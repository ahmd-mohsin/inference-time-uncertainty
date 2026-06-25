# Analyze the verification-generation gap: regimes, gap-vs-lift correlation, verdict.
import argparse, json, glob
from pathlib import Path


def spearman(x, y):
    """Spearman rho without scipy: Pearson on ranks."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0] * len(v)
        for rank, i in enumerate(order): r[i] = rank
        return r
    rx, ry = ranks(x), ranks(y)
    n = len(x); mx = sum(rx)/n; my = sum(ry)/n
    num = sum((a-mx)*(b-my) for a, b in zip(rx, ry))
    den = (sum((a-mx)**2 for a in rx) * sum((b-my)**2 for b in ry)) ** 0.5
    return num/den if den else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/verification_gap_qwen4b")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    d = Path(args.data_dir)
    recs = [json.load(open(f)) for f in sorted(d.glob("gap_*.json"))]
    recs = [r for r in recs if r["V_auc"] is not None]
    print(f"problems with verifier score: {len(recs)}")

    G = [r["G"] for r in recs]; V = [r["V_auc"] for r in recs]
    gap = [r["gap"] for r in recs]; lift = [r["selection_lift"] for r in recs]

    # ---- regimes ----
    HIV, LOG = 0.75, 0.5
    recover = [r for r in recs if r["V_auc"] >= HIV and r["G"] < LOG]
    ceiling = [r for r in recs if abs(r["gap"]) < 0.1]
    hardwall = [r for r in recs if r["V_auc"] < 0.55 and r["G"] < LOG]
    print("\n=== Regimes ===")
    print(f"  recoverable (V>={HIV}, G<{LOG})  : {len(recover)}  -> best-of-N should help")
    print(f"  no-gap (|V-G|<0.1)              : {len(ceiling)}  -> judging==generating")
    print(f"  hard-wall (V<0.55, G<{LOG})      : {len(hardwall)}  -> can't even verify")

    # ---- the headline test: does gap predict selection lift? ----
    rho = spearman(gap, lift)
    print("\n=== HEADLINE: does verification gap predict best-of-N lift over majority vote? ===")
    print(f"  Spearman(gap, selection_lift) = {rho:+.3f}")
    print(f"  (strong positive => gap is a real, cheap scalability signal)")

    # ---- aggregate verifier value ----
    mv = sum(r["mv_correct"] for r in recs) / len(recs)
    vs = sum(r["verifier_select_correct"] for r in recs) / len(recs)
    print(f"\n=== Selection accuracy (does the verifier beat majority vote?) ===")
    print(f"  majority-vote accuracy      : {mv:.3f}")
    print(f"  verifier-best-of-N accuracy : {vs:.3f}")
    print(f"  lift                        : {vs - mv:+.3f}")
    print(f"  mean V_auc={sum(V)/len(V):.3f}  mean G={sum(G)/len(G):.3f}  mean gap={sum(gap)/len(gap):+.3f}")

    json.dump({"n": len(recs), "spearman_gap_lift": rho,
               "mv_acc": mv, "verifier_acc": vs, "lift": vs-mv,
               "n_recoverable": len(recover), "n_nogap": len(ceiling),
               "n_hardwall": len(hardwall)}, open(d / "gap_analysis.json", "w"), indent=2)

    if not args.no_plot:
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
            cols = ["tab:green" if r["selection_lift"] > 0 else
                    ("tab:red" if r["selection_lift"] < 0 else "tab:gray") for r in recs]
            a1.scatter(G, V, c=cols, s=60, edgecolors="k")
            a1.plot([0, 1], [0, 1], "k--", alpha=0.4)
            a1.set_xlabel("G = pass@N (generation)"); a1.set_ylabel("V = verifier AUC")
            a1.set_title("Verification vs Generation\n(green=verifier helps, above diag=gap)")
            a2.scatter(gap, lift, s=60, edgecolors="k")
            a2.set_xlabel("gap = V - G"); a2.set_ylabel("selection lift (verifier - majority)")
            a2.set_title(f"Does gap predict lift?  Spearman={rho:+.2f}")
            plt.tight_layout(); plt.savefig(d / "gap_overview.png", dpi=150)
            print(f"\nsaved plot -> {d/'gap_overview.png'}")
        except Exception as e:
            print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
