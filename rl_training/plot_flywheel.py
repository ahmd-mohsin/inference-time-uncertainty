# Turn flywheel metrics.jsonl into the paper-defining figures.
#   Fig 1 (headline): pass@k curves, one per round -> does the whole curve rise each round?
#   Fig 2 (convergence): mean pass@k (full & hard) vs round -> fixed point, plateau, or collapse?
#   Fig 3 (mechanism): harvest yield + support-frontier migration (stuck/hard/solved) vs round.
# Also emits a LaTeX table (flywheel_table.tex) of per-round pass@k for the paper.
import argparse, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(metrics_path):
    return [json.loads(l) for l in open(metrics_path) if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="rl_training/runs/flywheel/metrics.jsonl")
    ap.add_argument("--out-dir", default="rl_training/runs/flywheel")
    ap.add_argument("--base-eval", default="", help="optional base passk json to draw as round -1 ref")
    a = ap.parse_args()
    R = load(a.metrics)
    if not R:
        print("no metrics yet"); return
    os.makedirs(a.out_dir, exist_ok=True)
    ks = sorted(int(k) for k in R[0]["passk_full"].keys())

    # ---- Fig 1: pass@k per round (headline) ----
    fig, ax = plt.subplots(figsize=(6, 4.2))
    cmap = plt.cm.viridis
    for rec in R:
        r = rec["round"]; c = cmap(r / max(1, len(R) - 1))
        ax.plot(ks, [rec["passk_full"][str(k)] for k in ks], "o-", color=c, label=f"round {r}")
    ax.set_xscale("log", base=2); ax.set_xticks(ks); ax.set_xticklabels(ks)
    ax.set_xlabel("k (samples)"); ax.set_ylabel("pass@k (full set)")
    ax.set_title("Iterated self-distillation: pass@k per round")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(f"{a.out_dir}/fig1_passk_per_round.pdf"); plt.close(fig)

    # ---- Fig 2: convergence (mean pass@k vs round, full + hard) ----
    fig, ax = plt.subplots(figsize=(6, 4.2))
    rounds = [rec["round"] for rec in R]
    ax.plot(rounds, [rec["mean_passk_full"] for rec in R], "o-", label="mean pass@k (full)")
    if any(rec.get("mean_passk_hard") for rec in R):
        ax.plot(rounds, [rec.get("mean_passk_hard") for rec in R], "s-", label="mean pass@k (hard set)")
    ax.set_xlabel("flywheel round"); ax.set_ylabel("mean pass@k")
    ax.set_title("Convergence: does support keep expanding?")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{a.out_dir}/fig2_convergence.pdf"); plt.close(fig)

    # ---- Fig 3: mechanism (harvest yield + frontier migration) ----
    fig, ax1 = plt.subplots(figsize=(6, 4.2))
    ax1.bar([r - 0.15 for r in rounds], [rec["harvest_yield"] for rec in R], width=0.3,
            color="#2c7fb8", label="harvest yield")
    ax1.set_xlabel("flywheel round"); ax1.set_ylabel("harvested tail rollouts", color="#2c7fb8")
    ax2 = ax1.twinx()
    for lab, col in [("stuck", "#d95f02"), ("hard", "#7570b3"), ("solved", "#1b9e77")]:
        ax2.plot(rounds, [rec["label_counts"].get(lab, 0) for rec in R], "^-", color=col, label=lab)
    ax2.set_ylabel("# problems by label")
    ax1.set_title("Mechanism: fuel (harvest) + support-frontier migration")
    l1, la1 = ax1.get_legend_handles_labels(); l2, la2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, la1 + la2, fontsize=7, loc="center right")
    fig.tight_layout(); fig.savefig(f"{a.out_dir}/fig3_mechanism.pdf"); plt.close(fig)

    # ---- LaTeX table ----
    with open(f"{a.out_dir}/flywheel_table.tex", "w") as f:
        f.write("\\begin{tabular}{r" + "c" * len(ks) + "cc}\n\\toprule\n")
        f.write("round & " + " & ".join(f"p@{k}" for k in ks) + " & harvest & hard\\_solved\\\\\n\\midrule\n")
        for rec in R:
            row = [str(rec["round"])] + [f"{rec['passk_full'][str(k)]:.3f}" for k in ks]
            row += [str(rec["harvest_yield"]), str(rec["label_counts"].get("solved", 0))]
            f.write(" & ".join(row) + "\\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    # ---- console summary (the key takeaway) ----
    print("=== FLYWHEEL SUMMARY ===")
    for rec in R:
        print(f"  round {rec['round']}: mean pass@k full={rec['mean_passk_full']:.4f} "
              f"hard={rec.get('mean_passk_hard')} harvest={rec['harvest_yield']} "
              f"labels={rec['label_counts']}")
    if len(R) >= 2:
        d = R[-1]["mean_passk_full"] - R[0]["mean_passk_full"]
        print(f"  NET mean-pass@k change over {len(R)-1} rounds: {d:+.4f} "
              f"({'EXPANDING' if d > 0.002 else 'PLATEAU/COLLAPSE'})")
    print(f"  figures -> {a.out_dir}/fig1_passk_per_round.pdf, fig2_convergence.pdf, fig3_mechanism.pdf")


if __name__ == "__main__":
    main()
