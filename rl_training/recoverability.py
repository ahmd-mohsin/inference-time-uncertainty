#!/usr/bin/env python3
"""E1 — Operationalize recoverability (Recoverability-Constrained RLVR pivot).

Turns pass@k result JSONs (per_problem: problem_id, n_correct, n_samples) into the paper's new
objects, cleanly separated:

  p_hat_q      = n_correct / N                          single-sample success MASS (answer-event)
  R_K(q)       = 1 - (1 - p_hat_q)^K                    recoverability at future budget K
  EffSupp(K,τ) = #{ q : R_K(q) >= τ }                   effective (recoverable) support at budget K
  pass@1       = mean_q p_hat_q                          sharpness  (separate axis)
  pass@k       = mean_q [1-(1-p_hat)^k] (analytic)       coverage curve (matches reported estimator)

NOTE (honesty): this is the ANSWER-EVENT level (a problem is "recoverable" if a correct answer is
findable within K samples). It is the E1 stepping stone. MODE-level recoverability (distinguishing
distinct reasoning strategies) requires the strategy-clustered multi-witness bank (E2) and replaces
p_hat_q with per-mode mass p_theta(M|q). The closed form R_K and the effective-support machinery are
identical once per-mode masses exist.

Usage:
  python -m rl_training.recoverability --tag base=PATH grpo=PATH floor=PATH \
         --budgets 16 64 256 1024 4096 --taus 0.5 0.9 --out DIR
Each PATH is a passk_*.json. Problems are aligned by problem_id across models.
"""
import argparse, json, os


def load(path):
    d = json.load(open(path))
    N = d.get("n_samples")
    pp = {}
    for e in d.get("per_problem", []):
        nc = e.get("n_correct")
        if nc is None:
            continue
        pp[int(e["problem_id"])] = nc / N
    return pp, N, d.get("dataset"), d.get("subset")


def R_K(p, K):
    return 1.0 - (1.0 - p) ** K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", default=[],
                    help="name=path pairs, e.g. base=foo.json (repeatable)")
    ap.add_argument("--budgets", type=int, nargs="+", default=[16, 64, 256, 1024, 4096])
    ap.add_argument("--taus", type=float, nargs="+", default=[0.5, 0.9])
    ap.add_argument("--fragile-lo", type=float, default=0.0, help="fragile band lower p_hat (base)")
    ap.add_argument("--fragile-hi", type=float, default=0.05, help="fragile band upper p_hat (base)")
    ap.add_argument("--ref", default="base", help="reference model name for lost/preserved analysis")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    models = {}
    meta = {}
    for spec in a.model:
        name, path = spec.split("=", 1)
        p, N, ds, sub = load(path)
        models[name] = p
        meta[name] = dict(N=N, dataset=ds, subset=sub, path=path, n=len(p))
    names = list(models.keys())
    # common problem set
    common = set.intersection(*[set(m.keys()) for m in models.values()])
    common = sorted(common)

    out = {"models": meta, "n_common": len(common), "budgets": a.budgets, "taus": a.taus,
           "pass1": {}, "eff_support": {}, "fragile": {}, "lost_preserved": {}}

    # --- pass@1 (sharpness) ---
    for nm in names:
        out["pass1"][nm] = sum(models[nm][q] for q in common) / len(common)

    # --- effective (recoverable) support: #{q: R_K(q) >= tau} ---
    for K in a.budgets:
        out["eff_support"][K] = {}
        for tau in a.taus:
            for nm in names:
                cnt = sum(1 for q in common if R_K(models[nm][q], K) >= tau)
                out["eff_support"][K][f"{nm}@tau{tau}"] = cnt

    # --- fragile-band focus (problems the REF model finds rare-but-real) ---
    ref = a.ref
    frag = [q for q in common if a.fragile_lo <= models[ref][q] <= a.fragile_hi]
    out["fragile"]["band"] = [a.fragile_lo, a.fragile_hi]
    out["fragile"]["ref"] = ref
    out["fragile"]["n_fragile"] = len(frag)
    for K in a.budgets:
        out["fragile"][f"recoverable@K{K}_tau0.5"] = {
            nm: sum(1 for q in frag if R_K(models[nm][q], K) >= 0.5) for nm in names}
    # mean p_hat on fragile band per model (are rare modes kept alive?)
    out["fragile"]["mean_phat"] = {nm: (sum(models[nm][q] for q in frag) / len(frag) if frag else 0.0)
                                   for nm in names}

    # --- lost/preserved recoverable modes vs ref, at each budget/tau ---
    for K in a.budgets:
        for tau in a.taus:
            key = f"K{K}_tau{tau}"
            rec = {nm: set(q for q in common if R_K(models[nm][q], K) >= tau) for nm in names}
            row = {}
            for nm in names:
                if nm == ref:
                    continue
                lost = rec[ref] - rec[nm]      # ref recoverable, model NOT -> RL destroyed recoverability
                gained = rec[nm] - rec[ref]
                row[nm] = dict(recoverable=len(rec[nm]), lost_vs_ref=len(lost), gained_vs_ref=len(gained))
            row[ref] = dict(recoverable=len(rec[ref]))
            out["lost_preserved"][key] = row

    # ---- print report ----
    print(f"=== E1 Recoverability | {len(common)} common problems | "
          f"dataset={meta[names[0]]['dataset']} subset={meta[names[0]]['subset']} N={meta[names[0]]['N']} ===\n")
    print("pass@1 (sharpness):  " + "  ".join(f"{nm}={out['pass1'][nm]:.4f}" for nm in names))
    print(f"\nEffective (recoverable) support  EffSupp(K,τ) = #{{q: R_K(q)≥τ}}  (of {len(common)}):")
    hdr = "  K".ljust(8) + "τ".ljust(6) + "".join(nm.ljust(12) for nm in names)
    print(hdr); print("  " + "-" * (len(hdr)))
    for K in a.budgets:
        for tau in a.taus:
            cells = "".join(str(out["eff_support"][K][f"{nm}@tau{tau}"]).ljust(12) for nm in names)
            print(f"  {str(K).ljust(6)}{str(tau).ljust(6)}{cells}")
    print(f"\nFragile band (ref={ref}, p_hat∈[{a.fragile_lo},{a.fragile_hi}]): "
          f"{len(frag)} problems | mean p_hat: "
          + "  ".join(f"{nm}={out['fragile']['mean_phat'][nm]:.4f}" for nm in names))
    for K in a.budgets:
        r = out["fragile"][f"recoverable@K{K}_tau0.5"]
        print(f"  recoverable@K={K} (τ0.5): " + "  ".join(f"{nm}={r[nm]}" for nm in names))
    print(f"\nLost/preserved recoverable modes vs {ref}:")
    for K in a.budgets:
        for tau in a.taus:
            row = out["lost_preserved"][f"K{K}_tau{tau}"]
            s = "  ".join(f"{nm}:rec={row[nm]['recoverable']}"
                          + (f",lost={row[nm]['lost_vs_ref']},gain={row[nm]['gained_vs_ref']}"
                             if nm != ref else "") for nm in names)
            print(f"  K={str(K).ljust(5)} τ={tau}:  {s}")

    if a.out:
        os.makedirs(a.out, exist_ok=True)
        fp = os.path.join(a.out, "recoverability_E1.json")
        json.dump(out, open(fp, "w"), indent=2)
        print(f"\nsaved -> {fp}")


if __name__ == "__main__":
    main()
