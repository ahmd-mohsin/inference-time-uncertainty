#!/usr/bin/env python3
"""E8 — The K·p ≈ 1 phase transition (mechanism figure for Recoverability-Constrained RLVR).

Claim: on-policy RL is *support-blind* to a mode of mass p once K·p < 1 — a group of K rollouts
almost never contains it, so it gets no gradient. Two observables, both closed-form in (K, p):

  observability   O_K(p) = 1 - (1-p)^K            (mode appears in a group of K)
  GRPO signal     S_K(p) = [1-(1-p)^K]·[1-p^K]    (group has BOTH a correct & incorrect rollout ->
                                                    nonzero group-relative advantage => real gradient)

S_K(p) -> 0 both as p->0 (never sampled: K·p<1 dead zone) and p->1 (no contrast). The off-policy
support floor's signal is teacher-forced => CONSTANT in p (=1), the only channel alive in the dead zone.

Causal reading (no synthetic suppression needed — RL *did* the suppression): base gives modes a
natural range of p; plain GRPO continued-RL pushes fragile problems' mass DOWN, across the K·p=1
boundary into the dead zone; the floor keeps them above it. We measure per-problem mass
p̂ = n_correct/N from the base/grpo/floor pass@k evals and count how many problems each model leaves
below the deployment boundary.

Usage: python -m rl_training.phase_transition --model base=PATH grpo=PATH floor=PATH \
       --deployK 256 1024 --out DIR
"""
import argparse, json, os
import numpy as np


def load_phat(path):
    d = json.load(open(path)); N = d["n_samples"]
    return {int(e["problem_id"]): e["n_correct"] / N for e in d["per_problem"]}, N


def O_K(p, K): return 1.0 - (1.0 - p) ** K
def S_K(p, K): return (1.0 - (1.0 - p) ** K) * (1.0 - p ** K)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", default=[], help="name=path pass@k json")
    ap.add_argument("--deployK", type=int, nargs="+", default=[256, 1024])
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    M = {}; N = None
    for spec in a.model:
        nm, p = spec.split("=", 1); M[nm], N = load_phat(p)
    names = list(M)
    common = sorted(set.intersection(*[set(m) for m in M.values()]))

    out = {"n_common": len(common), "N": N, "deployK": a.deployK, "curve": {}, "deadzone": {}, "signal": {}}

    # (1) universal transition curve: sweep p on a grid, show O_K, S_K vs K·p
    grid = np.concatenate([np.logspace(-4, 0, 60)])
    for K in a.deployK:
        out["curve"][K] = {"Kp": (grid * K).tolist(),
                           "O": [O_K(p, K) for p in grid],
                           "S": [S_K(p, K) for p in grid]}

    # (2) per-model: how many problems fall in the dead zone (K·p < 1) at deployment K,
    #     and mean GRPO signal S_K — the causal "RL pushed modes below the boundary" count.
    for K in a.deployK:
        out["deadzone"][K] = {}; out["signal"][K] = {}
        for nm in names:
            kp = np.array([M[nm][q] * K for q in common])
            out["deadzone"][K][nm] = int((kp < 1.0).sum())
            out["signal"][K][nm] = float(np.mean([S_K(M[nm][q], K) for q in common]))

    # (3) causal transitions vs base: of problems base kept ABOVE the boundary (K·p_base>=1),
    #     how many did each RL arm push BELOW (into the dead zone)? and vice-versa (rescued).
    ref = "base" if "base" in names else names[0]
    out["causal_vs_ref"] = {"ref": ref}
    for K in a.deployK:
        base_above = set(q for q in common if M[ref][q] * K >= 1.0)
        base_below = set(q for q in common if M[ref][q] * K < 1.0)
        row = {}
        for nm in names:
            if nm == ref: continue
            arm_below = set(q for q in common if M[nm][q] * K < 1.0)
            pushed_below = base_above & arm_below     # RL drove into dead zone (support-blinded)
            rescued = base_below - arm_below          # RL lifted out of dead zone
            row[nm] = dict(pushed_below=len(pushed_below), rescued=len(rescued))
        out["causal_vs_ref"][K] = row

    # ---- report ----
    print(f"=== E8 K·p phase transition | {len(common)} problems | N={N} ===\n")
    print("Universal transition (closed form): O_K, S_K cross their knee at K·p≈1.")
    for K in a.deployK:
        print(f"\n--- deployment K={K} ---")
        print("  dead-zone problems (K·p<1, on-policy support-blind):  "
              + "  ".join(f"{nm}={out['deadzone'][K][nm]}" for nm in names) + f"   (of {len(common)})")
        print("  mean GRPO learning signal S_K (higher=more trainable): "
              + "  ".join(f"{nm}={out['signal'][K][nm]:.3f}" for nm in names))
        cz = out["causal_vs_ref"][K]
        for nm in names:
            if nm == ref: continue
            print(f"  vs {ref}: {nm} pushed {cz[nm]['pushed_below']} problems BELOW K·p=1 "
                  f"(support-blinded), rescued {cz[nm]['rescued']}")
    print("\nOff-policy floor signal = 1 (constant in p) — the only channel alive in the dead zone.")
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        fp = os.path.join(a.out, "phase_transition_E8.json")
        json.dump(out, open(fp, "w"), indent=2)
        print(f"\nsaved -> {fp}")


if __name__ == "__main__":
    main()
