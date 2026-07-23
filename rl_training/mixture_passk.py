# Technique 2 (mixture-over-frozen-base), INFERENCE-TIME form: closed-form pass@k of a rollout-level
# mixture policy  pi_mix = (1-eps) pi_RL + eps pi_base,  computed directly from per-problem solve
# counts of the two models. NO GPU / NO retraining — pure composition of already-scored eval shards.
#
# Per problem, base solve-rate p_b = n_correct_base/N, RL solve-rate p_r = n_correct_RL/N.
# Each of k mixture samples is drawn from base w.p. eps else RL, so P(one sample wrong) =
# q = eps(1-p_b) + (1-eps)(1-p_r), and
#     pass@k_mix(problem) = 1 - q^k ,   pass@1_mix = eps*p_b + (1-eps)*p_r .
# Dataset curve = mean over problems. This is EXACT for a rollout-level mixture (each rollout from one
# model); it is a lower bound for a token-level mixture (which can also produce NEW correct paths).
#
# Guarantee (why this is a structural coverage floor, not a reward): since q <= max(1-p_b,1-p_r),
#   pass@k_mix >= max( (1-eps)-scaled RL , eps-scaled base ) ... concretely pass@k_mix >=
#   1-(1-min(p_b,p_r)... )  — coverage never falls below the better model's, up to the eps factor,
#   BY CONSTRUCTION (the base is always in the sampler). No training, no loss term.

from __future__ import annotations
import glob, json
import numpy as np


def _load_counts(shard_glob, N_default=256):
    allp = []
    for f in sorted(glob.glob(shard_glob)):
        allp += json.load(open(f)).get("per_problem", [])
    by = {}
    for p in allp:
        n = p.get("n_samples", N_default)
        by[p["problem_id"]] = (p["n_correct"], n)
    return by


def mixture_curve(base_glob, rl_glob, eps, ks=(1, 2, 4, 8, 16, 32, 64, 128, 256), N=256):
    b = _load_counts(base_glob, N); r = _load_counts(rl_glob, N)
    ids = sorted(set(b) & set(r))
    if not ids:
        return None
    pb = np.array([b[i][0] / b[i][1] for i in ids])
    pr = np.array([r[i][0] / r[i][1] for i in ids])
    q = eps * (1 - pb) + (1 - eps) * (1 - pr)          # P(one mixture sample wrong)
    return {k: float((1 - q ** k).mean()) for k in ks}, len(ids)


def analyze(base_glob, rl_glob, epsilons=(0.0, 0.05, 0.1, 0.15, 0.25, 0.5, 1.0),
            ks=(1, 16, 256), N=256, label=""):
    print(f"\n=== {label} (mixture: (1-eps)*RL + eps*base) ===")
    hdr = "eps  | " + " | ".join(f"pass@{k}" for k in ks)
    print(hdr); print("-" * len(hdr))
    rows = {}
    for eps in epsilons:
        cur, n = mixture_curve(base_glob, rl_glob, eps, ks=ks, N=N)
        rows[eps] = cur
        tag = "  (pure RL)" if eps == 0 else ("  (pure base)" if eps == 1 else "")
        print(f"{eps:.2f} | " + " | ".join(f"{cur[k]:.4f}" for k in ks) + tag + (f"   n={n}" if eps == 0 else ""))
    # find eps that maximizes pass@256 (or largest k) without dropping pass@1 more than 1% relative
    kmax = ks[-1]; k1 = ks[0]
    rl1 = rows[0.0][k1]
    best = max((e for e in epsilons if e < 1.0),
               key=lambda e: rows[e][kmax] if rows[e][k1] >= 0.99 * rl1 else -1)
    print(f">> best eps with <=1% pass@{k1} loss: eps={best:.2f} "
          f"-> pass@{kmax}={rows[best][kmax]:.4f} (pure-RL {rows[0.0][kmax]:.4f}, "
          f"pure-base {rows[1.0][kmax]:.4f}); pass@{k1}={rows[best][k1]:.4f} (RL {rl1:.4f})")
    return rows


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="glob for base scored shards")
    ap.add_argument("--rl", required=True, help="glob for RL(grpo/expM3) scored shards")
    ap.add_argument("--label", default="")
    ap.add_argument("--N", type=int, default=256)
    a = ap.parse_args()
    analyze(a.base, a.rl, label=a.label, N=a.N)
