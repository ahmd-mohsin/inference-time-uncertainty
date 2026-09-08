# DIRECTION 2 — objective-preserving verification densification: validate the estimator BEFORE any LLM compute.
# Target randomized verifier: success = pass all m i.i.d. sampled tests -> objective p^m (p = per-test pass prob).
# Run N>=m tests, pass c. Compare three reward estimators at MATCHED verification budget N:
#   binary_subset : 1[first m of the N tests all pass]      (one random m-subset; the naive sparse verifier)
#   Rhat_m        : C(c,m)/C(N,m)                            (Rao-Blackwellized: average over ALL m-subsets)
#   passfrac      : c/N                                      (dense but estimates p, the WRONG objective)
# Claims to check: (1) E[binary_subset]=E[Rhat_m]=p^m (same objective); E[passfrac]=p (different objective).
# (2) Var[Rhat_m] <= Var[binary_subset] (Rao-Blackwell). (3) REINFORCE policy-gradient variance is lower for
# Rhat_m at matched N, while it targets the SAME objective. Pure numpy; no dependencies on the RL stack.
import numpy as np
from math import comb

def estimators(c, N, m):
    binary_subset = 1.0 if c >= N else None  # placeholder; computed per-draw below (needs the actual subset)
    Rhat = comb(c, m) / comb(N, m) if c >= m else 0.0
    passfrac = c / N
    return Rhat, passfrac

def simulate(p, N, m, trials=200000, seed=0):
    rng = np.random.default_rng(seed)
    # draw N Bernoulli(p) test outcomes per trial
    outcomes = rng.random((trials, N)) < p
    c = outcomes.sum(1)
    # binary on a FIXED random m-subset (take first m columns = one random subset since tests are iid)
    binary_subset = outcomes[:, :m].all(1).astype(float)
    denom = comb(N, m)
    Rhat = np.array([comb(int(ci), m) if ci >= m else 0 for ci in c], dtype=float) / denom
    passfrac = c / N
    return binary_subset, Rhat, passfrac

def reinforce_grad_var(reward, score=1.0):
    # REINFORCE single-sample grad = reward * score; variance of the estimator (score held fixed / unit)
    g = reward * score
    return g.var()

def main():
    print(f"{'p':>4} {'N':>3} {'m':>2} | {'p^m':>7} | E[bin] E[Rhat] E[pf] | Var[bin] Var[Rhat] ratio | gradVar bin/Rhat")
    for p in (0.5, 0.7, 0.9):
        for (N, m) in ((4, 2), (8, 2), (8, 4), (12, 4)):
            bs, rh, pf = simulate(p, N, m)
            pm = p**m
            vb, vr = bs.var(), rh.var()
            gb, gr = reinforce_grad_var(bs), reinforce_grad_var(rh)
            print(f"{p:>4} {N:>3} {m:>2} | {pm:7.4f} | {bs.mean():.4f} {rh.mean():.4f} {pf.mean():.4f} | "
                  f"{vb:.5f} {vr:.5f} {vr/max(vb,1e-9):.3f} | {gb/max(gr,1e-12):.2f}x")
    print("\nInterpretation:")
    print(" - E[bin]==E[Rhat]==p^m  -> both estimate the SAME random-m-test success objective (objective preserved).")
    print(" - E[pf]==p (not p^m)    -> pass-fraction estimates a DIFFERENT (easier) objective; not objective-preserving.")
    print(" - Var[Rhat] <= Var[bin] (ratio<=1) -> Rao-Blackwell variance reduction at MATCHED budget N.")
    print(" - gradVar bin/Rhat >1x  -> REINFORCE gradient variance is lower for Rhat_m (denser, same objective).")
    print(" - Boundary (not shown): correlated tests / biased generators break E[bin]=p^m; must be tested in-env.")

if __name__ == "__main__":
    main()
