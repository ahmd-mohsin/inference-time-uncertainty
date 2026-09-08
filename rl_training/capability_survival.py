#!/usr/bin/env python3
"""Capability-Survival core objects (CPU-testable, no GPU).

The reframe's central object: a capability q is ALIVE at inference budget K / target recoverability tau
iff its single-sample success mass p satisfies  p >= eps(K, tau) = 1 - (1-tau)^(1/K).  We estimate p by
ACTUAL SAMPLED CORRECTNESS (binomial), never by teacher-forced sequence log-prob, and classify with a
Clopper-Pearson confidence interval into ALIVE / UNCERTAIN / EXTINCT-at-budget.

Also: the on-policy signal-starvation curve S_G(p) = 1 - (1-p)^G - p^G and the extinction boundary Gp~1.
"""
from __future__ import annotations
import math


def eps_budget(K: int, tau: float) -> float:
    """Minimum single-sample success mass for recoverability R_K = 1-(1-p)^K >= tau.
    eps = 1 - (1-tau)^(1/K).  This converts a deployment budget (K, tau) into a training constraint on p."""
    if not (0.0 < tau < 1.0):
        raise ValueError("tau in (0,1)")
    if K < 1:
        raise ValueError("K>=1")
    return 1.0 - (1.0 - tau) ** (1.0 / K)


def recoverability(p: float, K: int) -> float:
    """R_K(p) = 1 - (1-p)^K : probability a correct answer is found within K i.i.d. samples."""
    p = min(max(p, 0.0), 1.0)
    return 1.0 - (1.0 - p) ** K


def clopper_pearson(c: int, n: int, alpha: float = 0.05):
    """Exact binomial CI [lo, hi] for p given c successes in n trials (two-sided 1-alpha)."""
    if n <= 0:
        return 0.0, 1.0
    try:
        from scipy.stats import beta
        lo = 0.0 if c == 0 else beta.ppf(alpha / 2, c, n - c + 1)
        hi = 1.0 if c == n else beta.ppf(1 - alpha / 2, c + 1, n - c)
        return float(lo), float(hi)
    except Exception:
        # normal-approx fallback (only if scipy absent); conservative-ish
        if c == 0:
            return 0.0, 1.0 - (alpha / 2) ** (1.0 / n)
        if c == n:
            return (alpha / 2) ** (1.0 / n), 1.0
        p = c / n
        z = 1.959963984540054
        se = math.sqrt(p * (1 - p) / n)
        return max(0.0, p - z * se), min(1.0, p + z * se)


def classify_capability(c: int, n: int, K: int, tau: float, alpha: float = 0.05) -> str:
    """Three-way, statistically honest classification of one capability from sampled correctness.
      'alive'      : LCB(p) > eps      (certified recoverable at budget K)
      'extinct'    : UCB(p) < eps      (certified NOT recoverable at budget K)
      'uncertain'  : otherwise         (n too small to certify either way)
    NOTE: observing c=0 does NOT imply extinct — it usually lands 'uncertain' until n is large enough."""
    e = eps_budget(K, tau)
    lo, hi = clopper_pearson(c, n, alpha)
    if lo > e:
        return "alive"
    if hi < e:
        return "extinct"
    return "uncertain"


def n_to_certify_extinct(K: int, tau: float, alpha: float = 0.05) -> int:
    """Given c=0 successes, the smallest n whose CP upper bound falls below eps(K,tau) — i.e. how many
    samples you need to *certify* extinction. For c=0, UCB = 1-(alpha/2)^(1/n) < eps  =>  n bound."""
    e = eps_budget(K, tau)
    # UCB(0,n) = 1 - (alpha/2)^(1/n); solve 1-(alpha/2)^(1/n) < e  => (alpha/2)^(1/n) > 1-e
    # => (1/n) ln(alpha/2) > ln(1-e) => n > ln(alpha/2)/ln(1-e)
    return int(math.ceil(math.log(alpha / 2) / math.log(1 - e)))


def signal_prob(p: float, G: int) -> float:
    """S_G(p) = 1-(1-p)^G - p^G : probability a binary-reward GRPO group of G rollouts is INFORMATIVE
    (has both a correct and an incorrect sample -> nonzero advantage variance). ~ G*p for small p."""
    p = min(max(p, 0.0), 1.0)
    return 1.0 - (1.0 - p) ** G - p ** G


def expected_groups_to_signal(p: float, G: int) -> float:
    """E[# groups] before an informative positive group ~ 1/S_G(p) ~ 1/(G p) as p->0 (extinction barrier)."""
    s = signal_prob(p, G)
    return float("inf") if s <= 0 else 1.0 / s


def summarize(counts, n: int, K: int, tau: float, alpha: float = 0.05) -> dict:
    """Aggregate a dict {q: c} of sampled success counts (n samples each) into a survival report."""
    e = eps_budget(K, tau)
    cls = {q: classify_capability(c, n, K, tau, alpha) for q, c in counts.items()}
    alive = [q for q, k in cls.items() if k == "alive"]
    extinct = [q for q, k in cls.items() if k == "extinct"]
    uncertain = [q for q, k in cls.items() if k == "uncertain"]
    return {"K": K, "tau": tau, "eps": e, "n": n, "total": len(counts),
            "alive": len(alive), "uncertain": len(uncertain), "extinct": len(extinct),
            "alive_ids": alive, "extinct_ids": extinct, "uncertain_ids": uncertain}


if __name__ == "__main__":
    # sanity
    for K, tau in [(256, 0.9), (1024, 0.5)]:
        e = eps_budget(K, tau)
        print(f"K={K} tau={tau}: eps={e:.3e}  n_to_certify_extinct(c=0)={n_to_certify_extinct(K,tau)}")
    for G in [4, 8, 16]:
        for p in [0.5, 0.1, 0.01, 0.001]:
            print(f"G={G} p={p}: S_G={signal_prob(p,G):.4f}  E[groups]={expected_groups_to_signal(p,G):.1f}  Gp={G*p:.3f}")
