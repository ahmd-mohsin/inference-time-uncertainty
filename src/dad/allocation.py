# allocation.py
#
# Budget-aware sample allocation for DAD, replacing the fixed M, M/2, M/2
# schedule with the water-filling solution derived from the coordinate-ascent
# view of the method.
#
# Per round r the chosen (top-leverage) coordinate has:
#   - resolvable mass  Delta_r  (how much contested mass this coordinate can shed)
#   - per-sample rate  rho_r    (sharpness of the conditioned distribution)
#   - per-sample cost  c_r      (|x| + |W| + L tokens)
# Shift curve  g_r(M) = Delta_r (1 - e^{-rho_r M}),  g_r'(M) = Delta_r rho_r e^{-rho_r M}.
#
# Online policy (default): a round is allocated
#     M_r = (1/rho_r) * ln( Delta_r rho_r / (lambda c_r) ),   clamped to [m_min, m_max]
# and skipped/stopped when the initial marginal return per token is below the
# water level:  Delta_r rho_r / c_r <= lambda.
#
# `lambda` is the value-of-a-token. Online we set it from a single interpretable
# hyperparameter `min_marginal_gain` (mass per 1k tokens worth paying for) and
# fold in the remaining budget. `offline_water_fill` solves the exact multi-round
# KKT optimum for analysis / oracle comparisons.

import math
from dataclasses import dataclass


@dataclass
class AllocationConfig:
    token_budget: int = 40_000        # B: total generation tokens per problem
    m_min: int = 2                    # never fewer than this when a round runs
    m_max: int = 16                   # never more than this in one round
    rho_prior: float = 0.45           # per-sample resolution rate before evidence
    min_marginal_gain: float = 0.02   # contested-mass per 1k tokens worth paying
    probe_samples: int = 8            # width of the round-1 unconditional probe


def update_rho(prev_rho: float, observed_shift: float,
               resolvable_mass: float, samples: int,
               floor: float = 0.05, ceil: float = 3.0) -> float:
    """Re-estimate rho from the realized contraction of the last round.

    From g(M) = Delta (1 - e^{-rho M}) with observed shift s over M samples:
        rho = -ln(1 - s/Delta) / M
    Falls back to the prior when the round was degenerate.
    """
    if samples <= 0 or resolvable_mass <= 1e-9:
        return prev_rho
    frac = max(0.0, min(observed_shift / resolvable_mass, 0.999))
    if frac <= 0.0:
        # no progress: shrink rho so the next allocation is cautious
        return max(floor, prev_rho * 0.5)
    rho = -math.log(1.0 - frac) / samples
    rho = max(floor, min(rho, ceil))
    # light smoothing with the prior for stability across rounds
    return 0.5 * prev_rho + 0.5 * rho


def allocate_round(remaining_budget: int,
                   cost_per_sample: float,
                   resolvable_mass: float,
                   rho: float,
                   cfg: AllocationConfig):
    """Return (n_samples, stop).

    stop=True  -> the marginal return is below the water level; halt refinement.
    n_samples  -> integer width for this round, clamped and budget-capped.
    """
    if remaining_budget < cost_per_sample * cfg.m_min:
        return 0, True
    if resolvable_mass <= 1e-9 or rho <= 0.0:
        return 0, True

    # water level: contested mass per token we are willing to pay for
    lam = cfg.min_marginal_gain / 1000.0          # per-token units

    # initial marginal return per token of this coordinate
    g0_per_token = (resolvable_mass * rho) / max(cost_per_sample, 1.0)
    if g0_per_token <= lam:
        return 0, True                            # Eq. (10): stop test

    # closed-form width, Eq. (9):  M = (1/rho) ln( Delta rho / (lambda c) )
    m_star = (1.0 / rho) * math.log((resolvable_mass * rho) / (lam * cost_per_sample))
    m = int(round(m_star))
    m = max(cfg.m_min, min(m, cfg.m_max))

    # budget cap
    budget_cap = int(remaining_budget // max(cost_per_sample, 1.0))
    m = max(0, min(m, budget_cap))
    if m < cfg.m_min:
        return 0, True
    return m, False


# ----------------------------------------------------------------------
# offline oracle: exact multi-round water-filling (for analysis / figures)
# ----------------------------------------------------------------------
def offline_water_fill(deltas, rhos, costs, budget, m_cap=64):
    """Exact KKT optimum of  max sum_r Delta_r(1-e^{-rho_r M_r})  s.t. sum_r M_r c_r <= B.

    Equalizes g_r'(M_r)/c_r = lambda across active rounds (water-filling).
    Returns the per-round sample counts (floats; round for use).
    """
    R = len(deltas)

    def widths(lam):
        ms = []
        for d, r, c in zip(deltas, rhos, costs):
            if d <= 0 or r <= 0 or (d * r) / c <= lam:
                ms.append(0.0)
            else:
                ms.append(min(m_cap, (1.0 / r) * math.log((d * r) / (lam * c))))
        return ms

    def spent(lam):
        return sum(m * c for m, c in zip(widths(lam), costs))

    # bisection on lambda: spent is decreasing in lambda
    lo, hi = 1e-12, 1.0
    # ensure bracket
    while spent(lo) < budget and lo > 1e-15:
        lo /= 10.0
    for _ in range(100):
        mid = math.sqrt(lo * hi)
        if spent(mid) > budget:
            lo = mid
        else:
            hi = mid
    return widths(math.sqrt(lo * hi))