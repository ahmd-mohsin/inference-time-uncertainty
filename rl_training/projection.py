# Technique 1 — HARD PROJECTION: after each GRPO step, project the policy back onto the feasible set
#   F = { theta : log pi_theta(y_q | q) >= log pi_ref(y_q | q) + log(alpha)  for all banked (q,y_q) }
# by taking teacher-forced correction gradient steps on ONLY the violating banked traces, until they
# are feasible again (or a step cap is hit). This is projected-gradient-style constrained
# optimization: a PENALTY discourages violation softly; PROJECTION restores feasibility HARD every
# step. Off-policy (teacher-forced on the bank) so it can raise prob on a mode the policy no longer
# samples — which on-policy reward cannot (no rollout -> no gradient).
#
# This file holds the PURE control-flow/math (no torch model), unit-tested on CPU in
# tests/test_projection.py. The model-coupled step lives in coverage_trainer.py and CALLS these.

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence
import math


def floor_logp(ref_logp: float, alpha: float) -> float:
    """Per-trace feasibility floor: ref + log(alpha), alpha in (0,1]."""
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha must be in (0,1], got {alpha}")
    return ref_logp + math.log(alpha)


def violations(policy_logp: Sequence[float], ref_logp: Sequence[float],
               alpha: float) -> list[int]:
    """Indices of banked traces currently BELOW their floor (need correction). Strictly-below only:
    a trace exactly at the floor is feasible (closed set), so it is NOT selected."""
    if len(policy_logp) != len(ref_logp):
        raise ValueError("policy_logp and ref_logp length mismatch")
    return [i for i, (pl, rl) in enumerate(zip(policy_logp, ref_logp))
            if pl < floor_logp(rl, alpha) - 1e-9]


def max_violation(policy_logp: Sequence[float], ref_logp: Sequence[float], alpha: float) -> float:
    """Largest floor-gap (>=0). 0 => fully feasible. Used to decide early-stop of the projection."""
    g = 0.0
    for pl, rl in zip(policy_logp, ref_logp):
        g = max(g, floor_logp(rl, alpha) - pl)
    return max(g, 0.0)


@dataclass
class ProjectionConfig:
    alpha: float = 0.5          # floor slack: p_theta >= alpha * p_ref (alpha=1 forbids ANY drop)
    max_steps: int = 5          # cap on correction sub-steps per GRPO step (bounded cost)
    lr: float = 1e-5            # correction sub-step lr (small; only fixes violations)
    batch_size: int = 4         # banked traces per correction sub-step (small: huge vocab)
    tol: float = 0.0            # stop when max_violation <= tol (in nats)
    every: int = 1              # run projection every `every` GRPO steps (amortize cost)


@dataclass
class ProjectionState:
    """Bookkeeping the trainer updates each GRPO step; kept here so it is testable."""
    step: int = 0
    last_max_violation: float = 0.0
    last_n_violations: int = 0
    total_correction_steps: int = 0
    history: list = field(default_factory=list)

    def record(self, max_v: float, n_v: int, corr_steps: int):
        self.last_max_violation = max_v
        self.last_n_violations = n_v
        self.total_correction_steps += corr_steps
        self.history.append({"step": self.step, "max_violation": max_v,
                             "n_violations": n_v, "corr_steps": corr_steps})


def should_project(step: int, cfg: ProjectionConfig) -> bool:
    """Amortization gate: only project on steps divisible by cfg.every (every=1 -> always)."""
    return cfg.every <= 1 or (step % cfg.every == 0)


def batches(indices: Sequence[int], batch_size: int):
    """Yield violating-trace index batches for correction sub-steps (deterministic order)."""
    for i in range(0, len(indices), max(1, batch_size)):
        yield list(indices[i:i + batch_size])


def projection_loop_plan(policy_logp_fn, ref_logp: Sequence[float], cfg: ProjectionConfig):
    """PURE simulation of the correction loop given a callable policy_logp_fn() -> current logps.
    Returns the list of (sub_step, violating_batches) the trainer should execute, plus final state.
    Used both to drive the real loop and to unit-test convergence/termination logic on CPU with a
    mock policy_logp_fn (no model). Guarantees: <= max_steps sub-steps; stops early when feasible."""
    plan = []
    for sub in range(cfg.max_steps):
        pl = policy_logp_fn()
        if max_violation(pl, ref_logp, cfg.alpha) <= cfg.tol:
            break
        vio = violations(pl, ref_logp, cfg.alpha)
        if not vio:
            break
        plan.append((sub, list(batches(vio, cfg.batch_size))))
        # the real trainer would apply gradient steps here, changing policy_logp_fn's output;
        # in the pure plan we just record what WOULD be corrected this sub-step.
    return plan
