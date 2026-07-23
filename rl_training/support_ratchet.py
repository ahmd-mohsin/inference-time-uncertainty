# Support Ratchet — coverage preservation as a ONE-SIDED CONSTRAINT (not a reward).
#
# Motivation (docs: /tmp/support_ratchet_design.md, responds to UCPO 2605.00365 Cor 4.1):
# RLVR prunes rare-correct modes IRREVERSIBLY — once a correct mode's sampling prob drops below
# ~1/K it gets zero on-policy gradient and dies. Every reward/advantage-shaping method (UCPO, PKPO,
# RiskPO, ...) can only act on modes the policy STILL samples, so none can save an already-pruned
# mode. We instead impose a CONSTRAINT anchored to an external bank of base-correct traces: the
# policy's log-prob on each banked correct trace must not fall below a reference-relative floor.
#
# Key properties that make this a different object than reward-shaping:
#   * acts on log pi_theta(y|q) DIRECTLY (added to the loss), not through the reward/advantage;
#   * ONE-SIDED (ratchet): penalize only DROPPING BELOW the floor, never being above -> never pulls
#     the policy toward base breadth -> no pass@1 tax (unlike two-sided KL-to-ref / UCPO uniformity);
#   * anchored to EXTERNAL traces -> protects a mode even when the online policy no longer samples it
#     (a reward literally cannot: no rollout -> no gradient). This is what attacks irreversibility.
#
# This module is PURE TORCH and has no trl/vllm dependency, so it is unit-testable in isolation
# (see tests/test_support_ratchet.py).

from __future__ import annotations

import torch


def sequence_logprob(logits: torch.Tensor, labels: torch.Tensor,
                     mask: torch.Tensor) -> torch.Tensor:
    """Teacher-forced summed log-prob of each sequence: sum_t log p(label_t | prefix).

    Args:
      logits: (B, T, V) next-token logits (already shifted OR we shift here — see below).
      labels: (B, T) token ids of the completion (pad anywhere mask==0).
      mask:   (B, T) 1.0 on completion tokens to score, 0.0 on prompt/pad tokens.

    Convention: logits[:, t] predicts labels[:, t] (caller aligns; the standard HF pattern is
    logits[:, :-1] vs labels[:, 1:], which the caller should apply before calling). Returns (B,)
    summed log-prob over masked positions. Summed (not mean) because the floor is reference-relative
    (ell_ref - ell_theta), so per-token length cancels and a raw sum is the correct comparable.
    """
    # Memory-efficient: log p(label) = logit[label] - logsumexp(logits). We avoid materializing the
    # full (B,T,V) log_softmax tensor (V~151k for Qwen -> 14GB+ OOM); instead gather the target logit
    # and use logsumexp over V, both O(B*T). Row-by-row over T keeps the peak at one (B,V) slice.
    logits = logits.float()
    B, T, V = logits.shape
    gathered = torch.gather(logits, -1, labels.unsqueeze(-1)).squeeze(-1)  # (B,T) target logits
    # logsumexp over vocab per (b,t); chunk over T to cap peak memory on huge V.
    lse = torch.empty((B, T), dtype=logits.dtype, device=logits.device)
    step = max(1, 256 // max(1, V // 4096))                               # ~ up to 256 positions/chunk
    for s in range(0, T, step):
        lse[:, s:s + step] = torch.logsumexp(logits[:, s:s + step, :], dim=-1)
    tok_logp = (gathered - lse) * mask                                    # (B,T)
    return tok_logp.sum(dim=-1)                                           # (B,)


def ratchet_penalty(policy_logp: torch.Tensor, ref_logp: torch.Tensor,
                    alpha: float = 0.5, reduction: str = "mean") -> torch.Tensor:
    """One-sided reference-anchored support floor penalty.

    For each banked base-correct trace q with reference log-prob ell_ref = ref_logp[q]:
        floor_q      = ell_ref + log(alpha)          # allow the policy prob to drop by up to 1/alpha
        penalty_q    = relu( floor_q - policy_logp )  # ONLY active when it drops BELOW the floor

    alpha in (0,1]: alpha=1 forbids ANY drop below reference; alpha=0.5 tolerates a 2x prob drop;
    smaller alpha = looser floor. relu makes it one-sided: being above the floor costs nothing, so
    the constraint never pulls the policy back toward the base (no pass@1 tax).

    Args:
      policy_logp: (B,) current-policy summed seq log-prob on the banked traces.
      ref_logp:    (B,) reference (base/init) summed seq log-prob, precomputed once when banking.
      alpha:       floor slack in (0,1].
      reduction:   'mean' | 'sum' | 'none'.
    Returns: scalar (mean/sum) or (B,) penalty (>=0).
    """
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha must be in (0,1], got {alpha}")
    floor = ref_logp + torch.log(torch.as_tensor(alpha, dtype=ref_logp.dtype,
                                                  device=ref_logp.device))
    pen = torch.clamp(floor - policy_logp, min=0.0)   # relu(floor - logp): one-sided
    if reduction == "none":
        return pen
    if reduction == "sum":
        return pen.sum()
    return pen.mean()


def dual_update(mu: float, mean_penalty: float, kappa: float = 0.0,
                eta_mu: float = 0.1, mu_max: float = 5.0) -> float:
    """Optional Lagrange-dual ascent on the multiplier mu.

    Increase mu when the mean constraint violation exceeds a small slack target kappa, decrease when
    the constraint is satisfied; clip to [0, mu_max]. Fixed-mu training just skips this.
    """
    mu_new = mu + eta_mu * (mean_penalty - kappa)
    return float(min(max(mu_new, 0.0), mu_max))


def fraction_modes_alive(policy_logp: torch.Tensor, ref_logp: torch.Tensor,
                         alpha: float = 0.5) -> float:
    """Diagnostic: fraction of banked base-correct modes still at/above the floor under the policy.
    This is the DIRECT measure of 'did we stop the irreversible pruning' — report it during training
    and as a headline metric (reward-shaping baselines have no analogous guarantee)."""
    floor = ref_logp + torch.log(torch.as_tensor(alpha, dtype=ref_logp.dtype,
                                                  device=ref_logp.device))
    return float((policy_logp >= floor).float().mean())
