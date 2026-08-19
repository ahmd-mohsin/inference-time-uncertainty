# E3 — Mode-mass preservation certificate (framework + validation)

Built `rl_training/recoverability_certificate.py`. The mathematical core of the pivot: turns
teacher-forced replay into a *certificate*, not a regularizer.

## The certificate (Jensen)
```
log p_θ(M|q) ≥ log p_0(M|q) + E_{y~ν_m}[ log π_θ(y|q) − log π_0(y|q) ]
```
so if the teacher-forced quantity `Δ_qm = E_{y~ν_m}[log π_θ − log π_0] ≥ log α`, then
`p_θ(M|q) ≥ α·p_0(M|q)` (sufficient, one-sided). With E1's closed form:
`R_K(M;θ) ≥ 1 − (1 − α·p_0(M|q))^K` — a finite-budget recoverability guarantee.

## What the module does
- Estimates `Δ_qm` from J witness traces/mode and reports a **lower confidence bound** `LCB(Δ_qm)`
  (Student-t) → the certificate is statistically calibrated, not a point estimate.
- A mode is **certified** iff `LCB(Δ_qm) ≥ log α` AND has ≥ `min_witness` traces.
- Flags **endangered modes** (`LCB < log α`) → these are exactly the modes E4's primal-dual should
  put preservation pressure on.
- `alpha_eff_lb = exp(min(LCB,0))` → the guaranteed mass-preservation multiplier per mode.

## Validation (CPU self-check, θ=base ⇒ Δ≡0)
- α=0.5 (logα=−0.693<0): **305/305 certified** ✓ (LCB=0 ≥ logα).
- α=1.5 (logα=+0.405>0): **0/305 certified** ✓ (LCB=0 < logα).
- `min_witness=8` gate correctly marks the thin ≤4-trace bank **low-confidence / non-certifiable** —
  i.e. the certificate refuses to certify without enough witnesses (the desired conservative behavior).

Plumbing (estimator + LCB + gating) confirmed correct.

## Status
- ✅ Certificate estimator + LCB + endangered-mode detection built and validated (CPU).
- ✅ GPU scorer emitted: `rl_training/score_bank_logprobs.py` — teacher-forces a checkpoint over a
  clustered bank and writes per-trace `{problem_id, mode_id, logp_theta, logp_ref}`. **No generation**
  (cheap forward pass). Launch-ready.
- ⏭ Real run needs: (1) E2 sampler → ≥8–128 witnesses/mode on a cluster; (2) `score_bank_logprobs.py`
  on base, r2-from-grpo, r2-from-floor; (3) `certify --scored ...` → % certified modes per arm. The
  headline claim becomes: *floor certifies markedly more mode-mass than plain GRPO, and the certified
  fraction predicts the E1 recoverability gap.*

## Pipeline now in place (E1→E3)
```
sample_base_solutions.py (GPU)  →  strategy_bank.py cluster (CPU)  →  score_bank_logprobs.py (GPU)
   witnesses w/ text                → per-problem modes               → per-trace logπ_θ, logπ_0
                                                                          ↓
                          recoverability_certificate.py certify (CPU) → certified-mode fraction + endangered set
                          recoverability.py (CPU)                      → mode-level R_K, effective support
```
Everything CPU-side is built and validated; only the two GPU passes (sample, score) are pending a cluster.
