# Award-Track Hypothesis Register (governing plan, 2026-09-10)
Companion to ADAPTIVE_FORGETTING_RESULTS.md (§1–§112) and PROGRESS_SNAPSHOT.md v4.
This file is the GOVERNING execution plan. Full text pasted by PI 2026-09-10; key structure below.

## Award-shaped thesis (evidence-supported)
Outcome-RL and verified rejection-FT share a per-prompt gradient direction (success-gradient identity, §111 P1).
The large OOD-transfer gap between them is produced by IDENTIFIABLE choices in the RL estimator, is PREDICTED before training
by measurable headroom / dead-group (D_fail) statistics, and can be REMOVED inside RL by fixing those choices — recovering
RFT-level transfer at on-policy cost. Clause 1 proved; clauses 2-4 are Tier 1/2.

## Three discipline rules (every cell)
- No RL result at < 4 training seeds. - Every effect ships its decisive control PREREGISTERED. - Bottleneck statistic measured
  BEFORE training and must predict the held-out result or it is not a mechanism.

## Tier 0 gates (P1-P5): trainer-equivalence(Adam-level RFT vs positives-only mean-NLL RL endpoint); real-MaxRL unit test;
  matched-experience manifest; per-run GRPO audit (all-fail/mixed/clip/KL, unbiased subset estimator); freeze dev vs confirmation splits.
## Tier 1 ladder (mechanism): R0 GRPO -> R1 -std-norm -> R2 -KL/-clip -> R3 -negatives -> R4 success-count weighting ->
  R5 mean-token NLL -> R6 replay -> R7 = RFT. Run BOTH directions; reversal controls. Anchors: GRPO group +0.00, no-std +0.025, RFT +0.19-0.21.
  H1.1 negatives, H1.2 prompt-weighting, H1.3 replay(minor), H1.4 interaction, H1.5 GRPO-fixed METHOD (only if rungs recover >=80%),
  H1.6 crossover (D_fail predicts gap; near-miss rate via DAG Tvalue infra).
## Tier 2 predictive laws (out-of-sample): H2.1 headroom two-var form predicts new-coverage (fit 6 cells, predict 3); H2.2 D_fail predicts
  RFT-GRPO gap; H2.3 coverage dynamics geometric approach ρ; H2.4 GENERALITY grid {1.5B,7B,3B,Llama-3.2-3B,Gemma-2-2b}×{comp,MATH}×
  {RFT+,GRPO-group,GRPO-no-std,MaxRL}; H2.5 reachability vs reliability as separate endpoints.
## Tier 3 (only post Tier-1 verdict): H3.1 structural-annotation as RL primer (lowers D_fail); H3.2 decomp-distill on DAG cost; H3.3 repair@14B cost.
## Retired (do NOT re-test) — §5 table: MaxRL-teaching, imitation!=RL, diagnostic-curricula, verifier-resolution, RL-explorer,
  source-preservation, delayed-value, difficulty-escalation, recomposition, value-supervision(§112), mass-placing(void), advantage-density-law.

## Order of ops: Wk1 Tier0 + start H2.4 harvests. Wk2-3 Tier1 ladder + H2.4 + locked H2.1-2.3 predictions. Wk4 verdict -> H1.5 or interaction. Wk5 Tier3/write.
## Award checklist: headline at >=2 families x >=2 sizes (H2.4); 2-direction ladder + reversal; >=2 preregistered out-of-sample predictions landed;
  reachability/reliability separate; method only if matches RFT OOD at matched compute + beats GRPO in-dist at >=2 D points; confirmation opened once.
