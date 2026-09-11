# ⚠️ THESIS FALSIFIED (§112) — see "STATUS" below before reading further.
# STATUS 2026-09-10: The H1 mask control (reviewer-demanded) FALSIFIED the value-supervision thesis:
# Tmask (placeholder, no values) 0.850 ≈ Tvalue (correct values) 0.857; irrelevant values HURT (0.698). The gain is STRUCTURAL
# ANNOTATION of intermediates (scaffolding), NOT verified execution values — a known-adjacent formatting effect, not a novel
# verified-supervision method. This draft's original thesis (below) is retained only for record; the paper pivots to the
# CHARACTERIZATION (RFT>>GRPO + coverage/headroom + the DAG composition-bottleneck diagnostic + null battery), reframed WITHOUT
# any impossibility law. No award-caliber positive method emerged across recomposition/decomposition/value-supervision.
#
# --- original (now-falsified) draft below ---
# Which Execution States Teach Compositional Generalization?
_(method-centered alt-title if the selector wins: "Learning Compositional Reasoning from Distinguishing Execution States")_
_Working draft — 2026-09-10. Numbers from ADAPTIVE_FORGETTING_RESULTS.md §§79–111; not yet independently replicated._

## Abstract (draft opening — measured facts only; method claims await experiments)
Verified post-training supplies reliable examples of successful computation, but successful programs alone do not reveal which
additional supervision improves transfer to unfamiliar compositions. We study this in executable tasks that separate local
computations from their dependencies (a DAG domain with branching, merges, and intermediate reuse). On held-out DAG structures,
training with correct intermediate execution values improves a 1.5B model from **78.7% to 85.7%** (3 seeds, non-overlapping ranges),
while **scrambled**-value supervision reaches **78.0%** — the gain is the data-flow signal, not target length. This motivates a precise
question: *which* execution observations teach the dependencies needed for structural generalization, and how do their benefits depend
on the learner, the update rule, and the compute budget?

## 1. Thesis (testable program, NOT a law)
Verified intermediate supervision improves structural transfer when it makes computational distinctions the baseline update does not
learn efficiently. The benefit depends on the unresolved computation, the supervision content/cost, and the update procedure. We do NOT
claim self-improvement cannot expand competence, nor a scale-invariant absolute gain.

## 2. What is established (motivation + boundary conditions)
- **The composition bottleneck is real and localized** (E1-DAG, §106/§107b): with correct component code given, a 1.5B model still fails
  to WIRE the data-flow — oracle-component recovery 9.9% (DAG) vs 100% (linear pipelines), rising to 52.7% at 7B (capability-gated).
- **Correct intermediate values teach transferable wiring** (§109/§110): Tvalue 0.857 vs plain 0.787 (+0.07, 3 seeds); placebo (scrambled
  values) 0.780 ≈ plain → the *signal*, not length. Gain is capability-gated: +0.07 at 1.5B → +0.015 at 7B.
- **Boundary conditions (motivation only):** RFT>>GRPO for OOD from a shared checkpoint (+0.21, §91); a bank-identity coverage effect
  (§89, headroom-dependent); a null battery of ceiling-break attempts (repair/archive/delayed-value/escalation/recomposition, §84b/85/87/90b/107).
  These frame the problem; they do NOT prove an impossibility law (§111 retraction).

## 3. Theory (three modest, correct propositions — no universal law)
- **P1 success-gradient identity:** grad p_θ(x) = p_θ(x)·E[grad log π(y|x) | V=1]; RFT and outcome-RL share a gradient direction — the
  RFT>>GRPO gap is a weighting/normalization/collection-policy effect, an empirical question (§111).
- **P2 mixed-group formula:** P_mixed = 1 − p^G − (1−p)^G; quantifies when terminal GRPO groups carry zero reward-advantage signal (motivates H3).
- **P3 submodular coverage:** the state-selection objective F(Q) is monotone submodular under equal costs (greedy 1−1/e); budgeted costs need
  the budgeted treatment. (Classical; used as structure, not a new theorem.)
- **Info-theoretic caveat:** given program+input+interpreter the trace adds no Shannon information; the SFT benefit is bounded-learner usability.

## 4. Experiments (the contribution, if they pass)
- **H1 — WHY Tvalue works (RUNNING):** B, Tvalue, Ttrue-irrelevant (correct but disconnected aux values, length-matched), Tmask
  (placeholder), Tscramble. Advance the data-flow interpretation only if relevant values beat true-irrelevant AND formatting controls.
  Also test value POSITION (before/interleaved vs after code).
- **H2 — the method (state selection):** full-Tvalue vs random-state vs junction-heuristic vs distinguishing-state (candidate-error coverage
  F(Q)) vs extra-terminal-examples, at matched information budget AND matched end-to-end compute. Goal: retain most of the gain with fewer
  auxiliary tokens, or beat Tvalue at equal budget. Stop the selector if random/junction matches it.
- **H3 — RL:** from a shared checkpoint, {supervised, terminal-GRPO, full-process-GRPO, selected-state-GRPO} × supervision; log all-failure-group
  fraction, mixed-process fraction, first-success improvement, held-out OOD. Reproduce a faithful process-GRPO baseline (StepCodeReasoner), not a weak terminal one.
- **H4 — scaling by difficulty, not size:** crossed {local-op easy/hard} × {dependency-demand low/high} at 1.5B/7B; a development-calibrated
  hard-7B suite below saturation; predict a SELECTIVE benefit on measured dependency errors. Diagnostic: live-intermediate count under legal reorderings.

## 5. Downstream validation
Post-train a smaller code model with verified execution info from training programs; MBPP→HumanEval harness (finish + validate pass@1,
timeouts, hidden tests) + a controlled structural code split with hidden tests. Second model family for external replication before more sizes.
Full cost ledger (train + generated inference tokens + latency). Do NOT smuggle oracle values/reference code/search into the "one-attempt" eval.

## 6. Positioning vs closest prior (must reproduce a baseline, not rename)
StepCodeReasoner (verified execution-state targets + step-GRPO) is the closest — reproduce an execution-supervision baseline and
distinguish our selection rule / prediction of where it helps. SemCoder, Process-Supervised RL for Code, CAPER, Do-Models-Read-What-They-Write,
Do-Code-Semantics-Help (contrary evidence), Executable Counterfactuals (RL>SFT transfer — do not dismiss). Contribution = a predictive
selection rule + advantage over close practical controls, NOT "first to use execution values."

## 7. Stopping rules (pre-committed)
- true-irrelevant ≈ Tvalue → narrow to generic auxiliary-supervision, drop the data-flow claim.
- random-state ≈ distinguishing-state → keep the simpler method, drop the expensive selector.
- extra-terminal-examples ≈ process supervision → it's a supervision-allocation result, not process-specific.
- process-RL adds nothing over supervised continuation → keep SFT, don't force RL.
- effect confined to one synthetic setting → report as a controlled finding, don't claim generality.

## 8. Figures
F1 supervision-content ablation (B/relevant/irrelevant/mask/scramble) · F2 method × crossed difficulty · F3 accuracy vs compute & aux-tokens ·
F4 terminal vs process RL + all-failure-group diagnostics · F5 fresh structural OOD + natural code + 2nd model family.

## 9. Honest status
Positive method (Tvalue) is real, seed-robust, placebo-validated, capability-gated. The paper's strength is a PREDICTIVE account of which
verified process signals to provide and when they help — not a law and not a guaranteed award. High-upside outcome: a selection rule that
survives close baselines + external replication with a real compute advantage.
