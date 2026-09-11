# Evidence Extract — everything usable in the paper (verified self-improvement / coverage program)
_2026-09-11. Source: ADAPTIVE_FORGETTING_RESULTS.md §79–§114. Every number below is a recorded ledger result (not re-derived).
Organized by paper role: A failures→motivation · B theory/theorems · C empirical positives · D instruments · E mechanism ladder ·
F validations/controls · G retractions (honesty). Section tags (§NN) are the ledger anchors._

============================================================
## A. FAILURES / NULL BATTERY  (motivation + "obvious methods don't survive controls")
Each row: apparent gain → the decisive control that killed it → verdict.
- **Self-repair from execution feedback (§87, §92b):** recovers only 3% (1.5B) / 16% (7B) / 40% (14B) of the base pass@k=0 frontier;
  capability-gated, no distillation gain shown → small, not a fix.
- **Archive / source preservation (§82, §84b P2):** +0.025 at R2 but crossover by R3; current-only 0.657 vs +archive-on-uncovered 0.637 vs
  archive-on-shared 0.607 (dilution) → NULL.
- **Delayed-value / next-generation-teaching update selection (§85 P3):** J_next from low-immediate source 0.69 vs high-immediate 0.82 →
  next-gen value is monotone with immediate accuracy → NULL (immediate acc suffices).
- **Difficulty / curriculum escalation (§90b):** escalation d5→d6→d7 = 0.750 vs fixed-easy d5 0.695 (apparent +0.055) BUT fixed-HARD
  d7→d7→d7 = 0.770 > 0.750 → the gain was train-test proximity → NULL vs fair control.
- **Decomposition-distillation (§95/§95b/§104):** +0.06 (1.5B) / +0.02 (7B) / +0.015 (14B) OOD, BUT token-instrumented cost = 2.1× an RFT
  round, ~1.1–1.5× cost-to-matched-accuracy → modest, NOT an efficiency win; and composition is trivial in the linear domain (§103b) so it is
  a local-solving scaffold, not a composition teacher.
- **Self-expanding decomposition loop (§99/§99b):** DEC dominates RFT each round (R2 +0.13) but converges (R3 +0.015); depth-8 dec 0.810 vs
  rft 0.795 → does NOT diverge.
- **Recomposition (post-hoc rewiring) (§107):** held-out DAG wiring B(plain refs) 0.83 > C(random-aug) 0.81 > F(recomposition) 0.77 → rewiring HURTS → FALSIFIED.
- **Verified execution-VALUE supervision (§109/§110/§112):** Tvalue 0.857 vs B 0.787 (+0.07, seed-robust) LOOKED like a positive method, BUT
  the placeholder control Tmask 0.850 ≈ Tvalue → it is STRUCTURAL ANNOTATION, not the values; scrambled Tscram 0.780 ≈ B; irrelevant Tirrel
  0.698 HURTS → formatting effect (scratchpad-adjacent), NOT verified process supervision.
- **Estimator knobs (negatives, weighting) (§114):** R3 zero-negatives 0.548, R4 success-count weighting 0.567, both ≈ GRPO 0.555 (≠ RFT 0.745)
  → the RL OBJECTIVE is NOT the cause of the transfer gap.
- **Cross-domain (math) coverage mechanism (§86, §88):** math-instruct base run DISCARDED (no headroom = unfair); fair weak-base math shows
  iterative compounding (+0.046) but the coverage-CAUSAL decomposition is NULL (full 0.500 ≈ blocked 0.493) → mechanism is domain/headroom-dependent.

============================================================
## B. THEORY / FORMAL RESULTS  (modest, correct — no universal law)
- **Success-gradient identity (§111):** for a parameter-independent binary verifier V, p_θ(x)=E_{y~π}[V]; then
  ∇_θ p_θ(x) = p_θ(x)·E[∇_θ log π(y|x) | V=1]. ⇒ within-prompt-normalized RFT gradient on current-policy successes = ∇log p_θ(x), and the
  binary-reward policy gradient is p_θ(x) × the SAME direction. RFT and outcome-RL are NOT distinct gradient families ⇒ any transfer gap is
  procedural (weighting/pooling/normalization/collection), not objective-fundamental. [Grounds §114's negative.]
- **Operational reachability + Headroom Bound (§101, supersedes buggy §98):** R_{C,τ}(G)={x : s_G(x;C)≥τ}, s = Pr(verified solution within budget C).
  Thm 1 (support-boundedness): verified-only training gives NO direct gradient on unreached x; gains there = MEASURED generalization (not a free set).
  Cor 2 (headroom-gating): new-coverage gain ≤ headroom · transferable-value ⇒ predicts the size/domain curve; reachability ≠ reliability.
  Thm 3 (acquisition-cost factorization): with m components, retry cost c_i, success q_i, local verify v_i — E[C_whole]=(Σc_i+v_whole)/Πq_i vs
  E[C_local]=Σ(c_i+v_i)/q_i + assembly; decomposition helps only under stated assumptions (valid decomp, sound local verify, reusable comps).
- **Mixed-group formula (§79/H3):** P_mixed = 1 − p^G − (1−p)^G (fraction of GRPO groups with both a success and a failure); at p=0.01,G=8 ≈ 0.077
  → quantifies when terminal GRPO groups carry zero reward-advantage signal.
- **Unbiased dead-group estimator (§79):** D(G)=E[p^G+(1−p)^G] with unbiased subset form [C(c,G)+C(K−c,G)]/C(K,G); plug-in p̂^G+(1−p̂)^G is
  upward-biased for G>1. Use D_fail=E[(1−p)^G] (regime-separated), not merged D.
- **Submodular coverage of state-selection F(Q) (§98/register):** weighted max-coverage; monotone submodular under equal costs (greedy 1−1/e);
  budgeted costs need budgeted treatment. (Classical structure, not a new theorem.)
- **Info-theoretic caveat (§111):** given program+input+interpreter, the execution trace is deterministic (adds no Shannon information); any SFT
  benefit is bounded-learner USABILITY, not new information — consistent with §112 (values don't help beyond format).

============================================================
## C. EMPIRICAL POSITIVE RESULTS  (the paper's spine)
### C1. RFT ≫ GRPO for OOD from an identical checkpoint (§91, §113, §113b)
- 1.5B comp depth-7 OOD, 3 seeds: shared 0.555 → RFT+ 0.745/0.750/0.760 (mean 0.752); GRPO(group) 0.555/0.530/0.535 (0.540);
  GRPO(no-std) 0.580. Gap +0.21, non-overlapping.
- GRPO learns-but-doesn't-transfer (§91c): train reward 0.375 → ~0.60 peak while OOD flat.
- Generality (RFT+ − GRPO-group): deepseek-coder-1.3B +0.095 (0.495 vs 0.400), Qwen-Coder-1.5B +0.190, Qwen-Coder-3B +0.110 (0.905 vs 0.795),
  Qwen-Coder-7B RFT+ +0.025 (0.895; GRPO side pending server-mode). Sign invariant across 2 families × 4 sizes; magnitude shrinks with capability.
- Math cross-domain (§91e): shared 0.468 → RFT+ 0.489 > GRPO 0.474 (direction replicates, headroom-limited).
### C2. Coverage is causal (removal-controlled) (§84, §89)
- §84 (depth-7): full R3 bank 0.65 vs coverage-blocked 0.42 (+0.23) vs random-removal 0.59 (+0.06 size).
- §89 1000-panel (n=400 OOD): full 0.790 vs blocked 0.625 (+0.165 NEW-COVERAGE) ≈ random-removal 0.795 (NOT volume). Coverage R1 549→R2 734→R3 866.
### C3. Headroom gates it — 4-size curve (§93, §93b, §97, §105, §105b)
- full−blocked (compositional, fixed depth): 1.5B +0.165 → 3B ~0 (0.885 vs 0.890) → 7B ~0 (0.890 vs 0.885) → 14B ~0 (0.940 vs 0.930).
- Return-with-headroom: 7B depth-9 (79% base cov) −0.010; 14B depth-9 (91%) +0.005; 7B depth-14 (58% cov = real headroom) +0.020.
- Diverse-dataset dissociation (§97): iterative COMPOUNDING replicates everywhere (math +0.046, GSM8K +0.038 R3 vs matched base-3x), but the
  coverage-CAUSAL full≫blocked effect appears ONLY at 1.5B-compositional (high headroom).
### C4. Frozen vs improving sampler (§82b, §83b) — matched cost
- Improving iterative current-only: R3 0.753, R4 0.798 vs base-3x matched-cost one-shot 0.513 (+0.24); coverage 225→290→336→368.
- Fresh depth-7: frozen-base matched-round 0.427→0.487 vs improving 0.400→0.667 (+0.18). Frozen saturates; improving compounds.
### C5. Decomposition-distillation (modest positive, for completeness) (§95/§95b)
- Distilling decompose-recovered frontier solutions beats plain RFT single-shot: 1.5B 0.72 vs 0.66 (+0.06), 7B 0.90 vs 0.88 (+0.02); ∝ headroom.

============================================================
## D. INSTRUMENTS (reusable contributions)
- **DAG composition testbed (§106, §107b):** branching/merge/intermediate-reuse tasks where data-flow WIRING is provably the bottleneck.
  Oracle-components diagnostic: give correct per-op code, ask to wire → recovery of base zero-success cohort = 9.9% at 1.5B (27/272), 52.7% at
  7B (59/112). Contrast: LINEAR pipelines recover 100% at 1.5B/7B/14B and depths 7/9 (§103b) — composition is free there. ⇒ composition is a
  genuine, capability-gated bottleneck only in non-linear structure. (comp_dag.py)
- **Unbiased dead-group / D_fail estimator (§79)** (comp_pcount.py) — see B.
- **Matched controls toolkit:** same-prompts, matched-cost (token-instrumented), matched-accept-rate, count-matched, random-removal, fixed-hard,
  placebo (scrambled/mask targets), fresh-eval, frozen-base matched-round.

============================================================
## E. MECHANISM LADDER — "the objective is not the cause" (§114, grounded in §111)
From the shared checkpoint (1.5B comp, depth-7 OOD, 3 seeds): R0 GRPO(group) 0.555; R1 GRPO(no-std) 0.580; R3 zero-negatives 0.548;
R4 success-count weighting 0.567; RFT+ 0.745. ⇒ Removing negatives (H1.1) and adopting RFT's implicit success-count weighting (H1.2) BOTH leave
GRPO at baseline → the two ways the estimators differ do NOT close the gap. R4 IS RFT's implicit weighting applied ONLINE → the lever is
offline-broad-replay / training-prompt COVERAGE, not the estimator advantage shape. [PENDING: R6/coverage factorial running — does 4× per-step
prompt coverage close it?]  Prior partial: std-norm removal ≈ 24% of the GSM8K→MATH gap (§56-E1).

============================================================
## F. VALIDATIONS / CONTROLS THAT SURVIVED  (what makes the positives trustworthy)
- Seed CIs: RFT+ vs GRPO at 1.5B, 3 seeds, non-overlapping (§91d); Tvalue vs B, 3 seeds, non-overlapping (§109).
- Removal control (§89): full ≈ random-removal, ≫ coverage-blocked → identity of removed examples matters, not volume.
- Placebo control (§112): Tmask (no values) ≈ Tvalue; Tscram ≈ B → isolates format from content.
- Matched-cost (§82b/§104): iterative vs base-3x; token-instrumented DEC vs RFT.
- Fair-control reversal (§90b): fixed-hard > escalation → kills the proximity confound.
- GRPO sanity (§91c): train reward rises → GRPO learns, so flat OOD is a transfer failure not an optimization failure.
- Discarded-unfair-comparison discipline (§86): math-instruct ceiling run deleted, re-run on fair weak base.

============================================================
## G. RETRACTIONS (honesty box — verbatim for appendix)  (§79, §111, §112, §114)
Retracted, with reason: MaxRL "weak-solver-best-teacher" (coverage-fragile; didn't reproduce at 7B); "imitation≠RL reversal" (n=1 noise);
diagnostic-composition curricula / sparse bridges (coverage/freq artifact); verifier-resolution (bank-size artifact); RL-as-explorer
(fair-sampler control); source-preservation as a method (crossover/dilution); delayed-value selection (monotone with immediate acc); difficulty
escalation (fixed-hard control); recomposition (hurts wiring); execution-value supervision (placeholder ties); the §44 mass-placing axis
(gradient-reduction/LR artifact, §56-E0a); advantage-density "as a law" (biased estimator; confounded points); the universal "amplify-not-expand"
law and "RFT expands / RL reweights" mechanism (success-gradient identity + measured generalization, §111). The paper claims NONE of these.

============================================================
## H. OPEN / PENDING (in-flight, not yet evidence)
- Coverage/R6 factorial (4× per-step prompt coverage; does it close the gap?) — RUNNING.
- Math-domain estimator ladder (R3/R4 cross-domain) — RUNNING.
- 7B/14B GRPO server-mode (complete the generality grid's GRPO side).
- Preregistered out-of-sample predictions (H2.1 headroom, H2.2 D_fail) — NOT yet run; required to call the curve a "law".
