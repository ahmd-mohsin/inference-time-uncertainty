# Counterfactual Recovery Optimization — Paper Narrative (working draft, 2026-09-02)

*Curated end-to-end: motivation → what we measured → what we learned (incl. the nulls) → the method
we are now building. This is the spine to grow the NeurIPS submission on; the next experiment wave
plugs into §7.*

---

## 0. One-paragraph thesis

RLVR (RL with verifiable rewards) is usually described as causing "reasoning mode collapse," and a
growing literature tries to *preserve diversity* (SetPO, DPH-RL, DMPO, Uniqueness-Aware RL). We show, on
Qwen-Math, Llama-3.1 and DeepSeek, that the collapse is **routing compression, not capability erasure**
— the model stops *choosing* many strategies but can still *execute* them — and, crucially, that
**preserving or injecting diversity yields no downstream benefit** (coverage, adaptation, uncertainty,
pass@k) because reasoning modes are functionally redundant. The value of the latent repertoire appears
only in one regime: **after a failure**. A reasoning alternative matters exactly when it succeeds where
the model's own retry would fail. We formalize this as a **counterfactual recovery advantage**
`A^rec(s,z) = Q(s,z) − Q(s,retry)` and build **Counterfactual Recovery Optimization (CRO)**: learn *when*
to switch vs retry, *how* to switch (latent recovery options), and consolidate causally-useful
recoveries back into the base policy. The effect is domain-gated by genuine functional complementarity —
strong in executable code, absent in math — which the theory predicts.

---

## 1. Motivation

- **The field's premise:** RLVR sharpens pass@1 but "collapses" the diversity of reasoning, and this is
  assumed to be a capability problem worth preventing → a wave of diversity-preserving RL methods.
- **The unexamined gap:** "mode collapse" is measured as a *marginal* quantity (mode probability /
  trajectory diversity). That conflates two very different things:
  - **Routing** `ρ(m|q)` — how often strategy m is *chosen*.
  - **Competence** `c(m,q) = P(correct | do(M=m))` — whether it can still be *executed*.
  A drop in marginal mode mass cannot distinguish "stopped choosing it" from "can no longer do it."
- **Our question:** *What actually collapses under RLVR — and does any downstream task benefit from the
  diversity everyone is trying to preserve?*

---

## 2. The decomposition (measurement contribution)

For problem q and reasoning strategy m, separate three factors — a mode contributes only if all are
non-trivial (the **ACV** view):
- **Accessibility** `ρ(m|q)` — routing (measured from free generations + classifier, and from log-prob
  of entering the strategy).
- **Competence** `c(m,q)` — forced-mode correctness (prefix-forced, high adherence).
- **Value / complementarity** `v(m|q)` — does m solve what other strategies cannot.

Non-identifiability result: from marginal `s_m ≈ ρ_m·c_m` alone you cannot recover ρ or c, so **marginal
mode collapse cannot establish capability loss.**

---

## 3. What we found about the collapse (mechanism — validated)

- **F1 Routing compresses (ρ↓).** Behavioral routing entropy falls (Qwen 3.51→2.71 effective
  strategies/problem); log-prob of entering named strategies drops −0.34 nats/tok (Qwen), −0.67 (Llama);
  the support-floor preserves routing (≈ base). Two independent measures agree.
- **F2 Competence is preserved/improved (c↑).** High-N (32 forced samples), base_c-controlled: on modes
  the base is genuinely competent at (base_c ≥ 0.2), grpo Δc = **+0.077** (Qwen), **+0.169** (Llama),
  87–94% retained. (An earlier 4-sample pass showed spurious erosion — regression-to-mean; removed at N=32.)
  DeepSeek is the boundary case (train/eval mismatch).
- **F3 Δlogρ-vs-Δc map.** Of routing-suppressed (q,m) pairs, ~90% keep/raise competence (QII); only ~10%
  are true erasure — and that split is largely base-competence level, not a clean mechanism (honest caveat).
- **F4 Theory.** Softmax routing gradient `∂J/∂z_m = ρ_m(c_m − J)` suppresses below-average routes; the
  shared-parameter executor gives `Δc_m ≈ η Σ_j ρ_j⟨∇c_m,∇c_j⟩`, so competence can *rise* as routing
  falls. (Direct gradient-cosine measurement attempted; deferred — single-GPU OOM.)

**Message:** *knowing ≠ choosing.* RLVR compresses the routing distribution while preserving the
conditional reasoning repertoire.

---

## 4. Does diversity help downstream? Four honest nulls

We tested the "diversity is useful" hypothesis unusually hard. All negative, and all *unconditional /
single-episode*:
- **N1 Adaptation / optionality** — collapsed vs preserved forks adapt equally (or collapsed better) to
  new math shifts. No penalty from collapse.
- **N2 Coverage / stratified pass@k** — strategy-stratified sampling ≈ iid (Δ ∈ [−0.05,+0.01]); pass@32
  identical; oracle single-strategy < iid. Across Qwen/Llama/DeepSeek, math *and* code.
- **N3 Functional redundancy** — mean per-mode complementarity v ≈ 0.014; 8–9 of 14 strategies solve each
  solvable problem; almost none uniquely.
- **N4 Diagnostic uncertainty (D_CF)** — forced-strategy answer-disagreement never beats iid
  self-consistency for error detection; routing-masking law weak/inconsistent.

**Why:** marginal *and* diagnostic diversity are decoupled from capability because modes are functionally
redundant. This reconciles our nulls with the literature's positive diversity results: **diversity helps
only where functional complementarity `v` is high — which does not occur unconditionally in real tasks.**

---

## 5. The turn: diversity's value is conditional on failure

The nulls are all *unconditional*. The right object is **failure-conditioned**:
`v⁻_m = P(m succeeds | default failed)` — which can be ≫ 0 even when unconditional v ≈ 0.

- **Positive result (code):** on problems the default route *fails*, switching to a different reasoning
  strategy recovers **more than iid-retry at matched budget** — mean +0.17, up to +0.39 on HumanEval
  (Qwen-Instruct, Llama). First downstream win in the project.
- **Domain-gated:** in math the same test is **negative** (G_switch −0.04 to −0.17) — forced math
  "strategies" are lower-quality reasoning with no complementarity. Code "strategies" are genuinely
  different algorithms with complementary failure modes.

**Bottom line:** *monoculture is efficient before failure and a liability after it.* Exploit first;
diversify only after evidence of failure — and only where routes are genuinely complementary.

---

## 6. Counterfactual Recovery Atlas (justifies the method)

Scaled failure × recovery-option landscape on executable code (Qwen-Instruct + Llama × MBPP×2 +
HumanEval; 8 engineering recovery options; exec-verified). Over default-failed problems:
- **oracle − best-fixed = +0.22** (per-failure router headroom over any single strategy; +0.14 over
  iid-retry).
- **21%** of failures have a recovery option beating iid-retry by > 0.2.
- **Heterogeneous specialization:** 7–8/8 options are each the best recovery for some failure
  (`algo_replace` modal; `boundary`, `defensive`, `root_cause`, `complexity`, `builtins` win distinct
  subsets). **Different failures need different fixes.**

⇒ **GO** to build a per-failure recovery method.

---

## 6b. CONFIRMED POSITIVES + the pivot to Active Recovery Diagnosis (2026-09-03)

Data now spans code_recover (MBPP+HumanEval × Qwen-Instruct/Llama/Qwen-Coder/Qwen2.5-Coder-14B, incl.
denser n_rec=24) + a complete TACO Atlas (4 families × easy/med/hard) + a math negative control — all
death-proofed to `rl_training/runs_pulled/`. Two results are locked in:

- **POSITIVE — matched-budget diversified portfolio beats iid-retry.** On default-FAILED problems, spending
  a fixed retry budget B *across* recovery options recovers more than B iid retries. Fair B=12 subsample
  from each pool: **MBPP+HE portfolio 0.264 vs iid 0.192 = +0.072** (n=834 failures); pooled code +0.028;
  TACO +0.012. Per-family at denser B=24: Qwen-Instruct **+0.134**, Qwen-Coder +0.060, Llama +0.052. This
  is the project's robust downstream win.
- **NEGATIVE (honest) — the learned single-option router is a NO-GO.** Predicting the one best recovery
  option from the failed state (TF-IDF(question+err+failed-code)→LR, 5-fold CV) gives routed recovery
  0.091 < iid 0.192; closes <30% of the oracle gap. The oracle headroom is mostly *ties* (most routable
  failures yield to many options), so "which option" is not identifiable from `s_F`. (`CRO_ROUTER_RESULTS.md`.)
- **Domain-gated** (negative control): switching *hurts* in math — G_switch Llama −0.174, Qwen-Coder −0.091,
  Qwen-Instruct +0.000.

**The pivot the router failure forces (award-level thesis):** the failed state is *partially observable*.
Recovery actions have BOTH control value and **information value** — each attempt is also an experiment
that diagnoses the latent failure mechanism Z (wrong-algorithm / invariant / edge-case / complexity /
spec). So effective recovery is **sequential experimental design over complementary policies**, not
retry, static diversity, or one-shot routing. Three methodological objects (§7 rewritten around these):
  1. **Failure-Surface Diversity** `D_fail = 1 − mean_{i<j} Corr(E_i, E_j)` (E_z = 1{option z failed}) —
     the claim: *useful diversity is diversity of failure surfaces, not of generated text.* Test: D_fail
     predicts portfolio gain across cells while option-entropy / mode-count do not. (Explains every prior
     null: math has semantically distinct strategies but C_ij≈1 ⇒ v≈0; code has C_ij<1 ⇒ portfolio helps.)
  2. **Marginal recovery coverage** `Δ_i(z|S) = F_i(S∪{z}) − F_i(S)` — train options for *complementary*
     failure coverage, not reward or A^rec.
  3. **Active diagnostic recovery** `Q(h_t,a) = Q_solve(h_t,a) + β·Q_info(h_t,a)`, `Q_info ≈ I(Z;O_t|h_t,a)`
     — the best FIRST recovery action ≠ the best standalone action; pick early actions that most reduce
     uncertainty about what to try next. Target: recovery predictability rises with observations
     (e.g. 30%→48%→65%), and adaptive beats every static portfolio by +8–15 pts at equal compute.

(Offline analyses of D_fail / marginal coverage / adaptive-vs-static simulation on the existing recovery
matrix are in `analyze_failure_surface.py` → `FAILURE_SURFACE_RESULTS.md`; sequential/active-diagnosis
data collection is the on-GPU harness `seq_recover.py`.)

## 7. Methodology — CRO: Counterfactual Recovery Optimization  ⟵ (next experiment wave plugs in here)

**Rollout topology change vs GRPO:** instead of G iid attempts from x, run x → y₀ → (on failure) a set
of *matched counterfactual branches*: iid-retries **and** alternative recovery options, from the *same*
failure state.

**Central quantity — counterfactual recovery advantage:**
`A^rec(s,z) = Q(s,z) − Q(s,retry)`  — a repair is useful only if it beats the model's own retry (not just
if it's correct). Rewards *marginal recovery*, not generic diversity.

**Components:**
1. **Failure-conditioned gate** `g_φ(z | s_F)` over actions `{retry, z₁…z_M}` — RETRY is an explicit
   action (math ⇒ sometimes retry beats switch). Trained toward `q*(z) ∝ exp((A^rec − λ·cost)/τ)`.
2. **Learned latent recovery options** `π_θ(y | s_F, z)` (soft prompts / LoRA / discrete latents)
   replacing hand-written strategies; inspect emergent options post-hoc.
3. **A^rec-weighted executor learning** — train recovery policy on `[A^rec]_+`-weighted successful
   recoveries (those that worked where retry did not).
4. **Counterfactual consolidation** — distill high-A^rec recovered solutions back into the *initial*
   policy: failure → discover alternate solution → verify → amortize into pass@1 (self-improving loop /
   auto-curriculum).

**STATUS UPDATE (2026-09-03, experiment #1 run — see `CRO_ROUTER_RESULTS.md`):** the learned single-option
router `g_φ(z|s_F)` is a **NO-GO** — held-out CV routed recovery underperforms even iid-retry
(router−iid = −0.11 MBPP, −0.02 TACO) and closes <30% of the oracle gap. The oracle gap is mostly *ties*
(most routable failures are recovered by many options; label collapses to `root_cause`), so the per-problem
best option is not option-routable from `(question+fail_code+fail_err)`. **Reframe that survives:** the real
recovery win is **matched-budget diversification** (spread the retry budget across options — the earlier
+0.04…+0.27) NOT routing to one option. So component 1 becomes a *retry-vs-diversify* gate (binary) +
failure-conditioned entropy increase, not an 8-way option selector. Domain-gating reconfirmed (math
G_switch Llama −0.174 / Qwen-Coder −0.091 / Qwen-Instruct +0.000).

**GRPO SELF-REPAIR RESULT (2026-09-03, `GRPO_SELF_REPAIR_RESULTS.md`):** trained Qwen2.5-Coder-7B via GRPO
(exec-verify reward, error_only rollout, mbpp_train) → eval error_only on held-out mbpp-test/he. **NULL /
NO-GO:** on the reliable set (mbpp, n~127) trained≈base (seq−iid −0.070→−0.056; still below iid); the
HumanEval +0.227 is confounded (trained iid dropped 0.652→0.500, n~22). RL did not amplify the antidote at
this scale. A scaled retry (2× data/steps, marginal-recovery reward) is running. **Bottom line: the fix is
prompting (hide the buggy code), not RL** — a clean, honest result. The paper's contribution is the
mechanism + antidote, not a trained self-repair win.

**Headline equation reviewers remember:** `A^rec(s,z) = Q(s,z) − Q(s,retry)`.
**Conceptual contribution:** *failures are not zero-reward trajectories; they are branch points at which
the value of alternative policies becomes identifiable.*

**Positioning vs prior art** (must differentiate): CodeRescue / PROBE (recovery routing over
reflect/replan/escalate), Fission-GRPO & CARE (learn from failures), BPO (branch at intermediate
states), BAPO (reuse hard examples). Our novelty = the *combination*: matched retry-vs-switch
counterfactuals + `A^rec` objective + learned latent recovery options + failure-gate with RETRY action +
A^rec-weighted consolidation.

---

## 8. Evaluation plan (to be filled by the next wave)
- **Primary:** competitive programming (CodeContests / TACO / APPS) — needs a stdin/stdout exec verifier
  (not yet built; current verifier = HumanEval `check()` + MBPP asserts).
- **Secondary:** NP/combinatorial optimization, controlled planning (high genuine complementarity).
- **Negative control:** math (CRO should learn "don't switch").
- **Baselines:** GRPO, DPH-RL, SetPO, CARE, Fission-GRPO, BPO, BAPO.
- **Target results:** ≥1.5–2.5× retry recovery on failures; learned router closes >60–70% of oracle gap;
  +8–15 pts end-to-end on initially-failed tasks at matched compute; consolidation lifts pass@1;
  frontier expansion (previously-unsolved → solved); recovery scaling beats iid-retry scaling.

---

## 9. Honest scope / limitations
- The mechanism (ρ↓/c↑) is solid on Qwen+Llama; DeepSeek boundary; gradient-cosine not yet measured.
- All diversity value is **domain-gated by complementarity** — real in code, absent in math (math is the
  clean negative control, not a failure of the idea).
- Recovery Atlas frac(A^rec>0.2)=0.21 is borderline; larger + harder benchmarks needed to firm it.
- Qwen-Coder code jobs recurrently crash shards on the main node (infra, not scientific).

## Source records
`FINDINGS_AND_NOVELTY.md` (ACV, F1–F7), `ROUTING_VS_COMPETENCE_RESULTS.md`,
`ADAPTATION_OPTIONALITY_RESULTS.md`, `DCF_DIAGNOSTIC_DIVERSITY_RESULTS.md`,
`MONOCULTURE_RECOVERY_RESULTS.md`, `CRO_ATLAS_RESULTS.md`, `STATUS_AND_ALL_RESULTS.md`.
Harnesses: strategy_probe / route_logprob / free_route / stratified_passk / code_passk / code_recover /
math_recover / cf_disagree / analyze_probe / analyze_acv (+ go_*.sh launchers).
