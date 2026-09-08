# Consolidated Status & Results — RLVR Self-Repair (as of 2026-09-03)

One page for the whole project. Detailed writeups linked at the end.

---

## TL;DR

- A diffuse "diversity under RLVR" question was sharpened, through a chain of honestly-reported
  negatives, into one deployable, counterintuitive result: **showing a code model its own failed code
  degrades its next attempt (anchoring, seq−iid = −0.11); hiding the code and keeping only the error
  removes the penalty (→ parity with iid, +0.018 ± 0.069 over 17 runs).**
- Everything sophisticated I tried FAILED to beat iid resampling: single-option router, failure-surface
  diversity, sequential self-diagnosis (full history), GRPO self-repair (v1 + v2). Iid is a stubbornly
  strong baseline for RLVR'd code models.
- The paper's spine is the **anchoring mechanism + hide-the-code antidote**; the negatives are reported.

**One-line recipe for practitioners:** *on a code failure, give the model the error signature (or
"try again"), never paste back its own failed code — it never hurts, helps weaker models most, and is
strongest at low retry budgets.*

---

## Findings (locked, with numbers)

| # | Result | Bench / n | Effect | Verdict |
|---|---|---|---|---|
| F1 | Routing compresses under RLVR | Qwen+Llama+DeepSeek | H(ρ) 3.51→2.71 eff. strategies; log-p of strategy-prefix −0.34/−0.67 nats/tok | **Confirmed** |
| F2 | Competence preserved / rises | 14 forced strategies, N=32 | Δc = +0.077 Qwen, +0.169 Llama | **Confirmed** |
| N1 | Adaptation optionality | math shift | ≤ base | **Null** |
| N2 | Stratified pass@k coverage | math+code, k up to 32 | Δ ∈ [−0.05, +0.01] | **Null** |
| N3 | Functional complementarity | 14 strategies | mean v ≈ 0.014 | **Null** |
| N4 | Diagnostic diversity (D_CF) | math | never beats iid self-consistency | **Null** |
| P1 | Matched-budget portfolio > iid (code) | MBPP+HE, B=12 | +0.072 (fair), HE per-family +0.14…+0.33 | **Positive** (budget-gated; shrinks by n_rec=48) |
| P1' | Math negative control | math_recover | G_switch −0.17 / −0.09 / +0.00 | **Confirms domain gating** |
| **A1** | **Anchoring mechanism** (full-mode self-repair) | 21 cells | **seq−iid = −0.110 mean** | **Robust, headline** |
| **A2** | **Hide-code antidote** (error-only) | 17 cells | **+0.018 ± 0.069 (parity)** — ll_mbpp +0.072, qi_mbpp +0.061, qc14_mbpp −0.092 | **Positive, model-gated** |
| A3 | Antidote regime | T={3,6,10} | Advantage FRONT-LOADS at low budget | **Confirmed** |
| A4 | Error-label variant | +0.00 vs error-only | Naming the bug class adds nothing | **Null** |
| R1 | Learned single-option router | 4 families | routed 0.078 vs iid 0.184; closes <30% oracle gap | **NO-GO** |
| R2 | Failure-surface diversity `D_fail` | 26 cells | corr(D_fail, portfolio-iid) = −0.35 (wrong sign; math has highest D_fail but negative gain) | **Refuted** |
| R3 | Full-mode sequential self-diagnosis | 4 families × MBPP/HE | anchors; seq−iid down to −0.29 | **NO-GO** |
| **R4** | **GRPO self-repair (v1 + v2)** | Qwen-Coder-7B, LoRA, exec-verify reward | v1 mbpp seq−iid −0.056; v2 mbpp −0.016 (SEQ 0.434 < base 0.500); HE noisy | **NO-GO at this scale** |

---

## Paper spine (one-paragraph thesis)

RLVR does not erase reasoning capability — it **compresses routing while preserving competence**
(ρ↓, c↑). *Diversity per se* buys nothing downstream (four unconditional nulls) because modes are
functionally redundant; conditioning on failure, a **matched-budget diverse portfolio** modestly beats
iid retry in code (domain-gated: negative in math). Trying to route to a *single* best recovery from the
failed state fails — the failed state is not option-identifying (ties dominate). But the *mechanism* by
which self-repair fails is clean: **feeding a model its own failed code makes it worse (anchoring:
−0.11)**; hiding the code and keeping only the error **restores parity with iid** and helps weaker
models at low retry budgets. Training the model to exploit its own error feedback via GRPO (execute-
verify reward, LoRA) did not clear the +0.10 bar at 200 or 400 steps: **the fix is prompting, not RL, at
this scale**.

---

## Method-level headline (what to cite)

**Anchoring in RLVR code self-repair.** Given `(problem, failed_code, error)`, a sequential attempt
conditioned on the FULL failed history recovers ≈0.11 fewer failures per attempt than plain iid retry
across families/benches. The failure is not incapacity — it is *anchoring on the buggy prefix*. Removing
the code from the feedback (keeping the error signature) removes the penalty. Naming the bug class
(SyntaxError / wrong-output / etc.) adds nothing beyond the raw error text; RL training on this signal
did not amplify it (v1 + v2 null). Interpretation: RLVR-tuned code models cannot exploit their own
execution feedback zero-shot — the prompt-engineering fix outperforms the training fix at this scale.

---

## Everything that was tried, and the verdict

| Direction | Method | Verdict |
|---|---|---|
| Routing vs competence decomposition | forced-mode competence + logp of strategy prefix + free-generation routing | ρ↓/c↑ confirmed |
| Preserved-diversity RL (SetPO / DPH-F floor) benefit | pass@k, adaptation, functional rank | Null (4×) |
| Diagnostic uncertainty via strategy-disagreement | D_CF vs iid self-consistency | Null |
| Matched-budget diverse portfolio | code_recover / TACO Atlas (8 recovery options) | Positive (code), budget-gated; domain-gated (math negative) |
| Learned single-option recovery router | TF-IDF(question+err+failed_code) → LR, 5-fold CV | NO-GO (closes <30% oracle gap; label collapses to `root_cause`) |
| Failure-Surface Diversity `D_fail` | error-decorrelation across options → portfolio gain | Refuted (r = −0.35; math has highest D_fail but negative gain) |
| Marginal option coverage | greedy set-cover | Real but modest (0.048 → 0.160 over 8 options) |
| Offline adaptive-vs-static portfolio | binary-outcome-conditioned greedy | Weak (+0.012) — lower bound |
| Sequential active self-diagnosis (full history) | multi-turn seq_recover, show code + error | Anchors, hurts (seq−iid up to −0.29) |
| Anchoring antidote (error only, hide code) | seq_recover --diag-mode error_only | Removes penalty → parity with iid; helps weaker/low-budget |
| Failure-class labelling | error_label variant | Null over raw error |
| GRPO self-repair v1 (200 steps, 405 prompts) | TRL, LoRA, exec-verify reward | NO-GO (mbpp seq−iid −0.056) |
| GRPO self-repair v2 (400 steps, 495 prompts) | same + scaled | Confirms NO-GO (mbpp seq−iid −0.016) |
| Math negative control (recovery) | 3 families, 14 math strategies | Domain-gating confirmed (G_switch < 0) |
| TACO competitive-programming Atlas | 4 families × EASY/MED/HARD | Router headroom real on easy/med, collapses on hard |

---

## Deliverables on this laptop (all death-proofed)

Reports (`DELIVERABLES/report/`):
- **`STATUS_AND_ALL_RESULTS.md`** ← this file
- `PAPER_NARRATIVE_CRO.md` — curated paper spine
- `SEQ_DIAGNOSIS_RESULTS.md` — anchoring + antidote (headline)
- `GRPO_SELF_REPAIR_RESULTS.md` — v1 + v2 null
- `FAILURE_SURFACE_RESULTS.md` — D_fail refuted, marginal coverage
- `CRO_ROUTER_RESULTS.md` — router NO-GO
- `MONOCULTURE_RECOVERY_RESULTS.md` — matched-budget portfolio positive
- `CRO_ATLAS_RESULTS.md` — 8-option recovery Atlas
- `DCF_DIAGNOSTIC_DIVERSITY_RESULTS.md`, `ADAPTATION_OPTIONALITY_RESULTS.md`, `ROUTING_VS_COMPETENCE_RESULTS.md`, `FINDINGS_AND_NOVELTY.md`

Data (`rl_training/runs_pulled/`):
- `seq/` — 21 full-mode + 17 error-only + 4 GRPO eval seq_qc{BASE,V2}* jsons
- `cro_router/`, `taco_atlas/`, `math_negctrl/` — Atlas cells
- `repair_ckpt/qc_final_adapter.safetensors` (v1) + `qc_v2_adapter.safetensors` (v2) + configs

Harnesses (`rl_training/`):
- `strategy_probe.py`, `route_logprob.py`, `free_route.py`, `stratified_passk.py`, `cf_disagree.py`,
  `code_passk.py`, `code_recover.py`, `math_recover.py`, `taco_recover.py` (TACO parquet fix),
  `analyze_failure_surface.py`, `build_cro_router.py`, `seq_recover.py` (+ `--diag-mode`),
  `dump_repair_data.py`, `train_repair_grpo.py` (exec-verify reward + LoRA)
- Launchers: `go_probe/route/free/sp/ff/pr/lsp/cf/code/recover/mrec/taco/seq/dump/repair.sh`
- Bootstrap: `bootstrap_fast.sh`, `fetch_big.py`

---

## Open moves (only if you re-engage)

1. **Marginal-recovery reward GRPO.** Current GRPO rewards any-repair; a marginal-recovery reward would
   only credit repairs that succeed *where a matched iid retry fails*. Concretely: for each rollout,
   sample K matched iid retries offline; reward = 1[repair passes AND ≥⌈K/2⌉ iid retries fail]. This
   forces the model to learn the *residual* skill iid doesn't already provide.
2. **Full-FT instead of LoRA** (or higher rank + longer training) on Qwen-Coder-7B. LoRA rank-16 shifted
   iid recovery on HumanEval (confound); full-FT may either help or make it worse and clarify the story.
3. **Harder-failure curriculum.** MBPP-train failures are limited (~500). TACO/APPS medium-hard failures
   are where the strong-model exception (14B-coder) lives; training there is the real test.
4. **Qwen-Instruct GRPO fix.** Both attempts hit a TRL `zip(strict)` internal error — not batch-size.
   A trl-version pin or minimal-repro debug would recover the family.
