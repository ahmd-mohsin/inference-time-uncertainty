# Diagnostic Diversity (Counterfactual Strategy Disagreement) — Results (2026-09-02)

**Hypothesis (H-new):** even though strategies don't *solve* different problems (v_coverage≈0), they may
*fail differently* — so counterfactual strategy disagreement `D_CF` (how much the answer changes when
you force a different reasoning mode) could be an **error/uncertainty signal that beats iid
self-consistency**, and should beat it *more after GRPO* (routing collapse makes iid samples correlated
→ overconfident: "routing-induced uncertainty masking").

**Setup:** `cf_disagree.py`, 16 iid + 3×14 forced-strategy samples/problem, extracted answers saved.
Default A0 = iid majority answer; error E=1[A0≠gold]; error-AUROC of iid self-consistency (1−agreement)
vs D_CF (fraction of strategies whose answer disagrees with A0) vs combined. 9 jobs: {base,grpo,floor} ×
{Qwen-Math/MATH-500, Qwen-Math/Olympiad, Llama/Olympiad}, n=200 problems.

## Results — error-AUROC (higher = better error detection)

| family/data | policy | err | iid | D_CF | D_CF−iid | iid+D_CF |
|---|---|---:|---:|---:|---:|---:|
| Qwen/math500 | base | 0.22 | 0.794 | 0.737 | −0.057 | 0.791 |
| | grpo | 0.23 | 0.781 | 0.754 | −0.027 | 0.797 |
| | floor | 0.24 | 0.834 | 0.802 | −0.032 | 0.850 |
| Qwen/olympiad | base | 0.58 | 0.840 | 0.770 | −0.070 | 0.852 |
| | grpo | 0.53 | 0.856 | 0.834 | −0.023 | 0.865 |
| | floor | 0.55 | 0.887 | 0.737 | −0.151 | 0.883 |
| Llama/olympiad | base | 0.78 | 0.779 | 0.699 | −0.080 | 0.780 |
| | grpo | 0.65 | 0.789 | 0.626 | −0.164 | 0.754 |
| | floor | 0.74 | 0.896 | 0.730 | −0.165 | 0.872 |

**Routing-masking law (does D_CF−iid grow base→grpo?):**
- Qwen/math500: base −0.057 → grpo −0.027 (**+0.030**, right direction)
- Qwen/olympiad: base −0.070 → grpo −0.023 (**+0.048**, right direction)
- Llama/olympiad: base −0.080 → grpo −0.164 (**−0.083**, wrong direction)

## Verdict — NULL (4th consecutive downstream null)

1. **D_CF never beats iid self-consistency.** In all 9 cells `D_CF−iid < 0`; iid AUROC (0.78–0.90) is the
   stronger error signal everywhere. `iid+D_CF` ≈ iid (adds ≤ +0.014). So forced-strategy answer
   disagreement is **not** a useful uncertainty signal on top of ordinary self-consistency.
2. **The masking law is weak and inconsistent.** On Qwen the D_CF disadvantage *shrinks* after GRPO
   (+0.03, +0.05 — directionally consistent with routing-induced masking) but **never flips positive**;
   on Llama it moves the *wrong* way. Not a robust law.
3. **Likely cause:** forcing a strategy degrades per-sample quality, so forced-strategy answers are
   noisier than free samples → their disagreement is *less* calibrated than natural iid disagreement.
   Diagnostic diversity exists in principle but is dominated by the noise the forcing introduces.

**Consistent pattern across the whole project:** the hidden repertoire — measured as coverage
(stratified pass@k), adaptation, code pass@k, and now diagnostic uncertainty (D_CF) — yields **no
downstream win**. Four independent downstream directions, all null. The robust, real result remains the
**mechanism**: RLVR does routing compression (ρ↓) with competence preserved/improved (c↑), and diversity
(marginal or diagnostic) carries no downstream value because modes are both functionally redundant
(v_coverage≈0) and diagnostically redundant (D_CF adds nothing to iid).

**Implication for the paper:** this is a **theory + empirical-correction** paper. The correct claim is
negative-but-important: *"marginal AND diagnostic reasoning diversity are decoupled from capability in
RLVR-trained math/code models; RL compresses routing while preserving competence, and neither coverage
nor uncertainty benefits from the suppressed modes."* No method win in any regime tested. The remaining
supporting evidence is the gradient-alignment mechanism (why ρ↓/c↑ is benign) — pending.

## Files
Raw: `rl_training/runs_pulled/probe_routing/cf_*.json`. Harness: `cf_disagree.py` + `go_cf.sh`.
Prior nulls: `ROUTING_VS_COMPETENCE_RESULTS.md` (coverage), `ADAPTATION_OPTIONALITY_RESULTS.md`,
`STATUS_AND_ALL_RESULTS.md` (code). Findings/novelty: `FINDINGS_AND_NOVELTY.md`.
