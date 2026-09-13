# PROGRESS SNAPSHOT v11 — verified-support program (complete state)
_2026-09-13. Single source of truth. Detail: ADAPTIVE_FORGETTING_RESULTS.md §120–§140. Method: rl_training/vsf_trainer.py. Memos: ASTRA_ADVICE.md, REGRESSION_BUDGET_MEMO.md, FEEDBACK_ACQUISITION_MEMO.md. Recipe: rl_training/queue/FULL_BOOTSTRAP.sh._

## 0. STATUS RIGHT NOW
**No experiment running** — all 9 clusters (72 GPUs) idle. A0 gate (feedback-acquisition memo) complete + committed. **All three external memos are now gate-killed** (Astra amplifier, regression-budget, feedback-acquisition) — no acquisition/retention wrapper beats RFT. The method search is closed; the defensible deliverable is the characterization+measurement paper.

## 1. HEADLINE (what the data supports)
**For verifiable OOD learning, decoupled verified replay (RFT) is the ceiling recipe: it acquires the most and regresses the least. On-policy outcome-RL (GRPO) acquires little and actively regresses; VSF (a support-floor) causally repairs both axes but does not beat RFT; and continued RL after RFT adds nothing.** Gains are governed by headroom × accessibility, not headroom alone. This is a characterization + measurement result, not a new-SOTA-method.

## 2. MECHANISM MATRIX (OOD accuracy; pushed)
| Cell (family) | base | GRPO | VSF | RFT |
|---|---|---|---|---|
| 1.5B-mid (k16, §138) | 0.370 | 0.430 | 0.585 | **0.745** |
| 1.5B-hard (k16) | 0.180 | 0.210 | 0.390 | **0.535** |
| 3B-mid (k16) | 0.545 | 0.525↓ | — | **0.845** |
| 3B-hard (k16) | 0.365 | 0.330↓ | — | **0.620** |
| 7B-hard (§124-129) | 0.475 | 0.455↓ | 0.575 | 0.567 |
| 9B-hard (Yi) | — | 0.395 | 0.470 | 0.520 |
| deepseek-6.7B | 0.425 | 0.350↓ | — | 0.485 |
_RFT ≫ VSF > GRPO ≈ base. non-starved GRPO(num_gen16)=plain GRPO (§128). Independent-hardware (cu126 nodes) reproduces the spine (§135)._

## 3. PER-PROBLEM α/β (the measurement contribution, §138, 1.5B-mid k16, n=200)
| Arm | α = acquire\|base=0 | β = regress\|base=1 | oracle O=E[max(base,·)] |
|---|---|---|---|
| GRPO | 0.214 | 0.203 | 0.505 |
| VSF | 0.389 | 0.081 | 0.615 |
| RFT | 0.635 | 0.068 | 0.770 |
_GRPO's flat net (0.43 vs 0.37) **hides** α=0.21 acquisition offset by β=0.20 regression — averaged accuracy conceals both. VSF improves both axes; RFT dominates both._

## 4. FINDINGS (pushed, controlled)
1. **RFT ≫ GRPO for OOD**, sign-invariant across 2 families × 4 sizes (§113); **real, not a starved baseline** (non-starved GRPO doesn't help, §128).
2. **Mechanism = acquisition vs regression** (§127/§138): GRPO acquires little + regresses (below base at 3B/7B/deepseek); RFT/VSF acquire. Shown in 3 families.
3. **It's replay/coverage, not the objective** (§114 estimator ladder + §111 success-gradient identity).
4. **Dissociation** (§134): anchoring (disjoint replay helps) + acquisition (same-pool coverage); prior-support (RFT→GRPO) best; estimator knobs null.
5. **VSF = causal repair, not SOTA** (§129-131/§138): reverses GRPO's regression, full RFT-parity at 7B, partial elsewhere; **RFT ≥ VSF everywhere**.
6. **Gains are accessibility-gated, NOT linear-in-headroom** (§136→§137): clean matrix showed RFT gain *decreases* at extreme headroom (bank can't be filled). Linear "law" was a mixed-source artifact — corrected.
7. **2nd-domain corroboration** (§137): MATH-500 L5 GRPO pass@1 0.073<base 0.083 (regresses in non-code domain).

## 5. HYPOTHESES KILLED BY THEIR OWN GATES (honest, not spin)
- **CTH coverage-targeting** — model-size-gated, weak-model-only (§122/§125).
- **Linear headroom "law"** — accessibility-gated; the clean matrix refuted the linear fit (§137).
- **Regression-budget / retention program (memo H1–H5)** — **KILLED at the gate** (§138/§139): H0 oracle O(GRPO)=0.505≪RFT; H1 closer GRPO-from-RFT = RFT within noise (+.03/−.015/+.005/−.010) → RL adds nothing after RFT → constraining it can't beat RFT. **Not funded H1–H5**, per the memo's own kill-rule.
- **Feedback-acquisition program (memo A0–A4)** — **KILLED at A0** (§140): a 32–82% larger matched-domain verified bank does NOT improve RFT (Δ = +.010/−.020/.000/+.045, mean +.009, within noise; flat even in the highest-headroom cell). The recipient is **saturated** → an acquisition amplifier has no headroom → the query-by-committee active selector (A1–A4) **not built**, per the memo's own kill-rule.
- Earlier: self-repair distillation, archive/source preservation, delayed-value, difficulty escalation, recomposition, execution-value supervision.

## 6. HONEST AWARD READ
The defensible paper is a **characterization + measurement** paper: RFT is the ceiling for verified OOD; the α/β hidden-acquisition decomposition; VSF as a causal repair; accessibility-gated gains; robust across families/domains/hardware; multiple hypotheses killed by pre-registered gates. This is **strong main-track / possibly spotlight**, NOT a clear award — because no *new method* beats RFT. **All three external method proposals now fail their own gates**: VSF (repairs but < RFT), the regression-budget program (§138/§139: RL-from-RFT adds nothing), and the feedback-acquisition amplifier (§140: recipient saturated, bigger bank doesn't help). They converge on one wall — **tuned RFT is the frontier the available verified data supports**. Award-tier would need a method that genuinely beats RFT (none found across all three memos) or a frontier-scale predictive law (accessibility-gated, needs 2-axis data + >9B, currently blocked by 40GB OOM). Reported honestly, without a manufactured winner.

## 7. DATA GAPS / INFRA
- Old nodes died at 24h TTL, lost 14B-GRPO/VSF + deepseek-VSF finals (§133, non-load-bearing). 7B/14B OOM on 40GB (colocate + server); no clean >9B point. 3B VSF arms failed to capture in the clean matrix.
- Env solved + committed (`FULL_BOOTSTRAP.sh`: vllm0.23 + transformers4.57.6 + torch/torchvision cu126 + nvidia-cuda-runtime-cu13 on LD_LIBRARY_PATH + flash-attn cu126 source-build; sft pins CUDA_VISIBLE_DEVICES=0). Launch cells from the MAIN via sshpass (laptop→pod SSM tunnels drop within ~5s).

## 8. NEXT (nodes idle; honest options)
- **Write the characterization+measurement paper** from §120–140 (recommended — the science is done and defensible; the method search is now closed, all three memos gate-killed).
- Non-essential fills: complete 3B VSF / L3 math RFT+VSF arms; a memory-light ≥7B path (TP/QLoRA) for a clean large cell.
- I will **not** re-fund gate-killed programs (H1–H5 regression-budget, A1–A4 feedback-acquisition) or claim a law/method the data doesn't support. A genuinely new amplifier would need to first show the recipient is NOT saturated (A0 says it is, at this data scale) — e.g. a fundamentally larger/different data source, not a re-harvest of the same domain.
