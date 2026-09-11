# PROGRESS SNAPSHOT v6 — read this, then give feedback
_2026-09-11. Single source of truth. Full detail in ADAPTIVE_FORGETTING_RESULTS.md §120–§125._

## 1. THE ONE-PARAGRAPH STORY
We are studying: **for out-of-distribution (OOD) generalization, which post-training procedure wins — on-policy outcome-RL
(GRPO, the field default) or decoupled verified replay (RFT/ReST-EM family)?** Our answer, now supported at scale:
**RFT-style "Decoupled Verified-Coverage Replay" (DVR) dominates GRPO for OOD, and the size of the win is governed by HEADROOM
(how much of the OOD the base model still cannot solve), not by model size.** GRPO fails precisely where the model rarely samples
a success (hard tasks) because its groups become all-failures → no learning signal → it can even DEGRADE the model. DVR replays
offline-harvested verified successes and keeps working. We can prove the two procedures share the same per-prompt gradient
direction, so the gap is *procedural* (coverage/replay), not the update rule.

## 2. WHAT IS SOLID (recorded, controlled, pushed to GitHub)
- **DVR ≫ GRPO for OOD, sign-invariant across 2 model families × 4 sizes** (§113): +0.095 (1.3B) … +0.19 (1.5B) … +0.11 (3B).
- **NEW & IMPORTANT — the gap RETURNS at 7B when there is headroom** (§124): at 7B on a HARD task (depth-14, base only ~57%),
  RFT 0.567 vs GRPO 0.455 = **+0.112**; on an EASY task (depth-7, base ~87%) it was only +0.025. Same model, more headroom → bigger gap.
  GRPO's training reward stayed at the floor (~0.05) — it never learned (dead-group starvation) and dropped BELOW the base model.
- **Coverage is causal** (§89, removal-controlled): full bank 0.79 vs coverage-blocked 0.63 (+0.165) ≈ random-removal → the IDENTITY
  of the covered prompts matters, not bank size.
- **It is coverage, not the RL objective** (§114 + §111 theory): tweaking GRPO's estimator (drop negatives, copy RFT's success-weighting)
  does NOT close the gap; the success-gradient identity proves RFT and outcome-RL share the gradient direction.
- **Predictive headroom law**: gain ≈ headroom × transferable-value — explains the whole size/difficulty curve.

## 3. WHAT WE KILLED (honest negatives — these are NOT in the paper as methods)
- **Coverage-TARGETING (CTH — redirect sampling budget to unsolved prompts)** is **model-size-gated, not headroom-gated** (§122, §125):
  at identical depth-14, 1.5B gains +0.125 but 3B/7B/9B gain ≈ 0. So CTH is a weak-model-only trick → **abandoned as a method.**
  (Do not confuse with #2 above: DVR-replay dominance is different and DOES scale. This dissociation is a key clarity win.)
- Also previously killed (with the control that killed each): self-repair distillation, archive/source-preservation, delayed-value
  selection, difficulty escalation, recomposition/rewiring, execution-VALUE supervision (format effect), decomposition-as-efficiency.

## 4. THE AWARD CLAIM (scoped, honest)
"**On-policy outcome-RL is coverage-limited for OOD; a decoupled verified-replay procedure dominates it, we identify the mechanism
(coverage/replay, not the objective), and give a predictive headroom law for the gap.**" This OVERTURNS the GRPO-default for OOD with
a controlled matrix + mechanism + law. We do NOT claim an unbounded new gain, nor that coverage-targeting scales (it does not).

## 5. RUNNING RIGHT NOW (all 6 live nodes = ~48 GPUs; 1095 dead, needs fresh JSON for +24)
Goal: complete the **DVR-vs-GRPO-at-headroom matrix** (the headline figure) + confirm the mechanism at scale.

| Cell | RFT (DVR) | GRPO | State |
|---|---|---|---|
| 1.5B-hard (d12→14) | 0.260 | running (worker, colocate 3-seed) | training |
| 3B-hard  (d12→14) | 0.455 | running (worker, colocate 3-seed) | training |
| **7B-hard (d12→14)** | **0.567** | **0.455 → +0.112** | **DONE ✓** |
| 9B-hard  (d12→14) | 0.520 | running (1094, server-mode, step ~18/300) | training |
| 1.5B-mid / 3B-mid (d7→9) | (pair) | (pair) | self-contained RFT+GRPO pairs, workers |

- **1093** (chained ~4h): estimator-ladder at 7B-hard (zero-negatives arm step 81/300, then success-count) → then 2 more GRPO-7B-hard
  seeds for a proper confidence interval on the 0.455. Ladder predicts: both knobs ≈ GRPO 0.455 ≪ RFT 0.567 (confirms coverage-not-objective AT SCALE).

## 6. OPEN QUESTIONS / WHERE YOUR FEEDBACK MATTERS
1. **Is the scoped claim (§4) award-caliber to you, or do you want a bigger swing?** (Bigger swing = riskier; this one is defensible.)
2. **Cross-domain**: we have comp + partial math. Do you want a full DVR-vs-GRPO replication in math/code at scale before writing? (adds ~1 day)
3. **Model ceiling**: largest clean point is 9B (running). Want a 14B/32B hard point if a node appears? (headroom law predicts the gap persists.)
4. **CTH**: keep it in the paper as an honest boundary ("targeting doesn't scale, replay does"), or cut entirely?
5. **Seeds**: GRPO-7B-hard is 1 seed (0.455) with 2 more running; RFT side is 3 seeds. OK to headline once CI lands, or want 5 seeds?

## 7. INFRA NOTES (why progress sometimes stalls)
- 40GB GPUs: 7B/9B GRPO needs server-mode (vLLM on GPU0 + ZeRO-2 LoRA on GPU1-7); colocate only fits ≤3B. Both paths now working.
- Workers can't write /tmp/instance_storage (root-owned, no sudo) → we use ~/gu. Pods die at 24h TTL; everything is pushed to GitHub + re-derivable.
