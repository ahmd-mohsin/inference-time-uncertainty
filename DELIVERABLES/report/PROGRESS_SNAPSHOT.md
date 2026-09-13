# PROGRESS SNAPSHOT v9 — verified-support / VSF program (complete state)
_2026-09-13. Single source of truth. Detail: ADAPTIVE_FORGETTING_RESULTS.md §120–§137. Method: rl_training/vsf_trainer.py. Advisor memo: ASTRA_ADVICE.md. Recipe: rl_training/queue/FULL_BOOTSTRAP.sh._

## 0. STATUS RIGHT NOW
**No experiment is running** — all 9 clusters (3 mains + 6 workers, 72 GPUs) are idle (0% GPU). The latest batch (clean matrix + L3 math) completed. Everything below is committed to GitHub.

## 1. HEADLINE
**For verifiable OOD learning the resource that matters is transferable VERIFIED SUPPORT — its acquisition AND preservation — not the number of on-policy updates.** On hard OOD, on-policy outcome-RL (GRPO) fails to acquire and can drop *below the base model*; decoupled verified replay (RFT/DVR) dominates; a persistent support-floor (VSF) partially repairs GRPO. The magnitude is governed by **headroom × accessibility**, not headroom alone.

## 2. MECHANISM MATRIX (accuracy on OOD; pushed)
| Cell (family) | base | GRPO | VSF | RFT | source |
|---|---|---|---|---|---|
| 1.5B-mid (clean) | 0.265 | 0.240 | 0.335 | 0.550 | §137 |
| 1.5B-hard (clean) | 0.095 | 0.090 | 0.220 | 0.300 | §137 |
| 3B-mid (clean) | 0.340 | 0.315 | — | 0.690 | §137 |
| 3B-hard (clean) | 0.225 | 0.215 | — | 0.460 | §137 |
| 3B-hard (orig) | 0.255 | 0.210 ↓ | 0.280 | 0.505 | §126-130 |
| **7B-hard** | 0.475 | 0.455 ↓ | **0.575** | 0.567 | §124-129 |
| 9B-hard (Yi) | — | 0.395 | 0.470 | 0.520 | §129-131 |
| deepseek-6.7B | 0.425 | 0.350 ↓ | — | 0.485 | §126 |
_Read: RFT ≫ VSF > GRPO ≈ base everywhere. GRPO ends at/below base (no acquisition; regresses at 3B/7B/deepseek). VSF partial-repair at 1.5B, FULL RFT-parity at 7B. non-starved GRPO (num_gen16) = plain GRPO (§128)._

## 3. FINDINGS (all pushed, controlled)
1. **DVR ≫ GRPO for OOD**, sign-invariant across families/sizes (§113). **Deficit is real** — survives non-starved GRPO (§128).
2. **Mechanism = acquisition vs regression (α/β, §127)**: GRPO ends below base (damages held-out capability); RFT/VSF end above (acquire). Shown in **3 families** (Qwen, Yi, deepseek).
3. **It's coverage/replay, not the objective** (§114 estimator-knob ladder + §111 success-gradient identity).
4. **Dissociation (§134)**: replay benefit = **anchoring** (even disjoint-prompt replay helps: 0.165>GRPO 0.095) + **acquisition** (same-pool coverage). **Prior-support (RFT→GRPO) is best (0.335)** — seed RL with verified support first. Estimator knobs (zero-neg/KL) null.
5. **VSF repairs GRPO** (§129-131) — reverses the below-base regression at every size, full RFT-parity at 7B, partial at 1.5B/3B. Honest scope: **RFT ≥ VSF everywhere** → VSF is the causal *repair / drop-in fix*, not a strict SOTA winner.
6. **Predictive law is accessibility-gated, NOT linear (§136→§137 correction)**: the clean same-protocol matrix showed RFT gain *decreases* at extreme headroom (hard depth → base can't sample successes → smaller bank → less acquisition). gain ≈ headroom × accessibility (inverted-U). §136's linear R²=0.87 was a mixed-source artifact — caught + corrected by the clean matrix.
7. **2nd domain (§137 L3)**: on MATH-500 level-5, GRPO pass@1 0.073 < base 0.083 (regresses in a **non-code** domain), pass@4 up (redistribution). Corroborates §40. → regression is not a comp-generator artifact.

## 4. KILLED (honest negatives — not methods)
Coverage-TARGETING (CTH) is model-size-gated → weak-model-only, abandoned (§122/§125). Also killed w/ controls: self-repair distillation, archive/source preservation, delayed-value, difficulty escalation, recomposition, execution-value supervision, decomposition-as-efficiency, the linear headroom "law" (§136).

## 5. HONEST AWARD READ
The paper is **strong / spotlight-plausible**, not a guaranteed award. Defensible spine: outcome-RL regresses on hard OOD; verified support is the resource; VSF is the causal repair; holds across 3 families + 2 domains + independent hardware; law is accessibility-gated. **What's still needed for award-tier**: (a) a clean predictive-law fit that survives (the accessibility-gated form needs 2-axis data), (b) a literature audit vs self-imitation/ReST (novelty risk), (c) a frontier-scale point. Not yet claimed as a "law".

## 6. DATA LOST / GAPS (honest)
- Old nodes (1093/1094) died at 24h TTL with 14B-GRPO/VSF + deepseek-VSF finals in-flight, unpushed (§133). Non-load-bearing.
- 7B/14B keep OOMing on 40 GB (colocate + server) — 7B/9B covered by old-node runs; no clean >9B point.
- Clean-matrix 3B VSF arms failed to capture; L3 math RFT/VSF arms not run; 7B math GRPO OOM'd.

## 7. INFRA (validated, committed to rl_training/queue/)
Fresh pytorch-base-24.12 → working vllm0.23 stack: `vllm0.23 + transformers4.57.6 + torch/torchvision cu126 + nvidia-cuda-runtime-cu13 (LD_LIBRARY_PATH, for vllm's cu13 engine) + flash-attn cu126 source-build`; sft_train must pin CUDA_VISIBLE_DEVICES=0. Launch cells from the MAIN via sshpass (main→worker internal link stable; laptop→pod SSM tunnels drop within ~5s so multi-second launches must run on-node). Everything pushed to GitHub; nodes die at 24h TTL.

## 8. NEXT (nodes idle; say the word)
- Complete **L3 math row** (RFT+VSF math arms) for a full 2nd-domain comparison.
- **2-axis matrix** (headroom × bank-accessibility) to fit the corrected accessibility-gated law legitimately.
- Re-run **3B VSF** arms (failed capture); memory-light **7B** path (TP or QLoRA) for a clean ≥7B cell.
- Literature audit (self-imitation / ReST / balanced-replay) before any method-novelty claim.
