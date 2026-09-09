# Verified-Experience Transfer — Current Findings & Search Plan
*Status digest for steering. Full detail: `ADAPTIVE_FORGETTING_RESULTS.md` (§37–§63). Updated 2026-09-09.*

## One-line state
Two exciting hooks died under scrutiny; a modest, well-controlled result survives; we are now **searching for a regime where an authorship×RL effect is large and robust** before making any novel-mechanism claim.

## What we set out to show
A policy's *solving* competence ≠ the *teaching* value of its verified traces — and that this can be turned into a method that makes RL post-training transfer better.

## What SURVIVES (honest, replicated)
- **SFT-authored verified traces are the best for imitation transfer**, consistently but by a **small margin (+0.02–0.04)**, across **3 model families (Qwen-3B/7B, Phi-3.5-mini) × 2 domains (MATH-500, SVAMP)**.
- **CONFOUND (unresolved):** the SFT producer is also the **strongest solver** (GSM8K 0.700 vs GRPO 0.514). So "better teacher" is not yet separated from "better solver."

## What FAILED (retracted honestly — these are the important negatives)
| Hook | Result | Why it died |
|------|--------|-------------|
| MaxRL "weak solver teaches better" (§56) | RETRACTED | edge vanished on matched prompts; didn't reproduce at 7B (coverage-fragile) |
| SAC-RL "consolidate RL policy on SFT traces" as a method (§58) | DEMOTED to baseline | Gate A: fresh-SFT on SFT-authored (0.397) ≥ SAC-RL (0.386) at **lower cost**; RL-init redundant on OOD *and* in-domain |
| "best-for-imitation ≠ best-for-RL" reversal (§63) | RETRACTED | n=1 crossover was noise; at 4 seeds GRPO-auth 0.381 < SFT-auth 0.398 (t=−1.25) |
| "bigger margins on harder OOD" (AMC) | NULL | margin difficulty-invariant (+0.021), not bigger |
| MaxRL-approx label | CORRECTED | our arm is `R−p̂` (centered, no std-norm), **not** the published MaxRL estimator |

## Key numbers (OOD MATH-500, Qwen2.5-3B, LoRA)
| recipe | OOD | notes |
|--------|-----|-------|
| Fresh base + SFT-authored traces | **0.397** | best; simple SFT/distillation, no RL |
| SAC-RL (GRPO policy + SFT-authored) | 0.386 | baseline; RL-init adds nothing |
| SFT-only / fresh + GRPO-authored | 0.365–0.370 | |
| rejection-FT (3 seeds) | 0.357 | |
| raw GRPO | 0.295 | |
Gate B post-RL (4 seeds): SFT-authored 0.398 [.382,.414] vs GRPO-authored 0.381 [.361,.402] — **no reversal**.

## Honest tier assessment
Rigorous but **modest / conference-tier, not award-tier**. Effects are small, the large one (vs raw GRPO) is against a weak baseline, both novel hooks failed replication, and there is **no clean causal mechanism**.

## Literature we must beat (crowded)
RLT (NeurIPS 2025, closest — student-reward teachers + init for RL), SOAR (student-improvement reward), SEAL/Self-Adapting-LMs (RL over generated training data), PEAR (best-for-SFT ≠ best-for-RL), Distilled-RL. "Choose a better teacher" is **not** enough.

## OPTION 2 — the search now running (find a LARGE, robust authorship×RL effect)
Hypothesis to earn: *verified traces differ in how well they prepare a learner for **subsequent RL**, and this gap is large in the right regime.* Gate B found nothing, but was weak on three axes — the search targets each:

1. **RL budget too short** (100 steps, GSM8K saturated). → **Probe 1 (LAUNCHED):** 300-step GRPO continuation on the reversal pair (SFT-auth vs GRPO-auth), multi-seed, eval OOD. Does a longer RL budget differentiate initializations?
2. **No headroom** (recipients already ~0.70 on GSM8K). → **Probe 2 (candidate):** continue RL on a task with real headroom (MATH-train / harder problems) where RL can actually move, eval OOD.
3. **Regime/scale** — a task where RL ≫ SFT (exploration matters: search/code w/ execution reward) so initialization plausibly gates RL learnability; and/or bigger scale (7B, full-FT).

### Decision rule (pre-registered)
Build the RL-aware author (Gate C) **only if** an authorship×RL effect appears that is (a) **large** (≫ the ±0.02 imitation deltas), (b) **replicated** (≥4 seeds, non-overlapping CIs), (c) **RL-specific** (beats a matched continued-SFT control). Otherwise, write the modest rigorous paper.

## WHERE I NEED YOUR STEER ("next tones")
- **Which regime for the large effect?** (a) longer RL only, (b) headroom task (MATH-train / hard math), (c) exploration-heavy task where RL≫SFT (code w/ exec reward, countdown/search), (d) bigger scale (7B/full-FT).
- **Acceptable fallback?** If no large effect appears, are you OK publishing the modest, honest "source-quality of verified traces for transfer + careful nulls" paper — or keep searching?
- **Gated models:** Llama/Gemma are HF-gated (401). Accept licenses on the token account if you want those families.

## Assets ready (no recompute needed)
- Producer banks (base/SFT/GRPO/MaxRL-authored, Qwen-3B) on all 3 clusters; 7B-authored banks (SFT/GRPO) harvested as shards on C1.
- Fresh recipients per bank; GRPO policies (e1_C_grpo_s0/1/2, grpo_q7b_s0); SFT producers (3B, 7B).
- 3 clusters live (C1/C2/C3, 24 GPUs on mains; workers available).
