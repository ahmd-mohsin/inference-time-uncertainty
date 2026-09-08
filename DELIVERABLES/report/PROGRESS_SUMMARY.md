# Progress Summary — for review & next-step feedback
_2026-09-08 · 72-GPU run (3 fresh clusters) · all results in ADAPTIVE_FORGETTING_RESULTS.md, pushed to GitHub main_

## Thesis (settled this run)
**The update rule governs OOD transfer of verified experience — because RL can only *reweight* probability
mass, not *place* it.** SFT (an M-projection / mass-placing operator) puts probability mass on OOD-correct
traces; GRPO's advantage-weighted gradient can only sharpen mass that already exists (ρ≈0 on unreached
regions). This one principle explains every result below.

## Headline results (matched verified experience, Qwen2.5-3B unless noted)

| # | Experiment | Result | Verdict |
|---|-----------|--------|---------|
| §37 | Code (HumanEval OOD, MBPP-train) | base 0.566 → SFT 0.662 = **+0.096** | SFT≫GRPO, 4th domain |
| §40 | Dense OOD @7B | OlympiadBench C 0.135 vs A 0.100; full-MATH C 0.382 vs A 0.323; **MMLU-Pro C 0.408 vs A 0.304 (+0.104 cross-domain)** | SFT≫GRPO everywhere |
| §42 | Tuned-GRPO fairness | KL=0 0.304 / 2×-steps 0.328 / group16 0.328 (MATH-500) | gap survives tuning (signal-structural) |
| §44 | **Mass-placing axis** (GRPO+self-distill, β_sd 0.05–20, 3 seeds, 2 benchmarks) | flat **0.335±0.01**, never reaches SFT 0.410 | **GRPO+rehearsal CAN'T reconstruct SFT (irreducible)** |
| §44b | OlympiadBench hard-tier | base 0.074 / rescue 0.081 / **SFT 0.104** | SFT wins hardest tier |
| §45 | Reverse: SFT→GRPO (3-seed decay) | 0.410 → **0.433** (monotonic rise, no erosion) | GRPO safely *sharpens* placed mass |
| §46 | H5 subspace localization | attn-only 0.360, MLP-only 0.354 (both ≈ all 0.367) | mass-placing **redundantly distributed**, not localized |
| §48 | H6 SFT-ignition (tiny seed) | N=1–100 seed → GRPO stays ~0.33 | **no cheap ignition** — need substantial SFT |
| §49 | H9 in-dist crossover | SFT>GRPO in-dist (**0.705 vs 0.51–0.53** tuned) AND OOD (0.411 vs 0.326) | **no crossover — advantage is GENERAL, not OOD-specific** |

## Mechanism (coherent across all of the above)
- GRPO **from base** → ρ≈0, transfers ~0; even a bolted-on NLL rehearsal term can't fix it (§44, 20× dose, 3 seeds).
- SFT **places** the mass → transfers (in-dist +0.22, OOD +0.09–0.12).
- GRPO **from an SFT'd model** → sharpens the placed mass (§45, 0.410→0.433). Order/precondition, not mutual destruction.
- The mass-placing is redundant across attn+MLP (§46) and requires a *substantial* seed, not a token one (§48).

## Honesty ledger — 5 refuted pre-registrations (each strengthens the core)
H1 monotonic axis · H2 GRPO-rescue · §45 erosion · H5 MLP-localization · H6 cheap-ignition · H9 in-dist-crossover.
Every *optimistic/cheap* shortcut failed; the core (SFT places, GRPO reweights) survived every attack and generalized.

## Paper spine (all pushed)
§37 code · §40 dense cross-domain · §41 prior-art positioning · §42 tuned-GRPO fairness · §44 irreducibility ·
§44b hard-tier · §45 order/precondition · §46 subspace · §48 ignition · §49 generality · §THEORY (Thm11 hybrid).

## Fleet / assets
72 GPU (3 fresh clusters, us-west-2). 113 eval JSONs + adapters death-proofed to `checkpoints_pulled/fresh_0908/`.
Old HF continued-RL checkpoints cleared per request. Current work uses LOCAL adapters (no HF dependency).

## Open questions / candidate next steps (awaiting your feedback)
1. **H8 entropy-collapse signature** (queued): is OOD transfer ∝ retained generation entropy; is GRPO's failure
   premature entropy collapse? — a direct *measurement* of the mechanism (not another optimistic guess).
2. **H7 does "verified" matter?**: SFT on unverified/diverse traces — is it the operator or the correctness?
3. **Scale**: 7B/14B version of §44/§45 (7B colocate OOMs on 40GB → needs server-mode; setup cost).
4. **Framing decision**: §49 shows the effect is *general* (in-dist too), not OOD-only. Do we reframe the paper
   around "verified-experience efficiency of the update rule" with OOD as the sharpest case, or keep OOD-centric?
5. **Multi-family**: replicate §44/§45 on Llama/Mistral/DeepSeek (have adapters from earlier §27).
6. **H10 mass-transplant moonshot**: graft SFT LoRA delta A→B (same family) then RL — does placed mass port?
