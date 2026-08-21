# E8-interventional @150 steps — causal mode-preservation confirmed (25-step null resolved)

Ran the gold-standard causal design at proper horizon: from base Qwen2.5-Math-7B, two **identical**
full-FT GRPO arms (150 steps each), the ONLY difference being the off-policy support floor. Run in
parallel on two nodes (plain on main, floor on worker1). Then teacher-forced each final model over the
base-correct bank (1055 traces, `ref_logprob = log π_base`); Δ = logπ_trained − logπ_base.

## Result — floor causally preserves mode-mass that plain RL erodes

| Δ from base (1055 mode-witnesses) | mean | median | % ≥ α-floor (log0.5) | % preserved (Δ≥0) | % collapsed (Δ<−10) |
|---|---:|---:|---:|---:|---:|
| **plain GRPO (150)** | **−1.96** | −0.96 | 46.2% | 34.7% | **4.1%** |
| **GRPO + floor (150)** | **+1.38** | +0.52 | **85.0%** | 67.6% | **0.0%** |

- **Paired: floor − plain = +3.34 nats; floor > plain on 69.0% of witnesses.**
- Identical training, floor is the sole difference ⇒ **causal**: plain RL drives base mode-mass down
  (mean −1.96, 4.1% collapse toward extinction); the floor holds it up (mean +1.38, **0% collapse**).
- **This resolves the 25-step null** (E8_interventional_findings.md): at 25 steps neither arm had moved
  mass (median Δ=0); at 150 steps the divergence is clear and large. Confirms the mechanism needs
  training horizon to manifest — exactly as predicted.

## Honest limitation: no sharp K·p≈1 gradient in THIS run
Binned by training-budget K·p (K=8, boundary at p=0.125), the floor−plain gap is **flat** (~+3.3 nats
across every bin; plain collapse ~3–5% throughout):

| K·p bin | n | Δplain | Δfloor | gap | plain collapse | floor collapse |
|---|---:|---:|---:|---:|---:|---:|
| <0.5 | 180 | −2.04 | +1.50 | 3.55 | 3.3% | 0% |
| 0.5–1 | 205 | −1.77 | +1.42 | 3.20 | 3.9% | 0% |
| 1–2 | 294 | −2.00 | +1.55 | 3.54 | 3.1% | 0% |
| 2–4 | 348 | −2.00 | +1.18 | 3.19 | 5.5% | 0% |
| >4 | 28 | −1.84 | +1.03 | 2.87 | 3.6% | 0% |

The floor helps **uniformly** over the p-range this fragile-band bank spans — it does not show the
*differential* below-vs-above `K·p=1` effect. Two reasons: (i) the bank's p_base is concentrated
(K·p mostly 0.5–4; little mass at the deep K·p≪1 tail); (ii) 150 steps is moderate — not enough to
drive the very-low-p modes to differential extinction. **The sharp `K·p≈1` transition remains the
contribution of the observational E8** (which predicted E1's exact extinctions); the interventional
run's job — proving the floor *causally* preserves mode-mass under identical RL — is done.

## Takeaway
- ✅ Causal claim supported: identical RL, floor prevents the mode-mass erosion (−1.96→+1.38 nats;
  4.1%→0% collapse) that plain GRPO causes. Clean, +3.3-nat, 69%-paired.
- ✅ 25-step null explained (horizon effect), now resolved at 150 steps.
- ⚠️ K·p-differential transition not shown here (flat gap) — needs a wider-p bank and/or longer
  training; the observational E8 carries the K·p≈1 boundary claim.

## Artifacts (laptop, safe)
`e8_long/scored_e8plain150.jsonl`, `e8_long/scored_e8floor150.jsonl` (bank line-aligned, ref_logprob =
base). Trained models `e8_{plain,floor}` (150 steps) on node nvme (ephemeral; not pulled — 15GB each).
save_total_limit=3 reaped early checkpoints, so per-step trajectory unavailable (only 130/140/150).

## Next to strengthen
- Wider-p bank (include Omni-MATH / very-low-p modes) + score at multiple checkpoints (raise
  save_total_limit) → recover the K·p gradient.
- This + observational E8 + E3-unbiased together make the mechanism case; E6 baselines are the
  remaining reviewer demand.
