# E8-interventional — plain-GRPO vs GRPO+floor from base (25 steps) — NULL / underpowered

Ran the gold-standard causal design: from base Qwen2.5-Math-7B, two arms with **identical** data/steps,
the ONLY difference being the off-policy support floor. `go_e8_arm.sh plain 25` and `... floor 25`
(full-FT, ZeRO-3), then teacher-forced base / plain / floor over the 11,664-witness bank; Δ = logπ_trained − logπ_base.

## Result (honest): no phase transition at 25 steps
| Δ from base (11,664 witnesses) | mean | median | % suppressed (Δ<−5) |
|---|---:|---:|---:|
| plain-GRPO (25 steps) | −0.16 | 0.00 | 0.5% |
| GRPO+floor (25 steps) | −0.20 | 0.00 | 0.1% |

- Paired: floor > plain on only **30.5%** of witnesses; mean gap Δfloor−Δplain = **−0.05 nats** (≈0).
- Binned by training-budget K·p (K=8, boundary K·p=1 at p=0.125): the gap `Δfloor−Δplain` is flat and
  near-zero across all bins (−0.13 … −0.00 … −0.08) — **no `K·p≈1` transition.**

## Why it's null (not a refutation) — it's underpowered on training duration
25 GRPO steps barely perturb the base policy (median Δ = 0.00; only ~0.5% of modes moved at all). The
phase transition requires the plain arm to have **actually driven modes below the boundary** — which
takes many steps. The round-1 forks needed **400 steps** to open the coverage gap; the round-2
continued-RL arms used **100 steps**. At 25 steps neither arm has suppressed modes yet, so there is
nothing for the floor to differentiate. This is a **statement about the experiment's horizon**, not
about the mechanism.

## Where the interventional signal actually lives: E3-unbiased (adequate scale)
**E3-unbiased already IS the interventional experiment at proper training duration.** It is exactly
"identical continued RL (100 steps) from a plain fork vs a floor fork," and at that scale the effect
is large and clean (held out): plain collapses **39%** of base modes (Δ<−10), floor **2.3%**;
paired floor > plain on 77%; +8.44 nats mean. So the causal claim is supported — it just needs
~100+ steps to manifest, which the 25-step dedicated run does not reach.

## Options to get a clean dedicated interventional figure
1. **Accept E3-unbiased as the interventional evidence** (it is identical-RL fork-vs-fork at 100 steps,
   held out) — recommended; the dedicated short run adds nothing beyond confirming the horizon effect.
2. **Longer dedicated intervention**: rerun both arms at ~150–200 steps from base (≈1.5–2h/arm on
   8×A100) and re-measure Δ vs K·p — expect the transition to emerge as the plain arm crosses the
   boundary. Higher cost, cleaner single figure.
3. **Artificial-suppression variant**: force-lower a controlled mode's prefix prob, then a few
   plain-GRPO vs floor steps, measure recovery — the most direct but most engineering.

## Artifacts (laptop, safe)
`e3_heldout/scored_e8_{plain,floor}.jsonl`, `scored_hb_base.jsonl`, `E8_interventional.csv`.
Trained arm models `e8_{plain,floor}` were on node nvme (ephemeral).

## Takeaway
The dedicated 25-step interventional run is **negative/underpowered** and I'm recording it as such.
The mechanism's causal evidence stands on E3-unbiased (100-step, held out) + the E8 observational
phase-transition figure (which predicts E1's extinctions). If we want a standalone interventional
figure, it needs the longer-horizon rerun (option 2).
