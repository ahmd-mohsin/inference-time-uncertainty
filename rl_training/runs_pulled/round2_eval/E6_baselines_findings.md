# E6 — Baselines: floor vs global-KL vs plain GRPO (mode-mass)

First baseline landed: **GRPO + global KL-to-base** (β=0.04), the canonical coverage-preservation
comparator. Trained 150 steps from base under the **identical setup** as the E8 plain/floor arms
(same data, steps, group size), then teacher-forced over the base-correct bank (1055 witnesses);
Δ = logπ − logπ_base.

## 3-way result
| arm (150 steps from base) | mean Δ | median Δ | % ≥ α-floor (log0.5) | % preserved (Δ≥0) | % collapsed (Δ<−10) |
|---|---:|---:|---:|---:|---:|
| plain GRPO | −1.96 | −0.96 | 46.2% | 34.7% | 4.1% |
| **GRPO + global KL** (β=0.04) | −0.73 | −0.30 | 57.1% | 44.0% | 0.5% |
| **GRPO + floor (ours)** | **+1.38** | **+0.52** | **85.0%** | **67.6%** | **0.0%** |

Paired vs floor: floor − plain = +3.34 nats (floor>plain 69%); **floor − global-KL = +2.11 nats
(floor>KL 60%)**.

## Reading
- **Global KL works — partially.** As a coverage method it cuts catastrophic collapse (4.1%→0.5%)
  and lifts mean Δ (−1.96→−0.73), confirming it's a real baseline (not a strawman).
- **The floor beats it decisively.** Only the floor has **positive** mean Δ (it *raises* base mode
  mass, not merely slows its decay), 85% vs 57% of modes above the α=0.5 floor, and **0% collapse**.
  +2.11 nats over global-KL, floor>KL on 60% of witnesses.
- **Why:** global KL applies *symmetric, uniform* pressure toward base everywhere (taxing pass@1 and
  still letting rare modes drift); the floor is *one-sided* (only resists dropping below the floor)
  and *off-policy/teacher-forced* on the exact fragile modes, so its signal stays strong precisely
  where those modes are — the qualitative distinction the method claims.

This is the reviewer-critical result: **we beat the standard KL baseline on mode preservation, not
just vanilla GRPO.**

## Status of the baseline suite
- ✅ plain GRPO (β=0)
- ✅ **global KL** (β=0.04) — this doc
- ✅ floor (ours, expSR)
- ⏳ PBA (per-problem base anchoring) — implement as train_grpo variant (prompt-risk-gated KL)
- ⏳ UCPO (uniformity among correct rollouts) — implement
- ⏳ DPH-RL (mass-covering base-replay) — implement
- ⏳ BBG (Bayesian boundary gating) — implement
(All to be run at the same 150-step / fragile-band setup and added to this table.)

## Artifacts
`e8_long/scored_e6globalkl.jsonl` (bank line-aligned; ref_logprob=base). Model `e6_globalkl` (150) on
node nvme (ephemeral).
