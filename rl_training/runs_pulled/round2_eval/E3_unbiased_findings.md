# E3-unbiased — mode-mass certificate on a HELD-OUT base bank

Removes the provenance caveat from E3-real. Fresh base bank sampled on this cluster
(`sample_base_solutions.py`, N=256, Olympiad hard band) → **11,664 base-correct witnesses the floor
model NEVER trained on** → strategy-clustered → teacher-forced base / r2-grpo / r2-floor (parallel,
3 GPUs) → Δ = log π_arm − log π_base per witness. Bank + scored files pulled to laptop
(`e2_heldout_bank/`, `e3_heldout/`).

## Result (held out — the fair test)

| Δ = log π_arm − log π_base (11,664 witnesses) | mean | median | % ≥ α-floor (log 0.5) | % preserved (Δ≥0) | % collapsed (Δ<−10) |
|---|---:|---:|---:|---:|---:|
| **r2-from-floor** | **−1.68** | −1.00 | **46.2%** | 41.9% | **2.3%** |
| **r2-from-grpo**  | **−10.11** | −6.25 | 22.6% | 20.8% | **39.0%** |

- **Paired: floor − grpo = +8.44 nats; floor preserves more mode-mass on 77.1% of witnesses.**
- **Catastrophic-collapse rate (Δ<−10, i.e. pushed toward the extinction regime): grpo 39.0% vs
  floor 2.3% — a ~17× difference**, on traces floor never saw. This is the cleanest statement of the
  mechanism: identical continued RL from the plain fork collapses ~2 of every 5 base reasoning modes;
  from the coverage-preserved fork, ~1 in 40.
- Per-mode certified fraction (≥8 witnesses, LCB ≥ log α, α=0.5): floor 9/399 vs grpo 0/399. Formal
  certification is conservative at α=0.5 on held-out traces (floor's mean Δ is ~−1, so most modes'
  LCB dips just below the floor); the collapse-rate and paired gap are the robust signal.

## Comparison to E3-real (biased bank) — as predicted
| | mean Δ floor | mean Δ grpo | floor−grpo | floor collapse | grpo collapse |
|---|---:|---:|---:|---:|---:|
| E3-real (floor's own round-1 bank) | +4.40 | −9.99 | +14.4 | 0.4% | 40.5% |
| **E3-unbiased (held-out)** | **−1.68** | **−10.11** | **+8.44** | **2.3%** | **39.0%** |

Direction identical; magnitude smaller on held-out traces (floor drifts ~1.7 nats down on unseen
modes vs +4.4 on trained ones) — exactly the predicted "same direction, smaller magnitude." Crucially,
**grpo's collapse behavior is bank-independent (~39–40%)** — plain continued RL destroys base
mode-mass regardless of which base traces you measure — while floor keeps collapse near-zero.

## Reading for the paper
The claim is not "floor certifies every mode" — it is "coverage-preserving RL prevents the
catastrophic collapse of base reasoning modes that plain RL causes." Held-out evidence: **grpo
collapses 39% of base modes; floor 2.3%.** This is the mechanism behind the round-2 large-k ceiling
gain (r2-floor ≥ r2-grpo at k≥32) and the E1 extinction counts (grpo drove 2 problems to zero
recoverable mass; floor 0), now measured at the reasoning-mode-witness level on data floor never saw.

## Artifacts (laptop, safe)
`e2_heldout_bank/bank_e2_olympiad_shard{0..7}.jsonl` (11,664 witnesses), `e3_heldout/scored_hb_{base,
grpo,floor}.jsonl`, `e3_heldout/bank_clustered.jsonl`. Reproduce: cluster → score_bank_logprobs ×3 →
join by line order (Δ_arm = logp_arm − logp_base).

## Caveats remaining
- Strategy clustering is still the heuristic proxy (LLM-judge labeling is the E2-final upgrade).
- α=0.5 is a strict floor; the certified fraction would rise with a tuned α or the E4 primal-dual
  (which targets only endangered modes). Collapse-rate is the α-independent headline.
- Single seed; E7 adds 3 training seeds.
