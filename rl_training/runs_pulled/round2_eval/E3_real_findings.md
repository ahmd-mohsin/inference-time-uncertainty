# E3 (real) — Mode-mass certificate on the actual r2 models

Ran the full E3 loop on a **live GPU** (cluster mi-048d12e6e61c0049f, node 10.3.189.179): clustered the
existing base-correct bank into modes, teacher-forced **both round-2 checkpoints** over it (cheap
forward pass, no generation), and certified. Scored files + certify JSONs in `e3/`.

## Pipeline executed
1. `strategy_bank.py cluster --emit-bank` → `bank_clustered.jsonl` (1055 base traces, 305 problems,
   `mode_id` per trace; `ref_logprob` = log π₀ already present).
2. `score_bank_logprobs.py` (teacher-forced) on **r2-from-grpo ckpt-100** and **r2-from-floor ckpt-100**
   → per-trace `logp_theta` (+ `logp_ref`).
3. `recoverability_certificate.py certify` → `Δ_qm = logπ_θ − logπ₀`, LCB, certified fraction.

## Result — the mechanism, measured directly (huge, clean effect)

`Δ = log π_θ(y|q) − log π₀(y|q)` on base-correct **mode witnesses** (1055 traces):

| | mean Δ | median Δ | % ≥ α-floor (log 0.5) | % mass-preserved (Δ≥0) | % collapsed (Δ<−10) |
|---|---:|---:|---:|---:|---:|
| **r2-from-floor** | **+4.40** | +2.30 | **84.1%** | 78.6% | 0.4% |
| **r2-from-grpo**  | **−9.99** | −6.74 | 25.1% | 19.5% | **40.5%** |

- **Paired (same traces): floor − grpo = +14.4 nats; floor higher on 90.9% of traces.**
- **Certified modes (per-mode LCB ≥ log α, α=0.5): floor 402/675 (59.6%) vs grpo 91/675 (13.5%)** —
  the floor fork certifies **~4.4× more** reasoning-mode mass.

**Reading.** Identical unconstrained round-2 RL, run on the two round-1 forks: continued RL from the
plain fork drives base-correct mode mass **down ~10 nats** (40% of modes collapse by >10 nats — i.e.
pushed toward the extinction regime E1 measured), while continued RL from the coverage-preserved fork
**keeps mode mass at or above base** (median +2.3 nats, 84% still satisfy the α=0.5 support floor).
This is the paper's central mechanism — now measured at the **reasoning-mode-witness level**, paired,
91% consistent, and stated as a **certified fraction**, not a single-trace log-prob. It is a >14-nat
effect vs the saturated pass@256 gap of 0.007, so it belongs as the mechanism figure.

## Honest caveats
- **Bank provenance (important):** this bank is the round-1 *floor* coverage bank, so the floor lineage
  is not held out from it. What IS a fair test: these are the **round-2, fully-unconstrained** continued
  models — the floor constraint acted only in round 1, yet its preserved mass **survived identical
  round-2 RL** while the plain fork's collapsed. The unbiased confirmation is to rebuild the bank from a
  **fresh base sample not used in floor training** (E2 `sample_base_solutions.py`) and re-score; expect
  the same direction, smaller magnitude.
- **Thin bank / LCB:** ~3.5 traces/problem → ~1.6/mode, so the per-mode LCB is conservative (that's why
  certified 59.6% < trace-level 84%). The 128–1024-witness bank (E2-scale) will tighten the certificate
  and let us raise `min_witness` from the pilot value of 1.
- **Answer/strategy clustering** is still the heuristic proxy; LLM-judge labeling (E2 final) will firm
  up mode identity.

## Artifacts
`e3/scored_{floor,grpo}.jsonl` (per-trace logprobs), `e3/certify_{floor,grpo}.json` (per-mode Δ/LCB/
certified). Reproduce: cluster → `score_bank_logprobs.py` → `certify`.

## Next
- E2 `sample_base_solutions.py` on a **held-out** base sample (Olympiad hard + Omni-MATH) → unbiased
  bank → re-run E3 (removes the provenance caveat, gives the headline magnitude).
- E4 primal-dual can now target the **endangered modes** this certificate surfaces.
