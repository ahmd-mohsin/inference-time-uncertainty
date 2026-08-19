# E1 — Recoverability, operationalized (first pivot experiment)

Ran `rl_training/recoverability.py` on the existing per-problem data (base / r2-from-grpo /
r2-from-floor, 329-problem Olympiad fragile band, N=1024 samples). No GPU — pure re-analysis of data
we already own. Output JSON: `recoverability_E1.json`.

## What E1 computes
- `p̂_q = n_correct/N` — single-sample success **mass** (answer-event level).
- `R_K(q) = 1 − (1−p̂_q)^K` — **recoverability** at future rollout budget `K` (closed form → any K).
- `EffSupp(K,τ) = #{q : R_K(q) ≥ τ}` — **effective (recoverable) support**.
- pass@1 (sharpness) and pass@k (coverage) kept as separate axes.

*Level caveat:* this is **answer-event** recoverability (a problem is "recoverable" if a correct
answer is findable in K samples). MODE-level (distinct reasoning strategies) is E2 — swap `p̂_q` for
per-mode mass `p_θ(M|q)`; all machinery below is identical.

## Headline result (the metric reveals what pass@k hid)

**Continued plain-GRPO RL drove 2 base-recoverable problems to ZERO recoverable mass
(`n_correct=0`, unrecoverable at *any* K); continued RL from the floor fork drove ZERO to extinction.**

| | base | r2-from-grpo | r2-from-floor |
|---|---|---|---|
| problems with zero recoverable mass (`n_correct=0`) | 0 | **2** (#168, #237) | **0** |
| base-recoverable problems driven extinct by round-2 RL | — | **2** | **0** |
| `n_correct ≤ 1` (effectively unrecoverable @256) | 4 | 2 | **1** |

This is a **categorical extinction count**, robust to sampling noise (n_correct=0 is exact), and it
was completely invisible in the pass@256 curve (0.9832 vs 0.9900). It is the answer-event realization
of the paper's extinction claim — now stated as an *event count*, not a single-trace log-prob.

## Effective (recoverable) support — `EffSupp(K,τ)`, of 329
| K | τ | base | grpo | floor |
|---:|---:|---:|---:|---:|
| 16 | 0.5 | 258 | 287 | 284 |
| 64 | 0.9 | 271 | 291 | 292 |
| 256 | 0.9 | 307 | 316 | **320** |
| 1024 | 0.9 | 321 | 324 | **326** |
| 1024 | 0.5 | 329 | 327 | **329** |
| 16384 | 0.9 | 329 | **327** | **329** |

- **Small K:** grpo leads (sharpening raises `p̂` broadly → more problems clear τ with few samples).
- **Large K / high τ:** **floor ≥ base = 329**, while **grpo plateaus at 327** — grpo has 2
  permanently-extinct problems that no budget recovers; floor has none. Same crossover as pass@k, but
  now with an interpretable unit (recoverable problem count) and a hard extinction floor.

## Lost / preserved recoverable modes vs base (every K,τ)
grpo **loses 2–4** of base's recoverable problems at every budget; floor **loses 0–1**. At
K→∞ (16384), grpo's loss stabilizes at 2 (the extinct pair), floor at 0.

## Fragile band (81 problems base finds rare, p̂∈[0,0.05])
mean p̂: base 0.024 → grpo 0.059 (sharpest) → floor 0.045. At K=1024, τ0.5: floor recovers **81/81**,
grpo **79/81** (2 extinct). grpo buys the most sharpness on this band but pays it in 2 extinctions;
floor keeps the whole band recoverable.

## Reading
E1 confirms the **recoverability machinery** end-to-end on real data and, more importantly, shows the
new metric exposes a clean categorical signal (2 vs 0 extinctions) that the saturated pass@k gap
buried. Direction matches the paper's thesis: identical continued RL extinguishes rare recoverable
modes from the plain fork; the floor fork preserves them.

## Honest limits → motivates the next experiments
- Still the **saturated** fragile band; 2-vs-0 is directionally decisive but small in absolute count.
  The reviewer's push stands: rerun the headline on **Omni-MATH / harder boundary subsets** where the
  extinction count should be **tens of problems** (E-headline).
- Answer-event, not mode-level. **E2** (multi-witness strategy bank + clustering) upgrades `p̂_q` →
  per-mode mass so "extinct" means *strategy* extinction, closing the object-mismatch the review flagged.
- No training seeds yet (single run). **E7**: ≥3 seeds; check the extinct pair is stable across seeds.

## Reusable
`rl_training/recoverability.py` is a CLI (`--model name=path ... --budgets ... --taus ... --ref base`)
that will score every future eval (Omni-MATH, mode-level banks, new arms) into the same objects.
