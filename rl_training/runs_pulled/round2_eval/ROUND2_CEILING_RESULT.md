# Round-2 Continued-RL Ceiling — 3-way pass@k (HEADLINE)

**Setup.** Two round-1 forks of Qwen2.5-Math-7B — `grpo` (plain GRPO) and `floor` (GRPO + support-floor,
coverage-preserving) — each carried to step 400, then given **identical plain full-FT GRPO continued RL**
to step 100 (`r2-from-grpo`, `r2-from-floor`). Evaluated pass@k against `base` on the **fragile/hard
Olympiad band** (n_problems = 329, subset = hard, n_samples = 1024). Base pass@k reused from round-1.

| k | base | r2-from-grpo | r2-from-floor | floor − grpo |
|---:|---:|---:|---:|---:|
| 1 | 0.1728 | **0.3780** | 0.2986 | −0.0794 |
| 2 | 0.2953 | **0.5212** | 0.4518 | −0.0694 |
| 4 | 0.4539 | **0.6510** | 0.6053 | −0.0457 |
| 8 | 0.6160 | **0.7601** | 0.7366 | −0.0235 |
| 16 | 0.7519 | **0.8456** | 0.8381 | −0.0075 |
| 32 | 0.8524 | 0.9067 | **0.9069** | +0.0002 |
| 64 | 0.9175 | 0.9456 | **0.9494** | +0.0037 |
| 128 | 0.9546 | 0.9696 | **0.9754** | +0.0058 |
| 256 | 0.9763 | 0.9832 | **0.9900** | +0.0068 |

**Result — a crossover at k≈32.**
- **Small k (pass@1–16):** plain GRPO wins. Continued RL from the plain fork sharpens the policy and
  lifts greedy/low-k accuracy hard (pass@1 0.17 → 0.38).
- **Large k (pass@32–256, the coverage ceiling):** the **coverage-preserved `floor` fork overtakes
  plain GRPO**, and the gap widens monotonically with k (+0.0037 @64 → +0.0068 @256, reaching 0.990 @256).

**Takeaway (the paper's thesis).** Continued RL is necessary to realize the benefit of coverage
preservation: the support-floor fork does not look better mid-training at low k, but it reaches a
**higher continued-RL coverage ceiling** — plain GRPO trades away the large-k tail that the floor
fork keeps. Both continued-RL arms dominate `base` everywhere.

Models (durable): HF `muahmed7338/cov-r2-from-{grpo,floor}-7b` @ checkpoint-100.
Result JSONs: `passk_r2_{grpo,floor}_frag.json` (this dir); base `round1_fork/passk_base_r1_frag.json`.
