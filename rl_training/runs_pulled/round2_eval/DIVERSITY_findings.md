# Diversity campaign — expSR (frozen method) vs plain GRPO across families & datasets

Each cell: full pipeline on that (model, dataset)'s fragile band — difficulty prepass -> base-correct
bank (+ref_logprob) -> train PLAIN vs expSR (150 steps, identical) -> teacher-forced Δ=logπ−logπ_base
over the cell's own bank. Run on new node mi-076bf1 (2026-08-24) after migrating from the dying node.

| cell | fragile(hard) | bank | arm | mean Δ | %≥α-floor | %preserved | %collapsed |
|---|---:|---:|---|---:|---:|---:|---:|
| **DeepSeek-Math-7B × Olympiad** | 333 | 2015 | plain | −0.18 | 71.1% | 59.0% | 0.1% |
|  |  |  | **expSR** | **+0.92** | **84.1%** | **73.7%** | **0.0%** |
| Qwen3-8B × Olympiad | 199 | ~1298 | plain | (running) | | | |
| Qwen2.5-Math-7B × Omni-MATH | (prepass on-node) | | (running) | | | |

Reference (Qwen2.5-Math-7B × Olympiad, from method-freeze): plain −1.96, expSR +1.38 (gap +3.34).

## Reading (DeepSeek, first cell)
Effect **generalizes in direction** to a new family: expSR preserves > plain (floor−plain=+1.10 nats,
+13pts above floor, expSR raises mass / plain loses it). MAGNITUDE smaller than Qwen-Math (DeepSeek's
plain barely collapses: −0.18, 0.1%), consistent with difficulty resonance — the gain scales with how
much plain GRPO would otherwise collapse. Honest: direction robust across families; strength setting-dependent.
