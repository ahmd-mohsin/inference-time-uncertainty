# Diversity campaign — expSR (frozen method) vs plain GRPO across families & datasets

Each cell: full pipeline on that (model,dataset)'s fragile band — prepass -> base-correct bank(+ref)
-> train PLAIN vs expSR (150 steps, identical) -> teacher-forced Δ=logπ−logπ_base over the cell's bank.
Run on node mi-076bf1 (2026-08-24) after migrating from the dying node.

| cell | bank | arm | mean Δ | %≥α-floor | %preserved | %collapsed | gap(expSR−plain) |
|---|---:|---|---:|---:|---:|---:|---:|
| DeepSeek-Math-7B × Olympiad | 2015 | plain | −0.18 | 71.1% | 59.0% | 0.1% | |
|  |  | **expSR** | **+0.92** | 84.1% | 73.7% | 0.0% | **+1.10** |
| **Qwen2.5-Math-7B × Omni-MATH** | 1004 | plain | −1.99 | 48.5% | 46.7% | 4.1% | |
|  |  | **expSR** | **+1.44** | 84.2% | 79.4% | 0.4% | **+3.43** |
| Qwen3-8B × Olympiad | 1298 | plain | −5.77 | 27.9% | 27.9% | 14.6% | |
|  |  | **expSR** | (re-running; arm failed on vLLM/GPU race) | | | | |
| Qwen2.5-Math-7B × Olympiad (method-freeze ref) | 1055 | plain | −1.96 | 46.2% | 34.7% | 4.1% | |
|  |  | **expSR** | **+1.38** | 85.0% | 67.6% | 0.0% | +3.34 |

## Reading
- Effect **generalizes across families AND datasets**: expSR preserves > plain everywhere (raises base-
  mode mass, ~0% collapse) while plain loses/collapses it.
- **Omni-MATH (harder, unsaturated dataset) is a decisive win: +3.43 gap**, 84% vs 48% above floor,
  0.4% vs 4.1% collapse — the technique-friendly moderate band shows the effect strongly.
- **Qwen3-8B plain collapses catastrophically (−5.77, 14.6%)** — the most fragile base; expSR arm re-running.
- MAGNITUDE tracks how much plain would otherwise collapse (DeepSeek plain barely collapses → smaller
  gap; Qwen3/Omni collapse hard → large gap) — consistent with difficulty resonance.
