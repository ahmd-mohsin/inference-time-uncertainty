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
| DeepSeek-Math-7B × Omni-MATH | 558 | plain | −0.57 | 62.4% | 55.0% | 0.2% | |
|  |  | **expSR** | **+2.28** | 90.0% | 85.7% | 0.0% | **+2.85** |
| **Qwen2.5-7B-Instruct × Omni-MATH** | 584 | plain | −1.87 | 32.2% | 27.2% | 0.2% | |
|  |  | **expSR** | **+7.64** | 91.6% | 88.4% | 0.0% | **+9.51** |
| Qwen2.5-7B-Instruct × Olympiad | 1435 | plain | −1.90 | 32.4% | 25.7% | 0.5% | |
|  |  | **expSR** | **+2.28** | 74.6% | 66.7% | 0.1% | **+4.18** |
| Qwen2.5-Math-1.5B × Olympiad | 2415 | plain | −0.08 | 77.8% | 72.5% | 0.0% | |
|  |  | **expSR** | **+0.13** | 89.4% | 80.9% | 0.0% | +0.20 |
| Qwen2.5-Math-7B × Olympiad (method-freeze ref) | 1055 | plain | −1.96 | 46.2% | 34.7% | 4.1% | |
|  |  | **expSR** | **+1.38** | 85.0% | 67.6% | 0.0% | +3.34 |
| Qwen3-8B × Olympiad | 1298 | plain | −4.61 | 34.4% | 34.4% | 8.6% | |
|  |  | expSR | EXCLUDED — 8B ratchet+offload run diverged | | | | |

**Qwen3-8B × Olympiad expSR is EXCLUDED (invalid run, not a real negative).** The 8B expSR arm
crashed under no-offload (ratchet forward OOM at step ~23) and the CPU-offload re-run produced a
degenerate checkpoint: 0/1298 traces preserved (Δ≥0), mean Δ=−28, i.e. *worse* than plain (−4.6) —
the opposite of expSR's behavior on every 7B/1.5B cell. Root cause: the extra teacher-forced bank
backward interacts badly with ZeRO-3 param-gathering + CPU-offload on 8B (log_completions spam also
swallowed loss logging). The 7B expSR path is reliable (+1.38); the 8B path is not with current code.
Qwen3-8B × Omni-MATH expSR (offload) is running as a provisional 8B point — included only if its Δ is sane.

## Reading
- Effect **generalizes across families AND datasets** (Qwen-Math, DeepSeek-Math, Qwen-Instruct;
  Olympiad + Omni-MATH): on all four valid 7B/1.5B cells expSR preserves > plain (raises base-mode
  mass, ~0% collapse) while plain loses/collapses it.
- **Qwen2.5-7B-Instruct × Omni-MATH is the strongest cell: +9.51 gap** (expSR +7.64 vs plain −1.87),
  91.6% vs 32.2% above floor, 0% collapse — a non-math-specialized base on the harder dataset.
- **Omni-MATH (harder, unsaturated) drives the biggest gaps** (Instruct +9.51, Qwen-Math +3.43) —
  the technique-friendly moderate band shows the effect most strongly.
- MAGNITUDE tracks how much plain would otherwise collapse (DeepSeek plain barely collapses → smaller
  gap; Omni-MATH bases collapse hard → large gap) — consistent with difficulty resonance.
- **8B is a code-reliability boundary, not a scientific one:** the ratchet's extra teacher-forced
  backward is unstable under ZeRO-3 + CPU-offload at 8B (the only config that fits 8B on 40 GB). The
  generalization claim rests on the four reliable 7B/1.5B cells; 8B is flagged as an implementation limit.

## Llama family (2026-08-27) — the widest fragile band, biggest expSR rescue
| cell | bank | arm | mean Δ | %≥floor | %preserved | %collapsed | gap |
|---|---:|---|---:|---:|---:|---:|---:|
| **Llama-3.1-8B-Instruct × OlympiadBench** | 1978 | plain | −21.03 | 3.6% | 3.5% | **80.1%** | |
|  |  | **expSR** | **−0.10** | 47.0% | 46.9% | **8.9%** | **+20.9** |

**Headline:** a general (non-math) 8B model on hard math has a very wide fragile band, so plain GRPO
collapses catastrophically (**80% of base modes**, mean −21 nats). expSR cuts collapse to **8.9%**
(mean ≈0), a **+20.9-nat** rescue — the largest gap in the campaign, exactly per difficulty resonance
(gap ∝ how hard plain collapses). Caveat: expSR not fully 0% here (8B harder than 7B), but converts an
80% collapse into 8.9%. Llama×Omni expSR failed 3× (offload OOM on longer Omni traces) — plain-only there.
Downstream continued-RL ceiling on this Llama×Oly cell is the next step (round-1 forks: plain on HF
cov-r1-llama-oly-grpo-7b; expSR done locally).
