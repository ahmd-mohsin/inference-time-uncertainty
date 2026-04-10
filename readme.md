# RMI-Guided Tree Search for Mathematical Reasoning

## Inference-Time Compute via Diversity-Aware Step-Level Search

**Author:** Muhammad — Stanford University
**Target:** NeurIPS 2025 Submission
**Model:** Qwen/Qwen2.5-Math-7B-Instruct + Qwen/Qwen2.5-Math-PRM-7B
**Benchmarks:** AIME 2024, AMC, MATH500, DeepMath

---

## What This Project Does

This project implements **RMI-Tree** (Reverse Mutual Information-Guided Tree Search), a novel inference-time algorithm that improves mathematical reasoning on hard competition problems (AIME, AMC) by combining **quality scoring** (via a Process Reward Model) with **diversity scoring** (via KL divergence between solution branches).

The core equation:

```
Score(node) = R(node) + λ · D_KL( p(·|node) || (1/n) Σ_i p(·|sibling_i) )
```

Where:
- `R(node)` = Process Reward Model score (quality)
- `D_KL(...)` = KL divergence of this node's continuation distribution from the average sibling distribution (diversity)
- `λ` = hyperparameter balancing quality vs diversity

## Why This Works (Motivation from Prior Phases)

This project builds on four prior phases of research that established:

1. **Entropy inversion** (Phase 1, n=500): The model is more confident when wrong (H=0.252) than when right (H=0.446). Internal uncertainty signals are uninformative.

2. **Strategy homogeneity** (Phase 3): Temperature sampling at T=0.8 produces identical wrong strategies across K=3 continuations for most incorrect problems.

3. **Embedding perturbations fail** (Phase 2, Phase 4): Both PTCS and EGMI are absorbed by LayerNorm before they can redirect reasoning.

RMI-Tree solves this differently: it does not rely on the model detecting its own uncertainty. Instead, it forces **inter-branch diversity** by rewarding nodes whose continuation distributions diverge from the pool. The diversity signal comes from branch comparison, not self-assessment.

## How It Differs From REBASE

REBASE (Wu et al., ICLR 2025) uses PRM-only scoring:
```
Score_REBASE(node) = R(node)
```

RMI-Tree adds a diversity term:
```
Score_RMI(node) = R(node) + λ · D_KL(p_node || p_pool_avg)
```

On easy problems (GSM8K), diversity adds little because there's typically one natural strategy. On hard problems (AIME), where 2-3 genuinely different solution strategies exist, the diversity term prevents premature convergence to a single (possibly wrong) approach.

## Project Structure

```
~/inference-time-uncertainty/
├── src/
│   ├── search/                        # NEW — RMI tree search
│   │   ├── __init__.py
│   │   ├── tree.py                    # StepNode, SolutionTree
│   │   ├── diversity.py               # KL divergence, cosine divergence
│   │   ├── rmi_tree_search.py         # Core RMI-Tree algorithm
│   │   ├── rebase_baseline.py         # REBASE ablation (λ=0)
│   │   └── sampling_vote.py           # Sampling+vote baseline
│   ├── reward/                        # NEW — PRM interface
│   │   ├── __init__.py
│   │   └── prm.py                     # QwenMathPRM, DummyPRM
│   ├── uncertainty/                   # FROM PRIOR PHASES
│   │   ├── entropy_filter.py
│   │   ├── semantic_zone.py
│   │   ├── disagreement.py
│   │   ├── mixture_injector.py
│   │   └── pretokencommitment.py
│   ├── inference/                     # FROM PRIOR PHASES
│   │   ├── digte_generator.py
│   │   ├── egmi_generator.py
│   │   └── ptcs_generator.py
│   ├── baselines/                     # FROM PRIOR PHASES
│   │   ├── greedy.py
│   │   ├── beam_search.py
│   │   ├── adadec_math.py
│   │   ├── entropy_only_expansion.py
│   │   └── prompt_only.py
│   ├── data/
│   │   ├── dataset.py                 # load_math500, load_aime, format_prompt
│   │   └── model_loader.py            # ModelLoader
│   └── evaluation/
│       ├── metrics.py
│       ├── compute_matched.py
│       └── trigger_analysis.py
├── configs/
│   ├── inference_config.yaml          # Old config (greedy/DIGTE/EGMI)
│   └── rmi_search_config.yaml         # New config (RMI-Tree)
├── run_inference.py                   # Old entry point
├── run_rmi_search.py                  # New entry point (RMI-Tree)
├── run_ablation_sweep.py              # Full ablation automation
├── analyze_rmi_results.py             # Results tables and analysis
├── calibrate.py
├── calibrate_egmi.py
└── run_commitment_analysis.py
```

## Algorithm (Step by Step)

1. Given problem x, sample N first-step completions from the policy model at temperature T → depth-0 nodes

2. At each depth d:
   - Score all leaf nodes with the PRM: R(node)
   - Compute each node's KL divergence from its sibling pool: div(node)
   - Compute combined score: Score(node) = R(node) + λ · div(node)
   - Allocate expansion widths via softmax-normalized scores: W_j = Round(B_d · softmax(Score_j / T_b))
   - For each node, sample W_j children (next reasoning steps)
   - Update budget: B_{d+1} = B_d - n_completed

3. Repeat until budget exhausted or max_depth reached

4. Aggregate completed solutions via weighted majority vote (weight = PRM reward)

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_solutions` | 32 | Total solution budget N |
| `lambda_diversity` | 0.5 | Weight on diversity term (0 = REBASE) |
| `max_depth` | 20 | Maximum reasoning steps |
| `max_step_tokens` | 512 | Max tokens per step |
| `sampling_temperature` | 0.7 | Policy model sampling temperature |
| `balance_temperature` | 0.1 | Softmax temperature for budget allocation |
| `continuation_topk` | 50 | Top-k tokens stored for KL computation |
| `aggregation` | weighted_vote | How to pick final answer |
| `max_new_tokens` | 8192 | Total token limit (high for AIME) |

## Dependencies

```
torch>=2.0
transformers>=4.40
datasets
jsonlines
numpy
pyyaml
```

Optional (for PRM quantization):
```
bitsandbytes
accelerate
```