# Topological Persistence Ceiling Detector

## Core Intuition

When an LLM generates K reasoning chains for a problem, those chains form a point cloud
in a high-dimensional space. The *topology* of that point cloud — its connected
components, loops, and voids — reveals whether the model's solution space is fully
explored or still has undiscovered structure.

If the topology stabilizes quickly (Betti numbers freeze with few chains), the model
has exhausted its representational capacity for this problem. Adding more chains will
not reveal new solution strategies — the model has hit a **ceiling** that only weight
updates can break through.

If the topology keeps evolving (new loops/features appearing as chains are added),
there is still unexplored structure — more test-time compute **will** help.

## Objective

Given 8 reasoning chains from a model:
1. Detect whether the model has hit a representational ceiling for this problem
2. Predict whether additional test-time compute will improve accuracy
3. Compare IID chains vs. DAD-conditioned chains to measure whether conditioning
   genuinely expands the solution manifold or just reshuffles within it

## Method

1. **Sample** K chains (default 8) using vLLM for batched generation
2. **Embed** each chain as either:
   - A single point (mean-pooled hidden states) — cheap but lossy
   - A curve (sequence of step-level embeddings) — preserves path structure
3. **Compute distances** between chains (cosine for points, DTW/Fréchet for curves)
4. **Persistent homology** on the distance matrix via Vietoris-Rips filtration:
   - H₀: connected components (how many distinct answer clusters)
   - H₁: loops (multiple distinct strategies connecting the same endpoints)
   - H₂: voids (higher-order strategic structure)
5. **Ceiling detection** from the persistence diagram:
   - Topology frozen (no H₁ features) → ceiling
   - H₁ features with long lifetimes → scalable
   - Betti curve convergence rate → how fast topology stabilizes
6. **Comparison**: repeat with DAD-conditioned chains to test if conditioning
   creates new topological features

## Grounding Paper

**"Geometry Score: A Method For Comparing Generative Adversarial Networks"**
Khrulkov & Oseledets, ICML 2018 — https://arxiv.org/abs/1802.02664

Uses persistent homology on point clouds of generated samples to detect mode collapse.
We adapt this: replace "GAN outputs" with "reasoning trajectories" and "mode collapse"
with "diversity collapse / model ceiling."

## Architecture

```
topological_persistence/
├── config.py            # Dataclass configs (sampling, topology, embedding, experiment)
├── sampler.py           # VLLMSampler, HFSampler, VLLMHiddenStateSampler
├── embeddings.py        # Point/curve/step-level trajectory embedding extraction
├── distances.py         # Cosine, DTW, Fréchet distance matrices
├── persistence.py       # Vietoris-Rips persistent homology (ripser/gudhi)
├── ceiling_detector.py  # Topological signals → ceiling/scalable verdict
├── conditioning.py      # DAD workspace conditioning for non-IID comparison
├── visualization.py     # Persistence diagrams, Betti curves, comparison plots
├── pipeline.py          # End-to-end orchestration per problem
├── analysis.py          # Post-hoc aggregation and ground-truth validation
└── run.py               # CLI entry point
```

## Usage

```bash
# Default: Qwen3-32B via vLLM, 8 chains, AIME 2024, curve embeddings
python -m topological_persistence.run

# Custom configuration
python -m topological_persistence.run \
    --model Qwen/Qwen3-32B \
    --dataset aime_2024 \
    --n-problems 10 \
    --n-chains 8 \
    --representation curve \
    --output-dir data/topological_outputs

# Without vLLM (uses HuggingFace transformers, extracts hidden states directly)
python -m topological_persistence.run --no-vllm --representation point

# With YAML config
python -m topological_persistence.run --config configs/topo_config.yaml
```

## Dependencies

Beyond the base environment:
```
pip install ripser persim matplotlib scipy
# OR
pip install gudhi matplotlib scipy
# For vLLM:
pip install vllm
```

## Key Signals

| Signal | Low Value | High Value |
|--------|-----------|------------|
| H₁ features | No loops → single strategy cluster (ceiling) | Multiple loops → diverse strategies (scalable) |
| H₁ max lifetime | Short-lived features → noise (ceiling) | Long-lived → robust structural diversity (scalable) |
| Betti convergence rate | Fast convergence → topology fully revealed (ceiling) | Slow convergence → still discovering (scalable) |
| Diversity gain (IID→Cond) | No gain → conditioning doesn't help (ceiling) | Gain → conditioning opens new paths (scalable) |
