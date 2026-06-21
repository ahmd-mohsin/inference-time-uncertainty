# Configuration for the topological persistence ceiling detector.
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SamplingConfig:
    model_name: str = "Qwen/Qwen3-32B"
    n_chains: int = 8
    max_new_tokens: int = 8192
    temperature: float = 0.7
    top_p: float = 0.95
    dtype: str = "bfloat16"
    tensor_parallel_size: int = 1
    use_vllm: bool = True
    enable_thinking: bool = True


@dataclass
class TopologyConfig:
    max_homology_dim: int = 2
    n_radii: int = 100
    distance_metric: str = "cosine"
    stability_window: int = 3
    betti_convergence_threshold: float = 0.05


@dataclass
class EmbeddingConfig:
    representation: str = "curve"
    curve_distance: str = "dtw"
    step_pooling: str = "last_token"
    hidden_layer: int = -1
    max_steps_per_chain: int = 64
    subsample_tokens: int = 128


@dataclass
class ExperimentConfig:
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    dataset: str = "aime_2024"
    n_problems: int = 5
    output_dir: str = "data/topological_outputs"
    seed: int = 42
    conditioned_chains: bool = True
    n_conditioned_chains: int = 8


def load_config(path: Optional[str] = None) -> ExperimentConfig:
    if path is None:
        return ExperimentConfig()
    import yaml
    with open(path) as f:
        raw = yaml.safe_load(f)
    cfg = ExperimentConfig()
    if "sampling" in raw:
        cfg.sampling = SamplingConfig(**raw["sampling"])
    if "topology" in raw:
        cfg.topology = TopologyConfig(**raw["topology"])
    if "embedding" in raw:
        cfg.embedding = EmbeddingConfig(**raw["embedding"])
    for k in ("dataset", "n_problems", "output_dir", "seed", "conditioned_chains", "n_conditioned_chains"):
        if k in raw:
            setattr(cfg, k, raw[k])
    return cfg
