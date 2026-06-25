# Config for the verification-generation gap study.
from dataclasses import dataclass


@dataclass
class GapConfig:
    # --- model ---
    # Qwen3-4B is the small Qwen3 reasoning model (there is no Qwen3-7B; for a 7B use
    # Qwen/Qwen2.5-7B-Instruct via --model). Single GPU each; we data-parallel across 8.
    model_name: str = "Qwen/Qwen3-4B"
    dtype: str = "bfloat16"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9

    # --- generation (G) ---
    n_chains: int = 16            # chains per problem; G = pass@k over these
    gen_temperature: float = 0.7
    gen_top_p: float = 0.95
    # 32768, NOT 16384: the topo run truncated 41% of chains at 16k, contaminating labels
    # (a cut-off chain is "no answer", not "wrong"). Bigger budget lets chains finish.
    max_new_tokens: int = 32768

    # --- verification (V) ---
    # Self-verification is a SHORT yes/no judgment -> cheap. We let the model think briefly
    # then commit to a verdict token we can score.
    verify_temperature: float = 0.0   # deterministic judgment
    verify_max_tokens: int = 2048
    n_verify_samples: int = 1         # >1 => average P(YES) over samples (temp>0) for a
                                      #        smoother verifier score; 1 is greedy.

    # --- data ---
    dataset: str = "aime_all"         # AIME 2024+2025+2026 = 90 problems
    n_problems: int = 90
    seed: int = 42
    output_dir: str = "data/verification_gap_qwen4b"
