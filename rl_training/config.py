# Config for RL post-training experiments (docs/RL.md).
#
# Thesis: standard GRPO sharpens pass@1 but shrinks pass@k (Yue et al. 2504.13837).
# We add (A) a group-relative semantic-novelty reward that protects rare *correct* modes,
# (B) off-policy harvesting of the model's own rare high-k successes, and
# (C) hard-problem targeting — to expand support (keep base-level large-k pass@k while
# gaining at small k).
from dataclasses import dataclass, field


@dataclass
class RLConfig:
    # --- model / engine ---
    # Small models; 4-node accelerate+DeepSpeed handles these comfortably.
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"   # or Qwen/Qwen3-4B
    dtype: str = "bfloat16"
    use_lora: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    lora_target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj")

    # --- data ---
    dataset: str = "aime_all"        # aime_all | math500 | gsm8k (src/data/dataset.py)
    n_problems: int = -1             # -1 = all
    seed: int = 42
    # Component C: restrict to problems with headroom (low pass@1, pass@k>0).
    # Built offline by difficulty_prepass.py; path to its json (or "" to skip filtering).
    difficulty_json: str = ""
    hard_only: bool = True           # if difficulty_json given, keep only "hard" problems

    # --- GRPO core (maps to trl.GRPOConfig) ---
    num_generations: int = 8         # G = group size (also drives novelty grouping)
    # 16k: reasoning models need room to think (truncation contaminates labels, as the
    # topo runs showed at 41% cut-off). Long completions are memory-heavy, so we OFFLOAD
    # optimizer+params to CPU (ZeRO-3 offload) and shrink per-device batch -> see
    # accelerate_zero3_offload.yaml + per_device_train_batch_size below.
    max_completion_length: int = 16384
    vllm_max_model_length_long: int = 18432   # 16384 + prompt headroom
    gen_temperature: float = 1.0
    gen_top_p: float = 1.0
    learning_rate: float = 1e-6
    beta: float = 0.0                # KL coeff; 0 = TRL default (no ref-KL)
    epsilon: float = 0.2
    num_train_steps: int = 500
    # 16k completions are activation-heavy: one sequence per device at a time, recover
    # effective batch via grad accumulation. 8 gen/prompt * 1 device-bs means each step
    # processes a group; grad-accum 8 -> effective 8 prompts/update.
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    scale_rewards: str = "group"     # TRL default
    use_vllm: bool = True
    vllm_mode: str = "colocate"
    # For 16k completions on 40GB A100: ZeRO-3 CPU-offloads optimizer+params (frees GPU),
    # vLLM gets a smaller fraction (it still needs the full model for gen, but a 0.35
    # fraction + 18k KV context is enough since training activations are now the priority).
    # expandable_segments fights the fragmentation OOM seen at long context.
    vllm_gpu_memory_utilization: float = 0.35
    vllm_enable_sleep_mode: bool = False
    vllm_max_model_length: int = 18432   # 16384 + prompt headroom

    # --- Component A: group-relative semantic-novelty reward ---
    novelty_enabled: bool = True
    novelty_lambda: float = 0.5      # reward = correct * (1 + lambda * novelty)
    novelty_correct_only: bool = True  # novelty measured only among CORRECT rollouts
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    novelty_metric: str = "cosine"   # cosine | euclidean ; distance in [0,1]-ish

    # --- Component B: off-policy tail harvesting (policy distillation) ---
    harvest_enabled: bool = True
    harvest_every_steps: int = 50    # interleave a harvest+SFT phase every K steps
    harvest_k: int = 64              # samples per hard problem during harvest
    harvest_temperature: float = 1.0
    harvest_max_keep: int = 2        # keep up to this many distinct correct rollouts/problem
    harvest_sft_epochs: int = 1
    harvest_sft_lr: float = 1e-6

    # --- output ---
    output_dir: str = "rl_training/runs/exp"
    run_name: str = "rl_expand"
    log_completions: bool = True
    save_steps: int = 100


@dataclass
class EvalConfig:
    """pass@k evaluation — the decisive metric (does the crossover disappear?)."""
    model_path: str = "Qwen/Qwen2.5-7B-Instruct"   # base, or a trained checkpoint dir
    dataset: str = "aime_all"
    n_problems: int = -1
    seed: int = 42
    k_values: tuple = (1, 2, 4, 8, 16, 32, 64, 128, 256)
    n_samples: int = 256             # generate this many, compute pass@k for all k<=this
    temperature: float = 1.0
    top_p: float = 1.0
    max_new_tokens: int = 4096
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    output_dir: str = "rl_training/runs/eval"
    tag: str = "base"                # label for this curve (base|grpo|ours-A|ours-AB)
