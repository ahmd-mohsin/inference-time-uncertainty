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
    # Context window = 16384 ("max token length"). Generation budget fits within it with
    # headroom for the (short) AIME prompt. This is the fix for the smoke truncation:
    # completions are NOT capped tiny (1024 truncated every chain -> pass@k=0), they get
    # ~14k tokens to think, inside a 16k context.
    max_model_len: int = 16384
    max_completion_length: int = 14336
    gen_temperature: float = 1.0
    gen_top_p: float = 1.0
    learning_rate: float = 1e-6
    beta: float = 0.0                # KL coeff; 0 = TRL default (no ref-KL)
    epsilon: float = 0.2
    num_train_steps: int = 500
    # Each arm runs on a full 8-GPU node (4 nodes x 8 A100 = 32 GPUs). ZeRO-3 shards the
    # training states across the 8 GPUs, so memory is comfortable at 16k context; bs=2
    # per device with grad-accum 4 -> effective 8 prompts/update.
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    scale_rewards: str = "group"     # TRL default
    use_vllm: bool = True
    vllm_mode: str = "colocate"
    # colocate vLLM: 0.4 fraction holds model+KV for generation; ZeRO-3 sharding leaves
    # the rest for training. vLLM context = 16384 (matches max_model_len). expandable
    # segments reduces long-context fragmentation.
    vllm_gpu_memory_utilization: float = 0.4
    vllm_enable_sleep_mode: bool = False
    vllm_max_model_length: int = 16384

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
