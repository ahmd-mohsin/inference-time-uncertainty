# GRPO training entry point (docs/RL.md). Built on TRL GRPOTrainer.
#
# Reward = correctness_reward + novelty_bonus (Component A). Ablate to plain GRPO by
# --no-novelty (reward_weights=[1,0]). LoRA + colocate vLLM. Launch with `accelerate launch`
# (DeepSpeed ZeRO-3 across the 4 nodes); see scripts/rl_*.sh.
#
# Component B (off-policy harvesting) is an OUTER alternating loop in scripts/rl_train_full.sh:
#   train_grpo (K steps) -> harvest.py harvest -> harvest.py sft -> resume from checkpoint.
# This script handles one GRPO segment.

import argparse, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_training.config import RLConfig
from rl_training.data import build_dataset
from rl_training.rewards import correctness_reward, make_novelty_bonus


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=RLConfig.model_name)
    p.add_argument("--dataset", default=RLConfig.dataset)
    p.add_argument("--n-problems", type=int, default=RLConfig.n_problems)
    p.add_argument("--difficulty-json", default=RLConfig.difficulty_json)
    p.add_argument("--output-dir", default=RLConfig.output_dir)
    p.add_argument("--num-generations", type=int, default=RLConfig.num_generations)
    p.add_argument("--num-train-steps", type=int, default=RLConfig.num_train_steps)
    p.add_argument("--lr", type=float, default=RLConfig.learning_rate)
    p.add_argument("--beta", type=float, default=RLConfig.beta)
    p.add_argument("--max-completion-length", type=int, default=RLConfig.max_completion_length)
    p.add_argument("--novelty-lambda", type=float, default=RLConfig.novelty_lambda)
    p.add_argument("--no-novelty", action="store_true", help="ablation: plain GRPO")
    p.add_argument("--no-vllm", action="store_true")
    p.add_argument("--resume-from", default="", help="checkpoint dir to resume (Component B loop)")
    return p.parse_args()


def _resolve_local_model(model_id: str) -> str:
    """Return a local snapshot dir for model_id if cached, else model_id unchanged.

    Under HF_HUB_OFFLINE=1, TRL's colocate vLLM ModelConfig validation can fail to resolve
    a bare HF id ('Qwen/Qwen3-8B') even when cached. Passing the concrete snapshot path
    sidesteps any hub lookup for BOTH the HF trainer and vLLM.
    """
    if os.path.isdir(model_id):
        return model_id
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(model_id)  # offline -> returns cached snapshot dir
    except Exception:
        return model_id


def main():
    a = build_args()
    a.model = _resolve_local_model(a.model)
    cfg = RLConfig(model_name=a.model, dataset=a.dataset, n_problems=a.n_problems,
                   difficulty_json=a.difficulty_json, output_dir=a.output_dir,
                   num_generations=a.num_generations, num_train_steps=a.num_train_steps,
                   learning_rate=a.lr, beta=a.beta,
                   max_completion_length=a.max_completion_length,
                   novelty_lambda=a.novelty_lambda,
                   novelty_enabled=not a.no_novelty, use_vllm=not a.no_vllm)

    from trl import GRPOTrainer, GRPOConfig
    from peft import LoraConfig

    train_dataset = build_dataset(cfg.dataset, cfg.model_name, cfg.n_problems, cfg.seed,
                                  cfg.difficulty_json, cfg.hard_only)
    print(f"train dataset: {len(train_dataset)} problems (hard-targeted={bool(cfg.difficulty_json)})")

    # reward functions: correctness always; novelty optional (ablation)
    reward_funcs = [correctness_reward]
    reward_weights = [1.0]
    if cfg.novelty_enabled:
        reward_funcs.append(make_novelty_bonus(cfg.embedding_model, cfg.novelty_lambda,
                                               cfg.novelty_metric, cfg.novelty_correct_only))
        reward_weights.append(1.0)

    grpo_args = GRPOConfig(
        output_dir=cfg.output_dir, run_name=cfg.run_name,
        learning_rate=cfg.learning_rate, beta=cfg.beta, epsilon=cfg.epsilon,
        num_generations=cfg.num_generations,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        max_completion_length=cfg.max_completion_length,
        temperature=cfg.gen_temperature, top_p=cfg.gen_top_p,
        max_steps=cfg.num_train_steps, scale_rewards=cfg.scale_rewards,
        reward_weights=reward_weights,
        use_vllm=cfg.use_vllm, vllm_mode=cfg.vllm_mode,
        vllm_server_host=cfg.vllm_server_host, vllm_server_port=cfg.vllm_server_port,
        logging_steps=10, save_steps=cfg.save_steps, log_completions=cfg.log_completions,
        bf16=True, gradient_checkpointing=True,
        # non-reentrant checkpointing required with LoRA — reentrant recompute mismatches
        # metadata and raises CheckpointError mid-step.
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    peft_config = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_target_modules), task_type="CAUSAL_LM",
    ) if cfg.use_lora else None

    trainer = GRPOTrainer(
        model=cfg.model_name, args=grpo_args, reward_funcs=reward_funcs,
        train_dataset=train_dataset, peft_config=peft_config,
    )
    trainer.train(resume_from_checkpoint=a.resume_from or None)
    trainer.save_model(cfg.output_dir)
    print(f"saved GRPO model -> {cfg.output_dir}")


if __name__ == "__main__":
    main()
