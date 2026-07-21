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
    p.add_argument("--gradient-accumulation-steps", type=int,
                   default=RLConfig.gradient_accumulation_steps,
                   help="override grad-accum (used by the DP-vLLM launcher to hold the effective "
                        "batch constant when training on fewer ranks)")
    p.add_argument("--novelty-lambda", type=float, default=RLConfig.novelty_lambda)
    p.add_argument("--no-novelty", action="store_true", help="ablation: plain GRPO")
    # EXPERIMENT A: fragile-band curriculum (oversample base-pass@1 in [lo,hi])
    p.add_argument("--curriculum", action="store_true",
                   help="ExpA: oversample fragile-band problems (needs --difficulty-json with pass1)")
    p.add_argument("--frag-lo", type=float, default=0.02)
    p.add_argument("--frag-hi", type=float, default=0.30)
    p.add_argument("--frag-oversample", type=int, default=3)
    # EXPERIMENT C: rarity-weighted correctness bonus (instead of / with semantic novelty)
    p.add_argument("--rarity-bonus", action="store_true",
                   help="ExpC: add rarity-weighted correctness reward (up-weights rare-correct modes)")
    p.add_argument("--rarity-lambda", type=float, default=0.5)
    # METHOD 3: coverage-in-the-loop reward (marginal pass@k contribution: lam/n_correct_in_group)
    p.add_argument("--coverage-reward", action="store_true",
                   help="M3: reward correct rollouts by marginal group-coverage value (lam/n_correct)")
    p.add_argument("--coverage-lambda", type=float, default=1.0)
    p.add_argument("--no-vllm", action="store_true")
    p.add_argument("--resume-from", default="", help="checkpoint dir to resume (Component B loop)")
    p.add_argument("--init-adapter", default="", help="warm-start: load this saved LoRA adapter "
                   "dir as the starting weights (no optimizer state needed, unlike --resume-from)")
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
    # oursAB feeds a previous segment's output (seg_r_sft / seg_r) in as --model. If that is a
    # bare LoRA adapter, merge it into its base so GRPOTrainer/vLLM get a loadable full model,
    # then attach a fresh LoRA on top for this segment.
    from rl_training.model_utils import merge_adapter_if_needed
    a.model = merge_adapter_if_needed(a.model)
    a.model = _resolve_local_model(a.model)
    cfg = RLConfig(model_name=a.model, dataset=a.dataset, n_problems=a.n_problems,
                   difficulty_json=a.difficulty_json, output_dir=a.output_dir,
                   num_generations=a.num_generations, num_train_steps=a.num_train_steps,
                   learning_rate=a.lr, beta=a.beta,
                   max_completion_length=a.max_completion_length,
                   gradient_accumulation_steps=a.gradient_accumulation_steps,
                   novelty_lambda=a.novelty_lambda,
                   novelty_enabled=not a.no_novelty, use_vllm=not a.no_vllm)

    from trl import GRPOTrainer, GRPOConfig
    from peft import LoraConfig

    train_dataset = build_dataset(cfg.dataset, cfg.model_name, cfg.n_problems, cfg.seed,
                                  cfg.difficulty_json, cfg.hard_only,
                                  curriculum=a.curriculum, frag_lo=a.frag_lo, frag_hi=a.frag_hi,
                                  frag_oversample=a.frag_oversample)
    print(f"train dataset: {len(train_dataset)} rows (curriculum={a.curriculum}, "
          f"hard-targeted={bool(cfg.difficulty_json) and not a.curriculum})")

    # reward functions: correctness always; novelty optional (ablation)
    reward_funcs = [correctness_reward]
    reward_weights = [1.0]
    if cfg.novelty_enabled:
        reward_funcs.append(make_novelty_bonus(cfg.embedding_model, cfg.novelty_lambda,
                                               cfg.novelty_metric, cfg.novelty_correct_only))
        reward_weights.append(1.0)
    if a.rarity_bonus:  # EXPERIMENT C
        from rl_training.rewards import make_rarity_bonus
        reward_funcs.append(make_rarity_bonus(a.rarity_lambda))
        reward_weights.append(1.0)
    if a.coverage_reward:  # METHOD 3: coverage-in-the-loop
        from rl_training.rewards import make_coverage_reward
        reward_funcs.append(make_coverage_reward(a.coverage_lambda))
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
        logging_steps=10, save_steps=cfg.save_steps, save_total_limit=cfg.save_total_limit,
        log_completions=cfg.log_completions,
        bf16=True, gradient_checkpointing=True,
        # non-reentrant checkpointing required with LoRA — reentrant recompute mismatches
        # metadata and raises CheckpointError mid-step.
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # Always let GRPOTrainer build the model the SAME way (model id + peft_config) so DeepSpeed
    # ZeRO-2 shards it correctly. Pre-loading a full model ourselves (PeftModel.from_pretrained
    # on rank 0) caused CUDA OOM — rank 0 held an unsharded copy. Instead, for WARM-START we
    # load the saved adapter weights INTO the already-built (sharded) LoRA layers afterward.
    peft_config = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_target_modules), task_type="CAUSAL_LM",
    ) if cfg.use_lora else None
    trainer = GRPOTrainer(
        model=cfg.model_name, args=grpo_args, reward_funcs=reward_funcs,
        train_dataset=train_dataset, peft_config=peft_config,
    )
    if a.init_adapter:
        # WARM-START: overwrite the freshly-initialized LoRA weights with the saved adapter
        # (resumes policy weights from a prior run; optimizer/LR-schedule restart fresh, which
        # re-warms in a few steps at lr=1e-6). Memory profile == the working from-scratch path.
        # Saved keys look like 'base_model.model.model...lora_A.weight'; the live PEFT module's
        # state_dict uses the adapter name ('...lora_A.default.weight'). load_peft_weights +
        # set_peft_model_state_dict(adapter_name=...) handles that remap. We verify the load by
        # asserting non-trivial key overlap so a silent no-op (all-missing) can't pass unnoticed.
        from safetensors.torch import load_file
        print(f">> WARM-START: loading saved LoRA weights from {a.init_adapter}")
        sd = load_file(os.path.join(a.init_adapter, "adapter_model.safetensors"))
        # Saved keys are '...lora_A.weight'; the live PEFT module names the active adapter,
        # i.e. '...lora_A.default.weight'. Insert the adapter name so keys match exactly.
        # (Validated: this maps all 504 keys with 0 missing/unexpected.)
        remap = {}
        for k, v in sd.items():
            nk = k.replace(".lora_A.weight", ".lora_A.default.weight") \
                  .replace(".lora_B.weight", ".lora_B.default.weight")
            remap[nk] = v
        missing, unexpected = trainer.model.load_state_dict(remap, strict=False)
        missing_lora = [k for k in missing if "lora_" in k]
        print(f">> WARM-START: loaded {len(remap)} keys; missing_lora={len(missing_lora)} "
              f"unexpected={len(unexpected)}")
        if missing_lora or unexpected:
            raise RuntimeError(f"WARM-START key mismatch: missing_lora={len(missing_lora)} "
                               f"unexpected={len(unexpected)} (expected 0/0)")
    trainer.train(resume_from_checkpoint=a.resume_from or None)
    trainer.save_model(cfg.output_dir)
    print(f"saved GRPO model -> {cfg.output_dir}")


if __name__ == "__main__":
    main()
