# ARM C: SFT on the model's own verified-correct GSM8K trajectories (LoRA r32, matches the GRPO adapter for a
# fair A-vs-C comparison at fixed GSM8K experience). Prompt-completion loss (completion only). Saves checkpoints.
# Usage: python -m rl_training.sft_train --data <sftdata.jsonl> --model Qwen/Qwen2.5-3B --out <dir> \
#   --max-steps 400 --save-steps 100 --seed 0
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--model", default="Qwen/Qwen2.5-3B")
    ap.add_argument("--out", required=True); ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--save-steps", type=int, default=100); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-5); ap.add_argument("--bsz", type=int, default=8)
    # H5 subspace localization: restrict LoRA to attention-only or MLP-only to find WHERE the
    # mass-placing (OOD transfer) lives. all = the standard arm-C set.
    ap.add_argument("--target-modules", default="all", choices=["all", "attn", "mlp"])
    # H-A branch / RL→SFT: start SFT from an existing (GRPO) adapter checkpoint — merge it into the base,
    # then attach a fresh LoRA on top (same pattern as train_grpo's merge_adapter_if_needed warm-start).
    ap.add_argument("--init-adapter", default="")
    a = ap.parse_args()
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig
    from trl import SFTTrainer, SFTConfig
    ds = load_dataset("json", data_files=a.data)["train"]
    print(f"[sft] {len(ds)} verified trajectories from {a.data}")
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    base = a.model
    if a.init_adapter:
        from rl_training.model_utils import merge_adapter_if_needed
        base = merge_adapter_if_needed(a.init_adapter)   # merge GRPO ckpt into base → SFT fresh LoRA on top
        print(f"[sft] init-adapter merged: {a.init_adapter} -> {base}")
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, trust_remote_code=True)
    _tm = {"all": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
           "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
           "mlp": ["gate_proj", "up_proj", "down_proj"]}[a.target_modules]
    print(f"[sft] target_modules={a.target_modules}: {_tm}")
    peft = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.0, task_type="CAUSAL_LM", target_modules=_tm)
    cfg = SFTConfig(output_dir=a.out, per_device_train_batch_size=a.bsz, gradient_accumulation_steps=1,
                    learning_rate=a.lr, max_steps=a.max_steps, save_steps=a.save_steps, save_total_limit=20,
                    logging_steps=10, bf16=True, seed=a.seed, report_to="none", max_length=1536,
                    completion_only_loss=True, warmup_ratio=0.03, lr_scheduler_type="cosine")
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, peft_config=peft, processing_class=tok)
    trainer.train()
    trainer.save_model(a.out)
    print(f"[sft] done -> {a.out}")

if __name__ == "__main__":
    main()
