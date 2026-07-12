# Component B: off-policy tail harvesting (docs/RL.md §4.2).
#
# Sample the CURRENT policy at large k on hard problems, keep the rare correct rollouts
# (the tail on-policy GRPO would never reinforce because it's low-prob), and write them as
# an SFT dataset. A subsequent SFT pass on these rollouts injects the path into the
# high-probability region — self-distillation of the model's own tail. This is the
# mechanism that turns "redistribution within support" into support EXPANSION.
#
# Run as: sample (vLLM) -> dedup distinct correct -> jsonl ; then SFT (sft_step).
# Kept separate from the GRPO process so vLLM (harvest) and the trainer don't fight for GPU.

import argparse, json, os, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import (get_inference_dataset, format_prompt,
                              extract_numeric_answer, normalize_answer, answers_match)
from rl_training.model_utils import merge_adapter_if_needed
from rl_training.safe_match import safe_is_correct


def harvest(model_path, dataset, difficulty_json, k, max_keep, max_new_tokens,
            temperature, tensor_parallel_size, out_jsonl, n_problems=-1, all_problems=False,
            max_pass_rate=1.0, max_total=0):
    """Sample model at k on HARD problems; write distinct correct (prompt, completion) pairs.

    all_problems=True is the CONTROL for methodology fix #2: harvest correct rollouts from ALL
    problems (not just the hard band).

    SELECTIVE HARVEST (levers to stop Component B from flattening pass@1): our full-pipeline result
    showed oursABC recovers base-level COVERAGE but loses A's pass@1 gain — because SFT on the whole
    correct set pulls the model back toward base breadth. Two knobs make B target only the TRULY-LOST
    tail:
      max_pass_rate: skip a problem if the model already solves it easily (correct_frac > this among
                     the k samples). Harvesting easy problems just reinforces the dominant mode and
                     flattens the sharpened peak; we only want the RARE successes (the tail RL kills).
      max_total    : hard cap on total harvested rollouts (a gentler, smaller SFT set)."""
    from vllm import LLM, SamplingParams
    from transformers import AutoConfig

    # A GRPO segment saves a bare LoRA adapter; vLLM needs a full model -> merge first.
    model_path = merge_adapter_if_needed(model_path)

    problems = get_inference_dataset({"dataset": {"name": dataset, "split": "test",
                                                  "n_problems": n_problems, "seed": 42}})
    keep_ids = None
    if difficulty_json and os.path.exists(difficulty_json) and not all_problems:
        diff = json.load(open(difficulty_json))
        keep_ids = {d["problem_id"] for d in diff.get("per_problem", []) if d["label"] == "hard"}
    if keep_ids is not None:
        problems = [p for p in problems if p["problem_id"] in keep_ids]
    if all_problems:
        print(f"harvest: --all-problems CONTROL — harvesting from all {len(problems)} problems (not just hard)")
    if not problems:
        print("harvest: no hard problems; nothing to do"); return 0

    try:
        cap = int(getattr(AutoConfig.from_pretrained(model_path, trust_remote_code=True),
                          "max_position_embeddings", max_new_tokens))
    except Exception:
        cap = max_new_tokens
    mml = min(max_new_tokens + 1024, cap)

    llm = LLM(model=model_path, dtype="bfloat16", trust_remote_code=True,
              tensor_parallel_size=tensor_parallel_size, max_model_len=mml,
              gpu_memory_utilization=0.9, enable_prefix_caching=True)
    sp = SamplingParams(n=k, max_tokens=mml - 1024, temperature=temperature, top_p=1.0,
                        stop=["<|im_end|>", "<|endoftext|>"])
    prompts = [format_prompt(p, model_path) for p in problems]
    outs = llm.generate(prompts, sp)

    n_written = 0; n_skipped_easy = 0
    Path(out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w") as f:
        for p, prompt, o in zip(problems, prompts, outs):
            gold = str(p.get("gold_answer", ""))
            # score all k first so we know how RARE this problem's success is
            scored = [safe_is_correct(s.text, gold) for s in o.outputs]
            n_correct = sum(1 for ok, _ in scored if ok)
            if n_correct == 0:
                continue
            correct_frac = n_correct / max(1, len(scored))
            # SELECTIVE: only harvest the genuinely-rare tail. If the model already solves this
            # problem easily (correct_frac > max_pass_rate), re-SFTing it just reinforces the
            # dominant mode and flattens pass@1 — exactly what over-broad B did. Skip it.
            if correct_frac > max_pass_rate:
                n_skipped_easy += 1
                continue
            seen, kept = set(), 0
            for (ok, pred), s in zip(scored, o.outputs):
                if kept >= max_keep:
                    break
                if not ok:
                    continue
                if max_total and n_written >= max_total:
                    break
                # dedup by normalized reasoning signature so we keep DISTINCT correct paths
                sig = normalize_answer(pred) + "|" + str(len(s.text) // 200)
                if sig in seen:
                    continue
                seen.add(sig)
                f.write(json.dumps({"prompt": prompt, "completion": s.text,
                                    "problem_id": p["problem_id"]}) + "\n")
                kept += 1; n_written += 1
            if max_total and n_written >= max_total:
                print(f"harvest: hit max_total={max_total} cap"); break
    print(f"harvest: wrote {n_written} tail rollouts (skipped {n_skipped_easy} easy problems, "
          f"correct_frac>{max_pass_rate}) -> {out_jsonl}")
    return n_written


def sft_step(model_path, sft_jsonl, output_dir, lr, epochs, use_lora=True,
             lora_r=32, lora_alpha=64):
    """Off-policy SFT on harvested rollouts (TRL SFTTrainer). Saves to output_dir."""
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig
    # SFT base must be a full model: if the GRPO segment output is a bare adapter, merge it so we
    # SFT a clean full model + one fresh LoRA (not a bare adapter or a stacked double-adapter).
    model_path = merge_adapter_if_needed(model_path)
    ds = load_dataset("json", data_files=sft_jsonl, split="train")
    # SFTTrainer with prompt+completion columns trains on completion tokens only.
    args = SFTConfig(output_dir=output_dir, num_train_epochs=epochs, learning_rate=lr,
                     per_device_train_batch_size=1, gradient_accumulation_steps=8,
                     bf16=True, logging_steps=5, save_strategy="no",
                     gradient_checkpointing=True, completion_only_loss=True)
    peft_config = None
    if use_lora:
        from peft import LoraConfig
        peft_config = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.0,
                                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                 "gate_proj", "up_proj", "down_proj"],
                                 task_type="CAUSAL_LM")
    trainer = SFTTrainer(model=model_path, args=args, train_dataset=ds, peft_config=peft_config)
    trainer.train()
    trainer.save_model(output_dir)
    print(f"sft_step: saved -> {output_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["harvest", "sft"], required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--dataset", default="aime_all")
    ap.add_argument("--difficulty-json", default="")
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--max-keep", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--out-jsonl", default="rl_training/runs/harvest.jsonl")
    ap.add_argument("--output-dir", default="rl_training/runs/sft")
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--all-problems", action="store_true",
                    help="CONTROL: harvest from ALL problems, not just the hard band (isolates tail effect)")
    ap.add_argument("--max-pass-rate", type=float, default=1.0,
                    help="selective harvest: skip problems the model already solves with correct_frac>this (keep only rare tail)")
    ap.add_argument("--max-total", type=int, default=0,
                    help="cap total harvested rollouts (0=unlimited); a gentler, smaller SFT set")
    a = ap.parse_args()
    if a.mode == "harvest":
        harvest(a.model_path, a.dataset, a.difficulty_json, a.k, a.max_keep,
                a.max_new_tokens, a.temperature, a.tensor_parallel_size, a.out_jsonl,
                all_problems=a.all_problems, max_pass_rate=a.max_pass_rate, max_total=a.max_total)
    else:
        sft_step(a.model_path, a.out_jsonl, a.output_dir, a.lr, a.epochs)


if __name__ == "__main__":
    main()
