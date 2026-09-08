# Self-repair GRPO: train a model to fix its own failures given ONLY the error (buggy code hidden — the
# anchoring-antidote recipe). Rollout prompt = repair prompt (problem + real error); reward = does the
# generated code PASS the unit tests (execute-verify). TRL GRPOTrainer + LoRA + colocate vLLM.
#
# Consumes repair_*.jsonl from dump_repair_data (columns: prompt, test, entry, mbpp, problem_id).
# Usage (via go_repair.sh / accelerate launch):
#   python -m rl_training.train_repair_grpo --model <dir> --data <jsonl> --output-dir <ckpt> --steps 200
import argparse, json, os, sys, tempfile, subprocess, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def extract_code(text):
    m = re.findall(r"```(?:python)?\n(.*?)```", text or "", re.DOTALL)
    return m[0] if m else (text or "")

def run_tests(code, test, entry, mbpp, timeout=10):
    prog = code + "\n" + test + "\n" + ("" if mbpp else f"check({entry})\n")
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(prog); path = f.name
        r = subprocess.run(["python3", path], capture_output=True, timeout=timeout)
        ok = r.returncode == 0
    except Exception:
        ok = False
    finally:
        try: os.unlink(path)
        except Exception: pass
    return ok

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="repair_*.jsonl")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--max-completion-length", type=int, default=1024)
    ap.add_argument("--vllm-mode", default="colocate", choices=["colocate","server"])
    a = ap.parse_args()
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer
    from peft import LoraConfig

    rows = [json.loads(l) for l in open(a.data) if l.strip()]
    ds = Dataset.from_list([{"prompt": r["prompt"], "test": r["test"],
                             "entry": r.get("entry"), "mbpp": bool(r.get("mbpp"))} for r in rows])
    print(f"[repair-grpo] {len(ds)} repair prompts from {a.data}")

    def reward_pass(completions, test, entry, mbpp, **kw):
        # completions: list[str] (standard format). test/entry/mbpp: per-sample columns (list-aligned).
        out = []
        for c, t, e, m in zip(completions, test, entry, mbpp):
            code = extract_code(c if isinstance(c, str) else c[-1]["content"])
            out.append(1.0 if run_tests(code, t, e, m) else 0.0)
        return out

    cfg = GRPOConfig(
        output_dir=a.output_dir, learning_rate=a.lr, per_device_train_batch_size=a.num_generations,
        num_generations=a.num_generations, max_completion_length=a.max_completion_length,
        max_steps=a.steps, save_steps=max(a.steps//4,1), save_total_limit=2, logging_steps=1,
        gradient_accumulation_steps=1, bf16=True, beta=0.0, use_vllm=True, vllm_mode=a.vllm_mode,
        report_to="none", temperature=1.0, top_p=0.95,
    )
    peft_cfg = LoraConfig(r=a.lora_r, lora_alpha=a.lora_r*2, lora_dropout=0.05, task_type="CAUSAL_LM",
                          target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
    trainer = GRPOTrainer(model=a.model, reward_funcs=[reward_pass], args=cfg,
                          train_dataset=ds, peft_config=peft_cfg)
    trainer.train()
    trainer.save_model(a.output_dir)
    open(os.path.join(a.output_dir,"REPAIR_TRAIN_DONE"),"w").write("ok")
    print(">> repair-grpo training complete ->", a.output_dir)

if __name__ == "__main__":
    main()
