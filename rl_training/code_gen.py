# CODE eval PHASE 1 (gen only, vLLM) — generate k completions/problem, save jsonl with test metadata.
# Scoring is done SEPARATELY by code_score.py (no vLLM alive) to avoid the vLLM<->test-subprocess fork deadlock.
# Usage: python -m rl_training.code_gen --model-path <hf|adapter> --bench humaneval --tag X --k 8
import argparse, json, os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl_training.seq_recover import chat
PROMPT = ("You are an expert Python programmer. Write a correct, complete solution in a single ```python block.\n\nProblem: {q}\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True); ap.add_argument("--bench", default="humaneval")
    ap.add_argument("--tag", default="cg"); ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--output-dir", default="/tmp/instance_storage/gu/code_out")
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    from rl_training.code_passk import load_bench
    from rl_training.model_utils import merge_adapter_if_needed
    mp = merge_adapter_if_needed(a.model_path)
    items = load_bench(a.bench)
    llm = LLM(model=mp, dtype="bfloat16", trust_remote_code=True, max_model_len=2048,
              gpu_memory_utilization=float(os.environ.get("EVAL_GPU_MEM", 0.85)), enforce_eager=True)
    sp = SamplingParams(n=a.k, temperature=0.8, top_p=0.95, max_tokens=768, stop=["<|im_end|>", "<|endoftext|>"])
    outs = llm.generate([chat(mp, PROMPT.replace("{q}", it["prompt"])) for it in items], sp)
    Path(a.output_dir).mkdir(parents=True, exist_ok=True)
    fp = Path(a.output_dir)/f"cg_{a.tag}.jsonl"
    with open(fp, "w") as f:
        for it, o in zip(items, outs):
            f.write(json.dumps({"test": it["test"], "entry": it.get("entry"), "mbpp": bool(it.get("mbpp")),
                                "completions": [s.text for s in o.outputs]}) + "\n")
    print(f"[code_gen {a.tag} {a.bench}] wrote {len(items)} problems x {a.k} -> {fp}")

if __name__ == "__main__":
    main()
