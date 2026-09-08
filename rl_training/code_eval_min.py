# Minimal robust code pass@1/solve@k eval (sidesteps code_passk's stratified machinery which hung).
# Usage: python -m rl_training.code_eval_min --model-path <hf|adapter> --bench humaneval --tag X --k 8
import argparse, json, os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl_training.seq_recover import chat
PROMPT = ("You are an expert Python programmer. Write a correct, complete solution in a single ```python block.\n\nProblem: {q}\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True); ap.add_argument("--bench", default="humaneval")
    ap.add_argument("--tag", default="ce"); ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--output-dir", default="/tmp/instance_storage/gu/eval_out")
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    from rl_training.code_passk import load_bench
    from rl_training.rewards import _passvec, _extract_code
    from rl_training.model_utils import merge_adapter_if_needed
    mp = merge_adapter_if_needed(a.model_path)
    items = load_bench(a.bench)
    llm = LLM(model=mp, dtype="bfloat16", trust_remote_code=True, max_model_len=2048,
              gpu_memory_utilization=float(os.environ.get("EVAL_GPU_MEM", 0.85)), enforce_eager=True)
    sp = SamplingParams(n=a.k, temperature=0.8, top_p=0.95, max_tokens=768, stop=["<|im_end|>", "<|endoftext|>"])
    outs = llm.generate([chat(mp, PROMPT.replace("{q}", it["prompt"])) for it in items], sp)
    p1_sum = 0.0; solved_any = 0
    for it, o in zip(items, outs):
        passes = 0
        for s in o.outputs:
            pv = _passvec(_extract_code(s.text), it["test"], it.get("entry"), bool(it.get("mbpp")))
            if pv and all(pv): passes += 1
        p1_sum += passes / max(len(o.outputs), 1)
        solved_any += int(passes > 0)
    n = len(items)
    out = {"tag": a.tag, "bench": a.bench, "model": mp.split("/")[-1], "n": n,
           "pass@1": p1_sum / max(n, 1), "solve@k": solved_any / max(n, 1)}
    Path(a.output_dir).mkdir(parents=True, exist_ok=True)
    json.dump(out, open(Path(a.output_dir)/f"ce_{a.tag}.json", "w"))
    print(f"[code_eval_min {a.tag} {a.bench}] pass@1={out['pass@1']:.4f} solve@{a.k}={out['solve@k']:.4f} n={n}")

if __name__ == "__main__":
    main()
