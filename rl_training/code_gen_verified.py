# CODE-DOMAIN arm-C harvest: sample base on MBPP-train, execute tests, keep PASSING completions -> SFT jsonl.
# Generalizes the "SFT-on-verified" experiment beyond math. OOD eval = HumanEval (code_passk --bench humaneval).
# Usage: python -m rl_training.code_gen_verified --model-path <hf|adapter> --n 400 --k 4 --tag codeC --shard-index S --num-shards N [--merge]
import argparse, json, os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl_training.seq_recover import chat

PROMPT = ("You are an expert Python programmer. Write a correct, complete solution to the problem below "
          "in a single ```python code block.\n\nProblem: {q}\n")

def run(a):
    from vllm import LLM, SamplingParams
    from rl_training.code_passk import load_bench
    from rl_training.rewards import _passvec, _extract_code
    items = load_bench("mbpp_train")          # disjoint train split (no HumanEval contamination)
    if a.n > 0: items = items[:a.n]
    if a.num_shards > 1: items = items[a.shard_index::a.num_shards]
    mp = a.model_path
    llm = LLM(model=mp, dtype="bfloat16", trust_remote_code=True, tensor_parallel_size=1, max_model_len=2048,
              gpu_memory_utilization=float(os.environ.get("EVAL_GPU_MEM", 0.85)), enable_prefix_caching=True, enforce_eager=True)
    sp = SamplingParams(n=a.k, temperature=1.0, top_p=0.95, max_tokens=1024, stop=["<|im_end|>", "<|eot_id|>", "<|endoftext|>"])
    outs = llm.generate([chat(mp, PROMPT.replace("{q}", it["prompt"])) for it in items], sp)
    recs = []
    for it, o in zip(items, outs):
        prompt = chat(mp, PROMPT.replace("{q}", it["prompt"]))
        for s in o.outputs:
            code = _extract_code(s.text)
            pv = _passvec(code, it["test"], it.get("entry"), bool(it.get("mbpp")))
            if pv and all(pv):                # ALL tests pass = verified-correct
                recs.append({"prompt": prompt, "completion": s.text.strip()})
                break                         # one verified trace/problem (matched-experience)
    Path(a.output_dir).mkdir(parents=True, exist_ok=True)
    fp = Path(a.output_dir)/f"codedata_{a.tag}.shard{a.shard_index}-of-{a.num_shards}.jsonl"
    with open(fp, "w") as f:
        for r in recs: f.write(json.dumps(r) + "\n")
    print(f"[code_gen_verified {a.tag} s{a.shard_index}] {len(recs)}/{len(items)} verified-passing")

def merge(a):
    recs = []
    for s in range(a.num_shards):
        fp = Path(a.output_dir)/f"codedata_{a.tag}.shard{s}-of-{a.num_shards}.jsonl"
        if fp.exists(): recs += [l for l in open(fp) if l.strip()]
    fp = Path(a.output_dir)/f"codedata_{a.tag}.jsonl"
    with open(fp, "w") as f: f.writelines(recs)
    print(f"[code_gen_verified {a.tag}] merged {len(recs)} verified code traces -> {fp}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="Qwen/Qwen2.5-7B"); ap.add_argument("--n", type=int, default=400); ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--output-dir", default="/tmp/instance_storage/gu/sft_data"); ap.add_argument("--tag", default="codeC")
    ap.add_argument("--shard-index", type=int, default=0); ap.add_argument("--num-shards", type=int, default=1); ap.add_argument("--merge", action="store_true")
    a = ap.parse_args(); merge(a) if a.merge else run(a)

if __name__ == "__main__":
    main()
