#!/usr/bin/env python3
# E2 GPU sampler (SHARDED) — generate N base solutions/problem WITH TEXT, verify, keep base-correct.
# One shard per GPU. Writes per-shard jsonl incrementally (death-tolerant). Merge shards after.
# Usage (per shard):
#   CUDA_VISIBLE_DEVICES=i python sample_base_solutions.py --model <base> --n 256 \
#     --dataset olympiad_bench --difficulty-json <diff> --subset hard \
#     --num-shards 8 --shard-index i --out /path/bank_shardI.jsonl
import argparse, json, os, sys
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
sys.path.insert(0, os.getcwd())
from vllm import LLM, SamplingParams
from src.data.dataset import get_inference_dataset, format_prompt
from rl_training.safe_match import safe_is_correct

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--n", type=int, default=256)
ap.add_argument("--dataset", default="olympiad_bench")
ap.add_argument("--difficulty-json", default="")
ap.add_argument("--subset", default="hard")
ap.add_argument("--num-shards", type=int, default=1)
ap.add_argument("--shard-index", type=int, default=0)
ap.add_argument("--out", required=True)
ap.add_argument("--max-new-tokens", type=int, default=4096)
ap.add_argument("--temperature", type=float, default=1.0)
ap.add_argument("--max-keep", type=int, default=64, help="cap base-correct witnesses saved/problem")
a = ap.parse_args()

probs = get_inference_dataset({"dataset": {"name": a.dataset, "split": "test",
                                           "n_problems": -1, "seed": 42}})
if a.difficulty_json and os.path.exists(a.difficulty_json):
    diff = json.load(open(a.difficulty_json))
    keep = {d["problem_id"] for d in diff.get("per_problem", []) if d.get("label") == a.subset}
    probs = [p for p in probs if int(p["problem_id"]) in keep]
# shard by problem index
probs = [p for i, p in enumerate(probs) if i % a.num_shards == a.shard_index]
print(f"[shard {a.shard_index}/{a.num_shards}] {len(probs)} problems, n={a.n}", flush=True)

llm = LLM(model=a.model, tensor_parallel_size=1, max_model_len=4096, gpu_memory_utilization=0.9)
sp = SamplingParams(n=a.n, temperature=a.temperature, max_tokens=a.max_new_tokens)

kept = 0
with open(a.out, "w") as w:
    for p in probs:
        prompt = format_prompt(p, a.model)
        gold = str(p.get("gold_answer", ""))
        outs = llm.generate([prompt], sp)[0].outputs
        n_saved = 0
        for o in outs:
            if n_saved >= a.max_keep:
                break
            if safe_is_correct(o.text, gold)[0]:
                w.write(json.dumps({"problem_id": int(p["problem_id"]), "prompt": prompt,
                                    "completion": o.text, "gold": gold}) + "\n")
                w.flush()
                n_saved += 1
                kept += 1
        print(f"  q{p['problem_id']}: kept {n_saved}/{a.n}", flush=True)
print(f"[shard {a.shard_index}] DONE, kept {kept} base-correct witnesses -> {a.out}", flush=True)
