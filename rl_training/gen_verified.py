# ARM C data: sample base Qwen2.5-3B on GSM8K-train, keep VERIFIED-CORRECT completions -> SFT jsonl.
# Same experience as GRPO baseline (GSM8K-train, self-generated), different update (SFT vs RL). Sharded.
# Usage: python -m rl_training.gen_verified --model-path Qwen/Qwen2.5-3B --n 900 --k 4 --tag sft_s0 \
#   --shard-index S --num-shards 8 [--merge]
import argparse, json, os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl_training.seq_recover import chat
from rl_training.panel_eval import extract, match, PROMPT

def run(a):
    from vllm import LLM, SamplingParams
    from datasets import load_dataset
    d = load_dataset("openai/gsm8k", "main")["train"]
    items = [{"q": r["question"], "gold": r["answer"].split("####")[-1].strip().replace(",", "")} for r in d]
    if a.n > 0: items = items[:a.n]
    if a.num_shards > 1: items = items[a.shard_index::a.num_shards]
    mp = a.model_path
    llm = LLM(model=mp, dtype="bfloat16", trust_remote_code=True, tensor_parallel_size=1, max_model_len=2048,
              gpu_memory_utilization=float(os.environ.get("EVAL_GPU_MEM", 0.85)), enable_prefix_caching=True, enforce_eager=True)
    sp = SamplingParams(n=a.k, temperature=1.0, top_p=0.95, max_tokens=1024, stop=["<|im_end|>", "<|eot_id|>", "<|endoftext|>"])
    outs = llm.generate([chat(mp, PROMPT.replace("{q}", it["q"])) for it in items], sp)
    recs = []
    for it, o in zip(items, outs):
        prompt = chat(mp, PROMPT.replace("{q}", it["q"]))
        for s in o.outputs:
            if match(extract(s.text), it["gold"]):
                recs.append({"prompt": prompt, "completion": s.text.strip()})
                break  # one verified trajectory per problem (matched-experience: same as one GRPO success)
    Path(a.output_dir).mkdir(parents=True, exist_ok=True)
    fp = Path(a.output_dir)/f"sftdata_{a.tag}.shard{a.shard_index}-of-{a.num_shards}.jsonl"
    with open(fp, "w") as f:
        for r in recs: f.write(json.dumps(r) + "\n")
    print(f"[gen_verified {a.tag} s{a.shard_index}] {len(recs)}/{len(items)} verified-correct")

def merge(a):
    recs = []
    for s in range(a.num_shards):
        fp = Path(a.output_dir)/f"sftdata_{a.tag}.shard{s}-of-{a.num_shards}.jsonl"
        if fp.exists(): recs += [l for l in open(fp) if l.strip()]
    fp = Path(a.output_dir)/f"sftdata_{a.tag}.jsonl"
    with open(fp, "w") as f: f.writelines(recs)
    print(f"[gen_verified {a.tag}] merged {len(recs)} verified trajectories -> {fp}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="Qwen/Qwen2.5-3B"); ap.add_argument("--n", type=int, default=900); ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--output-dir", default="/tmp/instance_storage/gu/sft_data"); ap.add_argument("--tag", default="sft")
    ap.add_argument("--shard-index", type=int, default=0); ap.add_argument("--num-shards", type=int, default=1); ap.add_argument("--merge", action="store_true")
    a = ap.parse_args(); merge(a) if a.merge else run(a)

if __name__ == "__main__":
    main()
