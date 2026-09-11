# BBH arm-C harvest: sample base on BBH TRAIN tasks, keep exact-match-correct traces -> SFT jsonl. Sharded.
import argparse, json, os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl_training.seq_recover import chat
from rl_training.bbh_util import load_bbh, bbh_match, BBH_PROMPT, BBH_TRAIN

def run(a):
    from vllm import LLM, SamplingParams
    items = []
    for t in BBH_TRAIN: items += load_bbh(t)
    if a.num_shards > 1: items = items[a.shard_index::a.num_shards]
    mp = a.model_path
    llm = LLM(model=mp, dtype="bfloat16", trust_remote_code=True, max_model_len=2048,
              gpu_memory_utilization=float(os.environ.get("EVAL_GPU_MEM", 0.85)), enforce_eager=True)
    sp = SamplingParams(n=a.k, temperature=1.0, top_p=0.95, max_tokens=768, stop=["<|im_end|>", "<|endoftext|>"])
    outs = llm.generate([chat(mp, BBH_PROMPT.replace("{q}", it["q"])) for it in items], sp)
    recs = []
    for it, o in zip(items, outs):
        p = chat(mp, BBH_PROMPT.replace("{q}", it["q"]))
        for s in o.outputs:
            if bbh_match(s.text, it["gold"]): recs.append({"prompt": p, "completion": s.text.strip()}); break
    Path(a.output_dir).mkdir(parents=True, exist_ok=True)
    fp = Path(a.output_dir)/f"bbhdata_{a.tag}.shard{a.shard_index}-of-{a.num_shards}.jsonl"
    open(fp, "w").write("\n".join(json.dumps(r) for r in recs))
    print(f"[bbh_gen {a.tag} s{a.shard_index}] {len(recs)}/{len(items)} verified-correct")

def merge(a):
    recs = []
    for s in range(a.num_shards):
        fp = Path(a.output_dir)/f"bbhdata_{a.tag}.shard{s}-of-{a.num_shards}.jsonl"
        if fp.exists(): recs += [l for l in open(fp) if l.strip()]
    open(Path(a.output_dir)/f"bbhdata_{a.tag}.jsonl", "w").write("".join(x if x.endswith("\n") else x+"\n" for x in recs))
    print(f"[bbh_gen {a.tag}] merged {len(recs)} traces")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="Qwen/Qwen2.5-7B"); ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--output-dir", default="/tmp/instance_storage/gu/sft_data"); ap.add_argument("--tag", default="bbhC")
    ap.add_argument("--shard-index", type=int, default=0); ap.add_argument("--num-shards", type=int, default=1); ap.add_argument("--merge", action="store_true")
    a = ap.parse_args(); merge(a) if a.merge else run(a)
if __name__ == "__main__": main()
