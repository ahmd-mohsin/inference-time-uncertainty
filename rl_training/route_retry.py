# Exp4 ROUTING: does failed-code conditioning collapse the retry's algorithm-routing toward the failed
# parent's mode, and does the certificate restore routing entropy? For each default-FAILED problem, take
# the parent's algorithm label (classify), then generate retries under 3 conditions — IID (q only),
# FULL (q + failed code + error), CERT (q + counterexample, code erased) — classify each retry's algorithm,
# and measure H(retry algo) + P(retry algo == parent algo). Prediction: FULL lowers entropy + raises
# parent-match (routing collapse toward the failed mode); CERT restores entropy toward IID. Inference-only.
import argparse, json, os, sys, math
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl_training.code_passk import load_bench, extract_code, run_tests, classify
from rl_training.seq_recover import base_task, chat
from rl_training.certify import make_certificate

def entropy(labels):
    if not labels: return 0.0
    from collections import Counter
    n = len(labels); c = Counter(labels)
    return -sum((k/n)*math.log(k/n) for k in c.values())

def prompt(it, cond, code, err):
    if cond == "iid":  return chat("qwen", base_task(it))
    if cond == "full": return chat("qwen", base_task(it)+f"\n\nA previous attempt failed:\n```python\n{code}\n```\nError: {err}\nGive a corrected solution.")
    c = make_certificate(code, it) or f"failed: {err}"
    return chat("qwen", base_task(it)+f"\n\nA previous attempt failed. {c}")

def run(a):
    from vllm import LLM, SamplingParams
    from rl_training.model_utils import merge_adapter_if_needed
    mp = merge_adapter_if_needed(a.model_path)
    items = load_bench(a.bench)
    if a.max_problems > 0: items = items[:a.max_problems]
    if a.num_shards > 1: items = items[a.shard_index::a.num_shards]
    llm = LLM(model=mp, dtype="bfloat16", trust_remote_code=True, tensor_parallel_size=1, max_model_len=4096,
              gpu_memory_utilization=float(os.environ.get("EVAL_GPU_MEM",0.85)), enable_prefix_caching=True, enforce_eager=True)
    def gen(prompts, n):
        return llm.generate(prompts, SamplingParams(n=n, temperature=1.0, top_p=0.95, max_tokens=1024, stop=["<|im_end|>","<|eot_id|>","<|endoftext|>"]))
    d0 = gen([chat(mp, base_task(it)) for it in items], 1)
    failed = []
    for it, o in zip(items, d0):
        code = extract_code(o.outputs[0].text); ok, err = run_tests(code, it, return_err=True)
        if not ok: failed.append({"it": it, "code": code, "err": err, "plabel": classify(code)})
    K = a.k_retry; per = []
    for cond in ("iid", "full", "cert"):
        outs = gen([prompt(f["it"], cond, f["code"], f["err"]) for f in failed], K)
        for f, o in zip(failed, outs):
            labs = [classify(extract_code(s.text)) for s in o.outputs]
            f.setdefault("cond", {})[cond] = {"entropy": entropy(labs), "parent_match": sum(1 for l in labs if l == f["plabel"])/max(len(labs),1)}
    for f in failed: per.append({"id": f["it"]["id"], "plabel": f["plabel"], "cond": f["cond"]})
    out = {"tag": a.tag, "model": mp, "bench": a.bench, "n": len(per), "per_problem": per}
    Path(a.output_dir).mkdir(parents=True, exist_ok=True)
    fp = Path(a.output_dir)/(f"route_{a.tag}.shard{a.shard_index}-of-{a.num_shards}.json" if a.num_shards>1 else f"route_{a.tag}.json")
    json.dump(out, open(fp,"w")); print(f"[{a.tag} shard {a.shard_index}] n={len(per)} -> {fp}")

def merge(a):
    import statistics as st
    per = []
    for s in range(a.num_shards):
        fp = Path(a.output_dir)/f"route_{a.tag}.shard{s}-of-{a.num_shards}.json"
        if fp.exists(): per += json.load(open(fp))["per_problem"]
    res = {}
    for cond in ("iid", "full", "cert"):
        H = [p["cond"][cond]["entropy"] for p in per if cond in p.get("cond",{})]
        PM = [p["cond"][cond]["parent_match"] for p in per if cond in p.get("cond",{})]
        res[cond] = {"mean_entropy": st.mean(H) if H else None, "mean_parent_match": st.mean(PM) if PM else None}
    out = {"tag": a.tag, "n": len(per), "routing": res, "per_problem": per}
    json.dump(out, open(Path(a.output_dir)/f"route_{a.tag}.json","w"))
    print(f"[{a.tag}] n={len(per)}  " + "  ".join(f"{c}: H={res[c]['mean_entropy']:.3f} pmatch={res[c]['mean_parent_match']:.3f}" for c in ("iid","full","cert")))
    print("  prediction: full LOWER H + HIGHER pmatch than iid (routing collapse to parent); cert restores toward iid")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path"); ap.add_argument("--bench", default="mbpp")
    ap.add_argument("--max-problems", type=int, default=-1); ap.add_argument("--k-retry", type=int, default=8)
    ap.add_argument("--output-dir", default="/tmp/instance_storage/gu/eval_out"); ap.add_argument("--tag", default="route")
    ap.add_argument("--shard-index", type=int, default=0); ap.add_argument("--num-shards", type=int, default=1); ap.add_argument("--merge", action="store_true")
    a = ap.parse_args(); merge(a) if a.merge else run(a)

if __name__ == "__main__":
    main()
