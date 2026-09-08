# CEGIS baseline (reviewer-critical: counterexamples are prior art via CEGIS/MENTOR-style repair). On default-FAILED
# problems, matched-budget K, compare FOUR feedback conditions and test whether COUNTEREXAMPLE-ONLY (proposal erased)
# beats FAILED-CODE+COUNTEREXAMPLE (CEGIS-style, proposal retained → anchoring):
#   iid            : retry from original prompt (no feedback)
#   code_plus_cex  : failed code shown + counterexample (classic CEGIS/local-repair)
#   cex_only       : counterexample only, code HIDDEN (Forget-to-Repair)
#   sketch_plus_cex: bug-free signature/skeleton (no failed body) + counterexample
# Prediction: cex_only >= code_plus_cex (removing the proposal avoids anchoring), both use the same evidence.
import argparse, json, os, sys, re
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl_training.code_passk import load_bench, extract_code, run_tests
from rl_training.seq_recover import base_task, chat
from rl_training.certify import make_certificate

def signature_of(code):
    """First `def ...:` line (bug-free skeleton = structure without the failed body)."""
    for line in (code or "").splitlines():
        if re.match(r"\s*def\s+\w+\s*\(", line):
            return line.strip().rstrip(":") + ":\n    ..."
    return None

def run(a):
    from vllm import LLM, SamplingParams
    from rl_training.model_utils import merge_adapter_if_needed
    mp = merge_adapter_if_needed(a.model_path)
    items = load_bench(a.bench)
    if a.max_problems > 0: items = items[:a.max_problems]
    if a.num_shards > 1: items = items[a.shard_index::a.num_shards]
    llm = LLM(model=mp, dtype="bfloat16", trust_remote_code=True, tensor_parallel_size=1, max_model_len=4096,
              gpu_memory_utilization=float(os.environ.get("EVAL_GPU_MEM",0.85)), enable_prefix_caching=True, enforce_eager=True)
    def gen(prompts, n): return llm.generate(prompts, SamplingParams(n=n, temperature=1.0, top_p=0.95, max_tokens=1024, stop=["<|im_end|>","<|eot_id|>","<|endoftext|>"]))
    d0 = gen([chat(mp, base_task(it)) for it in items], 1)
    failed = []
    for it, o in zip(items, d0):
        code = extract_code(o.outputs[0].text); ok, err = run_tests(code, it, return_err=True)
        if not ok:
            cert = make_certificate(code, it)
            if cert: failed.append({"it": it, "code": code, "err": err, "cert": cert})
    if len(failed) < 3:
        json.dump({"tag":a.tag,"n":0,"per_arm":{}}, open(Path(a.output_dir)/f"cegis_{a.tag}.shard{a.shard_index}-of-{a.num_shards}.json","w")); print("too few"); return
    K = a.k
    def build(arm, f):
        task = base_task(f["it"]); cert = f["cert"]
        if arm == "iid": return chat(mp, task)
        if arm == "code_plus_cex":
            return chat(mp, task + f"\n\nYour previous attempt:\n```python\n{f['code'][:1200]}\n```\n{cert}")
        if arm == "cex_only":
            return chat(mp, task + "\n\n" + cert)
        if arm == "sketch_plus_cex":
            sk = signature_of(f["code"]) or ""
            return chat(mp, task + (f"\n\nUse this exact signature:\n```python\n{sk}\n```\n" if sk else "\n\n") + cert)
    ARMS = ["iid","code_plus_cex","cex_only","sketch_plus_cex"]
    hits = {arm:[0,0] for arm in ARMS}
    for arm in ARMS:
        outs = gen([build(arm, f) for f in failed], K)
        for f, o in zip(failed, outs):
            rec = any(run_tests(extract_code(s.text), f["it"]) for s in o.outputs)
            hits[arm][0] += int(rec); hits[arm][1] += 1
    out = {"tag":a.tag,"model":mp,"bench":a.bench,"n_failed":len(failed),"K":K,
           "per_arm":{arm:{"recovered":hits[arm][0],"total":hits[arm][1]} for arm in ARMS}}
    Path(a.output_dir).mkdir(parents=True, exist_ok=True)
    fp = Path(a.output_dir)/(f"cegis_{a.tag}.shard{a.shard_index}-of-{a.num_shards}.json" if a.num_shards>1 else f"cegis_{a.tag}.json")
    json.dump(out, open(fp,"w")); print(f"[{a.tag} s{a.shard_index}] failed={len(failed)} -> {fp}")

def merge(a):
    agg = {};
    for s in range(a.num_shards):
        fp = Path(a.output_dir)/f"cegis_{a.tag}.shard{s}-of-{a.num_shards}.json"
        if not fp.exists(): continue
        d = json.load(open(fp))
        for arm, r in d.get("per_arm", {}).items():
            agg.setdefault(arm,[0,0]); agg[arm][0]+=r["recovered"]; agg[arm][1]+=r["total"]
    res = {arm:{"rate":agg[arm][0]/max(agg[arm][1],1),"n":agg[arm][1]} for arm in agg}
    iid = res.get("iid",{}).get("rate",0)
    out={"tag":a.tag,"iid":iid,"per_arm":res,"delta_vs_iid":{a2:res[a2]["rate"]-iid for a2 in res}}
    json.dump(out, open(Path(a.output_dir)/f"cegis_{a.tag}.json","w"))
    print(f"[{a.tag}] n={res.get('iid',{}).get('n',0)}")
    for arm in ["iid","code_plus_cex","cex_only","sketch_plus_cex"]:
        if arm in res: print(f"  {arm:<16} rate={res[arm]['rate']:.3f}  Δiid={res[arm]['rate']-iid:+.3f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path"); ap.add_argument("--bench", default="mbpp")
    ap.add_argument("--max-problems", type=int, default=-1); ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--output-dir", default="/tmp/instance_storage/gu/eval_out"); ap.add_argument("--tag", default="cegis")
    ap.add_argument("--shard-index", type=int, default=0); ap.add_argument("--num-shards", type=int, default=1); ap.add_argument("--merge", action="store_true")
    a = ap.parse_args(); merge(a) if a.merge else run(a)

if __name__ == "__main__":
    main()
