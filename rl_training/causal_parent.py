# Exp3 CAUSAL: does the failed PROPOSAL causally shape the retry's hypothesis space? For each problem,
# elicit TWO distinct failed parents (different algorithm signatures) A,B. Repair from each under two
# conditions: FULL (show parent code+error → should anchor) and CERT (show only the counterexample, code
# erased → should NOT anchor). Measure parent-recurrence: does a repair resemble ITS parent more than the
# other parent? recurrence = sim(repair_from_X, X) − sim(repair_from_X, other). Prediction:
# full: recurrence ≫ 0 (proposal causally anchors); cert: recurrence ≈ 0 (proposal erased).
import argparse, json, os, sys, re
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl_training.code_passk import load_bench, extract_code, run_tests, classify
from rl_training.seq_recover import base_task, chat
from rl_training.certify import make_certificate

def toks(code):
    return set(re.findall(r"[A-Za-z_]\w+", code or ""))
def jac(a, b):
    A, B = toks(a), toks(b)
    return len(A & B) / max(len(A | B), 1)
def sim(repair, parent):
    """blend: algorithm-label match (classify) + token Jaccard."""
    return 0.5*(1.0 if classify(repair) == classify(parent) else 0.0) + 0.5*jac(repair, parent)

def repair_prompt(it, parent_code, err, cond):
    if cond == "full":
        fb = (f"A previous attempt failed. Here is that attempt:\n```python\n{parent_code}\n```\nError: {err}\n"
              f"Fix it and give a corrected full solution in a ```python block.")
    else:  # cert
        c = make_certificate(parent_code, it) or f"Previous attempt failed: {err}"
        fb = f"A previous attempt failed. {c}"
    return chat("qwen", base_task(it) + "\n\n" + fb)

def run(a):
    from vllm import LLM, SamplingParams
    from rl_training.model_utils import merge_adapter_if_needed
    mp = merge_adapter_if_needed(a.model_path)
    items = load_bench(a.bench)
    if a.max_problems > 0: items = items[:a.max_problems]
    if a.num_shards > 1: items = items[a.shard_index::a.num_shards]
    llm = LLM(model=mp, dtype="bfloat16", trust_remote_code=True, tensor_parallel_size=1, max_model_len=4096,
              gpu_memory_utilization=float(os.environ.get("EVAL_GPU_MEM",0.85)), enable_prefix_caching=True, enforce_eager=True)
    def gen(prompts, n=1, temp=1.0):
        sp = SamplingParams(n=n, temperature=temp, top_p=0.95, max_tokens=1024, stop=["<|im_end|>","<|eot_id|>","<|endoftext|>"])
        return llm.generate(prompts, sp)
    # elicit failed parents
    pool = gen([chat(mp, base_task(it)) for it in items], 8)
    per = []
    for it, o in zip(items, pool):
        fails = []
        for s in o.outputs:
            code = extract_code(s.text); ok, err = run_tests(code, it, return_err=True)
            if not ok: fails.append((code, err, classify(code)))
        if len(fails) < 2: continue
        A = fails[0]
        B = next((f for f in fails[1:] if f[2] != A[2]), None) or (max(fails[1:], key=lambda f: 1-jac(f[0], A[0])))
        per.append({"id": it["id"], "it": it, "pA": A, "pB": B})
    # repairs from each parent under each condition
    R = a.n_repair
    prompts, meta = [], []
    for p in per:
        for cond in ("full", "cert"):
            for src, par in (("A", p["pA"]), ("B", p["pB"])):
                prompts.append(repair_prompt(p["it"], par[0], par[1], cond)); meta.append((p["id"], cond, src))
    outs = gen(prompts, R)
    byk = {}
    for (pid, cond, src), o in zip(meta, outs):
        byk[(pid, cond, src)] = [extract_code(s.text) for s in o.outputs]
    rec = []
    for p in per:
        row = {"id": p["id"], "labA": p["pA"][2], "labB": p["pB"][2]}
        for cond in ("full", "cert"):
            vals = []
            for src, par, other in (("A", p["pA"], p["pB"]), ("B", p["pB"], p["pA"])):
                for rp in byk.get((p["id"], cond, src), []):
                    vals.append(sim(rp, par[0]) - sim(rp, other[0]))   # recurrence toward own parent
            row[f"recurrence_{cond}"] = sum(vals)/max(len(vals),1)
        rec.append(row)
    out = {"tag": a.tag, "model": mp, "bench": a.bench, "n_problems": len(per), "per_problem": rec}
    Path(a.output_dir).mkdir(parents=True, exist_ok=True)
    fp = Path(a.output_dir)/(f"causal_{a.tag}.shard{a.shard_index}-of-{a.num_shards}.json" if a.num_shards>1 else f"causal_{a.tag}.json")
    json.dump(out, open(fp,"w")); print(f"[{a.tag} shard {a.shard_index}] n={len(per)} -> {fp}")

def merge(a):
    import statistics as st
    per = []
    for s in range(a.num_shards):
        fp = Path(a.output_dir)/f"causal_{a.tag}.shard{s}-of-{a.num_shards}.json"
        if fp.exists(): per += json.load(open(fp))["per_problem"]
    full = [p["recurrence_full"] for p in per if "recurrence_full" in p]
    cert = [p["recurrence_cert"] for p in per if "recurrence_cert" in p]
    out = {"tag": a.tag, "n": len(per),
           "recurrence_full_mean": st.mean(full) if full else None,
           "recurrence_cert_mean": st.mean(cert) if cert else None, "per_problem": per}
    json.dump(out, open(Path(a.output_dir)/f"causal_{a.tag}.json","w"))
    print(f"[{a.tag}] n={len(per)} recurrence FULL={out['recurrence_full_mean']} CERT={out['recurrence_cert_mean']} "
          f"(prediction: full≫0 anchors to parent, cert≈0)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path"); ap.add_argument("--bench", default="mbpp")
    ap.add_argument("--max-problems", type=int, default=-1); ap.add_argument("--n-repair", type=int, default=4)
    ap.add_argument("--output-dir", default="/tmp/instance_storage/gu/eval_out"); ap.add_argument("--tag", default="causal")
    ap.add_argument("--shard-index", type=int, default=0); ap.add_argument("--num-shards", type=int, default=1); ap.add_argument("--merge", action="store_true")
    a = ap.parse_args(); merge(a) if a.merge else run(a)

if __name__ == "__main__":
    main()
