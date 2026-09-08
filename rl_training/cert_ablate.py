# Certificate ABLATION: two reviewer-critical controls in one run, on default-FAILED problems (code always
# hidden; single-shot K retries per variant; recovery = any pass).
#   INFO-DENSITY ladder: iid, C0 (bare notice), C1 (failed-tests), C2 (failing input), C3 (input+got),
#                        C4 (input+got+expected = full certificate), C5 (multi-counterexample).
#   CORRUPTION controls: wronginput (bogus input, real got/expected), wrongexpected (real input/got,
#                        corrupted expected), shuffled (an entirely different problem's certificate).
# Prediction: recovery rises with info level and peaks at C4/C5; corrupted certs DROP toward/below iid —
# proving the model uses the EVIDENCE CONTENT, not just a fresh-restart prompt.
import argparse, json, os, sys, random, re
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl_training.code_passk import load_bench, extract_code, run_tests
from rl_training.seq_recover import base_task, chat
from rl_training.certify import extract_counterexample, cert_string, _assert_pairs, _time_limit, _CertTimeout

def multi_cert(code, item):
    """Up to 3 distinct failing counterexamples from one failed program (for C5)."""
    ns = {}
    try:
        with _time_limit(5): exec(code, ns)
    except (Exception, _CertTimeout): return cert_string(extract_counterexample(code, item), 4)
    entry = item.get("entry")
    if entry and entry in ns: ns.setdefault("candidate", ns[entry])
    lines = []
    for lhs, rhs in _assert_pairs(item.get("test", "")):
        try:
            with _time_limit(5): got = eval(lhs, ns); exp = eval(rhs, ns)
        except (Exception, _CertTimeout): continue
        if got != exp:
            lines.append(f"`{lhs.strip()[:120]}` → got `{repr(got)[:80]}`, correct `{rhs.strip()[:60]}`")
        if len(lines) >= 3: break
    if not lines: return cert_string(extract_counterexample(code, item), 4)
    return "Concrete counterexamples:\n- " + "\n- ".join(lines) + "\nWrite a fresh, correct solution in a ```python block."

def corrupt_expected(exp):
    if exp is None: return "0"
    e = exp.strip()
    if re.fullmatch(r"-?\d+", e): return str(int(e) + 7)
    if e in ("True","False"): return "False" if e=="True" else "True"
    return e + " + 1"  # deliberately wrong

def build_variants(ce, ce_other, code, item):
    """dict variant -> feedback string (code hidden). ce=this problem's counterexample; ce_other=another problem's."""
    v = {}
    for lvl in range(5): v[f"c{lvl}"] = cert_string(ce, lvl)
    v["c5_multi"] = multi_cert(code, item)
    # corruptions (all level-4 shaped)
    if ce:
        v["wronginput"] = (f"Concrete counterexample: the call `{(ce_other or ce)['input']}` returned `{ce['got']}` "
                           f"but the correct answer is `{ce['expected']}`. Write a fresh, correct solution.")
        v["wrongexpected"] = (f"Concrete counterexample: the call `{ce['input']}` returned `{ce['got']}` but the "
                              f"correct answer is `{corrupt_expected(ce['expected'])}`. Write a fresh, correct solution.")
    if ce_other:
        v["shuffled"] = cert_string(ce_other, 4)
    return {k: val for k, val in v.items() if val}

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
            ce = extract_counterexample(code, it)
            if ce: failed.append({"it": it, "code": code, "ce": ce})
    if len(failed) < 3:
        json.dump({"tag":a.tag,"n":0,"per_variant":{}}, open(Path(a.output_dir)/f"ablate_{a.tag}.shard{a.shard_index}-of-{a.num_shards}.json","w")); print("too few"); return
    ces = [f["ce"] for f in failed]
    variants = {}
    for i, f in enumerate(failed):
        f["variants"] = build_variants(f["ce"], ces[(i+ len(ces)//2) % len(ces)], f["code"], f["it"])
    allv = sorted(set().union(*[set(f["variants"]) for f in failed])) + ["iid"]
    K = a.k
    hits = {v: [0,0] for v in allv}  # [recovered, total]
    for v in allv:
        prompts, idx = [], []
        for i, f in enumerate(failed):
            if v == "iid": prompts.append(chat(mp, base_task(f["it"]))); idx.append(i)
            elif v in f["variants"]: prompts.append(chat(mp, base_task(f["it"]) + "\n\n" + f["variants"][v])); idx.append(i)
        if not prompts: continue
        outs = gen(prompts, K)
        for i, o in zip(idx, outs):
            rec = any(run_tests(extract_code(s.text), failed[i]["it"]) for s in o.outputs)
            hits[v][0] += int(rec); hits[v][1] += 1
    out = {"tag": a.tag, "model": mp, "bench": a.bench, "n_failed": len(failed), "K": K,
           "per_variant": {v: {"recovered": hits[v][0], "total": hits[v][1], "rate": hits[v][0]/max(hits[v][1],1)} for v in allv}}
    Path(a.output_dir).mkdir(parents=True, exist_ok=True)
    fp = Path(a.output_dir)/(f"ablate_{a.tag}.shard{a.shard_index}-of-{a.num_shards}.json" if a.num_shards>1 else f"ablate_{a.tag}.json")
    json.dump(out, open(fp,"w")); print(f"[{a.tag} shard {a.shard_index}] failed={len(failed)} -> {fp}")

def merge(a):
    agg = {}
    for s in range(a.num_shards):
        fp = Path(a.output_dir)/f"ablate_{a.tag}.shard{s}-of-{a.num_shards}.json"
        if not fp.exists(): continue
        d = json.load(open(fp))
        for v, r in d.get("per_variant", {}).items():
            agg.setdefault(v, [0,0]); agg[v][0]+=r["recovered"]; agg[v][1]+=r["total"]
    res = {v: {"rate": agg[v][0]/max(agg[v][1],1), "n": agg[v][1]} for v in agg}
    iid = res.get("iid",{}).get("rate",0)
    out = {"tag": a.tag, "iid_rate": iid, "per_variant": res,
           "delta_vs_iid": {v: res[v]["rate"]-iid for v in res}}
    json.dump(out, open(Path(a.output_dir)/f"ablate_{a.tag}.json","w"))
    print(f"[{a.tag}] iid={iid:.3f}")
    for v in ["c0","c1","c2","c3","c4","c5_multi","wronginput","wrongexpected","shuffled"]:
        if v in res: print(f"  {v:<14} rate={res[v]['rate']:.3f}  Δiid={res[v]['rate']-iid:+.3f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path"); ap.add_argument("--bench", default="mbpp")
    ap.add_argument("--max-problems", type=int, default=-1); ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--output-dir", default="/tmp/instance_storage/gu/eval_out"); ap.add_argument("--tag", default="ablate")
    ap.add_argument("--shard-index", type=int, default=0); ap.add_argument("--num-shards", type=int, default=1); ap.add_argument("--merge", action="store_true")
    a = ap.parse_args(); merge(a) if a.merge else run(a)

if __name__ == "__main__":
    main()
