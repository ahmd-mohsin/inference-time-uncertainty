# STEP-0 tightening (reviewer): is the RAW-EVID repair-reward gap already present at init (before any training)?
# For each base-model first-attempt FAILURE, build BOTH repair states from the SAME failure — EVID=(q,err) and
# RAW=(q,err,failed code) — generate K repairs under each, and score every reward variant AND full-correctness
# (all-pass). If the gap exists at step 0, the §1c "learnability" reading is undermined; report honestly.
# Also reports full-correctness (the real target) RAW vs EVID at init. 8-shard DP + merge (sums, mergeable).
# Usage: python -m rl_training.step0_reward --model-path <dir> --bench mbpp --tag step0_qc --k 8 --n-probe 4 \
#        --shard-index S --num-shards 8   [--merge]
import argparse, json, os, sys
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl_training.code_passk import load_bench, extract_code, run_tests
from rl_training.seq_recover import base_task, chat

REPAIR_EVID = ("\n\nA previous attempt at this problem FAILED with this error:\n{err}\n"
               "(the failed code is hidden on purpose). Diagnose the likely cause and write a correct, "
               "complete solution in a ```python block.")
REPAIR_RAW = ("\n\nYour previous attempt (below) FAILED with this error:\n{err}\n"
              "```python\n{code}\n```\nFix it and write a correct, complete solution in a ```python block.")
VARIANTS = ["binary", "fraction", "residual", "fraction_bonus", "cert_residual", "allpass"]

def reward_variants(vec, pfail, alpha=0.5, lam=1.0):
    if not vec: return None
    allp = all(vec); frac = sum(vec) / len(vec)
    have = bool(pfail) and len(pfail) == len(vec)
    if have:
        Fm = [i for i in range(len(vec)) if not pfail[i]]; Sm = [i for i in range(len(vec)) if pfail[i]]
        fixed = (sum(1 for i in Fm if vec[i]) / len(Fm)) if Fm else 0.0
        regr = (sum(1 for i in Sm if not vec[i]) / len(Sm)) if Sm else 0.0
        residual = fixed - lam * regr
    else:
        residual = frac
    return {"binary": 1.0 if allp else 0.0, "fraction": frac, "residual": residual,
            "fraction_bonus": frac + (alpha if allp else 0.0),
            "cert_residual": residual + (alpha if allp else 0.0), "allpass": 1.0 if allp else 0.0}

def run(a):
    from vllm import LLM, SamplingParams
    from rl_training.model_utils import merge_adapter_if_needed
    from rl_training.rewards import _passvec
    mp = merge_adapter_if_needed(a.model_path)
    items = load_bench(a.bench)
    if a.max_problems > 0: items = items[:a.max_problems]
    if a.num_shards > 1: items = items[a.shard_index::a.num_shards]
    llm = LLM(model=mp, dtype="bfloat16", trust_remote_code=True, tensor_parallel_size=1, max_model_len=4096,
              gpu_memory_utilization=float(os.environ.get("EVAL_GPU_MEM", 0.85)), enable_prefix_caching=True, enforce_eager=True)
    def gen(prompts, n, mt=1024):
        return llm.generate(prompts, SamplingParams(n=n, temperature=1.0, top_p=0.95, max_tokens=mt,
                            stop=["<|im_end|>", "<|eot_id|>", "<|endoftext|>"]))
    # elicit first-attempt failures (same problems feed BOTH contexts -> matched)
    probe = gen([chat(mp, base_task(it)) for it in items], a.n_probe)
    cases = []
    for it, o in zip(items, probe):
        for s in o.outputs:
            c = extract_code(s.text); ok, e = run_tests(c, it, return_err=True)
            if not ok:
                pf = _passvec(c, it["test"], it.get("entry"), bool(it.get("mbpp")))
                cases.append({"it": it, "err": e, "code": c, "pfail": pf}); break
    if len(cases) < 2:
        Path(a.output_dir).mkdir(parents=True, exist_ok=True)
        json.dump({"tag": a.tag, "n_cases": 0}, open(Path(a.output_dir)/f"step0_{a.tag}.shard{a.shard_index}-of-{a.num_shards}.json", "w"))
        print("too few failures"); return
    prompts, meta = [], []
    for ci, c in enumerate(cases):
        q = base_task(c["it"])
        prompts.append(chat(mp, q + REPAIR_EVID.format(err=c["err"]))); meta.append((ci, "EVID"))
        prompts.append(chat(mp, q + REPAIR_RAW.format(err=c["err"], code=c["code"][:1500]))); meta.append((ci, "RAW"))
    outs = gen(prompts, a.k)
    sums = {ctx: defaultdict(float) for ctx in ("EVID", "RAW")}; cnt = defaultdict(int)
    for (ci, ctx), o in zip(meta, outs):
        c = cases[ci]; it = c["it"]
        for s in o.outputs:
            vec = _passvec(extract_code(s.text), it["test"], it.get("entry"), bool(it.get("mbpp")))
            rv = reward_variants(vec, c["pfail"])
            if rv is None: continue
            for k, v in rv.items(): sums[ctx][k] += v
            cnt[ctx] += 1
    out = {"tag": a.tag, "model": mp.split("/")[-1], "bench": a.bench, "n_cases": len(cases),
           "sums": {ctx: dict(sums[ctx]) for ctx in sums}, "counts": {ctx: cnt[ctx] for ctx in cnt}}
    Path(a.output_dir).mkdir(parents=True, exist_ok=True)
    fp = Path(a.output_dir)/(f"step0_{a.tag}.shard{a.shard_index}-of-{a.num_shards}.json" if a.num_shards > 1 else f"step0_{a.tag}.json")
    json.dump(out, open(fp, "w")); print(f"[step0 {a.tag} s{a.shard_index}] cases={len(cases)}")

def merge(a):
    sums = {ctx: defaultdict(float) for ctx in ("EVID", "RAW")}; cnt = defaultdict(int); n = 0
    for s in range(a.num_shards):
        fp = Path(a.output_dir)/f"step0_{a.tag}.shard{s}-of-{a.num_shards}.json"
        if not fp.exists(): continue
        d = json.load(open(fp))
        if not d.get("n_cases"): continue
        n += d["n_cases"]
        for ctx in ("EVID", "RAW"):
            for k, v in d["sums"].get(ctx, {}).items(): sums[ctx][k] += v
            cnt[ctx] += d["counts"].get(ctx, 0)
    means = {ctx: {k: sums[ctx][k]/max(cnt[ctx], 1) for k in sums[ctx]} for ctx in sums}
    out = {"tag": a.tag, "n_cases": n, "counts": dict(cnt), "means": means,
           "gap_EVID_minus_RAW": {k: means["EVID"].get(k, 0) - means["RAW"].get(k, 0) for k in VARIANTS}}
    json.dump(out, open(Path(a.output_dir)/f"step0_{a.tag}.json", "w"))
    print(f"[step0 {a.tag}] n_cases={n}  (K repairs per context)")
    print(f"  {'variant':14s} {'EVID':>8s} {'RAW':>8s} {'EVID-RAW':>10s}")
    for k in VARIANTS:
        e = means["EVID"].get(k, 0); r = means["RAW"].get(k, 0)
        print(f"  {k:14s} {e:8.3f} {r:8.3f} {e-r:+10.3f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="Qwen/Qwen2.5-Coder-7B-Instruct"); ap.add_argument("--bench", default="mbpp")
    ap.add_argument("--n-probe", type=int, default=4); ap.add_argument("--k", type=int, default=8); ap.add_argument("--max-problems", type=int, default=-1)
    ap.add_argument("--output-dir", default="/tmp/instance_storage/gu/eval_out"); ap.add_argument("--tag", default="step0")
    ap.add_argument("--shard-index", type=int, default=0); ap.add_argument("--num-shards", type=int, default=1); ap.add_argument("--merge", action="store_true")
    a = ap.parse_args(); merge(a) if a.merge else run(a)

if __name__ == "__main__":
    main()
