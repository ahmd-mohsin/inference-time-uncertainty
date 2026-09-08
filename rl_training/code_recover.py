# Failure-conditioned RECOVERY (monoculture direction #3). Isolates diversity's RECOVERY value from
# raw extra-budget: for each code problem, a DEFAULT attempt (n_def iid); then TWO matched-budget
# recovery arms of n_rec samples each: (A) iid-retry (same default prompt) vs (B) strategy-SWITCH
# (forced diverse strategies). On problems the default FAILS, does switch recover more than iid-retry
# at EQUAL budget? If yes, the hidden repertoire's value is recovery, not exploration.
#
# Reuses code_passk machinery (exec-verify). 8-GPU DP + merge.
# Usage: python -m rl_training.code_recover --model-path <dir> --bench humaneval --tag rec_qc_he \
#   --shard-index S --num-shards 8 --n-def 8 --n-rec 12
import argparse, json, os, sys, random
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl_training.code_passk import (load_bench, extract_code, run_tests, classify, build_prompt,
                                     STRATEGIES, STRAT_NAMES)


def run(a):
    from vllm import LLM, SamplingParams
    from rl_training.model_utils import merge_adapter_if_needed
    mp = merge_adapter_if_needed(a.model_path)
    items = load_bench(a.bench)
    if a.max_problems > 0: items = items[:a.max_problems]
    if a.num_shards > 1: items = items[a.shard_index::a.num_shards]
    _gm = float(os.environ.get("EVAL_GPU_MEM", 0.85)); _eager = os.environ.get("EVAL_ENFORCE_EAGER","1")=="1"
    llm = LLM(model=mp, dtype="bfloat16", trust_remote_code=True, tensor_parallel_size=1, max_model_len=4096,
              gpu_memory_utilization=_gm, enable_prefix_caching=True, enforce_eager=_eager)
    def gen(prompts, n):
        sp = SamplingParams(n=n, temperature=1.0, top_p=0.95, max_tokens=1024, stop=["<|im_end|>","<|eot_id|>","<|endoftext|>"])
        return llm.generate(prompts, sp)
    # DEFAULT (n_def) + IID-RETRY (n_rec, same prompt) = one call at n_def+n_rec on the plain prompt
    plain_out = gen([build_prompt(it, mp) for it in items], a.n_def + a.n_rec)
    # SWITCH-RETRY: n_rec samples spread across strategies (n_rec//len + remainder), one call per strategy
    per_strat = max(1, a.n_rec // len(STRAT_NAMES) + 1)
    sw_prompts, sw_idx = [], []
    for pi, it in enumerate(items):
        for m in STRAT_NAMES:
            sw_prompts.append(build_prompt(it, mp, m)); sw_idx.append((pi, m))
    sw_out = gen(sw_prompts, per_strat)
    per = []
    for it, o in zip(items, plain_out):
        outs = list(o.outputs)
        deff = []; fail_code = ""; fail_err = ""
        for s in outs[:a.n_def]:
            code = extract_code(s.text); ok, err = run_tests(code, it, return_err=True)
            deff.append(ok)
            if not ok and not fail_code:   # capture first failed default as router feature
                fail_code, fail_err = code[:2000], err
        iidr = [run_tests(extract_code(s.text), it) for s in outs[a.n_def:a.n_def+a.n_rec]]
        per.append({"id": it["id"], "default": deff, "iid_retry": iidr, "switch": [],
                    "question": it.get("prompt", "")[:1500], "fail_code": fail_code, "fail_err": fail_err})
    for (pi, m), o in zip(sw_idx, sw_out):
        it = items[pi]
        for s in o.outputs:
            per[pi]["switch"].append({"correct": run_tests(extract_code(s.text), it), "strategy": m})
    out = {"tag": a.tag, "model": mp, "bench": a.bench, "n_problems": len(items),
           "n_def": a.n_def, "n_rec": a.n_rec, "per_problem": per}
    Path(a.output_dir).mkdir(parents=True, exist_ok=True)
    fp = (Path(a.output_dir)/(f"rec_{a.tag}.shard{a.shard_index}-of-{a.num_shards}.json" if a.num_shards>1 else f"rec_{a.tag}.json"))
    json.dump(out, open(fp,"w")); print(f"[{a.tag} shard {a.shard_index}/{a.num_shards}] {len(items)} -> {fp}")


def merge(a):
    parts=[]
    for s in range(a.num_shards):
        fp=Path(a.output_dir)/f"rec_{a.tag}.shard{s}-of-{a.num_shards}.json"
        if not fp.exists(): raise FileNotFoundError(f"missing shard {s}")
        parts.append(json.load(open(fp)))
    per=[pp for part in parts for pp in part["per_problem"]]; nrec=parts[0]["n_rec"]
    # among problems where DEFAULT failed (no default sample correct), recovery rate for iid-retry vs switch
    # matched budget: iid_retry has n_rec samples; switch: sample n_rec from its pool
    nfail=0; iid_rec=0; sw_rec=0
    import random
    for p in per:
        if any(p["default"]): continue
        nfail+=1
        if any(p["iid_retry"]): iid_rec+=1
        pool=[x["correct"] for x in p["switch"]]
        # subsample n_rec from switch pool to match budget
        idx=random.sample(range(len(pool)), min(nrec,len(pool)))
        if any(pool[i] for i in idx): sw_rec+=1
    out={"tag":a.tag,"model":parts[0]["model"],"bench":parts[0]["bench"],"n_problems":len(per),
         "n_default_failed":nfail,"iid_retry_recovery":iid_rec/max(nfail,1),"switch_recovery":sw_rec/max(nfail,1),
         "switch_minus_iid":(sw_rec-iid_rec)/max(nfail,1),"per_problem":per}
    fp=Path(a.output_dir)/f"rec_{a.tag}.json"; json.dump(out,open(fp,"w"))
    print(f"[{a.tag}] n={len(per)} default_failed={nfail}  iid_retry_recov={out['iid_retry_recovery']:.3f} "
          f"switch_recov={out['switch_recovery']:.3f}  switch-iid={out['switch_minus_iid']:+.3f} (matched budget n_rec={nrec})")
    print(f"saved -> {fp}")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model-path"); ap.add_argument("--bench",default="humaneval")
    ap.add_argument("--max-problems",type=int,default=-1); ap.add_argument("--n-def",type=int,default=8); ap.add_argument("--n-rec",type=int,default=12)
    ap.add_argument("--output-dir",default="/tmp/instance_storage/gu/eval_out"); ap.add_argument("--tag",default="rec")
    ap.add_argument("--shard-index",type=int,default=0); ap.add_argument("--num-shards",type=int,default=1); ap.add_argument("--merge",action="store_true")
    a=ap.parse_args(); merge(a) if a.merge else run(a)


if __name__=="__main__":
    main()
