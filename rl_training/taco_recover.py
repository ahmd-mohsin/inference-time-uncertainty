# Counterfactual Recovery Atlas at SCALE on TACO (competitive programming, stdin/stdout verified).
# Produces the (failed_state, Q_i(z) action matrix, A^rec) dataset that CRO experiments #1 (router),
# #2 (A^rec ablation) and #4 (consolidation) consume. Same structure as code_recover but with a real
# stdin/stdout execution verifier + difficulty tiers.
#
# Per problem (stdin-type TACO): DEFAULT (n_def iid) -> on failure, matched IID-RETRY (n_rec) + each of
# the 8 recovery strategies (switch pool). Verify by running the code with each test input on stdin and
# comparing stdout. 8-GPU DP over problems + merge (oracle / best-fixed / A^rec).
#
# Usage: python -m rl_training.taco_recover --model-path <dir> --tag taco_qc --difficulty MEDIUM \
#   --max-problems 400 --shard-index S --num-shards 8 --n-def 4 --n-rec 12
import argparse, json, os, sys, subprocess, tempfile, random
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl_training.code_passk import extract_code, STRATEGIES, STRAT_NAMES


def load_taco(difficulty, cap):
    # newer `datasets` dropped loading-script support (BAAI/TACO ships TACO.py) -> load the
    # HF auto-converted parquet (always present under refs/convert/parquet) directly with pandas.
    import pandas as pd
    from huggingface_hub import HfApi, hf_hub_download
    tok = open(os.path.expanduser("~/.hf_token")).read().strip() if os.path.exists(os.path.expanduser("~/.hf_token")) else None
    api = HfApi(token=tok)
    # BAAI/TACO ships raw parquet under ALL/ in the MAIN revision (bypasses the loading script).
    files = [f for f in api.list_repo_files("BAAI/TACO", repo_type="dataset")
             if f.endswith(".parquet") and "test" in f.lower()]
    rows = []
    for f in files:
        p = hf_hub_download("BAAI/TACO", f, repo_type="dataset", token=tok)
        rows.append(pd.read_parquet(p))
    d = pd.concat(rows, ignore_index=True).to_dict("records")
    items = []
    for r in d:
        if difficulty and difficulty != "ALL" and (r.get("difficulty") or "").upper() != difficulty.upper():
            continue
        try:
            io = json.loads(r["input_output"]) if isinstance(r.get("input_output"), str) else r.get("input_output")
        except Exception:
            continue
        if not io or io.get("fn_name"):   # keep stdin/stdout problems only (skip call-based)
            continue
        ins = io.get("inputs") or []; outs = io.get("outputs") or []
        if not ins or len(ins) != len(outs):
            continue
        items.append({"id": f"taco/{len(items)}", "question": r["question"],
                      "inputs": ins[:8], "outputs": outs[:8]})  # cap tests/problem for cost
        if cap > 0 and len(items) >= cap:
            break
    return items


def run_stdin(code, ins, outs, timeout=8, return_err=False):
    """True iff code, run as a script, matches expected stdout on ALL given test cases.
    If return_err, also return a short failure signature (stderr tail / 'wrong-output' /
    'timeout') for CRO router features."""
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(code); path = f.name
    except Exception:
        return (False, "harness:tmpfile") if return_err else False
    ok = True; err = ""
    try:
        for inp, exp in zip(ins, outs):
            si = inp if isinstance(inp, str) else "\n".join(map(str, inp))
            eo = exp if isinstance(exp, str) else "\n".join(map(str, exp))
            try:
                r = subprocess.run(["python3", path], input=si, capture_output=True, text=True, timeout=timeout)
            except subprocess.TimeoutExpired:
                ok = False; err = "timeout"; break
            except Exception as e:
                ok = False; err = f"harness:{type(e).__name__}"; break
            if r.returncode != 0:
                ok = False; err = ((r.stderr or "").strip().splitlines() or ["nonzero-exit"])[-1][:200]; break
            if r.stdout.strip() != eo.strip():
                ok = False; err = "wrong-output"; break
    finally:
        try: os.unlink(path)
        except Exception: pass
    return (ok, err) if return_err else ok


def build_prompt(it, mp, strategy=None):
    instr = (STRATEGIES[strategy] + " ") if strategy else ""
    user = (f"{instr}Solve this competitive-programming problem. Read input from stdin, write the answer "
            f"to stdout. Return ONLY a ```python code block.\n\n{it['question']}")
    sysmsg = "You are an expert competitive programmer."
    ml = mp.lower()
    if any(k in ml for k in ["qwen","deepseek"]):
        return f"<|im_start|>system\n{sysmsg}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    if "llama" in ml:
        return (f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{sysmsg}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
    return f"<|im_start|>system\n{sysmsg}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"


def run(a):
    from vllm import LLM, SamplingParams
    from rl_training.model_utils import merge_adapter_if_needed
    mp = merge_adapter_if_needed(a.model_path)
    items = load_taco(a.difficulty, a.max_problems)
    if a.num_shards > 1: items = items[a.shard_index::a.num_shards]
    _gm = float(os.environ.get("EVAL_GPU_MEM", 0.85)); _eager = os.environ.get("EVAL_ENFORCE_EAGER","1")=="1"
    llm = LLM(model=mp, dtype="bfloat16", trust_remote_code=True, tensor_parallel_size=1, max_model_len=4096,
              gpu_memory_utilization=_gm, enable_prefix_caching=True, enforce_eager=_eager)
    def gen(prompts, n):
        sp = SamplingParams(n=n, temperature=1.0, top_p=0.95, max_tokens=1536, stop=["<|im_end|>","<|eot_id|>","<|endoftext|>"])
        return llm.generate(prompts, sp)
    plain = gen([build_prompt(it, mp) for it in items], a.n_def + a.n_rec)
    per_strat = max(1, a.n_rec // len(STRAT_NAMES) + 1)
    sp2, sidx = [], []
    for pi, it in enumerate(items):
        for m in STRAT_NAMES: sp2.append(build_prompt(it, mp, m)); sidx.append((pi, m))
    sout = gen(sp2, per_strat) if sp2 else []
    per = []
    for it, o in zip(items, plain):
        outs = list(o.outputs)
        deff = []; fail_code = ""; fail_err = ""
        for s in outs[:a.n_def]:
            code = extract_code(s.text); ok, err = run_stdin(code, it["inputs"], it["outputs"], return_err=True)
            deff.append(ok)
            if not ok and not fail_code:   # capture first failed default as router feature
                fail_code, fail_err = code[:2000], err
        iidr = [run_stdin(extract_code(s.text), it["inputs"], it["outputs"]) for s in outs[a.n_def:a.n_def+a.n_rec]]
        per.append({"id": it["id"], "default": deff, "iid_retry": iidr, "switch": [],
                    "question": it["question"][:1500], "fail_code": fail_code, "fail_err": fail_err})
    for (pi, m), o in zip(sidx, sout):
        it = items[pi]
        for s in o.outputs:
            per[pi]["switch"].append({"correct": run_stdin(extract_code(s.text), it["inputs"], it["outputs"]), "strategy": m})
    out = {"tag": a.tag, "model": mp, "difficulty": a.difficulty, "n_problems": len(items),
           "n_def": a.n_def, "n_rec": a.n_rec, "strategies": STRAT_NAMES, "per_problem": per}
    Path(a.output_dir).mkdir(parents=True, exist_ok=True)
    fp = (Path(a.output_dir)/(f"taco_{a.tag}.shard{a.shard_index}-of-{a.num_shards}.json" if a.num_shards>1 else f"taco_{a.tag}.json"))
    json.dump(out, open(fp,"w")); print(f"[{a.tag} shard {a.shard_index}/{a.num_shards}] {len(items)} -> {fp}")


def merge(a):
    import statistics
    parts=[]
    for s in range(a.num_shards):
        fp=Path(a.output_dir)/f"taco_{a.tag}.shard{s}-of-{a.num_shards}.json"
        if not fp.exists(): raise FileNotFoundError(f"missing shard {s}")
        parts.append(json.load(open(fp)))
    per=[pp for part in parts for pp in part["per_problem"]]; strat=parts[0]["strategies"]; nrec=parts[0]["n_rec"]
    fails=[p for p in per if not any(p["default"])]; nf=len(fails)
    if nf==0: json.dump({"tag":a.tag,"n_problems":len(per),"n_default_failed":0}, open(Path(a.output_dir)/f"taco_{a.tag}.json","w")); print(f"[{a.tag}] no failures"); return
    iid=sw=0; oracle=[]; bfacc={m:[] for m in strat}
    for p in fails:
        if any(p["iid_retry"]): iid+=1
        by={m:[] for m in strat}
        for x in p["switch"]: by[x["strategy"]].append(x["correct"])
        r={m:(1 if any(by[m]) else 0) for m in strat}
        pool=[x["correct"] for x in p["switch"]]; idx=random.sample(range(len(pool)),min(nrec,len(pool))) if pool else []
        if any(pool[i] for i in idx): sw+=1
        oracle.append(max(r.values()) if r else 0)
        for m in strat: bfacc[m].append(r[m])
    out={"tag":a.tag,"model":parts[0]["model"],"difficulty":parts[0]["difficulty"],"n_problems":len(per),
         "n_default_failed":nf,"default_solve_rate":sum(1 for p in per if any(p["default"]))/len(per),
         "iid_retry_recovery":iid/nf,"switch_recovery":sw/nf,"G_switch":(sw-iid)/nf,
         "oracle_recovery":statistics.mean(oracle),"best_fixed_recovery":max(statistics.mean(bfacc[m]) for m in strat),
         "per_problem":per}
    json.dump(out,open(Path(a.output_dir)/f"taco_{a.tag}.json","w"))
    print(f"[{a.tag}] nprob={len(per)} defsolve={out['default_solve_rate']:.2f} nfail={nf} iid={out['iid_retry_recovery']:.3f} "
          f"switch={out['switch_recovery']:.3f} G={out['G_switch']:+.3f} oracle={out['oracle_recovery']:.3f} bestfix={out['best_fixed_recovery']:.3f}")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model-path"); ap.add_argument("--difficulty",default="MEDIUM")
    ap.add_argument("--max-problems",type=int,default=400); ap.add_argument("--n-def",type=int,default=4); ap.add_argument("--n-rec",type=int,default=12)
    ap.add_argument("--output-dir",default="/tmp/instance_storage/gu/eval_out"); ap.add_argument("--tag",default="taco")
    ap.add_argument("--shard-index",type=int,default=0); ap.add_argument("--num-shards",type=int,default=1); ap.add_argument("--merge",action="store_true")
    a=ap.parse_args(); merge(a) if a.merge else run(a)


if __name__=="__main__":
    main()
