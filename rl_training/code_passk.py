# APPLIED downstream experiment (high-v domain = CODE). Tests whether mode-diverse (stratified)
# sampling beats iid pass@k when strategies are FUNCTIONALLY COMPLEMENTARY — the regime the ACV theory
# predicts diversity should help (unlike math where v≈0).
#
# Per problem: sample iid completions AND strategy-forced completions (iterative / recursive / DP /
# greedy / builtins / brute-force), execute each against the unit tests (subprocess, timeout), record
# (correct, strategy). Merge computes iid vs strategy-stratified pass@k + functional complementarity v.
#
# Benchmarks: HumanEval (openai_humaneval) and MBPP (mbpp). 8-GPU data-parallel over problems + merge.
# Usage: python -m rl_training.code_passk --model-path <dir> --bench humaneval --tag code_qc_he \
#   --shard-index S --num-shards 8 --n-iid 24 --n-forced 4
import argparse, json, os, sys, subprocess, tempfile, random, re
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STRATEGIES = {
    # 8 engineering-specific RECOVERY options (for the Counterfactual Recovery Atlas). Each is a
    # distinct way to *re-approach* a problem after a failed attempt.
    "root_cause":   "Carefully re-read the problem, identify the root cause of common mistakes, then implement correctly.",
    "boundary":     "Focus on boundary and edge cases (empty, single element, max/min, overflow); handle them explicitly.",
    "algo_replace": "Use a fundamentally different algorithm/data structure than the obvious one.",
    "complexity":   "Assume the naive approach is too slow; use an efficient algorithm (dynamic programming, sorting, hashing, two-pointers).",
    "defensive":    "Write defensively: validate inputs, avoid index/type errors, handle exceptions and special cases.",
    "rewrite":      "Ignore any prior approach and rewrite the solution cleanly from scratch.",
    "spec_reread":  "Re-read the specification literally; match the exact required output format and constraints.",
    "builtins":     "Implement concisely using Python built-ins / standard library (itertools, collections, math).",
}
STRAT_NAMES = list(STRATEGIES)


def load_bench(bench):
    from datasets import load_dataset
    items = []
    if bench == "humaneval":
        d = load_dataset("openai/openai_humaneval")["test"]
        for r in d:
            items.append({"id": r["task_id"], "prompt": r["prompt"], "test": r["test"],
                          "entry": r["entry_point"]})
    elif bench in ("mbpp", "mbpp_train"):
        split = "train" if bench == "mbpp_train" else "test"   # disjoint train set for GRPO (no eval contamination)
        d = load_dataset("google-research-datasets/mbpp")[split]
        for r in d:
            # mbpp: text + code + test_list; build a prompt from text + first assert's function name
            tests = "\n".join(r["test_list"])
            items.append({"id": f"mbpp/{r['task_id']}", "prompt": r["text"] + "\n", "test": tests,
                          "entry": None, "mbpp": True})
    elif bench == "mbpp_plus":
        # EvalPlus MBPP+ : base asserts (test_list) drive regime/evidence/counterexamples (assert-parseable),
        # while the harder augmented harness (r["test"], ~35x tests, self-executing) is the SOLVE gate (plus_test).
        d = load_dataset("evalplus/mbppplus")["test"]
        for r in d:
            items.append({"id": f"mbpp+/{r['task_id']}", "prompt": r["prompt"] + "\n",
                          "test": "\n".join(r["test_list"]), "plus_test": r["test"],
                          "entry": None, "mbpp": True})
    else:
        raise SystemExit(f"bad bench {bench}")
    return items


def extract_code(text):
    m = re.findall(r"```(?:python)?\n(.*?)```", text or "", re.DOTALL)
    return m[0] if m else (text or "")


def run_tests(code, item, timeout=10, return_err=False):
    """Execute code + tests in a subprocess; return True iff exit 0. If return_err, return (ok, err)
    where err is a short failure signature (stderr tail / 'timeout' / 'nonzero-exit') for router features."""
    if item.get("plus_test"):
        prog = code + "\n" + item["plus_test"] + "\n"   # EvalPlus harder harness (self-executing) = solve gate
    elif item.get("mbpp"):
        prog = code + "\n" + item["test"] + "\n"
    else:
        prog = code + "\n" + item["test"] + "\n" + f"check({item['entry']})\n"
    ok = False; err = ""
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(prog); path = f.name
        r = subprocess.run(["python3", path], capture_output=True, timeout=timeout)
        ok = r.returncode == 0
        if not ok:
            se = (r.stderr or b"").decode("utf-8", "replace")
            err = (se.strip().splitlines() or ["nonzero-exit"])[-1][:200]
    except subprocess.TimeoutExpired:
        err = "timeout"
    except Exception as e:
        err = f"harness:{type(e).__name__}"
    finally:
        try: os.unlink(path)
        except Exception: pass
    return (ok, err) if return_err else ok


def classify(code):
    """label by which strategy signal dominates the code (regex heuristics)."""
    t = code or ""
    sig = {
        "recursive": len(re.findall(r"\breturn\b.*\b(\w+)\(", t)),  # rough
        "dynamic_prog": len(re.findall(r"\b(dp|memo|cache|lru_cache)\b", t, re.I)),
        "builtins": len(re.findall(r"\b(sorted|sum|map|filter|zip|itertools|collections|math\.)\b", t)),
        "iterative": len(re.findall(r"\bfor\b|\bwhile\b", t)),
        "brute_force": len(re.findall(r"\ball\b|\bany\b|permutations|combinations|product", t)),
        "greedy": len(re.findall(r"\bgreedy\b|max\(|min\(", t)),
    }
    best = max(sig, key=lambda k: sig[k])
    return best if sig[best] > 0 else "iterative"


def build_prompt(item, model_path, strategy=None):
    instr = ""
    if strategy:
        instr = STRATEGIES[strategy] + " "
    if item.get("mbpp"):
        user = (f"{instr}Write a Python function for the following task. Return ONLY a ```python code block.\n\n"
                f"Task: {item['prompt']}\nYour code must satisfy tests like:\n{item['test'].splitlines()[0]}")
    else:
        user = (f"{instr}Complete the following Python function. Return the FULL function in a ```python "
                f"code block.\n\n{item['prompt']}")
    sysmsg = "You are an expert Python programmer."
    ml = model_path.lower()
    if any(k in ml for k in ["qwen", "deepseek"]):
        return f"<|im_start|>system\n{sysmsg}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    if "llama" in ml:
        return (f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{sysmsg}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n")
    return f"<|im_start|>system\n{sysmsg}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"


def run(a):
    from vllm import LLM, SamplingParams
    from rl_training.model_utils import merge_adapter_if_needed
    mp = merge_adapter_if_needed(a.model_path)
    items = load_bench(a.bench)
    if a.max_problems > 0: items = items[:a.max_problems]
    if a.num_shards > 1: items = items[a.shard_index::a.num_shards]
    _gm = float(os.environ.get("EVAL_GPU_MEM", 0.85)); _eager = os.environ.get("EVAL_ENFORCE_EAGER","1")=="1"
    llm = LLM(model=mp, dtype="bfloat16", trust_remote_code=True, tensor_parallel_size=1,
              max_model_len=4096, gpu_memory_utilization=_gm, enable_prefix_caching=True, enforce_eager=_eager)
    def gen(prompts, n):
        sp = SamplingParams(n=n, temperature=1.0, top_p=0.95, max_tokens=1024,
                            stop=["<|im_end|>","<|eot_id|>","<|endoftext|>"])
        return llm.generate(prompts, sp)
    # iid pool
    iid_out = gen([build_prompt(it, mp) for it in items], a.n_iid)
    # forced pool (per strategy)
    fprompts, fidx = [], []
    for pi, it in enumerate(items):
        for m in STRAT_NAMES:
            fprompts_p = build_prompt(it, mp, m); fprompts.append(fprompts_p); fidx.append((pi, m))
    f_out = gen(fprompts, a.n_forced) if fprompts else []
    per = []
    for it, o in zip(items, iid_out):
        samps = []
        for s in o.outputs:
            code = extract_code(s.text); ok = run_tests(code, it)
            samps.append({"correct": ok, "strategy": classify(code), "forced": None})
        per.append({"id": it["id"], "samples": samps})
    for (pi, m), o in zip(fidx, f_out):
        it = items[pi]
        for s in o.outputs:
            code = extract_code(s.text); ok = run_tests(code, it)
            per[pi]["samples"].append({"correct": ok, "strategy": m, "forced": m})
    out = {"tag": a.tag, "model": mp, "bench": a.bench, "n_problems": len(items),
           "n_iid": a.n_iid, "n_forced": a.n_forced, "strategies": STRAT_NAMES, "per_problem": per}
    Path(a.output_dir).mkdir(parents=True, exist_ok=True)
    fp = (Path(a.output_dir)/(f"code_{a.tag}.shard{a.shard_index}-of-{a.num_shards}.json"
          if a.num_shards>1 else f"code_{a.tag}.json"))
    json.dump(out, open(fp,"w"))
    print(f"[{a.tag} shard {a.shard_index}/{a.num_shards}] {len(items)} problems -> {fp}")


def _passk(samps, k, strat, trials=400):
    n=len(samps)
    if n==0: return 0.0
    if k>=n: return float(any(s["correct"] for s in samps))
    hit=0
    for _ in range(trials):
        if strat:
            by={}
            for i,s in enumerate(samps): by.setdefault(s["strategy"],[]).append(i)
            pools=[list(g) for g in by.values()]
            for pl in pools: random.shuffle(pl)
            random.shuffle(pools); picks=[]
            while len(picks)<k:
                prog=False
                for pl in pools:
                    if pl and len(picks)<k: picks.append(pl.pop()); prog=True
                if not prog: break
            idx=picks
        else: idx=random.sample(range(n),k)
        if any(samps[i]["correct"] for i in idx): hit+=1
    return hit/trials


def merge(a):
    parts=[]
    for s in range(a.num_shards):
        fp=Path(a.output_dir)/f"code_{a.tag}.shard{s}-of-{a.num_shards}.json"
        if not fp.exists(): raise FileNotFoundError(f"missing shard {s}: {fp}")
        parts.append(json.load(open(fp)))
    per=[pp for part in parts for pp in part["per_problem"]]
    ks=[k for k in [1,2,4,8,16] if k<=parts[0]["n_iid"]]
    # complementarity v: frac solvable problems where forced-strategies have complementary success
    def vprob(p):
        by={}
        for s in p["samples"]:
            if s["forced"]: by.setdefault(s["forced"],[]).append(s["correct"])
        solved=[m for m,vv in by.items() if any(vv)]
        return (len(solved)==1) if solved else None  # uniquely solved by one strategy
    solv=[vprob(p) for p in per if vprob(p) is not None]
    v=sum(solv)/len(solv) if solv else 0.0
    iid={k:sum(_passk(p["samples"],k,False) for p in per)/len(per) for k in ks}
    strat={k:sum(_passk(p["samples"],k,True) for p in per)/len(per) for k in ks}
    out={"tag":a.tag,"model":parts[0]["model"],"bench":parts[0]["bench"],"n_problems":len(per),
         "complementarity_v":v,"iid_passk":iid,"stratified_passk":strat,"per_problem":per}
    fp=Path(a.output_dir)/f"code_{a.tag}.json"; json.dump(out,open(fp,"w"))
    print(f"[{a.tag}] MERGED {len(per)} problems  v={v:.3f}")
    print(f"  {'k':>3}{'iid':>8}{'stratified':>11}{'gain':>8}")
    for k in ks: print(f"  {k:>3}{iid[k]:>8.3f}{strat[k]:>11.3f}{strat[k]-iid[k]:>+8.3f}")
    print(f"saved -> {fp}")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model-path"); ap.add_argument("--bench",default="humaneval")
    ap.add_argument("--max-problems",type=int,default=-1); ap.add_argument("--n-iid",type=int,default=24)
    ap.add_argument("--n-forced",type=int,default=4); ap.add_argument("--output-dir",default="/tmp/instance_storage/gu/eval_out")
    ap.add_argument("--tag",default="code"); ap.add_argument("--shard-index",type=int,default=0)
    ap.add_argument("--num-shards",type=int,default=1); ap.add_argument("--merge",action="store_true")
    a=ap.parse_args(); merge(a) if a.merge else run(a)


if __name__=="__main__":
    main()
