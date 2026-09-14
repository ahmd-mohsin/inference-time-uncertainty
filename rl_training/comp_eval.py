# Calibration + frozen-model eval for the compositional experiment (§64). PATCHED (§150): records per-problem success
# COUNT c out of k (not just any-of-k coverage) + accepts --seed, enabling pass@1, the Chen et al. unbiased pass@k
# estimator, and analytic null turnover (p_i=c_i/k). Usage: python -m rl_training.comp_eval --model M --pool P --k 16 --seed S --temperature 0.8
import argparse, json, os, sys, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import random as _random
from rl_training.comp_tasks import verify_solution, _rand_records


def _ensure_test_inputs(r):
    ti = r.get("test_inputs")
    if ti:
        return ti
    rng = _random.Random(abs(hash(r["prompt"])) % 10 ** 7)
    return [_rand_records(rng, rng.randint(6, 10)) for _ in range(6)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0, help="vLLM sampling seed (independent draws for null-turnover)")
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--tag", default="comp")
    ap.add_argument("--out-dir", default="/tmp/instance_storage/gu/eval_out")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.pool) if l.strip()][: a.n]
    if os.path.isdir(a.model) and os.path.exists(os.path.join(a.model, "adapter_config.json")) \
            and not os.path.exists(os.path.join(a.model, "config.json")):
        from rl_training.model_utils import merge_adapter_if_needed
        a.model = merge_adapter_if_needed(a.model)
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    llm = LLM(model=a.model, trust_remote_code=True, dtype="bfloat16",
              gpu_memory_utilization=float(os.environ.get("EVAL_GPU_MEM", "0.5")),
              max_model_len=2048, enforce_eager=True)
    sp = SamplingParams(n=a.k, temperature=a.temperature, top_p=0.95, max_tokens=a.max_tokens, seed=a.seed)

    def chat(p):
        try:
            return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        except Exception:
            return p + "\n"

    outs = llm.generate([chat(r["prompt"]) for r in rows], sp)
    from rl_training.comp_tasks import NONCOMMUTING
    pairset = {frozenset(p) for p in NONCOMMUTING}
    perprob = []; cov = 0
    for r, o in zip(rows, outs):
        task = {"prog": r["prog"], "test_inputs": _ensure_test_inputs(r)}
        c = sum(1 for comp in o.outputs if verify_solution(comp.text, task))   # success COUNT out of k
        kk = len(o.outputs); cov += int(c > 0)
        prog = r["prog"]; fam = "none"
        for i in range(len(prog) - 1):
            if frozenset((prog[i], prog[i + 1])) in pairset:
                fam = "%s|%s" % tuple(sorted((prog[i], prog[i + 1]))); break
        perprob.append({"prog": prog, "fam": fam, "ok": int(c > 0), "c": c, "k": kk})
    n = len(rows)
    pass1 = sum(p["c"] / p["k"] for p in perprob) / n          # mean single-attempt success (pass@1)
    coverage = cov / n                                          # pass@k (any-of-k) = old 'ok' metric
    res = {"tag": a.tag, "model": a.model, "pool": a.pool, "n": n, "k": a.k, "seed": a.seed,
           "pass1": pass1, "coverage_passk": coverage, "acc": coverage, "per_problem": perprob}
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, f"comp_{a.tag}.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"[comp_eval {a.tag}] n={n} k={a.k} seed={a.seed} pass1={pass1:.4f} coverage@k={coverage:.4f}")


if __name__ == "__main__":
    main()
