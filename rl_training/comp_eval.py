# Calibration + frozen-model eval for the compositional experiment (§64). Runs vLLM on a pool jsonl (or the component
# bank), executes each emitted `solve` via comp_tasks.verify_solution, reports pass rate overall and by kind.
# Workstream-A gate: component accuracy (~80-95%) and composed accuracy (~20-60%) => real composition gap.
# Usage: python -m rl_training.comp_eval --model <path|id> --pool /path/pool.jsonl --k 1 [--n 200] [--tag t]
import argparse, json, os, sys, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import random as _random
from rl_training.comp_tasks import verify_solution, _rand_records


def _ensure_test_inputs(r):
    ti = r.get("test_inputs")
    if ti:
        return ti
    rng = _random.Random(abs(hash(r["prompt"])) % 10 ** 7)   # component-bank rows carry no test_inputs
    return [_rand_records(rng, rng.randint(6, 10)) for _ in range(6)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pool", required=True, help="pool jsonl with prompt,prog,test_inputs")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--k", type=int, default=1, help="samples/prompt; k=1 greedy-ish for frozen accuracy")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--tag", default="comp")
    ap.add_argument("--out-dir", default="/tmp/instance_storage/gu/eval_out")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.pool) if l.strip()][: a.n]
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    llm = LLM(model=a.model, trust_remote_code=True, dtype="bfloat16",
              gpu_memory_utilization=float(os.environ.get("EVAL_GPU_MEM", "0.85")),
              max_model_len=2048, enforce_eager=True)
    sp = SamplingParams(n=a.k, temperature=(a.temperature if a.k > 1 else 0.0),
                        top_p=0.95, max_tokens=a.max_tokens)

    def chat(p):
        msgs = [{"role": "user", "content": p}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            return p + "\n"

    prompts = [chat(r["prompt"]) for r in rows]
    outs = llm.generate(prompts, sp)
    by_kind = collections.defaultdict(lambda: [0, 0])
    npass = 0
    for r, o in zip(rows, outs):
        task = {"prog": r["prog"], "test_inputs": _ensure_test_inputs(r)}
        # pass@1 semantics: any of k sampled solutions verifies (k=1 => the single greedy solution)
        ok = any(verify_solution(c.text, task) for c in o.outputs)
        npass += int(ok)
        kind = r.get("kind", "?").split(":")[0]
        by_kind[kind][0] += int(ok); by_kind[kind][1] += 1
    acc = npass / len(rows)
    res = {"tag": a.tag, "model": a.model, "pool": a.pool, "n": len(rows), "k": a.k,
           "acc": acc, "by_kind": {k: v[0] / v[1] for k, v in by_kind.items()},
           "by_kind_counts": {k: v for k, v in by_kind.items()}}
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, f"comp_{a.tag}.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"[comp_eval {a.tag}] n={len(rows)} k={a.k} acc={acc:.4f} by_kind=" +
          " ".join(f"{k}:{v[0]/v[1]:.3f}({v[1]})" for k, v in by_kind.items()))


if __name__ == "__main__":
    main()
