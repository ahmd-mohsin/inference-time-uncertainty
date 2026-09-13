"""CCT novelty sub-gate (before C1): measure the RFT model's DIRECT competence per op on single-block tasks.

C0 (§144) showed conditional-contract recovers 12k components, but skewed to trivial ops (concat=50%). The memo's
C0 continue-criterion demands the recovered supervision be on contexts the model STILL ERRS ON — not re-copies of
mastered operators. Here we probe, per op, whether the RFT model can produce that transform DIRECTLY (single-block
task, fresh inputs). direct_mastery[op] = pass@k coverage. Then novelty-adjusted recovered supervision =
  sum_op  C0_recovered[op] * (1 - direct_mastery[op]).
If most recoveries are on ops the model already masters standalone -> weak (STOP). If recoveries concentrate on ops
it fails directly -> genuinely novel supervision -> proceed to C1.

This is op-level novelty (not full multi-block context) — the honest first cut given C0 stored op counts, not contexts.
"""
import argparse, json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl_training.comp_tasks import _rand_records, _extract, PRIMS
from rl_training.comp_dag import render, BIN
from rl_training.cct_c0 import audit_candidate

UNARY_OPS = list(PRIMS); BIN_OPS = list(BIN)

def _single_block_dag(op):
    if op in BIN:
        return {"steps": [{"var": "v1", "kind": "bin", "op": op, "args": ["records", "records"]}], "out": "v1"}
    return {"steps": [{"var": "v1", "kind": "un", "op": op, "args": ["records"]}], "out": "v1"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ops", required=True, help="comma-separated ops this shard probes")
    ap.add_argument("--n-per-op", type=int, default=40)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ops = [o for o in a.ops.split(",") if o]

    tasks = []
    rr = random.Random(a.seed0 * 131 + 3)
    for op in ops:
        dag = _single_block_dag(op)
        for i in range(a.n_per_op):
            recs = _rand_records(rr, rr.randint(6, 10))
            tasks.append({"op": op, "dag": dag, "records": recs, "prompt": render(dag)})

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    llm = LLM(model=a.model, trust_remote_code=True, dtype="bfloat16",
              gpu_memory_utilization=float(os.environ.get("GEN_GPU_MEM", "0.45")),
              max_model_len=2048, enforce_eager=True)
    sp = SamplingParams(n=a.k, temperature=a.temperature, top_p=0.95, max_tokens=512)

    def chat(p):
        try: return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        except Exception: return p + "\n"

    outs = llm.generate([chat(t["prompt"]) for t in tasks], sp)
    rng = random.Random(a.seed0 + 7)
    per = {op: {"tasks": 0, "solved_anyk": 0, "cand": 0, "cand_pass": 0} for op in ops}
    for t, o in zip(tasks, outs):
        op = t["op"]; per[op]["tasks"] += 1; any_ok = False
        for c in o.outputs:
            code = _extract(c.text)
            if not code:
                per[op]["cand"] += 1; continue
            try:
                blocks = audit_candidate(t["records"], t["dag"], code, rng)
            except Exception:
                blocks = None
            per[op]["cand"] += 1
            if blocks and blocks[0]["whole"]:
                per[op]["cand_pass"] += 1; any_ok = True
        if any_ok: per[op]["solved_anyk"] += 1
    for op in ops:
        d = per[op]; d["mastery_passk"] = round(d["solved_anyk"] / max(1, d["tasks"]), 3)
        d["mastery_pass1"] = round(d["cand_pass"] / max(1, d["cand"]), 3)
    with open(a.out, "w") as f:
        json.dump({"model": a.model, "seed0": a.seed0, "per_op": per}, f, indent=2)
    print("[cct_novelty]", {op: per[op]["mastery_passk"] for op in ops}, "->", a.out)

if __name__ == "__main__":
    main()
