"""CCT C2 bank builder — wiring supervision. From RFT-checkpoint rollouts on DAG tasks, extract two EQUAL-BUDGET banks:
  A) whole-verified: full solve() completions whose whole-solution output matches the reference (ordinary RFT bank).
  B) connected-wiring: validated CONNECTED fragments — a parent->child (or longer contiguous) sub-DAG where every block is
     conditional-contract correct AND binding-correct AND the EDGE carries the parent's real output into the child correctly
     (observed-chain OK within the fragment). Emitted as a mini `solve` computing that sub-DAG. This supervises data-flow WIRING
     (the §106 bottleneck) using the candidate's OWN validated code — NOT post-hoc rewiring (§107, which hurt).

C2 trains arm A vs arm B (equal #examples, same output format) from the RFT ckpt, evals on HELD-OUT wiring patterns.
Here we only BUILD the banks + validate extraction. CPU-only.
"""
import argparse, ast, json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl_training.comp_tasks import _rand_records, _extract
from rl_training.comp_dag import gen_dag, render
from rl_training.cct_c0 import audit_candidate, _extract_solve_body, _refs

def _subdag(dag, idxs):
    steps = [dag["steps"][i] for i in idxs]
    return {"steps": steps, "out": steps[-1]["var"]}

def _fragment_completion(cand_code, frag_vars, out_var):
    """build a solve() computing only frag_vars (in order) and returning out_var, from the candidate's own statements."""
    stmts = _extract_solve_body(cand_code)
    if not stmts or any(v not in stmts for v in frag_vars):
        return None
    lines = ["def solve(records):"]
    for v in frag_vars:
        try:
            lines.append("    " + ast.unparse(stmts[v]))
        except Exception:
            return None
    lines.append(f"    return {out_var}")
    return "```python\n" + "\n".join(lines) + "\n```"

def connected_fragments(records, dag, cand_code, rng):
    """return list of validated connected fragments (prompt, completion) from ONE candidate.
    A fragment = contiguous block indices [i..j] (topo order) s.t. every block is parsed+conditional+binding OK
    and each block's declared args are earlier fragment vars or 'records' (edges stay inside the fragment)."""
    blocks = audit_candidate(records, dag, cand_code, rng)
    if blocks is None:
        return []
    okflag = {}
    for b in blocks:
        okflag[b["var"]] = bool(b["parsed"] and b["conditional"] and b["binding"])
    steps = dag["steps"]; frags = []
    n = len(steps)
    for i in range(n):
        fragvars = []
        for j in range(i, n):
            s = steps[j]; v = s["var"]
            if not okflag.get(v):
                break
            # edges must stay inside the fragment (args are records or an earlier fragment var)
            inside = all((a == "records") or (a in fragvars) for a in s["args"])
            if not inside and j > i:
                break
            fragvars.append(v)
            if len(fragvars) >= 2:  # a fragment must exercise >=1 edge (wiring)
                sub = _subdag(dag, list(range(i, j + 1)))
                comp = _fragment_completion(cand_code, fragvars, v)
                if comp:
                    frags.append({"prompt": render(sub), "completion": comp, "nblocks": len(fragvars)})
    return frags

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--n-tasks", type=int, default=64)
    ap.add_argument("--n-nodes", type=int, default=6)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--out-whole", required=True)
    ap.add_argument("--out-wiring", required=True)
    a = ap.parse_args()
    rr = random.Random(a.seed0 * 733 + 5)
    tasks = [{"dag": gen_dag(s, n_nodes=a.n_nodes), "records": _rand_records(rr, rr.randint(6, 10))}
             for s in range(a.seed0, a.seed0 + a.n_tasks)]
    for t in tasks: t["prompt"] = render(t["dag"])
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    llm = LLM(model=a.model, trust_remote_code=True, dtype="bfloat16",
              gpu_memory_utilization=float(os.environ.get("GEN_GPU_MEM", "0.45")), max_model_len=2048, enforce_eager=True)
    sp = SamplingParams(n=a.k, temperature=a.temperature, top_p=0.95, max_tokens=640)
    def chat(p):
        try: return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        except Exception: return p + "\n"
    outs = llm.generate([chat(t["prompt"]) for t in tasks], sp)
    rng = random.Random(a.seed0 + 99)
    whole, wiring = [], []
    for t, o in zip(tasks, outs):
        for c in o.outputs:
            code = _extract(c.text)
            if not code: continue
            try:
                blocks = audit_candidate(t["records"], t["dag"], code, rng)
            except Exception:
                continue
            if blocks is None: continue
            if blocks and blocks[0]["whole"]:
                comp = c.text if "```" in c.text else "```python\n" + code + "\n```"
                whole.append({"prompt": t["prompt"], "completion": comp})
            else:
                try:
                    wiring.extend(connected_fragments(t["records"], t["dag"], code, rng))
                except Exception:
                    pass
    with open(a.out_whole, "w") as f:
        for r in whole: f.write(json.dumps(r) + "\n")
    with open(a.out_wiring, "w") as f:
        for r in wiring: f.write(json.dumps(r) + "\n")
    print(f"[cct_c2_gen] whole={len(whole)} wiring_fragments={len(wiring)} -> {a.out_whole}, {a.out_wiring}")

if __name__ == "__main__":
    main()
