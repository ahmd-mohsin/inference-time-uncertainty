# DAG compositional domain: operations form a DAG (branching, merge, intermediate reuse), NOT a linear chain.
# Correct data-flow WIRING (which intermediate feeds which op) is the non-trivial part here — unlike linear pipelines
# where composition is free (§103b). Modes: --mode emit (write pool) | --mode diag (E1 oracle-components diagnostic).
import argparse, json, random, signal, os
from copy import deepcopy
from rl_training.comp_tasks import _rand_records, PRIMS, _extract

def _concat(a, b): return a + b
def _union_id(a, b):
    seen = set(); out = []
    for r in a + b:
        if r["id"] not in seen: seen.add(r["id"]); out.append(r)
    return out
def _zip_sum(a, b):
    out = []
    for x, y in zip(a, b):
        z = dict(x); z["val"] = x["val"] + y["val"]; out.append(z)
    return out
BIN = {"concat": _concat, "union_id": _union_id, "zip_sum": _zip_sum}
BIN_DESC = {"concat": "concatenate list {a} followed by list {b}",
            "union_id": "concatenate {a} then {b}, then stably drop records whose id already appeared (keep first)",
            "zip_sum": "pair up {a} and {b} by position and add their val fields (stop at the shorter list)"}
BIN_CODE = {"concat": "def op(a, b):\n    return a + b",
            "union_id": "def op(a, b):\n    seen=set(); out=[]\n    for r in a+b:\n        if r['id'] not in seen: seen.add(r['id']); out.append(r)\n    return out",
            "zip_sum": "def op(a, b):\n    out=[]\n    for x,y in zip(a,b):\n        z=dict(x); z['val']=x['val']+y['val']; out.append(z)\n    return out"}
UNARY = list(PRIMS)
from rl_training.comp_tasks import _STEP_PHRASE

def gen_dag(seed, n_nodes=6):
    rng = random.Random(seed)
    steps = []; nvars = ["records"]
    for i in range(n_nodes):
        v = f"v{i+1}"
        if len(nvars) >= 2 and rng.random() < 0.45:
            a, b = rng.sample(nvars, 2); op = rng.choice(list(BIN))
            steps.append({"var": v, "kind": "bin", "op": op, "args": [a, b]})
        else:
            src = rng.choice(nvars); op = rng.choice(UNARY)
            steps.append({"var": v, "kind": "un", "op": op, "args": [src]})
        nvars.append(v)
    return {"steps": steps, "out": nvars[-1]}

def run_dag(records, dag):
    env = {"records": deepcopy(records)}
    for s in dag["steps"]:
        if s["kind"] == "un": env[s["var"]] = PRIMS[s["op"]]["fn"](deepcopy(env[s["args"][0]]))
        else: env[s["var"]] = BIN[s["op"]](deepcopy(env[s["args"][0]]), deepcopy(env[s["args"][1]]))
    return env[dag["out"]]

def render(dag):
    lines = []
    for s in dag["steps"]:
        if s["kind"] == "un":
            lines.append(f"{s['var']} = ({_STEP_PHRASE[s['op']]}) applied to {s['args'][0]}")
        else:
            lines.append(f"{s['var']} = " + BIN_DESC[s["op"]].format(a=s["args"][0], b=s["args"][1]))
    recs = _rand_records(random.Random(0), 6)
    return ("You are given a list of records (dicts with keys id, grp, val, ok, ts). Compute these named intermediate "
            "lists IN ORDER (each vN is a list; note which inputs each uses — the data-flow is a DAG, not a simple chain):\n"
            + "\n".join(lines) + f"\nReturn `def solve(records):` that computes these and returns {dag['out']}. "
            "Return ONLY a ```python code block.\nExample records = " + json.dumps(recs))

def oracle_block(dag):
    # correct code for each op (components); the model must WIRE them per the DAG
    out = []
    for s in dag["steps"]:
        if s["kind"] == "un":
            code = "def op(r):\n" + "\n".join("    " + l for l in PRIMS[s["op"]]["code"].split("\n"))
            out.append(f"# {s['var']} = op({s['args'][0]}) where:\n{code}")
        else:
            out.append(f"# {s['var']} = op({s['args'][0]}, {s['args'][1]}) where:\n{BIN_CODE[s['op']]}")
    return "\n".join(out)

def verify(text, task, timeout=5):
    src = _extract(text); ns = {}
    try:
        signal.signal(signal.SIGALRM, lambda *a: (_ for _ in ()).throw(TimeoutError())); signal.alarm(timeout)
        exec(src, ns); solve = ns.get("solve")
        if not callable(solve): signal.alarm(0); return False
        for inp in task["test_inputs"]:
            try: got = solve(deepcopy(inp))
            except Exception: signal.alarm(0); return False
            if got != run_dag(inp, task["dag"]): signal.alarm(0); return False
        signal.alarm(0); return True
    except Exception:
        try: signal.alarm(0)
        except Exception: pass
        return False

def make_task(seed, n_nodes):
    dag = gen_dag(seed, n_nodes)
    ti = [_rand_records(random.Random(seed*100+j), random.Random(seed).randint(6, 10)) for j in range(6)]
    return {"dag": dag, "test_inputs": ti, "prompt": render(dag), "seed": seed, "n_nodes": n_nodes}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["emit", "diag"], default="emit")
    ap.add_argument("--n", type=int, default=300); ap.add_argument("--nodes", type=int, default=6)
    ap.add_argument("--seed0", type=int, default=0); ap.add_argument("--out", default="")
    ap.add_argument("--model", default=""); ap.add_argument("--k", type=int, default=8); ap.add_argument("--stats-out", default="")
    a = ap.parse_args()
    tasks = [make_task(a.seed0 + i, a.nodes) for i in range(a.n)]
    if a.mode == "emit":
        with open(a.out, "w") as f:
            for t in tasks: f.write(json.dumps(t) + "\n")
        print(f"wrote {len(tasks)} DAG tasks (nodes={a.nodes}) -> {a.out}"); return
    # diag: base pass@k + oracle-components recovery (does WIRING/composition limit?)
    from vllm import LLM, SamplingParams
    llm = LLM(model=a.model, dtype="bfloat16", trust_remote_code=True, max_model_len=4096,
              gpu_memory_utilization=float(os.environ.get("GEN_GPU_MEM", 0.5)), enforce_eager=True)
    tok = llm.get_tokenizer()
    def wrap(p): return tok.apply_chat_template([{"role":"user","content":p}], tokenize=False, add_generation_prompt=True)
    sp = SamplingParams(n=a.k, temperature=1.0, top_p=0.95, max_tokens=1536, stop=["<|im_end|>","<|endoftext|>"])
    outs = llm.generate([wrap(t["prompt"]) for t in tasks], sp)
    frontier = [t for t, o in zip(tasks, outs) if not any(verify(s.text, t) for s in o.outputs)]
    cprompts = [wrap(t["prompt"] + "\n\nHere is CORRECT code for each step's operation (you must WIRE them per the DAG above):\n```python\n"
                     + oracle_block(t["dag"]) + "\n```\nAssemble `def solve(records):` following the exact data-flow. Return ONLY a ```python code block.")
                for t in frontier]
    rec = 0
    if cprompts:
        couts = llm.generate(cprompts, sp)
        rec = sum(1 for t, o in zip(frontier, couts) if any(verify(s.text, t) for s in o.outputs))
    stats = {"n": len(tasks), "nodes": a.nodes, "frontier_zero_success_at_k": len(frontier), "budget_k": a.k,
             "oracle_components_recovered": rec, "oracle_components_recovery_rate": round(rec/max(1,len(frontier)), 3)}
    print("[dag_diag]", json.dumps(stats))
    if a.stats_out: json.dump(stats, open(a.stats_out, "w"))

if __name__ == "__main__": main()
