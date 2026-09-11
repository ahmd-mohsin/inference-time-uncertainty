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

def reference_solve_code(dag):
    ops_u = sorted({s["op"] for s in dag["steps"] if s["kind"] == "un"})
    ops_b = sorted({s["op"] for s in dag["steps"] if s["kind"] == "bin"})
    L = ["def solve(records):"]
    for op in ops_u:
        L.append(f"    def u_{op}(r):")
        for ln in PRIMS[op]["code"].split("\n"): L.append("        " + ln)
        L.append("        return r")
    for op in ops_b:
        for ln in BIN_CODE[op].replace("def op(", f"def b_{op}(").split("\n"): L.append("    " + ln)
    L.append("    records = [dict(x) for x in records]")
    for s in dag["steps"]:
        if s["kind"] == "un": L.append(f"    {s['var']} = u_{s['op']}([dict(x) for x in {s['args'][0]}])")
        else: L.append(f"    {s['var']} = b_{s['op']}([dict(x) for x in {s['args'][0]}], [dict(x) for x in {s['args'][1]}])")
    L.append(f"    return {dag['out']}")
    return "\n".join(L)

def _example_records(): return _rand_records(random.Random(0), 6)

def _intermediates(dag):
    env = {"records": [dict(x) for x in _example_records()]}; ann = {}
    for s in dag["steps"]:
        if s["kind"] == "un": env[s["var"]] = PRIMS[s["op"]]["fn"](deepcopy(env[s["args"][0]]))
        else: env[s["var"]] = BIN[s["op"]](deepcopy(env[s["args"][0]]), deepcopy(env[s["args"][1]]))
        ann[s["var"]] = env[s["var"]]
    return ann

def value_annotated_code(dag):
    # ROUTE B (distillation form): inline each intermediate's VERIFIED value on the example -> dense wiring/data-flow signal
    ann = _intermediates(dag); code = reference_solve_code(dag); out = []
    for l in code.split("\n"):
        out.append(l)
        s = l.strip()
        for v in ann:
            if s.startswith(v + " = "):
                out.append(" " * (len(l) - len(l.lstrip())) + f"# {v} on example == {json.dumps(ann[v])[:140]}")
    return "\n".join(out)

def plan_then_code(dag):
    # explicit data-flow PLAN (dependency graph) before code -> separates wiring from op-writing
    plan = ["# data-flow plan (each vN's inputs):"]
    for s in dag["steps"]: plan.append(f"#   {s['var']} <- {s['op']}(" + ", ".join(s["args"]) + ")")
    plan.append(f"#   output = {dag['out']}")
    return "\n".join(plan) + "\n" + reference_solve_code(dag)

def value_scram_code(dag):
    # PLACEBO: same code + same-length value comments, but VALUES SCRAMBLED across vars (wrong data-flow).
    # Isolates whether Tvalue's gain is the data-flow SIGNAL vs merely longer targets.
    ann = _intermediates(dag); keys = list(ann); vals = [ann[k] for k in keys]
    random.Random(abs(hash(str(dag["steps"]))) % (2**31)).shuffle(vals)
    scram = {k: vals[i] for i, k in enumerate(keys)}
    code = reference_solve_code(dag); out = []
    for l in code.split("\n"):
        out.append(l); s = l.strip()
        for v in scram:
            if s.startswith(v + " = "):
                out.append(" " * (len(l) - len(l.lstrip())) + f"# {v} on example == {json.dumps(scram[v])[:140]}")
    return "\n".join(out)

def target_code(dag, target):
    if target == "value": return value_annotated_code(dag)
    if target == "plan": return plan_then_code(dag)
    if target == "vscram": return value_scram_code(dag)
    return reference_solve_code(dag)

def recompose(dag, seed):
    # keep the same op multiset + node count; REWIRE args (which prior vars feed each op) -> new valid DAG
    rng = random.Random(seed); steps = []; nvars = ["records"]
    for s in dag["steps"]:
        v = s["var"]
        if s["kind"] == "bin" and len(nvars) >= 2:
            a, b = rng.sample(nvars, 2); steps.append({"var": v, "kind": "bin", "op": s["op"], "args": [a, b]})
        else:
            src = rng.choice(nvars)
            op = s["op"] if s["kind"] == "un" else UNARY[rng.randrange(len(UNARY))]
            steps.append({"var": v, "kind": "un", "op": op, "args": [src]})
        nvars.append(v)
    return {"steps": steps, "out": nvars[-1]}

def make_task(seed, n_nodes):
    dag = gen_dag(seed, n_nodes)
    ti = [_rand_records(random.Random(seed*100+j), random.Random(seed).randint(6, 10)) for j in range(6)]
    return {"dag": dag, "test_inputs": ti, "prompt": render(dag), "seed": seed, "n_nodes": n_nodes}

def valid_task(seed, n_nodes):
    # keep only DAGs whose reference executes on all test inputs (filters schema-mismatch DAGs)
    t = make_task(seed, n_nodes)
    try:
        for inp in t["test_inputs"]: run_dag(inp, t["dag"])
        return t
    except Exception:
        return None

def gen_valid(n, n_nodes, seed0):
    out = []; s = seed0
    while len(out) < n:
        t = valid_task(s, n_nodes)
        if t is not None: out.append(t)
        s += 1
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["emit", "diag", "bank", "eval"], default="emit")
    ap.add_argument("--arm", choices=["B", "F", "C"], default="B")
    ap.add_argument("--target", choices=["plain", "value", "plan", "vscram"], default="plain")
    ap.add_argument("--n", type=int, default=300); ap.add_argument("--nodes", type=int, default=6)
    ap.add_argument("--seed0", type=int, default=0); ap.add_argument("--out", default="")
    ap.add_argument("--model", default=""); ap.add_argument("--k", type=int, default=8); ap.add_argument("--stats-out", default="")
    a = ap.parse_args()
    if a.mode == "bank":
        # SFT bank {prompt, completion=reference_solve_code}. B: n straight refs. F: n/2 refs + n/2 recomposed (rewired,
        # same ops). C: n/2 refs + n/2 fresh-random DAGs (matched count, no interface focus).
        base = gen_valid((a.n + 1)//2 if a.arm != "B" else a.n, a.nodes, a.seed0)
        recs = [{"prompt": t["prompt"], "completion": target_code(t["dag"], a.target)} for t in base]
        if a.arm == "F":
            for i, t in enumerate(base):
                rc = recompose(t["dag"], a.seed0 + 500000 + i)
                tt = {"dag": rc, "test_inputs": t["test_inputs"]}
                try:
                    for inp in tt["test_inputs"]: run_dag(inp, rc)
                    recs.append({"prompt": render(rc), "completion": reference_solve_code(rc)})
                except Exception: pass
        elif a.arm == "C":
            extra = gen_valid(len(base), a.nodes, a.seed0 + 900000)
            recs += [{"prompt": t["prompt"], "completion": reference_solve_code(t["dag"])} for t in extra]
        with open(a.out, "w") as f:
            for r in recs: f.write(json.dumps(r) + "\n")
        print(f"[bank arm={a.arm}] wrote {len(recs)} SFT records -> {a.out}"); return
    if a.mode == "eval":
        from vllm import LLM, SamplingParams
        tasks = gen_valid(a.n, a.nodes, a.seed0)
        llm = LLM(model=a.model, dtype="bfloat16", trust_remote_code=True, max_model_len=4096,
                  gpu_memory_utilization=float(os.environ.get("EVAL_GPU_MEM", 0.5)), enforce_eager=True)
        tok = llm.get_tokenizer()
        def wrap(p): return tok.apply_chat_template([{"role":"user","content":p}], tokenize=False, add_generation_prompt=True)
        outs = llm.generate([wrap(t["prompt"]) for t in tasks],
                            SamplingParams(n=a.k, temperature=0.8, top_p=0.95, max_tokens=1536, stop=["<|im_end|>","<|endoftext|>"]))
        soln = sum(1 for t, o in zip(tasks, outs) if any(verify(s.text, t) for s in o.outputs))
        acc = round(soln / max(1, len(tasks)), 4)
        print(f"[dag_eval] acc={acc} n={len(tasks)} k={a.k}")
        if a.stats_out: json.dump({"acc": acc, "n": len(tasks), "k": a.k}, open(a.stats_out, "w"))
        return
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
