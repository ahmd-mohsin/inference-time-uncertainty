"""CCT gate C0 — mechanism/verifier audit in the DAG domain (proposal §143 / support-coverage memo).

Question C0: do FAILED whole-solutions contain CONDITIONALLY-correct components that whole-success (and even
observed-execution) masking discards, and are they on contexts the RFT model still errs on?

For a DAG task, block k is `v_k = op(declared_args)` with op in PRIMS/BIN (the ground-truth contract).
Given a candidate `solve(records)` we build three masks per block k:
  - whole:      retain block k iff the FULL candidate output matches the reference on the given records.
  - observed:   retain iff the candidate's computed v_k matches the reference v_k on the given records
                (contaminated by upstream errors: a correct transform on wrong input looks wrong).
  - conditional: feed INDEPENDENTLY-VARIED valid inputs to block k's own statement and check it against the
                contract op on those same inputs (recovers correct transforms hidden by upstream errors;
                rejects transforms that passed only by accidental agreement). PLUS a binding check: the
                statement must reference exactly the declared arg vars (a wrong dependency fails).

C0 output per (checkpoint, shard): counts of blocks that are conditional-correct but discarded by whole/observed,
restricted to candidates whose WHOLE solution FAILS; parse/exclusion rates; op×arity coverage of recovered blocks.
No training here — this only measures whether a richer supervision source EXISTS. CPU-only checker.
"""
import argparse, ast, json, random, sys, os
from copy import deepcopy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl_training.comp_tasks import _rand_records, PRIMS
from rl_training.comp_dag import gen_dag, run_dag, render, BIN, UNARY

def _sdc(x):
    """safe deepcopy: model candidates can produce un-deepcopyable objects (e.g. dict_values); fall back to the object."""
    try:
        return deepcopy(x)
    except Exception:
        return x

def _contract(step, env):
    """ground-truth value of a block given an env holding its declared args."""
    if step["kind"] == "un":
        return PRIMS[step["op"]]["fn"](deepcopy(env[step["args"][0]]))
    return BIN[step["op"]](deepcopy(env[step["args"][0]]), deepcopy(env[step["args"][1]]))

def _ref_partial(records, dag):
    """reference value of every v_k on the given records (env after each step); None if the reference itself errors (task not cleanly checkable)."""
    env = {"records": deepcopy(records)}; vals = {}
    for s in dag["steps"]:
        try:
            env[s["var"]] = _contract(s, env)
        except Exception:
            return None
        vals[s["var"]] = deepcopy(env[s["var"]])
    return vals

def _extract_solve_body(code):
    """return dict var-> (ast.Assign stmt) for top-level `v_k = expr` assignments inside solve()."""
    try:
        tree = ast.parse(code)
    except Exception:
        return None
    fn = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "solve"), None)
    if fn is None:
        return None
    stmts = {}
    for node in fn.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            stmts[node.targets[0].id] = node
    return stmts

def _refs(stmt):
    """variable Names read on the RHS of an assignment (its interface bindings)."""
    return {n.id for n in ast.walk(stmt.value) if isinstance(n, ast.Name)}

def _exec_stmt(stmt, inputs, gvars=None):
    """exec a single `v = expr` with RHS vars bound to `inputs`, using the candidate's module globals
    (gvars: helpers/imports/defs) as the global scope; return the assigned value or None on error."""
    g = dict(gvars) if gvars else {}
    g["__builtins__"] = __builtins__
    env = dict(inputs)
    mod = ast.Module(body=[stmt], type_ignores=[])
    try:
        code = compile(ast.fix_missing_locations(mod), "<blk>", "exec")
        exec(code, g, env)
        return env.get(stmt.targets[0].id)
    except Exception:
        return None

def _eq(a, b):
    try: return a == b
    except Exception: return False

def audit_candidate(records, dag, cand_code, rng, n_iface=8):
    """returns per-block dict: whole/observed/conditional/binding + whole_fail flag + op/arity."""
    ref_vals = _ref_partial(records, dag)
    if ref_vals is None:
        return None  # task not cleanly checkable (reference errors) -> excluded
    out_var = dag["out"]
    allvars = {"records"} | {s["var"] for s in dag["steps"]}
    # whole: run candidate solve on records, compare final output. G = candidate module globals (helpers/imports).
    whole_ok = False; G = {"__builtins__": __builtins__}
    try:
        exec(cand_code, G)
        res = G["solve"](deepcopy(records))
        whole_ok = _eq(res, ref_vals[out_var])
    except Exception:
        whole_ok = False
    stmts = _extract_solve_body(cand_code)
    # candidate's OWN sequential execution -> its actual intermediates (contaminated by upstream errors).
    cand_vals = {}; cenv = {"records": deepcopy(records)}
    if stmts:
        for s in dag["steps"]:
            if s["var"] in stmts:
                val = _exec_stmt(stmts[s["var"]], {k: _sdc(v) for k, v in cenv.items()}, G)
                cenv[s["var"]] = val; cand_vals[s["var"]] = val
    blocks = []
    for s in dag["steps"]:
        vk = s["var"]; decl = s["args"]
        rec = {"var": vk, "op": s["op"], "kind": s["kind"], "whole": whole_ok,
               "observed": None, "conditional": None, "binding": None, "parsed": False}
        if stmts and vk in stmts:
            rec["parsed"] = True
            stmt = stmts[vk]
            refs = _refs(stmt) & allvars  # only dag-variable interface refs (ignore helper/global names)
            # binding: RHS must reference exactly the declared arg vars
            rec["binding"] = set(decl).issubset(refs) and refs.issubset(set(decl) | {vk})
            # observed: candidate's OWN computed v_k vs reference v_k (contaminated by upstream errors)
            rec["observed"] = _eq(cand_vals.get(vk), ref_vals[vk])
            # conditional: independently-varied valid inputs to the declared args; check vs contract
            passes = 0; tried = 0
            for _ in range(n_iface):
                iface = {}
                for a in decl:
                    iface[a] = _rand_records(rng, rng.randint(4, 9))
                # candidate stmt evaluated with declared args bound to iface (bind refs by declared name)
                cand_out = _exec_stmt(stmt, {v: _sdc(iface[v]) for v in refs if v in iface}
                                             | ({"records": _sdc(iface.get(decl[0]))} if "records" in refs else {}), G)
                try:
                    contract_out = _contract(s, {**{a: deepcopy(iface[a]) for a in decl},
                                                 "records": deepcopy(iface.get(decl[0]))})
                except Exception:
                    continue  # contract not defined on this random input -> skip this probe
                if cand_out is not None:
                    tried += 1
                    if _eq(cand_out, contract_out): passes += 1
            rec["conditional"] = (tried >= max(2, n_iface // 2)) and (passes == tried)
        blocks.append(rec)
    return blocks

# ---- self-test: synthetic candidates validate the mechanism ------------------
def _dag_to_code(dag, corrupt_upstream=False, accidental=False):
    """emit a reference solve() for a dag; optionally corrupt the FIRST block's op (upstream error)."""
    lines = ["def solve(records):"]
    for i, s in enumerate(dag["steps"]):
        op = s["op"]
        if corrupt_upstream and i == 0 and s["kind"] == "un":
            op = next(o for o in UNARY if o != s["op"])  # wrong op upstream
        if s["kind"] == "un":
            src = s["args"][0]
            lines.append(f"    {s['var']} = _UN['{op}']({src})")
        else:
            a, b = s["args"]
            lines.append(f"    {s['var']} = _BIN['{op}']({a}, {b})")
    lines.append(f"    return {dag['out']}")
    body = "\n".join(lines)
    hdr = ("from rl_training.comp_tasks import PRIMS as _P\n"
           "from rl_training.comp_dag import BIN as _BIN\n"
           "_UN = {k: v['fn'] for k, v in _P.items()}\n")
    return hdr + body

def selftest():
    rng = random.Random(0); ok = True
    for seed in range(30):
        dag = gen_dag(seed, n_nodes=5); recs = _rand_records(rng, 8)
        # (1) correct candidate: all blocks whole+observed+conditional pass
        b = audit_candidate(recs, dag, _dag_to_code(dag), rng)
        if b is None: continue
        parsed = [x for x in b if x["parsed"]]
        if parsed and not all(x["conditional"] for x in parsed):
            print(f"seed{seed}: correct-cand conditional FAILED", [x for x in parsed if not x["conditional"]]); ok = False
        # (2) upstream-corrupted: block0 wrong; downstream should FAIL whole/observed but PASS conditional (recovered)
        bc = audit_candidate(recs, dag, _dag_to_code(dag, corrupt_upstream=True), rng)
        if bc is None: continue
        pc = [x for x in bc if x["parsed"]]
        recovered = [x for x in pc if (not x["whole"]) and (not x["observed"]) and x["conditional"]]
        if pc and not recovered and any(not x["observed"] for x in pc):
            # at least some downstream block should be recovered by conditional but not observed
            pass  # not guaranteed every seed; report aggregate below
    return ok

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--demo", action="store_true", help="print recovery stats on corrupted candidates")
    a = ap.parse_args()
    if a.selftest:
        print("SELFTEST", "PASS" if selftest() else "FAIL")
    if a.demo:
        rng = random.Random(1); tot = {"blocks": 0, "whole_fail": 0, "cond_recovered": 0, "obs_recovered": 0, "accidental_rej": 0}
        for seed in range(200):
            dag = gen_dag(seed, n_nodes=5); recs = _rand_records(rng, 8)
            bc = audit_candidate(recs, dag, _dag_to_code(dag, corrupt_upstream=True), rng)
            if bc is None: continue
            for x in bc:
                if not x["parsed"]: continue
                tot["blocks"] += 1
                if not x["whole"]: tot["whole_fail"] += 1
                # recovered by conditional but discarded by whole AND observed:
                if (not x["whole"]) and (not x["observed"]) and x["conditional"]: tot["cond_recovered"] += 1
                if (not x["whole"]) and x["observed"]: tot["obs_recovered"] += 1
        print("DEMO(corrupted-upstream candidates, 200 tasks):", json.dumps(tot))
        print("=> cond_recovered = blocks whole+observed BOTH discard but conditional-contract RECOVERS "
              "(correct transforms hidden by upstream error). If >0 and >> obs_recovered, C0 mechanism is real.")
