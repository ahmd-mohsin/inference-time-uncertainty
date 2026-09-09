# Compiler-backed COMPOSITIONAL task generator (§64 directive). Domain: data-processing pipelines over lists of
# records (list[dict]). ~16 typed primitives + routing {sequential, reuse-intermediate, branch, combine-two}.
# A task = a pipeline PROGRAM; the model must emit `def solve(records): ...` which is EXECUTED on stored held-out
# inputs and compared to the reference executor's output (exact structural match). Ground truth is executable.
#
# Core object: NONCOMMUTING primitive pairs (dedup∘filter ≠ filter∘dedup, sort∘take ≠ take∘sort, filter∘agg ≠ agg∘filter).
# Primitives are TOTAL (schema-safe .get defaults) so no pipeline crashes -> a crash never confounds the RL reward.
# Each task stores TEST_INPUTS; for DIAGNOSTIC tasks a majority of them provably SEPARATE the ordered pair (so a
# wrong/swapped solution fails), making the task genuinely discriminating.
#
# Pools (matched count/depth/primitive-universe):
#   A repeated-relationships : embedded noncommuting pairs kept in canonical (train) order
#   B random-new-compositions: random routings
#   C diagnostic-compositions: routings that FORCE a noncommuting pair to separate on the stored test inputs
import argparse, json, random, re, signal
from copy import deepcopy

GRPS = list("ABCDE")

def _rand_records(rng, n):
    return [{"id": rng.randint(1, 12), "grp": rng.choice(GRPS), "val": rng.randint(-9, 20),
             "ok": rng.random() < 0.6, "ts": rng.randint(0, 30)} for _ in range(n)]

# ---- primitives (TOTAL: .get defaults so pipelines never crash) ------------
def _p(fn, code, kind): return {"fn": fn, "code": code, "kind": kind}

PRIMS = {
    "filter_ok":      _p(lambda r: [x for x in r if x.get("ok", False)],
                         "r = [x for x in r if x.get('ok', False)]", "filter"),
    "filter_val_pos": _p(lambda r: [x for x in r if x.get("val", 0) > 0],
                         "r = [x for x in r if x.get('val', 0) > 0]", "filter"),
    "filter_grpA":    _p(lambda r: [x for x in r if x.get("grp") == "A"],
                         "r = [x for x in r if x.get('grp') == 'A']", "filter"),
    "dedup_id":       _p(lambda r: _dedup(r, "id"),
                         "seen=set(); r=[x for x in r if not (x.get('id') in seen or seen.add(x.get('id')))]", "dedup"),
    "dedup_grp":      _p(lambda r: _dedup(r, "grp"),
                         "seen=set(); r=[x for x in r if not (x.get('grp') in seen or seen.add(x.get('grp')))]", "dedup"),
    "sort_val":       _p(lambda r: sorted(r, key=lambda x: x.get("val", 0)),
                         "r = sorted(r, key=lambda x: x.get('val', 0))", "sort"),
    "sort_ts":        _p(lambda r: sorted(r, key=lambda x: x.get("ts", 0)),
                         "r = sorted(r, key=lambda x: x.get('ts', 0))", "sort"),
    "reverse":        _p(lambda r: list(reversed(r)), "r = list(reversed(r))", "order"),
    "take2":          _p(lambda r: r[:2], "r = r[:2]", "slice"),
    "take3":          _p(lambda r: r[:3], "r = r[:3]", "slice"),
    "drop1":          _p(lambda r: r[1:], "r = r[1:]", "slice"),
    "map_val_inc":    _p(lambda r: [{**x, "val": x.get("val", 0) + 1} for x in r],
                         "r = [{**x, 'val': x.get('val', 0) + 1} for x in r]", "map"),
    "map_val_abs":    _p(lambda r: [{**x, "val": abs(x.get("val", 0))} for x in r],
                         "r = [{**x, 'val': abs(x.get('val', 0))} for x in r]", "map"),
    "group_sum_val":  _p(lambda r: _group_agg(r, sum),
                         "from collections import OrderedDict as _OD\n_g=_OD()\nfor x in r: _g.setdefault(x.get('grp'),[]).append(x.get('val',0))\nr=[{'grp':k,'val':sum(v)} for k,v in _g.items()]", "agg"),
    "group_count":    _p(lambda r: _group_cnt(r),
                         "from collections import OrderedDict as _OD\n_g=_OD()\nfor x in r: _g[x.get('grp')]=_g.get(x.get('grp'),0)+1\nr=[{'grp':k,'val':v} for k,v in _g.items()]", "agg"),
    "sum_val":        _p(lambda r: [{"val": sum(x.get("val", 0) for x in r)}],
                         "r = [{'val': sum(x.get('val', 0) for x in r)}]", "reduce"),
}
PRIM_NAMES = list(PRIMS)

def _dedup(r, key):
    seen = set(); out = []
    for x in r:
        k = x.get(key)
        if k not in seen: seen.add(k); out.append(x)
    return out

def _group_agg(r, agg):
    from collections import OrderedDict
    g = OrderedDict()
    for x in r: g.setdefault(x.get("grp"), []).append(x.get("val", 0))
    return [{"grp": k, "val": agg(v)} for k, v in g.items()]

def _group_cnt(r):
    from collections import OrderedDict
    g = OrderedDict()
    for x in r: g[x.get("grp")] = g.get(x.get("grp"), 0) + 1
    return [{"grp": k, "val": v} for k, v in g.items()]

NONCOMMUTING = [
    ("dedup_id", "filter_ok"), ("dedup_id", "filter_val_pos"), ("dedup_grp", "filter_grpA"),
    ("sort_val", "take2"), ("sort_val", "take3"), ("sort_ts", "take2"),
    ("filter_val_pos", "group_sum_val"), ("filter_ok", "group_count"),
    ("map_val_inc", "filter_val_pos"),
]

def run_program(records, prog):
    r = deepcopy(records)
    for name in prog: r = PRIMS[name]["fn"](r)
    return r

def reference_solution(prog):
    lines = ["def solve(records):", "    r = [dict(x) for x in records]"]
    for name in prog:
        for ln in PRIMS[name]["code"].split("\n"): lines.append("    " + ln)
    lines.append("    return r")
    return "\n".join(lines)

_STEP_PHRASE = {
    "filter_ok": "keep only records where ok is true", "filter_val_pos": "keep only records with val > 0",
    "filter_grpA": "keep only records whose grp == 'A'",
    "dedup_id": "stably remove later records repeating an already-seen id (keep the first)",
    "dedup_grp": "stably keep the first record of each grp",
    "sort_val": "sort by val ascending (stable)", "sort_ts": "sort by ts ascending (stable)",
    "reverse": "reverse the list order", "take2": "keep the first 2 records", "take3": "keep the first 3 records",
    "drop1": "drop the first record", "map_val_inc": "add 1 to each record's val",
    "map_val_abs": "replace each val with its absolute value",
    "group_sum_val": "group by grp (first-seen order) and sum val into records {grp, val}",
    "group_count": "group by grp (first-seen order) and count into records {grp, val}",
    "sum_val": "reduce to a single record {val: total of all val}",
}

def render_prompt(records, prog):
    steps = "; then ".join(_STEP_PHRASE[n] for n in prog)
    return ("You are given a list of records (dicts with keys id, grp, val, ok, ts). Write a Python function "
            f"`solve(records)` that applies this pipeline IN ORDER: {steps}. Order matters. Return the resulting "
            "list. Return ONLY a ```python code block defining `solve`.\n"
            f"Example: records = {json.dumps(records)}")

# ---- verifier (uses the task's STORED test inputs) -------------------------
class _TO(Exception): pass
def _to(s, f): raise _TO()
def _extract(text):
    m = re.findall(r"```(?:python)?\s*(.*?)```", text, re.S)
    return m[-1] if m else text

def verify_solution(text, task, timeout=5):
    src = _extract(text); ns = {}
    try:
        signal.signal(signal.SIGALRM, _to); signal.alarm(timeout)
        exec(src, ns); solve = ns.get("solve")
        if not callable(solve): signal.alarm(0); return False
        for inp in task["test_inputs"]:
            try: got = solve(deepcopy(inp))
            except Exception: signal.alarm(0); return False
            if got != run_program(inp, task["prog"]): signal.alarm(0); return False
        signal.alarm(0); return True
    except Exception:
        try: signal.alarm(0)
        except Exception: pass
        return False

# ---- test-input construction ----------------------------------------------
def _sample_inputs(rng, n_inp=6, nrec=(6, 10)):
    return [_rand_records(rng, rng.randint(*nrec)) for _ in range(n_inp)]

def _separating_inputs(prog, pair, seed, n_want=6, pool=60):
    """Inputs where run(prog) with the pair in given order != run(prog with the pair swapped). Ensures the task
    genuinely tests the interaction (a swapped-order solution fails on >=half)."""
    rng = random.Random(seed * 131 + 5)
    swp = _swap_pair(prog, pair)
    sep, other = [], []
    for _ in range(pool):
        inp = _rand_records(rng, rng.randint(6, 10))
        (sep if run_program(inp, prog) != run_program(inp, swp) else other).append(inp)
        if len(sep) >= n_want: break
    # require majority separating; pad with a few non-separating for realism
    out = sep[:n_want]
    return out if len(out) >= max(2, n_want // 2) else None

def _swap_pair(prog, pair):
    p, q = pair; out = list(prog)
    for j in range(len(out) - 1):
        if out[j] == p and out[j + 1] == q: out[j], out[j + 1] = q, p; break
    return out

# ---- tasks + pools ---------------------------------------------------------
def gen_task(seed, depth=3, prog=None, kind="generic"):
    rng = random.Random(seed)
    if prog is None: prog = [rng.choice(PRIM_NAMES) for _ in range(depth)]
    recs = _rand_records(rng, rng.randint(6, 10))
    return {"seed": seed, "prog": prog, "records": recs, "prompt": render_prompt(recs, prog),
            "reference": reference_solution(prog), "test_inputs": _sample_inputs(random.Random(seed * 131 + 5)),
            "kind": kind, "depth": len(prog)}

def build_pool(which, n_tasks, depth=3, seed0=0):
    tasks = []; i = 0
    while len(tasks) < n_tasks:
        s = seed0 + i; i += 1; rng = random.Random(s)
        if which == "C":
            pair = NONCOMMUTING[rng.randrange(len(NONCOMMUTING))]
            n_distract = max(0, depth - 2)
            distract = [rng.choice(PRIM_NAMES) for _ in range(n_distract)]
            prog = distract + list(pair)                    # pair kept adjacent + ordered = the diagnostic
            ti = _separating_inputs(prog, pair, s)
            if ti is None: continue
            recs = ti[0]
            tasks.append({"seed": s, "prog": prog, "records": recs, "prompt": render_prompt(recs, prog),
                          "reference": reference_solution(prog), "test_inputs": ti,
                          "kind": "diagnostic:%s>%s" % pair, "depth": len(prog)})
        elif which == "A":
            prog = _canonicalize_pairs([rng.choice(PRIM_NAMES) for _ in range(depth)])
            tasks.append(gen_task(s, depth=depth, prog=prog, kind="repeated"))
        else:
            tasks.append(gen_task(s, depth=depth, kind="random"))
    return tasks

def _canonicalize_pairs(prog):
    canon = {}
    for (p, q) in NONCOMMUTING: canon[(p, q)] = True; canon[(q, p)] = False
    out = list(prog)
    for j in range(len(out) - 1):
        a, b = out[j], out[j + 1]
        if canon.get((a, b)) is False and canon.get((b, a)) is True:
            out[j], out[j + 1] = b, a
    return out

def build_matched_pools(n_tasks, depth=3, seed0=0):
    """Return (A,B,C) with MATCHED aggregate primitive frequency so the ONLY systematic difference is ARRANGEMENT
    (adjacency/order of noncommuting pairs). C is built first (its diagnostics fix the primitive multiset); A and B
    draw primitives from C's empirical frequency, then A canonicalizes any noncommuting pair (train order, non-
    separating) while B permutes randomly. Residual per-task interaction differences are the intervention, not a leak."""
    import collections, bisect
    C = build_pool("C", n_tasks, depth=depth, seed0=seed0)
    freq = collections.Counter()
    for t in C:
        for p in t["prog"]: freq[p] += 1
    names = list(freq); weights = [freq[n] for n in names]
    cum = []; s = 0
    for w in weights: s += w; cum.append(s)
    def draw(rng): return names[bisect.bisect_left(cum, rng.random() * s)]
    A, B = [], []
    for i in range(n_tasks):
        rng = random.Random(800000 + seed0 + i)
        base = [draw(rng) for _ in range(depth)]          # SAME multiset for the A/B pair
        B.append(gen_task(800000 + seed0 + i, depth=depth, prog=list(base), kind="random"))
        A.append(gen_task(900000 + seed0 + i, depth=depth, prog=_canonicalize_pairs(base), kind="repeated"))
    return A, B, C

def _units(prog):
    """Coverage units of a program: each primitive + each ADJACENT ordered pair (the interaction contexts)."""
    u = set(prog)
    for i in range(len(prog) - 1):
        u.add((prog[i], prog[i + 1]))
    return u

def build_setcover_pool(n_tasks, depth=5, seed0=0, cand_mult=6):
    """BET B: greedily SELECT compositions to maximize coverage of (primitive + adjacent-ordered-pair) units — a
    weighted set-cover curriculum. Compared against RANDOM selection from the SAME candidate universe at matched n.
    If coverage-selection > random at matched budget, the surviving coverage effect (§66d) becomes a validated selector."""
    rng = random.Random(seed0)
    cand = [gen_task(seed0 + i, depth=depth) for i in range(n_tasks * cand_mult)]
    covered = set(); chosen = []; pool = list(range(len(cand)))
    while len(chosen) < n_tasks and pool:
        best_i, best_gain = None, -1
        for i in pool:
            g = len(_units(cand[i]["prog"]) - covered)
            if g > best_gain: best_gain, best_i = g, i
        if best_gain <= 0:  # everything covered — reset to keep filling with fresh coverage cycles
            covered = set(); continue
        covered |= _units(cand[best_i]["prog"]); chosen.append(cand[best_i]); pool.remove(best_i)
    # top up if needed
    while len(chosen) < n_tasks and pool:
        chosen.append(cand[pool.pop(0)])
    return chosen[:n_tasks]

def component_bank(n_per_prim=40, seed0=10000):
    rows = []; s = seed0
    for name in PRIM_NAMES:
        for _ in range(n_per_prim):
            t = gen_task(s, prog=[name], kind="component"); s += 1
            rows.append({"prompt": t["prompt"], "completion": "```python\n" + t["reference"] + "\n```", "prog": t["prog"]})
    return rows

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", choices=["A", "B", "C", "components", "selftest"], default="selftest")
    ap.add_argument("--n", type=int, default=200); ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=0); ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.emit == "selftest":
        pool = build_pool("C", 6, depth=3)
        for t in pool:
            ok = verify_solution("```python\n" + t["reference"] + "\n```", t)
            wrong = reference_solution(_swap_pair(t["prog"], tuple(t["kind"].split(":")[1].split(">"))))
            bad = verify_solution("```python\n" + wrong + "\n```", t)
            print(f"[C] {t['kind']:34} ref_ok={ok} swapped_ok={bad}  (want True / False)")
        print("components covered:", len(component_bank(n_per_prim=1)), "/", len(PRIM_NAMES))
    elif a.emit == "components":
        rows = component_bank(); out = a.out or "/tmp/comp_components.jsonl"
        open(out, "w").write("\n".join(json.dumps(r) for r in rows))
        print(f"wrote {len(rows)} -> {out}")
    else:
        pool = build_pool(a.emit, a.n, depth=a.depth, seed0=a.seed0)
        out = a.out or f"/tmp/comp_pool_{a.emit}.jsonl"
        with open(out, "w") as f:
            for t in pool: f.write(json.dumps({k: t[k] for k in ("prompt", "prog", "kind", "seed", "reference", "test_inputs")}) + "\n")
        print(f"wrote {len(pool)} pool-{a.emit} -> {out}")
