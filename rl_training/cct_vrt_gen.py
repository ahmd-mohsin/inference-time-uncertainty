"""VRT bank builder (data-level relational supervision, §147 forward plan). DAG-native, reference-generated (no self-RFT
confound). Emits four EQUAL-BUDGET training banks + held-out eval pools:
  A whole      : reference solve() completions for DAG tasks.
  B +rename    : A plus harmless-rename counterparts (vars consistently renamed in prompt+code; semantics preserved).
  F +edit      : A plus meaningful-edit counterparts (one binding arg changed -> a DIFFERENT verified reference solve).
  G relational : A plus BOTH rename and edit counterparts (the combined relational supervision).
The relational hypothesis is realized as DATA (train the model to keep code invariant under renames, and to CHANGE the
dependency under meaningful edits), per the review's "supervision construction, not a new loss." Standalone inline
reference code (restricted to inline-able ops) so completions are valid self-contained SFT targets. CPU-only.
"""
import argparse, json, random, re, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl_training.comp_tasks import _rand_records, PRIMS
from rl_training.comp_dag import gen_dag, run_dag, _STEP_PHRASE, BIN, BIN_DESC
from rl_training.cct_c0 import audit_candidate

# inline-able ops only (single-line PRIMS code + 2 simple BIN)
INLINE_UN = ["filter_ok","filter_val_pos","filter_grpA","sort_val","sort_ts","reverse","take2","take3","drop1","map_val_inc","map_val_abs","sum_val"]
INLINE_BIN = {"concat":"{a} + {b}",
              "zip_sum":"[{{**x, 'val': x.get('val',0)+y.get('val',0)}} for x, y in zip({a}, {b})]"}

def _un_expr(op, arg):
    # PRIMS code is "r = EXPR"; take EXPR, replace standalone list var r -> arg
    code = PRIMS[op]["code"]
    rhs = code.split("=", 1)[1].strip()
    return re.sub(r"\br\b", arg, rhs)

def gen_inline_dag(seed, n_nodes=5):
    rng = random.Random(seed); steps=[]; nvars=["records"]
    for i in range(n_nodes):
        v=f"v{i+1}"
        if len(nvars)>=2 and rng.random()<0.4:
            a,b=rng.sample(nvars,2); op=rng.choice(list(INLINE_BIN))
            steps.append({"var":v,"kind":"bin","op":op,"args":[a,b]})
        else:
            src=rng.choice(nvars); op=rng.choice(INLINE_UN)
            steps.append({"var":v,"kind":"un","op":op,"args":[src]})
        nvars.append(v)
    return {"steps":steps,"out":nvars[-1]}

def ref_code(dag):
    lines=["def solve(records):"]
    for s in dag["steps"]:
        if s["kind"]=="un":
            lines.append(f"    {s['var']} = {_un_expr(s['op'], s['args'][0])}")
        else:
            a,b=s["args"]; lines.append(f"    {s['var']} = {INLINE_BIN[s['op']].format(a=a,b=b)}")
    lines.append(f"    return {dag['out']}")
    return "\n".join(lines)

def render_dag(dag):
    lines=[]
    for s in dag["steps"]:
        if s["kind"]=="un": lines.append(f"{s['var']} = ({_STEP_PHRASE[s['op']]}) applied to {s['args'][0]}")
        else: lines.append(f"{s['var']} = "+BIN_DESC[s['op']].format(a=s['args'][0],b=s['args'][1]))
    return ("You are given a list of records (dicts with keys id, grp, val, ok, ts). Compute these named intermediate "
            "lists IN ORDER (data-flow is a DAG):\n"+"\n".join(lines)+
            f"\nReturn `def solve(records):` that computes these and returns {dag['out']}. Return ONLY a ```python code block.")

def _rename_map(dag, rng):
    vs=[s["var"] for s in dag["steps"]]
    pool=[f"t{i}" for i in range(1,40)]; rng.shuffle(pool)
    return {v:pool[i] for i,v in enumerate(vs)}

def apply_rename(dag, m):
    def r(x): return m.get(x,x)
    steps=[{"var":r(s["var"]),"kind":s["kind"],"op":s["op"],"args":[r(a) for a in s["args"]]} for s in dag["steps"]]
    return {"steps":steps,"out":r(dag["out"])}

def meaningful_edit(dag, rng):
    """change ONE block's binding arg to a different earlier var -> a different task (dependency changed)."""
    idxs=[i for i,s in enumerate(dag["steps"]) if i>=1]
    rng.shuffle(idxs)
    for i in idxs:
        s=dag["steps"][i]; earlier=[st["var"] for st in dag["steps"][:i]]+["records"]
        for ai,a in enumerate(s["args"]):
            alts=[e for e in earlier if e!=a]
            if alts:
                e2=dict(dag); steps=[dict(x) for x in dag["steps"]]
                steps[i]=dict(s); steps[i]["args"]=list(s["args"]); steps[i]["args"][ai]=rng.choice(alts)
                e2["steps"]=steps; return e2, i
    return None, None

def _fenced(code): return "```python\n"+code+"\n```"

def _verify(dag, recs):
    try:
        b=audit_candidate(recs, dag, ref_code(dag), random.Random(0))
        return b is not None and b[0]["whole"]
    except Exception:
        return False

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--seed0",type=int,default=0); ap.add_argument("--n-tasks",type=int,default=400)
    ap.add_argument("--n-nodes",type=int,default=5); ap.add_argument("--outdir",required=True)
    ap.add_argument("--eval",action="store_true",help="emit held-out eval pools instead of train banks")
    a=ap.parse_args(); os.makedirs(a.outdir,exist_ok=True)
    rng=random.Random(a.seed0*17+1)
    A,rename,edit=[],[],[]
    edit_pairs=[]
    for s in range(a.seed0, a.seed0+a.n_tasks):
        dag=gen_inline_dag(s, a.n_nodes); recs=_rand_records(rng, rng.randint(6,10))
        if not _verify(dag, recs): continue
        A.append({"prompt":render_dag(dag),"completion":_fenced(ref_code(dag))})
        m=_rename_map(dag,rng); d2=apply_rename(dag,m)
        if _verify(d2,recs): rename.append({"prompt":render_dag(d2),"completion":_fenced(ref_code(d2))})
        de,ei=meaningful_edit(dag,rng)
        if de and _verify(de,recs):
            edit.append({"prompt":render_dag(de),"completion":_fenced(ref_code(de))})
            edit_pairs.append({"orig_prompt":render_dag(dag),"edit_prompt":render_dag(de),
                               "orig_completion":_fenced(ref_code(dag)),"edit_completion":_fenced(ref_code(de))})
    def dump(name,rows):
        with open(os.path.join(a.outdir,name),"w") as f:
            for r in rows: f.write(json.dumps(r)+"\n")
    if a.eval:
        dump("eval_whole.jsonl",A); dump("eval_editpairs.jsonl",edit_pairs)
        print(f"[vrt_gen EVAL] whole={len(A)} edit_pairs={len(edit_pairs)} -> {a.outdir}")
        return
    # equal budget across arms: base N whole; augmented arms add counterparts but we cap total to 2N so budget comparable
    dump("A_whole.jsonl", A)
    dump("rename.jsonl", rename); dump("edit.jsonl", edit)
    dump("edit_pairs.jsonl", edit_pairs)
    print(f"[vrt_gen] whole={len(A)} rename={len(rename)} edit={len(edit)} pairs={len(edit_pairs)} -> {a.outdir}")

if __name__=="__main__":
    main()
