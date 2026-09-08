# CRPO causal matrix (Causal Decontamination RL, flagship MEASUREMENT — no training needed).
# Formalizes self-repair state s=(q,P,E): P=failed proposal, E=verifier evidence (counterexample).
# Two causal quantities, both measured on VERIFIER-OBSERVABLE repair behavior (distribution over
# per-test pass-vectors of K sampled repairs), so the metric is objective:
#
#   Proposal Direct Effect  DE_P = D_JS( pi(Y|q,E,P_A) || pi(Y|q,E,P_B) )     [hold evidence, vary proposal]  -> want 0
#   Evidence Effect         IE_E = D_JS( pi(Y|q,P,E_1) || pi(Y|q,P,E_2) )     [hold proposal, vary evidence]  -> want >>0
#   Causal Signal Ratio     CSR  = IE_E / (DE_P + eps)                          good repair: CSR>>1 ; anchored: CSR<<1
#
# Measured under two CONTEXT constructions of the same base model:
#   RAW  : repair prompt shows the failed code P + evidence E  (full-history / CEGIS-anchored state)
#   EVID : repair prompt shows only evidence E, code hidden    (proposal-erased state = Forget-to-Repair)
# Prediction: RAW has large DE_P (contamination: which failed code you show changes the repair even at
# fixed evidence); EVID drives DE_P->0 by construction WHILE retaining IE_E (does not discard signal).
# That is the causal argument for proposal-invariant state, quantified.
#
# DE_P needs two distinct failed proposals P_A,P_B that share a COMMON failing test (so evidence E is
# identical). IE_E needs one proposal with >=2 distinct failing tests (two different counterexamples E_1,E_2).
# Usage: python -m rl_training.causal_matrix --model-path <dir> --bench mbpp --tag cm_qc \
#          --shard-index S --num-shards 8   [--merge]
import argparse, json, math, os, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl_training.code_passk import load_bench, extract_code, run_tests, classify
from rl_training.seq_recover import base_task, chat
from rl_training.certify import _assert_pairs, _time_limit
from rl_training.rewards import _passvec

# ---------- verifier-observable repair distribution + JS divergence ----------
def _sig(code, test, entry, mbpp):
    """Signature of a repair = its per-test pass vector (tuple of 0/1). '<crash>' if it won't run/parse."""
    try:
        v = _passvec(code, test, entry, mbpp)
    except Exception:
        return ("<crash>",)
    return tuple(int(x) for x in v) if v else ("<empty>",)

def js_divergence(sigsA, sigsB):
    """Jensen-Shannon divergence (base 2, in [0,1]) between two empirical categorical distributions
    over repair pass-vector signatures."""
    ca, cb = Counter(sigsA), Counter(sigsB)
    na, nb = sum(ca.values()), sum(cb.values())
    if na == 0 or nb == 0:
        return float("nan")
    keys = set(ca) | set(cb)
    def _kl(p, q):
        s = 0.0
        for k in keys:
            pk = p.get(k, 0) / na
            if pk > 0:
                mk = 0.5 * (p.get(k, 0) / na + q.get(k, 0) / nb)
                s += pk * math.log2(pk / mk)
        return s
    return 0.5 * _kl(ca, cb) + 0.5 * _kl(cb, ca)

# ---------- evidence rendering ----------
def _ce_from_test(code, item, lhs, rhs):
    """Counterexample dict for ONE specific assert (lhs==rhs) of this item, evaluated against `code`."""
    ns = {}
    try:
        with _time_limit(5):
            exec(code, ns)
    except Exception as e:
        return {"input": lhs.strip()[:160], "got": f"<{type(e).__name__}>", "expected": rhs.strip()[:80]}
    entry = item.get("entry")
    if entry and entry in ns:
        ns.setdefault("candidate", ns[entry])
    try:
        with _time_limit(5):
            got = eval(lhs, ns)
        got = repr(got)[:120]
    except Exception as e:
        got = f"<raised {type(e).__name__}>"
    return {"input": lhs.strip()[:160], "got": got, "expected": rhs.strip()[:80]}

def evidence_str(ce):
    return (f"A previous attempt failed a test. On the call `{ce['input']}` it produced `{ce['got']}` "
            f"but the expected result is `{ce['expected']}`.")

def prompt_raw(mp, it, code, ce):
    fb = (f"Your previous attempt (below) failed.\n```python\n{code[:1500]}\n```\n{evidence_str(ce)}\n"
          f"Fix it and write a correct, complete solution in a ```python block.")
    return chat(mp, base_task(it) + "\n\n" + fb)

def prompt_evid(mp, it, ce):
    fb = (f"A previous attempt (hidden) failed. {evidence_str(ce)}\n"
          f"Diagnose the likely cause and write a correct, complete solution in a ```python block.")
    return chat(mp, base_task(it) + "\n\n" + fb)

def _failing_asserts(code, item, max_check=200):
    """Return list of (lhs,rhs) asserts that FAIL for `code` (evidence sources)."""
    out = []
    ns = {}
    try:
        with _time_limit(5):
            exec(code, ns)
    except Exception:
        return [(lhs, rhs) for lhs, rhs in _assert_pairs(item.get("test", ""))[:max_check]]
    entry = item.get("entry")
    if entry and entry in ns:
        ns.setdefault("candidate", ns[entry])
    for lhs, rhs in _assert_pairs(item.get("test", ""))[:max_check]:
        try:
            with _time_limit(5):
                got = eval(lhs, ns); exp = eval(rhs, ns)
            if got != exp:
                out.append((lhs, rhs))
        except Exception:
            out.append((lhs, rhs))
    return out

def run(a):
    from vllm import LLM, SamplingParams
    from rl_training.model_utils import merge_adapter_if_needed
    mp = merge_adapter_if_needed(a.model_path)
    items = load_bench(a.bench)
    if a.max_problems > 0: items = items[:a.max_problems]
    if a.num_shards > 1: items = items[a.shard_index::a.num_shards]
    llm = LLM(model=mp, dtype="bfloat16", trust_remote_code=True, tensor_parallel_size=1, max_model_len=4096,
              gpu_memory_utilization=float(os.environ.get("EVAL_GPU_MEM", 0.85)), enable_prefix_caching=True, enforce_eager=True)
    def gen(prompts, n=1, temp=1.0, mt=1024):
        sp = SamplingParams(n=n, temperature=temp, top_p=0.95, max_tokens=mt, stop=["<|im_end|>", "<|eot_id|>", "<|endoftext|>"])
        return llm.generate(prompts, sp)

    # 1) elicit a pool of failed proposals per problem
    pool = gen([chat(mp, base_task(it)) for it in items], a.n_pool)
    cases = []   # each: two failed proposals P_A,P_B sharing a common failing test + P_A has >=2 failing tests
    for it, o in zip(items, pool):
        test, entry, mbpp = it.get("test", ""), it.get("entry"), bool(it.get("mbpp"))
        fails = []
        for s in o.outputs:
            code = extract_code(s.text)
            ok, _ = run_tests(code, it, return_err=True)
            if not ok:
                fa = _failing_asserts(code, it)
                if fa: fails.append({"code": code, "fa": fa, "cls": classify(code)})
        if len(fails) < 2:
            continue
        # pick P_A with >=2 distinct failing asserts (for IE_E), then P_B (different code) sharing a
        # failing assert with P_A (for DE_P at fixed evidence)
        PA = next((f for f in fails if len({(l, r) for l, r in f["fa"]}) >= 2), None)
        if PA is None:
            continue
        pa_fa = {(l, r) for l, r in PA["fa"]}
        PB = next((f for f in fails if f["code"] != PA["code"] and pa_fa & {(l, r) for l, r in f["fa"]}), None)
        if PB is None:
            continue
        shared = list(pa_fa & {(l, r) for l, r in PB["fa"]})[0]           # common failing test -> identical E
        e1, e2 = PA["fa"][0], next(x for x in PA["fa"] if x != PA["fa"][0])  # two distinct E for same P_A
        cases.append({"it": it, "PA": PA["code"], "PB": PB["code"],
                      "shared": shared, "e1": e1, "e2": e2})
    if len(cases) < 3:
        Path(a.output_dir).mkdir(parents=True, exist_ok=True)
        json.dump({"tag": a.tag, "n": 0}, open(Path(a.output_dir) / f"cm_{a.tag}.shard{a.shard_index}-of-{a.num_shards}.json", "w"))
        print(f"[{a.tag} s{a.shard_index}] too few cases ({len(cases)})"); return

    # 2) build the 4 conditions per case; K repairs each
    K = a.k
    prompts, meta = [], []
    def add(kind, ci, it, code, lhs, rhs):
        ce = _ce_from_test(code, it, lhs, rhs)
        p = prompt_raw(mp, it, code, ce) if kind.startswith("raw") else prompt_evid(mp, it, ce)
        prompts.append(p); meta.append((kind, ci))
    for ci, c in enumerate(cases):
        it, (sl, sr) = c["it"], c["shared"]
        # DE_P: fixed evidence = shared test; vary proposal (raw shows code, evid hides it)
        add("raw_deP_A", ci, it, c["PA"], sl, sr)
        add("raw_deP_B", ci, it, c["PB"], sl, sr)
        add("evid_deP_A", ci, it, c["PA"], sl, sr)   # evid: same E -> identical prompt for A and B (invariant by construction)
        # IE_E: fixed proposal P_A; vary evidence e1 vs e2
        add("raw_ieE_1", ci, it, c["PA"], c["e1"][0], c["e1"][1])
        add("raw_ieE_2", ci, it, c["PA"], c["e2"][0], c["e2"][1])
        add("evid_ieE_1", ci, it, c["PA"], c["e1"][0], c["e1"][1])
        add("evid_ieE_2", ci, it, c["PA"], c["e2"][0], c["e2"][1])
    outs = gen(prompts, K)
    # collect signatures per (kind,ci)
    bucket = {}
    for (kind, ci), o in zip(meta, outs):
        it = cases[ci]["it"]
        test, entry, mbpp = it.get("test", ""), it.get("entry"), bool(it.get("mbpp"))
        sigs = [_sig(extract_code(s.text), test, entry, mbpp) for s in o.outputs]
        solved = [1 if all(x == 1 for x in sg) else 0 for sg in sigs if sg and sg[0] != "<crash>" and "<" not in str(sg[0])]
        bucket[(kind, ci)] = {"sigs": sigs, "solve": sum(1 for sg in sigs if sg and all(x == 1 for x in sg)) / max(len(sigs), 1)}

    # 3) per-case DE_P / IE_E under raw and evid; then average
    rows = []
    for ci in range(len(cases)):
        def sg(k): return bucket.get((k, ci), {}).get("sigs", [])
        deP_raw = js_divergence(sg("raw_deP_A"), sg("raw_deP_B"))
        deP_evid = 0.0  # evid prompt for A and B at shared E is byte-identical -> distribution invariant by construction
        ieE_raw = js_divergence(sg("raw_ieE_1"), sg("raw_ieE_2"))
        ieE_evid = js_divergence(sg("evid_ieE_1"), sg("evid_ieE_2"))
        rows.append({"deP_raw": deP_raw, "deP_evid": deP_evid, "ieE_raw": ieE_raw, "ieE_evid": ieE_evid,
                     "solve_raw_A": bucket[("raw_deP_A", ci)]["solve"], "solve_evid_A": bucket[("evid_deP_A", ci)]["solve"]})
    out = {"tag": a.tag, "model": mp, "bench": a.bench, "n_cases": len(cases), "k": K, "per_case": rows}
    Path(a.output_dir).mkdir(parents=True, exist_ok=True)
    fp = Path(a.output_dir) / (f"cm_{a.tag}.shard{a.shard_index}-of-{a.num_shards}.json" if a.num_shards > 1 else f"cm_{a.tag}.json")
    json.dump(out, open(fp, "w"))
    import statistics as st
    def m(k):
        v = [r[k] for r in rows if r[k] == r[k]]
        return st.mean(v) if v else float("nan")
    print(f"[{a.tag} s{a.shard_index}] cases={len(cases)} DE_P_raw={m('deP_raw'):.3f} IE_E_raw={m('ieE_raw'):.3f} "
          f"IE_E_evid={m('ieE_evid'):.3f} CSR_raw={m('ieE_raw')/(m('deP_raw')+1e-6):.2f}")

def merge(a):
    import statistics as st
    rows = []
    for s in range(a.num_shards):
        fp = Path(a.output_dir) / f"cm_{a.tag}.shard{s}-of-{a.num_shards}.json"
        if not fp.exists(): continue
        d = json.load(open(fp))
        if d.get("n_cases"): rows += d["per_case"]
    def m(k):
        v = [r[k] for r in rows if r.get(k) == r.get(k) and r.get(k) is not None]
        return st.mean(v) if v else float("nan")
    eps = 1e-6
    out = {"tag": a.tag, "n_cases": len(rows),
           "DE_P_raw": m("deP_raw"), "DE_P_evid": m("deP_evid"),
           "IE_E_raw": m("ieE_raw"), "IE_E_evid": m("ieE_evid"),
           "CSR_raw": m("ieE_raw") / (m("deP_raw") + eps),
           "CSR_evid": m("ieE_evid") / (m("deP_evid") + eps),
           "solve_raw": m("solve_raw_A"), "solve_evid": m("solve_evid_A")}
    json.dump(out, open(Path(a.output_dir) / f"cm_{a.tag}.json", "w"))
    print(f"[{a.tag}] n={out['n_cases']} | DE_P raw={out['DE_P_raw']:.3f} evid={out['DE_P_evid']:.3f} | "
          f"IE_E raw={out['IE_E_raw']:.3f} evid={out['IE_E_evid']:.3f} | CSR raw={out['CSR_raw']:.2f} evid={out['CSR_evid']:.1f} | "
          f"solve raw={out['solve_raw']:.3f} evid={out['solve_evid']:.3f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path"); ap.add_argument("--bench", default="mbpp")
    ap.add_argument("--max-problems", type=int, default=-1)
    ap.add_argument("--n-pool", type=int, default=10, help="samples/problem to elicit two failed proposals")
    ap.add_argument("--k", type=int, default=8, help="repairs per condition (repair distribution)")
    ap.add_argument("--output-dir", default="/tmp/instance_storage/gu/eval_out"); ap.add_argument("--tag", default="cm")
    ap.add_argument("--shard-index", type=int, default=0); ap.add_argument("--num-shards", type=int, default=1); ap.add_argument("--merge", action="store_true")
    a = ap.parse_args(); merge(a) if a.merge else run(a)

if __name__ == "__main__":
    main()
