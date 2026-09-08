#!/usr/bin/env python3
"""ACV analysis (Accessibility rho, Competence c, Value/complementarity v) — offline, from existing
probe (c), route (rho logp) and stratified (per-sample) JSONs. Tests the new theory:
  1. QIII (harmful, dc<0) vs QII (benign, dc>=0): what distinguishes true erasure? (base c, base rho,
     functional uniqueness v).
  2. per-mode complementarity v_m: fraction of problems where forcing m is the ONLY strategy that
     solves (base). Correlate with routing suppression / competence change.
  3. v -> stratified-gain: per problem, does higher functional complementarity predict a positive
     stratified-minus-iid pass@k gain? (The ACV prediction for when diversity helps.)
"""
import json, os, sys, statistics as st

D = sys.argv[1] if len(sys.argv) > 1 else "rl_training/runs_pulled/probe_routing"
TAU = 0.25


def load(f):
    p = os.path.join(D, f)
    return json.load(open(p)) if os.path.exists(p) and os.path.getsize(p) > 100 else None


def c_index(d):
    idx = {}
    for p in d["per_problem"]:
        for m, fs in p["forced"].items():
            if fs:
                idx[(p["problem_id"], m)] = sum(x["correct"] for x in fs) / len(fs)
    return idx


def r_index(d):
    return {(p["problem_id"], m): p["route"][m]["logp_tok"] for p in d["per_problem"] for m in p["route"]}


def complementarity_v(cb, strategies, problem_ids):
    """v_m = frac of solvable problems where m is the ONLY strategy solving (c>=TAU). base competence."""
    uniq = {m: 0 for m in strategies}; solvable = 0
    for q in problem_ids:
        solvers = [m for m in strategies if cb.get((q, m), 0) >= TAU]
        if solvers:
            solvable += 1
            if len(solvers) == 1:
                uniq[solvers[0]] += 1
    return {m: uniq[m] / max(solvable, 1) for m in strategies}, solvable


def main():
    pb, pg = load("probe_probe_base_pfx.json"), load("probe_probe_grpo_pfx.json")
    rb, rg = load("route_route_base.json"), load("route_route_grpo.json")
    if not (pb and pg and rb and rg):
        print("missing qm probe/route files; have:", [f for f in os.listdir(D) if f.startswith(('probe_probe','route_route'))][:8]); return
    cb, cg = c_index(pb), c_index(pg)
    rbi, rgi = r_index(rb), r_index(rg)
    strategies = pb["strategies"]
    qids = sorted({q for (q, m) in cb})
    vmode, solvable = complementarity_v(cb, strategies, qids)

    # 1. QII vs QIII discriminant
    QII, QIII = [], []
    for k in set(cb) & set(cg) & set(rbi) & set(rgi):
        q, m = k
        dr = rgi[k] - rbi[k]; dc = cg[k] - cb[k]
        if dr >= -0.05:      # only routing-suppressed pairs
            continue
        rec = {"base_c": cb[k], "base_rho": rbi[k], "v": vmode[m], "dc": dc}
        (QII if dc >= 0 else QIII).append(rec)
    def mean(rows, key): return sum(r[key] for r in rows) / len(rows) if rows else 0.0
    print("=" * 78)
    print(f"1. QII (benign, dc>=0) vs QIII (harmful erasure, dc<0)  [routing-suppressed pairs]")
    print("=" * 78)
    print(f"  QII  n={len(QII):4}  base_c={mean(QII,'base_c'):.3f}  base_rho(logp/tok)={mean(QII,'base_rho'):.3f}  uniqueness_v={mean(QII,'v'):.4f}")
    print(f"  QIII n={len(QIII):4}  base_c={mean(QIII,'base_c'):.3f}  base_rho(logp/tok)={mean(QIII,'base_rho'):.3f}  uniqueness_v={mean(QIII,'v'):.4f}")
    print("  (ACV prediction: QIII has LOWER base_c and/or HIGHER uniqueness_v than QII)")

    # 2. per-mode complementarity vs competence change
    print("\n" + "=" * 78)
    print("2. per-mode functional complementarity v_m (frac problems only m solves, base)")
    print("=" * 78)
    dc_mode = {m: [] for m in strategies}
    for k in set(cb) & set(cg):
        dc_mode[k[1]].append(cg[k] - cb[k])
    print(f"  solvable problems (base, tau={TAU}): {solvable}")
    print(f"  {'strategy':16}{'v_m':>8}{'mean_dc':>9}")
    for m in sorted(strategies, key=lambda x: -vmode[x]):
        mdc = sum(dc_mode[m]) / len(dc_mode[m]) if dc_mode[m] else 0
        print(f"  {m:16}{vmode[m]:>8.4f}{mdc:>+9.3f}")
    print(f"  mean v across modes = {sum(vmode.values())/len(vmode):.4f}  (near 0 => modes redundant => diversity worthless)")

    # 3. v -> stratified gain per problem (from sp_qm_base samples)
    sp = load("sp_sp_qm_base.json")
    if sp:
        print("\n" + "=" * 78)
        print("3. per-problem complementarity vs stratified-minus-iid pass@8 gain (qm base)")
        print("=" * 78)
        import random
        def passk(samps, k, strat, trials=300):
            n=len(samps)
            if k>=n: return float(any(s["correct"] for s in samps))
            hit=0
            for _ in range(trials):
                if strat:
                    by={}
                    for i,s in enumerate(samps): by.setdefault(s["strategy"],[]).append(i)
                    gs=list(by.values()); random.shuffle(gs); picks=[]
                    pools=[list(g) for g in gs]
                    for pl in pools: random.shuffle(pl)
                    while len(picks)<k:
                        prog=False
                        for pl in pools:
                            if pl and len(picks)<k: picks.append(pl.pop()); prog=True
                        if not prog: break
                    idx=picks
                else:
                    idx=random.sample(range(n),k)
                if any(samps[i]["correct"] for i in idx): hit+=1
            return hit/trials
        gains=[]; comps=[]
        for p in sp["per_problem"]:
            s=p["samples"]
            # per-problem complementarity proxy: # distinct strategies among CORRECT samples
            corr_strats=len({x["strategy"] for x in s if x["correct"]})
            g=passk(s,8,True)-passk(s,8,False)
            gains.append(g); comps.append(corr_strats)
        # correlate
        import math
        n=len(gains); mg=sum(gains)/n; mc=sum(comps)/n
        cov=sum((gains[i]-mg)*(comps[i]-mc) for i in range(n))/n
        sg=math.sqrt(sum((x-mg)**2 for x in gains)/n); sc=math.sqrt(sum((x-mc)**2 for x in comps)/n)
        corr=cov/(sg*sc) if sg*sc>0 else 0
        print(f"  mean stratified-iid gain@8 = {mg:+.3f}  |  corr(distinct-correct-strategies, gain) = {corr:+.3f}")
        print("  (ACV prediction: gain rises with per-problem complementarity; if modes redundant, gain~0 everywhere)")


if __name__ == "__main__":
    main()
