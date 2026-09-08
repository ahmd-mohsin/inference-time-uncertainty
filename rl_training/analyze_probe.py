#!/usr/bin/env python3
"""Analyze routing-vs-competence probe outputs (local). Prints:
  1. Summary table per policy: default pass@1, routing rho, forced adherence, forced competence.
  2. Per-strategy competence delta vs base (which specific strategies lost/kept executability).
  3. Functional necessity/redundancy: per problem, strategies that SOLVE it when forced ->
     redundancy (how many strategies solve each solvable problem) + uniquely-necessary modes.
"""
import json, sys, os
from collections import defaultdict

DIR = sys.argv[1] if len(sys.argv) > 1 else "rl_training/runs_pulled/probe_routing"
FILES = {  # tag -> (policy, dataset)
    "probe_base": ("base", "Olympiad"), "probe_grpo": ("grpo", "Olympiad"),
    "probe_floor": ("floor", "Olympiad"), "probe_dphf": ("dphf", "Olympiad"),
    "probe_base_omni": ("base", "Omni"), "probe_grpo_omni": ("grpo", "Omni"),
    "probe_floor_omni": ("floor", "Omni"), "probe_dphf_omni": ("dphf", "Omni"),
    "probe_grpo_m500": ("grpo", "MATH500"),
}


def load(tag):
    fp = os.path.join(DIR, f"probe_{tag}.json")
    if not os.path.exists(fp) or os.path.getsize(fp) < 100:
        return None
    return json.load(open(fp))


def strat_competence(d):
    """per-strategy forced competence c(m) and adherence, averaged over problems."""
    strategies = d["strategies"]
    comp = {m: [] for m in strategies}
    adh = {m: [] for m in strategies}
    for p in d["per_problem"]:
        for m in strategies:
            f = p["forced"].get(m, [])
            if not f:
                continue
            comp[m].append(sum(x["correct"] for x in f) / len(f))
            adh[m].append(sum(x["adhered"] for x in f) / len(f))
    comp = {m: (sum(v) / len(v) if v else 0.0) for m, v in comp.items()}
    adh = {m: (sum(v) / len(v) if v else 0.0) for m, v in adh.items()}
    return comp, adh


def necessity(d, thresh=0.25):
    """Per problem: which strategies solve it (forced competence >= thresh). Returns:
      solvable problems, redundancy (mean #strategies that solve a solvable problem),
      per-strategy 'unique necessity' count (this strategy is the ONLY one solving the problem)."""
    strategies = d["strategies"]
    n_solvable = 0
    redundancy = []
    unique_need = {m: 0 for m in strategies}
    for p in d["per_problem"]:
        solvers = [m for m in strategies
                   if p["forced"].get(m) and (sum(x["correct"] for x in p["forced"][m]) / len(p["forced"][m])) >= thresh]
        if solvers:
            n_solvable += 1
            redundancy.append(len(solvers))
            if len(solvers) == 1:
                unique_need[solvers[0]] += 1
    return {"n_solvable": n_solvable,
            "mean_redundancy": (sum(redundancy) / len(redundancy) if redundancy else 0.0),
            "unique_need": unique_need,
            "n_uniquely_necessary_problems": sum(1 for r in redundancy if r == 1)}


def quadrant_map(dir_, probe_base, probe_grpo, route_base, route_grpo, dc_thresh=0.02, dr_thresh=0.05):
    """Exp #3 make-or-break map: for each (q,m), Delta log rho (grpo-base, routing) vs
    Delta c (grpo-base, competence). Classify into 4 quadrants. The hoped-for result: most
    'collapsed' (Delta rho << 0) pairs sit in QII (Delta c >= 0) = routing suppression w/o erasure."""
    pb, pg = load(probe_base), load(probe_grpo)
    rb = json.load(open(os.path.join(dir_, f"route_{route_base}.json")))
    rg = json.load(open(os.path.join(dir_, f"route_{route_grpo}.json")))
    if not (pb and pg):
        print("  (probe files missing — run prefix probes first)"); return
    # index competence + routing by (problem_id, strategy)
    def c_index(d):
        idx = {}
        for p in d["per_problem"]:
            for m, f in p["forced"].items():
                if f:
                    idx[(p["problem_id"], m)] = sum(x["correct"] for x in f) / len(f)
        return idx
    def r_index(d):
        return {(p["problem_id"], m): p["route"][m]["logp_tok"] for p in d["per_problem"] for m in p["route"]}
    cb, cg = c_index(pb), c_index(pg)
    rbi, rgi = r_index(rb), r_index(rg)
    quad = {"QI": 0, "QII": 0, "QIII": 0, "QIV": 0, "flat": 0}
    collapsed_pairs = 0; collapsed_in_QII = 0
    for k in set(cb) & set(cg) & set(rbi) & set(rgi):
        dr = rgi[k] - rbi[k]        # routing shift
        dc = cg[k] - cb[k]          # competence shift
        if abs(dr) < dr_thresh and abs(dc) < dc_thresh:
            quad["flat"] += 1; continue
        if dr > 0 and dc >= 0: quad["QI"] += 1
        elif dr < 0 and dc >= 0: quad["QII"] += 1
        elif dr < 0 and dc < 0: quad["QIII"] += 1
        else: quad["QIV"] += 1
        if dr <= -dr_thresh:  # routing-collapsed pairs
            collapsed_pairs += 1
            if dc >= 0: collapsed_in_QII += 1
    print("\n" + "=" * 92)
    print("EXP#3  Delta log rho (routing) vs Delta c (competence), grpo - base  [make-or-break map]")
    print("=" * 92)
    print(f"  QI  (rho+, c+ chosen more & better) : {quad['QI']}")
    print(f"  QII (rho-, c+ SUPPRESSED, competence kept/up) : {quad['QII']}   <-- the hoped-for bucket")
    print(f"  QIII(rho-, c- true erasure)        : {quad['QIII']}")
    print(f"  QIV (rho+, c- amplified despite worse) : {quad['QIV']}")
    print(f"  flat: {quad['flat']}")
    if collapsed_pairs:
        print(f"  of {collapsed_pairs} routing-collapsed pairs, {collapsed_in_QII} "
              f"({100*collapsed_in_QII/collapsed_pairs:.0f}%) kept/improved competence (QII)")


PFX = {  # high-adherence prefix probes -> (policy, dataset)
    "probe_base_pfx": ("base", "Olympiad"), "probe_grpo_pfx": ("grpo", "Olympiad"),
    "probe_floor_pfx": ("floor", "Olympiad"), "probe_dphf_pfx": ("dphf", "Olympiad"),
    "probe_base_omni_pfx": ("base", "Omni"), "probe_grpo_omni_pfx": ("grpo", "Omni"),
    "probe_floor_omni_pfx": ("floor", "Omni"), "probe_dphf_omni_pfx": ("dphf", "Omni"),
    "probe_grpo_m500_pfx": ("grpo", "MATH500"),
}


def _cmatrix(d, tau=0.25):
    """binary competence matrix rows=problems cols=strategies (c(m,q) >= tau)."""
    S = d["strategies"]; rows = []
    for p in d["per_problem"]:
        row = []
        for m in S:
            f = p["forced"].get(m, [])
            c = (sum(x["correct"] for x in f) / len(f)) if f else 0.0
            row.append(1 if c >= tau else 0)
        rows.append(row)
    return S, rows


def oracle_and_repertoire(dir_, eps=0.05, tau=0.25):
    """Oracle-router gap + minimal sufficient repertoire (greedy set-cover) + functional rank."""
    import math
    print("\n" + "=" * 92)
    print(f"OFFLINE: oracle-router gap / minimal repertoire / functional rank (tau={tau})")
    print("=" * 92)
    print(f"{'dataset':9} {'policy':6} {'default':>8} {'A_rand':>7} {'A_fixed':>8} {'A_oracle':>9} "
          f"{'gap':>6} {'|Smin|':>7} {'r_eff':>6}")
    for f, (pol, ds) in PFX.items():
        d = load(f)
        if d is None:
            continue
        S, C = _cmatrix(d, tau)
        n = len(C)
        if not n:
            continue
        # continuous c(m,q) for oracle/fixed/random
        Cc = []
        for p in d["per_problem"]:
            Cc.append([(sum(x["correct"] for x in p["forced"][m]) / len(p["forced"][m]))
                       if p["forced"].get(m) else 0.0 for m in S])
        A_default = sum(sum(x["correct"] for x in p["default"]) / len(p["default"])
                        for p in d["per_problem"] if p["default"]) / max(sum(1 for p in d["per_problem"] if p["default"]), 1)
        A_oracle = sum(max(row) for row in Cc) / n
        A_fixed = max(sum(row[j] for row in Cc) / n for j in range(len(S)))
        A_rand = sum(sum(row) / len(row) for row in Cc) / n
        # F(S)=frac problems covered (max binary over S). greedy min set to reach (1-eps)*F(all)
        F_all = sum(1 for row in C if max(row)) / n
        target = (1 - eps) * F_all
        chosen, covered = [], [False] * n
        cols = list(range(len(S)))
        while (sum(covered) / n) < target and cols:
            best = max(cols, key=lambda j: sum(1 for i in range(n) if C[i][j] and not covered[i]))
            gain = sum(1 for i in range(n) if C[i][best] and not covered[i])
            if gain == 0:
                break
            for i in range(n):
                if C[i][best]:
                    covered[i] = True
            chosen.append(S[best]); cols.remove(best)
        # functional rank r_eff from SVD of continuous competence matrix
        try:
            import numpy as np
            M = np.array(Cc)
            sv = np.linalg.svd(M, compute_uv=False)
            sv = sv[sv > 1e-9]; ps = sv / sv.sum()
            r_eff = math.exp(-(ps * np.log(ps)).sum())
        except Exception:
            r_eff = float("nan")
        print(f"{ds:9} {pol:6} {A_default:>8.3f} {A_rand:>7.3f} {A_fixed:>8.3f} {A_oracle:>9.3f} "
              f"{A_oracle-A_default:>+6.3f} {len(chosen):>7} {r_eff:>6.2f}")
    print("  gap = A_oracle - default = latent capability hidden by mis-routing (the routed-repertoire headroom)")


def main():
    data = {t: load(t) for t in FILES}
    print("=" * 92)
    print("1. SUMMARY  (rho=default routing mode-mass [regex proxy], c=forced competence)")
    print("=" * 92)
    print(f"{'dataset':9} {'policy':6} {'n_prob':>6} {'pass@1':>7} {'rho':>6} {'active/14':>9} {'adher':>6} {'COMPETENCE':>11}")
    for ds in ["Olympiad", "Omni", "MATH500"]:
        for t, (pol, d_ds) in FILES.items():
            if d_ds != ds or data[t] is None:
                continue
            s = data[t]["summary"]
            print(f"{ds:9} {pol:6} {data[t]['n_problems']:>6} {s['default_pass1']:>7.3f} "
                  f"{s['default_mode_mass_mean']:>6.3f} {s['routing_active_strategies']:>9} "
                  f"{s['forced_adherence_mean']:>6.3f} {s['forced_competence_mean']:>11.3f}")

    # 2. per-strategy competence delta grpo - base (Olympiad + Omni)
    for ds, bt, gt in [("Olympiad", "probe_base", "probe_grpo"), ("Omni", "probe_base_omni", "probe_grpo_omni")]:
        if data[bt] is None or data[gt] is None:
            continue
        cb, _ = strat_competence(data[bt]); cg, _ = strat_competence(data[gt])
        print("\n" + "=" * 92)
        print(f"2. PER-STRATEGY forced competence — {ds}   (delta = grpo - base; negative = erased)")
        print("=" * 92)
        print(f"{'strategy':16} {'base_c':>7} {'grpo_c':>7} {'delta':>7}")
        for m in sorted(cb, key=lambda k: cg[k] - cb[k]):
            print(f"{m:16} {cb[m]:>7.3f} {cg[m]:>7.3f} {cg[m]-cb[m]:>+7.3f}")

    # 3. functional necessity / redundancy
    print("\n" + "=" * 92)
    print("3. FUNCTIONAL NECESSITY / REDUNDANCY  (strategy solves problem if forced-c >= 0.25)")
    print("=" * 92)
    print(f"{'dataset':9} {'policy':6} {'solvable':>8} {'mean_redund':>11} {'uniq_nec_probs':>14}")
    for ds in ["Olympiad", "Omni", "MATH500"]:
        for t, (pol, d_ds) in FILES.items():
            if d_ds != ds or data[t] is None:
                continue
            n = necessity(data[t])
            print(f"{ds:9} {pol:6} {n['n_solvable']:>8} {n['mean_redundancy']:>11.2f} "
                  f"{n['n_uniquely_necessary_problems']:>14}")
    # which strategies are the uniquely-necessary ones (base, Olympiad)
    if data["probe_base"]:
        n = necessity(data["probe_base"])
        top = sorted(n["unique_need"].items(), key=lambda x: -x[1])[:6]
        print(f"\n  base/Olympiad uniquely-necessary strategies (count of problems only they solve):")
        print("   " + ", ".join(f"{m}:{c}" for m, c in top if c > 0))

    # EXP#3 quadrant map — needs prefix probes (probe_*_pfx) + routing (route_*). Auto-run if present.
    if os.path.exists(os.path.join(DIR, "route_route_grpo.json")) and \
       os.path.exists(os.path.join(DIR, "probe_probe_grpo_pfx.json")):
        quadrant_map(DIR, "probe_base_pfx", "probe_grpo_pfx", "route_base", "route_grpo")
    # OFFLINE oracle/repertoire/rank on whatever prefix probes are present
    if os.path.exists(os.path.join(DIR, "probe_probe_base_pfx.json")):
        oracle_and_repertoire(DIR)


if __name__ == "__main__":
    main()
