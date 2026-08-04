#!/usr/bin/env python3
"""Lost-set retention + fragile-band + matched-pass@1 analysis (Tier-A metrics).

The aggregate pass@k dilutes the signal (~90% of problems saturated). The RIGHT metrics
(professor review):
  1. LOST SET: problems BASE solves at k but GRPO does NOT. Of those, what fraction does each
     method retain? McNemar/sign test on this paired binary outcome. Effect size 10-20x the mean.
  2. Fragile-band-only pass@k (base pass@1 <= 0.10).
  3. Per-problem paired deltas.

Runs entirely on local eval shard JSONs (per_problem: problem_id, n_correct, [n_samples]).
"""
import json, glob, sys
from math import comb


def load_arm(evaldir, tag, n_default):
    """Merge shard files -> {problem_id: n_correct}, and n_samples."""
    d = {}
    ns = n_default
    for f in sorted(glob.glob(f"{evaldir}/passk_{tag}.shard*.json")):
        o = json.load(open(f))
        for p in o.get("per_problem", []):
            d[p["problem_id"]] = p["n_correct"]
            ns = p.get("n_samples", ns)
    return d, ns


def passk(nc, n, k):
    if nc <= 0: return 0.0
    if n - nc < k: return 1.0
    return 1.0 - comb(n - nc, k) / comb(n, k)


def solves_at_k(nc, n, k, thresh=0.5):
    """Binary: does this problem count as 'solved' at budget k? (pass@k >= thresh)."""
    return passk(nc, n, k) >= thresh


def mcnemar(b, c):
    """Exact McNemar (binomial) on discordant pairs b, c. Returns two-sided p."""
    from math import comb as C
    n = b + c
    if n == 0: return 1.0
    x = min(b, c)
    p = sum(C(n, i) for i in range(0, x + 1)) / (2 ** n) * 2
    return min(1.0, p)


def analyze(evaldir, arms, kbig, n_default, label):
    print(f"\n{'='*70}\n{label}  (k={kbig})\n{'='*70}")
    data = {a: load_arm(evaldir, a, n_default) for a in arms}
    ns = {a: data[a][1] for a in arms}
    d = {a: data[a][0] for a in arms}
    # short display names (strip the _oly7b/_aime7b/_olyfrag1024 suffix)
    short = {a: a.split("_")[0] for a in arms}
    base_arm = next((a for a in arms if short[a] == "base"), None)
    grpo_arm = next((a for a in arms if short[a] == "grpo"), None)
    ids = set.intersection(*[set(d[a].keys()) for a in arms])
    print(f"problems in common: {len(ids)} | n_samples: {ns}")
    if not ids: return

    # --- aggregate pass@kbig (the diluted metric, for reference) ---
    print("\n[aggregate pass@%d]" % kbig)
    for a in arms:
        v = sum(passk(d[a][i], ns[a], kbig) for i in ids) / len(ids)
        print(f"  {short[a]:9s} {v:.4f}")

    # --- LOST SET: base solves @kbig, grpo does not (needs base + grpo) ---
    if base_arm and grpo_arm:
        base, grpo = d[base_arm], d[grpo_arm]
        nb, ng = ns[base_arm], ns[grpo_arm]
        lost = [i for i in ids if solves_at_k(base[i], nb, kbig)
                and not solves_at_k(grpo[i], ng, kbig)]
        print(f"\n[LOST SET] base solves @{kbig} but GRPO doesn't: {len(lost)} problems")
        for a in arms:
            if a in (base_arm, grpo_arm) or not lost: continue
            retained = sum(1 for i in lost if solves_at_k(d[a][i], ns[a], kbig))
            print(f"  {short[a]:9s} retains {retained}/{len(lost)} = {retained/len(lost):.3f} of GRPO's lost set")

    # --- McNemar: each method vs GRPO on who-solves-what @kbig (needs grpo) ---
    if grpo_arm:
        grpo = d[grpo_arm]; ng = ns[grpo_arm]
        print(f"\n[McNemar: method vs GRPO, who-solves-what @{kbig}] (paired, all common problems)")
        for a in arms:
            if a == grpo_arm: continue
            b = sum(1 for i in ids if solves_at_k(d[a][i], ns[a], kbig)
                    and not solves_at_k(grpo[i], ng, kbig))    # method solves, grpo doesn't
            c = sum(1 for i in ids if not solves_at_k(d[a][i], ns[a], kbig)
                    and solves_at_k(grpo[i], ng, kbig))        # grpo solves, method doesn't
            p = mcnemar(b, c)
            sig = "SIG" if p < 0.05 else "ns"
            print(f"  {short[a]:9s} vs grpo: +{b} / -{c}  (net {b-c:+d})  McNemar p={p:.4f} {sig}")

    # --- fragile band (base pass@1 <= 0.10) restricted pass@k (needs base) ---
    if base_arm:
        base = d[base_arm]; nb = ns[base_arm]
        frag = [i for i in ids if passk(base[i], nb, 1) <= 0.10]
        print(f"\n[FRAGILE BAND] base pass@1<=0.10: {len(frag)} problems — pass@{kbig} on subset:")
        for a in arms:
            if not frag: break
            v = sum(passk(d[a][i], ns[a], kbig) for i in frag) / len(frag)
            print(f"  {short[a]:9s} {v:.4f}")


if __name__ == "__main__":
    E = "/Users/cmohsinm/inference-time-uncertainty/rl_training/runs_pulled/coverage_7b_mi036f/eval"
    # 7B Olympiad: tags are <arm>_oly7b
    analyze(E, ["base_oly7b", "grpo_oly7b", "expSR_oly7b", "expPROJ_oly7b"], 256, 256,
            "7B OlympiadBench (k=256, n=572)")
    # 7B AIME: tags <arm>_aime7b
    analyze(E, ["base_aime7b", "grpo_aime7b", "expSR_aime7b", "expPROJ_aime7b"], 256, 256,
            "7B AIME (k=256, n=90)")
    # 7B fragile-band k=1024 (no base here — grpo/expSR/expPROJ only)
    analyze(E, ["grpo_olyfrag1024", "expSR_olyfrag1024", "expPROJ_olyfrag1024"], 1024, 1024,
            "7B Olympiad fragile-band (k=1024, n=329)")
