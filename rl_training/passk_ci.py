# Bootstrap confidence intervals + cross-seed aggregation for pass@k curves.
#
# METHODOLOGY FIX #1: the Round-1 arm gaps (~0.018 at k=256) are within 1-2 standard errors on
# 500 problems, so they need error bars to be publishable. This tool takes one or more per-seed
# passk_{tag}[.seedS].json files per arm and produces, per k:
#   - mean pass@k across problems (and across seeds if multiple)
#   - a bootstrap CI by resampling PROBLEMS with replacement (the dominant variance source)
#   - optional across-seed spread
# and prints a paired base-vs-arm delta with its CI (does the crossover gap exclude 0?).
#
# Usage:
#   python -m rl_training.passk_ci --eval-dir rl_training/runs/eval --arms base,grpo,oursA,oursAB
#   python -m rl_training.passk_ci --eval-dir DIR --arms base,oursAB --paired base   # delta CIs vs base
# Pure stdlib (json/random/statistics) — no torch/vllm, runs anywhere including locally.
import argparse, glob, json, os, random
from statistics import mean


def _load_arm(eval_dir, tag):
    """Load all per-seed jsons for an arm: passk_{tag}.json and passk_{tag}.seed*.json.
    Returns list of per_problem dicts {problem_id: {k: 0/1}} — one per seed file."""
    files = sorted(set(glob.glob(os.path.join(eval_dir, f"passk_{tag}.json")) +
                       glob.glob(os.path.join(eval_dir, f"passk_{tag}.seed*.json"))))
    reps = []
    for f in files:
        d = json.load(open(f))
        pm = {p["problem_id"]: {int(k): (v if isinstance(v, (int, float)) else v)
                                for k, v in p["pass_at_k"].items()} for p in d["per_problem"]}
        reps.append((f, pm))
    return reps


def _ks(reps):
    any_pm = reps[0][1]
    any_prob = next(iter(any_pm.values()))
    return sorted(any_prob.keys())


def _curve_over(pm, ids, k):
    return mean(pm[i][k] for i in ids)


def bootstrap_ci(pm, k, ids, n_boot=2000, alpha=0.05, rng=None):
    """Percentile bootstrap CI for mean pass@k, resampling PROBLEMS with replacement."""
    rng = rng or random.Random(0)
    n = len(ids)
    point = _curve_over(pm, ids, k)
    boots = []
    for _ in range(n_boot):
        sample = [ids[rng.randrange(n)] for _ in range(n)]
        boots.append(mean(pm[i][k] for i in sample))
    boots.sort()
    lo = boots[int((alpha / 2) * n_boot)]
    hi = boots[int((1 - alpha / 2) * n_boot)]
    return point, lo, hi


def paired_delta_ci(pm_a, pm_base, k, ids, n_boot=2000, alpha=0.05, rng=None):
    """Bootstrap CI on the PAIRED per-problem delta (arm - base) at k. If the CI excludes 0 the
    difference is significant. Paired (same problems resampled together) => tighter, correct test."""
    rng = rng or random.Random(0)
    n = len(ids)
    point = mean(pm_a[i][k] - pm_base[i][k] for i in ids)
    boots = []
    for _ in range(n_boot):
        sample = [ids[rng.randrange(n)] for _ in range(n)]
        boots.append(mean(pm_a[i][k] - pm_base[i][k] for i in sample))
    boots.sort()
    return point, boots[int((alpha / 2) * n_boot)], boots[int((1 - alpha / 2) * n_boot)]


def _merge_seeds(reps):
    """Average per-problem 0/1 pass@k across seed replicates (only problems present in ALL)."""
    pms = [pm for _, pm in reps]
    common = set(pms[0])
    for pm in pms[1:]:
        common &= set(pm)
    ks = _ks(reps)
    merged = {i: {k: mean(pm[i][k] for pm in pms) for k in ks} for i in common}
    return merged, sorted(common), len(pms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--arms", required=True, help="comma-sep tags, e.g. base,grpo,oursA,oursAB")
    ap.add_argument("--paired", default="", help="baseline tag for paired delta CIs (e.g. base)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default="", help="optional json to write the curves+CIs")
    a = ap.parse_args()
    rng = random.Random(1234)

    arms = [t for t in a.arms.split(",") if t]
    loaded = {}
    for tag in arms:
        reps = _load_arm(a.eval_dir, tag)
        if not reps:
            print(f"  !! {tag}: no eval json found, skipping"); continue
        merged, ids, n_seeds = _merge_seeds(reps)
        loaded[tag] = (merged, ids, n_seeds, [f for f, _ in reps])

    # restrict every arm to the COMMON problem set for fair comparison
    common = None
    for tag, (merged, ids, _, _) in loaded.items():
        common = set(ids) if common is None else (common & set(ids))
    common = sorted(common)
    ks = _ks(next(iter(loaded.values()))[3] and [(None, loaded[arms[0]][0])] or [])  # ks from merged
    ks = sorted(next(iter(loaded.values()))[0][common[0]].keys())
    print(f"Arms: {list(loaded)} | common problems: {len(common)} | seeds/arm: "
          f"{ {t: loaded[t][2] for t in loaded} }\n")

    result = {"n_problems": len(common), "arms": {}}
    for tag, (merged, _, n_seeds, files) in loaded.items():
        print(f"=== {tag} (n_seeds={n_seeds}) ===")
        result["arms"][tag] = {"n_seeds": n_seeds, "curve": {}}
        for k in ks:
            pt, lo, hi = bootstrap_ci(merged, k, common, n_boot=a.n_boot, rng=rng)
            result["arms"][tag]["curve"][k] = {"mean": pt, "lo": lo, "hi": hi}
            print(f"  k={k:>3}: {pt:.3f}  [{lo:.3f}, {hi:.3f}]")
        print()

    if a.paired and a.paired in loaded:
        base_pm = loaded[a.paired][0]
        print(f"=== PAIRED delta vs {a.paired} (CI excludes 0 => significant) ===")
        result["paired_vs"] = a.paired; result["deltas"] = {}
        for tag in loaded:
            if tag == a.paired:
                continue
            result["deltas"][tag] = {}
            print(f"  {tag} - {a.paired}:")
            for k in ks:
                pt, lo, hi = paired_delta_ci(loaded[tag][0], base_pm, k, common, n_boot=a.n_boot, rng=rng)
                sig = "" if (lo <= 0 <= hi) else "  *SIG*"
                result["deltas"][tag][k] = {"delta": pt, "lo": lo, "hi": hi, "sig": bool(sig)}
                print(f"    k={k:>3}: {pt:+.3f}  [{lo:+.3f}, {hi:+.3f}]{sig}")
            print()

    if a.out:
        json.dump(result, open(a.out, "w"), indent=2)
        print(f"saved -> {a.out}")


if __name__ == "__main__":
    main()
