# Analyze topological persistence results and validate ceiling predictions.
#
# This script:
# 1. Loads all problem results and summarizes the topological signals
# 2. Validates predictions by running N>>8 chains and checking if accuracy improves
# 3. Computes the scaling curve: does majority-vote accuracy increase with more chains?
# 4. Correlates H1 features / topology-frozen with actual scalability
#
# Usage:
#   python -m topological_persistence.analyze_results --results-dir data/topological_outputs
#   python -m topological_persistence.analyze_results --validate --n-validation 64

import argparse
import json
import logging
import time
from pathlib import Path
from collections import Counter

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_results(results_dir: str) -> list[dict]:
    results = []
    for p in sorted(Path(results_dir).glob("problem_*.json")):
        with open(p) as f:
            results.append(json.load(f))
    return results


def majority_vote(answers: list[str]) -> str:
    counts = Counter(a for a in answers if a.strip())
    if not counts:
        return ""
    return counts.most_common(1)[0][0]


def answers_match_simple(pred: str, gold: str) -> bool:
    if not pred or not gold:
        return False
    pred = pred.strip().strip("$").rstrip(".,")
    gold = gold.strip().strip("$").rstrip(".,")
    if pred == gold:
        return True
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except (ValueError, TypeError):
        return False


def maj_at_k(answers: list[str], gold: str, k: int, n_trials: int = 500) -> float:
    """P(majority vote of k sampled chains is correct). Includes blanks as dilution."""
    if not answers:
        return 0.0
    correct = 0
    for _ in range(n_trials):
        subset = list(np.random.choice(answers, size=min(k, len(answers)), replace=False)) \
            if k <= len(answers) else list(np.random.choice(answers, size=k, replace=True))
        mv = majority_vote(subset)
        if answers_match_simple(mv, gold):
            correct += 1
    return correct / n_trials


def pass_at_k(answers: list[str], gold: str, k: int, n_trials: int = 500) -> float:
    """P(at least one of k sampled chains is correct) — the true 'does more compute help' test."""
    if not answers:
        return 0.0
    n = len(answers)
    correct_mask = np.array([answers_match_simple(a, gold) for a in answers])
    if correct_mask.sum() == 0:
        return 0.0
    if correct_mask.sum() == n:
        return 1.0
    hits = 0
    for _ in range(n_trials):
        idx = np.random.choice(n, size=min(k, n), replace=False) if k <= n \
            else np.random.choice(n, size=k, replace=True)
        if correct_mask[idx].any():
            hits += 1
    return hits / n_trials


def analyze_problem(result: dict) -> dict:
    pid = result["problem_id"]
    gold = result["gold_answer"]
    answers_iid = result["answers_iid"]
    answers_cond = result.get("answers_conditioned", [])
    sig = result["signal"]
    ncd = result.get("signal_ncd", {})
    comp = result.get("comparison", {})

    mv_iid = majority_vote(answers_iid)
    correct_iid = answers_match_simple(mv_iid, gold)
    n_correct_iid = sum(1 for a in answers_iid if answers_match_simple(a, gold))
    n_truncated_iid = sum(1 for a in answers_iid if not a.strip())

    mv_cond = majority_vote(answers_cond)
    correct_cond = answers_match_simple(mv_cond, gold)
    n_correct_cond = sum(1 for a in answers_cond if answers_match_simple(a, gold))

    all_answers = answers_iid + answers_cond
    pass_4 = pass_at_k(all_answers, gold, 4)
    pass_8 = pass_at_k(all_answers, gold, 8)
    pass_16 = pass_at_k(all_answers, gold, 16)

    unique_answers = set(a for a in answers_iid if a.strip())

    return {
        "problem_id": pid,
        "gold": gold,
        "verdict": sig["verdict"],
        "h1_features": sig["h1_n_features"],
        "h1_max_lifetime": sig["h1_max_lifetime"],
        "topology_frozen": sig["topology_frozen"],
        "ceiling_prob": sig["ceiling_probability"],
        "diversity_score": sig["diversity_score"],
        "majority_vote_iid": mv_iid,
        "correct_iid": correct_iid,
        "n_correct_iid": n_correct_iid,
        "n_truncated_iid": n_truncated_iid,
        "unique_answers_iid": len(unique_answers),
        "majority_vote_cond": mv_cond,
        "correct_cond": correct_cond,
        "n_correct_cond": n_correct_cond,
        "pass@4": pass_4,
        "pass@8": pass_8,
        "pass@16": pass_16,
        "ncd_mean": ncd.get("mean_ncd", 0),
        "diversity_gain": comp.get("diversity_gain", 0),
        "new_topo_features": comp.get("new_topological_features", False),
    }


def print_analysis(analyses: list[dict]):
    print("\n" + "=" * 80)
    print("TOPOLOGICAL PERSISTENCE CEILING DETECTION — RESULTS ANALYSIS")
    print("=" * 80)

    print("\n--- Per-Problem Summary ---\n")
    print(f"{'Prob':>4} {'Gold':>6} {'MV_IID':>8} {'Correct':>7} {'Trunc':>5} "
          f"{'H1':>3} {'Lifetime':>8} {'Frozen':>6} {'Verdict':>10}")
    print("-" * 75)
    for a in analyses:
        print(f"{a['problem_id']:>4} {a['gold']:>6} {a['majority_vote_iid']:>8} "
              f"{'✓' if a['correct_iid'] else '✗':>7} {a['n_truncated_iid']:>5} "
              f"{a['h1_features']:>3} {a['h1_max_lifetime']:>8.3f} "
              f"{'Yes' if a['topology_frozen'] else 'No':>6} {a['verdict']:>10}")

    print("\n--- Pass@K Analysis ---\n")
    print(f"{'Prob':>4} {'Verdict':>10} {'Pass@4':>7} {'Pass@8':>7} {'Pass@16':>7} "
          f"{'Scaling?':>8}")
    print("-" * 55)
    for a in analyses:
        scales = "Yes" if a["pass@16"] > a["pass@4"] + 0.05 else "No"
        print(f"{a['problem_id']:>4} {a['verdict']:>10} {a['pass@4']:>7.3f} "
              f"{a['pass@8']:>7.3f} {a['pass@16']:>7.3f} {scales:>8}")

    print("\n--- Ceiling Prediction Validation ---\n")
    ceiling_probs = [a for a in analyses if a["verdict"] == "CEILING_REACHED"]
    scalable_probs = [a for a in analyses if a["verdict"] == "SCALABLE"]

    print(f"CEILING problems ({len(ceiling_probs)}):")
    for a in ceiling_probs:
        print(f"  Problem {a['problem_id']}: correct={a['correct_iid']}, "
              f"pass@16={a['pass@16']:.3f}, scaling={'Yes' if a['pass@16'] > a['pass@4'] + 0.05 else 'No'}")
    print(f"\nSCALABLE problems ({len(scalable_probs)}):")
    for a in scalable_probs:
        print(f"  Problem {a['problem_id']}: correct={a['correct_iid']}, "
              f"pass@16={a['pass@16']:.3f}, scaling={'Yes' if a['pass@16'] > a['pass@4'] + 0.05 else 'No'}")

    print("\n--- Conditioning Effect ---\n")
    print(f"{'Prob':>4} {'Correct_IID':>11} {'Correct_Cond':>12} {'Div_Gain':>9} {'New_Topo':>8}")
    print("-" * 50)
    for a in analyses:
        print(f"{a['problem_id']:>4} {'✓' if a['correct_iid'] else '✗':>11} "
              f"{'✓' if a['correct_cond'] else '✗':>12} "
              f"{a['diversity_gain']:>9.4f} {'Yes' if a['new_topo_features'] else 'No':>8}")

    print("\n--- Key Findings ---\n")
    print("1. TOPOLOGY ↔ DIVERSITY CORRELATION:")
    h1_values = [a["h1_features"] for a in analyses]
    unique_values = [a["unique_answers_iid"] for a in analyses]
    if len(h1_values) > 2:
        corr = np.corrcoef(h1_values, unique_values)[0, 1]
        print(f"   Correlation(H1_features, unique_answers) = {corr:.3f}")
    else:
        print(f"   H1 features: {h1_values}, Unique answers: {unique_values}")

    print("\n2. CEILING PREDICTION ACCURACY:")
    correct_ceiling = sum(1 for a in ceiling_probs
                          if a["pass@16"] <= a["pass@4"] + 0.05)
    correct_scalable = sum(1 for a in scalable_probs
                           if a["pass@16"] > a["pass@4"] + 0.05 or a["n_correct_iid"] > 0)
    total = len(ceiling_probs) + len(scalable_probs)
    accuracy = (correct_ceiling + correct_scalable) / max(total, 1)
    print(f"   Ceiling predictions correct: {correct_ceiling}/{len(ceiling_probs)}")
    print(f"   Scalable predictions correct: {correct_scalable}/{len(scalable_probs)}")
    print(f"   Overall accuracy: {accuracy:.1%}")

    print("\n3. H1 LIFETIME AS A SIGNAL:")
    for a in sorted(analyses, key=lambda x: x["h1_max_lifetime"], reverse=True):
        status = "correct" if a["correct_iid"] else f"wrong (gold={a['gold']})"
        print(f"   Problem {a['problem_id']}: H1_lifetime={a['h1_max_lifetime']:.3f}, "
              f"answer={a['majority_vote_iid'] or 'NONE'}, {status}")

    print("\n4. NCD vs HIDDEN-STATE TOPOLOGY DIVERGENCE:")
    for a in analyses:
        print(f"   Problem {a['problem_id']}: NCD={a['ncd_mean']:.3f}, "
              f"H1={a['h1_features']}, verdict={a['verdict']}")


def validate_with_more_chains(results_dir: str, n_validation: int = 64,
                              dataset: str = None, model: str = None,
                              shard_index: int = 0, num_shards: int = 1):
    """Generate N>>8 chains per problem to test whether the ceiling prediction holds.

    Ceiling test logic (using pass@k = P(>=1 correct in k samples)):
      - CEILING prediction is CORRECT if scaling does NOT help:
          pass@N <= pass@8 + delta  (no new correct solutions emerge from more compute)
      - SCALABLE prediction is CORRECT if scaling helps OR it is already solvable:
          pass@N > pass@8 + delta, OR pass@8 already high (>0.5)
    We log BOTH maj@k (selection accuracy) and pass@k (coverage / does-the-answer-exist).
    """
    import sys, os, re
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from topological_persistence.config import load_config
    from topological_persistence.pipeline import format_problem_prompt
    from src.data.dataset import get_inference_dataset
    from vllm import LLM, SamplingParams

    cfg = load_config()
    if dataset:
        cfg.dataset = dataset
    if model:
        cfg.sampling.model_name = model

    results = load_results(results_dir)
    dataset_cfg = {"dataset": {"name": cfg.dataset, "split": "test",
                                "n_problems": cfg.n_problems, "seed": cfg.seed}}
    problems = get_inference_dataset(dataset_cfg)

    logger.info(f"Loading vLLM (TP={cfg.sampling.tensor_parallel_size}) for validation...")
    llm = LLM(
        model=cfg.sampling.model_name,
        dtype=cfg.sampling.dtype,
        tensor_parallel_size=cfg.sampling.tensor_parallel_size,
        trust_remote_code=True,
        max_model_len=16384 + 512,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.9,
    )

    by_id = {r["problem_id"]: r for r in results}
    delta = 0.10
    ks = [4, 8, 16, 32, 64]

    val_suffix = f"_shard{shard_index}" if num_shards > 1 else ""
    val_path = Path(results_dir) / f"validation{val_suffix}.json"

    validation = []
    for i, problem in enumerate(problems):
        pid = problem.get("problem_id", i)
        if pid not in by_id:
            continue
        if num_shards > 1 and (pid % num_shards) != shard_index:
            continue
        gold = problem.get("gold_answer", "")
        verdict = by_id[pid]["signal"]["verdict"]
        h1 = by_id[pid]["signal"]["h1_n_features"]

        logger.info(f"[{i+1}/{len(problems)}] Problem {pid} (verdict={verdict}) — {n_validation} chains...")
        prompt = format_problem_prompt(problem, cfg.sampling.model_name)
        params = SamplingParams(
            n=n_validation,
            max_tokens=cfg.sampling.max_new_tokens,
            temperature=cfg.sampling.temperature,
            top_p=cfg.sampling.top_p,
            stop=["<|im_end|>", "<|endoftext|>"],
        )
        outputs = llm.generate([prompt], params)[0]
        answers = []
        for out in outputs.outputs:
            matches = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", out.text)
            answers.append(matches[-1].strip() if matches else "")

        pass_curve = {k: pass_at_k(answers, gold, k) for k in ks if k <= n_validation}
        maj_curve = {k: maj_at_k(answers, gold, k) for k in ks if k <= n_validation}
        n_correct = sum(1 for a in answers if answers_match_simple(a, gold))

        kmax = max(pass_curve.keys())
        pass_gain = pass_curve[kmax] - pass_curve.get(8, 0)
        scales = pass_gain > delta
        already_solved = pass_curve.get(8, 0) > 0.5

        # Honest scoring: a SCALABLE prediction is correct ONLY if compute actually helped
        # (pass@max meaningfully exceeds pass@8). The old `scales or already_solved` rule
        # rescued every already-solvable problem and made SCALABLE nearly unfalsifiable
        # (10/11 "correct" SCALABLE calls had not actually scaled). Problems already solved
        # at 8 carry no scaling signal -> they are scored None (excluded), same as UNCERTAIN.
        if already_solved and not scales:
            # no headroom to test the prediction either way
            prediction_correct = None
        elif verdict == "CEILING_REACHED":
            prediction_correct = not scales
        elif verdict == "SCALABLE":
            prediction_correct = scales
        else:
            prediction_correct = None

        # distinguish solved-ceiling from stuck-ceiling
        ceiling_kind = None
        if verdict == "CEILING_REACHED":
            ceiling_kind = "solved" if already_solved else "stuck"

        rec = {
            "problem_id": pid, "gold": gold, "verdict": verdict, "h1_features": h1,
            "n_correct_of_N": n_correct, "n_validation": n_validation,
            "pass_curve": pass_curve, "maj_curve": maj_curve,
            "pass_gain_8_to_max": pass_gain, "actually_scales": scales,
            "already_solved_at_8": already_solved, "ceiling_kind": ceiling_kind,
            "prediction_correct": prediction_correct,
        }
        validation.append(rec)
        logger.info(f"  pass@8={pass_curve.get(8,0):.2f} pass@{kmax}={pass_curve[kmax]:.2f} "
                    f"gain={pass_gain:+.2f} scales={scales} pred_correct={prediction_correct}")

        # incremental save so a crash mid-run preserves progress
        with open(val_path, "w") as f:
            json.dump(validation, f, indent=2)

    print("\n" + "=" * 90)
    print(f"VALIDATION: ceiling predictions vs actual scaling (N={n_validation} chains)")
    print("=" * 90)
    print(f"\n{'Prob':>4} {'Verdict':>9} {'Kind':>6} {'H1':>3} {'pass@8':>7} {'pass@64':>8} "
          f"{'gain':>6} {'maj@8':>6} {'maj@64':>7} {'scales':>7} {'pred_ok':>7}")
    print("-" * 90)
    for v in validation:
        kmax = max(v["pass_curve"].keys())
        print(f"{v['problem_id']:>4} {v['verdict'][:8]:>9} {str(v['ceiling_kind'] or '-'):>6} "
              f"{v['h1_features']:>3} {v['pass_curve'].get(8,0):>7.2f} {v['pass_curve'][kmax]:>8.2f} "
              f"{v['pass_gain_8_to_max']:>+6.2f} {v['maj_curve'].get(8,0):>6.2f} "
              f"{v['maj_curve'][kmax]:>7.2f} {'Yes' if v['actually_scales'] else 'No':>7} "
              f"{'OK' if v['prediction_correct'] else 'X':>7}")

    judged = [v for v in validation if v["prediction_correct"] is not None]
    n_ok = sum(1 for v in judged if v["prediction_correct"])
    print(f"\nOverall prediction accuracy: {n_ok}/{len(judged)} "
          f"({n_ok/max(len(judged),1):.0%})")

    # break down by verdict
    for verdict in ["CEILING_REACHED", "SCALABLE"]:
        sub = [v for v in judged if v["verdict"] == verdict]
        if sub:
            ok = sum(1 for v in sub if v["prediction_correct"])
            print(f"  {verdict}: {ok}/{len(sub)} correct")
    stuck = [v for v in validation if v["ceiling_kind"] == "stuck"]
    solved = [v for v in validation if v["ceiling_kind"] == "solved"]
    print(f"  CEILING breakdown: {len(solved)} solved-ceiling, {len(stuck)} stuck-ceiling")

    print(f"\nValidation saved to {val_path}")


def merge_validation_shards(results_dir: str, num_shards: int):
    """Combine validation_shard*.json into validation.json (sorted by problem_id)."""
    merged = []
    for s in range(num_shards):
        sp = Path(results_dir) / f"validation_shard{s}.json"
        if sp.exists():
            merged.extend(json.load(open(sp)))
    merged.sort(key=lambda v: v["problem_id"])
    out = Path(results_dir) / "validation.json"
    json.dump(merged, open(out, "w"), indent=2)

    judged = [v for v in merged if v["prediction_correct"] is not None]
    n_ok = sum(1 for v in judged if v["prediction_correct"])
    print(f"\nMerged {len(merged)} problems from {num_shards} shards -> {out}")
    print(f"Overall prediction accuracy: {n_ok}/{len(judged)} "
          f"({n_ok/max(len(judged),1):.0%})")
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="data/topological_outputs")
    parser.add_argument("--validate", action="store_true",
                        help="Run N>>8 chains to validate ceiling predictions")
    parser.add_argument("--n-validation", type=int, default=64,
                        help="Number of chains for validation (default 64)")
    parser.add_argument("--dataset", default=None, help="Override dataset for validation")
    parser.add_argument("--model", default=None, help="Override model for validation")
    # Data-parallel validation: one process per GPU, each --shard-index i --num-shards N.
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--merge-only", action="store_true",
                        help="Merge validation_shard*.json into validation.json and exit")
    args = parser.parse_args()

    if args.merge_only:
        merge_validation_shards(args.results_dir, args.num_shards)
        return

    results = load_results(args.results_dir)
    if not results:
        print(f"No results found in {args.results_dir}")
        return

    # full per-problem analysis printout only on the single/primary process
    if args.shard_index == 0:
        analyses = [analyze_problem(r) for r in results]
        print_analysis(analyses)

    if args.validate:
        validate_with_more_chains(args.results_dir, args.n_validation,
                                  dataset=args.dataset, model=args.model,
                                  shard_index=args.shard_index, num_shards=args.num_shards)


if __name__ == "__main__":
    main()
