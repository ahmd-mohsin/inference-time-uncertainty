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


def pass_at_k(answers: list[str], gold: str, k: int, n_trials: int = 500) -> float:
    valid = [a for a in answers if a.strip()]
    if not valid:
        return 0.0
    correct = 0
    for _ in range(n_trials):
        subset = list(np.random.choice(valid, size=min(k, len(valid)), replace=True))
        mv = majority_vote(subset)
        if answers_match_simple(mv, gold):
            correct += 1
    return correct / n_trials


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


def validate_with_more_chains(results_dir: str, n_validation: int = 64):
    """Generate many more chains to test whether 'SCALABLE' problems actually benefit."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from topological_persistence.config import load_config
    from topological_persistence.sampler import get_sampler
    from topological_persistence.pipeline import format_problem_prompt
    from src.data.dataset import get_inference_dataset

    cfg = load_config()
    results = load_results(results_dir)

    dataset_cfg = {"dataset": {"name": cfg.dataset, "split": "test",
                                "n_problems": cfg.n_problems, "seed": cfg.seed}}
    problems = get_inference_dataset(dataset_cfg)

    cfg.sampling.use_vllm = True
    sampler = get_sampler(cfg.sampling)

    validation = []
    for i, (problem, result) in enumerate(zip(problems, results)):
        pid = problem.get("problem_id", i)
        gold = problem.get("gold_answer", "")
        verdict = result["signal"]["verdict"]

        logger.info(f"Validating problem {pid} (verdict={verdict}) with {n_validation} chains...")
        prompt = format_problem_prompt(problem, cfg.sampling.model_name)

        from vllm import SamplingParams
        params = SamplingParams(
            n=n_validation,
            max_tokens=cfg.sampling.max_new_tokens,
            temperature=cfg.sampling.temperature,
            top_p=cfg.sampling.top_p,
            stop=["<|im_end|>", "<|endoftext|>"],
        )
        outputs = sampler._vllm.generate([prompt], params)[0]
        answers = []
        for out in outputs.outputs:
            import re
            matches = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", out.text)
            ans = matches[-1].strip() if matches else ""
            answers.append(ans)

        scaling_curve = {}
        for k in [4, 8, 16, 32, 64]:
            if k <= n_validation:
                scaling_curve[k] = pass_at_k(answers, gold, k)

        improved = scaling_curve.get(64, 0) > scaling_curve.get(8, 0) + 0.05

        validation.append({
            "problem_id": pid,
            "gold": gold,
            "verdict": verdict,
            "scaling_curve": scaling_curve,
            "actually_scales": improved,
            "prediction_correct": (verdict == "SCALABLE" and improved) or
                                  (verdict == "CEILING_REACHED" and not improved),
        })

        logger.info(f"  Scaling curve: {scaling_curve}")
        logger.info(f"  Verdict={verdict}, Actually scales={improved}, "
                    f"Prediction correct={validation[-1]['prediction_correct']}")

    print("\n" + "=" * 80)
    print("VALIDATION: Does the ceiling prediction hold with N=64 chains?")
    print("=" * 80)
    print(f"\n{'Prob':>4} {'Verdict':>10} {'Pass@8':>7} {'Pass@16':>7} "
          f"{'Pass@32':>7} {'Pass@64':>7} {'Scales?':>7} {'Correct?':>8}")
    print("-" * 70)
    for v in validation:
        sc = v["scaling_curve"]
        print(f"{v['problem_id']:>4} {v['verdict']:>10} "
              f"{sc.get(8, 0):>7.3f} {sc.get(16, 0):>7.3f} "
              f"{sc.get(32, 0):>7.3f} {sc.get(64, 0):>7.3f} "
              f"{'Yes' if v['actually_scales'] else 'No':>7} "
              f"{'✓' if v['prediction_correct'] else '✗':>8}")

    n_correct = sum(1 for v in validation if v["prediction_correct"])
    print(f"\nOverall prediction accuracy: {n_correct}/{len(validation)} "
          f"({n_correct/max(len(validation),1):.0%})")

    out_path = Path(results_dir) / "validation.json"
    with open(out_path, "w") as f:
        json.dump(validation, f, indent=2)
    print(f"\nValidation saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="data/topological_outputs")
    parser.add_argument("--validate", action="store_true",
                        help="Run N>>8 chains to validate ceiling predictions")
    parser.add_argument("--n-validation", type=int, default=64,
                        help="Number of chains for validation (default 64)")
    args = parser.parse_args()

    results = load_results(args.results_dir)
    if not results:
        print(f"No results found in {args.results_dir}")
        return

    analyses = [analyze_problem(r) for r in results]
    print_analysis(analyses)

    if args.validate:
        validate_with_more_chains(args.results_dir, args.n_validation)


if __name__ == "__main__":
    main()
