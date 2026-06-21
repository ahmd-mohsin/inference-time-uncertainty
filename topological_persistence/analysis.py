# Post-hoc analysis: load results, aggregate signals, validate against ground truth.
import json
import numpy as np
from pathlib import Path
from typing import Optional


def load_results(output_dir: str) -> list[dict]:
    results = []
    for p in sorted(Path(output_dir).glob("problem_*.json")):
        with open(p) as f:
            results.append(json.load(f))
    return results


def majority_vote_accuracy(answers: list[str], gold: str) -> bool:
    from collections import Counter
    if not answers:
        return False
    counts = Counter(a for a in answers if a.strip())
    if not counts:
        return False
    majority = counts.most_common(1)[0][0]
    try:
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.data.dataset import answers_match
        return answers_match(majority, gold)
    except ImportError:
        return majority.strip() == gold.strip()


def compute_scaling_curve(answers_by_k: dict[int, list[str]], gold: str) -> dict[int, float]:
    curve = {}
    for k, answers in sorted(answers_by_k.items()):
        n_correct = 0
        n_trials = 100
        for _ in range(n_trials):
            subset = np.random.choice(answers, size=min(k, len(answers)), replace=True).tolist()
            if majority_vote_accuracy(subset, gold):
                n_correct += 1
        curve[k] = n_correct / n_trials
    return curve


def validate_ceiling_predictions(results: list[dict]) -> dict:
    correct_ceiling = 0
    correct_scalable = 0
    total_ceiling = 0
    total_scalable = 0

    for r in results:
        if "error" in r or "signal" not in r:
            continue
        verdict = r["signal"]["verdict"]
        gold = r.get("gold_answer", "")
        answers_iid = r.get("answers_iid", [])
        answers_cond = r.get("answers_conditioned", [])

        solved_iid = majority_vote_accuracy(answers_iid, gold)
        solved_cond = majority_vote_accuracy(answers_cond, gold)
        improved = solved_cond and not solved_iid

        if verdict == "CEILING_REACHED":
            total_ceiling += 1
            if not improved:
                correct_ceiling += 1
        elif verdict == "SCALABLE":
            total_scalable += 1
            if improved or solved_iid:
                correct_scalable += 1

    return {
        "ceiling_precision": correct_ceiling / max(total_ceiling, 1),
        "scalable_precision": correct_scalable / max(total_scalable, 1),
        "total_ceiling": total_ceiling,
        "total_scalable": total_scalable,
    }


def aggregate_signals(results: list[dict]) -> dict:
    signals = [r["signal"] for r in results if "signal" in r]
    if not signals:
        return {}
    return {
        "mean_ceiling_prob": np.mean([s["ceiling_probability"] for s in signals]),
        "mean_h1_features": np.mean([s["h1_n_features"] for s in signals]),
        "mean_diversity": np.mean([s["diversity_score"] for s in signals]),
        "fraction_frozen": np.mean([s["topology_frozen"] for s in signals]),
        "verdicts": {
            "CEILING_REACHED": sum(1 for s in signals if s["verdict"] == "CEILING_REACHED"),
            "SCALABLE": sum(1 for s in signals if s["verdict"] == "SCALABLE"),
            "UNCERTAIN": sum(1 for s in signals if s["verdict"] == "UNCERTAIN"),
        },
    }
