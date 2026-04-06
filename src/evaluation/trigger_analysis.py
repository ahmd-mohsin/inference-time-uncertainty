import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class TriggerAnalyzer:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def load_results(self, path: str) -> list[dict]:
        import jsonlines
        with jsonlines.open(path) as reader:
            return list(reader)

    def trigger_position_stats(self, results: list[dict]) -> dict:
        positions, entropies, disagreements = [], [], []
        in_correct, in_incorrect = 0, 0

        for r in results:
            for step in r.get("trace", []):
                if not step.get("expansion_triggered"):
                    continue
                positions.append(step["position"])
                entropies.append(step["entropy"])
                disagreements.append(step["disagreement_score"])
                if r["correct"]:
                    in_correct += 1
                else:
                    in_incorrect += 1

        if not positions:
            return {"n_expansion_triggers": 0}

        n_correct_probs = sum(1 for r in results if r["correct"])
        n_incorrect_probs = sum(1 for r in results if not r["correct"])

        return {
            "n_expansion_triggers": len(positions),
            "mean_trigger_position": float(np.mean(positions)),
            "median_trigger_position": float(np.median(positions)),
            "mean_entropy_at_trigger": float(np.mean(entropies)),
            "mean_disagreement_at_trigger": float(np.mean(disagreements)),
            "triggers_in_correct": in_correct,
            "triggers_in_incorrect": in_incorrect,
            "trigger_rate_in_correct_solutions": in_correct / max(1, n_correct_probs),
            "trigger_rate_in_incorrect_solutions": in_incorrect / max(1, n_incorrect_probs),
        }

    def expansion_effectiveness(self, results: list[dict]) -> dict:
        exp_correct = [r for r in results if r.get("n_expansion_triggers", 0) > 0 and r["correct"]]
        exp_incorrect = [r for r in results if r.get("n_expansion_triggers", 0) > 0 and not r["correct"]]
        no_exp_correct = [r for r in results if r.get("n_expansion_triggers", 0) == 0 and r["correct"]]
        no_exp_incorrect = [r for r in results if r.get("n_expansion_triggers", 0) == 0 and not r["correct"]]

        n_exp = len(exp_correct) + len(exp_incorrect)
        n_no_exp = len(no_exp_correct) + len(no_exp_incorrect)

        return {
            "n_problems_with_expansion": n_exp,
            "n_problems_without_expansion": n_no_exp,
            "accuracy_with_expansion": len(exp_correct) / max(1, n_exp),
            "accuracy_without_expansion": len(no_exp_correct) / max(1, n_no_exp),
            "accuracy_delta": (len(exp_correct) / max(1, n_exp)) - (len(no_exp_correct) / max(1, n_no_exp)),
        }

    def semantic_zone_stats(self, results: list[dict]) -> dict:
        zone_total = zone_triggered = non_zone_total = non_zone_triggered = 0
        for r in results:
            for step in r.get("trace", []):
                if step.get("in_semantic_zone"):
                    zone_total += 1
                    if step.get("entropy_triggered"):
                        zone_triggered += 1
                else:
                    non_zone_total += 1
                    if step.get("entropy_triggered"):
                        non_zone_triggered += 1

        return {
            "zone_tokens_total": zone_total,
            "zone_entropy_trigger_rate": zone_triggered / max(1, zone_total),
            "non_zone_tokens_total": non_zone_total,
            "non_zone_entropy_trigger_rate": non_zone_triggered / max(1, non_zone_total),
            "zone_fraction": zone_total / max(1, zone_total + non_zone_total),
        }

    def full_analysis(
        self, results: list[dict], save_path: Optional[str] = None
    ) -> dict:
        analysis = {
            "trigger_positions": self.trigger_position_stats(results),
            "expansion_effectiveness": self.expansion_effectiveness(results),
            "semantic_zone_stats": self.semantic_zone_stats(results),
        }
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w") as f:
                json.dump(analysis, f, indent=2)
            logger.info(f"Saved trigger analysis to {save_path}")

        logger.info("Trigger analysis:")
        for section, data in analysis.items():
            logger.info(f"  [{section}]")
            for k, v in data.items():
                if isinstance(v, float):
                    logger.info(f"    {k}: {v:.4f}")
                else:
                    logger.info(f"    {k}: {v}")

        return analysis