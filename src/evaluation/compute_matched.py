import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _load_jsonlines(path: str) -> list[dict]:
    import jsonlines
    with jsonlines.open(path) as reader:
        return list(reader)


class ComputeMatchedEvaluator:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def load_results(self, path: str) -> list[dict]:
        return _load_jsonlines(path)

    def accuracy_at_budget(self, results: list[dict], token_budget: int) -> float:
        valid = [r for r in results if r["total_tokens"] <= token_budget]
        if not valid:
            return 0.0
        return float(np.mean([r["correct"] for r in valid]))

    def pareto_curve(
        self,
        results_by_method: dict[str, list[dict]],
        n_steps: int = 20,
    ) -> dict[str, list[dict]]:
        all_tokens = [
            r["total_tokens"]
            for rs in results_by_method.values()
            for r in rs
        ]
        budgets = np.linspace(min(all_tokens), max(all_tokens), n_steps).astype(int).tolist()
        return {
            method: [
                {"budget": b, "accuracy": self.accuracy_at_budget(results, b)}
                for b in budgets
            ]
            for method, results in results_by_method.items()
        }

    def budget_matched_comparison(
        self,
        method_results: dict[str, list[dict]],
        reference: str = "greedy",
    ) -> dict:
        if reference not in method_results:
            logger.warning(f"Reference method '{reference}' not in results")
            return {}
        ref_mean = float(np.mean([r["total_tokens"] for r in method_results[reference]]))
        budget_cap = ref_mean * 1.05
        out = {}
        for method, results in method_results.items():
            method_mean = float(np.mean([r["total_tokens"] for r in results]))
            if method_mean <= budget_cap:
                acc = float(np.mean([r["correct"] for r in results]))
                within = True
            else:
                capped = [r for r in results if r["total_tokens"] <= budget_cap]
                acc = float(np.mean([r["correct"] for r in capped])) if capped else 0.0
                within = False
            out[method] = {
                "accuracy": acc,
                "mean_tokens": method_mean,
                "token_ratio": method_mean / max(1.0, ref_mean),
                "within_budget": within,
            }
        return out

    def ablation_table(self, method_results: dict[str, list[dict]]) -> list[dict]:
        rows = []
        for method, results in method_results.items():
            if not results:
                continue
            rows.append({
                "method": method,
                "n": len(results),
                "accuracy": float(np.mean([r["correct"] for r in results])),
                "mean_tokens": float(np.mean([r["total_tokens"] for r in results])),
                "mean_expansion_tokens": float(np.mean([r.get("total_expansion_tokens", 0) for r in results])),
                "entropy_trigger_rate": float(np.mean([r.get("trigger_rate_entropy", 0) for r in results])),
                "expansion_trigger_rate": float(np.mean([r.get("trigger_rate_expansion", 0) for r in results])),
                "mean_wall_time": float(np.mean([r.get("wall_time_sec", 0) for r in results])),
            })
        rows.sort(key=lambda x: -x["accuracy"])
        return rows

    def print_ablation_table(self, rows: list[dict]) -> None:
        header = (
            f"{'Method':<28} {'Acc':>6} {'Tokens':>8} "
            f"{'ExpTok':>7} {'H-Rate':>7} {'X-Rate':>7} {'Time':>7}"
        )
        sep = "-" * len(header)
        logger.info(sep)
        logger.info(header)
        logger.info(sep)
        for r in rows:
            logger.info(
                f"{r['method']:<28} "
                f"{r['accuracy']:>6.4f} "
                f"{r['mean_tokens']:>8.1f} "
                f"{r['mean_expansion_tokens']:>7.1f} "
                f"{r['entropy_trigger_rate']:>7.4f} "
                f"{r['expansion_trigger_rate']:>7.4f} "
                f"{r['mean_wall_time']:>7.2f}"
            )
        logger.info(sep)

    def save(self, data: dict, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved to {path}")