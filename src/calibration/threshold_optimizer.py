import json
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
import jsonlines

logger = logging.getLogger(__name__)


def _load_jsonlines(path: str) -> list[dict]:
    records = []
    with jsonlines.open(path) as reader:
        for r in reader:
            records.append(r)
    return records


def _score_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    triggered: np.ndarray,
    metric: str,
    auroc: float,
) -> float:
    if metric == "auroc":
        return auroc
    tp = float((triggered * labels).sum())
    fp = float((triggered * (1 - labels)).sum())
    fn = float(((1 - triggered) * labels).sum())
    if metric == "precision":
        return tp / (tp + fp + 1e-9)
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    return 2 * precision * recall / (precision + recall + 1e-9)


class ThresholdOptimizer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        cal = cfg["calibration"]

        self.entropy_range = np.linspace(
            cal["entropy_threshold_range"]["min"],
            cal["entropy_threshold_range"]["max"],
            cal["entropy_threshold_range"]["n_steps"],
        )
        self.disagreement_range = np.linspace(
            cal["disagreement_threshold_range"]["min"],
            cal["disagreement_threshold_range"]["max"],
            cal["disagreement_threshold_range"]["n_steps"],
        )

        self.metric = cal.get("metric", "auroc")
        self.min_trigger_rate = cal.get("min_trigger_rate", 0.005)
        self.max_trigger_rate = cal.get("max_trigger_rate", 0.15)

    def load_token_records(self, path: str) -> list[dict]:
        records = _load_jsonlines(path)
        logger.info(f"Loaded {len(records)} token records from {path}")
        return records

    def load_disagreement_records(self, path: str) -> list[dict]:
        records = _load_jsonlines(path)
        logger.info(f"Loaded {len(records)} disagreement records from {path}")
        return records

    def optimize_entropy_threshold(self, token_records: list[dict]) -> dict:
        logger.info(
            f"Optimizing tau_e — metric={self.metric} "
            f"trigger_rate=[{self.min_trigger_rate},{self.max_trigger_rate}]"
        )

        zone_records = [r for r in token_records if r["in_semantic_zone"]]
        if not zone_records:
            logger.warning("No records in semantic zone — returning default tau_e=1.0")
            return {"tau_e": 1.0, "entropy_auroc_full": 0.5, "entropy_trigger_rate": 0.0}

        entropies = np.array([r["entropy"] for r in zone_records], dtype=np.float64)
        labels = np.array(
            [int(not r["is_correct_step"]) for r in zone_records], dtype=np.float64
        )

        nan_mask = np.isfinite(entropies)
        entropies = entropies[nan_mask]
        labels = labels[nan_mask]

        if len(entropies) == 0:
            logger.warning("All entropy values are nan/inf — returning default tau_e=1.5")
            return {"tau_e": 1.5, "entropy_auroc_full": 0.5, "entropy_trigger_rate": 0.0}

        if labels.sum() == 0 or labels.sum() == len(labels):
            logger.warning("All-same labels in zone — using median entropy")
            tau = float(np.median(entropies))
            return {
                "tau_e": tau,
                "entropy_auroc_full": 0.5,
                "entropy_trigger_rate": float((entropies > tau).mean()),
            }

        try:
            auroc = float(roc_auc_score(labels, entropies))
        except Exception:
            auroc = 0.5

        best_tau = None
        best_score = -1.0
        best_rate = 0.0

        for tau in self.entropy_range:
            triggered = (entropies > tau).astype(np.float64)
            rate = float(triggered.mean())
            if rate < self.min_trigger_rate or rate > self.max_trigger_rate:
                continue
            score = _score_threshold(labels, entropies, triggered, self.metric, auroc)
            if score > best_score:
                best_score = score
                best_tau = float(tau)
                best_rate = rate

        if best_tau is None:
            logger.warning(
                "No tau_e met trigger rate constraint — using median entropy as fallback"
            )
            best_tau = float(np.median(entropies))
            best_rate = float((entropies > best_tau).mean())

        logger.info(
            f"tau_e={best_tau:.4f}  {self.metric}={best_score:.4f}  "
            f"trigger_rate={best_rate:.4f}  auroc_full={auroc:.4f}"
        )

        return {
            "tau_e": best_tau,
            f"entropy_{self.metric}": best_score,
            "entropy_trigger_rate": best_rate,
            "entropy_auroc_full": auroc,
        }

    def optimize_disagreement_threshold(
        self, disagreement_records: list[dict]
    ) -> dict:
        logger.info(
            f"Optimizing tau_d — metric={self.metric} "
            f"trigger_rate=[{self.min_trigger_rate},{self.max_trigger_rate * 2}]"
        )

        triggered_records = [
            r for r in disagreement_records if r.get("entropy_triggered", False)
        ]
        if not triggered_records:
            logger.warning("No entropy-triggered records — returning default tau_d=0.3")
            return {
                "tau_d": 0.3,
                "disagreement_auroc_full": 0.5,
                "disagreement_trigger_rate": 0.0,
            }

        d_scores = np.array(
            [r["disagreement_score"] for r in triggered_records], dtype=np.float64
        )
        labels = np.array(
            [int(not r["is_correct_step"]) for r in triggered_records], dtype=np.float64
        )

        if labels.sum() == 0 or labels.sum() == len(labels):
            logger.warning("All-same disagreement labels — using 0.3 as fallback")
            return {
                "tau_d": 0.3,
                "disagreement_auroc_full": 0.5,
                "disagreement_trigger_rate": float((d_scores > 0.3).mean()),
            }

        try:
            auroc = float(roc_auc_score(labels, d_scores))
        except Exception:
            auroc = 0.5

        best_tau = None
        best_score = -1.0
        best_rate = 0.0
        max_rate = self.max_trigger_rate * 2

        for tau in self.disagreement_range:
            triggered = (d_scores > tau).astype(np.float64)
            rate = float(triggered.mean())
            if rate < self.min_trigger_rate or rate > max_rate:
                continue
            score = _score_threshold(labels, d_scores, triggered, self.metric, auroc)
            if score > best_score:
                best_score = score
                best_tau = float(tau)
                best_rate = rate

        if best_tau is None:
            logger.warning(
                "No tau_d met trigger rate constraint — using 0.3 as fallback"
            )
            best_tau = 0.3
            best_rate = float((d_scores > best_tau).mean())

        logger.info(
            f"tau_d={best_tau:.4f}  {self.metric}={best_score:.4f}  "
            f"trigger_rate={best_rate:.4f}  auroc_full={auroc:.4f}"
        )

        return {
            "tau_d": best_tau,
            f"disagreement_{self.metric}": best_score,
            "disagreement_trigger_rate": best_rate,
            "disagreement_auroc_full": auroc,
        }

    def compare_signals(self, token_records: list[dict]) -> dict:
        zone_records = [r for r in token_records if r["in_semantic_zone"]]
        if not zone_records:
            return {}
        labels = np.array(
            [int(not r["is_correct_step"]) for r in zone_records], dtype=np.float64
        )
        if labels.sum() == 0 or labels.sum() == len(labels):
            return {}

        results: dict = {}
        signal_arrays = {
            "entropy_auroc": np.array([r["entropy"] for r in zone_records]),
            "neg_margin_auroc": np.array([-r["logit_margin"] for r in zone_records]),
            "neg_top1_auroc": np.array([-r["top1_prob"] for r in zone_records]),
        }
        for name, scores in signal_arrays.items():
            try:
                results[name] = float(roc_auc_score(labels, scores))
            except Exception:
                results[name] = 0.5

        logger.info("Signal comparison (AUROC for predicting step errors):")
        for k, v in results.items():
            logger.info(f"  {k}: {v:.4f}")

        return results

    def save_thresholds(
        self,
        model_short_name: str,
        tau_e: float,
        tau_d: float,
        metadata: dict,
        path: str,
    ) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        existing: dict = {}
        if Path(path).exists():
            with open(path) as f:
                existing = json.load(f)

        existing[model_short_name] = {
            "tau_e": tau_e,
            "tau_d": tau_d,
            **{k: v for k, v in metadata.items() if not isinstance(v, dict)},
            "signal_comparison": metadata.get("signal_comparison", {}),
        }

        with open(path, "w") as f:
            json.dump(existing, f, indent=2)

        logger.info(f"Saved thresholds for '{model_short_name}' to {path}")
        logger.info(f"  tau_e={tau_e:.4f}  tau_d={tau_d:.4f}")

    def load_thresholds(
        self, model_short_name: str, path: str
    ) -> tuple[float, float]:
        if not Path(path).exists():
            raise FileNotFoundError(f"Thresholds file not found: {path}")
        with open(path) as f:
            thresholds = json.load(f)
        if model_short_name not in thresholds:
            raise KeyError(
                f"No thresholds for '{model_short_name}' in {path}. "
                f"Available: {list(thresholds.keys())}"
            )
        entry = thresholds[model_short_name]
        tau_e = float(entry["tau_e"])
        tau_d = float(entry["tau_d"])
        logger.info(
            f"Loaded thresholds for '{model_short_name}': "
            f"tau_e={tau_e:.4f}  tau_d={tau_d:.4f}"
        )
        return tau_e, tau_d