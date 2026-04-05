import logging
from pathlib import Path

import numpy as np
import jsonlines

logger = logging.getLogger(__name__)


def _get_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 11,
            "figure.dpi": 150,
            "axes.grid": True,
            "grid.alpha": 0.3,
        })
        return plt
    except ImportError:
        logger.warning("matplotlib not available — skipping plots")
        return None


class CalibrationAnalyzer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model_short_name = cfg["model"]["short_name"]
        self.figures_dir = Path(cfg["output"]["figures_dir"])
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def load_records(self, path: str) -> list[dict]:
        with jsonlines.open(path) as reader:
            return list(reader)

    def plot_entropy_distribution(
        self, token_records: list[dict], tau_e: float, save: bool = True
    ) -> dict:
        zone_records = [r for r in token_records if r["in_semantic_zone"]]
        if not zone_records:
            return {}

        correct_entropies = [r["entropy"] for r in zone_records if r["is_correct_step"]]
        incorrect_entropies = [
            r["entropy"] for r in zone_records if not r["is_correct_step"]
        ]

        stats = {
            "correct_mean_entropy": float(np.mean(correct_entropies)) if correct_entropies else 0.0,
            "incorrect_mean_entropy": float(np.mean(incorrect_entropies)) if incorrect_entropies else 0.0,
            "n_correct_zone_tokens": len(correct_entropies),
            "n_incorrect_zone_tokens": len(incorrect_entropies),
            "trigger_rate_at_tau_e": float(
                np.mean([r["entropy"] > tau_e for r in zone_records])
            ),
        }

        if not save:
            return stats

        plt = _get_matplotlib()
        if plt is None:
            return stats

        fig, ax = plt.subplots(figsize=(8, 5))
        bins = np.linspace(0, 5, 50)
        if correct_entropies:
            ax.hist(
                correct_entropies, bins=bins, alpha=0.6,
                label="Correct steps", color="steelblue", density=True,
            )
        if incorrect_entropies:
            ax.hist(
                incorrect_entropies, bins=bins, alpha=0.6,
                label="Incorrect steps", color="crimson", density=True,
            )
        ax.axvline(
            tau_e, color="black", linestyle="--", linewidth=2,
            label=f"$\\tau_e$ = {tau_e:.3f}",
        )
        ax.set_xlabel("Shannon Entropy $H_t$")
        ax.set_ylabel("Density")
        ax.set_title(
            f"Entropy Distribution at Semantic Load Zones\n{self.model_short_name}"
        )
        ax.legend()
        out_path = self.figures_dir / f"entropy_dist_{self.model_short_name}.png"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved → {out_path}")
        stats["entropy_dist_plot"] = str(out_path)
        return stats

    def plot_disagreement_distribution(
        self, disagreement_records: list[dict], tau_d: float, save: bool = True
    ) -> dict:
        triggered = [r for r in disagreement_records if r.get("entropy_triggered", False)]
        if not triggered:
            return {}

        correct_d = [r["disagreement_score"] for r in triggered if r["is_correct_step"]]
        incorrect_d = [
            r["disagreement_score"] for r in triggered if not r["is_correct_step"]
        ]

        stats = {
            "correct_mean_disagreement": float(np.mean(correct_d)) if correct_d else 0.0,
            "incorrect_mean_disagreement": float(np.mean(incorrect_d)) if incorrect_d else 0.0,
            "n_triggered_correct": len(correct_d),
            "n_triggered_incorrect": len(incorrect_d),
            "expand_rate_at_tau_d": float(
                np.mean([r["disagreement_score"] > tau_d for r in triggered])
            ),
        }

        if not save:
            return stats

        plt = _get_matplotlib()
        if plt is None:
            return stats

        fig, ax = plt.subplots(figsize=(8, 5))
        bins = np.linspace(0, 1, 40)
        if correct_d:
            ax.hist(
                correct_d, bins=bins, alpha=0.6,
                label="Correct steps", color="steelblue", density=True,
            )
        if incorrect_d:
            ax.hist(
                incorrect_d, bins=bins, alpha=0.6,
                label="Incorrect steps", color="crimson", density=True,
            )
        ax.axvline(
            tau_d, color="black", linestyle="--", linewidth=2,
            label=f"$\\tau_d$ = {tau_d:.3f}",
        )
        ax.set_xlabel("Semantic Disagreement Score $d_t$")
        ax.set_ylabel("Density")
        ax.set_title(
            f"Disagreement Distribution at Entropy-Triggered Positions\n"
            f"{self.model_short_name}"
        )
        ax.legend()
        out_path = self.figures_dir / f"disagreement_dist_{self.model_short_name}.png"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved → {out_path}")
        stats["disagreement_dist_plot"] = str(out_path)
        return stats

    def plot_signal_comparison_roc(
        self, token_records: list[dict], save: bool = True
    ) -> dict:
        from sklearn.metrics import roc_auc_score, roc_curve

        zone_records = [r for r in token_records if r["in_semantic_zone"]]
        if not zone_records:
            return {}

        labels = np.array(
            [int(not r["is_correct_step"]) for r in zone_records], dtype=np.float64
        )
        if labels.sum() == 0 or labels.sum() == len(labels):
            return {}

        signals = {
            "Entropy ($H_t$)": np.array([r["entropy"] for r in zone_records]),
            "Neg. Logit Margin": np.array([-r["logit_margin"] for r in zone_records]),
            "Neg. Top-1 Prob": np.array([-r["top1_prob"] for r in zone_records]),
        }
        colors = ["crimson", "steelblue", "seagreen"]

        results: dict = {}
        plt = _get_matplotlib() if save else None
        fig = ax = None
        if plt is not None:
            fig, ax = plt.subplots(figsize=(7, 6))

        for (name, scores), color in zip(signals.items(), colors):
            try:
                auroc = float(roc_auc_score(labels, scores))
                results[name] = auroc
                if ax is not None:
                    fpr, tpr, _ = roc_curve(labels, scores)
                    ax.plot(fpr, tpr, label=f"{name} (AUC={auroc:.3f})", color=color, linewidth=2)
            except Exception as e:
                logger.warning(f"ROC failed for {name}: {e}")

        if ax is not None:
            ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(
                f"Signal Comparison — Error Prediction ROC\n{self.model_short_name}"
            )
            ax.legend()
            out_path = self.figures_dir / f"roc_comparison_{self.model_short_name}.png"
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Saved → {out_path}")
            results["roc_plot"] = str(out_path)

        return results

    def print_summary(
        self,
        tau_e: float,
        tau_d: float,
        entropy_stats: dict,
        disagreement_stats: dict,
        signal_comparison: dict,
    ) -> None:
        sep = "=" * 60
        logger.info(sep)
        logger.info(f"CALIBRATION SUMMARY — {self.model_short_name}")
        logger.info(sep)
        logger.info(f"  tau_e = {tau_e:.4f}   (entropy threshold)")
        logger.info(f"  tau_d = {tau_d:.4f}   (disagreement threshold)")
        logger.info("")
        logger.info("  Entropy stage (semantic load zone tokens):")
        for k, v in entropy_stats.items():
            if isinstance(v, float):
                logger.info(f"    {k}: {v:.4f}")
        logger.info("")
        logger.info("  Disagreement stage (entropy-triggered tokens):")
        for k, v in disagreement_stats.items():
            if isinstance(v, float):
                logger.info(f"    {k}: {v:.4f}")
        logger.info("")
        logger.info("  Signal comparison — AUROC for predicting step errors:")
        for k, v in signal_comparison.items():
            if isinstance(v, float):
                logger.info(f"    {k}: {v:.4f}")
        logger.info(sep)