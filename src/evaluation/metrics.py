import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ProblemResult:
    problem_id: int
    source: str
    question: str
    gold_answer: str
    predicted_answer: Optional[str]
    correct: bool
    total_tokens: int
    n_entropy_triggers: int
    n_expansion_triggers: int
    total_expansion_tokens: int
    trigger_rate_entropy: float
    trigger_rate_expansion: float
    wall_time_sec: float
    decoding_mode: str
    has_boxed: bool = False
    level: str = ""
    problem_type: str = ""


@dataclass
class AggregateMetrics:
    decoding_mode: str
    n_problems: int
    accuracy: float
    accuracy_with_boxed: float
    mean_tokens: float
    median_tokens: float
    std_tokens: float
    mean_wall_time: float
    total_wall_time: float
    mean_entropy_trigger_rate: float
    mean_expansion_trigger_rate: float
    mean_expansion_tokens: float
    expansion_overhead_pct: float
    accuracy_by_level: dict = field(default_factory=dict)
    accuracy_by_type: dict = field(default_factory=dict)


class MetricsAggregator:
    def __init__(self, decoding_mode: str, cfg: dict):
        self.decoding_mode = decoding_mode
        self.cfg = cfg
        self.results: list[ProblemResult] = []

    def add(self, result: ProblemResult) -> None:
        self.results.append(result)

    def compute(self) -> AggregateMetrics:
        if not self.results:
            return AggregateMetrics(
                decoding_mode=self.decoding_mode, n_problems=0,
                accuracy=0.0, accuracy_with_boxed=0.0,
                mean_tokens=0.0, median_tokens=0.0, std_tokens=0.0,
                mean_wall_time=0.0, total_wall_time=0.0,
                mean_entropy_trigger_rate=0.0, mean_expansion_trigger_rate=0.0,
                mean_expansion_tokens=0.0, expansion_overhead_pct=0.0,
            )

        tokens = [r.total_tokens for r in self.results]
        exp_tokens = [r.total_expansion_tokens for r in self.results]
        mean_tok = float(np.mean(tokens))
        mean_exp = float(np.mean(exp_tokens))
        base_tok = max(1.0, mean_tok - mean_exp)
        overhead_pct = (mean_exp / base_tok * 100) if mean_exp > 0 else 0.0

        with_boxed = [r for r in self.results if r.has_boxed]
        acc_boxed = sum(r.correct for r in with_boxed) / max(1, len(with_boxed))

        by_level: dict[str, list[bool]] = {}
        by_type: dict[str, list[bool]] = {}
        for r in self.results:
            if r.level:
                by_level.setdefault(r.level, []).append(r.correct)
            if r.problem_type:
                by_type.setdefault(r.problem_type, []).append(r.correct)

        return AggregateMetrics(
            decoding_mode=self.decoding_mode,
            n_problems=len(self.results),
            accuracy=float(np.mean([r.correct for r in self.results])),
            accuracy_with_boxed=float(acc_boxed),
            mean_tokens=mean_tok,
            median_tokens=float(np.median(tokens)),
            std_tokens=float(np.std(tokens)),
            mean_wall_time=float(np.mean([r.wall_time_sec for r in self.results])),
            total_wall_time=float(np.sum([r.wall_time_sec for r in self.results])),
            mean_entropy_trigger_rate=float(np.mean([r.trigger_rate_entropy for r in self.results])),
            mean_expansion_trigger_rate=float(np.mean([r.trigger_rate_expansion for r in self.results])),
            mean_expansion_tokens=mean_exp,
            expansion_overhead_pct=overhead_pct,
            accuracy_by_level={k: float(np.mean(v)) for k, v in by_level.items()},
            accuracy_by_type={k: float(np.mean(v)) for k, v in by_type.items()},
        )

    def save_results(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        import jsonlines
        with jsonlines.open(path, mode="w") as writer:
            writer.write_all([asdict(r) for r in self.results])
        logger.info(f"Saved {len(self.results)} results to {path}")

    def save_metrics(self, metrics: AggregateMetrics, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(metrics), f, indent=2)
        logger.info(f"Saved aggregate metrics to {path}")

    def print_summary(self, metrics: AggregateMetrics) -> None:
        sep = "=" * 60
        logger.info(sep)
        logger.info(f"RESULTS — {metrics.decoding_mode.upper()}")
        logger.info(sep)
        logger.info(f"  Problems          : {metrics.n_problems}")
        logger.info(f"  Accuracy          : {metrics.accuracy:.4f}  ({metrics.accuracy*100:.1f}%)")
        logger.info(f"  Accuracy (boxed)  : {metrics.accuracy_with_boxed:.4f}")
        logger.info(f"  Mean tokens       : {metrics.mean_tokens:.1f}")
        logger.info(f"  Median tokens     : {metrics.median_tokens:.1f}")
        logger.info(f"  Expansion overhead: {metrics.expansion_overhead_pct:.1f}%")
        logger.info(f"  H trigger rate    : {metrics.mean_entropy_trigger_rate:.4f}")
        logger.info(f"  Expansion rate    : {metrics.mean_expansion_trigger_rate:.4f}")
        logger.info(f"  Mean exp tokens   : {metrics.mean_expansion_tokens:.1f}")
        logger.info(f"  Mean wall time(s) : {metrics.mean_wall_time:.2f}")
        logger.info(f"  Total wall time(s): {metrics.total_wall_time:.1f}")
        if metrics.accuracy_by_level:
            logger.info("  By level:")
            for lvl in sorted(metrics.accuracy_by_level):
                logger.info(f"    {lvl}: {metrics.accuracy_by_level[lvl]:.4f}")
        if metrics.accuracy_by_type:
            logger.info("  By type (top 5):")
            for t, v in sorted(metrics.accuracy_by_type.items(), key=lambda x: -x[1])[:5]:
                logger.info(f"    {t}: {v:.4f}")
        logger.info(sep)


def compare_methods(metrics_list: list[AggregateMetrics]) -> dict:
    if not metrics_list:
        return {}
    baseline = next((m for m in metrics_list if m.decoding_mode == "greedy"), metrics_list[0])
    return {
        m.decoding_mode: {
            "accuracy": m.accuracy,
            "delta_vs_greedy": m.accuracy - baseline.accuracy,
            "mean_tokens": m.mean_tokens,
            "token_ratio_vs_greedy": m.mean_tokens / max(1.0, baseline.mean_tokens),
            "expansion_overhead_pct": m.expansion_overhead_pct,
            "mean_entropy_trigger_rate": m.mean_entropy_trigger_rate,
            "mean_expansion_trigger_rate": m.mean_expansion_trigger_rate,
            "mean_wall_time": m.mean_wall_time,
            "speed_ratio_vs_greedy": m.mean_wall_time / max(1.0, baseline.mean_wall_time),
        }
        for m in metrics_list
    }