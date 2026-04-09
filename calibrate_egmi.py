"""
calibrate_egmi.py

Computes the low-entropy threshold tau_low for EGMI from existing calibration
token data. The threshold is set at the p20 of the entropy distribution within
the strategy window (first W tokens of each generated sequence).

The intuition: the bottom 20% of entropy values are the most confident token
commitments. These are the positions EGMI targets for mixture injection.

Usage:
    python calibrate_egmi.py \
        --token_data data/calibration_outputs/token_data_qwen2.5-math-7b.jsonl \
        --thresholds_file data/calibration_outputs/learned_thresholds.json \
        --model_short_name qwen2.5-math-7b \
        --strategy_window 40 \
        --low_entropy_pct 20
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import jsonlines

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def calibrate(args) -> None:
    logger.info(f"Loading token data from {args.token_data}")
    records = list(jsonlines.open(args.token_data))
    logger.info(f"Loaded {len(records)} token records")

    strategy_entropies = []
    for r in records:
        pos = r.get("position", r.get("step", None))
        entropy = r.get("entropy", None)
        if pos is None or entropy is None:
            continue
        if pos < args.strategy_window:
            if not (entropy != entropy) and entropy < 15.0:
                strategy_entropies.append(float(entropy))

    if not strategy_entropies:
        logger.warning("No strategy-window entropy values found — check field names in token data")
        logger.info("Available fields in first record:")
        if records:
            logger.info(str(list(records[0].keys())))
        return

    strategy_entropies = sorted(strategy_entropies)
    total = len(strategy_entropies)
    logger.info(f"Strategy-window tokens (pos < {args.strategy_window}): {total}")

    tau_low = float(np.percentile(strategy_entropies, args.low_entropy_pct))

    for pct in [5, 10, 20, 30, 50]:
        val = np.percentile(strategy_entropies, pct)
        trigger_rate = sum(1 for e in strategy_entropies if e < val) / total
        logger.info(f"  p{pct}: tau_low={val:.5f}  injection_rate={trigger_rate:.3f}")

    logger.info(f"Selected tau_low = {tau_low:.5f} (p{args.low_entropy_pct})")

    path = args.thresholds_file
    existing = {}
    if Path(path).exists():
        with open(path) as f:
            existing = json.load(f)

    existing.setdefault(args.model_short_name, {})
    existing[args.model_short_name]["tau_low_egmi"] = tau_low
    existing[args.model_short_name]["egmi_strategy_window"] = args.strategy_window
    existing[args.model_short_name]["egmi_low_entropy_pct"] = args.low_entropy_pct

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)
    logger.info(f"Saved tau_low={tau_low:.5f} to {path}[{args.model_short_name}]")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--token_data",
                   default="data/calibration_outputs/token_data_qwen2.5-math-7b.jsonl")
    p.add_argument("--thresholds_file",
                   default="data/calibration_outputs/learned_thresholds.json")
    p.add_argument("--model_short_name", default="qwen2.5-math-7b")
    p.add_argument("--strategy_window", type=int, default=40)
    p.add_argument("--low_entropy_pct", type=float, default=20)
    return p.parse_args()


if __name__ == "__main__":
    setup_logging()
    calibrate(parse_args())