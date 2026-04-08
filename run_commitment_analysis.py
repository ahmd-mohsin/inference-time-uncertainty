"""
run_commitment_analysis.py

Validates the Pre-Token Commitment Signal on existing MATH500 predictions
before running full PTCS inference. Asks two empirical questions:

Q1: Is commitment divergence predictive of final answer correctness?
    (AUC of divergence score as a binary predictor of incorrect answers)

Q2: Do incorrect problems show higher or lower divergence than correct ones?
    (This tells us whether the signal fires in the right direction)

Run on 25-50 problems first. Expected runtime ~2 min for 25 problems with K=5.

Usage:
    python run_commitment_analysis.py \
        --predictions data/inference_outputs/predictions_qwen2.5-math-7b_greedy.jsonl \
        --config configs/inference_config.yaml \
        --model_short_name qwen2.5-math-7b \
        --n_problems 50 \
        --k_probes 5 \
        --noise_std 0.01 \
        --out data/commitment_analysis.json
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
import yaml
import jsonlines
import numpy as np

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def compute_auc(scores: list[float], labels: list[int]) -> float:
    pairs = sorted(zip(scores, labels), key=lambda x: -x[0])
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tp = fp = 0
    auc = 0.0
    prev_fp = 0
    for _, label in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
            auc += tp * (fp - prev_fp)
            prev_fp = fp
    return auc / (n_pos * n_neg)


def run_analysis(args) -> None:
    import torch
    from src.data.model_loader import ModelLoader
    from src.data.dataset import format_prompt, load_math500
    from src.uncertainty.pretokencommitment import PreTokenCommitmentDetector

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.model_short_name:
        cfg["model"]["short_name"] = args.model_short_name

    model, tokenizer = ModelLoader(cfg).load()
    device = cfg["model"]["device"]

    detector = PreTokenCommitmentDetector(
        k_probes=args.k_probes,
        noise_std=args.noise_std,
        divergence_threshold=args.divergence_threshold,
        device=device,
    )

    with jsonlines.open(args.predictions) as reader:
        preds = list(reader)

    if args.n_problems > 0:
        preds = preds[:args.n_problems]

    pred_map = {p["problem_id"]: p for p in preds}
    problems = load_math500(n_problems=-1)
    problems = [p for p in problems if p["problem_id"] in pred_map]
    if args.n_problems > 0:
        problems = problems[:args.n_problems]

    logger.info(f"Running commitment analysis on {len(problems)} problems")
    logger.info(f"K={args.k_probes}  noise_std={args.noise_std}  threshold={args.divergence_threshold}")

    records = []
    divergences_correct = []
    divergences_incorrect = []

    for i, problem in enumerate(problems):
        pred = pred_map.get(problem["problem_id"])
        if pred is None:
            continue

        prompt = format_prompt(problem, cfg["model"]["name"])
        prompt_ids = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=4096
        )["input_ids"].to(device)

        t0 = time.time()
        probe = detector.probe(model, prompt_ids)
        probe_time = time.time() - t0

        is_correct = bool(pred.get("correct", False))

        record = {
            "problem_id": problem["problem_id"],
            "correct": is_correct,
            "divergence": probe.divergence,
            "is_unstable": probe.is_unstable,
            "selected_probe_idx": probe.selected_probe_idx,
            "probe_time_sec": probe_time,
            "greedy_answer": pred.get("extracted_answer"),
            "gold_answer": problem["gold_answer"],
        }
        records.append(record)

        if is_correct:
            divergences_correct.append(probe.divergence)
        else:
            divergences_incorrect.append(probe.divergence)

        if (i + 1) % 10 == 0 or i == 0:
            logger.info(
                f"  [{i+1}/{len(problems)}] "
                f"D={probe.divergence:.4f}  unstable={probe.is_unstable}  "
                f"correct={is_correct}  probe_time={probe_time:.2f}s"
            )

    all_divs = [r["divergence"] for r in records]
    all_labels = [int(not r["correct"]) for r in records]

    auc = compute_auc(all_divs, all_labels)

    mean_correct = float(np.mean(divergences_correct)) if divergences_correct else 0.0
    mean_incorrect = float(np.mean(divergences_incorrect)) if divergences_incorrect else 0.0
    unstable_rate = sum(1 for r in records if r["is_unstable"]) / max(1, len(records))

    summary = {
        "n_problems": len(records),
        "n_correct": len(divergences_correct),
        "n_incorrect": len(divergences_incorrect),
        "mean_divergence_correct": mean_correct,
        "mean_divergence_incorrect": mean_incorrect,
        "divergence_delta": mean_incorrect - mean_correct,
        "auc_divergence_predicts_incorrect": auc,
        "unstable_rate": unstable_rate,
        "k_probes": args.k_probes,
        "noise_std": args.noise_std,
        "divergence_threshold": args.divergence_threshold,
        "divergence_percentiles": {
            "p25": float(np.percentile(all_divs, 25)),
            "p50": float(np.percentile(all_divs, 50)),
            "p75": float(np.percentile(all_divs, 75)),
            "p90": float(np.percentile(all_divs, 90)),
            "p95": float(np.percentile(all_divs, 95)),
        },
    }

    logger.info("=" * 60)
    logger.info("COMMITMENT ANALYSIS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Problems:               {summary['n_problems']}")
    logger.info(f"  Correct / Incorrect:    {summary['n_correct']} / {summary['n_incorrect']}")
    logger.info(f"  Mean D (correct):       {mean_correct:.5f}")
    logger.info(f"  Mean D (incorrect):     {mean_incorrect:.5f}")
    logger.info(f"  Delta (incorr - corr):  {summary['divergence_delta']:+.5f}")
    logger.info(f"  AUC (D → incorrect):    {auc:.4f}  [0.5=chance, 1.0=perfect]")
    logger.info(f"  Unstable rate:          {unstable_rate:.1%}")
    logger.info(f"  Divergence p50/p90/p95: "
                f"{summary['divergence_percentiles']['p50']:.5f} / "
                f"{summary['divergence_percentiles']['p90']:.5f} / "
                f"{summary['divergence_percentiles']['p95']:.5f}")
    logger.info("=" * 60)

    if summary["divergence_delta"] > 0:
        logger.info("SIGNAL DIRECTION: incorrect problems have HIGHER divergence (expected)")
    elif summary["divergence_delta"] < 0:
        logger.info("SIGNAL DIRECTION: incorrect problems have LOWER divergence (entropy inversion pattern)")
    else:
        logger.info("SIGNAL DIRECTION: no difference")

    if auc > 0.6:
        logger.info(f"AUC={auc:.3f}: divergence is a useful predictor — proceed with PTCS")
    elif auc > 0.5:
        logger.info(f"AUC={auc:.3f}: weak signal — adjust noise_std or k_probes before PTCS")
    else:
        logger.info(f"AUC={auc:.3f}: no signal — signal may be inverted or noise_std is too high")

    output = {"summary": summary, "records": records}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results saved to {args.out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True,
                   help="Path to existing greedy/digte predictions jsonl")
    p.add_argument("--config", default="configs/inference_config.yaml")
    p.add_argument("--model_short_name", default=None)
    p.add_argument("--n_problems", type=int, default=50)
    p.add_argument("--k_probes", type=int, default=5)
    p.add_argument("--noise_std", type=float, default=0.01)
    p.add_argument("--divergence_threshold", type=float, default=0.05)
    p.add_argument("--out", default="data/commitment_analysis.json")
    return p.parse_args()


if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    run_analysis(args)