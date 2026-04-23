"""
Convergence Data Collection: Per-round kappa and H_ans tracking.

Runs DAD on a dataset and records per-round convergence metrics
for each problem, split by whether DAD ultimately got the answer correct.

Usage:
    python run_convergence_data.py --config configs/dad_config.yaml \
        --dataset gsm8k --n_problems 200

Output:
    results/convergence_raw.json     — per-problem per-round data
    results/convergence_summary.json — aggregated stats for plotting
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from collections import defaultdict

import torch
import yaml
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Convergence Data Collection")
    parser.add_argument("--config", default="configs/dad_config.yaml")
    parser.add_argument("--dataset", default="gsm8k")
    parser.add_argument("--n_problems", type=int, default=200)
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()

    setup_logging()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cfg["dataset"]["name"] = args.dataset
    cfg["dataset"]["n_problems"] = args.n_problems

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_short = cfg["model"].get("name", "model").split("/")[-1]
    prefix = f"{model_short}_{args.dataset}"

    logger.info("=" * 60)
    logger.info("Convergence Data Collection")
    logger.info("=" * 60)
    logger.info(f"  Model:    {cfg['model']['name']}")
    logger.info(f"  Dataset:  {args.dataset}")
    logger.info(f"  N:        {args.n_problems}")
    logger.info(f"  M:        {cfg['dad']['m_samples']}")
    logger.info(f"  R:        {cfg['dad']['max_rounds']}")

    # Load model and data
    from src.data.model_loader import ModelLoader
    from src.data.dataset import get_inference_dataset, format_prompt, answers_match

    model, tokenizer = ModelLoader(cfg).load()

    if hasattr(model, 'config'):
        old_max_pos = getattr(model.config, 'max_position_embeddings', None)
        if old_max_pos and old_max_pos < 32768:
            model.config.max_position_embeddings = 32768
            logger.info(f"  Fixed max_position_embeddings: {old_max_pos} -> 32768")

    problems = get_inference_dataset(cfg)
    logger.info(f"Loaded {len(problems)} problems")

    # Import DAD generator
    from src.dad.dad_generator import DADGenerator
    generator = DADGenerator(model, tokenizer, cfg)

    # Collect data
    all_data = []
    raw_path = output_dir / f"{prefix}_convergence_raw.json"

    n_correct = 0
    pbar = tqdm(problems, desc="Convergence", unit="prob")

    for prob_idx, problem in enumerate(pbar):
        prompt = format_prompt(problem, cfg["model"]["name"])
        prompt_ids = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=4096,
        )["input_ids"].to(cfg["model"]["device"])

        t0 = time.time()
        try:
            gen = generator.generate(prompt_ids, problem_text=problem["question"])
            dad_answer = gen.extracted_answer
            dad_correct = answers_match(dad_answer, problem["gold_answer"])
            if dad_correct:
                n_correct += 1

            entry = {
                "problem_id": problem["problem_id"],
                "gold_answer": problem["gold_answer"],
                "dad_answer": dad_answer,
                "correct": dad_correct,
                "n_rounds": gen.n_rounds,
                "n_total_generations": gen.n_total_generations,
                "total_tokens": gen.total_tokens,
                "wall_time": time.time() - t0,
                # Per-round data
                "entropy_per_round": gen.answer_entropy_per_round,
                "confidence_per_round": gen.confidence_per_round,
                # Disagreement map from final round
                "n_agreed_final": gen.disagreement_map.get("n_agreed", 0),
                "n_disputed_final": gen.disagreement_map.get("n_disputed", 0),
                "final_answer_entropy": gen.disagreement_map.get("answer_entropy", 0),
                "final_confidence": gen.disagreement_map.get("confidence", 0),
                "final_answer_dist": gen.disagreement_map.get("answer_distribution", {}),
            }
        except Exception as e:
            logger.warning(f"Problem {problem['problem_id']} failed: {e}")
            entry = {
                "problem_id": problem["problem_id"],
                "gold_answer": problem["gold_answer"],
                "dad_answer": "",
                "correct": False,
                "n_rounds": 0,
                "n_total_generations": 0,
                "total_tokens": 0,
                "wall_time": time.time() - t0,
                "entropy_per_round": [],
                "confidence_per_round": [],
                "n_agreed_final": 0,
                "n_disputed_final": 0,
                "final_answer_entropy": 0,
                "final_confidence": 0,
                "final_answer_dist": {},
                "error": str(e),
            }

        all_data.append(entry)

        # Incremental save
        with open(raw_path, "w") as f:
            json.dump(all_data, f, indent=2, default=str)

        acc = n_correct / (prob_idx + 1)
        pbar.set_postfix(
            acc=f"{acc:.3f}",
            rounds=entry["n_rounds"],
            correct="Y" if entry["correct"] else "N",
        )

        torch.cuda.empty_cache()

    pbar.close()

    # ── Compute summary stats for MATLAB ─────────────────────────
    max_rounds = max(e["n_rounds"] for e in all_data if e["n_rounds"] > 0)

    correct_problems = [e for e in all_data if e["correct"]]
    wrong_problems = [e for e in all_data if not e["correct"]]

    summary = {
        "model": cfg["model"]["name"],
        "dataset": args.dataset,
        "n_problems": len(all_data),
        "n_correct": len(correct_problems),
        "n_wrong": len(wrong_problems),
        "accuracy": len(correct_problems) / len(all_data) if all_data else 0,
        "max_rounds": max_rounds,
        "rounds": {},
    }

    for r in range(max_rounds):
        round_key = f"round_{r + 1}"

        # Correct problems
        ent_correct = [e["entropy_per_round"][r] for e in correct_problems
                       if r < len(e["entropy_per_round"])]
        conf_correct = [e["confidence_per_round"][r] for e in correct_problems
                        if r < len(e["confidence_per_round"])]

        # Wrong problems
        ent_wrong = [e["entropy_per_round"][r] for e in wrong_problems
                     if r < len(e["entropy_per_round"])]
        conf_wrong = [e["confidence_per_round"][r] for e in wrong_problems
                      if r < len(e["confidence_per_round"])]

        summary["rounds"][round_key] = {
            "correct": {
                "n_active": len(ent_correct),
                "entropy_mean": float(np.mean(ent_correct)) if ent_correct else None,
                "entropy_std": float(np.std(ent_correct)) if ent_correct else None,
                "entropy_median": float(np.median(ent_correct)) if ent_correct else None,
                "confidence_mean": float(np.mean(conf_correct)) if conf_correct else None,
                "confidence_std": float(np.std(conf_correct)) if conf_correct else None,
                "confidence_median": float(np.median(conf_correct)) if conf_correct else None,
            },
            "wrong": {
                "n_active": len(ent_wrong),
                "entropy_mean": float(np.mean(ent_wrong)) if ent_wrong else None,
                "entropy_std": float(np.std(ent_wrong)) if ent_wrong else None,
                "entropy_median": float(np.median(ent_wrong)) if ent_wrong else None,
                "confidence_mean": float(np.mean(conf_wrong)) if conf_wrong else None,
                "confidence_std": float(np.std(conf_wrong)) if conf_wrong else None,
                "confidence_median": float(np.median(conf_wrong)) if conf_wrong else None,
            },
            "all": {
                "n_active": len(ent_correct) + len(ent_wrong),
                "entropy_mean": float(np.mean(ent_correct + ent_wrong)) if (ent_correct + ent_wrong) else None,
                "confidence_mean": float(np.mean(conf_correct + conf_wrong)) if (conf_correct + conf_wrong) else None,
            },
        }

    # MATLAB-friendly flat arrays
    matlab = {
        "problem_ids": [e["problem_id"] for e in all_data],
        "correct": [int(e["correct"]) for e in all_data],
        "n_rounds": [e["n_rounds"] for e in all_data],
        "total_tokens": [e["total_tokens"] for e in all_data],
    }

    # Pad entropy/confidence arrays to max_rounds with -1
    for r in range(max_rounds):
        ent_key = f"entropy_r{r+1}"
        conf_key = f"confidence_r{r+1}"
        matlab[ent_key] = []
        matlab[conf_key] = []
        for e in all_data:
            ent = e["entropy_per_round"]
            conf = e["confidence_per_round"]
            matlab[ent_key].append(ent[r] if r < len(ent) else -1)
            matlab[conf_key].append(conf[r] if r < len(conf) else -1)

    summary["matlab"] = matlab

    summary_path = output_dir / f"{prefix}_convergence_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ── Print summary ────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"Accuracy: {summary['accuracy']:.4f} ({summary['n_correct']}/{summary['n_problems']})")
    logger.info(f"\n{'Round':<8} {'Group':<10} {'N':>5} {'H_mean':>8} {'H_std':>8} {'K_mean':>8} {'K_std':>8}")
    logger.info("-" * 60)
    for r_key, r_data in summary["rounds"].items():
        for group in ["correct", "wrong"]:
            g = r_data[group]
            if g["n_active"] > 0:
                logger.info(
                    f"{r_key:<8} {group:<10} {g['n_active']:>5} "
                    f"{g['entropy_mean']:>8.3f} {g['entropy_std']:>8.3f} "
                    f"{g['confidence_mean']:>8.3f} {g['confidence_std']:>8.3f}"
                )
    logger.info("=" * 60)
    logger.info(f"Raw data: {raw_path}")
    logger.info(f"Summary:  {summary_path}")


if __name__ == "__main__":
    main()