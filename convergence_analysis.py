"""
Convergence & Per-Problem Analysis Data Collection

Runs DAD on a dataset and stores detailed per-round data for each problem:
  - Per-round: confidence, answer entropy, n_agreed, n_disputed, 
    answer distribution, workspace length, tokens used
  - Per-problem: greedy/maj/dad correctness, category classification
  - Overall: convergence curves, per-problem categorization table

Usage:
    python run_convergence_analysis.py --config configs/dad_config_qwen3.yaml \
        --dataset aime_2024 --n_problems -1

Output:
    results/convergence_data.json  — full per-round data for every problem
    results/per_problem_cats.json  — categorization (A/B/C/D) for each problem
    results/convergence_summary.json — aggregated stats per round
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from collections import defaultdict, Counter

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


def run_all_methods(model, tokenizer, problems, cfg):
    """Run greedy, sampling+vote, and DAD on all problems.
    Returns detailed per-problem results with per-round DAD data.
    """
    from src.data.dataset import (
        format_prompt, extract_boxed_answer, extract_numeric_answer,
        answers_match, normalize_answer,
    )
    from src.dad.dad_generator import DADGenerator

    device = cfg["model"]["device"]
    dad_cfg = cfg.get("dad", {})
    max_tokens_greedy = cfg["model"].get("max_new_tokens", 2048)
    max_tokens_dad = dad_cfg.get("max_gen_tokens", 2048)
    m_samples = dad_cfg.get("m_samples", 8)
    temperature = dad_cfg.get("temperature", 0.7)
    top_p = dad_cfg.get("top_p", 0.95)

    generator = DADGenerator(model, tokenizer, cfg)

    all_results = []

    pbar = tqdm(problems, desc="Full Analysis", unit="prob")

    for prob_idx, problem in enumerate(pbar):
        prompt = format_prompt(problem, cfg["model"]["name"])
        prompt_ids = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=4096,
        )["input_ids"].to(device)

        result = {
            "problem_id": problem["problem_id"],
            "question": problem["question"],
            "gold_answer": problem["gold_answer"],
            "source": problem.get("source", ""),
            "level": problem.get("level", ""),
            "problem_type": problem.get("problem_type", ""),
        }

        # ── 1. Greedy ────────────────────────────────────────────
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                input_ids=prompt_ids,
                max_new_tokens=max_tokens_greedy,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        greedy_gen = out[0, prompt_ids.shape[1]:]
        greedy_text = tokenizer.decode(greedy_gen, skip_special_tokens=True)
        greedy_answer = extract_boxed_answer(greedy_text) or extract_numeric_answer(greedy_text) or ""
        greedy_correct = answers_match(greedy_answer, problem["gold_answer"])
        greedy_tokens = len(greedy_gen)
        greedy_wall = time.time() - t0

        result["greedy"] = {
            "answer": greedy_answer,
            "correct": greedy_correct,
            "tokens": greedy_tokens,
            "wall_time": greedy_wall,
        }

        del out
        torch.cuda.empty_cache()

        # ── 2. Sampling + Vote ───────────────────────────────────
        t0 = time.time()
        sample_answers = []
        sample_tokens = 0
        with torch.no_grad():
            for _ in range(m_samples):
                out = model.generate(
                    input_ids=prompt_ids,
                    max_new_tokens=max_tokens_dad,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                gen_ids = out[0, prompt_ids.shape[1]:]
                gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                ans = extract_boxed_answer(gen_text) or extract_numeric_answer(gen_text) or ""
                sample_answers.append(ans)
                sample_tokens += len(gen_ids)
                del out
                torch.cuda.empty_cache()

        maj_wall = time.time() - t0

        # Majority vote
        normalized = [normalize_answer(a) for a in sample_answers]
        counts = Counter(normalized)
        best_norm = counts.most_common(1)[0][0] if counts else ""
        best_ans = next((a for a, n in zip(sample_answers, normalized) if n == best_norm), "")
        maj_correct = answers_match(best_ans, problem["gold_answer"])

        result["sampling_vote"] = {
            "answers": sample_answers,
            "answer_distribution": dict(counts),
            "majority_answer": best_ans,
            "correct": maj_correct,
            "tokens": sample_tokens,
            "wall_time": maj_wall,
        }

        # ── 3. DAD (with per-round data capture) ─────────────────
        t0 = time.time()
        try:
            gen = generator.generate(prompt_ids, problem_text=problem["question"])
            dad_answer = gen.extracted_answer
            dad_correct = answers_match(dad_answer, problem["gold_answer"])

            result["dad"] = {
                "answer": dad_answer,
                "correct": dad_correct,
                "tokens": gen.total_tokens,
                "wall_time": gen.wall_time_sec,
                "n_rounds": gen.n_rounds,
                "n_total_generations": gen.n_total_generations,
                # Per-round convergence data
                "answer_entropy_per_round": gen.answer_entropy_per_round,
                "confidence_per_round": gen.confidence_per_round,
                # Final disagreement map
                "disagreement_map": gen.disagreement_map,
            }
        except Exception as e:
            logger.warning(f"DAD failed on problem {problem['problem_id']}: {e}")
            dad_correct = False
            result["dad"] = {
                "answer": "",
                "correct": False,
                "tokens": 0,
                "wall_time": time.time() - t0,
                "n_rounds": 0,
                "n_total_generations": 0,
                "answer_entropy_per_round": [],
                "confidence_per_round": [],
                "disagreement_map": {},
                "error": str(e),
            }

        torch.cuda.empty_cache()

        # ── 4. Categorize the problem ────────────────────────────
        g = result["greedy"]["correct"]
        m = result["sampling_vote"]["correct"]
        d = result["dad"]["correct"]

        if g and m and d:
            category = "A"  # All methods correct
            cat_desc = "all_correct"
        elif not g and not m and not d:
            category = "D"  # All methods wrong
            cat_desc = "all_wrong"
        elif d and not m:
            category = "C"  # Only DAD correct (unique DAD contribution)
            cat_desc = "dad_only"
        elif m and not d:
            category = "E"  # Only voting correct, DAD wrong
            cat_desc = "maj_only"
        elif not g and m and d:
            category = "B"  # Both voting and DAD help
            cat_desc = "both_help"
        elif g and not m and d:
            category = "F"  # Voting hurts, DAD recovers
            cat_desc = "vote_hurts_dad_recovers"
        elif g and not m and not d:
            category = "G"  # Voting and DAD both hurt
            cat_desc = "greedy_only"
        else:
            category = "X"  # Other
            cat_desc = "other"

        result["category"] = category
        result["category_desc"] = cat_desc

        all_results.append(result)

        # Update progress bar
        n = len(all_results)
        g_acc = sum(1 for r in all_results if r["greedy"]["correct"]) / n
        m_acc = sum(1 for r in all_results if r["sampling_vote"]["correct"]) / n
        d_acc = sum(1 for r in all_results if r["dad"]["correct"]) / n
        pbar.set_postfix(
            G=f"{g_acc:.3f}",
            M=f"{m_acc:.3f}",
            D=f"{d_acc:.3f}",
            cat=category,
        )

    pbar.close()
    return all_results


def compute_convergence_summary(all_results):
    """Aggregate per-round convergence statistics across all problems."""
    max_rounds = max(
        r["dad"]["n_rounds"] for r in all_results if r["dad"]["n_rounds"] > 0
    )

    summary = {}
    for round_idx in range(max_rounds):
        entropies = []
        confidences = []
        for r in all_results:
            ent_list = r["dad"].get("answer_entropy_per_round", [])
            conf_list = r["dad"].get("confidence_per_round", [])
            if round_idx < len(ent_list):
                entropies.append(ent_list[round_idx])
            if round_idx < len(conf_list):
                confidences.append(conf_list[round_idx])

        summary[f"round_{round_idx + 1}"] = {
            "n_problems_active": len(entropies),
            "mean_entropy": float(np.mean(entropies)) if entropies else None,
            "median_entropy": float(np.median(entropies)) if entropies else None,
            "std_entropy": float(np.std(entropies)) if entropies else None,
            "mean_confidence": float(np.mean(confidences)) if confidences else None,
            "median_confidence": float(np.median(confidences)) if confidences else None,
            "std_confidence": float(np.std(confidences)) if confidences else None,
        }

    return summary


def compute_category_table(all_results):
    """Compute per-problem categorization table."""
    cat_counts = Counter(r["category"] for r in all_results)
    cat_descs = {}
    for r in all_results:
        cat_descs[r["category"]] = r["category_desc"]

    table = {}
    for cat in sorted(cat_counts.keys()):
        problems_in_cat = [r for r in all_results if r["category"] == cat]
        table[cat] = {
            "description": cat_descs.get(cat, ""),
            "count": cat_counts[cat],
            "fraction": cat_counts[cat] / len(all_results),
            "problem_ids": [r["problem_id"] for r in problems_in_cat],
            "mean_dad_rounds": float(np.mean([
                r["dad"]["n_rounds"] for r in problems_in_cat if r["dad"]["n_rounds"] > 0
            ])) if any(r["dad"]["n_rounds"] > 0 for r in problems_in_cat) else 0,
        }

    return table


def print_summary(all_results, convergence_summary, category_table):
    """Print summary tables to console."""
    n = len(all_results)

    # Overall accuracy
    g_acc = sum(1 for r in all_results if r["greedy"]["correct"]) / n
    m_acc = sum(1 for r in all_results if r["sampling_vote"]["correct"]) / n
    d_acc = sum(1 for r in all_results if r["dad"]["correct"]) / n

    logger.info("=" * 60)
    logger.info("OVERALL ACCURACY")
    logger.info(f"  Greedy:         {g_acc:.4f} ({sum(1 for r in all_results if r['greedy']['correct'])}/{n})")
    logger.info(f"  Sampling+Vote:  {m_acc:.4f} ({sum(1 for r in all_results if r['sampling_vote']['correct'])}/{n})")
    logger.info(f"  DAD:            {d_acc:.4f} ({sum(1 for r in all_results if r['dad']['correct'])}/{n})")

    # Convergence
    logger.info("\nCONVERGENCE ACROSS ROUNDS")
    for round_key, stats in convergence_summary.items():
        logger.info(
            f"  {round_key}: active={stats['n_problems_active']}, "
            f"mean_H={stats['mean_entropy']:.3f}, "
            f"mean_kappa={stats['mean_confidence']:.3f}"
        )

    # Category table
    logger.info("\nPER-PROBLEM CATEGORIZATION")
    logger.info(f"{'Cat':<5} {'Description':<30} {'Count':>6} {'Fraction':>10} {'Avg Rounds':>12}")
    logger.info("-" * 65)
    for cat in sorted(category_table.keys()):
        info = category_table[cat]
        logger.info(
            f"{cat:<5} {info['description']:<30} {info['count']:>6} "
            f"{info['fraction']:>10.3f} {info['mean_dad_rounds']:>12.1f}"
        )
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Convergence & Per-Problem Analysis")
    parser.add_argument("--config", default="configs/dad_config_qwen3.yaml")
    parser.add_argument("--dataset", default="aime_2024")
    parser.add_argument("--n_problems", type=int, default=-1)
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()

    setup_logging()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cfg["dataset"]["name"] = args.dataset
    if args.n_problems != -1:
        cfg["dataset"]["n_problems"] = args.n_problems

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_short = cfg["model"].get("name", "model").split("/")[-1]
    dataset_name = cfg["dataset"]["name"]
    prefix = f"{model_short}_{dataset_name}"

    logger.info("=" * 60)
    logger.info("Convergence & Per-Problem Analysis")
    logger.info("=" * 60)
    logger.info(f"  Model:    {cfg['model']['name']}")
    logger.info(f"  Dataset:  {dataset_name}")
    logger.info(f"  M:        {cfg['dad']['m_samples']}")
    logger.info(f"  R:        {cfg['dad']['max_rounds']}")

    # Load model and data
    from src.data.model_loader import ModelLoader
    from src.data.dataset import get_inference_dataset

    model, tokenizer = ModelLoader(cfg).load()

    if hasattr(model, 'config'):
        old_max_pos = getattr(model.config, 'max_position_embeddings', None)
        if old_max_pos and old_max_pos < 32768:
            model.config.max_position_embeddings = 32768
            logger.info(f"  Fixed max_position_embeddings: {old_max_pos} -> 32768")

    problems = get_inference_dataset(cfg)
    logger.info(f"Loaded {len(problems)} problems")

    # Run all methods
    all_results = run_all_methods(model, tokenizer, problems, cfg)

    # Compute summaries
    convergence_summary = compute_convergence_summary(all_results)
    category_table = compute_category_table(all_results)

    # Print
    print_summary(all_results, convergence_summary, category_table)

    # Save everything
    # 1. Full per-problem data (for MATLAB plotting)
    full_path = output_dir / f"{prefix}_convergence_data.json"
    with open(full_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"Saved full data: {full_path}")

    # 2. Convergence summary
    conv_path = output_dir / f"{prefix}_convergence_summary.json"
    with open(conv_path, "w") as f:
        json.dump(convergence_summary, f, indent=2)
    logger.info(f"Saved convergence summary: {conv_path}")

    # 3. Category table
    cat_path = output_dir / f"{prefix}_categories.json"
    with open(cat_path, "w") as f:
        json.dump(category_table, f, indent=2)
    logger.info(f"Saved categories: {cat_path}")

    # 4. MATLAB-friendly flat format for convergence curves
    matlab_data = {
        "problem_ids": [],
        "gold_answers": [],
        "greedy_correct": [],
        "maj_correct": [],
        "dad_correct": [],
        "categories": [],
        "n_rounds": [],
        "entropy_round1": [],
        "entropy_round2": [],
        "entropy_round3": [],
        "confidence_round1": [],
        "confidence_round2": [],
        "confidence_round3": [],
        "n_agreed_final": [],
        "n_disputed_final": [],
        "greedy_tokens": [],
        "maj_tokens": [],
        "dad_tokens": [],
    }

    for r in all_results:
        matlab_data["problem_ids"].append(r["problem_id"])
        matlab_data["gold_answers"].append(r["gold_answer"])
        matlab_data["greedy_correct"].append(int(r["greedy"]["correct"]))
        matlab_data["maj_correct"].append(int(r["sampling_vote"]["correct"]))
        matlab_data["dad_correct"].append(int(r["dad"]["correct"]))
        matlab_data["categories"].append(r["category"])
        matlab_data["n_rounds"].append(r["dad"]["n_rounds"])

        ent = r["dad"].get("answer_entropy_per_round", [])
        conf = r["dad"].get("confidence_per_round", [])
        dmap = r["dad"].get("disagreement_map", {})

        matlab_data["entropy_round1"].append(ent[0] if len(ent) > 0 else -1)
        matlab_data["entropy_round2"].append(ent[1] if len(ent) > 1 else -1)
        matlab_data["entropy_round3"].append(ent[2] if len(ent) > 2 else -1)
        matlab_data["confidence_round1"].append(conf[0] if len(conf) > 0 else -1)
        matlab_data["confidence_round2"].append(conf[1] if len(conf) > 1 else -1)
        matlab_data["confidence_round3"].append(conf[2] if len(conf) > 2 else -1)
        matlab_data["n_agreed_final"].append(dmap.get("n_agreed", 0))
        matlab_data["n_disputed_final"].append(dmap.get("n_disputed", 0))
        matlab_data["greedy_tokens"].append(r["greedy"]["tokens"])
        matlab_data["maj_tokens"].append(r["sampling_vote"]["tokens"])
        matlab_data["dad_tokens"].append(r["dad"]["tokens"])

    matlab_path = output_dir / f"{prefix}_matlab_data.json"
    with open(matlab_path, "w") as f:
        json.dump(matlab_data, f, indent=2)
    logger.info(f"Saved MATLAB-friendly data: {matlab_path}")

    logger.info(f"\nAll outputs in: {output_dir}")


if __name__ == "__main__":
    main()