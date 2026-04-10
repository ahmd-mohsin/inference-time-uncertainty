import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import yaml
import numpy as np

logger = logging.getLogger(__name__)

MODES = ["rmi_tree", "rebase", "sampling_vote"]

def setup_logging(cfg: dict, mode: str) -> None:
    log_cfg = cfg.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO"), logging.INFO)
    short = cfg["model"]["short_name"]
    log_file = (
        log_cfg.get("log_file", f"logs/inference_{short}_{mode}.log")
        .replace("{model_short_name}", short)
        .replace("{decoding_mode}", mode)
    )
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)],
        force=True,
    )

def build_generator(mode: str, model, tokenizer, prm, cfg: dict):
    if mode == "rebase":
        from src.search.rebase_baseline import REBASEGenerator
        return REBASEGenerator(model, tokenizer, prm, cfg)

    elif mode == "rmi_tree":
        from src.search.rmi_tree_search import RMITreeSearchGenerator
        return RMITreeSearchGenerator(model, tokenizer, prm, cfg)

    elif mode == "sampling_vote":
        from src.search.sampling_vote import SamplingVoteGenerator
        return SamplingVoteGenerator(model, tokenizer, prm, cfg)

    raise ValueError(f"Unknown mode: {mode}")

def run(cfg: dict, mode: str) -> None:
    import jsonlines
    from src.data.dataset import get_inference_dataset, format_prompt, answers_match
    from src.data.model_loader import ModelLoader
    from src.reward.prm import load_prm

    loader = ModelLoader(cfg)
    model, tokenizer = loader.load()

    prm = load_prm(cfg)

    generator = build_generator(mode, model, tokenizer, prm, cfg)

    problems = get_inference_dataset(cfg)
    logger.info(f"Running {len(problems)} problems  mode={mode}")

    short = cfg["model"]["short_name"]
    lam = cfg.get("search", {}).get("lambda_diversity", 0.0)
    n_sol = cfg.get("search", {}).get("n_solutions", 32)
    mode_tag = f"{mode}_lam{lam}_n{n_sol}"

    pred_path = (
        cfg["output"]["predictions_file"]
        .replace("{model_short_name}", short)
        .replace("{decoding_mode}", mode_tag)
    )
    metrics_path = (
        cfg["output"]["metrics_file"]
        .replace("{model_short_name}", short)
        .replace("{decoding_mode}", mode_tag)
    )
    Path(pred_path).parent.mkdir(parents=True, exist_ok=True)

    results = []
    n_correct = 0

    with jsonlines.open(pred_path, mode="w") as writer:
        for i, problem in enumerate(problems):
            prompt = format_prompt(problem, cfg["model"]["name"])
            prompt_ids = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=4096
            )["input_ids"].to(cfg["model"]["device"])

            try:
                gen = generator.generate(prompt_ids, problem_text=problem["question"])
            except Exception as e:
                logger.warning(f"Problem {problem['problem_id']} failed: {e}")
                import traceback
                traceback.print_exc()
                continue

            correct = answers_match(gen.extracted_answer, problem["gold_answer"])
            if correct:
                n_correct += 1

            record = {
                "problem_id": problem["problem_id"],
                "source": problem.get("source", ""),
                "question": problem["question"],
                "gold_answer": problem["gold_answer"],
                "extracted_answer": gen.extracted_answer,
                "correct": correct,
                "generated_text": gen.generated_text,
                "total_tokens": gen.total_tokens,
                "wall_time_sec": gen.wall_time_sec,
                "n_completed_solutions": gen.n_completed_solutions,
                "n_prm_calls": gen.n_prm_calls,
                "tree_max_depth": gen.tree_max_depth,
                "mean_prm_reward": gen.mean_prm_reward,
                "mean_diversity_score": gen.mean_diversity_score,
                "selected_method": gen.selected_method,
                "decoding_mode": mode,
                "lambda_diversity": lam,
                "n_solutions_target": n_sol,
                "level": problem.get("level", ""),
                "problem_type": problem.get("problem_type", ""),
            }

            if cfg.get("output", {}).get("log_detail", False) and gen.all_solutions:
                record["all_solutions"] = gen.all_solutions

            writer.write(record)
            results.append(record)

            acc = n_correct / (i + 1)
            logger.info(
                f"  [{i+1}/{len(problems)}] "
                f"correct={correct}  running_acc={acc:.4f}  "
                f"answer={gen.extracted_answer}  gold={problem['gold_answer']}  "
                f"sols={gen.n_completed_solutions}  "
                f"tokens={gen.total_tokens}  "
                f"time={gen.wall_time_sec:.1f}s"
            )

    accuracy = n_correct / max(1, len(results))

    metrics = {
        "mode": mode,
        "model": cfg["model"]["name"],
        "prm": cfg.get("prm", {}).get("model_name", "none"),
        "dataset": cfg["dataset"]["name"],
        "n_problems": len(results),
        "n_correct": n_correct,
        "accuracy": accuracy,
        "lambda_diversity": lam,
        "n_solutions": n_sol,
        "aggregation": cfg.get("search", {}).get("aggregation", "weighted_vote"),
        "diversity_method": cfg.get("search", {}).get("diversity_method", "kl"),
        "balance_temperature": cfg.get("search", {}).get("balance_temperature", 0.1),
        "sampling_temperature": cfg.get("search", {}).get("sampling_temperature", 0.7),
        "max_new_tokens": cfg["model"].get("max_new_tokens", 8192),
        "mean_tokens": float(np.mean([r["total_tokens"] for r in results])) if results else 0,
        "median_tokens": float(np.median([r["total_tokens"] for r in results])) if results else 0,
        "mean_wall_time": float(np.mean([r["wall_time_sec"] for r in results])) if results else 0,
        "total_wall_time": float(np.sum([r["wall_time_sec"] for r in results])) if results else 0,
        "mean_prm_reward": float(np.mean([r["mean_prm_reward"] for r in results])) if results else 0,
        "mean_diversity": float(np.mean([r["mean_diversity_score"] for r in results])) if results else 0,
        "mean_solutions_per_problem": float(np.mean([r["n_completed_solutions"] for r in results])) if results else 0,
        "mean_prm_calls_per_problem": float(np.mean([r["n_prm_calls"] for r in results])) if results else 0,
    }

    from collections import defaultdict
    by_level = defaultdict(list)
    for r in results:
        lvl = r.get("level", "")
        if lvl:
            by_level[lvl].append(r["correct"])
    if by_level:
        metrics["accuracy_by_level"] = {
            k: float(np.mean(v)) for k, v in sorted(by_level.items())
        }

    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    sep = "=" * 60
    logger.info(sep)
    logger.info(f"RESULTS — {mode.upper()}")
    logger.info(sep)
    logger.info(f"  Dataset       : {metrics['dataset']}")
    logger.info(f"  Accuracy      : {accuracy:.4f}  ({n_correct}/{len(results)})")
    logger.info(f"  λ_diversity   : {metrics['lambda_diversity']}")
    logger.info(f"  N solutions   : {metrics['n_solutions']}")
    logger.info(f"  Aggregation   : {metrics['aggregation']}")
    logger.info(f"  Mean tokens   : {metrics['mean_tokens']:.0f}")
    logger.info(f"  Median tokens : {metrics['median_tokens']:.0f}")
    logger.info(f"  Mean time/q   : {metrics['mean_wall_time']:.1f}s")
    logger.info(f"  Total time    : {metrics['total_wall_time']:.0f}s")
    logger.info(f"  Mean PRM      : {metrics['mean_prm_reward']:.4f}")
    logger.info(f"  Mean diversity : {metrics['mean_diversity']:.4f}")
    logger.info(f"  Mean PRM calls: {metrics['mean_prm_calls_per_problem']:.1f}")
    if "accuracy_by_level" in metrics:
        logger.info("  By level:")
        for lvl, acc_l in metrics["accuracy_by_level"].items():
            logger.info(f"    {lvl}: {acc_l:.4f}")
    logger.info(sep)
    logger.info(f"Predictions → {pred_path}")
    logger.info(f"Metrics     → {metrics_path}")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RMI-Guided Tree Search for Math Reasoning")
    p.add_argument("--config", default="configs/rmi_search_config.yaml")
    p.add_argument("--mode", default="rmi_tree", choices=MODES)
    p.add_argument("--model", default=None, help="Override policy model name")
    p.add_argument("--model_short_name", default=None)
    p.add_argument("--prm_model", default=None, help="Override PRM model name")
    p.add_argument("--dataset", default=None)
    p.add_argument("--n_problems", type=int, default=None)
    p.add_argument("--n_solutions", type=int, default=None, help="Solution budget N")
    p.add_argument("--lambda_div", type=float, default=None, help="Diversity weight λ")
    p.add_argument("--aggregation", choices=["weighted_vote", "majority_vote", "best_reward"])
    p.add_argument("--diversity_method", choices=["kl", "hidden"])
    p.add_argument("--temperature", type=float, default=None, help="Sampling temperature")
    p.add_argument("--max_depth", type=int, default=None)
    p.add_argument("--log_detail", action="store_true")
    return p.parse_args()

def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.model:
        cfg["model"]["name"] = args.model
    if args.model_short_name:
        cfg["model"]["short_name"] = args.model_short_name
    if args.prm_model:
        cfg["prm"]["model_name"] = args.prm_model
    if args.dataset:
        cfg["dataset"]["name"] = args.dataset
    if args.n_problems is not None:
        cfg["dataset"]["n_problems"] = args.n_problems
    if args.n_solutions is not None:
        cfg["search"]["n_solutions"] = args.n_solutions
    if args.lambda_div is not None:
        cfg["search"]["lambda_diversity"] = args.lambda_div
    if args.aggregation:
        cfg["search"]["aggregation"] = args.aggregation
    if args.diversity_method:
        cfg["search"]["diversity_method"] = args.diversity_method
    if args.temperature is not None:
        cfg["search"]["sampling_temperature"] = args.temperature
    if args.max_depth is not None:
        cfg["search"]["max_depth"] = args.max_depth
    if args.log_detail:
        cfg["output"]["log_detail"] = True

    mode = args.mode
    if mode == "rebase":
        cfg["search"]["lambda_diversity"] = 0.0

    Path(cfg["output"]["dir"]).mkdir(parents=True, exist_ok=True)
    setup_logging(cfg, mode)

    logger.info("=" * 60)
    logger.info("RMI-Guided Tree Search for Math Reasoning")
    logger.info("=" * 60)
    logger.info(f"  mode           : {mode}")
    logger.info(f"  policy model   : {cfg['model']['name']}")
    logger.info(f"  PRM model      : {cfg['prm']['model_name']}")
    logger.info(f"  dataset        : {cfg['dataset']['name']}")
    logger.info(f"  n_problems     : {cfg['dataset'].get('n_problems', -1)}")
    logger.info(f"  N (solutions)  : {cfg['search']['n_solutions']}")
    logger.info(f"  λ (diversity)  : {cfg['search']['lambda_diversity']}")
    logger.info(f"  aggregation    : {cfg['search']['aggregation']}")
    logger.info(f"  diversity_meth : {cfg['search']['diversity_method']}")
    logger.info(f"  temperature    : {cfg['search']['sampling_temperature']}")
    logger.info(f"  T_balance      : {cfg['search']['balance_temperature']}")
    logger.info(f"  max_depth      : {cfg['search']['max_depth']}")
    logger.info(f"  max_tokens     : {cfg['model']['max_new_tokens']}")

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        logger.info(f"  GPU            : {props.name}  {props.total_mem / 1e9:.1f} GB")
    else:
        logger.warning("  No CUDA GPU — will be very slow.")

    run(cfg, mode)

if __name__ == "__main__":
    main()