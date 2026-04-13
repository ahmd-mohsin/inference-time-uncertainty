"""
run_dad.py — FIXED version

Fixes applied:
1. max_position_embeddings set to 32768 after model load (prevents 4096 truncation warning)
2. Leading zero normalization in voting (017 == 17)
3. Float gold answer handling in answers_match (142.0 == 142)
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

logger = logging.getLogger(__name__)


def setup_logging(log_file=None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


def _normalize_for_voting(answer):
    """Robust answer normalization for majority voting.
    Handles: leading zeros, float-to-int, whitespace.
    """
    if answer is None or answer == "":
        return ""
    from src.data.dataset import normalize_answer
    n = normalize_answer(answer)
    # Strip leading zeros: '017' -> '17', but '0' stays '0'
    n = n.lstrip("0") or "0"
    try:
        val = float(n)
        if val == int(val) and abs(val) < 1e15:
            return str(int(val))
        return f"{val:.8f}".rstrip("0").rstrip(".")
    except (ValueError, OverflowError):
        pass
    return n


def run_greedy(model, tokenizer, problems, cfg):
    from src.data.dataset import format_prompt, extract_boxed_answer, extract_numeric_answer, answers_match

    device = cfg["model"]["device"]
    max_tokens = cfg["model"].get("max_new_tokens", 2048)
    results = []

    for i, problem in enumerate(problems):
        prompt = format_prompt(problem, cfg["model"]["name"])
        prompt_ids = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=4096,
        )["input_ids"].to(device)

        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                input_ids=prompt_ids,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        gen_ids = out[0, prompt_ids.shape[1]:].tolist()
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        wall = time.time() - t0

        answer = extract_boxed_answer(gen_text) or extract_numeric_answer(gen_text) or ""
        correct = answers_match(answer, problem["gold_answer"])

        results.append({
            "problem_id": problem["problem_id"],
            "source": problem.get("source", ""),
            "question": problem["question"],
            "gold_answer": problem["gold_answer"],
            "extracted_answer": answer,
            "correct": correct,
            "generated_text": gen_text,
            "total_tokens": len(gen_ids),
            "wall_time_sec": wall,
            "method": "greedy",
            "level": problem.get("level", ""),
            "problem_type": problem.get("problem_type", ""),
        })

        if (i + 1) % 20 == 0 or i == 0:
            acc = sum(1 for r in results if r["correct"]) / len(results)
            logger.info(f"  Greedy [{i+1}/{len(problems)}] acc={acc:.4f} ans={answer} gold={problem['gold_answer']}")

        del out
        torch.cuda.empty_cache()

    return results


def run_sampling_vote(model, tokenizer, problems, cfg):
    from src.data.dataset import format_prompt, extract_boxed_answer, extract_numeric_answer, answers_match

    device = cfg["model"]["device"]
    dad_cfg = cfg.get("dad", {})
    n_samples = dad_cfg.get("m_samples", 8)
    max_tokens = dad_cfg.get("max_gen_tokens", 2048)
    temperature = dad_cfg.get("temperature", 0.7)
    top_p = dad_cfg.get("top_p", 0.95)
    results = []

    for i, problem in enumerate(problems):
        prompt = format_prompt(problem, cfg["model"]["name"])
        prompt_ids = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=4096,
        )["input_ids"].to(device)

        t0 = time.time()
        solutions = []
        with torch.no_grad():
            for _ in range(n_samples):
                out = model.generate(
                    input_ids=prompt_ids,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                gen_ids = out[0, prompt_ids.shape[1]:].tolist()
                gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                ans = extract_boxed_answer(gen_text) or extract_numeric_answer(gen_text) or ""
                solutions.append({"text": gen_text, "answer": ans, "tokens": len(gen_ids)})
                del out
                torch.cuda.empty_cache()

        wall = time.time() - t0

        # Majority vote with robust normalization
        answer_counts = defaultdict(int)
        answer_text = {}
        for s in solutions:
            norm = _normalize_for_voting(s["answer"])
            answer_counts[norm] += 1
            if norm not in answer_text:
                answer_text[norm] = s

        best_answer = max(answer_counts, key=answer_counts.get) if answer_counts else ""
        best_sol = answer_text.get(best_answer, solutions[-1] if solutions else {"text": "", "answer": ""})

        correct = answers_match(best_sol["answer"], problem["gold_answer"])

        results.append({
            "problem_id": problem["problem_id"],
            "source": problem.get("source", ""),
            "question": problem["question"],
            "gold_answer": problem["gold_answer"],
            "extracted_answer": best_sol["answer"],
            "correct": correct,
            "generated_text": best_sol["text"],
            "total_tokens": sum(s["tokens"] for s in solutions),
            "wall_time_sec": wall,
            "method": "sampling_vote",
            "n_samples": n_samples,
            "answer_distribution": dict(answer_counts),
            "level": problem.get("level", ""),
            "problem_type": problem.get("problem_type", ""),
        })

        if (i + 1) % 20 == 0 or i == 0:
            acc = sum(1 for r in results if r["correct"]) / len(results)
            logger.info(f"  SamplingVote [{i+1}/{len(problems)}] acc={acc:.4f} ans={best_sol['answer']} gold={problem['gold_answer']}")

    return results


def run_dad(model, tokenizer, problems, cfg):
    from src.data.dataset import format_prompt, answers_match
    from src.dad.dad_generator import DADGenerator

    generator = DADGenerator(model, tokenizer, cfg)
    results = []

    for i, problem in enumerate(problems):
        prompt = format_prompt(problem, cfg["model"]["name"])
        prompt_ids = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=4096,
        )["input_ids"].to(cfg["model"]["device"])

        t0 = time.time()
        try:
            gen = generator.generate(prompt_ids, problem_text=problem["question"])
        except Exception as e:
            logger.warning(f"Problem {problem['problem_id']} failed: {e}")
            import traceback; traceback.print_exc()
            torch.cuda.empty_cache()
            continue

        correct = answers_match(gen.extracted_answer, problem["gold_answer"])

        results.append({
            "problem_id": problem["problem_id"],
            "source": problem.get("source", ""),
            "question": problem["question"],
            "gold_answer": problem["gold_answer"],
            "extracted_answer": gen.extracted_answer,
            "correct": correct,
            "generated_text": gen.generated_text,
            "total_tokens": gen.total_tokens,
            "wall_time_sec": gen.wall_time_sec,
            "method": "dad",
            "n_rounds": gen.n_rounds,
            "n_total_generations": gen.n_total_generations,
            "answer_entropy_per_round": gen.answer_entropy_per_round,
            "confidence_per_round": gen.confidence_per_round,
            "disagreement_map": gen.disagreement_map,
            "level": problem.get("level", ""),
            "problem_type": problem.get("problem_type", ""),
        })

        if (i + 1) % 20 == 0 or i == 0:
            acc = sum(1 for r in results if r["correct"]) / len(results)
            logger.info(
                f"  DAD [{i+1}/{len(problems)}] acc={acc:.4f} "
                f"ans={gen.extracted_answer} gold={problem['gold_answer']} "
                f"rounds={gen.n_rounds} gens={gen.n_total_generations}"
            )

        torch.cuda.empty_cache()

    return results


def save_results(results, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
            f.flush()
    logger.info(f"Saved {len(results)} results to {path}")


def compute_metrics(results, method_name):
    if not results:
        return {}

    n = len(results)
    n_correct = sum(1 for r in results if r["correct"])
    accuracy = n_correct / n

    by_level = defaultdict(list)
    by_type = defaultdict(list)
    for r in results:
        if r.get("level"):
            by_level[r["level"]].append(r["correct"])
        if r.get("problem_type"):
            by_type[r["problem_type"]].append(r["correct"])

    metrics = {
        "method": method_name,
        "n_problems": n,
        "n_correct": n_correct,
        "accuracy": accuracy,
        "mean_tokens": float(np.mean([r["total_tokens"] for r in results])),
        "median_tokens": float(np.median([r["total_tokens"] for r in results])),
        "mean_wall_time": float(np.mean([r["wall_time_sec"] for r in results])),
        "total_wall_time": float(np.sum([r["wall_time_sec"] for r in results])),
        "accuracy_by_level": {k: float(np.mean(v)) for k, v in sorted(by_level.items())},
        "accuracy_by_type": {k: float(np.mean(v)) for k, v in sorted(by_type.items())},
    }
    return metrics


def print_comparison(all_metrics):
    sep = "=" * 70
    logger.info(sep)
    logger.info(f"{'Method':<25} {'Acc':>8} {'N':>6} {'Tokens':>10} {'Time/Q':>10}")
    logger.info("-" * 70)
    for m in all_metrics:
        logger.info(
            f"{m['method']:<25} {m['accuracy']:>8.4f} {m['n_problems']:>6} "
            f"{m['mean_tokens']:>10.0f} {m['mean_wall_time']:>10.1f}s"
        )
    logger.info(sep)

    if len(all_metrics) >= 2:
        baseline = all_metrics[0]
        for m in all_metrics[1:]:
            delta = m["accuracy"] - baseline["accuracy"]
            logger.info(
                f"  {m['method']} vs {baseline['method']}: "
                f"{'+' if delta >= 0 else ''}{delta:.4f} ({delta*100:+.1f}%)"
            )
    logger.info(sep)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dad_config.yaml")
    parser.add_argument("--mode", default="all", choices=["greedy", "sampling", "dad", "all"])
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--n_problems", type=int, default=None)
    parser.add_argument("--m_samples", type=int, default=None)
    parser.add_argument("--max_rounds", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.dataset:
        cfg["dataset"]["name"] = args.dataset
    if args.n_problems is not None:
        cfg["dataset"]["n_problems"] = args.n_problems
    if args.m_samples is not None:
        cfg["dad"]["m_samples"] = args.m_samples
    if args.max_rounds is not None:
        cfg["dad"]["max_rounds"] = args.max_rounds
    if args.temperature is not None:
        cfg["dad"]["temperature"] = args.temperature

    short = cfg["model"]["short_name"]
    setup_logging(f"logs/dad_{short}.log")

    logger.info("=" * 60)
    logger.info("Disagreement-Aware Distillation (DAD)")
    logger.info("=" * 60)
    logger.info(f"  model      : {cfg['model']['name']}")
    logger.info(f"  dataset    : {cfg['dataset']['name']}")
    logger.info(f"  n_problems : {cfg['dataset'].get('n_problems', -1)}")
    logger.info(f"  M samples  : {cfg['dad']['m_samples']}")
    logger.info(f"  max_rounds : {cfg['dad']['max_rounds']}")
    logger.info(f"  temperature: {cfg['dad']['temperature']}")
    logger.info(f"  mode       : {args.mode}")

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        logger.info(f"  GPU        : {props.name} {props.total_memory / 1e9:.1f}GB")

    from src.data.model_loader import ModelLoader
    from src.data.dataset import get_inference_dataset

    model, tokenizer = ModelLoader(cfg).load()

    # ============================================================
    # FIX 1: Override max_position_embeddings for all baselines
    # Qwen2.5-Math-7B supports 32K context but config says 4096
    # Without this, greedy/sampling solutions get truncated
    # ============================================================
    if hasattr(model, 'config'):
        old_max_pos = getattr(model.config, 'max_position_embeddings', None)
        if old_max_pos and old_max_pos < 32768:
            model.config.max_position_embeddings = 32768
            logger.info(f"  Fixed max_position_embeddings: {old_max_pos} -> 32768")

    problems = get_inference_dataset(cfg)
    logger.info(f"Loaded {len(problems)} problems")

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []

    if args.mode in ("greedy", "all"):
        logger.info("Running GREEDY baseline...")
        greedy_results = run_greedy(model, tokenizer, problems, cfg)
        save_results(greedy_results, out_dir / f"predictions_{short}_greedy.jsonl")
        gm = compute_metrics(greedy_results, "greedy")
        all_metrics.append(gm)
        logger.info(f"  Greedy accuracy: {gm['accuracy']:.4f}")

    if args.mode in ("sampling", "all"):
        logger.info(f"Running SAMPLING+VOTE (M={cfg['dad']['m_samples']})...")
        sv_results = run_sampling_vote(model, tokenizer, problems, cfg)
        save_results(sv_results, out_dir / f"predictions_{short}_sampling_m{cfg['dad']['m_samples']}.jsonl")
        sm = compute_metrics(sv_results, f"sampling_vote_m{cfg['dad']['m_samples']}")
        all_metrics.append(sm)
        logger.info(f"  Sampling+Vote accuracy: {sm['accuracy']:.4f}")

    if args.mode in ("dad", "all"):
        logger.info(f"Running DAD (M={cfg['dad']['m_samples']}, R={cfg['dad']['max_rounds']})...")
        dad_results = run_dad(model, tokenizer, problems, cfg)
        save_results(dad_results, out_dir / f"predictions_{short}_dad_m{cfg['dad']['m_samples']}_r{cfg['dad']['max_rounds']}.jsonl")
        dm = compute_metrics(dad_results, f"dad_m{cfg['dad']['m_samples']}_r{cfg['dad']['max_rounds']}")
        all_metrics.append(dm)
        logger.info(f"  DAD accuracy: {dm['accuracy']:.4f}")

    if all_metrics:
        print_comparison(all_metrics)

        metrics_path = out_dir / f"metrics_{short}_comparison.json"
        with open(metrics_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
        logger.info(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()