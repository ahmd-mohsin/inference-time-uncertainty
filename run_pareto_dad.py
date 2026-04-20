"""
Pareto DAD Sweep: Run DAD at multiple (M, R) configurations on MATH-500.

This script runs the same set of problems through multiple DAD configurations
and records (total_tokens, accuracy) for each. The problems are loaded once
and the model is loaded once; only the DAD parameters change between runs.

Usage:
    # Default: 100 MATH-500 problems, configs M in {4,8}, R in {1,2,3}
    python run_pareto_dad.py --config configs/dad_config.yaml --n_problems 100

    # Custom configs
    python run_pareto_dad.py --config configs/dad_config.yaml --n_problems 50 \
        --configs "4:1,4:2,4:3,8:1,8:2,8:3,16:2,16:3"

    # Also run greedy and Maj@N baselines for comparison
    python run_pareto_dad.py --config configs/dad_config.yaml --n_problems 100 --with_baselines

    # Chain after existing job
    while kill -0 <PID> 2>/dev/null; do sleep 5; done && \
    python run_pareto_dad.py --config configs/dad_config.yaml --n_problems 100 --with_baselines

Output:
    results/pareto_dad_sweep.json   — all (tokens, accuracy) data points
    results/pareto_dad_sweep.pdf    — Pareto curve figure
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


def run_greedy_on_problems(model, tokenizer, problems, cfg):
    """Run greedy on all problems, return per-problem results."""
    from src.data.dataset import format_prompt, extract_boxed_answer, extract_numeric_answer, answers_match

    device = cfg["model"]["device"]
    max_tokens = cfg["model"].get("max_new_tokens", 2048)
    results = []

    pbar = tqdm(problems, desc="Greedy", unit="prob")
    for problem in pbar:
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
        gen_len = out.shape[1] - prompt_ids.shape[1]
        gen_text = tokenizer.decode(out[0, prompt_ids.shape[1]:], skip_special_tokens=True)
        wall = time.time() - t0

        answer = extract_boxed_answer(gen_text) or extract_numeric_answer(gen_text) or ""
        correct = answers_match(answer, problem["gold_answer"])

        results.append({
            "problem_id": problem["problem_id"],
            "correct": correct,
            "tokens": gen_len,
            "wall_time": wall,
        })

        acc = sum(r["correct"] for r in results) / len(results)
        pbar.set_postfix(acc=f"{acc:.3f}")

        del out
        torch.cuda.empty_cache()

    pbar.close()
    return results


def run_majN_on_problems(model, tokenizer, problems, cfg, n_samples):
    """Run Maj@N on all problems, return per-problem results."""
    from src.data.dataset import (
        format_prompt, extract_boxed_answer, extract_numeric_answer,
        answers_match, normalize_answer,
    )
    from collections import Counter

    device = cfg["model"]["device"]
    dad_cfg = cfg.get("dad", {})
    max_tokens = dad_cfg.get("max_gen_tokens", 2048)
    temperature = dad_cfg.get("temperature", 0.7)
    top_p = dad_cfg.get("top_p", 0.95)
    results = []

    pbar = tqdm(problems, desc=f"Maj@{n_samples}", unit="prob")
    for problem in pbar:
        prompt = format_prompt(problem, cfg["model"]["name"])
        prompt_ids = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=4096,
        )["input_ids"].to(device)

        t0 = time.time()
        answers = []
        total_tok = 0
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
                gen_len = out.shape[1] - prompt_ids.shape[1]
                total_tok += gen_len
                gen_text = tokenizer.decode(out[0, prompt_ids.shape[1]:], skip_special_tokens=True)
                ans = extract_boxed_answer(gen_text) or extract_numeric_answer(gen_text) or ""
                answers.append(ans)
                del out
                torch.cuda.empty_cache()

        wall = time.time() - t0

        normalized = [normalize_answer(a) for a in answers]
        counts = Counter(normalized)
        best_norm = counts.most_common(1)[0][0] if counts else ""
        best_ans = next((a for a, n in zip(answers, normalized) if n == best_norm), "")
        correct = answers_match(best_ans, problem["gold_answer"])

        results.append({
            "problem_id": problem["problem_id"],
            "correct": correct,
            "tokens": total_tok,
            "wall_time": wall,
        })

        acc = sum(r["correct"] for r in results) / len(results)
        pbar.set_postfix(acc=f"{acc:.3f}")

    pbar.close()
    return results


def run_dad_on_problems(model, tokenizer, problems, cfg, m_samples, max_rounds):
    """Run DAD with specific M and R on all problems, return per-problem results."""
    from src.data.dataset import format_prompt, answers_match
    from src.dad.dad_generator import DADGenerator

    cfg_copy = json.loads(json.dumps(cfg))
    cfg_copy["dad"]["m_samples"] = m_samples
    cfg_copy["dad"]["max_rounds"] = max_rounds

    generator = DADGenerator(model, tokenizer, cfg_copy)
    results = []

    pbar = tqdm(problems, desc=f"DAD(M={m_samples},R={max_rounds})", unit="prob")
    for problem in pbar:
        prompt = format_prompt(problem, cfg_copy["model"]["name"])
        prompt_ids = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=4096,
        )["input_ids"].to(cfg_copy["model"]["device"])

        t0 = time.time()
        try:
            gen = generator.generate(prompt_ids, problem_text=problem["question"])
            correct = answers_match(gen.extracted_answer, problem["gold_answer"])
            tokens = gen.total_tokens
            rounds_used = gen.n_rounds
        except Exception as e:
            logger.warning(f"Problem {problem['problem_id']} failed: {e}")
            correct = False
            tokens = 0
            rounds_used = 0

        wall = time.time() - t0

        results.append({
            "problem_id": problem["problem_id"],
            "correct": correct,
            "tokens": tokens,
            "wall_time": wall,
            "rounds": rounds_used,
        })

        acc = sum(r["correct"] for r in results) / len(results)
        pbar.set_postfix(acc=f"{acc:.3f}", rounds=rounds_used)

        torch.cuda.empty_cache()

    pbar.close()
    return results


def summarize(results, method_name):
    n = len(results)
    n_correct = sum(r["correct"] for r in results)
    total_tokens = sum(r["tokens"] for r in results)
    total_wall = sum(r["wall_time"] for r in results)
    return {
        "method": method_name,
        "n_problems": n,
        "n_correct": n_correct,
        "accuracy": n_correct / n if n > 0 else 0,
        "mean_tokens": total_tokens / n if n > 0 else 0,
        "total_tokens": total_tokens,
        "mean_wall_time": total_wall / n if n > 0 else 0,
    }


def plot_pareto(sweep_data, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 11,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 10,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

    fig, ax = plt.subplots(figsize=(9, 6))

    styles = {
        "Greedy":  {"color": "#6B7280", "marker": "o", "ms": 12, "lw": 0},
        "Maj":     {"color": "#3B82F6", "marker": "s", "ms": 9,  "lw": 2.2},
        "DAD":     {"color": "#EF4444", "marker": "*", "ms": 15, "lw": 2.8},
    }

    grouped = defaultdict(list)
    for p in sweep_data:
        m = p["method"]
        if m.startswith("Greedy"):
            grouped["Greedy"].append(p)
        elif m.startswith("Maj"):
            grouped["Maj"].append(p)
        elif m.startswith("DAD"):
            grouped["DAD"].append(p)

    for family in ["Greedy", "Maj", "DAD"]:
        if family not in grouped:
            continue
        points = sorted(grouped[family], key=lambda x: x["mean_tokens"])
        tok = [p["mean_tokens"] / 1000 for p in points]  # convert to k
        acc = [p["accuracy"] * 100 for p in points]
        s = styles[family]

        label = {"Greedy": "Greedy", "Maj": "Maj@N", "DAD": "DAD (Ours)"}[family]

        if family == "Greedy":
            ax.scatter(tok, acc, s=s["ms"]**2, c=s["color"], marker=s["marker"],
                       edgecolors="white", linewidths=1.0, zorder=5, label=label)
        else:
            ax.plot(tok, acc, color=s["color"], linewidth=s["lw"],
                    marker=s["marker"], markersize=s["ms"],
                    markeredgecolor="white", markeredgewidth=0.8,
                    zorder=4 if family == "Maj" else 6, label=label)

        # Annotate each point
        for p, t, a in zip(points, tok, acc):
            config = p["method"].replace("Maj@", "N=").replace("DAD(", "").replace(")", "")
            offset_y = 1.2 if family == "DAD" else -1.5
            ax.annotate(config, (t, a),
                        textcoords="offset points", xytext=(6, offset_y),
                        fontsize=7.5, color=s["color"], alpha=0.85)

    # DAD Pareto shading
    if "DAD" in grouped:
        dad_pts = sorted(grouped["DAD"], key=lambda x: x["mean_tokens"])
        dad_tok = [p["mean_tokens"] / 1000 for p in dad_pts]
        dad_acc = [p["accuracy"] * 100 for p in dad_pts]
        ax.fill_between(dad_tok, dad_acc, min(dad_acc) - 2, alpha=0.05, color="#EF4444")

    ax.set_xscale("log")
    ax.set_xlabel("Mean Tokens per Problem (k, log scale)", fontsize=13)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_title("Accuracy vs. Token Budget: DAD Pareto Frontier (MATH-500)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", frameon=True, fancybox=True, edgecolor="#E5E7EB")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.2, linewidth=0.5)

    fig.savefig(output_dir / "pareto_dad_sweep.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(output_dir / "pareto_dad_sweep.png", format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {output_dir / 'pareto_dad_sweep.pdf'}")


def main():
    parser = argparse.ArgumentParser(description="Pareto DAD Sweep on MATH-500")
    parser.add_argument("--config", default="configs/dad_config.yaml")
    parser.add_argument("--n_problems", type=int, default=100)
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--configs", type=str, default="4:1,4:2,4:3,8:1,8:2,8:3",
                        help="Comma-separated M:R pairs for DAD sweep")
    parser.add_argument("--with_baselines", action="store_true",
                        help="Also run Greedy and Maj@{2,4,8,16}")
    parser.add_argument("--baseline_ns", type=str, default="2,4,8,16",
                        help="N values for Maj@N baselines")
    args = parser.parse_args()

    setup_logging()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cfg["dataset"]["name"] = "math500"
    cfg["dataset"]["n_problems"] = args.n_problems

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse DAD configs
    dad_configs = []
    for pair in args.configs.split(","):
        m, r = pair.strip().split(":")
        dad_configs.append((int(m), int(r)))

    # Parse baseline Ns
    baseline_ns = [int(x) for x in args.baseline_ns.split(",")]

    logger.info("=" * 60)
    logger.info("Pareto DAD Sweep")
    logger.info("=" * 60)
    logger.info(f"  Dataset:       math500 (first {args.n_problems})")
    logger.info(f"  DAD configs:   {dad_configs}")
    logger.info(f"  With baselines: {args.with_baselines}")
    if args.with_baselines:
        logger.info(f"  Baseline Ns:   {baseline_ns}")

    # Load model and data (once)
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

    sweep_data = []
    sweep_path = output_dir / "pareto_dad_sweep.json"

    def save_sweep():
        with open(sweep_path, "w") as f:
            json.dump(sweep_data, f, indent=2)

    # ── Baselines ─────────────────────────────────────────────
    if args.with_baselines:
        # Greedy
        logger.info("\n--- Greedy ---")
        greedy_results = run_greedy_on_problems(model, tokenizer, problems, cfg)
        s = summarize(greedy_results, "Greedy")
        sweep_data.append(s)
        logger.info(f"  Greedy: {s['accuracy']:.4f} acc, {s['mean_tokens']:.0f} tok/prob")
        save_sweep()

        # Maj@N
        for n in baseline_ns:
            logger.info(f"\n--- Maj@{n} ---")
            maj_results = run_majN_on_problems(model, tokenizer, problems, cfg, n)
            s = summarize(maj_results, f"Maj@{n}")
            sweep_data.append(s)
            logger.info(f"  Maj@{n}: {s['accuracy']:.4f} acc, {s['mean_tokens']:.0f} tok/prob")
            save_sweep()

    # ── DAD sweep ─────────────────────────────────────────────
    for m, r in dad_configs:
        logger.info(f"\n--- DAD(M={m}, R={r}) ---")
        dad_results = run_dad_on_problems(model, tokenizer, problems, cfg, m, r)
        s = summarize(dad_results, f"DAD(M={m},R={r})")
        sweep_data.append(s)
        logger.info(f"  DAD(M={m},R={r}): {s['accuracy']:.4f} acc, {s['mean_tokens']:.0f} tok/prob")

        # Save per-problem details
        details_path = output_dir / f"pareto_dad_M{m}_R{r}_details.json"
        with open(details_path, "w") as f:
            json.dump(dad_results, f, indent=2)

        save_sweep()

    # ── Summary ───────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info(f"{'Method':<25} {'Tokens/Prob':>12} {'Accuracy':>10} {'Wall/Prob':>10}")
    logger.info("-" * 60)
    for s in sorted(sweep_data, key=lambda x: x["mean_tokens"]):
        logger.info(
            f"{s['method']:<25} {s['mean_tokens']:>12.0f} "
            f"{s['accuracy']:>10.4f} {s['mean_wall_time']:>10.1f}s"
        )
    logger.info("=" * 60)

    # ── Plot ──────────────────────────────────────────────────
    plot_pareto(sweep_data, output_dir)
    logger.info(f"\nAll data saved to {sweep_path}")


if __name__ == "__main__":
    main()