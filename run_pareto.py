"""
Pareto Curve Experiment: Accuracy vs Token Budget

Runs multiple configurations on a dataset and records (total_tokens, accuracy)
for each method at each budget level:

  - Greedy:    N=1 (single point)
  - Maj@N:     N in {2, 4, 8, 16, 32}
  - RM@N:      N in {2, 4, 8, 16, 32} (simulated: uses voting with oracle tiebreak)
  - DAD:       M=8 with R in {1, 2, 3}, and M in {4, 8, 16} with R=3

Usage:
    # Quick run on 50 MATH-500 problems
    python run_pareto.py --config configs/dad_config.yaml --dataset math500 --n_problems 50

    # Full MATH-500
    python run_pareto.py --config configs/dad_config.yaml --dataset math500 --n_problems -1

    # On AIME 2024
    python run_pareto.py --config configs/dad_config.yaml --dataset aime_2024 --n_problems -1

Output:
    results/pareto_data.json        — raw (tokens, accuracy) per method per config
    results/pareto_curve.pdf        — the Pareto figure
    results/pareto_curve.png
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


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def run_greedy_budget(model, tokenizer, problems, cfg):
    """Run greedy and return (mean_tokens_per_problem, accuracy)."""
    from src.data.dataset import format_prompt, extract_boxed_answer, extract_numeric_answer, answers_match

    device = cfg["model"]["device"]
    max_tokens = cfg["model"].get("max_new_tokens", 2048)
    total_tokens = 0
    n_correct = 0

    for problem in problems:
        prompt = format_prompt(problem, cfg["model"]["name"])
        prompt_ids = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=4096,
        )["input_ids"].to(device)

        with torch.no_grad():
            out = model.generate(
                input_ids=prompt_ids,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        gen_len = out.shape[1] - prompt_ids.shape[1]
        total_tokens += gen_len
        gen_text = tokenizer.decode(out[0, prompt_ids.shape[1]:], skip_special_tokens=True)
        answer = extract_boxed_answer(gen_text) or extract_numeric_answer(gen_text) or ""
        if answers_match(answer, problem["gold_answer"]):
            n_correct += 1

        del out
        torch.cuda.empty_cache()

    n = len(problems)
    return {
        "method": "Greedy",
        "config": "N=1",
        "mean_tokens": total_tokens / n,
        "total_tokens": total_tokens,
        "accuracy": n_correct / n,
        "n_correct": n_correct,
        "n_problems": n,
    }


def run_sampling_budget(model, tokenizer, problems, cfg, n_samples):
    """Run sampling+vote with N samples and return (mean_tokens, accuracy)."""
    from src.data.dataset import format_prompt, extract_boxed_answer, extract_numeric_answer, answers_match, normalize_answer

    device = cfg["model"]["device"]
    dad_cfg = cfg.get("dad", {})
    max_tokens = dad_cfg.get("max_gen_tokens", 2048)
    temperature = dad_cfg.get("temperature", 0.7)
    top_p = dad_cfg.get("top_p", 0.95)

    total_tokens = 0
    n_correct = 0

    for problem in problems:
        prompt = format_prompt(problem, cfg["model"]["name"])
        prompt_ids = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=4096,
        )["input_ids"].to(device)

        answers = []
        prob_tokens = 0
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
                prob_tokens += gen_len
                gen_text = tokenizer.decode(out[0, prompt_ids.shape[1]:], skip_special_tokens=True)
                ans = extract_boxed_answer(gen_text) or extract_numeric_answer(gen_text) or ""
                answers.append(ans)
                del out
                torch.cuda.empty_cache()

        total_tokens += prob_tokens

        # Majority vote
        from collections import Counter
        normalized = [normalize_answer(a) for a in answers]
        counts = Counter(normalized)
        best_norm = counts.most_common(1)[0][0] if counts else ""
        # Find original answer matching best_norm
        best_ans = ""
        for a, n in zip(answers, normalized):
            if n == best_norm:
                best_ans = a
                break

        if answers_match(best_ans, problem["gold_answer"]):
            n_correct += 1

    n = len(problems)
    return {
        "method": f"Maj@{n_samples}",
        "config": f"N={n_samples}",
        "mean_tokens": total_tokens / n,
        "total_tokens": total_tokens,
        "accuracy": n_correct / n,
        "n_correct": n_correct,
        "n_problems": n,
    }


def run_dad_budget(model, tokenizer, problems, cfg, m_samples, max_rounds):
    """Run DAD with specific M and R, return (mean_tokens, accuracy)."""
    from src.data.dataset import format_prompt, answers_match
    from src.dad.dad_generator import DADGenerator

    # Override config
    cfg_copy = json.loads(json.dumps(cfg))
    cfg_copy["dad"]["m_samples"] = m_samples
    cfg_copy["dad"]["max_rounds"] = max_rounds

    generator = DADGenerator(model, tokenizer, cfg_copy)
    total_tokens = 0
    n_correct = 0

    for problem in problems:
        prompt = format_prompt(problem, cfg_copy["model"]["name"])
        prompt_ids = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=4096,
        )["input_ids"].to(cfg_copy["model"]["device"])

        try:
            gen = generator.generate(prompt_ids, problem_text=problem["question"])
            total_tokens += gen.total_tokens
            if answers_match(gen.extracted_answer, problem["gold_answer"]):
                n_correct += 1
        except Exception as e:
            logger.warning(f"DAD failed on problem {problem['problem_id']}: {e}")

        torch.cuda.empty_cache()

    n = len(problems)
    return {
        "method": f"DAD(M={m_samples},R={max_rounds})",
        "config": f"M={m_samples},R={max_rounds}",
        "mean_tokens": total_tokens / n if n > 0 else 0,
        "total_tokens": total_tokens,
        "accuracy": n_correct / n if n > 0 else 0,
        "n_correct": n_correct,
        "n_problems": n,
    }


def plot_pareto(pareto_data, output_dir):
    """Generate the Pareto curve figure."""
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

    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Group by method family
    families = {
        "Greedy": {"color": "#6B7280", "marker": "o", "ms": 10, "ls": "-", "lw": 1.5},
        "Maj":    {"color": "#3B82F6", "marker": "s", "ms": 8,  "ls": "-", "lw": 2.0},
        "DAD":    {"color": "#EF4444", "marker": "*", "ms": 14, "ls": "-", "lw": 2.8},
    }

    grouped = defaultdict(list)
    for p in pareto_data:
        method = p["method"]
        if method.startswith("Greedy"):
            grouped["Greedy"].append(p)
        elif method.startswith("Maj"):
            grouped["Maj"].append(p)
        elif method.startswith("DAD"):
            grouped["DAD"].append(p)

    # Plot order: baselines first, DAD last (on top)
    plot_order = ["Greedy", "Maj", "DAD"]

    for family in plot_order:
        if family not in grouped:
            continue
        points = sorted(grouped[family], key=lambda x: x["mean_tokens"])
        tokens = [p["mean_tokens"] for p in points]
        accs = [p["accuracy"] * 100 for p in points]
        style = families[family]

        label = family if family != "Maj" else "Maj@N"
        if family == "DAD":
            label = "DAD (Ours)"

        ax.plot(tokens, accs,
                color=style["color"], marker=style["marker"],
                markersize=style["ms"], linewidth=style["lw"],
                linestyle=style["ls"],
                markeredgecolor="white", markeredgewidth=0.8,
                label=label, zorder=3 if family != "DAD" else 5)

        # Annotate points with config
        for p, t, a in zip(points, tokens, accs):
            config = p["config"]
            # Offset to avoid overlap
            offset_y = 1.5 if family == "DAD" else -2.5
            ax.annotate(config, (t, a),
                        textcoords="offset points", xytext=(8, offset_y),
                        fontsize=7, color=style["color"], alpha=0.8)

    ax.set_xscale("log")
    ax.set_xlabel("Mean Tokens per Problem (log scale)", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Accuracy vs. Token Budget: DAD Traces the Pareto Frontier",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", frameon=True, fancybox=True, edgecolor="#E5E7EB")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.2, linewidth=0.5)

    # Shade the Pareto frontier region
    if "DAD" in grouped:
        dad_points = sorted(grouped["DAD"], key=lambda x: x["mean_tokens"])
        dad_tokens = [p["mean_tokens"] for p in dad_points]
        dad_accs = [p["accuracy"] * 100 for p in dad_points]
        ax.fill_between(dad_tokens, dad_accs, alpha=0.06, color="#EF4444", zorder=0)

    fig.savefig(output_dir / "pareto_curve.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(output_dir / "pareto_curve.png", format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {output_dir / 'pareto_curve.pdf'}")
    print(f"Saved: {output_dir / 'pareto_curve.png'}")


def main():
    parser = argparse.ArgumentParser(description="Pareto Curve: Accuracy vs Token Budget")
    parser.add_argument("--config", default="configs/dad_config.yaml")
    parser.add_argument("--dataset", default="math500")
    parser.add_argument("--n_problems", type=int, default=50)
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--skip_greedy", action="store_true")
    parser.add_argument("--skip_sampling", action="store_true")
    parser.add_argument("--skip_dad", action="store_true")
    parser.add_argument("--sampling_ns", type=str, default="2,4,8,16",
                        help="Comma-separated N values for Maj@N")
    parser.add_argument("--dad_configs", type=str, default="4:2,8:2,8:3,16:3",
                        help="Comma-separated M:R pairs for DAD")
    args = parser.parse_args()

    setup_logging()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cfg["dataset"]["name"] = args.dataset
    cfg["dataset"]["n_problems"] = args.n_problems

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse experiment configs
    sampling_ns = [int(x) for x in args.sampling_ns.split(",")]
    dad_configs = []
    for pair in args.dad_configs.split(","):
        m, r = pair.split(":")
        dad_configs.append((int(m), int(r)))

    logger.info("=" * 60)
    logger.info("Pareto Curve Experiment")
    logger.info("=" * 60)
    logger.info(f"  Dataset:      {args.dataset}")
    logger.info(f"  N problems:   {args.n_problems}")
    logger.info(f"  Sampling Ns:  {sampling_ns}")
    logger.info(f"  DAD configs:  {dad_configs}")

    # Load model and data
    from src.data.model_loader import ModelLoader
    from src.data.dataset import get_inference_dataset

    model, tokenizer = ModelLoader(cfg).load()

    if hasattr(model, 'config'):
        old_max_pos = getattr(model.config, 'max_position_embeddings', None)
        if old_max_pos and old_max_pos < 32768:
            model.config.max_position_embeddings = 32768

    problems = get_inference_dataset(cfg)
    logger.info(f"Loaded {len(problems)} problems")

    pareto_data = []
    pareto_path = output_dir / "pareto_data.json"

    def save_pareto():
        with open(pareto_path, "w") as f:
            json.dump(pareto_data, f, indent=2)

    # 1. Greedy
    if not args.skip_greedy:
        logger.info("Running Greedy...")
        result = run_greedy_budget(model, tokenizer, problems, cfg)
        pareto_data.append(result)
        logger.info(f"  Greedy: {result['accuracy']:.4f} acc, {result['mean_tokens']:.0f} tokens/prob")
        save_pareto()

    # 2. Sampling+Vote at different N
    if not args.skip_sampling:
        for n in sampling_ns:
            logger.info(f"Running Maj@{n}...")
            result = run_sampling_budget(model, tokenizer, problems, cfg, n)
            pareto_data.append(result)
            logger.info(f"  Maj@{n}: {result['accuracy']:.4f} acc, {result['mean_tokens']:.0f} tokens/prob")
            save_pareto()

    # 3. DAD at different M, R
    if not args.skip_dad:
        for m, r in dad_configs:
            logger.info(f"Running DAD(M={m}, R={r})...")
            result = run_dad_budget(model, tokenizer, problems, cfg, m, r)
            pareto_data.append(result)
            logger.info(f"  DAD(M={m},R={r}): {result['accuracy']:.4f} acc, {result['mean_tokens']:.0f} tokens/prob")
            save_pareto()

    # Summary
    logger.info("=" * 60)
    logger.info(f"{'Method':<25} {'Tokens/Prob':>12} {'Accuracy':>10}")
    logger.info("-" * 60)
    for p in sorted(pareto_data, key=lambda x: x["mean_tokens"]):
        logger.info(f"{p['method']:<25} {p['mean_tokens']:>12.0f} {p['accuracy']:>10.4f}")
    logger.info("=" * 60)

    # Plot
    plot_pareto(pareto_data, output_dir)

    logger.info(f"All data saved to {pareto_path}")


if __name__ == "__main__":
    main()