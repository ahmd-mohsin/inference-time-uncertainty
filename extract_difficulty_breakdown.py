"""
Extract per-difficulty-level accuracy from MATH-500 results.

This script reads the JSONL output from run_dad.py (which saves level info
per problem) and computes accuracy by difficulty level for each method.

Usage:
    # Option 1: Use existing results
    python extract_difficulty_breakdown.py --input results/math500_results.jsonl

    # Option 2: Run fresh on MATH-500 (50 problems) then extract
    python extract_difficulty_breakdown.py --run --n_problems 50

    # Option 3: Full MATH-500 (500 problems)
    python extract_difficulty_breakdown.py --run --n_problems -1

Output:
    difficulty_breakdown.json   — per-level accuracy for each method
    difficulty_heatmap.pdf      — heatmap figure
    difficulty_3d.pdf           — 3D surface figure
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from collections import defaultdict

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


def load_results(path):
    """Load JSONL results from run_dad.py output."""
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def compute_breakdown(results):
    """Compute per-level accuracy for each method.
    
    Expects each result dict to have:
        - level: str like "1", "2", ..., "5"
        - method: str like "greedy", "sampling_vote", "dad"
        - correct: bool
        - problem_id: int
    
    If results come from --mode all, there will be multiple entries
    per problem_id (one per method). If results come from the entropy
    analysis script, fields are greedy_correct, majority_correct, dad_correct.
    """
    
    # Detect format: run_dad.py format vs entropy_analysis format
    if "method" in results[0]:
        # run_dad.py format: separate lines per method
        return _breakdown_from_run_dad(results)
    elif "greedy_correct" in results[0]:
        # entropy_analysis format: all methods in one line
        return _breakdown_from_entropy(results)
    else:
        raise ValueError("Unknown result format. Expected 'method' or 'greedy_correct' field.")


def _breakdown_from_run_dad(results):
    """Extract breakdown from run_dad.py JSONL (one line per problem per method)."""
    # Group by level and method
    level_method_correct = defaultdict(lambda: defaultdict(list))
    
    for r in results:
        level = str(r.get("level", "")).strip()
        if not level or level == "":
            continue
        method = r.get("method", "")
        correct = r.get("correct", False)
        level_method_correct[level][method].append(correct)
    
    return _format_breakdown(level_method_correct)


def _breakdown_from_entropy(results):
    """Extract breakdown from entropy_analysis JSONL (all methods in one line)."""
    level_method_correct = defaultdict(lambda: defaultdict(list))
    
    for r in results:
        level = str(r.get("level", "")).strip()
        if not level or level == "":
            continue
        
        level_method_correct[level]["greedy"].append(r.get("greedy_correct", False))
        level_method_correct[level]["sampling_vote"].append(r.get("majority_correct", False))
        if r.get("dad_correct") is not None:
            level_method_correct[level]["dad"].append(r.get("dad_correct", False))
    
    return _format_breakdown(level_method_correct)


def _format_breakdown(level_method_correct):
    """Format the breakdown into a clean dict."""
    breakdown = {}
    
    # Sort levels numerically
    levels = sorted(level_method_correct.keys(), key=lambda x: int(x) if x.isdigit() else 99)
    
    for level in levels:
        methods = level_method_correct[level]
        breakdown[level] = {}
        for method, corrects in methods.items():
            n = len(corrects)
            n_correct = sum(corrects)
            acc = n_correct / n * 100 if n > 0 else 0.0
            breakdown[level][method] = {
                "n": n,
                "correct": n_correct,
                "accuracy": round(acc, 1),
            }
    
    return breakdown


def print_breakdown(breakdown):
    """Pretty-print the breakdown table."""
    # Collect all methods
    all_methods = set()
    for level_data in breakdown.values():
        all_methods.update(level_data.keys())
    
    # Canonical order
    method_order = ["greedy", "sampling_vote", "dad"]
    methods = [m for m in method_order if m in all_methods]
    methods.extend(sorted(all_methods - set(method_order)))
    
    # Header
    header = f"{'Level':<8} {'Count':>6}"
    for m in methods:
        header += f" {m:>15}"
    print("=" * len(header))
    print("ACCURACY BY DIFFICULTY LEVEL")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    
    for level in sorted(breakdown.keys(), key=lambda x: int(x) if x.isdigit() else 99):
        level_data = breakdown[level]
        # Get count from any method
        n = next(iter(level_data.values()))["n"] if level_data else 0
        row = f"  {level:<6} {n:>6}"
        for m in methods:
            if m in level_data:
                acc = level_data[m]["accuracy"]
                row += f" {acc:>14.1f}%"
            else:
                row += f" {'---':>15}"
        print(row)
    
    # Overall
    print("-" * len(header))
    row = f"{'Overall':<8}"
    total_n = sum(next(iter(d.values()))["n"] for d in breakdown.values() if d)
    row += f" {total_n:>6}"
    for m in methods:
        total_correct = sum(
            breakdown[l][m]["correct"] for l in breakdown if m in breakdown[l]
        )
        total_count = sum(
            breakdown[l][m]["n"] for l in breakdown if m in breakdown[l]
        )
        acc = total_correct / total_count * 100 if total_count > 0 else 0
        row += f" {acc:>14.1f}%"
    print(row)
    print("=" * len(header))


def generate_figures(breakdown, output_dir):
    """Generate heatmap and 3D figures from the breakdown data."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

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

    # Prepare data
    levels = sorted(breakdown.keys(), key=lambda x: int(x) if x.isdigit() else 99)
    method_order = ["greedy", "sampling_vote", "dad"]
    method_labels = {"greedy": "Greedy", "sampling_vote": "Maj@8", "dad": "DAD (Ours)"}
    method_colors = {
        "greedy": "#6B7280",
        "sampling_vote": "#3B82F6",
        "dad": "#EF4444",
    }
    methods = [m for m in method_order if any(m in breakdown[l] for l in levels)]

    # Build accuracy matrix: rows=levels, cols=methods
    acc_matrix = []
    for level in levels:
        row = []
        for m in methods:
            if m in breakdown[level]:
                row.append(breakdown[level][m]["accuracy"])
            else:
                row.append(0.0)
        acc_matrix.append(row)
    acc_matrix = np.array(acc_matrix)

    n_levels = len(levels)
    n_methods = len(methods)

    # ── Figure 1: Grouped bar chart by difficulty level ───────────
    fig1, ax1 = plt.subplots(figsize=(9, 5.5))

    x = np.arange(n_levels)
    width = 0.25

    for j, m in enumerate(methods):
        offset = (j - n_methods / 2 + 0.5) * width
        bars = ax1.bar(x + offset, acc_matrix[:, j], width,
                       color=method_colors.get(m, "#666"),
                       alpha=0.8, edgecolor="white", linewidth=0.8,
                       label=method_labels.get(m, m), zorder=3)
        # Value labels on bars
        for bar, val in zip(bars, acc_matrix[:, j]):
            if val > 0:
                ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                         f"{val:.0f}", ha="center", va="bottom", fontsize=8,
                         fontweight="bold" if m == "dad" else "normal",
                         color=method_colors.get(m, "#666"))

    # DAD gain annotations
    if "dad" in methods and "greedy" in methods:
        dad_idx = methods.index("dad")
        greedy_idx = methods.index("greedy")
        for i, level in enumerate(levels):
            gain = acc_matrix[i, dad_idx] - acc_matrix[i, greedy_idx]
            if gain > 1:
                ax1.annotate(f"+{gain:.0f}pp",
                             xy=(x[i] + (dad_idx - n_methods/2 + 0.5) * width,
                                 acc_matrix[i, dad_idx] + 4),
                             fontsize=7, color="#991B1B", fontweight="bold",
                             ha="center", va="bottom")

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Level {l}" for l in levels], fontsize=11)
    ax1.set_ylabel("Accuracy (%)", fontsize=12)
    ax1.set_xlabel("Difficulty Level (MATH-500)", fontsize=12)
    ax1.set_ylim(0, 110)
    ax1.legend(loc="upper right", frameon=True, fancybox=True, edgecolor="#E5E7EB")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(axis="y", alpha=0.2, linewidth=0.5)
    ax1.set_title("Accuracy by Difficulty Level: DAD Gains Increase with Hardness",
                   fontsize=13, fontweight="bold")

    # Add problem counts
    for i, level in enumerate(levels):
        n = breakdown[level][methods[0]]["n"]
        ax1.text(x[i], -6, f"n={n}", ha="center", fontsize=8, color="#6B7280")

    fig1.savefig(output_dir / "difficulty_bars.pdf", format="pdf", bbox_inches="tight")
    fig1.savefig(output_dir / "difficulty_bars.png", format="png", bbox_inches="tight", dpi=300)
    plt.close(fig1)
    print(f"Saved: {output_dir / 'difficulty_bars.pdf'}")

    # ── Figure 2: Heatmap ─────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(7, 5))

    im = ax2.imshow(acc_matrix, cmap="RdYlGn", aspect="auto", vmin=20, vmax=100)

    # Text annotations
    for i in range(n_levels):
        for j in range(n_methods):
            val = acc_matrix[i, j]
            color = "white" if val < 50 else "#1F2937"
            weight = "bold" if methods[j] == "dad" else "normal"
            ax2.text(j, i, f"{val:.0f}%", ha="center", va="center",
                     fontsize=12, fontweight=weight, color=color)

    # Highlight DAD column
    if "dad" in methods:
        dad_j = methods.index("dad")
        rect = plt.Rectangle((dad_j - 0.5, -0.5), 1, n_levels,
                              linewidth=2.5, edgecolor="#EF4444", facecolor="none")
        ax2.add_patch(rect)

    ax2.set_xticks(range(n_methods))
    ax2.set_xticklabels([method_labels.get(m, m) for m in methods], fontsize=11)
    ax2.set_yticks(range(n_levels))
    ax2.set_yticklabels([f"Level {l}" for l in levels], fontsize=11)
    ax2.set_xlabel("Method", fontsize=12)
    ax2.set_ylabel("Difficulty Level", fontsize=12)
    ax2.set_title("Accuracy Heatmap by Difficulty", fontsize=13, fontweight="bold")

    cb = plt.colorbar(im, ax=ax2, shrink=0.8)
    cb.set_label("Accuracy (%)", fontsize=11)

    fig2.savefig(output_dir / "difficulty_heatmap.pdf", format="pdf", bbox_inches="tight")
    fig2.savefig(output_dir / "difficulty_heatmap.png", format="png", bbox_inches="tight", dpi=300)
    plt.close(fig2)
    print(f"Saved: {output_dir / 'difficulty_heatmap.pdf'}")

    # ── Figure 3: DAD gain curve ──────────────────────────────────
    if "dad" in methods and len(methods) >= 2:
        fig3, ax3 = plt.subplots(figsize=(8, 5))

        dad_idx = methods.index("dad")
        for j, m in enumerate(methods):
            if m == "dad":
                continue
            gains = acc_matrix[:, dad_idx] - acc_matrix[:, j]
            ax3.plot(range(n_levels), gains, '-o',
                     color=method_colors.get(m, "#666"),
                     linewidth=2.2, markersize=8,
                     label=f"DAD vs. {method_labels.get(m, m)}")
            # Value labels
            for i, g in enumerate(gains):
                ax3.text(i, g + 0.8, f"+{g:.0f}", ha="center", va="bottom",
                         fontsize=9, color=method_colors.get(m, "#666"))

        ax3.set_xticks(range(n_levels))
        ax3.set_xticklabels([f"Level {l}" for l in levels], fontsize=11)
        ax3.set_ylabel("DAD Accuracy Gain (pp)", fontsize=12)
        ax3.set_xlabel("Difficulty Level", fontsize=12)
        ax3.set_title("DAD Gain Over Baselines by Difficulty",
                       fontsize=13, fontweight="bold")
        ax3.legend(loc="upper left", frameon=True, fancybox=True, edgecolor="#E5E7EB")
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)
        ax3.grid(alpha=0.2, linewidth=0.5)
        ax3.axhline(y=0, color="#9CA3AF", linewidth=0.8, linestyle="-")

        fig3.savefig(output_dir / "difficulty_gain_curve.pdf", format="pdf", bbox_inches="tight")
        fig3.savefig(output_dir / "difficulty_gain_curve.png", format="png", bbox_inches="tight", dpi=300)
        plt.close(fig3)
        print(f"Saved: {output_dir / 'difficulty_gain_curve.pdf'}")


def main():
    parser = argparse.ArgumentParser(description="Extract difficulty breakdown from MATH-500 results")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to existing results JSONL (from run_dad.py or run_entropy_analysis.py)")
    parser.add_argument("--run", action="store_true",
                        help="Run fresh experiment on MATH-500")
    parser.add_argument("--config", default="configs/dad_config.yaml")
    parser.add_argument("--n_problems", type=int, default=50)
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--no_figures", action="store_true")
    args = parser.parse_args()

    setup_logging()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.run:
        # Run fresh experiment
        import subprocess
        cmd = [
            "python", "run_dad.py",
            "--config", args.config,
            "--mode", "all",
            "--dataset", "math500",
            "--n_problems", str(args.n_problems),
        ]
        logger.info(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        # Find the most recent results file
        results_files = sorted(output_dir.glob("*math500*.jsonl"), key=lambda p: p.stat().st_mtime)
        if not results_files:
            logger.error("No results file found after run. Check run_dad.py output path.")
            sys.exit(1)
        args.input = str(results_files[-1])
        logger.info(f"Using results from: {args.input}")

    if args.input is None:
        # Try to find any existing results
        candidates = list(Path("results").glob("*math500*.jsonl")) + \
                     list(Path("results").glob("*entropy*.jsonl"))
        if candidates:
            args.input = str(sorted(candidates, key=lambda p: p.stat().st_mtime)[-1])
            logger.info(f"Auto-detected results: {args.input}")
        else:
            logger.error("No --input provided and no results found. Use --run or provide --input.")
            sys.exit(1)

    results = load_results(args.input)
    logger.info(f"Loaded {len(results)} result entries from {args.input}")

    # Check that levels are present
    has_levels = any(r.get("level", "") not in ("", None) for r in results)
    if not has_levels:
        logger.error("No difficulty level info found in results. "
                     "Make sure you ran on MATH-500 (which has Level 1-5).")
        sys.exit(1)

    breakdown = compute_breakdown(results)
    print_breakdown(breakdown)

    # Save breakdown
    breakdown_path = output_dir / "difficulty_breakdown.json"
    with open(breakdown_path, "w") as f:
        json.dump(breakdown, f, indent=2)
    logger.info(f"Saved breakdown to {breakdown_path}")

    if not args.no_figures:
        generate_figures(breakdown, output_dir)


if __name__ == "__main__":
    main()