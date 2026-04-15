"""
Plot Entropy Inversion Scatter from run_entropy_analysis.py output.

Usage:
    python plot_entropy_inversion.py --input results/entropy_analysis.jsonl

Generates:
    results/entropy_inversion.pdf
    results/entropy_inversion.png
"""

import argparse
import json
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


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


def load_results(path):
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/entropy_analysis.jsonl")
    parser.add_argument("--output_dir", default=None, help="Override output dir (default: same as input)")
    args = parser.parse_args()

    results = load_results(args.input)
    print(f"Loaded {len(results)} problems")

    from pathlib import Path
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path(args.input).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Extract data ──────────────────────────────────────────────
    h_ans_list = []
    maj_gain_list = []     # 1 if maj correct and greedy wrong, -1 if opposite, 0 otherwise
    dad_gain_list = []
    token_ent_list = []
    greedy_correct_list = []

    has_dad = any(r.get("dad_correct") is not None for r in results)

    for r in results:
        h_ans = r["answer_entropy"]
        greedy_ok = r["greedy_correct"]
        maj_ok = r["majority_correct"]

        # Gain = +1 if method correct and greedy wrong
        #        -1 if method wrong and greedy correct
        #         0 otherwise
        maj_gain = int(maj_ok) - int(greedy_ok)

        h_ans_list.append(h_ans)
        maj_gain_list.append(maj_gain)
        greedy_correct_list.append(greedy_ok)

        if has_dad:
            dad_ok = r.get("dad_correct", False)
            dad_gain = int(dad_ok) - int(greedy_ok)
            dad_gain_list.append(dad_gain)

        te = r.get("mean_sample_token_entropy")
        if te is not None:
            token_ent_list.append(te)

    h_ans = np.array(h_ans_list)
    maj_gain = np.array(maj_gain_list)

    # ── Figure 1: Main Scatter (H_ans vs gain) ────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))

    # Bin by H_ans for cleaner visualization
    bins = [0.0, 0.001, 0.5, 1.0, 1.5, 2.0, 3.5]
    bin_labels = ["0", "(0, 0.5]", "(0.5, 1]", "(1, 1.5]", "(1.5, 2]", ">2"]
    bin_centers = [0.0, 0.25, 0.75, 1.25, 1.75, 2.5]

    def bin_gains(h_vals, gain_vals):
        binned_means = []
        binned_counts = []
        binned_sems = []
        for i in range(len(bins) - 1):
            lo, hi = bins[i], bins[i + 1]
            if i == 0:
                mask = h_vals <= hi
            else:
                mask = (h_vals > lo) & (h_vals <= hi)
            gains_in_bin = gain_vals[mask]
            if len(gains_in_bin) > 0:
                # Convert gain to "accuracy improvement" percentage
                # gain is per-problem: +1, 0, or -1
                # Mean gain = fraction_improved - fraction_regressed
                mean_gain = gains_in_bin.mean() * 100  # as percentage points
                sem = gains_in_bin.std() / np.sqrt(len(gains_in_bin)) * 100 if len(gains_in_bin) > 1 else 0
                binned_means.append(mean_gain)
                binned_sems.append(sem)
            else:
                binned_means.append(0)
                binned_sems.append(0)
            binned_counts.append(len(gains_in_bin))
        return np.array(binned_means), np.array(binned_sems), binned_counts

    maj_means, maj_sems, maj_counts = bin_gains(h_ans, maj_gain)

    # Plot Maj@8
    ax.bar(np.array(bin_centers) - 0.12, maj_means, width=0.22, 
           color="#3B82F6", alpha=0.7, edgecolor="white", linewidth=0.8,
           label="Maj@8 gain over Greedy", zorder=3)
    ax.errorbar(np.array(bin_centers) - 0.12, maj_means, yerr=maj_sems,
                fmt="none", ecolor="#1E40AF", capsize=3, capthick=1.2, zorder=4)

    if has_dad and len(dad_gain_list) > 0:
        dad_gain = np.array(dad_gain_list)
        dad_means, dad_sems, dad_counts = bin_gains(h_ans, dad_gain)

        ax.bar(np.array(bin_centers) + 0.12, dad_means, width=0.22,
               color="#EF4444", alpha=0.7, edgecolor="white", linewidth=0.8,
               label="DAD gain over Greedy", zorder=3)
        ax.errorbar(np.array(bin_centers) + 0.12, dad_means, yerr=dad_sems,
                    fmt="none", ecolor="#991B1B", capsize=3, capthick=1.2, zorder=4)

    # Add count annotations
    for i, (bc, mc) in enumerate(zip(bin_centers, maj_counts)):
        ax.text(bc, -8, f"n={mc}", ha="center", va="top", fontsize=8, color="#6B7280")

    ax.axhline(y=0, color="#9CA3AF", linewidth=0.8, linestyle="-", zorder=1)
    ax.set_xticks(bin_centers)
    ax.set_xticklabels(bin_labels, fontsize=10)
    ax.set_xlabel("Answer Entropy $H_{\\mathrm{ans}}$ (from $M=8$ samples)", fontsize=12)
    ax.set_ylabel("Accuracy Gain over Greedy (pp)", fontsize=12)
    ax.set_title("Entropy Inversion: Voting Fails at $H_{\\mathrm{ans}}=0$, DAD Does Not", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", frameon=True, fancybox=True, edgecolor="#E5E7EB")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    # Highlight the H_ans=0 region
    ax.axvspan(-0.3, 0.15, alpha=0.08, color="#EF4444", zorder=0)
    ax.text(0.0, ax.get_ylim()[1] * 0.9, "Entropy\nInversion\nRegime", 
            ha="center", va="top", fontsize=8, color="#991B1B", fontstyle="italic")

    fig.savefig(out_dir / "entropy_inversion.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(out_dir / "entropy_inversion.png", format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {out_dir / 'entropy_inversion.pdf'}")
    print(f"Saved: {out_dir / 'entropy_inversion.png'}")

    # ── Figure 2: Token Entropy vs Answer Entropy ─────────────────
    # This validates that token entropy does NOT predict answer correctness
    if token_ent_list:
        fig2, ax2 = plt.subplots(figsize=(7, 5))

        te = np.array(token_ent_list[:len(h_ans_list)])
        if len(te) == len(h_ans_list):
            colors = ["#22C55E" if g else "#EF4444" for g in greedy_correct_list]
            ax2.scatter(te, h_ans, c=colors, alpha=0.6, s=50, edgecolors="white", linewidths=0.5, zorder=3)

            # Legend
            legend_elems = [
                Line2D([0], [0], marker="o", color="w", markerfacecolor="#22C55E", markersize=8, label="Greedy Correct"),
                Line2D([0], [0], marker="o", color="w", markerfacecolor="#EF4444", markersize=8, label="Greedy Wrong"),
            ]
            ax2.legend(handles=legend_elems, loc="upper right", frameon=True, fancybox=True, edgecolor="#E5E7EB")

            ax2.set_xlabel("Mean Per-Token Entropy (bits)", fontsize=12)
            ax2.set_ylabel("Answer Entropy $H_{\\mathrm{ans}}$ (bits)", fontsize=12)
            ax2.set_title("Token Entropy vs. Answer Entropy\n(Low token entropy does NOT imply correct answers)", 
                         fontsize=12, fontweight="bold")
            ax2.spines["top"].set_visible(False)
            ax2.spines["right"].set_visible(False)
            ax2.grid(alpha=0.2)

            fig2.savefig(out_dir / "token_vs_answer_entropy.pdf", format="pdf", bbox_inches="tight")
            fig2.savefig(out_dir / "token_vs_answer_entropy.png", format="png", bbox_inches="tight", dpi=300)
            plt.close(fig2)
            print(f"Saved: {out_dir / 'token_vs_answer_entropy.pdf'}")

    # ── Print summary table ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("ENTROPY INVERSION SUMMARY")
    print("=" * 60)
    print(f"{'H_ans Bin':<12} {'Count':>6} {'Greedy':>8} {'Maj@8':>8} {'Maj Gain':>10}", end="")
    if has_dad:
        print(f" {'DAD':>8} {'DAD Gain':>10}", end="")
    print()
    print("-" * 60)

    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        if i == 0:
            mask = h_ans <= hi
            label = "0"
        else:
            mask = (h_ans > lo) & (h_ans <= hi)
            label = f"({lo},{hi}]"

        subset = [r for r, m in zip(results, mask) if m]
        if not subset:
            continue

        n = len(subset)
        g_acc = sum(1 for r in subset if r["greedy_correct"]) / n * 100
        m_acc = sum(1 for r in subset if r["majority_correct"]) / n * 100
        m_gain = m_acc - g_acc

        print(f"{label:<12} {n:>6} {g_acc:>7.1f}% {m_acc:>7.1f}% {m_gain:>+9.1f}pp", end="")

        if has_dad:
            d_acc = sum(1 for r in subset if r.get("dad_correct", False)) / n * 100
            d_gain = d_acc - g_acc
            print(f" {d_acc:>7.1f}% {d_gain:>+9.1f}pp", end="")
        print()

    print("=" * 60)


if __name__ == "__main__":
    main()