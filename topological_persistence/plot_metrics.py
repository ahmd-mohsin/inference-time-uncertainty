# Visualize topological persistence metrics across solved problems.
#
# Reads problem_*.json from a results dir and produces a panel of diagnostic plots:
#   1. Verdict / ceiling-probability bar chart
#   2. H1 features and H1 max-lifetime per problem (the core ceiling signal)
#   3. DTW (hidden-state) distance-matrix heatmaps per problem
#   4. DTW vs NCD distance scatter (representational vs surface diversity)
#   5. IID vs conditioned diversity comparison
#   6. MDS 2D embedding of chains from the DTW matrix (visualize clustering/loops)
#   7. Answer-agreement vs topology summary
#
# Usage:
#   python -m topological_persistence.plot_metrics --results-dir data/topological_outputs \
#       --out-dir data/topological_outputs/figures

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_results(results_dir: str) -> list[dict]:
    out = []
    for p in sorted(Path(results_dir).glob("problem_*.json")):
        with open(p) as f:
            out.append(json.load(f))
    return out


def _verdict_color(v: str) -> str:
    return {"CEILING_REACHED": "tab:red", "SCALABLE": "tab:green",
            "UNCERTAIN": "tab:orange"}.get(v, "tab:gray")


def plot_verdict_overview(results, out_dir):
    pids = [r["problem_id"] for r in results]
    probs = [r["signal"]["ceiling_probability"] for r in results]
    colors = [_verdict_color(r["signal"]["verdict"]) for r in results]

    fig, ax = plt.subplots(figsize=(max(6, len(results) * 0.8), 5))
    ax.bar([str(p) for p in pids], probs, color=colors)
    ax.axhline(0.7, ls="--", c="red", alpha=0.5, label="ceiling threshold (0.7)")
    ax.axhline(0.4, ls="--", c="orange", alpha=0.5, label="uncertain threshold (0.4)")
    ax.set_xlabel("Problem ID")
    ax.set_ylabel("Ceiling probability")
    ax.set_title("Ceiling probability per problem (red=ceiling, green=scalable)")
    ax.set_ylim(0, 1.05)
    ax.legend()
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "01_verdict_overview.png", dpi=150)
    plt.close()


def plot_h1_signals(results, out_dir):
    pids = [str(r["problem_id"]) for r in results]
    h1_n = [r["signal"]["h1_n_features"] for r in results]
    h1_lt = [r["signal"]["h1_max_lifetime"] for r in results]
    colors = [_verdict_color(r["signal"]["verdict"]) for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.bar(pids, h1_n, color=colors)
    ax1.set_xlabel("Problem ID")
    ax1.set_ylabel("H1 feature count (loops)")
    ax1.set_title("Number of H1 loops (strategic corridors)")

    ax2.bar(pids, h1_lt, color=colors)
    ax2.set_xlabel("Problem ID")
    ax2.set_ylabel("H1 max lifetime (persistence)")
    ax2.set_title("Longest-lived H1 loop (robustness of diversity)")
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "02_h1_signals.png", dpi=150)
    plt.close()


def plot_distance_heatmaps(results, out_dir):
    n = len(results)
    ncols = min(n, 5)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for i, r in enumerate(results):
        D = np.array(r["distance_matrix_iid"])
        ax = axes[i]
        im = ax.imshow(D, cmap="viridis")
        ax.set_title(f"P{r['problem_id']} ({r['signal']['verdict'][:4]})", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)
    for j in range(len(results), len(axes)):
        axes[j].axis("off")
    fig.suptitle("DTW distance matrices between hidden-state trajectories (IID chains)")
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "03_dtw_heatmaps.png", dpi=150)
    plt.close()


def plot_dtw_vs_ncd(results, out_dir):
    fig, ax = plt.subplots(figsize=(7, 6))
    for r in results:
        D_dtw = np.array(r["distance_matrix_iid"])
        D_ncd = np.array(r["distance_matrix_ncd"])
        iu = np.triu_indices_from(D_dtw, k=1)
        dtw_vals = D_dtw[iu]
        ncd_vals = D_ncd[iu]
        # normalize DTW to [0,1] within-problem so problems are comparable
        if dtw_vals.max() > 0:
            dtw_norm = dtw_vals / dtw_vals.max()
        else:
            dtw_norm = dtw_vals
        ax.scatter(ncd_vals, dtw_norm, alpha=0.5,
                   color=_verdict_color(r["signal"]["verdict"]),
                   label=f"P{r['problem_id']} ({r['signal']['verdict'][:4]})")
    ax.set_xlabel("NCD (text/surface diversity)")
    ax.set_ylabel("DTW (hidden-state diversity, normalized)")
    ax.set_title("Surface diversity vs representational diversity\n(pairwise chain distances)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "04_dtw_vs_ncd.png", dpi=150)
    plt.close()


def plot_iid_vs_conditioned(results, out_dir):
    pids = [str(r["problem_id"]) for r in results]
    div_iid = [r["comparison"]["diversity_iid"] for r in results if r.get("comparison")]
    div_cond = [r["comparison"]["diversity_conditioned"] for r in results if r.get("comparison")]
    valid_pids = [str(r["problem_id"]) for r in results if r.get("comparison")]

    x = np.arange(len(valid_pids))
    w = 0.38
    fig, ax = plt.subplots(figsize=(max(7, len(valid_pids) * 0.9), 5))
    ax.bar(x - w / 2, div_iid, w, label="IID", color="tab:blue")
    ax.bar(x + w / 2, div_cond, w, label="Conditioned (DAD)", color="tab:purple")
    ax.set_xticks(x); ax.set_xticklabels(valid_pids)
    ax.set_xlabel("Problem ID")
    ax.set_ylabel("Topological diversity score")
    ax.set_title("Diversity: IID sampling vs DAD-conditioned sampling")
    ax.legend()
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "05_iid_vs_conditioned.png", dpi=150)
    plt.close()


def plot_mds_embeddings(results, out_dir):
    from sklearn.manifold import MDS
    n = len(results)
    ncols = min(n, 5)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for i, r in enumerate(results):
        D = np.array(r["distance_matrix_iid"])
        ax = axes[i]
        if D.shape[0] >= 3 and D.max() > 0:
            mds = MDS(n_components=2, dissimilarity="precomputed",
                      random_state=42, normalized_stress="auto")
            coords = mds.fit_transform(D)
            answers = r["answers_iid"]
            # color by answer (empty answer = gray)
            uniq = {a: j for j, a in enumerate(sorted(set(answers)))}
            cols = [plt.cm.tab10(uniq[a] % 10) if a.strip() else (0.6, 0.6, 0.6, 1.0)
                    for a in answers]
            ax.scatter(coords[:, 0], coords[:, 1], c=cols, s=80, edgecolors="k")
            for j, (xx, yy) in enumerate(coords):
                lbl = answers[j] if answers[j].strip() else "∅"
                ax.annotate(lbl, (xx, yy), fontsize=7, ha="center", va="center")
        ax.set_title(f"P{r['problem_id']} H1={r['signal']['h1_n_features']} "
                     f"({r['signal']['verdict'][:4]})", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    for j in range(len(results), len(axes)):
        axes[j].axis("off")
    fig.suptitle("2D MDS of chains from DTW matrix (labels = answers, ∅ = truncated)")
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "06_mds_embeddings.png", dpi=150)
    plt.close()


def plot_summary_scatter(results, out_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    for r in results:
        sig = r["signal"]
        answers = r["answers_iid"]
        n_valid = sum(1 for a in answers if a.strip())
        agreement = 0.0
        if n_valid > 0:
            from collections import Counter
            c = Counter(a for a in answers if a.strip())
            agreement = c.most_common(1)[0][1] / len(answers)
        ax.scatter(sig["h1_max_lifetime"], agreement, s=140,
                   color=_verdict_color(sig["verdict"]), edgecolors="k")
        ax.annotate(f"P{r['problem_id']}", (sig["h1_max_lifetime"], agreement),
                    fontsize=9, ha="left", va="bottom")
    ax.set_xlabel("H1 max lifetime (representational diversity)")
    ax.set_ylabel("Answer agreement fraction (majority / total)")
    ax.set_title("Topology vs answer agreement\n"
                 "(low-left = stuck & disagree; high = diverse paths)")
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "07_topology_vs_agreement.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="data/topological_outputs")
    parser.add_argument("--out-dir", default="data/topological_outputs/figures")
    args = parser.parse_args()

    results = load_results(args.results_dir)
    if not results:
        print(f"No results in {args.results_dir}")
        return
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    plot_verdict_overview(results, args.out_dir)
    plot_h1_signals(results, args.out_dir)
    plot_distance_heatmaps(results, args.out_dir)
    plot_dtw_vs_ncd(results, args.out_dir)
    plot_iid_vs_conditioned(results, args.out_dir)
    plot_mds_embeddings(results, args.out_dir)
    plot_summary_scatter(results, args.out_dir)

    print(f"Saved 7 figures to {args.out_dir}:")
    for f in sorted(Path(args.out_dir).glob("*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
