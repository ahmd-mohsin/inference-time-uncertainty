# Visualization of persistence diagrams, Betti curves, and ceiling signals.
import numpy as np
from pathlib import Path
from typing import Optional

from topological_persistence.persistence import TopologicalSignature
from topological_persistence.ceiling_detector import CeilingSignal


def plot_persistence_diagram(sig: TopologicalSignature, save_path: Optional[str] = None, title: str = ""):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    colors = ["tab:blue", "tab:orange", "tab:green"]
    labels = ["H₀ (components)", "H₁ (loops)", "H₂ (voids)"]

    max_val = 0.0
    for d in sig.diagrams:
        if d.birth.size > 0:
            max_val = max(max_val, d.birth.max(), d.death.max())
        ax.scatter(d.birth, d.death, c=colors[d.dimension % 3],
                   label=labels[d.dimension % 3], alpha=0.7, s=40)

    diag_line = np.linspace(0, max_val * 1.1, 100)
    ax.plot(diag_line, diag_line, "k--", alpha=0.3)
    ax.set_xlabel("Birth")
    ax.set_ylabel("Death")
    ax.set_title(title or "Persistence Diagram")
    ax.legend()
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_betti_curves(sig: TopologicalSignature, save_path: Optional[str] = None, title: str = ""):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    colors = ["tab:blue", "tab:orange", "tab:green"]
    labels = ["β₀", "β₁", "β₂"]

    for dim in range(min(sig.betti_curves.shape[0], 3)):
        ax.plot(sig.radii, sig.betti_curves[dim], c=colors[dim], label=labels[dim], linewidth=2)

    ax.set_xlabel("Radius (ε)")
    ax.set_ylabel("Betti Number")
    ax.set_title(title or "Betti Curves")
    ax.legend()
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_comparison(
    sig_iid: TopologicalSignature,
    sig_cond: TopologicalSignature,
    signal: CeilingSignal,
    save_path: Optional[str] = None,
    title: str = "",
):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for d in sig_iid.diagrams:
        if d.birth.size > 0:
            ax.scatter(d.birth, d.death, c=colors[d.dimension % 3], alpha=0.7, s=30)
    max_val = max((d.death.max() for d in sig_iid.diagrams if d.death.size > 0), default=1.0)
    ax.plot([0, max_val], [0, max_val], "k--", alpha=0.3)
    ax.set_title("IID Persistence Diagram")
    ax.set_xlabel("Birth")
    ax.set_ylabel("Death")

    ax = axes[0, 1]
    for d in sig_cond.diagrams:
        if d.birth.size > 0:
            ax.scatter(d.birth, d.death, c=colors[d.dimension % 3], alpha=0.7, s=30)
    max_val = max((d.death.max() for d in sig_cond.diagrams if d.death.size > 0), default=1.0)
    ax.plot([0, max_val], [0, max_val], "k--", alpha=0.3)
    ax.set_title("Conditioned Persistence Diagram")
    ax.set_xlabel("Birth")
    ax.set_ylabel("Death")

    ax = axes[1, 0]
    for dim in range(min(sig_iid.betti_curves.shape[0], 3)):
        ax.plot(sig_iid.radii, sig_iid.betti_curves[dim], c=colors[dim],
                linewidth=2, label=f"β{dim} (IID)")
        ax.plot(sig_cond.radii, sig_cond.betti_curves[dim], c=colors[dim],
                linewidth=2, linestyle="--", label=f"β{dim} (Cond)")
    ax.set_title("Betti Curves Comparison")
    ax.set_xlabel("Radius")
    ax.set_ylabel("Betti Number")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.axis("off")
    info = (
        f"Verdict: {signal.verdict}\n"
        f"Ceiling Probability: {signal.ceiling_probability:.2f}\n"
        f"Topology Frozen: {signal.topology_frozen}\n"
        f"H₁ Features: {signal.h1_n_features}\n"
        f"H₁ Max Lifetime: {signal.h1_max_lifetime:.4f}\n"
        f"Diversity Score: {signal.diversity_score:.4f}\n"
        f"Betti Conv. Rate: {signal.betti_convergence_rate:.4f}"
    )
    ax.text(0.1, 0.5, info, fontsize=12, fontfamily="monospace",
            verticalalignment="center", transform=ax.transAxes,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.set_title("Ceiling Detection Signal")

    fig.suptitle(title or "Topological Ceiling Analysis", fontsize=14)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
