# Ceiling detector: uses topological signatures to predict compute scalability.
import numpy as np
from dataclasses import dataclass
from typing import Optional

from topological_persistence.persistence import TopologicalSignature, PersistenceDiagram


@dataclass
class CeilingSignal:
    h0_stabilization_radius: float
    h1_max_lifetime: float
    h1_total_persistence: float
    h1_n_features: int
    betti_convergence_rate: float
    topology_frozen: bool
    diversity_score: float
    ceiling_probability: float
    verdict: str


def betti_convergence_rate(betti_curves: np.ndarray, radii: np.ndarray, window: int = 3) -> float:
    if betti_curves.shape[1] < window * 2:
        return 0.0
    total_variation = 0.0
    for dim in range(betti_curves.shape[0]):
        curve = betti_curves[dim]
        diffs = np.abs(np.diff(curve))
        n = len(diffs)
        first_half = diffs[:n // 2].sum()
        second_half = diffs[n // 2:].sum()
        if first_half > 0:
            total_variation += second_half / first_half
        else:
            total_variation += 0.0
    return total_variation / max(betti_curves.shape[0], 1)


def h0_stabilization(diagram_h0: PersistenceDiagram) -> float:
    if diagram_h0.n_features == 0:
        return 0.0
    deaths = np.sort(diagram_h0.death)
    if len(deaths) >= 2:
        return float(deaths[-2])
    return float(deaths[-1]) if len(deaths) > 0 else 0.0


def diversity_from_persistence(sig: TopologicalSignature) -> float:
    score = 0.0
    for d in sig.diagrams:
        if d.dimension == 0:
            score += d.total_persistence * 0.3
        elif d.dimension == 1:
            score += d.total_persistence * 1.0 + d.n_features * 0.5
        elif d.dimension == 2:
            score += d.total_persistence * 2.0 + d.n_features * 1.0
    return score


def detect_ceiling(
    sig_iid: TopologicalSignature,
    sig_conditioned: Optional[TopologicalSignature] = None,
    threshold: float = 0.3,
) -> CeilingSignal:
    h0_diag = next((d for d in sig_iid.diagrams if d.dimension == 0), None)
    h1_diag = next((d for d in sig_iid.diagrams if d.dimension == 1), None)

    h0_stab = h0_stabilization(h0_diag) if h0_diag else 0.0
    h1_max_lt = h1_diag.max_lifetime if h1_diag else 0.0
    h1_total = h1_diag.total_persistence if h1_diag else 0.0
    h1_n = h1_diag.n_features if h1_diag else 0

    conv_rate = betti_convergence_rate(sig_iid.betti_curves, sig_iid.radii)

    topology_frozen = (h1_n == 0 and conv_rate < 0.1)

    div_iid = diversity_from_persistence(sig_iid)
    div_cond = diversity_from_persistence(sig_conditioned) if sig_conditioned else None

    if div_cond is not None and div_iid > 0:
        diversity_gain = (div_cond - div_iid) / div_iid
    else:
        diversity_gain = 0.0

    ceiling_prob = 0.0
    if topology_frozen:
        ceiling_prob += 0.4
    if h1_max_lt < threshold:
        ceiling_prob += 0.3
    if div_cond is not None and diversity_gain < 0.1:
        ceiling_prob += 0.2
    if conv_rate < 0.05:
        ceiling_prob += 0.1
    ceiling_prob = min(ceiling_prob, 1.0)

    if ceiling_prob >= 0.7:
        verdict = "CEILING_REACHED"
    elif ceiling_prob >= 0.4:
        verdict = "UNCERTAIN"
    else:
        verdict = "SCALABLE"

    return CeilingSignal(
        h0_stabilization_radius=h0_stab,
        h1_max_lifetime=h1_max_lt,
        h1_total_persistence=h1_total,
        h1_n_features=h1_n,
        betti_convergence_rate=conv_rate,
        topology_frozen=topology_frozen,
        diversity_score=div_iid,
        ceiling_probability=ceiling_prob,
        verdict=verdict,
    )


def compare_topologies(
    sig_iid: TopologicalSignature,
    sig_conditioned: TopologicalSignature,
) -> dict:
    div_iid = diversity_from_persistence(sig_iid)
    div_cond = diversity_from_persistence(sig_conditioned)

    h1_iid = next((d for d in sig_iid.diagrams if d.dimension == 1), None)
    h1_cond = next((d for d in sig_conditioned.diagrams if d.dimension == 1), None)

    return {
        "diversity_iid": div_iid,
        "diversity_conditioned": div_cond,
        "diversity_gain": (div_cond - div_iid) / max(div_iid, 1e-8),
        "h1_features_iid": h1_iid.n_features if h1_iid else 0,
        "h1_features_conditioned": h1_cond.n_features if h1_cond else 0,
        "h1_max_lifetime_iid": h1_iid.max_lifetime if h1_iid else 0.0,
        "h1_max_lifetime_conditioned": h1_cond.max_lifetime if h1_cond else 0.0,
        "new_topological_features": (
            (h1_cond.n_features if h1_cond else 0) > (h1_iid.n_features if h1_iid else 0)
        ),
    }
