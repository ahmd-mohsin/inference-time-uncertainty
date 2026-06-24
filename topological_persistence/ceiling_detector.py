# Ceiling detector: predicts compute scalability from trajectory-ensemble diversity.
#
# NOTE on the redesign: the original verdict was driven by H1 persistent-homology
# features, which were shown to be non-predictive on K=8 chains (AUC 0.33 for
# actually_scales -- worse than chance). The decisive verdict now comes from
# answer-distribution diversity (the strongest cheap signal) with spectral effective
# rank as a difficulty covariate. H1 fields are retained for reference/plots only.
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
    # --- primary signals (answer-distribution + spectral); the H1 fields above are now
    #     reference-only. Default to None so the topology-only call path still works.
    effective_rank: Optional[float] = None
    spectral_gain: Optional[float] = None
    answer_entropy: Optional[float] = None
    n_unique_answers: Optional[int] = None
    verdict_source: str = "topology"  # set to "answer_spectral" by detect_ceiling_v2


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


def detect_ceiling_v2(
    answers_iid: list,
    points_iid: "np.ndarray",
    points_cond: Optional["np.ndarray"] = None,
    sig_iid: Optional[TopologicalSignature] = None,
    entropy_scalable: float = 0.4,
    entropy_ceiling: float = 0.05,
) -> CeilingSignal:
    """Primary detector. Verdict from answer-distribution diversity (validated strongest
    cheap signal) + spectral effective rank (difficulty covariate). H1 is reference-only.

    Decision logic (answer entropy in nats over the K chains):
      - High answer entropy  => chains disagree => competing hypotheses exist that more
        compute / selection can resolve            -> SCALABLE
      - Near-zero entropy with non-blank consensus => the model is locked onto a single
        answer; sampling more won't change it       -> CEILING_REACHED
      - In between                                  -> UNCERTAIN
    Spectral effective rank and its IID->conditioned gain are recorded as covariates (rank
    tracks difficulty; gain tests whether conditioning expands the subspace vs reshuffles).

    Note: this predicts "are there competing strategies to resolve", NOT correctness. A
    confident-but-wrong problem reads as CEILING by design (that is the intended meaning:
    sampling alone won't help; a weight update would be needed).
    """
    from topological_persistence import spectral

    ent = spectral.answer_entropy(answers_iid)
    n_unique = spectral.n_unique_answers(answers_iid)
    blank = spectral.blank_fraction(answers_iid)
    er = spectral.effective_rank(points_iid)
    sgain = spectral.spectral_gain(points_iid, points_cond) if points_cond is not None else None

    # reference H1 fields (kept for plots / comparison only)
    h1_diag = next((d for d in sig_iid.diagrams if d.dimension == 1), None) if sig_iid else None
    h0_diag = next((d for d in sig_iid.diagrams if d.dimension == 0), None) if sig_iid else None
    h1_n = h1_diag.n_features if h1_diag else 0
    h1_max_lt = h1_diag.max_lifetime if h1_diag else 0.0
    h1_total = h1_diag.total_persistence if h1_diag else 0.0
    h0_stab = h0_stabilization(h0_diag) if h0_diag else 0.0
    conv_rate = betti_convergence_rate(sig_iid.betti_curves, sig_iid.radii) if sig_iid else 0.0

    if ent >= entropy_scalable:
        verdict = "SCALABLE"
    elif ent <= entropy_ceiling:
        verdict = "CEILING_REACHED"
    else:
        verdict = "UNCERTAIN"

    # map entropy to a [0,1] ceiling probability (monotone-decreasing in entropy)
    ceiling_prob = float(np.clip(1.0 - ent / max(entropy_scalable, 1e-8), 0.0, 1.0))

    return CeilingSignal(
        h0_stabilization_radius=h0_stab,
        h1_max_lifetime=h1_max_lt,
        h1_total_persistence=h1_total,
        h1_n_features=h1_n,
        betti_convergence_rate=conv_rate,
        topology_frozen=(h1_n == 0 and conv_rate < 0.1),
        diversity_score=er,
        ceiling_probability=ceiling_prob,
        verdict=verdict,
        effective_rank=er,
        spectral_gain=sgain,
        answer_entropy=ent,
        n_unique_answers=n_unique,
        verdict_source="answer_spectral",
    )
