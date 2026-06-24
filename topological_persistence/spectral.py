# Spectral + answer-distribution diversity signals.
#
# Why this module exists: the persistent-homology pipeline (persistence.py +
# ceiling_detector H1 logic) was shown to be non-predictive on K=8 chains -- 8 points in
# ~5000-D suffer distance concentration (pairwise-distance CV ~3% under DTW), so every H1
# "loop" is noise, and H1_features had AUC 0.33 (worse than chance) for predicting whether
# a problem actually scales.
#
# Offline re-analysis (spectral_reanalysis.py) of the AIME-2026 run showed:
#   - spectral effective rank tracks problem DIFFICULTY (Spearman +0.69 vs coverage)
#     but not scalability directly  -> good difficulty covariate (Direction 2)
#   - answer-distribution stats (entropy / unique count) and NCD were the STRONGEST
#     cheap predictors and must be the baseline any geometric signal has to beat.
#
# So this module makes those signals first-class. Hidden-state geometry is kept as a
# difficulty covariate, NOT the decisive ceiling signal.

import numpy as np
from collections import Counter


# --------------------------------------------------------------------- spectral signals

def effective_rank(point_matrix: np.ndarray) -> float:
    """Entropy-based (participation-ratio) effective rank of a K x D point cloud.

    Mean-centered so we measure spread, not absolute position. Low value => chains live
    in a collapsed low-dimensional subspace (the Direction-2 ceiling intuition).
    Bounded in [0, min(K, D)]; for K=8 chains the max is ~8.
    """
    if point_matrix.ndim != 2 or point_matrix.shape[0] < 2:
        return 0.0
    M = point_matrix - point_matrix.mean(axis=0, keepdims=True)
    s = np.linalg.svd(M, compute_uv=False)
    s2 = s ** 2
    if s2.sum() <= 0:
        return 0.0
    p = s2 / s2.sum()
    return float(np.exp(-(p * np.log(p + 1e-12)).sum()))


def energy_rank(point_matrix: np.ndarray, energy: float = 0.95) -> int:
    """Number of singular values needed to capture `energy` fraction of variance."""
    if point_matrix.ndim != 2 or point_matrix.shape[0] < 2:
        return 0
    M = point_matrix - point_matrix.mean(axis=0, keepdims=True)
    s = np.linalg.svd(M, compute_uv=False)
    s2 = s ** 2
    if s2.sum() <= 0:
        return 0
    c = np.cumsum(s2) / s2.sum()
    return int(np.searchsorted(c, energy) + 1)


def spectral_gain(points_iid: np.ndarray, points_cond: np.ndarray) -> float:
    """Relative change in effective rank from IID -> conditioned chains.

    Direct test of the conditioning paradox (questions_and_directions Direction 2 /
    meta-question): does conditioning EXPAND the effective subspace (>0) or merely
    rotate/reshuffle within it (~0)?
    """
    er_iid = effective_rank(points_iid)
    er_cond = effective_rank(points_cond)
    if er_iid <= 1e-8:
        return 0.0
    return float((er_cond - er_iid) / er_iid)


# ------------------------------------------------------------- answer-distribution signals

def answer_entropy(answers: list[str]) -> float:
    """Shannon entropy (nats) of the answer distribution; blank answers dropped."""
    vals = [a.strip() for a in answers if a and a.strip()]
    if not vals:
        return 0.0
    counts = np.array(list(Counter(vals).values()), dtype=float)
    p = counts / counts.sum()
    return float(-(p * np.log(p + 1e-12)).sum())


def n_unique_answers(answers: list[str]) -> int:
    return len({a.strip() for a in answers if a and a.strip()})


def majority_fraction(answers: list[str]) -> float:
    """Fraction of (non-blank) chains agreeing with the modal answer; 0 if all blank."""
    vals = [a.strip() for a in answers if a and a.strip()]
    if not vals:
        return 0.0
    top = Counter(vals).most_common(1)[0][1]
    return top / len(vals)


def blank_fraction(answers: list[str]) -> float:
    if not answers:
        return 0.0
    return sum(1 for a in answers if not (a and a.strip())) / len(answers)
