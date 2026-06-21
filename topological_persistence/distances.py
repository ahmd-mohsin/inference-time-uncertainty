# Distance matrices between trajectories: cosine, DTW, Frechet.
import numpy as np
from typing import Optional


def cosine_distance_matrix(points: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normalized = points / norms
    sim = normalized @ normalized.T
    sim = np.clip(sim, -1.0, 1.0)
    return 1.0 - sim


def euclidean_distance_matrix(points: np.ndarray) -> np.ndarray:
    diff = points[:, None, :] - points[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


def dtw_distance(curve_a: np.ndarray, curve_b: np.ndarray) -> float:
    n, m = curve_a.shape[0], curve_b.shape[0]
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = np.linalg.norm(curve_a[i - 1] - curve_b[j - 1])
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return cost[n, m] / (n + m)


def frechet_distance(curve_a: np.ndarray, curve_b: np.ndarray) -> float:
    n, m = curve_a.shape[0], curve_b.shape[0]
    ca = np.full((n, m), -1.0)

    def _c(i, j):
        if ca[i, j] > -0.5:
            return ca[i, j]
        d = np.linalg.norm(curve_a[i] - curve_b[j])
        if i == 0 and j == 0:
            ca[i, j] = d
        elif i == 0:
            ca[i, j] = max(_c(0, j - 1), d)
        elif j == 0:
            ca[i, j] = max(_c(i - 1, 0), d)
        else:
            ca[i, j] = max(min(_c(i - 1, j), _c(i - 1, j - 1), _c(i, j - 1)), d)
        return ca[i, j]

    return _c(n - 1, m - 1)


def curve_distance_matrix(curves: list[np.ndarray], metric: str = "dtw") -> np.ndarray:
    n = len(curves)
    D = np.zeros((n, n))
    dist_fn = dtw_distance if metric == "dtw" else frechet_distance
    for i in range(n):
        for j in range(i + 1, n):
            d = dist_fn(curves[i], curves[j])
            D[i, j] = d
            D[j, i] = d
    return D


def compute_distance_matrix(embedding_data: dict, metric: str = "cosine") -> np.ndarray:
    if embedding_data["type"] == "point":
        points = embedding_data["points"]
        if metric == "cosine":
            return cosine_distance_matrix(points)
        return euclidean_distance_matrix(points)
    else:
        curves = embedding_data["curves"]
        curve_metric = "dtw" if metric in ("dtw", "cosine") else "frechet"
        return curve_distance_matrix(curves, metric=curve_metric)
