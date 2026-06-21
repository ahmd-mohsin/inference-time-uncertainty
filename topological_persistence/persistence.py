# Persistent homology computation and topological feature extraction.
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PersistenceDiagram:
    birth: np.ndarray
    death: np.ndarray
    dimension: int

    @property
    def lifetimes(self) -> np.ndarray:
        return self.death - self.birth

    @property
    def max_lifetime(self) -> float:
        if len(self.lifetimes) == 0:
            return 0.0
        return float(self.lifetimes.max())

    @property
    def total_persistence(self) -> float:
        return float(self.lifetimes.sum())

    @property
    def n_features(self) -> int:
        return len(self.birth)


@dataclass
class TopologicalSignature:
    diagrams: list[PersistenceDiagram] = field(default_factory=list)
    betti_curves: Optional[np.ndarray] = None
    radii: Optional[np.ndarray] = None

    @property
    def max_dim(self) -> int:
        return max((d.dimension for d in self.diagrams), default=-1)

    def betti_at_radius(self, r: float, dim: int) -> int:
        if self.betti_curves is None or self.radii is None:
            return 0
        idx = np.searchsorted(self.radii, r)
        idx = min(idx, len(self.radii) - 1)
        if dim < self.betti_curves.shape[0]:
            return int(self.betti_curves[dim, idx])
        return 0


def compute_persistence_ripser(distance_matrix: np.ndarray, max_dim: int = 2) -> list[PersistenceDiagram]:
    from ripser import ripser
    result = ripser(distance_matrix, maxdim=max_dim, distance_matrix=True)
    diagrams = []
    for dim, dgm in enumerate(result["dgms"]):
        finite_mask = np.isfinite(dgm[:, 1])
        birth = dgm[finite_mask, 0]
        death = dgm[finite_mask, 1]
        diagrams.append(PersistenceDiagram(birth=birth, death=death, dimension=dim))
    return diagrams


def compute_persistence_gudhi(distance_matrix: np.ndarray, max_dim: int = 2) -> list[PersistenceDiagram]:
    import gudhi
    rips = gudhi.RipsComplex(distance_matrix=distance_matrix, max_edge_length=np.inf)
    st = rips.create_simplex_tree(max_dimension=max_dim + 1)
    st.compute_persistence()
    diagrams = []
    for dim in range(max_dim + 1):
        pairs = st.persistence_intervals_in_dimension(dim)
        if len(pairs) == 0:
            diagrams.append(PersistenceDiagram(
                birth=np.array([]), death=np.array([]), dimension=dim
            ))
            continue
        finite_mask = np.isfinite(pairs[:, 1])
        birth = pairs[finite_mask, 0]
        death = pairs[finite_mask, 1]
        diagrams.append(PersistenceDiagram(birth=birth, death=death, dimension=dim))
    return diagrams


def compute_persistence(distance_matrix: np.ndarray, max_dim: int = 2) -> list[PersistenceDiagram]:
    try:
        return compute_persistence_ripser(distance_matrix, max_dim)
    except ImportError:
        pass
    try:
        return compute_persistence_gudhi(distance_matrix, max_dim)
    except ImportError:
        pass
    raise ImportError("Install either 'ripser' or 'gudhi' for persistent homology computation.")


def compute_betti_curves(diagrams: list[PersistenceDiagram], n_radii: int = 100) -> tuple[np.ndarray, np.ndarray]:
    max_death = 0.0
    for d in diagrams:
        if d.death.size > 0:
            max_death = max(max_death, d.death.max())
    if max_death == 0:
        max_death = 1.0

    radii = np.linspace(0, max_death * 1.1, n_radii)
    max_dim = max((d.dimension for d in diagrams), default=0)
    betti = np.zeros((max_dim + 1, n_radii), dtype=int)

    for d in diagrams:
        for i, r in enumerate(radii):
            alive = np.sum((d.birth <= r) & (d.death > r))
            betti[d.dimension, i] = alive

    return betti, radii


def compute_topological_signature(
    distance_matrix: np.ndarray, max_dim: int = 2, n_radii: int = 100
) -> TopologicalSignature:
    diagrams = compute_persistence(distance_matrix, max_dim)
    betti, radii = compute_betti_curves(diagrams, n_radii)
    return TopologicalSignature(diagrams=diagrams, betti_curves=betti, radii=radii)
