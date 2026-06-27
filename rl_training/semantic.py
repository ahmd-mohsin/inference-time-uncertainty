# Semantic embedding model for the group-relative novelty reward (Component A, docs/RL.md §4.1).
#
# Uses a PRE-TRAINED sentence-embedding model (default all-MiniLM-L6-v2) to embed each
# rollout's reasoning text, so "novelty" = semantic distance between *approaches*, not
# surface tokens. Loaded once, cached process-wide (the reward fn is called every step).
#
# NOTE (from our own prior negative result, topological_persistence/METHODOLOGY.md):
# raw high-dim hidden-state distances concentrate and become uninformative. A dedicated
# sentence-embedding model is L2-normalized and trained for semantic similarity, which
# sidesteps that failure — but we still validate separation in tests before trusting it.

import numpy as np

_MODEL_CACHE = {}


def get_embedder(model_name: str):
    """Lazily load and cache a SentenceTransformer (one per process)."""
    if model_name not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name, device=device)
    return _MODEL_CACHE[model_name]


def embed_texts(texts, model_name: str) -> np.ndarray:
    """Return (n, d) normalized embeddings for a list of texts."""
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)
    model = get_embedder(model_name)
    emb = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True,
                       show_progress_bar=False, batch_size=64)
    return emb.astype(np.float32)


def pairwise_novelty(embeddings: np.ndarray, metric: str = "cosine") -> np.ndarray:
    """Per-item mean distance to the OTHER items (the group-relative novelty score).

    embeddings: (n, d), L2-normalized. Returns (n,) where entry i = mean distance from i
    to all j != i. With <2 items, returns zeros (a lone correct rollout has no peers to be
    novel against -> no bonus, which is the intended behavior).
    """
    n = embeddings.shape[0]
    if n < 2:
        return np.zeros(n, dtype=np.float32)
    if metric == "cosine":
        sim = embeddings @ embeddings.T            # cosine sim (normalized inputs)
        dist = 1.0 - sim                           # cosine distance in [0, 2]
    else:  # euclidean
        diff = embeddings[:, None, :] - embeddings[None, :, :]
        dist = np.sqrt((diff ** 2).sum(-1))
    np.fill_diagonal(dist, 0.0)
    return (dist.sum(axis=1) / (n - 1)).astype(np.float32)
