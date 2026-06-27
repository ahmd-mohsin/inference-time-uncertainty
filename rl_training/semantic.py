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

def embed_texts(texts, model_name: str = None) -> np.ndarray:
    """Return (n, d) L2-normalized TF-IDF vectors for a list of texts.

    DESIGN: novelty must NOT instantiate a neural model inside the ZeRO-3 training process —
    DeepSpeed partitions any such model's params at runtime (regardless of zero3_init_flag),
    breaking its forward ('weight must be 2-D'). So we use a MODEL-FREE text representation:
    char/word n-gram TF-IDF (pure sklearn/numpy, no torch). It still captures *approach*
    diversity — different solution methods use different vocabulary/operators/structure —
    which is exactly what the group-relative novelty reward needs. Immune to DeepSpeed,
    needs no cache/download, and is fast on CPU. `model_name` is accepted for API
    compatibility but ignored.
    """
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)
    if len(texts) == 1:
        return np.ones((1, 1), dtype=np.float32)
    from sklearn.feature_extraction.text import TfidfVectorizer
    # word 1-2 grams capture method vocabulary ("substitution", "modular", "casework"...).
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=4096, sublinear_tf=True)
    try:
        X = vec.fit_transform(texts).toarray().astype(np.float32)
    except ValueError:
        # empty vocabulary (all-blank/degenerate) -> no novelty signal
        return np.zeros((len(texts), 1), dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


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
