# Trajectory embedding extraction: point, curve, step-level, and text-based representations.
import numpy as np
import gzip
from typing import Optional

from topological_persistence.config import EmbeddingConfig
from topological_persistence.sampler import Chain


def extract_step_boundaries(text: str) -> list[int]:
    boundaries = [0]
    lines = text.split("\n")
    pos = 0
    in_blank = False
    for line in lines:
        if line.strip() == "":
            in_blank = True
        else:
            if in_blank:
                boundaries.append(pos)
                in_blank = False
        pos += len(line) + 1
    return boundaries


def subsample_hidden_states(h: np.ndarray, max_points: int) -> np.ndarray:
    if h.shape[0] <= max_points:
        return h
    indices = np.linspace(0, h.shape[0] - 1, max_points, dtype=int)
    return h[indices]


def _has_hidden_states(chains: list[Chain]) -> bool:
    return any(c.hidden_states is not None and c.hidden_states.shape[0] > 0 for c in chains)


def get_point_embedding(chain: Chain, cfg: EmbeddingConfig) -> np.ndarray:
    if chain.hidden_states is not None and chain.hidden_states.shape[0] > 0:
        return chain.hidden_states.mean(axis=0)
    return np.zeros(1)


def get_curve_embedding(chain: Chain, cfg: EmbeddingConfig) -> np.ndarray:
    if chain.hidden_states is None or chain.hidden_states.shape[0] == 0:
        return np.zeros((1, 1))
    h = subsample_hidden_states(chain.hidden_states, cfg.max_steps_per_chain)
    return h


def get_step_embeddings(chain: Chain, cfg: EmbeddingConfig) -> np.ndarray:
    if chain.hidden_states is None or chain.hidden_states.shape[0] == 0:
        return np.zeros((1, 1))
    boundaries = extract_step_boundaries(chain.text)
    if len(boundaries) <= 1:
        return subsample_hidden_states(chain.hidden_states, cfg.max_steps_per_chain)

    h = chain.hidden_states
    n_tokens = h.shape[0]
    step_embeds = []
    for i in range(len(boundaries)):
        start_char = boundaries[i]
        end_char = boundaries[i + 1] if i + 1 < len(boundaries) else len(chain.text)
        ratio_start = start_char / max(len(chain.text), 1)
        ratio_end = end_char / max(len(chain.text), 1)
        t_start = int(ratio_start * n_tokens)
        t_end = int(ratio_end * n_tokens)
        if t_end > t_start:
            step_embeds.append(h[t_start:t_end].mean(axis=0))
    if step_embeds:
        return np.stack(step_embeds)
    return subsample_hidden_states(h, cfg.max_steps_per_chain)


def _ngram_vector(text: str, n: int = 3, vocab_size: int = 4096) -> np.ndarray:
    vec = np.zeros(vocab_size, dtype=np.float32)
    text_lower = text.lower()
    for i in range(len(text_lower) - n + 1):
        gram = text_lower[i:i+n]
        idx = hash(gram) % vocab_size
        vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def _step_ngram_curves(chain: Chain, cfg: EmbeddingConfig, n: int = 3, vocab_size: int = 2048) -> np.ndarray:
    boundaries = extract_step_boundaries(chain.text)
    if len(boundaries) <= 1:
        chunks = [chain.text[i:i+200] for i in range(0, len(chain.text), 200)]
    else:
        chunks = []
        for i in range(len(boundaries)):
            start = boundaries[i]
            end = boundaries[i+1] if i+1 < len(boundaries) else len(chain.text)
            chunks.append(chain.text[start:end])

    chunks = [c for c in chunks if c.strip()]
    if not chunks:
        return np.zeros((1, vocab_size), dtype=np.float32)

    max_steps = cfg.max_steps_per_chain
    if len(chunks) > max_steps:
        indices = np.linspace(0, len(chunks)-1, max_steps, dtype=int)
        chunks = [chunks[i] for i in indices]

    vecs = []
    for chunk in chunks:
        vec = np.zeros(vocab_size, dtype=np.float32)
        chunk_lower = chunk.lower()
        for j in range(len(chunk_lower) - n + 1):
            gram = chunk_lower[j:j+n]
            idx = hash(gram) % vocab_size
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        vecs.append(vec)
    return np.stack(vecs)


def ncd_distance_matrix(chains: list[Chain]) -> np.ndarray:
    n = len(chains)
    compressed = []
    for c in chains:
        data = c.text.encode("utf-8")
        compressed.append(len(gzip.compress(data)))

    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            combined = (chains[i].text + chains[j].text).encode("utf-8")
            c_ij = len(gzip.compress(combined))
            c_min = min(compressed[i], compressed[j])
            c_max = max(compressed[i], compressed[j])
            ncd = (c_ij - c_min) / max(c_max, 1)
            D[i, j] = ncd
            D[j, i] = ncd
    return D


def embed_chains(chains: list[Chain], cfg: EmbeddingConfig) -> dict:
    if not _has_hidden_states(chains):
        raise ValueError("No hidden states available. Hidden states are required for topological analysis.")

    if cfg.representation == "point":
        points = np.stack([get_point_embedding(c, cfg) for c in chains])
        return {"points": points, "type": "point"}
    elif cfg.representation == "curve":
        curves = [get_curve_embedding(c, cfg) for c in chains]
        return {"curves": curves, "type": "curve"}
    elif cfg.representation == "steps":
        steps = [get_step_embeddings(c, cfg) for c in chains]
        return {"curves": steps, "type": "curve"}

    points = np.stack([get_point_embedding(c, cfg) for c in chains])
    return {"points": points, "type": "point"}
