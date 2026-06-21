# Trajectory embedding extraction: point, curve, and step-level representations.
import numpy as np
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


def embed_chains(chains: list[Chain], cfg: EmbeddingConfig) -> dict:
    if cfg.representation == "point":
        points = np.stack([get_point_embedding(c, cfg) for c in chains])
        return {"points": points, "type": "point"}
    elif cfg.representation == "curve":
        curves = [get_curve_embedding(c, cfg) for c in chains]
        return {"curves": curves, "type": "curve"}
    elif cfg.representation == "steps":
        steps = [get_step_embeddings(c, cfg) for c in chains]
        return {"curves": steps, "type": "curve"}
    else:
        points = np.stack([get_point_embedding(c, cfg) for c in chains])
        return {"points": points, "type": "point"}
