import logging
from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np

from src.search.tree import StepNode

logger = logging.getLogger(__name__)

def compute_kl_divergence_topk(
    node_topk_ids: list[int],
    node_topk_logprobs: list[float],
    pool_topk_ids: list[list[int]],
    pool_topk_logprobs: list[list[float]],
    vocab_size: int = 152064,
    smoothing: float = 1e-8,
) -> float:
    if not pool_topk_ids:
        return 0.0

    all_ids = set(node_topk_ids)
    for ids in pool_topk_ids:
        all_ids.update(ids)
    all_ids = sorted(all_ids)

    def _build_prob_vec(topk_ids, topk_logprobs):
        id2lp = dict(zip(topk_ids, topk_logprobs))
        if topk_logprobs:
            topk_mass = sum(np.exp(lp) for lp in topk_logprobs)
        else:
            topk_mass = 0.0
        residual = max(smoothing, (1.0 - topk_mass)) / max(1, vocab_size - len(topk_ids))
        probs = []
        for tid in all_ids:
            if tid in id2lp:
                probs.append(max(smoothing, np.exp(id2lp[tid])))
            else:
                probs.append(residual)
        return np.array(probs, dtype=np.float64)

    p_node = _build_prob_vec(node_topk_ids, node_topk_logprobs)
    p_node /= p_node.sum()

    pool_probs = []
    for ids, lps in zip(pool_topk_ids, pool_topk_logprobs):
        pool_probs.append(_build_prob_vec(ids, lps))
    p_pool = np.mean(pool_probs, axis=0)
    p_pool /= p_pool.sum()

    kl = float(np.sum(p_node * np.log(p_node / p_pool)))
    return max(0.0, kl)

@torch.no_grad()
def compute_kl_divergence_full(
    node_logits: torch.Tensor,
    pool_logits: list[torch.Tensor],
) -> float:
    if not pool_logits:
        return 0.0

    p_node = F.softmax(node_logits.float(), dim=-1)
    log_p_node = F.log_softmax(node_logits.float(), dim=-1)

    pool_stack = torch.stack([F.softmax(lg.float(), dim=-1) for lg in pool_logits], dim=0)
    p_pool = pool_stack.mean(dim=0)
    log_p_pool = torch.log(p_pool + 1e-10)

    kl = float(torch.sum(p_node * (log_p_node - log_p_pool)).item())
    return max(0.0, kl)

@torch.no_grad()
def compute_hidden_state_divergence(
    node_hidden: torch.Tensor,
    pool_hiddens: list[torch.Tensor],
) -> float:
    if not pool_hiddens:
        return 0.0

    h = F.normalize(node_hidden.float(), dim=-1)
    pool_stack = torch.stack([F.normalize(ph.float(), dim=-1) for ph in pool_hiddens])
    centroid = F.normalize(pool_stack.mean(dim=0), dim=-1)

    cos_sim = float(torch.dot(h, centroid).item())
    return max(0.0, 1.0 - cos_sim)

def compute_combined_score(
    prm_reward: float,
    diversity: float,
    lambda_div: float = 0.5,
    prm_weight: float = 1.0,
) -> float:
    return prm_weight * prm_reward + lambda_div * diversity