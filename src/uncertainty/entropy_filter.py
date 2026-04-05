import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class EntropyRecord:
    position: int
    entropy: float
    top1_prob: float
    top2_prob: float
    logit_margin: float
    token_id: int
    token_str: str
    in_semantic_zone: bool
    triggered: bool


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    return -(probs * log_probs).sum(dim=-1)


def compute_logit_margin(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    sorted_probs, _ = probs.sort(dim=-1, descending=True)
    return sorted_probs[..., 0] - sorted_probs[..., 1]


def compute_top_probs(logits: torch.Tensor, k: int = 2) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    top_probs, _ = probs.topk(min(k, probs.shape[-1]), dim=-1)
    return top_probs


class EntropyPreFilter:
    def __init__(self, threshold: float = 1.0, min_tokens_before_trigger: int = 20):
        self.threshold = threshold
        self.min_tokens_before_trigger = min_tokens_before_trigger

    def should_trigger(
        self,
        logits: torch.Tensor,
        position: int,
        in_semantic_zone: bool,
    ) -> tuple[bool, float]:
        if position < self.min_tokens_before_trigger:
            return False, 0.0
        if not in_semantic_zone:
            return False, 0.0
        entropy = compute_entropy(logits.squeeze()).item()
        return entropy > self.threshold, entropy

    def full_record(
        self,
        logits: torch.Tensor,
        position: int,
        token_id: int,
        token_str: str,
        in_semantic_zone: bool,
    ) -> EntropyRecord:
        squeezed = logits.squeeze()
        entropy = compute_entropy(squeezed).item()
        margin = compute_logit_margin(squeezed).item()
        top_probs = compute_top_probs(squeezed, k=2)
        triggered = (
            position >= self.min_tokens_before_trigger
            and in_semantic_zone
            and entropy > self.threshold
        )
        return EntropyRecord(
            position=position,
            entropy=entropy,
            top1_prob=top_probs[0].item(),
            top2_prob=top_probs[1].item() if top_probs.shape[-1] > 1 else 0.0,
            logit_margin=margin,
            token_id=token_id,
            token_str=token_str,
            in_semantic_zone=in_semantic_zone,
            triggered=triggered,
        )