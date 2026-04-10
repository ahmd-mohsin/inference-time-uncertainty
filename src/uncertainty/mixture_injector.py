import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class InjectionRecord:
    position: int
    entropy: float
    top_k: int
    mixture_weight_spread: float


class EntropyGatedMixtureInjector:
    def __init__(
        self,
        model,
        low_entropy_threshold: float,
        strategy_window: int = 40,
        top_k: int = 2,
        device: str = "cuda",
    ):
        self.model = model
        self.low_entropy_threshold = low_entropy_threshold
        self.strategy_window = strategy_window
        self.top_k = top_k
        self.device = device
        self._embedding = model.get_input_embeddings()
        self._model_dtype = next(model.parameters()).dtype

    def should_inject(self, entropy: float, position: int) -> bool:
        return position < self.strategy_window and entropy < self.low_entropy_threshold

    @torch.no_grad()
    def build_mixture_embedding(self, logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits.float(), dim=-1)
        top_probs, top_ids = probs.topk(self.top_k, dim=-1)
        top_probs_norm = (top_probs / top_probs.sum()).to(self._model_dtype)
        top_embeds = self._embedding(top_ids).to(self._model_dtype)
        mixture = (top_probs_norm.unsqueeze(-1) * top_embeds).sum(dim=0)
        return mixture

    @torch.no_grad()
    def forward_with_mixture(
        self,
        current_embeds: torch.Tensor,
        mixture_embed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        new_embeds = torch.cat(
            [current_embeds, mixture_embed.to(self._model_dtype).unsqueeze(0).unsqueeze(0)],
            dim=1,
        )
        out = self.model(inputs_embeds=new_embeds, return_dict=True)
        next_logits = out.logits[0, -1, :]
        return next_logits, new_embeds

    def mixture_weight_spread(self, logits: torch.Tensor) -> float:
        probs = F.softmax(logits.float(), dim=-1)
        top_probs, _ = probs.topk(self.top_k, dim=-1)
        top_probs_norm = top_probs / top_probs.sum()
        return float(top_probs_norm[0] - top_probs_norm[-1])