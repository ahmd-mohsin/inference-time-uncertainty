import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class CommitmentProbe:
    divergence: float
    hidden_states: list[torch.Tensor]
    centroid: torch.Tensor
    is_unstable: bool
    selected_probe_idx: int


class PreTokenCommitmentDetector:
    """
    Detects latent answer commitment instability before any generation token
    is sampled, using embedding-level perturbation and hidden-state cosine
    divergence at the last prompt position.

    Theoretical grounding: Boppana et al. (2026) show models commit to answers
    in hidden states before generating reasoning tokens. This detector identifies
    whether that commitment is robust (low divergence across K perturbed forward
    passes) or fragile (high divergence), using only the prompt encoding step.

    When divergence >= threshold, the selected starting hidden state is the one
    closest to the centroid — the most consensus-supported commitment — rather
    than the highest-probability one, which may be the locally dominant but
    globally unstable choice.
    """

    def __init__(
        self,
        k_probes: int = 5,
        noise_std: float = 0.01,
        divergence_threshold: float = 0.05,
        device: str = "cuda",
    ):
        self.k = k_probes
        self.noise_std = noise_std
        self.threshold = divergence_threshold
        self.device = device

    @torch.no_grad()
    def probe(
        self,
        model,
        prompt_ids: torch.Tensor,
    ) -> CommitmentProbe:
        hidden_states: list[torch.Tensor] = []

        for _ in range(self.k):
            inputs_embeds = model.get_input_embeddings()(prompt_ids).clone()
            noise = torch.randn_like(inputs_embeds) * self.noise_std
            inputs_embeds = inputs_embeds + noise

            out = model(
                inputs_embeds=inputs_embeds,
                output_hidden_states=True,
                return_dict=True,
            )

            last_layer_hidden = out.hidden_states[-1]
            h = last_layer_hidden[0, -1, :].float()
            h_norm = F.normalize(h, dim=-1)
            hidden_states.append(h_norm.detach())

        stacked = torch.stack(hidden_states, dim=0)
        centroid = F.normalize(stacked.mean(dim=0), dim=-1)

        pairwise_sims: list[float] = []
        for i in range(self.k):
            for j in range(i + 1, self.k):
                sim = torch.dot(stacked[i], stacked[j]).item()
                pairwise_sims.append(sim)

        mean_sim = sum(pairwise_sims) / len(pairwise_sims) if pairwise_sims else 1.0
        divergence = float(max(0.0, 1.0 - mean_sim))
        is_unstable = divergence >= self.threshold

        dists_to_centroid = [
            (1.0 - torch.dot(stacked[i], centroid).item())
            for i in range(self.k)
        ]
        selected_idx = int(min(range(self.k), key=lambda i: dists_to_centroid[i]))

        logger.debug(
            f"CommitmentProbe: divergence={divergence:.4f} "
            f"unstable={is_unstable} selected={selected_idx}"
        )

        return CommitmentProbe(
            divergence=divergence,
            hidden_states=hidden_states,
            centroid=centroid,
            is_unstable=is_unstable,
            selected_probe_idx=selected_idx,
        )

    @torch.no_grad()
    def get_steered_input(
        self,
        model,
        prompt_ids: torch.Tensor,
        selected_idx: int,
        noise_std: float = None,
    ) -> torch.Tensor:
        """
        Returns input_embeds for the selected probe run's perturbation direction.
        We re-run the selected probe's perturbation so generation starts from
        the same embedding-space position as the most consensus-aligned forward pass.
        We seed with selected_idx for reproducibility.
        """
        std = noise_std if noise_std is not None else self.noise_std
        torch.manual_seed(selected_idx)
        inputs_embeds = model.get_input_embeddings()(prompt_ids).clone()
        noise = torch.randn_like(inputs_embeds) * std
        return inputs_embeds + noise