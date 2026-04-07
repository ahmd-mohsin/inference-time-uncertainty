import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class ContinuationDisagreement:
    position: int
    disagreement_score: float
    pairwise_similarities: list[float]
    continuation_token_ids: list[list[int]]
    continuation_log_probs: list[float]
    triggered: bool


class SemanticDisagreementDetector:
    def __init__(
        self,
        k_continuations: int = 3,
        continuation_length: int = 12,
        temperature: float = 0.8,
        disagreement_threshold: float = 0.3,
    ):
        self.k = k_continuations
        self.L = continuation_length
        self.temperature = temperature
        self.threshold = disagreement_threshold

    @torch.no_grad()
    def compute(
        self,
        model,
        input_ids: torch.Tensor,
        past_key_values,
        position: int,
    ) -> ContinuationDisagreement:
        continuation_ids_list: list[list[int]] = []
        continuation_log_probs_list: list[float] = []
        final_hidden_states: list[torch.Tensor] = []

        for _ in range(self.k):
            out = model.generate(
                input_ids=input_ids,
                max_new_tokens=self.L,
                do_sample=True,
                temperature=self.temperature,
                output_scores=True,
                output_hidden_states=True,
                return_dict_in_generate=True,
                pad_token_id=model.config.pad_token_id or model.config.eos_token_id,
                eos_token_id=model.config.eos_token_id,
            )

            new_ids = out.sequences[0, input_ids.shape[1]:].tolist()
            continuation_ids_list.append(new_ids)

            step_lp = 0.0
            if out.scores:
                for step_idx, score in enumerate(out.scores):
                    if step_idx < len(new_ids):
                        lp = F.log_softmax(score.squeeze(0), dim=-1)
                        step_lp += lp[new_ids[step_idx]].item()
            continuation_log_probs_list.append(step_lp)

            if out.hidden_states and len(out.hidden_states) > 0:
                last_step_hidden = out.hidden_states[-1]
                last_layer = last_step_hidden[-1]
                h = last_layer[0, -1, :].float()
                h_norm = F.normalize(h, dim=-1)
                final_hidden_states.append(h_norm)

        pairwise_sims: list[float] = []
        if len(final_hidden_states) >= 2:
            hidden_stack = torch.stack(final_hidden_states, dim=0)
            for i in range(len(final_hidden_states)):
                for j in range(i + 1, len(final_hidden_states)):
                    sim = torch.dot(hidden_stack[i], hidden_stack[j]).item()
                    pairwise_sims.append(float(sim))

        mean_sim = sum(pairwise_sims) / len(pairwise_sims) if pairwise_sims else 1.0
        disagreement = float(max(0.0, min(1.0, 1.0 - mean_sim)))
        triggered = disagreement > self.threshold

        return ContinuationDisagreement(
            position=position,
            disagreement_score=disagreement,
            pairwise_similarities=pairwise_sims,
            continuation_token_ids=continuation_ids_list,
            continuation_log_probs=continuation_log_probs_list,
            triggered=triggered,
        )

    def should_expand(self, record: ContinuationDisagreement) -> bool:
        return record.triggered


def _clone_past_key_values(past_key_values):
    if past_key_values is None:
        return None
    return tuple(
        tuple(kv.clone() for kv in layer)
        for layer in past_key_values
    )