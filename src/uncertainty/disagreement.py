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


def _clone_past_key_values(past_key_values):
    if past_key_values is None:
        return None
    return tuple(
        tuple(kv.clone() for kv in layer)
        for layer in past_key_values
    )


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
        device = input_ids.device

        continuation_ids_list: list[list[int]] = []
        continuation_log_probs_list: list[float] = []
        final_hidden_states: list[torch.Tensor] = []

        for _ in range(self.k):
            current_ids = input_ids.clone()
            current_past = _clone_past_key_values(past_key_values)
            step_log_prob = 0.0
            last_hidden: Optional[torch.Tensor] = None

            for step in range(self.L):
                out = model(
                    input_ids=current_ids[:, -1:],
                    past_key_values=current_past,
                    use_cache=True,
                    output_hidden_states=True,
                )
                logits = out.logits[:, -1, :]
                current_past = out.past_key_values

                last_hidden = out.hidden_states[-1][:, -1, :]

                scaled_logits = logits / max(self.temperature, 1e-6)
                probs = F.softmax(scaled_logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)

                log_prob = F.log_softmax(scaled_logits, dim=-1)
                step_log_prob += log_prob[0, next_token_id.item()].item()

                current_ids = torch.cat([current_ids, next_token_id], dim=1)

            sampled_ids = current_ids[0, input_ids.shape[1]:].tolist()
            continuation_ids_list.append(sampled_ids)
            continuation_log_probs_list.append(step_log_prob)

            if last_hidden is not None:
                h_norm = F.normalize(last_hidden.squeeze(0).float(), dim=-1)
                final_hidden_states.append(h_norm)

        pairwise_sims: list[float] = []
        if len(final_hidden_states) >= 2:
            hidden_stack = torch.stack(final_hidden_states, dim=0)
            for i in range(self.k):
                for j in range(i + 1, self.k):
                    if i < hidden_stack.shape[0] and j < hidden_stack.shape[0]:
                        sim = torch.dot(hidden_stack[i], hidden_stack[j]).item()
                        pairwise_sims.append(float(sim))

        if pairwise_sims:
            mean_sim = sum(pairwise_sims) / len(pairwise_sims)
        else:
            mean_sim = 1.0

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