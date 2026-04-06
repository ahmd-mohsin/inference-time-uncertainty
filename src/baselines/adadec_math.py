import logging
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.uncertainty.semantic_zone import SemanticLoadZoneClassifier
from src.uncertainty.entropy_filter import compute_entropy
from src.uncertainty.disagreement import _clone_past_key_values

logger = logging.getLogger(__name__)


@dataclass
class AdaDecResult:
    generated_ids: list[int]
    generated_text: str
    prompt_len: int
    total_tokens: int
    n_entropy_triggers: int
    n_expansion_triggers: int
    total_expansion_tokens: int
    trigger_rate_entropy: float
    trigger_rate_expansion: float
    wall_time_sec: float


class AdaDecGenerator:
    def __init__(
        self,
        model,
        tokenizer,
        zone_classifier: SemanticLoadZoneClassifier,
        tau_e: float,
        cfg: dict,
        lookahead_length: int = 5,
        lookahead_beam_size: int = 3,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.zone_classifier = zone_classifier
        self.tau_e = tau_e
        self.lookahead_length = lookahead_length
        self.lookahead_beam_size = lookahead_beam_size
        self.max_new_tokens = cfg["model"]["max_new_tokens"]
        self.eos_token_id = tokenizer.eos_token_id
        self.device = cfg["model"]["device"]
        self.min_tokens = cfg.get("decoding", {}).get("min_tokens_before_trigger", 20)

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor) -> AdaDecResult:
        t_start = time.time()
        current_ids = prompt_ids.clone()
        past = None
        prompt_len = prompt_ids.shape[1]
        generated_ids: list[int] = []
        n_pauses = 0

        for step in range(self.max_new_tokens):
            out = self.model(
                input_ids=current_ids if past is None else current_ids[:, -1:],
                past_key_values=past,
                use_cache=True,
            )
            past = out.past_key_values
            logits = out.logits[:, -1, :]
            greedy_id = logits.argmax(dim=-1).item()
            greedy_str = self.tokenizer.decode([greedy_id], skip_special_tokens=False)
            decoded = self.tokenizer.decode(current_ids[0], skip_special_tokens=True)
            in_zone = self.zone_classifier.is_in_zone(decoded, greedy_str)
            entropy = compute_entropy(logits.squeeze()).item()

            if step >= self.min_tokens and in_zone and entropy > self.tau_e:
                n_pauses += 1
                chosen_id = self._lookahead_rerank(current_ids, past, logits)
            else:
                chosen_id = greedy_id

            current_ids = torch.cat(
                [current_ids, torch.tensor([[chosen_id]], device=self.device)], dim=1
            )
            generated_ids.append(chosen_id)
            if chosen_id == self.eos_token_id:
                break

        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        total = len(generated_ids)

        return AdaDecResult(
            generated_ids=generated_ids,
            generated_text=generated_text,
            prompt_len=prompt_len,
            total_tokens=total,
            n_entropy_triggers=n_pauses,
            n_expansion_triggers=0,
            total_expansion_tokens=0,
            trigger_rate_entropy=n_pauses / max(1, total),
            trigger_rate_expansion=0.0,
            wall_time_sec=time.time() - t_start,
        )

    @torch.no_grad()
    def _lookahead_rerank(
        self,
        current_ids: torch.Tensor,
        past,
        logits: torch.Tensor,
    ) -> int:
        _, top_ids = F.softmax(logits, dim=-1).topk(self.lookahead_beam_size, dim=-1)
        candidates = top_ids[0].tolist()
        best_id = candidates[0]
        best_score = float("-inf")

        for cand_id in candidates:
            score = self._score_candidate(
                current_ids=current_ids,
                past=_clone_past_key_values(past),
                candidate_id=cand_id,
            )
            if score > best_score:
                best_score = score
                best_id = cand_id

        return best_id

    @torch.no_grad()
    def _score_candidate(
        self,
        current_ids: torch.Tensor,
        past,
        candidate_id: int,
    ) -> float:
        cand_tensor = torch.tensor([[candidate_id]], device=self.device)
        out = self.model(input_ids=cand_tensor, past_key_values=past, use_cache=True)
        cur_past = out.past_key_values
        cur_ids = torch.cat([current_ids, cand_tensor], dim=1)
        total_lp = 0.0

        for _ in range(self.lookahead_length):
            out = self.model(
                input_ids=cur_ids[:, -1:],
                past_key_values=cur_past,
                use_cache=True,
            )
            cur_past = out.past_key_values
            next_id = out.logits[:, -1, :].argmax(dim=-1).item()
            total_lp += F.log_softmax(out.logits[:, -1, :], dim=-1)[0, next_id].item()
            cur_ids = torch.cat(
                [cur_ids, torch.tensor([[next_id]], device=self.device)], dim=1
            )
            if next_id == self.eos_token_id:
                break

        return total_lp