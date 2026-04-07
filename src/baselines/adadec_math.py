import logging
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.uncertainty.semantic_zone import SemanticLoadZoneClassifier
from src.uncertainty.entropy_filter import compute_entropy

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
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        self.device = cfg["model"]["device"]
        self.min_tokens = cfg.get("decoding", {}).get("min_tokens_before_trigger", 20)

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor) -> AdaDecResult:
        t_start = time.time()

        base_out = self.model.generate(
            input_ids=prompt_ids,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
        )

        prompt_len = prompt_ids.shape[1]
        base_ids = base_out.sequences[0, prompt_len:].tolist()
        scores = base_out.scores

        generated_ids: list[int] = []
        n_pauses = 0
        current_seq = prompt_ids.clone()
        decoded_so_far = self.tokenizer.decode(prompt_ids[0], skip_special_tokens=True)

        for pos, (token_id, logits_t) in enumerate(zip(base_ids, scores)):
            logits_clean = torch.nan_to_num(
                logits_t.squeeze(0), nan=0.0, posinf=1e4, neginf=-1e4
            )
            token_str = self.tokenizer.decode([token_id], skip_special_tokens=False)
            in_zone = self.zone_classifier.is_in_zone(decoded_so_far, token_str)
            entropy = compute_entropy(logits_clean).item()

            if pos >= self.min_tokens and in_zone and entropy > self.tau_e:
                n_pauses += 1
                chosen_id = self._lookahead_rerank(current_seq, logits_clean)
            else:
                chosen_id = token_id

            generated_ids.append(chosen_id)
            chosen_str = self.tokenizer.decode([chosen_id], skip_special_tokens=False)
            decoded_so_far += chosen_str
            current_seq = torch.cat(
                [current_seq, torch.tensor([[chosen_id]], device=self.device)], dim=1
            )
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
    def _lookahead_rerank(self, current_seq: torch.Tensor, logits: torch.Tensor) -> int:
        _, top_ids = F.softmax(logits, dim=-1).topk(self.lookahead_beam_size)
        candidates = top_ids.tolist()
        best_id = candidates[0]
        best_score = float("-inf")

        for cand_id in candidates:
            cand_input = torch.cat(
                [current_seq, torch.tensor([[cand_id]], device=self.device)], dim=1
            )
            out = self.model.generate(
                input_ids=cand_input,
                max_new_tokens=self.lookahead_length,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=self.pad_token_id,
                eos_token_id=self.eos_token_id,
            )
            new_ids = out.sequences[0, cand_input.shape[1]:].tolist()
            total_lp = sum(
                F.log_softmax(s.squeeze(0), dim=-1)[tid].item()
                for s, tid in zip(out.scores, new_ids)
            )
            if total_lp > best_score:
                best_score = total_lp
                best_id = cand_id

        return best_id