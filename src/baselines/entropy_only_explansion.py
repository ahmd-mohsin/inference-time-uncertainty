import logging
import time
from dataclasses import dataclass

import torch

from src.uncertainty.semantic_zone import SemanticLoadZoneClassifier
from src.uncertainty.entropy_filter import compute_entropy

logger = logging.getLogger(__name__)

DELIBERATION_MARKER = "[VERIFY: resolve the following step before continuing]"


@dataclass
class EntropyOnlyResult:
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


class EntropyOnlyExpansionGenerator:
    def __init__(
        self,
        model,
        tokenizer,
        zone_classifier: SemanticLoadZoneClassifier,
        tau_e: float,
        cfg: dict,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.zone_classifier = zone_classifier
        self.tau_e = tau_e
        self.max_new_tokens = cfg["model"]["max_new_tokens"]
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        self.device = cfg["model"]["device"]
        self.min_tokens = cfg.get("decoding", {}).get("min_tokens_before_trigger", 20)
        dec_cfg = cfg.get("decoding", {})
        self.expansion_delta_l = dec_cfg.get("expansion_delta_l", 50)
        self.expansion_marker = dec_cfg.get("expansion_marker", DELIBERATION_MARKER)
        self.marker_ids = tokenizer.encode(self.expansion_marker, add_special_tokens=False)

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor) -> EntropyOnlyResult:
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
        n_expansions = 0
        total_expansion_tokens = 0
        tokens_used = 0
        remaining = self.max_new_tokens
        current_seq = prompt_ids.clone()
        decoded_so_far = self.tokenizer.decode(prompt_ids[0], skip_special_tokens=True)

        for pos, (token_id, logits_t) in enumerate(zip(base_ids, scores)):
            logits_clean = torch.nan_to_num(
                logits_t.squeeze(0), nan=0.0, posinf=1e4, neginf=-1e4
            )
            token_str = self.tokenizer.decode([token_id], skip_special_tokens=False)
            in_zone = self.zone_classifier.is_in_zone(decoded_so_far, token_str)
            entropy = compute_entropy(logits_clean).item()

            if tokens_used >= self.min_tokens and in_zone and entropy > self.tau_e and remaining > len(self.marker_ids) + 1:
                n_expansions += 1
                marker_tensor = torch.tensor([self.marker_ids], device=self.device)
                expand_input = torch.cat([current_seq, marker_tensor], dim=1)
                delta = min(self.expansion_delta_l, remaining - len(self.marker_ids))
                exp_out = self.model.generate(
                    input_ids=expand_input,
                    max_new_tokens=delta,
                    do_sample=False,
                    pad_token_id=self.pad_token_id,
                    eos_token_id=self.eos_token_id,
                )
                exp_ids = list(self.marker_ids) + exp_out[0, expand_input.shape[1]:].tolist()
                generated_ids.extend(exp_ids)
                total_expansion_tokens += len(exp_ids)
                tokens_used += len(exp_ids)
                remaining -= len(exp_ids)
                for eid in exp_ids:
                    etok = self.tokenizer.decode([eid], skip_special_tokens=False)
                    decoded_so_far += etok
                    current_seq = torch.cat(
                        [current_seq, torch.tensor([[eid]], device=self.device)], dim=1
                    )
                if any(t == self.eos_token_id for t in exp_ids) or remaining <= 0:
                    break

            generated_ids.append(token_id)
            decoded_so_far += token_str
            current_seq = torch.cat(
                [current_seq, torch.tensor([[token_id]], device=self.device)], dim=1
            )
            tokens_used += 1
            remaining -= 1
            if token_id == self.eos_token_id or remaining <= 0:
                break

        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        total = len(generated_ids)
        return EntropyOnlyResult(
            generated_ids=generated_ids, generated_text=generated_text,
            prompt_len=prompt_len, total_tokens=total,
            n_entropy_triggers=n_expansions, n_expansion_triggers=n_expansions,
            total_expansion_tokens=total_expansion_tokens,
            trigger_rate_entropy=n_expansions / max(1, total),
            trigger_rate_expansion=n_expansions / max(1, total),
            wall_time_sec=time.time() - t_start,
        )