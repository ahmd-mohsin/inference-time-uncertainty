import logging
import time
from dataclasses import dataclass

import torch

from src.uncertainty.semantic_zone import SemanticLoadZoneClassifier
from src.uncertainty.entropy_filter import compute_entropy
from src.uncertainty.disagreement import SemanticDisagreementDetector, _clone_past_key_values

logger = logging.getLogger(__name__)

DELIBERATION_MARKER = "[VERIFY: resolve the following step before continuing]"


@dataclass
class PromptOnlyResult:
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


class PromptOnlyGenerator:
    def __init__(
        self,
        model,
        tokenizer,
        zone_classifier: SemanticLoadZoneClassifier,
        disagreement_detector: SemanticDisagreementDetector,
        tau_e: float,
        tau_d: float,
        cfg: dict,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.zone_classifier = zone_classifier
        self.detector = disagreement_detector
        self.tau_e = tau_e
        self.tau_d = tau_d
        self.max_new_tokens = cfg["model"]["max_new_tokens"]
        self.eos_token_id = tokenizer.eos_token_id
        self.device = cfg["model"]["device"]
        self.min_tokens = cfg.get("decoding", {}).get("min_tokens_before_trigger", 20)

        dec_cfg = cfg.get("decoding", {})
        self.expansion_marker = dec_cfg.get("expansion_marker", DELIBERATION_MARKER)
        self.marker_ids = tokenizer.encode(self.expansion_marker, add_special_tokens=False)

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor) -> PromptOnlyResult:
        t_start = time.time()
        current_ids = prompt_ids.clone()
        past = None
        prompt_len = prompt_ids.shape[1]
        generated_ids: list[int] = []
        n_injections = 0
        tokens_used = 0

        while tokens_used < self.max_new_tokens:
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

            if tokens_used >= self.min_tokens and in_zone and entropy > self.tau_e:
                drec = self.detector.compute(
                    model=self.model,
                    input_ids=current_ids,
                    past_key_values=_clone_past_key_values(past),
                    position=tokens_used,
                )
                if drec.disagreement_score > self.tau_d:
                    n_injections += 1
                    marker_tensor = torch.tensor([self.marker_ids], device=self.device)
                    out_m = self.model(
                        input_ids=marker_tensor,
                        past_key_values=past,
                        use_cache=True,
                    )
                    past = out_m.past_key_values
                    current_ids = torch.cat([current_ids, marker_tensor], dim=1)
                    generated_ids.extend(self.marker_ids)
                    tokens_used += len(self.marker_ids)

                    if tokens_used >= self.max_new_tokens:
                        break

                    out = self.model(
                        input_ids=current_ids[:, -1:],
                        past_key_values=past,
                        use_cache=True,
                    )
                    past = out.past_key_values
                    greedy_id = out.logits[:, -1, :].argmax(dim=-1).item()

            current_ids = torch.cat(
                [current_ids, torch.tensor([[greedy_id]], device=self.device)], dim=1
            )
            generated_ids.append(greedy_id)
            tokens_used += 1

            if greedy_id == self.eos_token_id:
                break

        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        total = len(generated_ids)

        return PromptOnlyResult(
            generated_ids=generated_ids,
            generated_text=generated_text,
            prompt_len=prompt_len,
            total_tokens=total,
            n_entropy_triggers=n_injections,
            n_expansion_triggers=n_injections,
            total_expansion_tokens=0,
            trigger_rate_entropy=n_injections / max(1, total),
            trigger_rate_expansion=n_injections / max(1, total),
            wall_time_sec=time.time() - t_start,
        )