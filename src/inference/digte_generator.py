import logging
import time
from dataclasses import dataclass, field

import torch

from src.uncertainty.entropy_filter import EntropyPreFilter, compute_entropy
from src.uncertainty.semantic_zone import SemanticLoadZoneClassifier
from src.uncertainty.disagreement import SemanticDisagreementDetector, _clone_past_key_values

logger = logging.getLogger(__name__)

DELIBERATION_MARKER = "[VERIFY: resolve the following step before continuing]"


@dataclass
class StepTrace:
    position: int
    token_id: int
    token_str: str
    entropy: float
    in_semantic_zone: bool
    entropy_triggered: bool
    disagreement_score: float
    expansion_triggered: bool
    expansion_tokens: int
    latency_ms: float


@dataclass
class GenerationResult:
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
    trace: list[StepTrace] = field(default_factory=list)


class DIGTEGenerator:
    def __init__(
        self,
        model,
        tokenizer,
        zone_classifier: SemanticLoadZoneClassifier,
        entropy_filter: EntropyPreFilter,
        disagreement_detector: SemanticDisagreementDetector,
        cfg: dict,
        log_detail: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.zone_classifier = zone_classifier
        self.entropy_filter = entropy_filter
        self.disagreement_detector = disagreement_detector
        self.log_detail = log_detail

        dec_cfg = cfg.get("decoding", {})
        self.max_new_tokens = cfg["model"]["max_new_tokens"]
        self.expansion_delta_l = dec_cfg.get("expansion_delta_l", 50)
        self.expansion_marker = dec_cfg.get("expansion_marker", DELIBERATION_MARKER)
        self.marker_ids = tokenizer.encode(self.expansion_marker, add_special_tokens=False)
        self.eos_token_id = tokenizer.eos_token_id
        self.device = cfg["model"]["device"]

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor) -> GenerationResult:
        t_start = time.time()
        current_ids = prompt_ids.clone()
        past = None
        prompt_len = prompt_ids.shape[1]
        generated_ids: list[int] = []
        trace: list[StepTrace] = []
        n_entropy_triggers = 0
        n_expansion_triggers = 0
        total_expansion_tokens = 0
        tokens_used = 0
        remaining = self.max_new_tokens

        while tokens_used < self.max_new_tokens:
            step_t = time.time()

            out = self.model(
                input_ids=current_ids if past is None else current_ids[:, -1:],
                past_key_values=past,
                use_cache=True,
            )
            past = out.past_key_values
            logits = out.logits[:, -1, :]
            greedy_id = logits.argmax(dim=-1).item()
            greedy_str = self.tokenizer.decode([greedy_id], skip_special_tokens=False)

            decoded_so_far = self.tokenizer.decode(current_ids[0], skip_special_tokens=True)
            in_zone = self.zone_classifier.is_in_zone(decoded_so_far, greedy_str)
            entropy_triggered, entropy_val = self.entropy_filter.should_trigger(
                logits=logits, position=tokens_used, in_semantic_zone=in_zone,
            )

            disagreement_score = 0.0
            expansion_triggered = False
            expansion_tokens_added = 0

            if entropy_triggered:
                n_entropy_triggers += 1
                drec = self.disagreement_detector.compute(
                    model=self.model,
                    input_ids=current_ids,
                    past_key_values=_clone_past_key_values(past),
                    position=tokens_used,
                )
                disagreement_score = drec.disagreement_score

                if self.disagreement_detector.should_expand(drec):
                    n_expansion_triggers += 1
                    expansion_triggered = True
                    current_ids, past, expansion_ids = self._expand_in_place(
                        current_ids=current_ids,
                        past=past,
                        remaining=remaining,
                    )
                    generated_ids.extend(expansion_ids)
                    expansion_tokens_added = len(expansion_ids)
                    total_expansion_tokens += expansion_tokens_added
                    tokens_used += expansion_tokens_added
                    remaining -= expansion_tokens_added

                    if any(t == self.eos_token_id for t in expansion_ids) or remaining <= 0:
                        break

                    out2 = self.model(
                        input_ids=current_ids[:, -1:],
                        past_key_values=past,
                        use_cache=True,
                    )
                    past = out2.past_key_values
                    logits = out2.logits[:, -1, :]
                    greedy_id = logits.argmax(dim=-1).item()
                    greedy_str = self.tokenizer.decode([greedy_id], skip_special_tokens=False)

            next_tensor = torch.tensor([[greedy_id]], device=self.device)
            current_ids = torch.cat([current_ids, next_tensor], dim=1)
            generated_ids.append(greedy_id)
            tokens_used += 1
            remaining -= 1

            if self.log_detail:
                trace.append(StepTrace(
                    position=tokens_used - 1,
                    token_id=greedy_id,
                    token_str=greedy_str,
                    entropy=entropy_val,
                    in_semantic_zone=in_zone,
                    entropy_triggered=entropy_triggered,
                    disagreement_score=disagreement_score,
                    expansion_triggered=expansion_triggered,
                    expansion_tokens=expansion_tokens_added,
                    latency_ms=(time.time() - step_t) * 1000,
                ))

            if greedy_id == self.eos_token_id or remaining <= 0:
                break

        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        total = len(generated_ids)

        return GenerationResult(
            generated_ids=generated_ids,
            generated_text=generated_text,
            prompt_len=prompt_len,
            total_tokens=total,
            n_entropy_triggers=n_entropy_triggers,
            n_expansion_triggers=n_expansion_triggers,
            total_expansion_tokens=total_expansion_tokens,
            trigger_rate_entropy=n_entropy_triggers / max(1, total),
            trigger_rate_expansion=n_expansion_triggers / max(1, total),
            wall_time_sec=time.time() - t_start,
            trace=trace if self.log_detail else [],
        )

    @torch.no_grad()
    def _expand_in_place(
        self,
        current_ids: torch.Tensor,
        past,
        remaining: int,
    ) -> tuple[torch.Tensor, object, list[int]]:
        marker_tensor = torch.tensor([self.marker_ids], device=self.device)
        out = self.model(input_ids=marker_tensor, past_key_values=past, use_cache=True)
        past = out.past_key_values
        current_ids = torch.cat([current_ids, marker_tensor], dim=1)

        expansion_ids = list(self.marker_ids)
        delta = min(self.expansion_delta_l, remaining - len(self.marker_ids))

        for _ in range(max(0, delta)):
            out = self.model(
                input_ids=current_ids[:, -1:],
                past_key_values=past,
                use_cache=True,
            )
            past = out.past_key_values
            next_id = out.logits[:, -1, :].argmax(dim=-1).item()
            current_ids = torch.cat(
                [current_ids, torch.tensor([[next_id]], device=self.device)], dim=1
            )
            expansion_ids.append(next_id)
            if next_id == self.eos_token_id:
                break

        return current_ids, past, expansion_ids