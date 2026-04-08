import logging
import time
from dataclasses import dataclass, field

import torch

from src.uncertainty.entropy_filter import EntropyPreFilter, compute_entropy
from src.uncertainty.semantic_zone import SemanticLoadZoneClassifier
from src.uncertainty.disagreement import SemanticDisagreementDetector

logger = logging.getLogger(__name__)

_CONTINUATION_PROMPT = " Let me verify this step carefully and continue: "
_MAX_ENTROPY_TRIGGERS_PER_PROBLEM = 8


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


def _is_garbage_token(tok: str) -> bool:
    for c in tok:
        cp = ord(c)
        is_ascii_ext = cp < 0x0100
        is_greek = 0x0370 <= cp <= 0x03FF
        is_math_sym = 0x2200 <= cp <= 0x22FF
        is_arrows = 0x2190 <= cp <= 0x21FF
        if not (is_ascii_ext or is_greek or is_math_sym or is_arrows):
            return True
    return False


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
        self.expansion_delta_l = dec_cfg.get("expansion_delta_l", 30)
        self.continuation_temperature = dec_cfg.get("continuation_temperature", 0.3)
        self.max_triggers = dec_cfg.get("max_entropy_triggers_per_problem", _MAX_ENTROPY_TRIGGERS_PER_PROBLEM)
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        self.device = cfg["model"]["device"]

        self._continuation_ids = tokenizer.encode(
            _CONTINUATION_PROMPT, add_special_tokens=False
        )

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor) -> GenerationResult:
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
        trace: list[StepTrace] = []
        n_entropy_triggers = 0
        n_expansion_triggers = 0
        total_expansion_tokens = 0
        tokens_used = 0
        remaining = self.max_new_tokens

        current_seq = prompt_ids.clone()
        decoded_so_far = self.tokenizer.decode(prompt_ids[0], skip_special_tokens=True)

        for pos, (token_id, logits_t) in enumerate(zip(base_ids, scores)):
            step_t = time.time()
            logits_clean = torch.nan_to_num(
                logits_t.squeeze(0), nan=0.0, posinf=1e4, neginf=-1e4
            )
            token_str = self.tokenizer.decode([token_id], skip_special_tokens=False)
            in_zone = self.zone_classifier.is_in_zone(decoded_so_far, token_str)

            trigger_budget_remaining = n_entropy_triggers < self.max_triggers

            entropy_triggered, entropy_val = self.entropy_filter.should_trigger(
                logits=logits_clean.unsqueeze(0),
                position=tokens_used,
                in_semantic_zone=in_zone,
            )
            entropy_triggered = entropy_triggered and trigger_budget_remaining

            disagreement_score = 0.0
            expansion_triggered = False
            expansion_tokens_added = 0

            if entropy_triggered:
                n_entropy_triggers += 1
                budget_for_expansion = remaining - len(self._continuation_ids) - 1
                if budget_for_expansion > 5:
                    try:
                        drec = self.disagreement_detector.compute(
                            model=self.model,
                            input_ids=current_seq,
                            past_key_values=None,
                            position=tokens_used,
                        )
                        disagreement_score = drec.disagreement_score

                        if self.disagreement_detector.should_expand(drec):
                            n_expansion_triggers += 1
                            expansion_triggered = True

                            exp_ids = self._expand_from_current(
                                current_seq=current_seq,
                                remaining=budget_for_expansion,
                            )

                            generated_ids.extend(exp_ids)
                            expansion_tokens_added = len(exp_ids)
                            total_expansion_tokens += expansion_tokens_added
                            tokens_used += expansion_tokens_added
                            remaining -= expansion_tokens_added

                            for eid in exp_ids:
                                etok = self.tokenizer.decode(
                                    [eid], skip_special_tokens=False
                                )
                                decoded_so_far += etok
                                current_seq = torch.cat(
                                    [
                                        current_seq,
                                        torch.tensor([[eid]], device=self.device),
                                    ],
                                    dim=1,
                                )

                            if (
                                any(t == self.eos_token_id for t in exp_ids)
                                or remaining <= 0
                            ):
                                break

                    except Exception as e:
                        logger.debug(f"DIGTE expansion failed at pos {pos}: {e}")

            generated_ids.append(token_id)
            decoded_so_far += token_str
            current_seq = torch.cat(
                [current_seq, torch.tensor([[token_id]], device=self.device)],
                dim=1,
            )
            tokens_used += 1
            remaining -= 1

            if self.log_detail:
                trace.append(
                    StepTrace(
                        position=tokens_used - 1,
                        token_id=token_id,
                        token_str=token_str,
                        entropy=entropy_val,
                        in_semantic_zone=in_zone,
                        entropy_triggered=entropy_triggered,
                        disagreement_score=disagreement_score,
                        expansion_triggered=expansion_triggered,
                        expansion_tokens=expansion_tokens_added,
                        latency_ms=(time.time() - step_t) * 1000,
                    )
                )

            if token_id == self.eos_token_id or remaining <= 0:
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
    def _expand_from_current(
        self,
        current_seq: torch.Tensor,
        remaining: int,
    ) -> list[int]:
        cont_tensor = torch.tensor([self._continuation_ids], device=self.device)
        expand_input = torch.cat([current_seq, cont_tensor], dim=1)

        delta = min(self.expansion_delta_l, remaining - len(self._continuation_ids))
        if delta <= 0:
            return list(self._continuation_ids)

        temperature = self.continuation_temperature

        if temperature < 0.05:
            out = self.model.generate(
                input_ids=expand_input,
                max_new_tokens=delta,
                do_sample=False,
                pad_token_id=self.pad_token_id,
                eos_token_id=self.eos_token_id,
                repetition_penalty=1.05,
            )
        else:
            out = self.model.generate(
                input_ids=expand_input,
                max_new_tokens=delta,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                top_k=40,
                pad_token_id=self.pad_token_id,
                eos_token_id=self.eos_token_id,
                repetition_penalty=1.05,
            )

        new_ids = out[0, expand_input.shape[1]:].tolist()

        clean_ids = []
        for tid in new_ids:
            tok = self.tokenizer.decode([tid], skip_special_tokens=False)
            if _is_garbage_token(tok):
                logger.debug(f"Stopping expansion at garbage token: {repr(tok)}")
                break
            clean_ids.append(tid)

        return list(self._continuation_ids) + clean_ids