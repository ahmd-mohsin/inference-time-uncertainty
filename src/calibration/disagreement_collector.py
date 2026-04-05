import logging
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
from tqdm import tqdm
import jsonlines

from src.data.dataset import (
    get_calibration_dataset,
    format_prompt,
    extract_numeric_answer,
    answers_match,
)
from src.uncertainty.entropy_filter import compute_entropy
from src.uncertainty.semantic_zone import SemanticLoadZoneClassifier
from src.uncertainty.disagreement import (
    SemanticDisagreementDetector,
    _clone_past_key_values,
)

logger = logging.getLogger(__name__)


@dataclass
class DisagreementTokenRecord:
    problem_id: int
    position: int
    entropy: float
    disagreement_score: float
    in_semantic_zone: bool
    token_id: int
    token_str: str
    is_correct_step: bool
    final_answer_correct: bool
    entropy_triggered: bool


class DisagreementDataCollector:
    def __init__(
        self,
        model,
        tokenizer,
        zone_classifier: SemanticLoadZoneClassifier,
        detector: SemanticDisagreementDetector,
        tau_e: float,
        cfg: dict,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.zone_classifier = zone_classifier
        self.detector = detector
        self.tau_e = tau_e
        self.cfg = cfg
        self.device = cfg["model"]["device"]
        self.model_name = cfg["model"]["name"]
        self.max_new_tokens = cfg["model"]["max_new_tokens"]
        self.min_tokens = cfg.get("decoding", {}).get("min_tokens_before_trigger", 20)

    @torch.no_grad()
    def collect(self, save_path: str) -> list[DisagreementTokenRecord]:
        logger.info("Phase 1c — Collecting disagreement data at entropy-triggered positions")
        problems = get_calibration_dataset(self.cfg)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        all_records: list[DisagreementTokenRecord] = []

        with jsonlines.open(save_path, mode="w") as writer:
            for problem in tqdm(problems, desc="Collecting disagreement data"):
                try:
                    records = self._collect_problem(problem)
                    all_records.extend(records)
                    writer.write_all([asdict(r) for r in records])
                except Exception as e:
                    logger.warning(f"Skipping problem {problem['problem_id']}: {e}")

        logger.info(
            f"Collected {len(all_records)} disagreement records "
            f"from {len(problems)} problems"
        )
        logger.info(f"Saved to {save_path}")
        return all_records

    @torch.no_grad()
    def _collect_problem(self, problem: dict) -> list[DisagreementTokenRecord]:
        prompt = format_prompt(problem, self.model_name)
        prompt_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )["input_ids"].to(self.device)

        gold = problem["gold_answer"]
        eos_id = self.tokenizer.eos_token_id

        current_ids = prompt_ids.clone()
        past = None
        generated_ids: list[int] = []
        records: list[DisagreementTokenRecord] = []

        for pos in range(self.max_new_tokens):
            out = self.model(
                input_ids=current_ids if past is None else current_ids[:, -1:],
                past_key_values=past,
                use_cache=True,
            )
            past = out.past_key_values
            logits = out.logits[:, -1, :]
            next_id = logits.argmax(dim=-1).item()

            token_str = self.tokenizer.decode([next_id], skip_special_tokens=False)
            decoded_so_far = self.tokenizer.decode(
                current_ids[0], skip_special_tokens=True
            )
            in_zone = self.zone_classifier.is_in_zone(decoded_so_far, token_str)
            entropy = compute_entropy(logits.squeeze()).item()

            entropy_triggered = (
                pos >= self.min_tokens
                and in_zone
                and entropy > self.tau_e
            )

            disagreement_score = 0.0
            if entropy_triggered:
                try:
                    drec = self.detector.compute(
                        model=self.model,
                        input_ids=current_ids,
                        past_key_values=_clone_past_key_values(past),
                        position=pos,
                    )
                    disagreement_score = drec.disagreement_score
                except Exception as e:
                    logger.debug(f"Disagreement failed at pos {pos}: {e}")

            generated_ids.append(next_id)
            partial_text = self.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )
            pred = extract_numeric_answer(partial_text)
            is_correct_step = pred is not None and answers_match(pred, gold)

            records.append(
                DisagreementTokenRecord(
                    problem_id=problem["problem_id"],
                    position=pos,
                    entropy=entropy,
                    disagreement_score=disagreement_score,
                    in_semantic_zone=in_zone,
                    token_id=next_id,
                    token_str=token_str,
                    is_correct_step=is_correct_step,
                    final_answer_correct=False,
                    entropy_triggered=entropy_triggered,
                )
            )

            current_ids = torch.cat(
                [current_ids, torch.tensor([[next_id]], device=self.device)],
                dim=1,
            )
            if next_id == eos_id:
                break

        if records:
            final_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            final_pred = extract_numeric_answer(final_text)
            final_correct = answers_match(final_pred, gold)
            for r in records:
                r.final_answer_correct = final_correct

        return records