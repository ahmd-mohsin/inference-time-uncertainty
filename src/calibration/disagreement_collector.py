import logging
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn.functional as F
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
from src.uncertainty.disagreement import SemanticDisagreementDetector

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
            f"Collected {len(all_records)} disagreement records from {len(problems)} problems"
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

        output = self.model.generate(
            input_ids=prompt_ids,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=1.0,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        generated_ids = output.sequences[0, prompt_ids.shape[1]:].tolist()
        scores = output.scores
        full_sequence = output.sequences

        if not generated_ids:
            return []

        full_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        pred_answer = extract_numeric_answer(full_text)
        final_correct = answers_match(pred_answer, problem["gold_answer"])

        records: list[DisagreementTokenRecord] = []
        decoded_so_far = self.tokenizer.decode(prompt_ids[0], skip_special_tokens=True)
        prompt_len = prompt_ids.shape[1]

        for pos, (token_id, logits_t) in enumerate(zip(generated_ids, scores)):
            logits_clean = torch.nan_to_num(
                logits_t.squeeze(0), nan=0.0, posinf=1e4, neginf=-1e4
            )
            token_str = self.tokenizer.decode([token_id], skip_special_tokens=False)
            in_zone = self.zone_classifier.is_in_zone(decoded_so_far, token_str)
            entropy = compute_entropy(logits_clean).item()

            entropy_triggered = (
                pos >= self.min_tokens
                and in_zone
                and entropy > self.tau_e
            )

            disagreement_score = 0.0
            if entropy_triggered:
                prefix_ids = full_sequence[:, :prompt_len + pos]
                try:
                    drec = self.detector.compute(
                        model=self.model,
                        input_ids=prefix_ids,
                        past_key_values=None,
                        position=pos,
                    )
                    disagreement_score = drec.disagreement_score
                except Exception as e:
                    logger.debug(f"Disagreement failed at pos {pos}: {e}")

            records.append(DisagreementTokenRecord(
                problem_id=problem["problem_id"],
                position=pos,
                entropy=entropy,
                disagreement_score=disagreement_score,
                in_semantic_zone=in_zone,
                token_id=token_id,
                token_str=token_str,
                is_correct_step=final_correct,
                final_answer_correct=final_correct,
                entropy_triggered=entropy_triggered,
            ))
            decoded_so_far += token_str

        return records