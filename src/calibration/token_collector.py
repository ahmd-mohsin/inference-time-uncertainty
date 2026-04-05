import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

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
from src.uncertainty.entropy_filter import (
    compute_entropy,
    compute_logit_margin,
    compute_top_probs,
)
from src.uncertainty.semantic_zone import SemanticLoadZoneClassifier

logger = logging.getLogger(__name__)


@dataclass
class TokenRecord:
    problem_id: int
    position: int
    entropy: float
    logit_margin: float
    top1_prob: float
    top2_prob: float
    in_semantic_zone: bool
    token_id: int
    token_str: str
    is_correct_step: bool
    final_answer_correct: bool


class TokenDataCollector:
    def __init__(
        self,
        model,
        tokenizer,
        zone_classifier: SemanticLoadZoneClassifier,
        cfg: dict,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.zone_classifier = zone_classifier
        self.cfg = cfg
        self.device = cfg["model"]["device"]
        self.model_name = cfg["model"]["name"]
        self.max_new_tokens = cfg["model"]["max_new_tokens"]

    @torch.no_grad()
    def collect(self, save_path: str) -> list[TokenRecord]:
        logger.info("Phase 1a — Collecting token data on calibration dataset")
        problems = get_calibration_dataset(self.cfg)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        all_records: list[TokenRecord] = []

        with jsonlines.open(save_path, mode="w") as writer:
            for problem in tqdm(problems, desc="Collecting token data"):
                try:
                    records = self._collect_problem(problem)
                    all_records.extend(records)
                    writer.write_all([asdict(r) for r in records])
                except Exception as e:
                    logger.warning(f"Skipping problem {problem['problem_id']}: {e}")

        logger.info(
            f"Collected {len(all_records)} token records from {len(problems)} problems"
        )
        logger.info(f"Saved to {save_path}")
        return all_records

    @torch.no_grad()
    def _collect_problem(self, problem: dict) -> list[TokenRecord]:
        prompt = format_prompt(problem, self.model_name)
        prompt_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )["input_ids"].to(self.device)

        prompt_len = prompt_ids.shape[1]
        generated_ids: list[int] = []
        all_logits: list[torch.Tensor] = []

        current_ids = prompt_ids.clone()
        past = None
        eos_id = self.tokenizer.eos_token_id

        for _ in range(self.max_new_tokens):
            out = self.model(
                input_ids=current_ids if past is None else current_ids[:, -1:],
                past_key_values=past,
                use_cache=True,
            )
            past = out.past_key_values
            logits_t = out.logits[:, -1, :].squeeze(0).clone()
            all_logits.append(logits_t)

            next_id = logits_t.argmax(dim=-1).item()
            generated_ids.append(next_id)
            current_ids = torch.cat(
                [current_ids, torch.tensor([[next_id]], device=self.device)],
                dim=1,
            )
            if next_id == eos_id:
                break

        full_generated_text = self.tokenizer.decode(
            generated_ids, skip_special_tokens=True
        )
        pred_answer = extract_numeric_answer(full_generated_text)
        final_correct = answers_match(pred_answer, problem["gold_answer"])

        step_labels = self._label_steps(
            generated_ids=generated_ids,
            gold=problem["gold_answer"],
            final_correct=final_correct,
        )

        records: list[TokenRecord] = []
        decoded_so_far = self.tokenizer.decode(
            prompt_ids[0], skip_special_tokens=True
        )

        for pos, (token_id, logits_t, is_correct) in enumerate(
            zip(generated_ids, all_logits, step_labels)
        ):
            token_str = self.tokenizer.decode([token_id], skip_special_tokens=False)
            in_zone = self.zone_classifier.is_in_zone(decoded_so_far, token_str)

            entropy = compute_entropy(logits_t).item()
            margin = compute_logit_margin(logits_t).item()
            top_probs = compute_top_probs(logits_t, k=2)

            records.append(
                TokenRecord(
                    problem_id=problem["problem_id"],
                    position=pos,
                    entropy=entropy,
                    logit_margin=margin,
                    top1_prob=top_probs[0].item(),
                    top2_prob=top_probs[1].item() if top_probs.shape[-1] > 1 else 0.0,
                    in_semantic_zone=in_zone,
                    token_id=token_id,
                    token_str=token_str,
                    is_correct_step=is_correct,
                    final_answer_correct=final_correct,
                )
            )
            decoded_so_far += token_str

        return records

    def _label_steps(
        self,
        generated_ids: list[int],
        gold: str,
        final_correct: bool,
    ) -> list[bool]:
        n = len(generated_ids)
        if not final_correct:
            return [False] * n

        labels = [True] * n
        partial_text = ""
        for pos, token_id in enumerate(generated_ids):
            token_str = self.tokenizer.decode([token_id], skip_special_tokens=False)
            partial_text += token_str
            pred = extract_numeric_answer(partial_text)
            if pred is not None and answers_match(pred, gold):
                for i in range(pos):
                    labels[i] = False
                return labels

        return [True] * n