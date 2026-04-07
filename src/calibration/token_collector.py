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
        n_correct = 0
        n_incorrect = 0

        with jsonlines.open(save_path, mode="w") as writer:
            for problem in tqdm(problems, desc="Collecting token data"):
                try:
                    records = self._collect_problem(problem)
                    all_records.extend(records)
                    writer.write_all([asdict(r) for r in records])
                    if records:
                        if records[0].final_answer_correct:
                            n_correct += 1
                        else:
                            n_incorrect += 1
                except Exception as e:
                    logger.warning(f"Skipping problem {problem['problem_id']}: {e}")

        logger.info(
            f"Collected {len(all_records)} token records from {len(problems)} problems "
            f"(correct={n_correct}, incorrect={n_incorrect})"
        )
        if n_correct == 0 or n_incorrect == 0:
            logger.warning(
                "All problems have the same final_answer_correct value — "
                "label variance is zero. Consider using a harder dataset or weaker model."
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

        if not generated_ids:
            return []

        full_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        pred_answer = extract_numeric_answer(full_text)
        final_correct = answers_match(pred_answer, problem["gold_answer"])

        records: list[TokenRecord] = []
        decoded_so_far = self.tokenizer.decode(prompt_ids[0], skip_special_tokens=True)

        for pos, (token_id, logits_t) in enumerate(zip(generated_ids, scores)):
            logits_clean = torch.nan_to_num(
                logits_t.squeeze(0), nan=0.0, posinf=1e4, neginf=-1e4
            )
            token_str = self.tokenizer.decode([token_id], skip_special_tokens=False)
            in_zone = self.zone_classifier.is_in_zone(decoded_so_far, token_str)

            entropy = compute_entropy(logits_clean).item()
            margin = compute_logit_margin(logits_clean).item()
            top_probs = compute_top_probs(logits_clean, k=2)

            records.append(TokenRecord(
                problem_id=problem["problem_id"],
                position=pos,
                entropy=entropy,
                logit_margin=margin,
                top1_prob=top_probs[0].item(),
                top2_prob=top_probs[1].item() if top_probs.shape[-1] > 1 else 0.0,
                in_semantic_zone=in_zone,
                token_id=token_id,
                token_str=token_str,
                is_correct_step=final_correct,
                final_answer_correct=final_correct,
            ))
            decoded_so_far += token_str

        return records