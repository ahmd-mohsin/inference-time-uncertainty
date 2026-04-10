import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np

from src.search.rmi_tree_search import RMITreeResult

logger = logging.getLogger(__name__)

class SamplingVoteGenerator:

    def __init__(self, model, tokenizer, prm, cfg: dict):
        self.model = model
        self.tokenizer = tokenizer
        self.prm = prm

        self.device = cfg["model"]["device"]
        self.max_new_tokens = cfg["model"].get("max_new_tokens", 8192)
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

        search_cfg = cfg.get("search", {})
        self.n_solutions = search_cfg.get("n_solutions", 32)
        self.temperature = search_cfg.get("sampling_temperature", 0.7)
        self.top_p = search_cfg.get("top_p", 0.95)
        self.aggregation = search_cfg.get("aggregation", "weighted_vote")

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor, problem_text: str = "") -> RMITreeResult:
        t_start = time.time()
        prompt_len = prompt_ids.shape[1]

        solutions = []
        total_tokens = 0

        for k in range(self.n_solutions):
            out = self.model.generate(
                input_ids=prompt_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self.pad_token_id,
                eos_token_id=self.eos_token_id,
            )

            gen_ids = out[0, prompt_len:].tolist()
            gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            total_tokens += len(gen_ids)

            answer = self._extract_answer(gen_text)

            reward = self.prm.score_step(
                problem_text=problem_text,
                solution_so_far="",
                current_step=gen_text,
            )

            solutions.append({
                "node_id": k,
                "text": gen_text,
                "answer": answer,
                "prm_reward": reward,
                "diversity_score": 0.0,
                "depth": 1,
                "n_tokens": len(gen_ids),
            })

        if self.aggregation == "weighted_vote":
            best = self._weighted_majority_vote(solutions)
            method = "weighted_vote"
        elif self.aggregation == "majority_vote":
            best = self._majority_vote(solutions)
            method = "majority_vote"
        else:
            best = max(solutions, key=lambda s: s["prm_reward"])
            method = "best_reward"

        rewards = [s["prm_reward"] for s in solutions]

        return RMITreeResult(
            generated_text=best["text"],
            generated_ids=[],
            extracted_answer=best["answer"],
            all_solutions=solutions,
            prompt_len=prompt_len,
            total_tokens=total_tokens,
            wall_time_sec=time.time() - t_start,
            n_completed_solutions=len(solutions),
            n_prm_calls=len(solutions),
            tree_max_depth=1,
            mean_prm_reward=float(np.mean(rewards)),
            mean_diversity_score=0.0,
            selected_method=method,
            n_entropy_triggers=0,
            n_expansion_triggers=0,
            total_expansion_tokens=0,
            trigger_rate_entropy=0.0,
            trigger_rate_expansion=0.0,
        )

    def _weighted_majority_vote(self, solutions: list[dict]) -> dict:
        answer_scores: dict[str, float] = {}
        answer_best: dict[str, dict] = {}
        for sol in solutions:
            a = self._normalize(sol["answer"])
            if not a:
                continue
            answer_scores[a] = answer_scores.get(a, 0.0) + sol["prm_reward"]
            if a not in answer_best or sol["prm_reward"] > answer_best[a]["prm_reward"]:
                answer_best[a] = sol
        if not answer_scores:
            return max(solutions, key=lambda s: s["prm_reward"])
        best_answer = max(answer_scores, key=answer_scores.get)
        return answer_best[best_answer]

    def _majority_vote(self, solutions: list[dict]) -> dict:
        answer_counts: dict[str, int] = {}
        answer_best: dict[str, dict] = {}
        for sol in solutions:
            a = self._normalize(sol["answer"])
            if not a:
                continue
            answer_counts[a] = answer_counts.get(a, 0) + 1
            if a not in answer_best or sol["prm_reward"] > answer_best[a]["prm_reward"]:
                answer_best[a] = sol
        if not answer_counts:
            return max(solutions, key=lambda s: s["prm_reward"])
        best_answer = max(answer_counts, key=answer_counts.get)
        return answer_best[best_answer]

    def _extract_answer(self, text: str) -> Optional[str]:
        from src.data.dataset import extract_boxed_answer, extract_numeric_answer
        answer = extract_boxed_answer(text)
        if answer:
            return answer
        return extract_numeric_answer(text)

    def _normalize(self, answer: Optional[str]) -> str:
        if answer is None:
            return ""
        from src.data.dataset import normalize_answer
        return normalize_answer(answer)