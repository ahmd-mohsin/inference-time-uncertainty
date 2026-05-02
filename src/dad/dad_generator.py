import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import numpy as np

from src.dad.claim_extractor import profile_solution, extract_boxed_answer
from src.dad.disagreement_analyzer import (
    build_disagreement_map, format_workspace, DisagreementMap,
)

logger = logging.getLogger(__name__)


@dataclass
class DADResult:
    generated_text: str
    extracted_answer: Optional[str]
    all_solutions: list[dict] = field(default_factory=list)
    workspace_text: str = ""
    disagreement_map: Optional[dict] = None
    n_rounds: int = 0
    n_total_generations: int = 0
    total_tokens: int = 0
    wall_time_sec: float = 0.0
    answer_entropy_per_round: list[float] = field(default_factory=list)
    confidence_per_round: list[float] = field(default_factory=list)
    selected_method: str = ""
    # NEW: per-round disagreement maps for characterization
    per_round_disagreement_maps: list[dict] = field(default_factory=list)


class DADGenerator:
    def __init__(self, model, tokenizer, cfg: dict):
        self.model = model
        self.tokenizer = tokenizer
        self.device = cfg["model"]["device"]

        dad_cfg = cfg.get("dad", {})
        self.m_samples = dad_cfg.get("m_samples", 8)
        self.max_rounds = dad_cfg.get("max_rounds", 3)
        self.max_gen_tokens = dad_cfg.get("max_gen_tokens", 2048)
        self.temperature = dad_cfg.get("temperature", 0.7)
        self.top_p = dad_cfg.get("top_p", 0.95)
        self.confidence_threshold = dad_cfg.get("confidence_threshold", 0.8)
        self.workspace_max_tokens = dad_cfg.get("workspace_max_tokens", 800)
        self.refine_samples = dad_cfg.get("refine_samples", 4)

        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

        if hasattr(model, 'config') and model.config.max_position_embeddings < 32768:
            model.config.max_position_embeddings = 32768

    def _dmap_to_dict(self, dmap) -> dict:
        """Convert a DisagreementMap to a serializable dict."""
        return {
            "n_solutions": dmap.n_solutions,
            "answer_distribution": dmap.answer_distribution,
            "answer_entropy": dmap.answer_entropy,
            "majority_answer": dmap.majority_answer,
            "majority_answer_fraction": dmap.majority_answer_fraction,
            "n_agreed": len(dmap.agreed_claims),
            "n_disputed": len(dmap.disputed_claims),
            "confidence": dmap.confidence_score,
        }

    def generate(self, prompt_ids: torch.Tensor, problem_text: str = "") -> DADResult:
        t_start = time.time()
        all_solutions_across_rounds = []
        answer_entropies = []
        confidences = []
        per_round_dmaps = []  # NEW: store every round's disagreement map
        best_workspace = ""
        best_dmap = None

        base_prompt = self.tokenizer.decode(prompt_ids[0], skip_special_tokens=False)

        for round_idx in range(self.max_rounds):
            if round_idx == 0:
                current_prompt = base_prompt
                n_samples = self.m_samples
            else:
                refine_prompt = self._build_refine_prompt(base_prompt, best_workspace)
                current_prompt = refine_prompt
                n_samples = self.refine_samples

            solutions = self._sample_solutions(current_prompt, n_samples)
            all_solutions_across_rounds.extend(solutions)

            profiles = [profile_solution(s["text"], i) for i, s in enumerate(solutions)]
            dmap = build_disagreement_map(profiles)

            answer_entropies.append(dmap.answer_entropy)
            confidences.append(dmap.confidence_score)

            # Store this round's disagreement map
            per_round_dmaps.append(self._dmap_to_dict(dmap))

            best_workspace = format_workspace(problem_text, dmap, self.workspace_max_tokens)
            best_dmap = dmap

            logger.info(
                f"  Round {round_idx}: {len(solutions)} solutions, "
                f"H={dmap.answer_entropy:.2f}, "
                f"majority={dmap.majority_answer} ({dmap.majority_answer_fraction:.0%}), "
                f"confidence={dmap.confidence_score:.3f}, "
                f"agreed={len(dmap.agreed_claims)}, disputed={len(dmap.disputed_claims)}"
            )

            if dmap.confidence_score >= self.confidence_threshold and round_idx > 0:
                logger.info(f"  Early stop: confidence {dmap.confidence_score:.3f} >= {self.confidence_threshold}")
                break

            if dmap.answer_entropy == 0.0 and dmap.majority_answer_fraction == 1.0 and round_idx > 0:
                logger.info(f"  Early stop: unanimous agreement on {dmap.majority_answer}")
                break

            torch.cuda.empty_cache()

        final_profiles = [profile_solution(s["text"], i) for i, s in enumerate(all_solutions_across_rounds)]
        final_dmap = build_disagreement_map(final_profiles)

        best_solution = self._select_best_solution(all_solutions_across_rounds, final_dmap)

        dmap_dict = None
        if best_dmap:
            dmap_dict = self._dmap_to_dict(best_dmap)

        return DADResult(
            generated_text=best_solution["text"],
            extracted_answer=best_solution["answer"],
            all_solutions=[
                {"text": s["text"], "answer": s.get("answer", ""), "tokens": s.get("tokens", 0)}
                for s in all_solutions_across_rounds
            ],
            workspace_text=best_workspace,
            disagreement_map=dmap_dict,
            n_rounds=len(answer_entropies),
            n_total_generations=len(all_solutions_across_rounds),
            total_tokens=sum(s.get("tokens", 0) for s in all_solutions_across_rounds),
            wall_time_sec=time.time() - t_start,
            answer_entropy_per_round=answer_entropies,
            confidence_per_round=confidences,
            selected_method="disagreement_weighted_vote",
            per_round_disagreement_maps=per_round_dmaps,
        )

    @torch.no_grad()
    def _sample_solutions(self, prompt_text, n_samples):
        solutions = []
        prompt_ids = self.tokenizer(
            prompt_text, return_tensors="pt", truncation=True, max_length=4096,
        )["input_ids"].to(self.device)

        for _ in range(n_samples):
            out = self.model.generate(
                input_ids=prompt_ids,
                max_new_tokens=self.max_gen_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self.pad_token_id,
                eos_token_id=self.eos_token_id,
            )
            gen_ids = out[0, prompt_ids.shape[1]:].tolist()
            gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            answer = extract_boxed_answer(gen_text) or ""

            solutions.append({
                "text": gen_text,
                "answer": answer,
                "tokens": len(gen_ids),
            })
            del out
            torch.cuda.empty_cache()

        return solutions

    def _build_refine_prompt(self, base_prompt, workspace):
        system_msg = (
            "You are a helpful math assistant. Solve the following problem step by step. "
            "Show your reasoning clearly. Put your final answer in \\boxed{}."
        )

        parts = base_prompt.split("<|im_start|>user\n")
        if len(parts) >= 2:
            question_part = parts[1].split("<|im_end|>")[0]
        else:
            question_part = base_prompt

        refine_instruction = (
            f"{question_part}\n\n"
            f"Here is an analysis of multiple previous solution attempts:\n"
            f"<workspace>\n{workspace}\n</workspace>\n\n"
            f"Based on this analysis, provide a careful solution. "
            f"Pay special attention to any disputed claims and verify each step. "
            f"Do not simply repeat previous answers - check the reasoning independently."
        )

        prompt = (
            f"<|im_start|>system\n{system_msg}<|im_end|>\n"
            f"<|im_start|>user\n{refine_instruction}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        return prompt

    def _select_best_solution(self, all_solutions, dmap):
        from src.data.dataset import normalize_answer

        answer_scores = {}
        answer_best = {}

        for sol in all_solutions:
            ans = sol.get("answer", "")
            if not ans:
                continue
            norm = normalize_answer(ans)
            norm = norm.lstrip("0") or "0"
            try:
                val = float(norm)
                if val == int(val) and abs(val) < 1e15:
                    norm = str(int(val))
            except (ValueError, OverflowError):
                pass

            if norm not in answer_scores:
                answer_scores[norm] = 0.0
                answer_best[norm] = sol

            answer_scores[norm] += 1.0

            if len(sol.get("text", "")) > len(answer_best[norm].get("text", "")):
                answer_best[norm] = sol

        if not answer_scores:
            return all_solutions[-1] if all_solutions else {"text": "", "answer": ""}

        best_answer = max(answer_scores, key=answer_scores.get)
        return answer_best[best_answer]