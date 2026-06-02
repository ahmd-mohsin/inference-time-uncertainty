# dad_generator.py
#
# Adaptive Disagreement-Aware Distillation.
#
# Replaces the fixed (M, M/2, M/2) schedule with a budget-aware loop:
#   1. wide unconditional PROBE round to surface the contested structure;
#   2. each refinement round picks the single highest-LEVERAGE disputed
#      coordinate (Gauss-Southwell) and conditions on it;
#   3. the number of samples per round is set by WATER-FILLING under a token
#      budget, and the loop STOPS when the marginal contested-mass removed per
#      token drops below the water level.
#
# The refinement prompt is built with the tokenizer's chat template, so it is
# correct across model families (Qwen, LLaMA, Ministral, DeepSeek) instead of
# hard-coding ChatML markers.

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import torch

from src.dad.claim_extractor import profile_solution, extract_boxed_answer
from src.dad.disagreement_analyzer import (
    build_disagreement_map, format_workspace, DisagreementMap,
)
from src.dad.allocation import AllocationConfig, allocate_round, update_rho

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
    per_round_disagreement_maps: list[dict] = field(default_factory=list)
    # adaptive-allocation telemetry
    contested_mass_per_round: list[float] = field(default_factory=list)
    samples_per_round: list[int] = field(default_factory=list)
    chosen_coordinate_per_round: list[str] = field(default_factory=list)
    leverage_per_round: list[float] = field(default_factory=list)
    stop_reason: str = ""


class DADGenerator:
    def __init__(self, model, tokenizer, cfg: dict):
        self.model = model
        self.tokenizer = tokenizer
        self.device = cfg["model"]["device"]

        dad_cfg = cfg.get("dad", {})
        self.model_name = cfg["model"].get("name", "")
        self.m_samples = dad_cfg.get("m_samples", 8)
        self.max_rounds = dad_cfg.get("max_rounds", 3)
        self.max_gen_tokens = dad_cfg.get("max_gen_tokens", 2048)
        self.temperature = dad_cfg.get("temperature", 0.7)
        self.top_p = dad_cfg.get("top_p", 0.95)
        self.confidence_threshold = dad_cfg.get("confidence_threshold", 0.8)
        # Residual contested-mass tolerance for the "settled" stop: when the
        # ANSWER is unanimous (real, non-blank) and residual mass is below this,
        # remaining disagreement is extraction noise, not signal. Set 0 to force
        # exact claim convergence. Default 1.5 ~ a couple of noisy keys.
        self.contested_mass_floor = dad_cfg.get("contested_mass_floor", 1.5)
        self.workspace_max_tokens = dad_cfg.get("workspace_max_tokens", 800)
        self.system_prompt = dad_cfg.get(
            "system_prompt",
            "You are a helpful math assistant. Solve the problem step by step, "
            "show your reasoning clearly, and put your final answer in \\boxed{}.",
        )

        # adaptive allocation config (sensible default budget if unset)
        default_budget = self.m_samples * self.max_rounds * (self.max_gen_tokens + 800)
        self.alloc = AllocationConfig(
            token_budget=dad_cfg.get("token_budget", default_budget),
            m_min=dad_cfg.get("min_round_samples", 2),
            m_max=dad_cfg.get("max_round_samples", self.m_samples),
            rho_prior=dad_cfg.get("rho_prior", 0.45),
            min_marginal_gain=dad_cfg.get("min_marginal_gain", 0.02),
            probe_samples=self.m_samples,
        )

        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

        if hasattr(model, "config") and getattr(
            model.config, "max_position_embeddings", 1 << 30) < 32768:
            model.config.max_position_embeddings = 32768

    # ------------------------------------------------------------------
    def _dmap_to_dict(self, d: DisagreementMap) -> dict:
        return {
            "n_solutions": d.n_solutions,
            "answer_distribution": d.answer_distribution,
            "answer_entropy": d.answer_entropy,
            "majority_answer": d.majority_answer,
            "majority_answer_fraction": d.majority_answer_fraction,
            "n_agreed": len(d.agreed_claims),
            "n_disputed": len(d.disputed_claims),
            "confidence": d.confidence_score,
            "contested_mass": d.contested_mass,
            "top_leverage_key": d.top_leverage_key,
            "n_substantive_disputes": d.n_substantive_disputes,
            "top_leverage": d.disputed_claims[0].leverage if d.disputed_claims else 0.0,
        }

    # ------------------------------------------------------------------
    def generate(self, prompt_ids: torch.Tensor, problem_text: str = "") -> DADResult:
        t_start = time.time()
        all_solutions = []
        entropies, confidences = [], []
        contested, widths, coords, levs = [], [], [], []
        per_round_dmaps = []
        workspace = ""
        last_dmap = None
        stop_reason = "max_rounds"

        prompt_len = int(prompt_ids.shape[1])
        L_est = float(self.max_gen_tokens)          # generation-length estimate
        rho = self.alloc.rho_prior
        tokens_used = 0
        prev_psi = None

        for r in range(self.max_rounds):
            # ---- decide sample count for this round ----
            if r == 0:
                n = self.alloc.probe_samples
                prompt_text = self.tokenizer.decode(prompt_ids[0],
                                                    skip_special_tokens=False)
            else:
                ws_tokens = len(self.tokenizer.encode(workspace))
                cost = prompt_len + ws_tokens + L_est
                remaining = self.alloc.token_budget - tokens_used
                # resolvable mass = leverage of the chosen (top) coordinate
                delta = min(last_dmap.disputed_claims[0].leverage,
                            last_dmap.contested_mass) if last_dmap.disputed_claims else 0.0
                n, stop = allocate_round(remaining, cost, delta, rho, self.alloc)
                if stop or n <= 0:
                    stop_reason = "water_level" if delta > 0 else "converged"
                    break
                prompt_text = self._build_refine_prompt(problem_text, workspace)

            # ---- sample ----
            sols = self._sample_solutions(prompt_text, n)
            all_solutions.extend(sols)
            tokens_used += sum(s["tokens"] for s in sols)
            if sols:
                L_est = 0.5 * L_est + 0.5 * (sum(s["tokens"] for s in sols) / len(sols))

            # ---- analyze ----
            profiles = [profile_solution(s["text"], i) for i, s in enumerate(sols)]
            dmap = build_disagreement_map(profiles)

            entropies.append(dmap.answer_entropy)
            confidences.append(dmap.confidence_score)
            contested.append(dmap.contested_mass)
            widths.append(n)
            coords.append(dmap.top_leverage_key)
            levs.append(dmap.disputed_claims[0].leverage if dmap.disputed_claims else 0.0)
            per_round_dmaps.append(self._dmap_to_dict(dmap))

            workspace = format_workspace(problem_text, dmap, self.workspace_max_tokens)
            last_dmap = dmap

            # ---- update rho from the realized contraction ----
            if prev_psi is not None and r > 0:
                observed_shift = max(0.0, prev_psi - dmap.contested_mass)
                resolvable = max(prev_psi, 1e-6)
                rho = update_rho(rho, observed_shift, resolvable, n)
            prev_psi = dmap.contested_mass

            logger.info(
                f"  Round {r}: n={n}, H={dmap.answer_entropy:.2f}, "
                f"maj={dmap.majority_answer}({dmap.majority_answer_fraction:.0%}), "
                f"conf={dmap.confidence_score:.3f}, agreed={len(dmap.agreed_claims)}, "
                f"disputed={len(dmap.disputed_claims)}, Psi={dmap.contested_mass:.2f}, "
                f"top='{dmap.top_leverage_key}', rho={rho:.2f}, used={tokens_used}"
            )

            # ---- claim-level convergence (can fire on round 0) ----
            # (1) No majority-engaged disputed claims remain -> done.
            if not dmap.disputed_claims:
                stop_reason = "claims_converged"
                break
            # (2) "Settled": the ANSWER is unanimous on a REAL (non-blank) value
            # AND no SUBSTANTIVE dispute remains (only bare-variable/definition
            # noise). The two guards are essential:
            #   - non-blank majority: all-truncated chains must keep retrying;
            #   - zero substantive disputes: entropy inversion (unanimous answer
            #     with a real disputed claim) must keep refining, regardless of
            #     how small its residual mass happens to be.
            if (dmap.answer_entropy == 0.0
                    and dmap.majority_answer not in ("", None)
                    and dmap.majority_answer_fraction == 1.0
                    and dmap.n_substantive_disputes == 0
                    and dmap.contested_mass <= self.contested_mass_floor):
                stop_reason = "settled"
                break

            torch.cuda.empty_cache()

        # ---- final answer over all solutions across rounds ----
        final_profiles = [profile_solution(s["text"], i)
                           for i, s in enumerate(all_solutions)]
        final_dmap = build_disagreement_map(final_profiles)
        best = self._select_best_solution(all_solutions, final_dmap)

        return DADResult(
            generated_text=best["text"],
            extracted_answer=best.get("answer", ""),
            all_solutions=[{"text": s["text"], "answer": s.get("answer", ""),
                            "tokens": s.get("tokens", 0)} for s in all_solutions],
            workspace_text=workspace,
            disagreement_map=self._dmap_to_dict(last_dmap) if last_dmap else None,
            n_rounds=len(widths),
            n_total_generations=len(all_solutions),
            total_tokens=tokens_used,
            wall_time_sec=time.time() - t_start,
            answer_entropy_per_round=entropies,
            confidence_per_round=confidences,
            selected_method="adaptive_gauss_southwell_waterfill",
            per_round_disagreement_maps=per_round_dmaps,
            contested_mass_per_round=contested,
            samples_per_round=widths,
            chosen_coordinate_per_round=coords,
            leverage_per_round=levs,
            stop_reason=stop_reason,
        )

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _sample_solutions(self, prompt_text, n_samples):
        solutions = []
        prompt_ids = self.tokenizer(
            prompt_text, return_tensors="pt", truncation=True, max_length=8192,
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
            # boxed first; fall back to the rich extractor so a finished but
            # unboxed chain (e.g. "... m+n = 113.") still yields an answer.
            ans = extract_boxed_answer(gen_text)
            if not ans:
                try:
                    from src.data.dataset import extract_numeric_answer
                    ans = extract_numeric_answer(gen_text) or ""
                except Exception:
                    ans = ""
            solutions.append({
                "text": gen_text,
                "answer": ans,
                "tokens": len(gen_ids),
            })
            del out
            torch.cuda.empty_cache()
        return solutions

    # ------------------------------------------------------------------
    def _build_refine_prompt(self, problem_text: str, workspace: str) -> str:
        """Build a refinement prompt using the SAME per-model template as round 1.

        Reuses src.data.dataset.format_prompt (Qwen / LLaMA / Ministral / Gemma /
        DeepSeek aware) by folding the workspace into the question, so refinement
        rounds are formatted identically to the probe round and the baselines.
        """
        aug_question = (
            f"{problem_text}\n\n"
            f"Below is an analysis of several previous attempts at this problem:\n"
            f"<workspace>\n{workspace}\n</workspace>\n\n"
            f"Treat the AGREED FACTS as established. Re-derive the PRIMARY DISPUTE "
            f"explicitly and verify every line independently — do not simply copy a "
            f"previous answer. Put your final answer in \\boxed{{}}."
        )
        try:
            from src.data.dataset import format_prompt
            return format_prompt({"question": aug_question}, self.model_name)
        except Exception:
            return (f"System: {self.system_prompt}\n\nUser: {aug_question}\n\nAssistant:")

    # ------------------------------------------------------------------
    def _select_best_solution(self, all_solutions, dmap):
        from src.data.dataset import normalize_answer

        scores, best = {}, {}
        for sol in all_solutions:
            ans = sol.get("answer", "")
            if not ans:
                continue
            norm = normalize_answer(ans)
            norm = norm.lstrip("0") or "0"
            try:
                v = float(norm)
                if v == int(v) and abs(v) < 1e15:
                    norm = str(int(v))
            except (ValueError, OverflowError):
                pass
            if norm not in scores:
                scores[norm] = 0.0
                best[norm] = sol
            scores[norm] += 1.0
            if len(sol.get("text", "")) > len(best[norm].get("text", "")):
                best[norm] = sol

        if not scores:
            return all_solutions[-1] if all_solutions else {"text": "", "answer": ""}
        return best[max(scores, key=scores.get)]