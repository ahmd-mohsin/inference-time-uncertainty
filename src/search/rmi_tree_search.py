import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np

from src.search.tree import StepNode, SolutionTree
from src.search.diversity import (
    compute_kl_divergence_topk,
    compute_combined_score,
)

logger = logging.getLogger(__name__)


@dataclass
class RMITreeResult:
    generated_text: str
    generated_ids: list[int]
    extracted_answer: Optional[str]
    all_solutions: list[dict] = field(default_factory=list)
    prompt_len: int = 0
    total_tokens: int = 0
    wall_time_sec: float = 0.0
    n_entropy_triggers: int = 0
    n_expansion_triggers: int = 0
    total_expansion_tokens: int = 0
    trigger_rate_entropy: float = 0.0
    trigger_rate_expansion: float = 0.0
    n_completed_solutions: int = 0
    n_prm_calls: int = 0
    tree_max_depth: int = 0
    mean_prm_reward: float = 0.0
    mean_diversity_score: float = 0.0
    selected_method: str = "best_reward"


class RMITreeSearchGenerator:

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
        self.max_depth = search_cfg.get("max_depth", 20)
        self.max_step_tokens = search_cfg.get("max_step_tokens", 512)
        self.temperature = search_cfg.get("sampling_temperature", 0.7)
        self.balance_temperature = search_cfg.get("balance_temperature", 0.1)
        self.top_p = search_cfg.get("top_p", 0.95)
        self.max_path_tokens = search_cfg.get("max_path_tokens", 4096)

        self.lambda_div = search_cfg.get("lambda_diversity", 0.5)
        self.diversity_method = search_cfg.get("diversity_method", "kl")
        self.continuation_topk = search_cfg.get("continuation_topk", 50)
        self.aggregation = search_cfg.get("aggregation", "weighted_vote")

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor, problem_text: str = "") -> RMITreeResult:
        t_start = time.time()
        prompt_len = prompt_ids.shape[1]

        tree = SolutionTree()
        budget = self.n_solutions

        logger.info(
            f"RMI-Tree: Starting search with N={self.n_solutions}, "
            f"max_depth={self.max_depth}, λ={self.lambda_div}"
        )

        depth0_nodes = self._sample_first_steps(prompt_ids, budget, tree)
        completed_at_d0 = sum(1 for n in depth0_nodes if n.is_terminal)
        budget -= completed_at_d0
        tree.n_completed_solutions += completed_at_d0

        logger.info(
            f"  Depth 0: {len(depth0_nodes)} nodes, "
            f"{completed_at_d0} completed, budget={budget}"
        )

        for depth in range(1, self.max_depth + 1):
            if budget <= 0:
                break

            leaves = tree.get_leaves_at_depth(depth - 1)
            active_leaves = [n for n in leaves if not n.is_terminal and not n.children_ids]

            if not active_leaves:
                logger.info(f"  Depth {depth}: No active leaves, stopping.")
                break

            self._score_nodes_with_prm(active_leaves, problem_text, tree)
            self._compute_diversity_scores(active_leaves)
            widths = self._allocate_expansion_widths(active_leaves, budget)

            new_completed = 0
            for node, width in zip(active_leaves, widths):
                if width <= 0:
                    continue
                if len(node.path_token_ids) >= self.max_path_tokens:
                    node.is_terminal = True
                    node.is_complete = self._has_boxed_answer(node.path_text)
                    if node.is_complete:
                        node.extracted_answer = self._extract_answer(node.path_text)
                        new_completed += 1
                    continue

                children = self._expand_node(node, prompt_ids, width, tree)
                new_completed += sum(1 for c in children if c.is_terminal)

            budget -= new_completed
            tree.n_completed_solutions += new_completed

            n_total_leaves = len(tree.get_leaves_at_depth(depth))
            logger.info(
                f"  Depth {depth}: expanded {sum(widths)} children, "
                f"{new_completed} completed, budget={budget}, "
                f"total_leaves={n_total_leaves}"
            )

            torch.cuda.empty_cache()

        completed = tree.get_complete_solutions()
        if not completed:
            completed = tree.get_terminal_nodes()
        if not completed:
            all_nodes = sorted(tree.nodes.values(), key=lambda n: len(n.path_token_ids), reverse=True)
            for n in all_nodes[:self.n_solutions]:
                n.is_terminal = True
                n.is_complete = self._has_boxed_answer(n.path_text)
                if n.is_complete:
                    n.extracted_answer = self._extract_answer(n.path_text)
                    completed.append(n)

        if not completed:
            logger.warning("RMI-Tree: No solutions found, falling back to greedy.")
            return self._greedy_fallback(prompt_ids, prompt_len, t_start)

        result = self._aggregate_solutions(completed, tree, prompt_len, problem_text, t_start)
        return result

    @torch.no_grad()
    def _sample_one_step(
        self,
        input_ids: torch.Tensor,
        max_tokens: int,
    ) -> tuple[list[int], str, bool, Optional[torch.Tensor]]:
        seq_len_before = input_ids.shape[1]

        out = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_tokens,
            do_sample=(self.temperature >= 0.05),
            temperature=self.temperature if self.temperature >= 0.05 else None,
            top_p=self.top_p if self.temperature >= 0.05 else None,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True,
        )

        all_new_ids = out.sequences[0, seq_len_before:].tolist()

        if not all_new_ids:
            return [], "", True, None

        is_terminal = False
        step_ids = []
        decoded_so_far = ""
        newline_count = 0

        for i, tid in enumerate(all_new_ids):
            step_ids.append(tid)

            if tid == self.eos_token_id:
                is_terminal = True
                break

            tok_str = self.tokenizer.decode([tid], skip_special_tokens=False)
            decoded_so_far += tok_str

            if "\n" in tok_str:
                newline_count += tok_str.count("\n")
            else:
                newline_count = 0

            if i >= 5 and newline_count >= 2:
                break

            if "\\boxed{" in decoded_so_far and self._has_boxed_answer(decoded_so_far):
                is_terminal = True
                break

        text = self.tokenizer.decode(step_ids, skip_special_tokens=True)

        last_logits = None
        if out.scores and len(step_ids) <= len(out.scores):
            last_logits = out.scores[len(step_ids) - 1].squeeze(0)
            last_logits = torch.nan_to_num(last_logits, nan=0.0, posinf=1e4, neginf=-1e4)

        del out
        torch.cuda.empty_cache()

        return step_ids, text, is_terminal, last_logits

    @torch.no_grad()
    def _sample_first_steps(
        self,
        prompt_ids: torch.Tensor,
        n_samples: int,
        tree: SolutionTree,
    ) -> list[StepNode]:
        nodes = []
        for _ in range(n_samples):
            step_ids, step_text, is_terminal, logits_last = self._sample_one_step(
                input_ids=prompt_ids,
                max_tokens=self.max_step_tokens,
            )

            node = StepNode(
                node_id=tree.new_id(),
                depth=0,
                parent_id=None,
                token_ids=step_ids,
                text=step_text,
                path_token_ids=step_ids,
                path_text=step_text,
                is_terminal=is_terminal,
                is_complete=self._has_boxed_answer(step_text),
            )

            if node.is_complete:
                node.extracted_answer = self._extract_answer(step_text)
                node.is_terminal = True

            if logits_last is not None and not is_terminal:
                topk_probs, topk_ids = F.softmax(logits_last.float(), dim=-1).topk(
                    self.continuation_topk
                )
                node.continuation_topk_ids = topk_ids.cpu().tolist()
                node.continuation_topk_logprobs = torch.log(topk_probs + 1e-10).cpu().tolist()

            tree.add_node(node)
            tree.root_ids.append(node.node_id)
            tree.total_tokens_generated += len(step_ids)
            nodes.append(node)

        return nodes

    @torch.no_grad()
    def _expand_node(
        self,
        node: StepNode,
        prompt_ids: torch.Tensor,
        n_children: int,
        tree: SolutionTree,
    ) -> list[StepNode]:
        path_tensor = torch.tensor([node.path_token_ids], device=self.device)
        full_input = torch.cat([prompt_ids, path_tensor], dim=1)

        if full_input.shape[1] > self.max_path_tokens:
            full_input = full_input[:, :self.max_path_tokens]

        children = []
        for _ in range(n_children):
            remaining = self.max_path_tokens - full_input.shape[1]
            step_max = min(self.max_step_tokens, max(remaining, 64))

            step_ids, step_text, is_terminal, logits_last = self._sample_one_step(
                input_ids=full_input,
                max_tokens=step_max,
            )

            child = StepNode(
                node_id=tree.new_id(),
                depth=node.depth + 1,
                parent_id=node.node_id,
                token_ids=step_ids,
                text=step_text,
                path_token_ids=node.path_token_ids + step_ids,
                path_text=node.path_text + step_text,
                is_terminal=is_terminal,
                is_complete=self._has_boxed_answer(node.path_text + step_text),
            )

            if child.is_complete:
                child.extracted_answer = self._extract_answer(child.path_text)
                child.is_terminal = True

            if len(child.path_token_ids) >= self.max_path_tokens:
                child.is_terminal = True
                if not child.is_complete:
                    child.is_complete = self._has_boxed_answer(child.path_text)
                    if child.is_complete:
                        child.extracted_answer = self._extract_answer(child.path_text)

            if logits_last is not None and not child.is_terminal:
                topk_probs, topk_ids = F.softmax(logits_last.float(), dim=-1).topk(
                    self.continuation_topk
                )
                child.continuation_topk_ids = topk_ids.cpu().tolist()
                child.continuation_topk_logprobs = torch.log(topk_probs + 1e-10).cpu().tolist()

            tree.add_node(child)
            tree.total_tokens_generated += len(step_ids)
            children.append(child)

        node.expansion_width = n_children
        return children

    def _score_nodes_with_prm(self, nodes: list[StepNode], problem_text: str, tree: SolutionTree) -> None:
        for node in nodes:
            if node.parent_id is not None:
                parent = tree.nodes[node.parent_id]
                solution_so_far = parent.path_text
            else:
                solution_so_far = ""

            try:
                reward = self.prm.score_step(
                    problem_text=problem_text,
                    solution_so_far=solution_so_far,
                    current_step=node.text,
                )
            except Exception as e:
                logger.debug(f"PRM scoring failed for node {node.node_id}: {e}")
                reward = 0.5

            node.prm_reward = reward
            tree.n_prm_calls += 1

    def _compute_diversity_scores(self, nodes: list[StepNode]) -> None:
        if len(nodes) <= 1:
            for n in nodes:
                n.diversity_score = 0.0
                n.combined_score = compute_combined_score(n.prm_reward, 0.0, self.lambda_div)
            return

        pool_topk_ids = [n.continuation_topk_ids for n in nodes]
        pool_topk_logprobs = [n.continuation_topk_logprobs for n in nodes]

        for i, node in enumerate(nodes):
            sibling_ids = pool_topk_ids[:i] + pool_topk_ids[i + 1:]
            sibling_lps = pool_topk_logprobs[:i] + pool_topk_logprobs[i + 1:]

            if not node.continuation_topk_ids or not any(sibling_ids):
                div = 0.0
            elif self.diversity_method == "kl":
                div = compute_kl_divergence_topk(
                    node.continuation_topk_ids,
                    node.continuation_topk_logprobs,
                    [s for s in sibling_ids if s],
                    [s for s in sibling_lps if s],
                    vocab_size=len(self.tokenizer),
                )
            else:
                node_set = set(node.continuation_topk_ids[:10])
                overlaps = []
                for sib_ids in sibling_ids:
                    if not sib_ids:
                        continue
                    sib_set = set(sib_ids[:10])
                    jaccard = len(node_set & sib_set) / max(1, len(node_set | sib_set))
                    overlaps.append(jaccard)
                div = 1.0 - (sum(overlaps) / max(1, len(overlaps))) if overlaps else 0.0

            node.diversity_score = div
            node.combined_score = compute_combined_score(node.prm_reward, div, self.lambda_div)

    def _allocate_expansion_widths(self, nodes: list[StepNode], budget: int) -> list[int]:
        if not nodes or budget <= 0:
            return [0] * len(nodes)

        scores = np.array([n.combined_score for n in nodes], dtype=np.float64)

        scores_scaled = scores / max(self.balance_temperature, 1e-8)
        scores_scaled -= scores_scaled.max()
        exp_scores = np.exp(scores_scaled)
        softmax_weights = exp_scores / exp_scores.sum()

        raw_widths = budget * softmax_weights
        widths = np.round(raw_widths).astype(int)

        while widths.sum() > budget:
            idx = np.argmin(raw_widths - np.floor(raw_widths))
            if widths[idx] > 0:
                widths[idx] -= 1

        for i in range(len(widths)):
            if widths[i] == 0 and scores[i] > 0 and widths.sum() < budget:
                widths[i] = 1

        return widths.tolist()

    def _aggregate_solutions(self, completed, tree, prompt_len, problem_text, t_start):
        all_solutions = []
        for node in completed:
            answer = node.extracted_answer
            if answer is None:
                answer = self._extract_answer(node.path_text)

            all_solutions.append({
                "node_id": node.node_id,
                "text": node.path_text,
                "answer": answer,
                "prm_reward": node.prm_reward,
                "diversity_score": node.diversity_score,
                "depth": node.depth,
                "n_tokens": len(node.path_token_ids),
            })

        if self.aggregation == "weighted_vote":
            best = self._weighted_majority_vote(all_solutions)
            method = "weighted_vote"
        elif self.aggregation == "majority_vote":
            best = self._majority_vote(all_solutions)
            method = "majority_vote"
        else:
            best = max(all_solutions, key=lambda s: s["prm_reward"])
            method = "best_reward"

        rewards = [s["prm_reward"] for s in all_solutions]
        diversities = [s["diversity_score"] for s in all_solutions]

        return RMITreeResult(
            generated_text=best["text"],
            generated_ids=tree.nodes[best["node_id"]].path_token_ids,
            extracted_answer=best["answer"],
            all_solutions=all_solutions,
            prompt_len=prompt_len,
            total_tokens=tree.total_tokens_generated,
            wall_time_sec=time.time() - t_start,
            n_completed_solutions=len(all_solutions),
            n_prm_calls=tree.n_prm_calls,
            tree_max_depth=tree.max_depth(),
            mean_prm_reward=float(np.mean(rewards)) if rewards else 0.0,
            mean_diversity_score=float(np.mean(diversities)) if diversities else 0.0,
            selected_method=method,
            n_entropy_triggers=0,
            n_expansion_triggers=len(all_solutions),
            total_expansion_tokens=tree.total_tokens_generated,
            trigger_rate_entropy=0.0,
            trigger_rate_expansion=1.0,
        )

    def _weighted_majority_vote(self, solutions):
        answer_scores = {}
        answer_best = {}
        for sol in solutions:
            a = self._normalize_for_voting(sol["answer"])
            if not a:
                continue
            answer_scores[a] = answer_scores.get(a, 0.0) + sol["prm_reward"]
            if a not in answer_best or sol["prm_reward"] > answer_best[a]["prm_reward"]:
                answer_best[a] = sol
        if not answer_scores:
            return max(solutions, key=lambda s: s["prm_reward"])
        best_answer = max(answer_scores, key=answer_scores.get)
        return answer_best[best_answer]

    def _majority_vote(self, solutions):
        answer_counts = {}
        answer_best = {}
        for sol in solutions:
            a = self._normalize_for_voting(sol["answer"])
            if not a:
                continue
            answer_counts[a] = answer_counts.get(a, 0) + 1
            if a not in answer_best or sol["prm_reward"] > answer_best[a]["prm_reward"]:
                answer_best[a] = sol
        if not answer_counts:
            return max(solutions, key=lambda s: s["prm_reward"])
        best_answer = max(answer_counts, key=answer_counts.get)
        return answer_best[best_answer]

    def _has_boxed_answer(self, text):
        if "\\boxed{" not in text:
            return False
        depth = 0
        in_boxed = False
        i = 0
        while i < len(text):
            if text[i:i + 7] == "\\boxed{":
                in_boxed = True
                depth = 1
                i += 7
                continue
            if in_boxed:
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        return True
            i += 1
        return False

    def _extract_answer(self, text):
        from src.data.dataset import extract_boxed_answer, extract_numeric_answer
        answer = extract_boxed_answer(text)
        if answer:
            return answer
        return extract_numeric_answer(text)

    def _normalize_for_voting(self, answer):
        if answer is None:
            return ""
        from src.data.dataset import normalize_answer
        return normalize_answer(answer)

    def _greedy_fallback(self, prompt_ids, prompt_len, t_start):
        out = self.model.generate(
            input_ids=prompt_ids,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
        )
        gen_ids = out[0, prompt_len:].tolist()
        gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        answer = self._extract_answer(gen_text)
        return RMITreeResult(
            generated_text=gen_text,
            generated_ids=gen_ids,
            extracted_answer=answer,
            prompt_len=prompt_len,
            total_tokens=len(gen_ids),
            wall_time_sec=time.time() - t_start,
            n_completed_solutions=1,
            selected_method="greedy_fallback",
        )