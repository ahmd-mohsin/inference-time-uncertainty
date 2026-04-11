import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np

from src.search.tree import StepNode, SolutionTree
from src.search.diversity import compute_kl_divergence_topk, compute_combined_score

logger = logging.getLogger(__name__)

MAX_KL_CLAMP = 5.0


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

        if hasattr(model, 'config'):
            if model.config.max_position_embeddings < 32768:
                logger.info(f"Overriding max_position_embeddings from {model.config.max_position_embeddings} to 32768")
                model.config.max_position_embeddings = 32768

        self.device = cfg["model"]["device"]
        self.max_new_tokens = cfg["model"].get("max_new_tokens", 8192)
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

        search_cfg = cfg.get("search", {})
        self.n_solutions = search_cfg.get("n_solutions", 32)
        self.max_depth = search_cfg.get("max_depth", 15)
        self.max_step_tokens = search_cfg.get("max_step_tokens", 2048)
        self.temperature = search_cfg.get("sampling_temperature", 0.7)
        self.top_p = search_cfg.get("top_p", 0.95)
        self.max_path_tokens = search_cfg.get("max_path_tokens", 8192)

        self.lambda_div = search_cfg.get("lambda_diversity", 0.5)
        self.diversity_method = search_cfg.get("diversity_method", "kl")
        self.continuation_topk = search_cfg.get("continuation_topk", 50)
        self.aggregation = search_cfg.get("aggregation", "weighted_vote")

        self.k_expand = search_cfg.get("k_expand", 4)
        self.k_keep = search_cfg.get("k_keep", 2)
        self.max_active = search_cfg.get("max_active", 8)

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor, problem_text: str = "") -> RMITreeResult:
        t_start = time.time()
        prompt_len = prompt_ids.shape[1]
        tree = SolutionTree()
        completed_nodes = []

        logger.info(
            f"RMI-Tree: N={self.n_solutions}, k_expand={self.k_expand}, "
            f"k_keep={self.k_keep}, max_active={self.max_active}, "
            f"\u03bb={self.lambda_div}, max_depth={self.max_depth}, "
            f"max_step_tok={self.max_step_tokens}"
        )

        initial_nodes = self._sample_siblings(prompt_ids, None, self.k_expand, tree)
        self._score_sibling_group(initial_nodes, problem_text, tree)
        self._compute_sibling_diversity(initial_nodes)

        active = []
        for node in initial_nodes:
            if node.is_terminal:
                completed_nodes.append(node)
            else:
                active.append(node)

        logger.info(
            f"  Init: {len(initial_nodes)} siblings, "
            f"{len(completed_nodes)} completed, {len(active)} active, "
            f"PRM={[round(n.prm_reward,3) for n in initial_nodes]}, "
            f"DIV={[round(n.diversity_score,3) for n in initial_nodes]}"
        )

        max_iterations = self.n_solutions * 5
        iteration = 0

        while len(completed_nodes) < self.n_solutions and active and iteration < max_iterations:
            iteration += 1

            active.sort(key=lambda n: n.combined_score, reverse=True)
            if len(active) > self.max_active:
                active = active[:self.max_active]

            best_leaf = active[0]

            if best_leaf.depth >= self.max_depth or len(best_leaf.path_token_ids) >= self.max_path_tokens:
                active.remove(best_leaf)
                best_leaf.is_terminal = True
                best_leaf.is_complete = self._has_boxed_answer(best_leaf.path_text)
                if best_leaf.is_complete:
                    best_leaf.extracted_answer = self._extract_answer(best_leaf.path_text)
                completed_nodes.append(best_leaf)
                continue

            children = self._sample_siblings(prompt_ids, best_leaf, self.k_expand, tree)
            self._score_sibling_group(children, problem_text, tree)
            self._compute_sibling_diversity(children)

            active.remove(best_leaf)

            new_completed = [c for c in children if c.is_terminal]
            new_active = [c for c in children if not c.is_terminal]

            completed_nodes.extend(new_completed)
            remaining = self.n_solutions - len(completed_nodes)

            if remaining <= 0:
                logger.info(f"  Iter {iteration}: budget full ({len(completed_nodes)} solutions)")
                break

            new_active.sort(key=lambda n: n.combined_score, reverse=True)
            keep = min(self.k_keep, len(new_active))
            active.extend(new_active[:keep])

            if len(active) > self.max_active:
                active.sort(key=lambda n: n.combined_score, reverse=True)
                active = active[:self.max_active]

            logger.info(
                f"  Iter {iteration} (d={best_leaf.depth+1}): "
                f"{self.k_expand} siblings, {len(new_completed)} done, "
                f"kept {keep}, active={len(active)}, total_done={len(completed_nodes)}, "
                f"PRM={[round(c.prm_reward,3) for c in children]}, "
                f"DIV={[round(c.diversity_score,3) for c in children]}"
            )
            torch.cuda.empty_cache()

        if not completed_nodes:
            for node in active[:self.n_solutions]:
                node.is_terminal = True
                node.is_complete = self._has_boxed_answer(node.path_text)
                if node.is_complete:
                    node.extracted_answer = self._extract_answer(node.path_text)
                completed_nodes.append(node)

        if not completed_nodes:
            logger.warning("RMI-Tree: No solutions, falling back to greedy.")
            return self._greedy_fallback(prompt_ids, prompt_len, t_start)

        return self._aggregate_solutions(completed_nodes, tree, prompt_len, problem_text, t_start)

    def _sample_siblings(self, prompt_ids, parent_node, k, tree):
        if parent_node is None:
            input_ids = prompt_ids
            parent_path_ids, parent_path_text = [], ""
            depth, parent_id = 0, None
        else:
            path_tensor = torch.tensor([parent_node.path_token_ids], device=self.device)
            input_ids = torch.cat([prompt_ids, path_tensor], dim=1)
            if input_ids.shape[1] > self.max_path_tokens:
                input_ids = input_ids[:, :self.max_path_tokens]
            parent_path_ids = parent_node.path_token_ids
            parent_path_text = parent_node.path_text
            depth = parent_node.depth + 1
            parent_id = parent_node.node_id

        siblings = []
        for _ in range(k):
            remaining = self.max_path_tokens - input_ids.shape[1]
            step_max = min(self.max_step_tokens, max(remaining, 64))
            step_ids, step_text, is_terminal = self._sample_one_step(input_ids, step_max)

            node = StepNode(
                node_id=tree.new_id(), depth=depth, parent_id=parent_id,
                token_ids=step_ids, text=step_text,
                path_token_ids=parent_path_ids + step_ids,
                path_text=parent_path_text + step_text,
                is_terminal=is_terminal,
                is_complete=self._has_boxed_answer(parent_path_text + step_text),
            )
            if node.is_complete:
                node.extracted_answer = self._extract_answer(node.path_text)
                node.is_terminal = True
            if len(node.path_token_ids) >= self.max_path_tokens:
                node.is_terminal = True
                if not node.is_complete:
                    node.is_complete = self._has_boxed_answer(node.path_text)
                    if node.is_complete:
                        node.extracted_answer = self._extract_answer(node.path_text)

            if not node.is_terminal:
                self._get_continuation_dist(node, prompt_ids)

            tree.add_node(node)
            if depth == 0:
                tree.root_ids.append(node.node_id)
            tree.total_tokens_generated += len(step_ids)
            siblings.append(node)

            torch.cuda.empty_cache()

        return siblings

    @torch.no_grad()
    def _sample_one_step(self, input_ids, max_tokens):
        out = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_tokens,
            do_sample=(self.temperature >= 0.05),
            temperature=self.temperature if self.temperature >= 0.05 else None,
            top_p=self.top_p if self.temperature >= 0.05 else None,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
        )
        new_ids = out[0, input_ids.shape[1]:].tolist()
        del out
        torch.cuda.empty_cache()

        if not new_ids:
            return [], "", True

        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        is_terminal = (self.eos_token_id in new_ids) or self._has_boxed_answer(text)
        return new_ids, text, is_terminal

    @torch.no_grad()
    def _get_continuation_dist(self, node, prompt_ids):
        path_tensor = torch.tensor([node.path_token_ids[-64:]], device=self.device)
        trim_prompt = prompt_ids[:, -128:] if prompt_ids.shape[1] > 128 else prompt_ids
        ctx = torch.cat([trim_prompt, path_tensor], dim=1)

        out = self.model(input_ids=ctx, return_dict=True, use_cache=False)
        logits = out.logits[0, -1, :]
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)

        topk_probs, topk_ids = F.softmax(logits.float(), dim=-1).topk(self.continuation_topk)
        node.continuation_topk_ids = topk_ids.cpu().tolist()
        node.continuation_topk_logprobs = torch.log(topk_probs + 1e-10).cpu().tolist()

        del out, logits
        torch.cuda.empty_cache()

    def _score_sibling_group(self, siblings, problem_text, tree):
        for node in siblings:
            if node.parent_id is not None and node.parent_id in tree.nodes:
                solution_so_far = tree.nodes[node.parent_id].path_text
            else:
                solution_so_far = ""
            try:
                reward = self.prm.score_step(
                    problem_text=problem_text,
                    solution_so_far=solution_so_far,
                    current_step=node.text,
                )
            except Exception as e:
                logger.debug(f"PRM failed for node {node.node_id}: {e}")
                reward = 0.5
            node.prm_reward = reward
            tree.n_prm_calls += 1

    def _compute_sibling_diversity(self, siblings):
        non_terminal = [s for s in siblings if not s.is_terminal and s.continuation_topk_ids]

        if len(non_terminal) < 2:
            for s in siblings:
                s.diversity_score = 0.0
                s.combined_score = compute_combined_score(s.prm_reward, 0.0, self.lambda_div)
            return

        pool_ids = [s.continuation_topk_ids for s in non_terminal]
        pool_lps = [s.continuation_topk_logprobs for s in non_terminal]

        for s in siblings:
            if not s.continuation_topk_ids or s.is_terminal:
                s.diversity_score = 0.0
                s.combined_score = compute_combined_score(s.prm_reward, 0.0, self.lambda_div)
                continue

            idx = non_terminal.index(s)
            sib_ids = pool_ids[:idx] + pool_ids[idx+1:]
            sib_lps = pool_lps[:idx] + pool_lps[idx+1:]

            if self.diversity_method == "kl":
                div = compute_kl_divergence_topk(
                    s.continuation_topk_ids, s.continuation_topk_logprobs,
                    sib_ids, sib_lps, vocab_size=len(self.tokenizer),
                )
                div = min(div, MAX_KL_CLAMP)
            else:
                node_set = set(s.continuation_topk_ids[:10])
                overlaps = []
                for sid in sib_ids:
                    sib_set = set(sid[:10])
                    jaccard = len(node_set & sib_set) / max(1, len(node_set | sib_set))
                    overlaps.append(jaccard)
                div = 1.0 - (sum(overlaps) / max(1, len(overlaps))) if overlaps else 0.0

            s.diversity_score = div
            s.combined_score = compute_combined_score(s.prm_reward, div, self.lambda_div)

    def _aggregate_solutions(self, completed, tree, prompt_len, problem_text, t_start):
        for node in completed:
            if node.path_text.strip():
                try:
                    sol_reward, step_scores = self.prm.score_solution(
                        problem_text=problem_text, full_solution=node.path_text,
                    )
                    node.prm_reward = sol_reward
                    tree.n_prm_calls += 1
                except Exception:
                    if node.prm_reward == 0.0:
                        node.prm_reward = 0.5

        all_solutions = []
        for node in completed:
            answer = node.extracted_answer if node.extracted_answer else self._extract_answer(node.path_text)
            all_solutions.append({
                "node_id": node.node_id, "text": node.path_text, "answer": answer,
                "prm_reward": node.prm_reward, "diversity_score": node.diversity_score,
                "depth": node.depth, "n_tokens": len(node.path_token_ids),
            })

        if all_solutions:
            logger.info(
                f"  Aggregating {len(all_solutions)} solutions: "
                f"rewards={[round(s['prm_reward'],3) for s in all_solutions]}, "
                f"answers={[s['answer'] for s in all_solutions]}"
            )

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
            generated_text=best["text"], generated_ids=tree.nodes[best["node_id"]].path_token_ids,
            extracted_answer=best["answer"], all_solutions=all_solutions,
            prompt_len=prompt_len, total_tokens=tree.total_tokens_generated,
            wall_time_sec=time.time() - t_start,
            n_completed_solutions=len(all_solutions), n_prm_calls=tree.n_prm_calls,
            tree_max_depth=tree.max_depth(),
            mean_prm_reward=float(np.mean(rewards)) if rewards else 0.0,
            mean_diversity_score=float(np.mean(diversities)) if diversities else 0.0,
            selected_method=method,
            n_entropy_triggers=0, n_expansion_triggers=len(all_solutions),
            total_expansion_tokens=tree.total_tokens_generated,
            trigger_rate_entropy=0.0, trigger_rate_expansion=1.0,
        )

    def _weighted_majority_vote(self, solutions):
        answer_scores, answer_best = {}, {}
        for sol in solutions:
            a = self._normalize_for_voting(sol["answer"])
            if not a: continue
            answer_scores[a] = answer_scores.get(a, 0.0) + sol["prm_reward"]
            if a not in answer_best or sol["prm_reward"] > answer_best[a]["prm_reward"]:
                answer_best[a] = sol
        if not answer_scores:
            return max(solutions, key=lambda s: s["prm_reward"])
        return answer_best[max(answer_scores, key=answer_scores.get)]

    def _majority_vote(self, solutions):
        answer_counts, answer_best = {}, {}
        for sol in solutions:
            a = self._normalize_for_voting(sol["answer"])
            if not a: continue
            answer_counts[a] = answer_counts.get(a, 0) + 1
            if a not in answer_best or sol["prm_reward"] > answer_best[a]["prm_reward"]:
                answer_best[a] = sol
        if not answer_counts:
            return max(solutions, key=lambda s: s["prm_reward"])
        return answer_best[max(answer_counts, key=answer_counts.get)]

    def _has_boxed_answer(self, text):
        if "\\boxed{" not in text: return False
        depth = 0
        in_boxed = False
        i = 0
        while i < len(text):
            if text[i:i+7] == "\\boxed{":
                in_boxed = True
                depth = 1
                i += 7
                continue
            if in_boxed:
                if text[i] == "{": depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0: return True
            i += 1
        return False

    def _extract_answer(self, text):
        from src.data.dataset import extract_boxed_answer, extract_numeric_answer
        return extract_boxed_answer(text) or extract_numeric_answer(text)

    def _normalize_for_voting(self, answer):
        if answer is None: return ""
        from src.data.dataset import normalize_answer
        n = normalize_answer(answer)
        n = n.lstrip("0") or "0"
        try:
            val = float(n)
            if val == int(val) and abs(val) < 1e15:
                return str(int(val))
            return f"{val:.8f}".rstrip("0").rstrip(".")
        except (ValueError, OverflowError):
            pass
        return n

    def _greedy_fallback(self, prompt_ids, prompt_len, t_start):
        out = self.model.generate(
            input_ids=prompt_ids, max_new_tokens=self.max_new_tokens,
            do_sample=False, pad_token_id=self.pad_token_id, eos_token_id=self.eos_token_id,
        )
        gen_ids = out[0, prompt_len:].tolist()
        gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return RMITreeResult(
            generated_text=gen_text, generated_ids=gen_ids,
            extracted_answer=self._extract_answer(gen_text),
            prompt_len=prompt_len, total_tokens=len(gen_ids),
            wall_time_sec=time.time() - t_start,
            n_completed_solutions=1, selected_method="greedy_fallback",
        )