# Reward functions for GRPO (docs/RL.md §4.1, Component A).
#
# TRL contract (verified against installed trl/trl/rewards/accuracy_rewards.py):
#   reward_fn(prompts, completions, completion_ids, trainer_state, **kwargs) -> list[float|None]
#   - dataset extra columns arrive as kwargs aligned with completions (e.g. gold_answer)
#   - for standard (text) datasets, completions is a list[str]; for conversational it's
#     list[list[{role,content}]] — we handle both.
#   - completions for the same prompt form a group (size = num_generations). We recover
#     groups by identical prompt string, which is robust to batch layout.
#   - multiple reward funcs are summed (weighted by reward_weights). We expose TWO:
#       correctness_reward  -> {0,1}
#       novelty_bonus       -> lambda * novelty for CORRECT rollouts, else 0
#     so the effective reward is correct*(1 + lambda*novelty). Keeping them separate lets
#     TRL log each and lets us ablate by setting reward_weights=[1,0] (=plain GRPO).

import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# extract_numeric_answer is the robust 7-strategy extractor (boxed in full text, strip
# <think>, answer markers, bold, "= X", bare number, trailing LaTeX) — far better for
# reasoning-model output than boxed-only. Use it everywhere we read a model's answer.
from src.data.dataset import extract_numeric_answer, answers_match
from rl_training.semantic import embed_texts, pairwise_novelty


def _content(c):
    """Normalize a completion to text (handles conversational + standard formats)."""
    if isinstance(c, str):
        return c
    if isinstance(c, list) and c and isinstance(c[-1], dict):
        return c[-1].get("content", "")
    return str(c)


def _is_correct(text: str, gold: str) -> bool:
    pred = extract_numeric_answer(text)
    if pred is None:
        return False
    try:
        return bool(answers_match(pred, gold))
    except Exception:
        return False


def correctness_reward(completions, gold_answer=None, log_metric=None, **kwargs):
    """Binary verifiable reward: 1.0 if the boxed answer matches gold, else 0.0."""
    texts = [_content(c) for c in completions]
    gold = gold_answer if gold_answer is not None else [""] * len(texts)
    rewards = [1.0 if _is_correct(t, g) else 0.0 for t, g in zip(texts, gold)]
    if log_metric:
        log_metric("correct_frac", float(np.mean(rewards)) if rewards else 0.0)
    return rewards


def make_novelty_bonus(embedding_model: str, lam: float, metric: str = "cosine",
                       correct_only: bool = True):
    """Build the group-relative novelty bonus reward fn (Component A).

    For each prompt-group, embed the rollouts, compute each rollout's mean semantic
    distance to the OTHER rollouts in the group, and return lambda*novelty. If
    correct_only, novelty is computed *among correct rollouts only* and incorrect rollouts
    get 0 — this is the key design choice: we protect rare *correct* modes, not arbitrary
    diversity (which would also reward diverse wrong answers, like a plain entropy bonus).
    """
    def novelty_bonus(prompts, completions, gold_answer=None, log_metric=None, **kwargs):
        texts = [_content(c) for c in completions]
        gold = gold_answer if gold_answer is not None else [""] * len(texts)
        n = len(texts)
        bonus = [0.0] * n

        # group indices by identical prompt (robust to batch layout)
        groups = defaultdict(list)
        for i, p in enumerate(prompts):
            key = p if isinstance(p, str) else str(p)
            groups[key].append(i)

        for _, idxs in groups.items():
            # which members count toward novelty
            if correct_only:
                members = [i for i in idxs if _is_correct(texts[i], gold[i])]
            else:
                members = list(idxs)
            if len(members) < 2:
                continue  # no peers -> no novelty bonus
            emb = embed_texts([texts[i] for i in members], embedding_model)
            nov = pairwise_novelty(emb, metric=metric)   # (len(members),)
            for j, i in enumerate(members):
                bonus[i] = lam * float(nov[j])

        if log_metric and n:
            log_metric("novelty_bonus_mean", float(np.mean(bonus)))
            log_metric("novelty_nonzero_frac", float(np.mean([b > 0 for b in bonus])))
        return bonus

    novelty_bonus.__name__ = "novelty_bonus"
    return novelty_bonus


def make_rarity_bonus(lam: float = 0.5):
    """EXPERIMENT C — RARITY-WEIGHTED CORRECTNESS ('you found the needle, don't forget it').

    GRPO's group-relative advantage drives LOW-PROBABILITY correct modes toward zero: a correct
    rollout in a group that is already mostly correct gets small advantage, while a correct rollout
    that is RARE in its group (most peers wrong) is exactly the fragile-tail solution we must
    protect. This reward adds, for each CORRECT rollout, a bonus proportional to how rare
    correctness is in its group:  bonus_i = lam * (1 - correct_frac_group) for correct i, else 0.
    A lone correct rollout among 7 wrong (cfrac=1/8) gets ~lam*0.875; a correct rollout in an
    all-correct group gets ~0. Up-weights the gradient on rare-correct modes at the reward level,
    directly counteracting mode collapse — a purely count-based alternative to the semantic
    novelty bonus (no embedding model needed)."""
    def rarity_bonus(prompts, completions, gold_answer=None, log_metric=None, **kwargs):
        texts = [_content(c) for c in completions]
        gold = gold_answer if gold_answer is not None else [""] * len(texts)
        n = len(texts)
        bonus = [0.0] * n
        correct = [_is_correct(texts[i], gold[i]) for i in range(n)]
        groups = defaultdict(list)
        for i, p in enumerate(prompts):
            groups[p if isinstance(p, str) else str(p)].append(i)
        for _, idxs in groups.items():
            cfrac = float(np.mean([correct[i] for i in idxs])) if idxs else 0.0
            for i in idxs:
                if correct[i]:
                    bonus[i] = lam * (1.0 - cfrac)   # rarer correctness -> bigger bonus
        if log_metric and n:
            log_metric("rarity_bonus_mean", float(np.mean(bonus)))
        return bonus
    rarity_bonus.__name__ = "rarity_bonus"
    return rarity_bonus


def make_coverage_reward(lam: float = 1.0):
    """METHOD 3 — COVERAGE-IN-THE-LOOP (optimize the pass@k margin, not pass@1).

    KEY SUBTLETY: a reward that is CONSTANT across a group is a NO-OP in GRPO, because the
    group-relative advantage subtracts the group mean (a flat '1[any correct]' bonus cancels
    exactly -> zero gradient). So we do NOT broadcast a constant. Instead we reward each correct
    rollout by its MARGINAL contribution to group coverage: the fewer peers also got it right,
    the more this rollout is *the* thing keeping the problem solvable, so the bigger its reward.

        reward_i = correct_i * (1 + lam / n_correct_in_group)

    - A LONE correct rollout (n_correct=1) gets 1 + lam  -> maximally protected (it alone provides
      coverage of this problem; if the policy forgets it, pass@k drops).
    - One of many correct rollouts (n_correct=8) gets 1 + lam/8 -> ~baseline (redundant; losing it
      doesn't hurt coverage).
    This is non-constant within the group (survives GRPO normalization) and its gradient points
    toward PRESERVING at-least-one-correct (the pass@k quantity), rather than maximizing the
    fraction correct (pass@1). Correctness is still required (0 for wrong), so it never rewards
    diverse-wrong. Emitted as a SEPARATE reward fn (summed by TRL) so plain correctness is logged
    too; ablate by removing it."""
    def coverage_reward(prompts, completions, gold_answer=None, log_metric=None, **kwargs):
        texts = [_content(c) for c in completions]
        gold = gold_answer if gold_answer is not None else [""] * len(texts)
        n = len(texts)
        r = [0.0] * n
        correct = [_is_correct(texts[i], gold[i]) for i in range(n)]
        groups = defaultdict(list)
        for i, p in enumerate(prompts):
            groups[p if isinstance(p, str) else str(p)].append(i)
        n_lone = 0
        for _, idxs in groups.items():
            nc = sum(correct[i] for i in idxs)
            if nc == 0:
                continue
            if nc == 1:
                n_lone += 1
            for i in idxs:
                if correct[i]:
                    r[i] = lam / nc          # marginal coverage value; lone solver -> lam, redundant -> lam/nc
        if log_metric and n:
            log_metric("coverage_reward_mean", float(np.mean(r)))
            log_metric("lone_solver_groups", float(n_lone))
        return r
    coverage_reward.__name__ = "coverage_reward"
    return coverage_reward
