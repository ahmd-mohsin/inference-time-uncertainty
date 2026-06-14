"""
disagreement_miner.py
======================

Genuine, dense disagreement extraction for self-distillation on math reasoning.

THE CORE IDEA (why this is noise-free where step-clustering was not)
--------------------------------------------------------------------
Earlier consensus-decoding measured disagreement at the *step-text* level:
cluster K next-steps by `canonical_step_key`, call high cluster-spread
"disagreement". That is unsalvageably noisy, because two facts are true at once:

  * surface variation ("compute -33550/484" vs "approx -69.318") looks like
    disagreement but is the SAME fact -> false positives (the geometry blowup);
  * paraphrase forks that reconverge to the same answer are not disagreements
    at all -> more false positives (the prob-0 over-fork).

This module never clusters step text. It measures disagreement ONLY at the
quantity that normalizes cleanly: the FINAL ANSWER a branch leads to.

A reasoning prefix `p` induces a distribution over final answers:
    A(p) = { normalized_answer : count }  from K greedy rollouts conditioned on p.

A step that takes prefix p -> p' is a GENUINE, DENSE disagreement-resolution
point iff:
    (1) BEFORE the step, the answer is genuinely contested:
        >= `min_distinct` distinct answers each with support >= `min_support`
        (uncertain(A(p)) is True), AND
    (2) committing the step COLLAPSES that uncertainty:
        top answer fraction in A(p') >= `collapse_frac`
        (collapsed(A(p')) is True).

Arithmetic steps have A(p) = A(p') = {one answer} -> never (1) -> skipped.
Paraphrase forks reconverge to one answer -> never (1) -> skipped.
A single hallucinated branch has support 1 < min_support -> filtered by density.
Only steps where the model's *answer* genuinely splits and then gets decided
survive. That is the dense, concrete disagreement.

OUTPUT
------
A list of `GenuineDisagreement` records per problem: the agreed prefix, the
resolving step, the competing answers with their supports, and (if a gold
answer is supplied) which competitor was correct. These records are the unit
you feed to a teacher (Level-B contrast context) or use for calibration
(does the surviving answer == gold?).

This module is backend-agnostic. It depends only on a `Backend` with a single
`complete(prefix, n, max_new_tokens, do_sample)` method -- the HFBackend in
consensus_decoder.py already satisfies this via `generate_text` (see the
adapter in run_disagreement_mining.py).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Optional, Protocol, Tuple


# --------------------------------------------------------------------------- #
# Answer extraction / normalization. Reuse the project's canonicalizer if
# importable (sympy-backed, so 25/8 == 3.125 == \frac{25}{8} -> one bucket),
# else a safe fallback. NEVER cluster on step text -- only on these answers.
# --------------------------------------------------------------------------- #
try:
    from src.data.dataset import (  # type: ignore
        normalize_answer as _normalize_answer,
        extract_boxed_answer as _extract_boxed_answer,
        extract_numeric_answer as _extract_numeric_answer,
    )
except Exception:  # pragma: no cover - fallback for standalone use/testing
    def _extract_boxed_answer(text: str) -> Optional[str]:
        if not text:
            return None
        m = list(re.finditer(r"\\boxed\{([^{}]+)\}", text))
        return m[-1].group(1).strip() if m else None

    def _extract_numeric_answer(text: str) -> Optional[str]:
        if not text:
            return None
        nums = re.findall(r"-?\d+(?:\.\d+)?", text)
        return nums[-1] if nums else None

    def _normalize_answer(answer: Optional[str]) -> str:
        if answer is None:
            return ""
        s = str(answer).strip()
        s = s.replace(" ", "").rstrip(".")
        try:                      # collapse integer-valued floats: 204.0 -> 204
            f = float(s)
            if f.is_integer():
                return str(int(f))
            return repr(f)
        except ValueError:
            return s


# Sentinel for "this rollout produced no extractable answer". Kept DISTINCT
# from any real answer and (by default) excluded from the support that defines
# a disagreement -- a bag of blanks is not a disagreement, it's a truncation.
BLANK = "<blank>"


def extract_answer(text: str) -> str:
    """Normalized final answer of a (partial) rollout, or BLANK if none."""
    box = _extract_boxed_answer(text or "")
    cand = box if box else (_extract_numeric_answer(text or "") or "")
    norm = _normalize_answer(cand)
    return norm if norm else BLANK


# --------------------------------------------------------------------------- #
# Backend protocol. consensus_decoder.HFBackend.generate_text matches this
# (modulo a tiny adapter, provided in the runner).
# --------------------------------------------------------------------------- #
class Backend(Protocol):
    def complete(self, prefix: str, n: int, max_new_tokens: int,
                 do_sample: bool = True,
                 temperature: Optional[float] = None) -> List[str]:
        """Return `n` continuations of `prefix` (text after the prefix)."""
        ...


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
@dataclass
class AnswerDist:
    """Distribution over final answers induced by a prefix, from K rollouts."""
    counts: Counter = field(default_factory=Counter)   # normalized answer -> n
    k: int = 0                                          # total rollouts drawn

    # ---- views that EXCLUDE blanks (blanks are truncation, not opinion) ---- #
    def supported(self, min_support: int) -> Dict[str, int]:
        return {a: c for a, c in self.counts.items()
                if a != BLANK and c >= min_support}

    def n_answered(self) -> int:
        return sum(c for a, c in self.counts.items() if a != BLANK)

    def n_blank(self) -> int:
        return self.counts.get(BLANK, 0)

    def top(self) -> Tuple[Optional[str], int]:
        real = [(a, c) for a, c in self.counts.items() if a != BLANK]
        if not real:
            return None, 0
        a, c = max(real, key=lambda kv: kv[1])
        return a, c

    def top_fraction(self) -> float:
        """Fraction of NON-BLANK rollouts that agree on the top answer."""
        n = self.n_answered()
        if n == 0:
            return 0.0
        return self.top()[1] / n

    def entropy(self) -> float:
        """Shannon entropy (bits) over non-blank answers. 0 == unanimous."""
        n = self.n_answered()
        if n == 0:
            return 0.0
        h = 0.0
        for a, c in self.counts.items():
            if a == BLANK or c == 0:
                continue
            p = c / n
            h -= p * math.log2(p)
        return h

    def as_dict(self) -> dict:
        return {"counts": dict(self.counts), "k": self.k,
                "n_answered": self.n_answered(), "n_blank": self.n_blank(),
                "top_fraction": round(self.top_fraction(), 4),
                "entropy": round(self.entropy(), 4)}


@dataclass
class GenuineDisagreement:
    """One dense, concrete, answer-divergent fork."""
    problem_id: object
    anchor_index: int                 # which segment boundary this fork sits at
    agreed_prefix: str                # reasoning everyone shares up to the fork
    resolving_step: str               # the segment that collapsed the split
    dist_before: dict                 # AnswerDist.as_dict() BEFORE the step
    dist_after: dict                  # AnswerDist.as_dict() AFTER the step
    competing_answers: List[Tuple[str, int]]  # [(answer, support)] sorted desc
    resolved_to: Optional[str]        # answer the population collapsed onto
    # --- populated only when a gold answer is supplied (calibration) -------- #
    gold_answer: Optional[str] = None
    resolved_correctly: Optional[bool] = None   # resolved_to == gold ?
    gold_was_in_contention: Optional[bool] = None  # gold among competitors ?

    def as_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class MinerConfig:
    # --- answer-distribution sampling --- #
    k_rollouts: int = 6           # greedy-ish rollouts per anchor prefix
    rollout_max_tokens: int = 4096
    rollout_temperature: float = 0.7   # need diversity to SEE the split
    # --- anchoring (segmentation quality is irrelevant; see module docstring) #
    max_anchors: int = 10         # cap forks evaluated per problem (cost)
    min_segment_chars: int = 120  # don't anchor on tiny fragments
    # --- the density / concreteness filters (THE noise gate) --- #
    min_support: int = 2          # an answer must appear >= this to "count"
    min_distinct_before: int = 2  # >= this many supported answers == contested
    collapse_frac: float = 0.75   # after-step top fraction to call it resolved
    max_blank_frac: float = 0.5   # skip anchors whose bag is mostly truncation
    # --- base trajectory --- #
    base_max_tokens: int = 8192


# --------------------------------------------------------------------------- #
# Core mechanics
# --------------------------------------------------------------------------- #
def answer_distribution(backend: Backend, prefix: str, cfg: MinerConfig) -> AnswerDist:
    """Roll forward K times from `prefix` and bucket by FINAL ANSWER.

    This is the only place the model is queried for disagreement. Note we
    sample (temperature>0) so that a genuine split is actually visible; a
    purely greedy rollout would hide it.
    """
    outs = backend.complete(prefix, n=cfg.k_rollouts,
                            max_new_tokens=cfg.rollout_max_tokens,
                            do_sample=True, temperature=cfg.rollout_temperature)
    counts = Counter(extract_answer(o) for o in outs)
    return AnswerDist(counts=counts, k=cfg.k_rollouts)


def find_anchors(prompt: str, base_trajectory: str, cfg: MinerConfig) -> List[str]:
    """Cumulative prefixes at coarse segment boundaries of the base solution.

    IMPORTANT: anchor placement does NOT need to be clean. We are not
    clustering these segments; they only decide *where* we probe the answer
    distribution. A crude split is fine -- the answer-divergence test does all
    the discriminative work.
    """
    # Split on blank lines first; fall back to single newlines; then merge
    # tiny fragments so each segment is a meaningful chunk of reasoning.
    raw = re.split(r"\n\s*\n", base_trajectory)
    if len(raw) < 3:
        raw = base_trajectory.split("\n")
    segments: List[str] = []
    buf = ""
    for seg in raw:
        buf = (buf + "\n" + seg) if buf else seg
        if len(buf) >= cfg.min_segment_chars:
            segments.append(buf)
            buf = ""
    if buf:
        segments.append(buf)

    # Build cumulative prefixes: prompt, prompt+seg0, prompt+seg0+seg1, ...
    prefixes = [prompt]
    acc = prompt
    for seg in segments:
        acc = acc + "\n" + seg
        prefixes.append(acc)

    # Subsample to <= max_anchors evenly (keeps cost bounded on long solutions).
    if len(prefixes) > cfg.max_anchors:
        idx = [round(i * (len(prefixes) - 1) / (cfg.max_anchors - 1))
               for i in range(cfg.max_anchors)]
        # dedupe while preserving order
        seen, keep = set(), []
        for i in idx:
            if i not in seen:
                seen.add(i)
                keep.append(i)
        prefixes = [prefixes[i] for i in keep]
    return prefixes


def is_contested(dist: AnswerDist, cfg: MinerConfig) -> bool:
    """BEFORE-the-step gate: the answer is genuinely split."""
    if dist.n_answered() == 0:
        return False
    if dist.n_blank() / max(dist.k, 1) > cfg.max_blank_frac:
        return False  # mostly truncation -> not a disagreement, a token-budget bug
    supported = dist.supported(cfg.min_support)
    return len(supported) >= cfg.min_distinct_before


def is_resolved(dist: AnswerDist, cfg: MinerConfig) -> bool:
    """AFTER-the-step gate: the split collapsed onto one answer."""
    if dist.n_answered() == 0:
        return False
    return dist.top_fraction() >= cfg.collapse_frac


def _step_text(agreed_prefix: str, next_prefix: str) -> str:
    """The reasoning segment added between two consecutive anchor prefixes."""
    if next_prefix.startswith(agreed_prefix):
        return next_prefix[len(agreed_prefix):].strip()
    return next_prefix.strip()


# --------------------------------------------------------------------------- #
# Top-level entry point
# --------------------------------------------------------------------------- #
def mine_problem(backend: Backend,
                 problem_id: object,
                 prompt: str,
                 cfg: MinerConfig = MinerConfig(),
                 gold_answer: Optional[str] = None,
                 base_trajectory: Optional[str] = None
                 ) -> Tuple[List[GenuineDisagreement], dict]:
    """Mine all dense, concrete, answer-divergent forks for one problem.

    Returns (list_of_genuine_disagreements, telemetry). `gold_answer` is used
    ONLY for calibration labels on the records -- it is never used to decide
    whether a fork is genuine, so the extraction itself is fully teacher-free.
    """
    norm_gold = _normalize_answer(gold_answer) if gold_answer is not None else None

    # 1. One base trajectory just to lay down anchor points.
    if base_trajectory is None:
        base = backend.complete(prompt, n=1, max_new_tokens=cfg.base_max_tokens,
                                do_sample=False)
        base_trajectory = base[0] if base else ""

    # 2. Anchor prefixes.
    prefixes = find_anchors(prompt, base_trajectory, cfg)

    # 3. Answer distribution at each anchor (each computed once; consecutive
    #    forks share, so this is len(prefixes) rollout-batches, not 2x).
    dists = [answer_distribution(backend, p, cfg) for p in prefixes]

    # 4. A fork between anchor i and i+1 is genuine iff contested(before) and
    #    resolved(after).
    found: List[GenuineDisagreement] = []
    for i in range(len(prefixes) - 1):
        before, after = dists[i], dists[i + 1]
        if not is_contested(before, cfg):
            continue
        if not is_resolved(after, cfg):
            continue

        resolved_to, _ = after.top()
        competitors = sorted(before.supported(cfg.min_support).items(),
                             key=lambda kv: kv[1], reverse=True)

        rec = GenuineDisagreement(
            problem_id=problem_id,
            anchor_index=i,
            agreed_prefix=prefixes[i],
            resolving_step=_step_text(prefixes[i], prefixes[i + 1]),
            dist_before=before.as_dict(),
            dist_after=after.as_dict(),
            competing_answers=competitors,
            resolved_to=resolved_to,
        )
        if norm_gold is not None:
            rec.gold_answer = norm_gold
            rec.resolved_correctly = (resolved_to == norm_gold)
            rec.gold_was_in_contention = norm_gold in dict(competitors)
        found.append(rec)

    telemetry = {
        "problem_id": problem_id,
        "n_anchors": len(prefixes),
        "n_genuine_forks": len(found),
        "anchor_top_fractions": [round(d.top_fraction(), 3) for d in dists],
        "anchor_entropies": [round(d.entropy(), 3) for d in dists],
        "anchor_blank_fracs": [round(d.n_blank() / max(d.k, 1), 3) for d in dists],
    }
    return found, telemetry


# --------------------------------------------------------------------------- #
# Teacher-context builder (Level-B contrast). Optional, kept here so the unit
# of extraction and the unit of supervision live together.
# --------------------------------------------------------------------------- #
def build_contrast_context(rec: GenuineDisagreement) -> str:
    """Turn a genuine fork into privileged context for a teacher forward pass.

    This conditions the teacher on the DISAGREEMENT and its resolution -- NOT
    on the gold answer. That is the whole point: it is the signal available
    when no ground truth exists.
    """
    comp = " or ".join(f"{a} (supported by {c} rollouts)"
                       for a, c in rec.competing_answers)
    return (
        f"{rec.agreed_prefix}\n\n"
        f"[At this point the reasoning genuinely diverges: it could lead to "
        f"{comp}. The lines of reasoning that survived scrutiny converge on "
        f"{rec.resolved_to}. Continue the solution from here, committing to "
        f"that resolution.]\n\n"
        f"{rec.resolving_step}"
    )