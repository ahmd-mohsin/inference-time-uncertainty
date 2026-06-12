# consensus_decoder.py
#
# Step-synchronous consensus decoding (teacher-free, on-policy).
#
# One COMMITTED PREFIX C grows step by step. At each step we:
#   (1) BRANCH       : sample K one-step continuations from C            [doc step 1]
#   (2) CLUSTER+D    : group by SEMANTIC equivalence, measure disagreement D   [step 2]
#   (3) DECIDE       : low D -> commit consensus; high D -> resolve      [step 3]
#   (4) RECONVERGE?  : before paying to resolve, short greedy lookahead --
#                       benign forks (reconverge / dead-end) are handled free
#   (5) RESOLVE      : verify (sympy/model) > adjudicate (J jurors) > prune  [step 4]
#   (6) APPEND       : commit the winning step to C, repeat until \boxed  [step 5]
#
# The committed trajectory IS the answer AND a verified trace. Every step also
# records the data needed to LATER choose a training mode -- no weights are
# updated here:
#   * committed trace             -> Self-distillation SFT      (option 1)
#   * per-step consensus dist p    -> step-level KL targets       (option 2)
#   * winning vs losing steps      -> preference pairs for DPO/RL  (option 3)
# Use extract_sft_examples / extract_kl_targets / extract_dpo_pairs to harvest.
#
# The model is injected as a `step_model` (duck-typed) so the control flow is
# unit-testable without a real LLM. Use StepModel.from_hf(model, tokenizer, cfg)
# in production; a scripted mock is used in the tests.

import logging
import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Optional, Callable

logger = logging.getLogger(__name__)

# Direct imports from the project's dataset module (no fallback). These must
# resolve -- the decoder relies on the SAME answer machinery as the baselines.
from src.data.dataset import (
    normalize_answer as _normalize_answer,
    extract_numeric_answer as _extract_numeric_answer,
    extract_boxed_answer as _extract_boxed_answer,
)

try:
    import sympy
    from sympy.parsing.sympy_parser import parse_expr
except Exception:  # pragma: no cover
    sympy = None


# ======================================================================
# SECTION 1 -- semantic step equivalence (THE crux: over-fork vs over-merge)
# ======================================================================

_FILLER = {
    "so", "thus", "therefore", "hence", "then", "now", "let", "lets", "let's",
    "we", "have", "get", "got", "is", "are", "the", "a", "an", "me", "check",
    "see", "if", "this", "that", "which", "and", "to", "of", "it", "be", "can",
    "wait", "okay", "ok", "actually", "since", "as", "by", "using", "from",
    "step", "next", "first", "compute", "computing", "find", "finding", "value",
}


def _clean_math(s: str) -> str:
    """Light LaTeX -> sympy-parseable cleanup."""
    s = s.strip().strip("$")
    s = s.replace("\\left", "").replace("\\right", "").replace("\\,", " ")
    s = s.replace("\\cdot", "*").replace("\\times", "*").replace("\\div", "/")
    s = re.sub(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"((\1)/(\2))", s)
    s = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"sqrt(\1)", s)
    s = s.replace("\\pi", "pi")
    s = s.replace("^", "**")
    s = s.replace("{", "(").replace("}", ")")
    s = re.sub(r"\\[a-zA-Z]+", "", s)        # drop any remaining \commands
    s = s.replace("\u2212", "-")             # unicode minus
    # implicit multiplication: 17k -> 17*k, 2( -> 2*(, )x -> )*x, kx stays
    s = re.sub(r"(\d)([a-zA-Z(])", r"\1*\2", s)
    s = re.sub(r"(\))([a-zA-Z0-9(])", r"\1*\2", s)
    return s


def _sympy_equation_key(step: str) -> Optional[str]:
    """If the step asserts an equation A = B, return a sign-canonical key for
    (A - B) so that 'm = 2+17k', 'm = 17k+2', and '17k+2 = m' all collapse."""
    if sympy is None:
        return None
    # drop a leading filler clause ("set", "so we get", "thus") before the eqn
    step = re.sub(r"^\s*(set|so|thus|therefore|then|we get|we have|let)\b[:,]?\s*",
                  "", step.strip(), flags=re.IGNORECASE)
    # take the FIRST top-level equation in the step
    m = re.search(r"([^=\n]+)=([^=\n]+)", step)
    if not m:
        return None
    lhs, rhs = _clean_math(m.group(1)), _clean_math(m.group(2))
    try:
        loc = {"sqrt": sympy.sqrt, "pi": sympy.pi}
        diff = sympy.simplify(parse_expr(lhs, local_dict=loc)
                              - parse_expr(rhs, local_dict=loc))
    except Exception:
        return None
    if diff == 0:
        return "EQ:0"
    try:
        a, b = sympy.srepr(diff), sympy.srepr(sympy.simplify(-diff))
        return "EQ:" + (a if a <= b else b)     # sign-canonical
    except Exception:
        return None


def _text_key(step: str) -> str:
    """Fallback key: numbers present + content words (filler removed). Collapses
    '80. let me check it' and '80' toward each other via their shared number,
    but is admittedly weaker than the model-equivalence hook below."""
    nums = sorted(set(re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", step)))
    words = re.findall(r"[a-zA-Z]+", step.lower())
    words = sorted({w for w in words if w not in _FILLER and len(w) > 1})
    if nums:                      # number-dominated step -> key mainly on numbers
        return "N:" + ",".join(nums) + ("|" + ",".join(words[:2]) if words else "")
    return "T:" + " ".join(words)


def canonical_step_key(step: str) -> str:
    """Semantic key for clustering. Prefers symbolic equation canonicalization,
    falls back to a normalized-text key."""
    step = (step or "").strip()
    if not step:
        return "EMPTY"
    eq = _sympy_equation_key(step)
    if eq is not None:
        return eq
    return _text_key(step)


def steps_equivalent(a: str, b: str,
                     equivalence_fn: Optional[Callable[[str, str], bool]] = None) -> bool:
    if equivalence_fn is not None:
        try:
            return bool(equivalence_fn(a, b))
        except Exception:
            pass
    return canonical_step_key(a) == canonical_step_key(b)


# ======================================================================
# SECTION 2 -- clustering + disagreement measure D
# ======================================================================

@dataclass
class Cluster:
    key: str
    members: list = field(default_factory=list)   # indices into candidates
    representative: str = ""                       # medoid text (shortest = cleanest)

    @property
    def size(self) -> int:
        return len(self.members)


def cluster_steps(candidates,
                  equivalence_fn: Optional[Callable[[str, str], bool]] = None):
    """Group candidate steps. With an equivalence_fn (e.g. a cheap model call)
    we agglomerate by pairwise equivalence; otherwise we bucket by canonical key."""
    clusters: list[Cluster] = []
    for i, step in enumerate(candidates):
        placed = False
        for c in clusters:
            if steps_equivalent(step, candidates[c.members[0]], equivalence_fn):
                c.members.append(i)
                # medoid = shortest member (least filler, cleanest assertion)
                if len(step) < len(c.representative):
                    c.representative = step
                placed = True
                break
        if not placed:
            clusters.append(Cluster(key=canonical_step_key(step),
                                    members=[i], representative=step))
    clusters.sort(key=lambda c: c.size, reverse=True)
    return clusters


def disagreement(clusters, n):
    """Return (D, top_fraction, entropy, distribution{key->frac})."""
    if n == 0 or not clusters:
        return 1.0, 0.0, 0.0, {}
    dist = {c.representative: c.size / n for c in clusters}
    top = clusters[0].size / n
    H = -sum(p * math.log(p) for p in dist.values() if p > 0)
    return 1.0 - top, top, H, dist


# ======================================================================
# SECTION 3 -- lightweight semantics for the reconvergence lookahead
# ======================================================================

_CONTRA_MARKERS = (
    "contradiction", "no solution", "impossible", "cannot be", "can't be",
    "not possible", "undefined", "does not exist",
)


def _to_float(tok: str):
    try:
        if "/" in tok:
            a, b = tok.split("/")
            return float(a) / float(b)
        return float(tok)
    except Exception:
        return None


def detect_contradiction(text: str) -> bool:
    """Cheap dead-end detector. Catches explicit markers AND trig-range
    violations like the torus chain's 'sin theta = -19/6'."""
    t = (text or "").lower()
    if any(m in t for m in _CONTRA_MARKERS):
        return True
    for m in re.finditer(r"(sin|cos)\b[^=]{0,15}=\s*(-?\d+(?:\.\d+)?(?:/\d+)?)", t):
        v = _to_float(m.group(2))
        if v is not None and abs(v) > 1.0 + 1e-9:
            return True
    # negative discriminant phrased as sqrt of a negative number
    for m in re.finditer(r"sqrt\(\s*(-\d+(?:\.\d+)?)\s*\)", t):
        return True
    return False


def emerging_answer(text: str):
    """Best-effort 'where is this branch heading' signal for the lookahead."""
    if not text:
        return None
    box = _extract_boxed_answer(text)
    cand = box if box else (_extract_numeric_answer(text) or "")
    if not cand:
        return None
    try:
        return _normalize_answer(cand)
    except Exception:
        return str(cand).strip()


def has_boxed(text: str) -> bool:
    return bool(_extract_boxed_answer(text or ""))


# ======================================================================
# SECTION 4 -- production model wrapper (HF) + the StepModel interface
# ======================================================================

class HFBackend:
    """Thin wrapper around an HF model+tokenizer, mirroring DADGenerator's
    sampling. Returns generated TEXT (one step at a time)."""

    def __init__(self, model, tokenizer, cfg: dict):
        import torch
        self.torch = torch
        self.model = model
        self.tokenizer = tokenizer
        self.device = cfg["model"]["device"]
        dad = cfg.get("dad", {})
        self.temperature = dad.get("temperature", 0.7)
        self.top_p = dad.get("top_p", 0.95)
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        self.eos_token_id = tokenizer.eos_token_id

    def generate_text(self, prefix_text, n, max_new_tokens,
                      do_sample=True, temperature=None):
        """Generate `n` continuations of `prefix_text`. Sampling draws all n in a
        SINGLE batched call (num_return_sequences=n) -- this is the difference
        between tractable and hopeless on a step-synchronous decoder, since we
        regenerate from the growing prefix many times."""
        torch = self.torch
        enc = self.tokenizer(prefix_text, return_tensors="pt",
                             truncation=True, max_length=16384)
        ids = enc["input_ids"].to(self.device)
        attn = enc.get("attention_mask")
        kwargs = dict(
            input_ids=ids,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
        )
        if attn is not None:
            kwargs["attention_mask"] = attn.to(self.device)
        if do_sample:
            kwargs.update(do_sample=True, top_p=self.top_p,
                          temperature=(temperature if temperature is not None
                                       else self.temperature),
                          num_return_sequences=n)
        else:
            kwargs.update(do_sample=False, num_return_sequences=1)
        with torch.no_grad():
            out = self.model.generate(**kwargs)
        plen = ids.shape[1]
        return [self.tokenizer.decode(out[i, plen:], skip_special_tokens=True)
                for i in range(out.shape[0])]


def _first_step(text: str, max_chars: int = 1200) -> str:
    """Truncate a generation to a single reasoning step (first \\n\\n boundary,
    or a boxed answer, whichever comes first)."""
    text = text.lstrip("\n")
    box = re.search(r"\\boxed\{[^{}]+\}", text)
    para = text.find("\n\n")
    cut = len(text)
    if para != -1:
        cut = min(cut, para)
    if box:
        cut = min(cut, box.end())  # keep the box if it's the first thing
    step = text[:cut].strip()
    return step[:max_chars] if step else text[:max_chars].strip()


class StepModel:
    """Duck-typed interface the decoder depends on:
        sample_steps(prefix, n)           -> list[str]   (K one-step branches)
        greedy_rollout(prefix, step, h)   -> str         (lookahead continuation)
        adjudicate(prefix, cands, j)      -> int         (winning candidate index)
        verify_step(prefix, step)         -> Optional[bool]
    """

    def __init__(self, backend, cfg: dict):
        self.backend = backend
        dad = cfg.get("dad", {})
        self.step_max_tokens = dad.get("step_max_tokens", 256)
        self.problem_template = dad.get("system_prompt",
            "Solve the problem step by step. Put your final answer in \\boxed{}.")

    @classmethod
    def from_hf(cls, model, tokenizer, cfg):
        return cls(HFBackend(model, tokenizer, cfg), cfg)

    def sample_steps(self, prefix, n):
        raw = self.backend.generate_text(prefix, n, self.step_max_tokens,
                                         do_sample=True)
        return [_first_step(t) for t in raw]

    def greedy_rollout(self, prefix, step, h):
        cont = prefix + "\n" + step + "\n"
        out = self.backend.generate_text(cont, 1, self.step_max_tokens * h,
                                        do_sample=False)
        return out[0] if out else ""

    def adjudicate(self, prefix, cands, j):
        opts = "\n".join(f"[{i}] {c}" for i, c in enumerate(cands))
        prompt = (
            f"{prefix}\n\nTwo proposed NEXT steps conflict:\n{opts}\n\n"
            f"They cannot both be correct. Reason independently, then end your "
            f"reply with exactly 'CHOICE: <index>'.\n"
        )
        votes = Counter()
        outs = self.backend.generate_text(prompt, j, self.step_max_tokens * 3,
                                         do_sample=True)
        for o in outs:
            m = re.search(r"CHOICE:\s*\[?(\d+)\]?", o)
            if m and int(m.group(1)) < len(cands):
                votes[int(m.group(1))] += 1
            else:                      # fallback: whose emerging answer appears?
                for i, c in enumerate(cands):
                    ea = emerging_answer(c)
                    if ea and ea in o:
                        votes[i] += 1
                        break
        return votes.most_common(1)[0][0] if votes else 0

    def verify_step(self, prefix, step):
        """Return True/False if cheaply verifiable (numeric equation via sympy),
        else None to signal 'not locally verifiable'."""
        if sympy is not None:
            m = re.search(r"([^=\n]+)=([^=\n]+)", step)
            if m:
                try:
                    loc = {"sqrt": sympy.sqrt, "pi": sympy.pi}
                    lhs = parse_expr(_clean_math(m.group(1)), local_dict=loc)
                    rhs = parse_expr(_clean_math(m.group(2)), local_dict=loc)
                    if not (lhs.free_symbols or rhs.free_symbols):
                        return bool(sympy.simplify(lhs - rhs) == 0)
                except Exception:
                    pass
        return None


# ======================================================================
# SECTION 5 -- trajectory records (carry ALL downstream training signal)
# ======================================================================

@dataclass
class StepRecord:
    idx: int
    prefix: str                      # committed C BEFORE this step (the conditioning)
    candidates: list                 # raw K branches
    distribution: dict               # {representative_step: fraction}  -> KL target
    D: float
    top_fraction: float
    entropy: float
    decision: str                    # commit | benign_commit | resolved | pruned_to
    resolution_method: str           # none | verify | adjudicate | dead_end_prune
    committed_step: str
    losing_steps: list               # -> preference pairs (chosen=committed)
    process_reward: float            # consensus frac / verifier pass / juror margin
    lookahead: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass
class TrajectoryRecord:
    problem: str
    final_answer: Optional[str]
    committed_trajectory: str
    steps: list
    stop_reason: str
    n_steps: int
    n_model_calls: int
    wall_time_sec: float

    def to_dict(self):
        d = asdict(self)
        return d


# ---- the three later-training extractors (no weight updates here) ----

def extract_sft_examples(traj: TrajectoryRecord):
    """Option 1 -- self-distillation SFT: learn the committed trace, and each
    committed step from its own prefix."""
    out = [{"prompt": traj.problem, "completion": traj.committed_trajectory}]
    for s in traj.steps:
        out.append({"prompt": s.prefix, "completion": s.committed_step})
    return out


def extract_kl_targets(traj: TrajectoryRecord):
    """Option 2 -- step-level KL: soft consensus distribution over candidate
    steps at each fork (teacher-free target = your own branch consensus)."""
    targets = []
    for s in traj.steps:
        if len(s.distribution) > 1:          # only forks carry signal
            targets.append({"prefix": s.prefix,
                            "targets": s.distribution,
                            "committed": s.committed_step})
    return targets


def extract_dpo_pairs(traj: TrajectoryRecord):
    """Option 3 -- preference pairs for DPO / online preference RL:
    (chosen = committed step) > (rejected = a resolved-away / pruned step)."""
    pairs = []
    for s in traj.steps:
        for losing in s.losing_steps:
            if losing and losing.strip() and losing != s.committed_step:
                pairs.append({"prefix": s.prefix,
                              "chosen": s.committed_step,
                              "rejected": losing,
                              "source": s.resolution_method})
    return pairs


# ======================================================================
# SECTION 6 -- the decoder
# ======================================================================

class ConsensusDecoder:
    def __init__(self, step_model, cfg: dict, equivalence_fn=None):
        self.sm = step_model
        self.equiv = equivalence_fn
        dad = cfg.get("dad", {})
        self.K = dad.get("branch_k", 4)             # branches per step
        self.tau = dad.get("commit_tau", 0.75)      # consensus fraction to commit
        self.J = dad.get("adjudicators", 3)         # jurors for resolution
        self.h = dad.get("lookahead_steps", 2)      # reconvergence horizon
        self.max_steps = dad.get("max_steps", 64)

    # ------------------------------------------------------------------
    def generate(self, problem_text: str) -> TrajectoryRecord:
        t0 = time.time()
        C = problem_text.rstrip() + "\n"
        steps: list[StepRecord] = []
        calls = 0
        stop_reason = "max_steps"

        for idx in range(self.max_steps):
            # ---- (1) BRANCH ----
            cands = self.sm.sample_steps(C, self.K)
            calls += self.K
            cands = [c for c in cands if c is not None]
            if not cands:
                stop_reason = "no_candidates"
                break

            # ---- (2) CLUSTER + measure D ----
            clusters = cluster_steps(cands, self.equiv)
            D, top, H, dist = disagreement(clusters, len(cands))

            committed, decision, method, losing, reward, look = \
                self._decide(C, cands, clusters, top)
            calls += look.get("model_calls", 0)

            steps.append(StepRecord(
                idx=idx, prefix=C, candidates=cands, distribution=dist,
                D=D, top_fraction=top, entropy=H, decision=decision,
                resolution_method=method, committed_step=committed,
                losing_steps=losing, process_reward=reward, lookahead=look,
            ))

            if committed is None:           # everything pruned, nothing to commit
                stop_reason = "all_pruned"
                break

            # ---- (6) APPEND ----
            C = C + committed.strip() + "\n"
            logger.info(f"step {idx}: D={D:.2f} top={top:.2f} "
                        f"decision={decision}/{method} -> {committed[:60]!r}")

            if has_boxed(committed):
                stop_reason = "boxed"
                break

        return TrajectoryRecord(
            problem=problem_text,
            final_answer=emerging_answer(C),
            committed_trajectory=C,
            steps=steps,
            stop_reason=stop_reason,
            n_steps=len(steps),
            n_model_calls=calls,
            wall_time_sec=time.time() - t0,
        )

    # ------------------------------------------------------------------
    def _decide(self, C, cands, clusters, top):
        """Returns (committed_step, decision, method, losing_steps, reward, look)."""
        # ---- (3) low D: commit consensus, no extra compute ----
        if top >= self.tau:
            return clusters[0].representative, "commit", "none", [], top, {}

        # ---- (4) high D: reconvergence lookahead BEFORE paying to resolve ----
        reps = [c.representative for c in clusters[:2]]
        look = {"reps": reps, "downstream": [], "model_calls": 0}
        downstream_ans, alive = [], []
        for rep in reps:
            if detect_contradiction(rep):           # dead end at the step itself
                downstream_ans.append(("DEAD", rep))
                look["downstream"].append({"rep": rep, "verdict": "dead_end"})
                continue
            cont = self.sm.greedy_rollout(C, rep, self.h)
            look["model_calls"] += 1
            if detect_contradiction(cont):
                downstream_ans.append(("DEAD", rep))
                look["downstream"].append({"rep": rep, "verdict": "dead_end"})
            else:
                ea = emerging_answer(cont)
                downstream_ans.append((ea, rep))
                alive.append((ea, rep))
                look["downstream"].append({"rep": rep, "verdict": "alive",
                                           "emerging": ea})

        # all but one branch dead -> prune to the survivor (free resolution)
        if len(alive) == 1:
            survivor = alive[0][1]
            losers = [r for (_, r) in downstream_ans if r != survivor]
            return survivor, "pruned_to", "dead_end_prune", losers, 1.0, look

        if len(alive) == 0:
            return None, "pruned_to", "dead_end_prune", reps, 0.0, look

        # benign fork: the surviving branches reconverge to the SAME answer ->
        # answer-irrelevant (zero leverage). Commit the cleaner one, don't resolve.
        eas = [ea for (ea, _) in alive if ea is not None]
        if len(eas) >= 2 and len(set(eas)) == 1:
            cleaner = min((r for (_, r) in alive), key=len)
            losers = [r for (_, r) in alive if r != cleaner]
            return cleaner, "benign_commit", "reconverge", losers, top, look

        # ---- (5) load-bearing fork: resolve ----
        # 5a. verifiable step? commit the one that checks out.
        verdicts = []
        for (_, rep) in alive:
            v = self.sm.verify_step(C, rep)
            if v is not None:
                look["model_calls"] += 0      # sympy path is free; model path counted in sm
            verdicts.append(v)
        passed = [rep for (v, (_, rep)) in zip(verdicts, alive) if v is True]
        if len(passed) == 1:
            losers = [r for (_, r) in alive if r != passed[0]]
            return passed[0], "resolved", "verify", losers, 1.0, look

        # 5b. adjudicate with J jurors.
        cand_reps = [r for (_, r) in alive]
        win = self.sm.adjudicate(C, cand_reps, self.J)
        look["model_calls"] += self.J
        win = max(0, min(win, len(cand_reps) - 1))
        winner = cand_reps[win]
        losers = [r for r in cand_reps if r != winner]
        # juror-margin as the process reward
        return winner, "resolved", "adjudicate", losers, top, look


# ======================================================================
# convenience: decode + serialize to a JSONL-ready dict
# ======================================================================

def decode_problem(step_model, cfg, problem_text, equivalence_fn=None):
    dec = ConsensusDecoder(step_model, cfg, equivalence_fn)
    traj = dec.generate(problem_text)
    return traj, {
        "problem": traj.problem,
        "final_answer": traj.final_answer,
        "stop_reason": traj.stop_reason,
        "n_steps": traj.n_steps,
        "n_model_calls": traj.n_model_calls,
        "committed_trajectory": traj.committed_trajectory,
        "steps": [s.to_dict() for s in traj.steps],
        "sft_examples": extract_sft_examples(traj),
        "kl_targets": extract_kl_targets(traj),
        "dpo_pairs": extract_dpo_pairs(traj),
    }