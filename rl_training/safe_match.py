# Timeout-guarded answer matching.
#
# WHY: src.data.dataset.answers_match -> _canonical_via_sympy calls sympy parse_expr/nsimplify,
# which can hang FOREVER on a pathological expression (the try/except there catches exceptions,
# NOT infinite loops / catastrophic backtracking). A 1.5B model over tens of thousands of samples
# reliably emits such a completion — this hung a base pass@k eval for 16.5h and a grpo eval + an
# oursABC harvest similarly. Any code that scores MANY model completions must cap each match.
#
# Two layers of defense:
#  1. A LENGTH GUARD (primary). answers_match -> _canonical_via_sympy feeds the extracted string to
#     sympy parse_expr/nsimplify, which can spin in C for a very long time on a long/pathological
#     expression. SIGALRM CANNOT preempt that C work (the signal handler only runs when control
#     returns to Python), so a pure-signal guard is insufficient — it hung 3 eval shards for 30min+.
#     So we first reject implausibly long extracted answers (a real MATH answer is short) and only
#     then attempt the match. This is what actually stops the hang.
#  2. SIGALRM as a backstop for any remaining pure-Python slow path.
import signal

from src.data.dataset import extract_numeric_answer, answers_match

# A genuine final answer (number, fraction, short set/expr) is short. Anything longer is a runaway
# generation, not an answer — never worth feeding to sympy.
MAX_ANSWER_LEN = 64


class _Timeout(Exception):
    pass


def _fast_prefilter(pred, gold):
    """Cheap decisions that avoid sympy entirely. Returns True/False if decided, else None."""
    if pred is None:
        return False
    ps, gs = str(pred).strip(), str(gold).strip()
    if ps == gs:                       # exact string match — common, no sympy needed
        return True
    if len(ps) > MAX_ANSWER_LEN:       # runaway extraction — not a real answer, don't sympy it
        return False
    try:                               # both plain floats — decide directly
        return abs(float(ps) - float(gs)) < 1e-6
    except (ValueError, TypeError):
        return None                    # undecided -> fall through to full answers_match


def safe_is_correct(text, gold, timeout=5):
    """Extract a numeric answer from `text` and match against `gold`. Length-guarded (primary)
    plus a SIGALRM backstop. A stuck/oversized/erroring match counts as NOT correct."""
    try:
        pred = extract_numeric_answer(text)
    except Exception:
        return False, None
    fast = _fast_prefilter(pred, gold)
    if fast is not None:
        return fast, pred
    # undecided and short enough: attempt the full (sympy-capable) match under a signal backstop
    def _alarm(signum, frame):
        raise _Timeout()
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        return bool(answers_match(pred, gold)), pred
    except _Timeout:
        return False, pred
    except Exception:
        return False, pred
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)
