# Timeout-guarded answer matching.
#
# WHY: src.data.dataset.answers_match -> _canonical_via_sympy calls sympy parse_expr/nsimplify,
# which can hang FOREVER on a pathological expression (the try/except there catches exceptions,
# NOT infinite loops / catastrophic backtracking). A 1.5B model over tens of thousands of samples
# reliably emits such a completion — this hung a base pass@k eval for 16.5h and a grpo eval + an
# oursABC harvest similarly. Any code that scores MANY model completions must cap each match.
#
# Uses SIGALRM (main-thread, single-process only — true for our eval/harvest/prepass scripts).
import signal

from src.data.dataset import extract_numeric_answer, answers_match


class _Timeout(Exception):
    pass


def safe_is_correct(text, gold, timeout=5):
    """Extract a numeric answer from `text` and match against `gold`, guarded by a hard
    per-call timeout. A stuck parse (or any exception) counts as NOT correct."""
    def _alarm(signum, frame):
        raise _Timeout()
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        pred = extract_numeric_answer(text)
        return (pred is not None and answers_match(pred, gold)), pred
    except _Timeout:
        return False, None
    except Exception:
        return False, None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)
