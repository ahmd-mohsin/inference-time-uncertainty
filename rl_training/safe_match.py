# Timeout-guarded answer matching — HARD process-kill version.
#
# WHY: scoring a model completion runs (a) extract_numeric_answer, whose \boxed{} regexes
# CATASTROPHICALLY BACKTRACK on long adversarial output, and (b) answers_match -> sympy
# parse_expr/nsimplify, which can spin FOREVER. BOTH hang in C, where SIGALRM cannot preempt
# them (the handler only runs when control returns to Python). A prior signal+truncation guard
# still let base-model level-5 shards hang at 102% CPU for 5.5h.
#
# ROBUST FIX: do all the risky work (extract + match) in a PERSISTENT CHILD PROCESS and enforce a
# hard WALL-CLOCK timeout in the parent. On timeout the child is genuinely hung in C, so we SIGKILL
# it (survives C-level hangs, unlike SIGALRM) and respawn a fresh worker. Each match is thereby
# bounded to `timeout` seconds no matter what. A stuck/oversized/erroring result counts as WRONG.
import multiprocessing as _mp

from src.data.dataset import extract_numeric_answer, answers_match

MAX_ANSWER_LEN = 64      # a real MATH answer is short; longer extraction = runaway, not an answer
MAX_MATCH_CHARS = 2000   # answer lives at the tail; cap text fed to extraction (regex backtrack guard)

# fork (not spawn): the child inherits already-imported modules, so it does NOT re-execute the
# parent's __main__ (which for `python -m evaluate_passk` would re-init vLLM). The worker only calls
# answers_match (pure CPU/sympy) — it never touches CUDA — so forking after vLLM init is safe here.
_ctx = _mp.get_context("fork")
_worker = None
_conn = None


def _fast_prefilter(pred, gold):
    """Cheap decisions that avoid sympy entirely. Returns True/False if decided, else None."""
    if pred is None:
        return False
    ps, gs = str(pred).strip(), str(gold).strip()
    if ps == gs:
        return True
    if len(ps) > MAX_ANSWER_LEN:
        return False
    try:
        return abs(float(ps) - float(gs)) < 1e-6
    except (ValueError, TypeError):
        return None


def _worker_loop(conn):
    """Persistent child: read (pred, gold, _), send back is_correct(bool). Blocks in C on a
    pathological sympy input — fine, the parent SIGKILLs us and respawns."""
    while True:
        try:
            item = conn.recv()
        except EOFError:
            break
        if item is None:
            break
        pred, gold, _ = item
        try:
            conn.send(bool(answers_match(pred, gold)))
        except Exception:
            conn.send(False)


def _kill_worker():
    global _worker, _conn
    try:
        if _worker is not None and _worker.is_alive():
            _worker.kill()            # SIGKILL — the only thing that stops a C-level hang
            _worker.join(timeout=3)
    except Exception:
        pass
    _worker = None
    _conn = None


def _ensure_worker():
    global _worker, _conn
    if _worker is not None and _worker.is_alive():
        return
    _conn, child = _ctx.Pipe()
    _worker = _ctx.Process(target=_worker_loop, args=(child,), daemon=True)
    _worker.start()


def safe_is_correct(text, gold, timeout=5):
    """Match `text`'s answer against `gold`, bounded to `timeout` wall-clock seconds. The cheap,
    safe steps (tail-truncate, extract, exact/float prefilter) run in-process; ONLY the dangerous
    sympy match is delegated to a killable child. Returns (is_correct, pred); hang/error -> (False,None).

    NOTE: extract_numeric_answer's \\boxed{} regex can itself backtrack in C. We tail-truncate to
    MAX_MATCH_CHARS first (the documented mitigation); if that ever proves insufficient, move the
    extract into the child too. In practice truncation stops the extraction hang; the sympy match is
    the one that needed the process kill."""
    if text and len(text) > MAX_MATCH_CHARS:
        text = text[-MAX_MATCH_CHARS:]
    try:
        pred = extract_numeric_answer(text)
    except Exception:
        return (False, None)
    fast = _fast_prefilter(pred, gold)
    if fast is not None:
        return (bool(fast), pred)
    # undecided -> the sympy path, which can hang in C. Run it in the killable child, bounded.
    try:
        _ensure_worker()
        _conn.send((pred, gold, True))   # True = pred already extracted, child only does answers_match
        if _conn.poll(timeout):
            ok = _conn.recv()
            return (bool(ok), pred)
        _kill_worker()                   # hung in C -> SIGKILL, respawn next call
        return (False, pred)
    except (EOFError, BrokenPipeError, OSError, Exception):
        _kill_worker()
        return (False, pred)
