# Failure Certificate extractor (Forget-to-Repair, the (E_rich, ∅) condition). Given a FAILED program +
# its unit tests, produce a concrete counterexample — a failing (input → expected vs got) — that carries
# rich failure EVIDENCE while containing NONE of the failed program itself. Feeding C(y-) forward (code
# hidden) tests whether informative failure feedback survives proposal-erasure.
import ast, signal, contextlib

class _CertTimeout(Exception):
    pass

def _alarm_handler(signum, frame):
    raise _CertTimeout()

@contextlib.contextmanager
def _time_limit(seconds):
    """Hard wall-clock limit for exec/eval of untrusted candidate code (mbpp/HE solutions can infinite-loop
    and would otherwise hang an eval shard forever). Uses SIGALRM — only effective in the main thread; if
    unavailable (e.g. called from a worker thread) it degrades to no-limit rather than crashing."""
    try:
        old = signal.signal(signal.SIGALRM, _alarm_handler)
    except (ValueError, AttributeError):
        yield  # not main thread / no SIGALRM — run without the guard
        return
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)

def _assert_pairs(test_src):
    """(lhs_src, rhs_src) for every top-level `assert LHS == RHS` in the test source (mbpp test_list
    joined, or a HumanEval check() body)."""
    pairs = []
    try:
        tree = ast.parse(test_src)
    except Exception:
        return pairs
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare) and len(node.test.ops) == 1:
            if not isinstance(node.test.ops[0], ast.Eq):
                continue
            lhs = ast.get_source_segment(test_src, node.test.left)
            rhs = ast.get_source_segment(test_src, node.test.comparators[0])
            if lhs and rhs:
                pairs.append((lhs, rhs))
    return pairs

def extract_counterexample(code, item, max_check=300):
    """Return the RAW pieces {input, got, expected} of the first failing case (for info-density levels +
    corruption controls), or None if it passes."""
    ns = {}
    try:
        with _time_limit(5):
            exec(code, ns)
    except (Exception, _CertTimeout) as e:
        return {"input": "<program did not run>", "got": f"<{type(e).__name__}>", "expected": None}
    entry = item.get("entry")
    if entry and entry in ns:
        ns.setdefault("candidate", ns[entry])
    for lhs, rhs in _assert_pairs(item.get("test", ""))[:max_check]:
        try:
            with _time_limit(5):
                got = eval(lhs, ns)
        except Exception as e:
            return {"input": lhs.strip()[:160], "got": f"<raised {type(e).__name__}>", "expected": rhs.strip()[:80]}
        try:
            exp = eval(rhs, ns)
        except Exception:
            exp = None
        if exp is None or got != exp:
            return {"input": lhs.strip()[:160], "got": repr(got)[:120], "expected": rhs.strip()[:80]}
    return None

def cert_string(ce, level):
    """Information-density ladder C0..C4 from a counterexample dict."""
    if ce is None: return None
    if level == 0: return "A previous attempt was incorrect. Write a fresh solution in a ```python block."
    if level == 1: return "A previous attempt failed the tests. Diagnose and write a fresh solution in a ```python block."
    if level == 2: return f"A previous attempt failed on the call `{ce['input']}`. Write a fresh correct solution in a ```python block."
    if level == 3: return f"On the call `{ce['input']}` a previous attempt produced `{ce['got']}`. Write a fresh correct solution in a ```python block."
    return (f"Concrete counterexample: the call `{ce['input']}` returned `{ce['got']}` but the correct answer is "
            f"`{ce['expected']}`. Write a fresh, correct solution in a ```python block.")

def make_certificate(code, item, max_check=300):
    """Return a natural-language Failure Certificate string (no failed code), or None if the program
    actually passes / no counterexample can be isolated."""
    ns = {}
    try:
        with _time_limit(5):
            exec(code, ns)
    except _CertTimeout:
        return ("The submitted solution timed out (likely an infinite loop or far-too-slow algorithm). "
                "Rethink the algorithmic complexity, then write a fresh correct solution.")
    except Exception as e:
        return (f"The submitted solution does not even run — it raised {type(e).__name__}: "
                f"{str(e)[:150]}. A correct, importable function is required.")
    entry = item.get("entry")
    if entry and entry in ns:
        ns.setdefault("candidate", ns[entry])   # HumanEval asserts call candidate(...)
    for lhs, rhs in _assert_pairs(item.get("test", ""))[:max_check]:
        try:
            with _time_limit(5):
                got = eval(lhs, ns)
        except _CertTimeout:
            return (f"Evaluating `{lhs.strip()[:160]}` timed out (infinite loop or too-slow algorithm). "
                    f"The correct result should be `{rhs.strip()[:80]}`. Fix the complexity, then write a "
                    f"fresh correct solution.")
        except Exception as e:
            return (f"Concrete failure: evaluating `{lhs.strip()[:160]}` raised {type(e).__name__}: "
                    f"{str(e)[:120]}. The correct result should be `{rhs.strip()[:80]}`. Diagnose why this "
                    f"input breaks the approach, then write a fresh correct solution.")
        try:
            exp = eval(rhs, ns)
        except Exception:
            exp = None
        if exp is None or got != exp:
            return (f"Concrete counterexample: the call `{lhs.strip()[:160]}` returned `{repr(got)[:120]}` "
                    f"but the correct answer is `{rhs.strip()[:80]}`. Other cases may already be handled. "
                    f"Identify the specific logic error this reveals, then write a fresh, correct solution.")
    return None
