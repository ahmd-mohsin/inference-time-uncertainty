# claim_extractor.py
#
# Deterministic, single-model claim extraction for DAD.
# No secondary LLM call (honors the single-model constraint of the method).
#
# Key improvements over the original:
#   1. Value canonicalization via sympy (optional dependency, graceful fallback):
#      -75/24, -25/8, and -3.125 all canonicalize to the SAME key, so equivalent
#      claims cluster correctly instead of being scattered by surface form.
#   2. Expression-LHS equations: captures "4a + 3b + 2c = -25/8" (multi-term LHS),
#      which the original identifier-only regex silently dropped.
#   3. Equality chains: "... = -75/24 = -25/8" yields each consecutive relation.
#   4. Wider deductive markers and LaTeX stripping so fewer steps are missed.
#   5. A normalized clustering *key* is stored on every claim, decoupled from the
#      raw surface text, so the analyzer clusters on meaning, not on phrasing.

import re
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# optional sympy backend
# ----------------------------------------------------------------------
try:
    from sympy import sympify, Rational, nsimplify, S
    from sympy.parsing.sympy_parser import (
        parse_expr,
        standard_transformations,
        implicit_multiplication_application,
        convert_xor,
    )

    _SYMPY = True
    _TRANSFORMS = standard_transformations + (
        implicit_multiplication_application,
        convert_xor,
    )
except Exception:  # pragma: no cover
    _SYMPY = False
    _TRANSFORMS = None


# ----------------------------------------------------------------------
# data model
# ----------------------------------------------------------------------
@dataclass
class MathClaim:
    claim_type: str          # 'equation' | 'intermediate_result' | 'method' | 'variable_choice'
    content: str             # raw surface text (for the workspace / debugging)
    value: str               # canonical value string (used for clustering)
    source_solution_idx: int
    key: str = ""            # canonical clustering key (LHS / quantity name)


@dataclass
class SolutionProfile:
    solution_idx: int
    text: str
    final_answer: str
    claims: list[MathClaim] = field(default_factory=list)


# ----------------------------------------------------------------------
# latex / surface cleaning
# ----------------------------------------------------------------------
_LATEX_DROP = [
    r"\left", r"\right", r"\,", r"\!", r"\;", r"\:", r"\quad", r"\qquad",
    r"\displaystyle", r"\text", r"\mathrm", r"\mathbf", r"\bf", r"\;",
    r"\big", r"\Big", r"\bigl", r"\bigr", r"$", r"\(", r"\)", r"\[", r"\]",
]


def _strip_latex(s: str) -> str:
    s = s.strip()
    for tok in _LATEX_DROP:
        s = s.replace(tok, " ")
    # \frac{a}{b} -> (a)/(b) ; \dfrac{a}{b} likewise
    s = re.sub(r"\\d?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"((\1)/(\2))", s)
    # \sqrt{x} -> sqrt(x)
    s = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"sqrt(\1)", s)
    s = re.sub(r"\\sqrt\s*([0-9a-zA-Z])", r"sqrt(\1)", s)
    # remove remaining backslash-commands like \cdot \times \pi (keep a space)
    s = s.replace(r"\cdot", "*").replace(r"\times", "*").replace(r"\div", "/")
    s = re.sub(r"\\[a-zA-Z]+", " ", s)
    s = s.replace("^", "**")
    s = s.replace("{", "(").replace("}", ")")
    s = s.replace(",", "")  # thousands separators / list commas
    s = re.sub(r"°|\\circ", "", s)
    return s.strip()


_NUMERIC_RE = re.compile(r"^[\s+\-*/().0-9eE]+$")


def canonicalize_value(raw: str) -> str:
    """Return a canonical string for a math value/expression.

    Equivalent surface forms map to one string. Falls back to a cleaned
    lowercase string when sympy is unavailable or parsing fails.
    """
    if raw is None:
        return ""
    s = _strip_latex(str(raw))
    if not s:
        return ""
    if len(s) > 200:                     # guard against pathological spans
        return s.lower().strip()

    if _SYMPY:
        try:
            expr = parse_expr(s, transformations=_TRANSFORMS, evaluate=True)
            # collapse trivial floats to exact rationals where possible
            try:
                expr = nsimplify(expr, rational=True)
            except Exception:
                pass
            return _sstr(expr)
        except Exception:
            pass
    # fallback: whitespace-normalized lowercase
    return re.sub(r"\s+", "", s).lower()


def _sstr(expr) -> str:
    """Deterministic canonical string for a sympy expression."""
    from sympy import srepr  # noqa
    try:
        return str(expr)
    except Exception:
        return ""


# back-compat alias (disagreement_analyzer imports this name)
extract_numeric_value = canonicalize_value


def normalize_key(lhs: str) -> str:
    """Canonical key for the left-hand quantity of an equation.

    Keeps symbolic structure (does NOT evaluate), so 'a+b+c', 'a + b + c',
    and 'c+b+a' all map to one key, while remaining distinct from a value.
    """
    s = _strip_latex(str(lhs))
    if not s:
        return ""
    if _SYMPY and len(s) <= 120:
        try:
            expr = parse_expr(s, transformations=_TRANSFORMS, evaluate=False)
            return _sstr(expr)
        except Exception:
            pass
    return re.sub(r"\s+", "", s).lower()


def _has_symbol(canon: str) -> bool:
    """True if the canonical string contains a letter (i.e. a variable)."""
    return bool(re.search(r"[a-zA-Z]", canon)) and "sqrt" != canon


# ----------------------------------------------------------------------
# boxed answer
# ----------------------------------------------------------------------
def extract_boxed_answer(text: str):
    depth = 0
    start = None
    i = 0
    while i < len(text):
        if text[i:i + 7] == r"\boxed{":
            if start is None:
                start = i + 7
                depth = 1
                i += 7
                continue
        if start is not None:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i].strip()
        i += 1
    pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
    matches = re.findall(pattern, text)
    return matches[-1].strip() if matches else None


# ----------------------------------------------------------------------
# segmentation + equality parsing
# ----------------------------------------------------------------------
_DOT = "\x01"


def _segments(text: str):
    """Split a solution into clean reasoning segments.

    Splits on newlines, display-math delimiters, semicolons, sentence periods,
    and clause commas, while protecting decimal points (3.14 is not a boundary).
    Isolating each statement is what keeps an equation's key from being
    contaminated by surrounding prose.
    """
    t = re.sub(r"\\\[|\\\]|\\\(|\\\)", "\n", text)
    t = t.replace("$$", "\n")
    # protect decimals: 3.14 -> 3<DOT>14 so '.' is not a sentence boundary
    t = re.sub(r"(\d)\.(\d)", r"\1" + _DOT + r"\2", t)
    # boundaries: newline | ; | sentence-period+space | clause-comma+space
    for piece in re.split(r"\n|;|\.\s+|,\s+", t):
        piece = piece.replace(_DOT, ".").strip().rstrip(".")
        if piece:
            yield piece


# leading words that signal the segment is prose, not a bare equation
_STOPWORDS = {
    "let", "set", "define", "denote", "we", "so", "thus", "hence", "therefore",
    "then", "since", "because", "where", "and", "the", "is", "are", "compute",
    "computing", "using", "use", "get", "have", "obtain", "gives", "give",
    "equals", "this", "that", "which", "now", "first", "next", "finally",
    "adding", "solving", "substituting", "converting", "taking",
}


def _is_prose_lhs(lhs: str) -> bool:
    """True if the LHS looks like a sentence fragment rather than a math LHS."""
    if len(lhs) > 40:
        return True
    words = re.findall(r"[A-Za-z]{2,}", lhs)
    # multi-letter tokens that are English words (not log2/sqrt/sin/cos/etc.)
    mathy = {"log", "log2", "ln", "sqrt", "sin", "cos", "tan", "exp", "mod",
             "gcd", "lcm", "max", "min", "pi"}
    for w in words:
        if w.lower() in _STOPWORDS:
            return True
        if w.lower() not in mathy and len(w) >= 3:
            return True
    return False


# relational operators that are NOT plain equalities
_INEQ_TOKENS = [r"\le", r"\ge", r"\neq", r"\ne", r"\approx", r"\geq", r"\leq",
                "<=", ">=", "!=", "≈", "≤", "≥", "≠"]


def _split_equality_chain(seg: str):
    """Return the operands of an equality chain 'a = b = c' -> [a, b, c].

    Returns [] when the segment has no usable plain equality.
    """
    if "=" not in seg:
        return []
    if any(tok in seg for tok in _INEQ_TOKENS):
        return []
    # protect '==' and arrows; treat only single '=' as a relation
    s = seg.replace("==", "\x00").replace("=>", "\x00").replace("<=", "\x00")
    parts = [p.strip() for p in s.split("=")]
    parts = [p.replace("\x00", "==") for p in parts]
    parts = [p for p in parts if p]
    return parts if len(parts) >= 2 else []


def _clean_lhs(lhs: str) -> str:
    """Strip leading natural-language words from an LHS, keeping the math tail.

    'We compute 4a + 3b + 2c' -> '4a + 3b + 2c'. Returns '' if nothing math-like
    remains.
    """
    lhs = lhs.strip()
    toks = lhs.split()
    mathy = {"log", "log2", "ln", "sqrt", "sin", "cos", "tan", "exp", "mod",
             "gcd", "lcm", "max", "min", "pi"}
    # drop leading tokens that are clearly prose
    while toks:
        w = re.sub(r"[^A-Za-z]", "", toks[0]).lower()
        if w and w not in mathy and (w in _STOPWORDS or len(w) >= 3):
            toks.pop(0)
        else:
            break
    return " ".join(toks).strip()


def extract_equations(text, solution_idx):
    claims = []
    for seg in _segments(text):
        parts = _split_equality_chain(seg)
        if not parts:
            continue
        for lhs, rhs in zip(parts, parts[1:]):
            if len(rhs) < 1 or rhs.startswith("\\begin"):
                continue
            lhs = _clean_lhs(lhs)
            if not lhs or _is_prose_lhs(lhs):
                continue
            key = normalize_key(lhs)
            val = canonicalize_value(rhs)
            if not key or not val or not _has_symbol(key):
                continue
            # definitional identity (symbolic RHS, e.g. b = log2 y) vs
            # evaluation (numeric RHS, e.g. b = -3/8). Keep them in separate
            # clusters so a definition never dilutes a value's agreement ratio.
            ctype = "definition" if _has_symbol(val) else "equation"
            claims.append(MathClaim(
                claim_type=ctype,
                content=f"{lhs} = {rhs}",
                value=val,
                source_solution_idx=solution_idx,
                key=key,
            ))
    return claims


_DEDUCTIVE = (
    r"(?:therefore|thus|hence|so we have|so we get|we get|we have|we obtain|"
    r"this gives|which gives|it follows that|consequently|implies that|"
    r"the answer is|the result is|the value is|equals)"
)


def extract_intermediate_results(text, solution_idx):
    claims = []
    pat = re.compile(_DEDUCTIVE + r"\s*[,:]?\s*(.{2,120}?)(?:\.|\n|$)", re.IGNORECASE)
    for m in re.finditer(pat, text):
        content = m.group(1).strip().rstrip(".")
        if not content:
            continue
        # if the fragment is itself an equation, key on its LHS; else key on value
        parts = _split_equality_chain(content)
        if parts and len(parts) >= 2:
            key = normalize_key(parts[0])
            val = canonicalize_value(parts[-1])
        else:
            val = canonicalize_value(content)
            key = f"result::{val}"          # value-keyed result claim
        if not val:
            continue
        claims.append(MathClaim(
            claim_type="intermediate_result",
            content=content,
            value=val,
            source_solution_idx=solution_idx,
            key=key,
        ))
    return claims


_METHOD_PATTERNS = [
    (r"(coordinate\s+geometry)", "method"),
    (r"(trigonometry|trig\b)", "method"),
    (r"(the\s+(?:angle\s+bisector|power\s+of\s+a\s+point|Stewart|Menelaus|Ceva|"
     r"Pythagorean|law\s+of\s+(?:sines|cosines))[\w\s]*?(?:theorem|lemma)?)", "method"),
    (r"(modular\s+arithmetic|mod\s+\d+)", "method"),
    (r"(generating\s+functions?)", "method"),
    (r"(complementary\s+counting|casework|case\s+analysis|principle\s+of\s+inclusion)", "method"),
    (r"(strong\s+induction|induction)", "method"),
    (r"(substitution|change\s+of\s+variables?)", "method"),
    (r"(the\s+quadratic\s+formula|completing\s+the\s+square)", "method"),
    (r"(logarithm(?:ic)?\s+(?:system|identities|properties)|take\s+(?:the\s+)?log)", "method"),
    (r"(vieta'?s)", "method"),
    (r"(recursion|recurrence)", "method"),
]


def extract_method_choice(text, solution_idx):
    claims = []
    for pat, ctype in _METHOD_PATTERNS:
        for m in re.finditer(pat, text, re.IGNORECASE):
            name = re.sub(r"\s+", " ", m.group(1).strip().lower())
            claims.append(MathClaim(
                claim_type=ctype,
                content=name,
                value=name,
                source_solution_idx=solution_idx,
                key=f"method::{name}",
            ))
    # variable definitions: "let a = log2 x", "set x = ..."
    for m in re.finditer(r"(?:let|set|define|denote)\s+([A-Za-z])\s*=", text, re.IGNORECASE):
        v = m.group(1).strip().lower()
        claims.append(MathClaim(
            claim_type="variable_choice",
            content=f"let {v} = ...",
            value=v,
            source_solution_idx=solution_idx,
            key=f"varchoice::{v}",
        ))
    return claims


def profile_solution(text, solution_idx):
    answer = extract_boxed_answer(text)
    if answer is None:
        num = re.search(r"(?:answer is|answer:)\s*([^\n]{1,60})", text, re.IGNORECASE)
        answer = num.group(1).strip() if num else ""

    claims = []
    claims.extend(extract_equations(text, solution_idx))
    claims.extend(extract_intermediate_results(text, solution_idx))
    claims.extend(extract_method_choice(text, solution_idx))

    seen = set()
    deduped = []
    for c in claims:
        dk = (c.claim_type, c.key, c.value)
        if dk not in seen:
            seen.add(dk)
            deduped.append(c)

    return SolutionProfile(
        solution_idx=solution_idx,
        text=text,
        final_answer=answer or "",
        claims=deduped,
    )