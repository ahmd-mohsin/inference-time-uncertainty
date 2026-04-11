import re
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class MathClaim:
    claim_type: str
    content: str
    value: str
    source_solution_idx: int


@dataclass
class SolutionProfile:
    solution_idx: int
    text: str
    final_answer: str
    claims: list[MathClaim] = field(default_factory=list)


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


def extract_numeric_value(text):
    text = text.strip()
    text = text.replace(",", "")
    text = re.sub(r"\\circ|°", "", text)
    text = text.replace("{", "").replace("}", "")
    text = text.replace("\\left", "").replace("\\right", "")
    try:
        return str(float(text))
    except ValueError:
        pass
    frac = re.match(r"^\\?frac\s*\{?\s*(-?\d+)\s*\}?\s*\{?\s*(-?\d+)\s*\}?$", text)
    if frac:
        try:
            return str(float(frac.group(1)) / float(frac.group(2)))
        except (ValueError, ZeroDivisionError):
            pass
    return text.lower().strip()


def extract_equations(text, solution_idx):
    claims = []
    eq_patterns = [
        r"([a-zA-Z_]\w*)\s*=\s*([^\n,;]{1,80})",
        r"\\([a-zA-Z]+)\s*=\s*([^\n\\]{1,80})",
    ]
    for pat in eq_patterns:
        for m in re.finditer(pat, text):
            lhs = m.group(1).strip()
            rhs = m.group(2).strip().rstrip(".")
            if len(rhs) < 2 or rhs.startswith("\\begin"):
                continue
            claims.append(MathClaim(
                claim_type="equation",
                content=f"{lhs} = {rhs}",
                value=extract_numeric_value(rhs),
                source_solution_idx=solution_idx,
            ))
    return claims


def extract_intermediate_results(text, solution_idx):
    claims = []
    result_patterns = [
        r"(?:therefore|thus|hence|so|we get|we have|we obtain|this gives|which gives)\s*[,:]?\s*(.{5,120}?)(?:\.|$)",
        r"(?:the answer is|the result is|equals)\s+(.{3,80}?)(?:\.|$)",
    ]
    for pat in result_patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            content = m.group(1).strip()
            val = extract_numeric_value(content)
            claims.append(MathClaim(
                claim_type="intermediate_result",
                content=content,
                value=val,
                source_solution_idx=solution_idx,
            ))
    return claims


def extract_method_choice(text, solution_idx):
    claims = []
    method_patterns = [
        (r"(?:using|by|via|with)\s+(coordinate\s+geometry)", "method"),
        (r"(?:using|by|via|with)\s+(trigonometry|trig)", "method"),
        (r"(?:using|by|via|with)\s+(the\s+(?:angle|power|Stewart|Menelaus|Ceva)[\w\s]*(?:theorem|lemma))", "method"),
        (r"(?:using|by|via|with)\s+(modular\s+arithmetic)", "method"),
        (r"(?:using|by|via|with)\s+(generating\s+functions?)", "method"),
        (r"(?:using|by|via|with)\s+(casework|case\s+analysis)", "method"),
        (r"(?:using|by|via|with)\s+(induction)", "method"),
        (r"(?:using|by|via|with)\s+(substitution)", "method"),
        (r"(?:using|by|via|with)\s+(the\s+quadratic\s+formula)", "method"),
        (r"(?:let|setting|define)\s+([A-Za-z])\s*=", "variable_choice"),
    ]
    for pat, claim_type in method_patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            claims.append(MathClaim(
                claim_type=claim_type,
                content=m.group(1).strip().lower(),
                value=m.group(1).strip().lower(),
                source_solution_idx=solution_idx,
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
        key = (c.claim_type, c.value)
        if key not in seen:
            seen.add(key)
            deduped.append(c)

    return SolutionProfile(
        solution_idx=solution_idx,
        text=text,
        final_answer=answer or "",
        claims=deduped,
    )