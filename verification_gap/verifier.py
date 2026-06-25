# Verification prompt, answer parsing, and YES/NO scoring for the gap study.
import re


# ----------------------------------------------------------------- answer parsing

def extract_boxed(text: str) -> str:
    """Last \\boxed{...} content, handling one level of nested braces."""
    m = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    return m[-1].strip() if m else ""


def answers_match(pred: str, gold) -> bool:
    if pred is None or gold is None:
        return False
    p = str(pred).strip().strip("$").rstrip(".,")
    g = str(gold).strip().strip("$").rstrip(".,")
    if p == "" or g == "":
        return False
    if p == g:
        return True
    try:
        return abs(float(p) - float(g)) < 1e-6
    except (ValueError, TypeError):
        return False


# ----------------------------------------------------------------- verifier prompt

# We ask the model to judge a *proposed* answer. Crucial design points:
#   - It judges a candidate answer string, NOT one of its own chains -> tests verification
#     of an answer in isolation (the V capability), decoupled from generation.
#   - We force the verdict into a parseable form: "VERDICT: YES" / "VERDICT: NO".
#   - We ALWAYS include known-wrong candidates so V is measured as discrimination (AUC),
#     never raw accuracy -> immune to the model's yes-bias.

_VERIFY_SYSTEM = (
    "You are a careful math grader. You are given a competition problem and a PROPOSED "
    "final answer. Decide whether the proposed answer is correct. Think briefly, then end "
    "your response with exactly one line:\nVERDICT: YES\nor\nVERDICT: NO"
)


def build_verify_prompt(problem_question: str, candidate_answer: str, model_name: str) -> str:
    ml = model_name.lower()
    user = (f"Problem:\n{problem_question}\n\n"
            f"Proposed final answer: {candidate_answer}\n\n"
            f"Is this proposed answer correct? End with 'VERDICT: YES' or 'VERDICT: NO'.")
    if any(k in ml for k in ["qwen", "deepseek", "nemotron"]):
        return (f"<|im_start|>system\n{_VERIFY_SYSTEM}<|im_end|>\n"
                f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")
    if "llama" in ml:
        return (f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{_VERIFY_SYSTEM}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n")
    return (f"<|im_start|>system\n{_VERIFY_SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")


def parse_verdict(text: str):
    """Return 1.0 for YES, 0.0 for NO, None if unparseable.

    Looks for the LAST explicit 'VERDICT: YES/NO'; falls back to a trailing yes/no token.
    """
    m = re.findall(r"VERDICT:\s*(YES|NO)", text, flags=re.IGNORECASE)
    if m:
        return 1.0 if m[-1].upper() == "YES" else 0.0
    tail = text[-200:].lower()
    yes = tail.rfind("yes")
    no = tail.rfind("no")
    if yes == -1 and no == -1:
        return None
    return 1.0 if yes > no else 0.0
