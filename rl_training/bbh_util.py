# BBH (BIG-Bench-Hard) logic-reasoning domain: loader + robust exact-match verifier + prompt.
# Third domain (after math, code) — tests "update rule governs OOD transfer" on symbolic reasoning + Assumption A
# (compositional subskills): train one task family, eval a structurally-distant one.
import re

BBH_PROMPT = ("Solve the reasoning problem. Think briefly, then end with a line exactly:\nThe answer is <ANSWER>.\n\n{q}")

# task families for the train/OOD split (all exact-answer; ~250 ex each)
BBH_TRAIN = ["boolean_expressions", "web_of_lies", "navigate"]                       # logic/boolean family
BBH_OOD   = ["logical_deduction_three_objects", "tracking_shuffled_objects_three_objects", "date_understanding"]

def load_bbh(task, n=-1):
    from datasets import load_dataset
    d = None
    for did in (("lukaemon/bbh", task), ("maveriq/bigbenchhard", task), ("Joschka/big_bench_hard", task)):
        try:
            dd = load_dataset(did[0], did[1]); sp = "test" if "test" in dd else list(dd.keys())[0]; d = dd[sp]; break
        except Exception: continue
    if d is None: raise SystemExit(f"BBH task {task} unavailable")
    items = [{"q": r.get("input") or r.get("question") or "", "gold": str(r.get("target") or r.get("answer") or "").strip()} for r in d]
    return items[:n] if n > 0 else items

def _norm(x):
    x = str(x).strip().strip(".").strip()
    x = re.sub(r"^\(([A-Za-z])\)$", r"\1", x)          # (A) -> A
    return x.lower().replace(" ", "")

def bbh_match(text, gold):
    if not text: return False
    g = _norm(gold)
    # extract model answer: "The answer is X" > \boxed{} > last (X) > last line
    m = re.findall(r"answer is\s*:?\s*(.+)", text, re.I)
    cand = []
    if m: cand.append(m[-1].split("\n")[0])
    cand += re.findall(r"\\boxed\{([^}]*)\}", text)
    cand += re.findall(r"\(([A-Za-z])\)", text)
    cand.append(text.strip().split("\n")[-1])
    for c in cand:
        if _norm(c) == g: return True
        # gold may be "(A)" while model said "A", or contain the option text
        if g and (g in _norm(c) or _norm(c) in g) and len(g) > 1: return True
    return False
