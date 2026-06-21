# DAD-style workspace conditioning to generate non-IID chains for comparison.
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from topological_persistence.sampler import Chain


def build_disagreement_workspace(chains: list[Chain]) -> str:
    from collections import Counter
    answers = Counter()
    for c in chains:
        a = c.answer.strip()
        if a:
            answers[a] += 1

    n = len(chains)
    lines = [f"SOLUTION ANALYSIS ({n} attempts):", ""]
    lines.append("ANSWER DISTRIBUTION:")
    for ans, cnt in answers.most_common():
        pct = cnt / n * 100
        lines.append(f"  {ans}: {cnt}/{n} ({pct:.0f}%)")
    lines.append("")

    if len(answers) > 1:
        lines.append("DISAGREEMENT DETECTED:")
        majority = answers.most_common(1)[0][0] if answers else ""
        minority_answers = [a for a, _ in answers.most_common()[1:]]
        lines.append(f"  Majority answer: {majority}")
        lines.append(f"  Competing answers: {', '.join(minority_answers)}")
        lines.append("")

        lines.append("REPRESENTATIVE REASONING SNIPPETS:")
        shown = set()
        for c in chains:
            if c.answer in shown:
                continue
            shown.add(c.answer)
            snippet = c.text[:500].strip()
            lines.append(f"  [{c.answer}]: {snippet}...")
            lines.append("")
            if len(shown) >= 3:
                break

    lines.append("INSTRUCTION: Treat agreed facts as established. "
                 "Re-derive any disputed steps independently. "
                 "Put your final answer in \\boxed{}.")
    return "\n".join(lines)


def build_conditioned_prompt(problem_text: str, workspace: str, model_name: str) -> str:
    from src.data.dataset import format_prompt
    aug_question = (
        f"{problem_text}\n\n"
        f"Below is an analysis of several previous attempts at this problem:\n"
        f"<workspace>\n{workspace}\n</workspace>\n\n"
        f"Treat the AGREED FACTS as established. Re-derive any disputed claims "
        f"explicitly and verify every line independently. Put your final answer in \\boxed{{}}."
    )
    try:
        return format_prompt({"question": aug_question}, model_name)
    except Exception:
        system = ("You are a helpful math assistant. Solve the problem step by step, "
                  "show your reasoning clearly, and put your final answer in \\boxed{}.")
        return (f"<|im_start|>system\n{system}<|im_end|>\n"
                f"<|im_start|>user\n{aug_question}<|im_end|>\n"
                f"<|im_start|>assistant\n")
