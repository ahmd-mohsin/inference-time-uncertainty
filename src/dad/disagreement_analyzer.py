import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from src.dad.claim_extractor import SolutionProfile, extract_numeric_value

logger = logging.getLogger(__name__)


@dataclass
class ClaimCluster:
    claim_type: str
    content_key: str
    values: dict
    majority_value: str
    agreement_ratio: float
    supporting_solutions: list[int]


@dataclass
class DisagreementMap:
    n_solutions: int
    answer_distribution: dict
    answer_entropy: float
    majority_answer: str
    majority_answer_count: int
    majority_answer_fraction: float
    agreed_claims: list[ClaimCluster] = field(default_factory=list)
    disputed_claims: list[ClaimCluster] = field(default_factory=list)
    method_distribution: dict = field(default_factory=dict)
    confidence_score: float = 0.0


def compute_entropy(distribution):
    import math
    total = sum(distribution.values())
    if total == 0:
        return 0.0
    ent = 0.0
    for count in distribution.values():
        if count > 0:
            p = count / total
            ent -= p * math.log2(p)
    return ent


def build_disagreement_map(profiles: list[SolutionProfile]) -> DisagreementMap:
    n = len(profiles)
    if n == 0:
        return DisagreementMap(
            n_solutions=0,
            answer_distribution={},
            answer_entropy=0.0,
            majority_answer="",
            majority_answer_count=0,
            majority_answer_fraction=0.0,
        )

    answer_counts = Counter()
    for p in profiles:
        norm_ans = extract_numeric_value(p.final_answer)
        answer_counts[norm_ans] += 1

    majority_answer = answer_counts.most_common(1)[0][0] if answer_counts else ""
    majority_count = answer_counts[majority_answer]
    answer_entropy = compute_entropy(answer_counts)

    claim_groups = defaultdict(lambda: defaultdict(list))
    for p in profiles:
        for c in p.claims:
            key = (c.claim_type, c.content.split("=")[0].strip().lower() if "=" in c.content else c.content[:30].lower())
            claim_groups[key][c.value].append(p.solution_idx)

    agreed = []
    disputed = []
    for (ctype, ckey), value_map in claim_groups.items():
        total_mentions = sum(len(v) for v in value_map.values())
        if total_mentions < 2:
            continue
        best_val = max(value_map, key=lambda v: len(value_map[v]))
        best_count = len(value_map[best_val])
        ratio = best_count / total_mentions

        cluster = ClaimCluster(
            claim_type=ctype,
            content_key=ckey,
            values={v: idxs for v, idxs in value_map.items()},
            majority_value=best_val,
            agreement_ratio=ratio,
            supporting_solutions=[idx for idxs in value_map.values() for idx in idxs],
        )

        if ratio >= 0.8 and len(value_map) == 1:
            agreed.append(cluster)
        else:
            disputed.append(cluster)

    disputed.sort(key=lambda c: c.agreement_ratio)

    method_counts = Counter()
    for p in profiles:
        for c in p.claims:
            if c.claim_type == "method":
                method_counts[c.value] += 1

    confidence = majority_count / n
    if answer_entropy > 0:
        confidence *= (1.0 / (1.0 + answer_entropy))

    return DisagreementMap(
        n_solutions=n,
        answer_distribution=dict(answer_counts),
        answer_entropy=answer_entropy,
        majority_answer=majority_answer,
        majority_answer_count=majority_count,
        majority_answer_fraction=majority_count / n,
        agreed_claims=agreed,
        disputed_claims=disputed,
        method_distribution=dict(method_counts),
        confidence_score=confidence,
    )


def format_workspace(problem_text: str, dmap: DisagreementMap, max_tokens_approx: int = 800) -> str:
    lines = []
    lines.append(f"SOLUTION ANALYSIS ({dmap.n_solutions} attempts):")
    lines.append("")

    lines.append("ANSWER DISTRIBUTION:")
    for ans, cnt in sorted(dmap.answer_distribution.items(), key=lambda x: -x[1]):
        pct = cnt / dmap.n_solutions * 100
        lines.append(f"  {ans}: {cnt}/{dmap.n_solutions} ({pct:.0f}%)")
    lines.append("")

    if dmap.agreed_claims:
        lines.append("AGREED FACTS (all solutions concur):")
        for c in dmap.agreed_claims[:5]:
            lines.append(f"  - {c.content_key} = {c.majority_value}")
        lines.append("")

    if dmap.disputed_claims:
        lines.append("DISPUTED CLAIMS (solutions disagree):")
        for c in dmap.disputed_claims[:5]:
            vals = []
            for v, idxs in sorted(c.values.items(), key=lambda x: -len(x[1])):
                vals.append(f"{v} ({len(idxs)} solutions)")
            lines.append(f"  - {c.content_key}: {' vs '.join(vals)}")
        lines.append("")

    if dmap.method_distribution:
        lines.append("APPROACHES USED:")
        for method, cnt in sorted(dmap.method_distribution.items(), key=lambda x: -x[1]):
            lines.append(f"  - {method}: {cnt} solutions")
        lines.append("")

    if dmap.answer_entropy > 1.0:
        lines.append("NOTE: High disagreement on final answer. Carefully verify each step.")
    elif dmap.answer_entropy > 0.0:
        lines.append("NOTE: Some disagreement exists. Check the disputed claims above.")
    else:
        lines.append("NOTE: All solutions agree. Verify the shared reasoning is correct.")

    workspace = "\n".join(lines)

    char_limit = max_tokens_approx * 4
    if len(workspace) > char_limit:
        workspace = workspace[:char_limit] + "\n[truncated]"

    return workspace