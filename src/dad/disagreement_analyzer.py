# disagreement_analyzer.py
#
# Turns a set of SolutionProfiles into a structured disagreement map and a
# bounded workspace. New relative to the original:
#   * Claims clustered on canonical (type, key, value) instead of surface text.
#   * A best-effort reasoning DAG over claim keys (k' -> k if k's value/key
#     references the symbol named by k').
#   * Leverage l_k = sum_{j in {k} u desc(k) and disputed} (1 - alpha_j),
#     the cascade-aware score used for Gauss-Southwell coordinate selection.
#   * Contested mass Psi = sum_{k in F} mu_k (1 - alpha_k), the quantity the
#     refinement loop drives to zero.
#   * format_workspace foregrounds the single max-leverage disputed claim.

import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from src.dad.claim_extractor import SolutionProfile, canonicalize_value

logger = logging.getLogger(__name__)

AGREEMENT_THRESHOLD = 0.8


@dataclass
class ClaimCluster:
    claim_type: str
    content_key: str                 # canonical key
    values: dict                     # {canonical_value: [solution_idx, ...]}
    majority_value: str
    agreement_ratio: float           # alpha_k
    supporting_solutions: list[int]
    leverage: float = 0.0            # l_k (filled by compute_leverage)
    n_competing: int = 1


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
    contested_mass: float = 0.0      # Psi
    top_leverage_key: str = ""       # argmax_k l_k  (Gauss-Southwell coordinate)
    n_substantive_disputes: int = 0  # disputed claims excluding bare-var/definition noise


def compute_entropy(distribution) -> float:
    total = sum(distribution.values())
    if total == 0:
        return 0.0
    ent = 0.0
    for count in distribution.values():
        if count > 0:
            p = count / total
            ent -= p * math.log2(p)
    return ent


# ----------------------------------------------------------------------
# clustering
# ----------------------------------------------------------------------
def _cluster_claims(profiles):
    """Group claims across solutions by (claim_type, canonical key).

    CRITICAL: one vote per solution per key. A single long chain-of-thought
    asserts the same key many times with transient values (e.g. s = 9/(4-x)
    ... s = 2.5); counting every mention inflates the apparent disagreement.
    We keep each solution's SETTLED value (its last assertion of that key),
    so M solutions contribute at most M votes per key.
    """
    groups = defaultdict(lambda: defaultdict(list))
    for p in profiles:
        settled = {}                       # (type, key) -> last asserted value
        for c in p.claims:
            settled[(c.claim_type, c.key)] = c.value
        for (ctype, ckey), val in settled.items():
            groups[(ctype, ckey)][val].append(p.solution_idx)
    return groups


def _symbols_in(key: str) -> set:
    """Single-letter symbols referenced by a canonical key/value string."""
    return set(re.findall(r"(?<![A-Za-z])([a-zA-Z])(?![A-Za-z])", key or ""))


def build_claim_dag(clusters):
    """Best-effort dependency edges among clusters.

    Edge producer(symbol s) -> consumer(cluster) when the consumer's key or
    majority value references s and s is defined as another cluster's key.
    Returns {cluster_index: set(descendant_indices)} (transitive closure).
    """
    # map a defined symbol -> cluster index that defines it
    defines = {}
    for i, cl in enumerate(clusters):
        sk = cl.content_key.strip()
        if len(sk) == 1 and sk.isalpha():
            defines[sk] = i

    children = defaultdict(set)
    for j, cl in enumerate(clusters):
        refs = _symbols_in(cl.content_key) | _symbols_in(cl.majority_value)
        for s in refs:
            i = defines.get(s)
            if i is not None and i != j:
                children[i].add(j)

    # transitive closure (DAG; cap depth to avoid cycles from noisy keys)
    desc = {}
    for i in range(len(clusters)):
        stack = list(children.get(i, ()))
        seen = set()
        steps = 0
        while stack and steps < 10_000:
            steps += 1
            x = stack.pop()
            if x in seen or x == i:
                continue
            seen.add(x)
            stack.extend(children.get(x, ()))
        desc[i] = seen
    return desc


def compute_leverage(clusters, disputed_idx, desc):
    """l_k = sum over (self u descendants) that are disputed of (1 - alpha)."""
    disputed_set = set(disputed_idx)
    for i, cl in enumerate(clusters):
        subtree = ({i} | desc.get(i, set())) & disputed_set
        cl.leverage = sum((1.0 - clusters[j].agreement_ratio) for j in subtree)


def build_disagreement_map(profiles: list[SolutionProfile],
                           min_support: int = 2,
                           dispute_support_frac: float = 0.5) -> DisagreementMap:
    """Build the disagreement map.

    A cluster is only eligible to be DISPUTED when at least
    ceil(n * dispute_support_frac) solutions actually assert that key (a
    majority must engage before a step counts as contested). Keys touched by
    only a few solutions are sparse noise, not disagreement, and are excluded
    from F, from contested mass, and from leverage. The agreement ratio is
    computed over the supporting solutions, not over surface mentions.
    """
    n = len(profiles)
    if n == 0:
        return DisagreementMap(0, {}, 0.0, "", 0, 0.0)

    import math as _math
    dispute_min_support = max(2, _math.ceil(n * dispute_support_frac))

    # answer distribution — NEVER let an empty/blank answer vote. A truncated
    # chain that emits no answer must not form a spurious "" majority that
    # outvotes the one solution that actually finished.
    answer_counts = Counter()
    for p in profiles:
        a = canonicalize_value(p.final_answer)
        if a and a.strip():
            answer_counts[a] += 1
    if answer_counts:
        majority_answer = answer_counts.most_common(1)[0][0]
        majority_count = answer_counts[majority_answer]
    else:
        majority_answer, majority_count = "", 0
    answer_entropy = compute_entropy(answer_counts)

    # claim clusters (one vote per solution per key)
    groups = _cluster_claims(profiles)
    clusters = []
    for (ctype, ckey), value_map in groups.items():
        support = len({i for idxs in value_map.values() for i in idxs})
        if support < min_support:           # need >=2 solutions to compare
            continue
        best_val = max(value_map, key=lambda v: len(value_map[v]))
        ratio = len(value_map[best_val]) / support   # alpha over support
        clusters.append(ClaimCluster(
            claim_type=ctype,
            content_key=ckey,
            values={v: list(idxs) for v, idxs in value_map.items()},
            majority_value=best_val,
            agreement_ratio=ratio,
            supporting_solutions=[i for idxs in value_map.values() for i in idxs],
            n_competing=len(value_map),
        ))

    # agreed / disputed split with a support floor on disputes
    agreed_idx, disputed_idx = [], []
    for i, cl in enumerate(clusters):
        support = len(set(cl.supporting_solutions))
        if cl.agreement_ratio >= AGREEMENT_THRESHOLD and cl.n_competing == 1:
            agreed_idx.append(i)
        elif support >= dispute_min_support and cl.n_competing > 1:
            disputed_idx.append(i)          # genuine, majority-engaged dispute
        # else: sparse/ambiguous -> ignored (neither agreed nor contested)

    # DAG + leverage (cascade-aware)
    desc = build_claim_dag(clusters)
    compute_leverage(clusters, disputed_idx, desc)

    agreed = [clusters[i] for i in agreed_idx]
    disputed = [clusters[i] for i in disputed_idx]

    # Demote noisy coordinates for SELECTION: a bare single-letter variable key
    # (s, y, t, …) or a 'definition'-type cluster tends to collect spurious
    # leverage from many transient LHS mentions. Such a key should not be the
    # foregrounded dispute unless it carries a real numeric disagreement.
    def _is_noisy_coord(cl):
        k = (cl.content_key or "").strip()
        if cl.claim_type == "definition":
            return True
        if len(k) == 1 and k.isalpha():
            return True
        return False

    # Gauss-Southwell ordering: prefer non-noisy, highest leverage, then lowest alpha.
    disputed.sort(key=lambda c: (_is_noisy_coord(c), -c.leverage, c.agreement_ratio))

    contested_mass = sum((1.0 - c.agreement_ratio) for c in disputed)
    top_key = disputed[0].content_key if disputed else ""
    # number of SUBSTANTIVE disputes (exclude bare-variable / definition noise).
    # This, not raw mass, is what tells "real disagreement" from extraction noise.
    n_substantive = sum(1 for c in disputed if not _is_noisy_coord(c))

    method_counts = Counter()
    for p in profiles:
        for c in p.claims:
            if c.claim_type == "method":
                method_counts[c.value] += 1

    confidence = (majority_count / n)
    if answer_entropy > 0:
        confidence *= 1.0 / (1.0 + answer_entropy)

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
        contested_mass=contested_mass,
        top_leverage_key=top_key,
        n_substantive_disputes=n_substantive,
    )


# ----------------------------------------------------------------------
# workspace synthesis (Gauss-Southwell foregrounding)
# ----------------------------------------------------------------------
def format_workspace(problem_text: str, dmap: DisagreementMap,
                     max_tokens_approx: int = 800,
                     n_agreed: int = 6, n_disputed: int = 5) -> str:
    lines = [f"SOLUTION ANALYSIS ({dmap.n_solutions} attempts):", ""]

    lines.append("ANSWER DISTRIBUTION:")
    for ans, cnt in sorted(dmap.answer_distribution.items(), key=lambda x: -x[1]):
        pct = cnt / dmap.n_solutions * 100 if dmap.n_solutions else 0
        lines.append(f"  {ans}: {cnt}/{dmap.n_solutions} ({pct:.0f}%)")
    lines.append("")

    if dmap.agreed_claims:
        lines.append("AGREED FACTS (treat as established):")
        for c in dmap.agreed_claims[:n_agreed]:
            lines.append(f"  - {c.content_key} = {c.majority_value}")
        lines.append("")

    if dmap.disputed_claims:
        # the single highest-leverage dispute is shown first and called out
        top = dmap.disputed_claims[0]
        lines.append("PRIMARY DISPUTE TO RESOLVE (verify this step explicitly):")
        comp = " vs ".join(
            f"{v} ({len(idxs)} solutions)"
            for v, idxs in sorted(top.values.items(), key=lambda x: -len(x[1]))
        )
        lines.append(f"  - {top.content_key}: {comp}")
        lines.append("")

        if len(dmap.disputed_claims) > 1:
            lines.append("OTHER DISPUTED CLAIMS:")
            for c in dmap.disputed_claims[1:n_disputed]:
                comp = " vs ".join(
                    f"{v} ({len(idxs)})"
                    for v, idxs in sorted(c.values.items(), key=lambda x: -len(x[1]))
                )
                lines.append(f"  - {c.content_key}: {comp}")
            lines.append("")

    if dmap.method_distribution and len(dmap.method_distribution) > 1:
        lines.append("APPROACHES USED:")
        for method, cnt in sorted(dmap.method_distribution.items(), key=lambda x: -x[1]):
            lines.append(f"  - {method}: {cnt} solutions")
        lines.append("")

    if dmap.contested_mass > 1.0:
        lines.append("NOTE: High disagreement. Re-derive the primary disputed step "
                     "from the agreed facts and verify each line.")
    elif dmap.contested_mass > 0.0:
        lines.append("NOTE: Some disagreement remains. Focus on the primary dispute above.")
    else:
        lines.append("NOTE: Solutions agree. Verify the shared reasoning is correct.")

    workspace = "\n".join(lines)
    char_limit = max_tokens_approx * 4
    if len(workspace) > char_limit:
        workspace = workspace[:char_limit] + "\n[truncated]"
    return workspace