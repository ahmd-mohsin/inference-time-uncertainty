#!/usr/bin/env python3
"""
Estimate whether dropping the DAD probe width from M=8 to M=4 would hold accuracy,
using ONLY an existing M=8 DAD JSONL (no reruns).

For each problem we reconstruct the bag of 8 probe answers (non-blank answers from
disagreement_map.answer_distribution, plus the implied blanks), then EXACTLY
enumerate all C(8,4)=70 four-subsets and ask, for each subset, what the majority
vote would have been. From that we get, per problem:

  p_keep  = fraction of 4-subsets whose unique plurality winner == the M=8 majority
  p_tie   = fraction of 4-subsets where the M=8 majority is tied for the lead
  p_lose  = fraction where some OTHER answer (or blank) wins outright

A problem is "at risk" if p_lose is non-trivial. Expected M=4 accuracy is the sum
of per-problem keep-probabilities over correct problems (ties scored 0.5).

Also reports a token-savings estimate that accounts for refinement rounds, to show
that halving M does NOT halve tokens.
"""
import json, sys, itertools
from collections import Counter

DEFAULT_PATH = "/home/ahmed/inference-time-uncertainty/data/inference_outputs/Qwen3-8B_aime_2024_dad_20260603_141726/dad_m8_r3.jsonl"
TARGET_M = 4            # the probe width you're considering
TIE_RISK_THRESHOLD = 0.05   # flag a problem if p_lose+0.5*p_tie exceeds this


def norm(s):
    """Light normalization just for the gold-recoverability heuristic."""
    s = str(s).strip().lower()
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except Exception:
        return "".join(ch for ch in s if ch.isalnum())


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def probe_bag(rec):
    """Reconstruct the 8-element multiset of probe answers, including blanks."""
    dmap = rec.get("disagreement_map", {}) or {}
    dist = dmap.get("answer_distribution", {}) or {}
    spr = rec.get("samples_per_round") or [8]
    M = spr[0] if spr else 8
    bag = []
    for ans, c in dist.items():
        bag += [ans] * int(c)
    n_blank = max(0, M - len(bag))
    bag += [""] * n_blank
    # if dist somehow exceeds M, truncate (shouldn't happen)
    return bag[:M], M, n_blank


def vote(subset):
    """Pipeline-style vote: majority over NON-BLANK answers. Returns (winner_set, has_answer)."""
    counts = Counter(a for a in subset if a != "")
    if not counts:
        return set(), False
    top = max(counts.values())
    winners = {a for a, c in counts.items() if c == top}
    return winners, True


def analyze(rec):
    bag, M, n_blank = probe_bag(rec)
    dmap = rec.get("disagreement_map", {}) or {}
    m8_maj = dmap.get("majority_answer", "")
    m8_correct = bool(rec.get("correct"))
    gold = rec.get("gold_answer", "")

    # which (if any) distribution key equals gold -> recoverability upside detection
    gold_key = None
    for a in dmap.get("answer_distribution", {}):
        if norm(a) == norm(gold):
            gold_key = a
            break

    k = min(TARGET_M, M)
    subsets = list(itertools.combinations(range(M), k))
    keep = tie = lose = gold_wins = blank_outcome = 0
    for idx in subsets:
        sub = [bag[i] for i in idx]
        winners, has = vote(sub)
        if not has:
            blank_outcome += 1
            lose += 1
            continue
        # keep/tie/lose measured against the M=8 majority answer
        if winners == {m8_maj}:
            keep += 1
        elif m8_maj in winners:
            tie += 1
        else:
            lose += 1
        # separately: does the GOLD answer win this subset outright? (upside on wrong probs)
        if gold_key is not None and winners == {gold_key}:
            gold_wins += 1

    n = len(subsets)
    p_keep, p_tie, p_lose = keep / n, tie / n, lose / n
    p_blank = blank_outcome / n
    p_gold = gold_wins / n

    # expected M=4 correctness for this problem
    if m8_correct:
        exp_correct = p_keep + 0.5 * p_tie
    else:
        # already wrong at M=8; only upside is a subset where gold wins outright
        exp_correct = p_gold

    risk_score = (p_lose + 0.5 * p_tie) if m8_correct else 0.0
    return {
        "pid": rec.get("problem_id"),
        "gold": gold,
        "m8_majority": m8_maj,
        "m8_correct": m8_correct,
        "bag": Counter(bag),
        "n_blank": n_blank,
        "p_keep": p_keep, "p_tie": p_tie, "p_lose": p_lose,
        "p_blank": p_blank, "p_gold_recover": p_gold,
        "exp_correct_m4": exp_correct,
        "risk_score": risk_score,
        # tokens
        "total_tokens": rec.get("total_tokens", 0),
        "n_total_generations": rec.get("n_total_generations", M),
        "n_rounds": rec.get("n_rounds", 1),
        "stop_reason": rec.get("stop_reason", ""),
    }


def fmt_bag(c):
    parts = []
    for a, n in c.most_common():
        label = "BLANK" if a == "" else (a[:14] + ("…" if len(a) > 14 else ""))
        parts.append(f"{label}:{n}")
    return "{" + ", ".join(parts) + "}"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    rows = load(path)
    res = [analyze(r) for r in rows]

    N = len(res)
    m8_acc = sum(r["m8_correct"] for r in res) / N
    est_m4_acc = sum(r["exp_correct_m4"] for r in res) / N

    # token savings: dropping (M - TARGET_M) probe chains removes that many gens at the
    # problem's average per-generation cost. Refinement rounds are untouched.
    saved = []
    for r in res:
        g = max(1, r["n_total_generations"])
        per_gen = r["total_tokens"] / g
        dropped = max(0, 8 - TARGET_M)  # probe chains removed
        # can't drop more probe chains than existed in round 1
        new_tokens = max(r["total_tokens"] - dropped * per_gen, per_gen)
        saved.append((r["total_tokens"], new_tokens))
    mean_old = sum(s[0] for s in saved) / N
    mean_new = sum(s[1] for s in saved) / N

    print(f"\nFile: {path}")
    print(f"Problems: {N}   M=8 accuracy: {m8_acc:.3f} ({sum(r['m8_correct'] for r in res)}/{N})")
    print("=" * 100)
    print(f"{'pid':>4} {'gold':>6} {'M8ok':>5} {'p_keep':>7} {'p_tie':>6} {'p_lose':>7} "
          f"{'p_blank':>8} {'recov':>6}  probe_bag")
    print("-" * 100)
    for r in sorted(res, key=lambda x: -x["risk_score"]):
        flag = "  <-- AT RISK" if r["risk_score"] > TIE_RISK_THRESHOLD else ""
        print(f"{str(r['pid']):>4} {str(r['gold'])[:6]:>6} {str(r['m8_correct'])[:5]:>5} "
              f"{r['p_keep']:>7.3f} {r['p_tie']:>6.3f} {r['p_lose']:>7.3f} "
              f"{r['p_blank']:>8.3f} {r['p_gold_recover']:>6.3f}  {fmt_bag(r['bag'])}{flag}")
    print("=" * 100)

    at_risk = [r for r in res if r["risk_score"] > TIE_RISK_THRESHOLD]
    print(f"\nEstimated M={TARGET_M} accuracy: {est_m4_acc:.3f}  "
          f"(vs M=8 actual {m8_acc:.3f}, expected change {est_m4_acc - m8_acc:+.3f})")
    print(f"At-risk problems (could flip): {len(at_risk)}  -> "
          f"{[r['pid'] for r in at_risk] if at_risk else 'none'}")
    print(f"\nToken estimate (probe 8 -> {TARGET_M}, refinement rounds unchanged):")
    print(f"  mean tokens  : {mean_old:>10.0f}  ->  {mean_new:>10.0f}   "
          f"({(1 - mean_new / mean_old) * 100:.1f}% saved)")
    print(f"  NOTE: far less than 50% because cost is dominated by chain length and "
          f"refinement rounds,\n        not probe width. Verify the % against your full 30-problem file.")
    print()


if __name__ == "__main__":
    main()