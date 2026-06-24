# Offline spectral re-analysis of the AIME-2026 run.
#
# Motivation: the persistent-homology pipeline produced H1 features that are pure noise
# (8 points in ~5000-D -> distance concentration, CV~3% under DTW). This script tests the
# *alternative* signals proposed in docs/questions_and_directions.md on the SAME data,
# entirely offline from hidden_states.npz + chains_raw.json + validation.json:
#
#   - Direction 2: spectral effective rank of the trajectory ensemble (IID vs conditioned)
#   - Direction 10: NCD (surface compressibility) diversity
#   - baseline:     answer entropy / unique-answer count
#   - reference:    the original H1_n_features verdict
#
# It scores each signal against ground truth WITHOUT the `scales OR already_solved` rescue,
# and reports separately on the low-coverage subset (where scaling is actually measurable).
#
# Usage:
#   python -m topological_persistence.spectral_reanalysis \
#       --data-dir data/topological_outputs_aime2026

import argparse
import gzip
import json
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------------- signals

def effective_rank(point_matrix: np.ndarray) -> float:
    """Entropy-based (participation-ratio) effective rank of K x D mean-pooled chains.

    Low value => chains live in a collapsed low-dim subspace (Direction-2 ceiling signal).
    Mean-centered so we measure spread, not absolute position.
    """
    M = point_matrix - point_matrix.mean(axis=0, keepdims=True)
    s = np.linalg.svd(M, compute_uv=False)
    s2 = s ** 2
    if s2.sum() <= 0:
        return 0.0
    p = s2 / s2.sum()
    return float(np.exp(-(p * np.log(p + 1e-12)).sum()))


def energy_rank(point_matrix: np.ndarray, energy: float = 0.95) -> int:
    """# singular values needed to capture `energy` fraction of variance."""
    M = point_matrix - point_matrix.mean(axis=0, keepdims=True)
    s = np.linalg.svd(M, compute_uv=False)
    s2 = s ** 2
    if s2.sum() <= 0:
        return 0
    c = np.cumsum(s2) / s2.sum()
    return int(np.searchsorted(c, energy) + 1)


def answer_entropy(answers: list[str]) -> float:
    """Shannon entropy (nats) of the answer distribution; blanks dropped."""
    vals = [a for a in answers if a.strip()]
    if not vals:
        return 0.0
    _, counts = np.unique(vals, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log(p + 1e-12)).sum())


def ncd_mean(texts: list[str]) -> float:
    comp = [len(gzip.compress(t.encode("utf-8"))) for t in texts]
    n = len(texts)
    vals = []
    for i in range(n):
        for j in range(i + 1, n):
            cij = len(gzip.compress((texts[i] + texts[j]).encode("utf-8")))
            lo, hi = min(comp[i], comp[j]), max(comp[i], comp[j])
            vals.append((cij - lo) / max(hi, 1))
    return float(np.mean(vals)) if vals else 0.0


# ----------------------------------------------------------------------------- scoring

def pooled(H, pid: int, tag: str, n: int = 8) -> np.ndarray:
    return np.stack([H[f"{pid}_{tag}_{j}"].mean(axis=0) for j in range(n)])


def youden_threshold(scores: np.ndarray, labels: np.ndarray):
    """Best decision threshold maximizing (TPR - FPR); returns (thr, direction).

    direction=+1 means "score >= thr predicts positive (scalable)".
    """
    best = (-1.0, None, +1)
    order = np.unique(scores)
    for thr in order:
        for direction in (+1, -1):
            pred = (scores >= thr) if direction > 0 else (scores <= thr)
            tp = np.sum(pred & labels)
            fp = np.sum(pred & ~labels)
            fn = np.sum(~pred & labels)
            tn = np.sum(~pred & ~labels)
            tpr = tp / max(tp + fn, 1)
            fpr = fp / max(fp + tn, 1)
            j = tpr - fpr
            if j > best[0]:
                best = (j, float(thr), direction)
    return best[1], best[2], best[0]


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUC (P(score_pos > score_neg)); 0.5 = chance. Higher score => positive."""
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return float(wins / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/topological_outputs_aime2026")
    ap.add_argument("--low-coverage-thresh", type=int, default=40,
                    help="n_correct_of_N below this = problem with real headroom")
    args = ap.parse_args()

    d = Path(args.data_dir)
    H = np.load(d / "hidden_states.npz")
    raw = json.load(open(d / "chains_raw.json"))
    val = {v["problem_id"]: v for v in json.load(open(d / "validation.json"))}

    rows = []
    for pid in range(30):
        spid = str(pid)
        P = pooled(H, pid, "iid")
        Pc = pooled(H, pid, "cond")
        er = effective_rank(P)
        erc = effective_rank(Pc)
        iid_answers = [c["answer"] for c in raw[spid]["iid"]]
        iid_texts = [c["text"] for c in raw[spid]["iid"]]
        v = val[pid]
        rows.append({
            "pid": pid,
            "verdict": v["verdict"],
            "scales": bool(v["actually_scales"]),
            "solved8": bool(v["already_solved_at_8"]),
            "nc": v["n_correct_of_N"],
            "h1": v["h1_features"],
            "erank": er,
            "erank_cond": erc,
            "erank_gain": erc - er,
            "e95": energy_rank(P),
            "ans_entropy": answer_entropy(iid_answers),
            "n_unique": len({a for a in iid_answers if a.strip()}),
            "ncd": ncd_mean(iid_texts),
        })

    # ----- table
    hdr = (f"{'pid':>3}{'verdict':>9}{'scl':>4}{'nc':>4}{'H1':>3}"
           f"{'eRank':>7}{'eRgain':>7}{'e95':>4}{'Hans':>6}{'uniq':>5}{'NCD':>6}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['pid']:>3}{r['verdict'][:8]:>9}{('Y' if r['scales'] else '.'):>4}"
              f"{r['nc']:>4}{r['h1']:>3}{r['erank']:>7.2f}{r['erank_gain']:>+7.2f}"
              f"{r['e95']:>4}{r['ans_entropy']:>6.2f}{r['n_unique']:>5}{r['ncd']:>6.3f}")

    # ----- signal evaluation
    # Ground-truth label set 1: actually_scales (only 3 positives -> weak but report).
    # Ground-truth label set 2: "solvable-with-compute" coverage proxy -- n_correct_of_N as
    #   a CONTINUOUS difficulty target; report Spearman-like rank corr of each signal vs nc.
    def rankcorr(x, y):
        from scipy.stats import spearmanr
        return float(spearmanr(x, y).correlation)

    nc = np.array([r["nc"] for r in rows], float)
    print("\n=== Signal vs problem coverage (n_correct_of_N), Spearman rho ===")
    print("    (a real ceiling signal should track difficulty: low diversity <-> low coverage)")
    for key, label in [("erank", "effective_rank"), ("e95", "energy95_rank"),
                       ("h1", "H1_features"), ("ans_entropy", "answer_entropy"),
                       ("n_unique", "unique_answers"), ("ncd", "NCD_mean")]:
        x = np.array([r[key] for r in rows], float)
        print(f"  {label:>16}: rho(signal, coverage) = {rankcorr(x, nc):+.3f}")

    # ----- focus subset: real headroom
    low = [r for r in rows if r["nc"] < args.low_coverage_thresh]
    print(f"\n=== Low-coverage subset (n_correct_of_N < {args.low_coverage_thresh}): "
          f"{len(low)} problems, {sum(r['scales'] for r in low)} actually scale ===")
    lhdr = f"{'pid':>3}{'scl':>4}{'nc':>4}{'H1':>3}{'eRank':>7}{'Hans':>6}{'uniq':>5}{'NCD':>6}"
    print(lhdr)
    for r in low:
        print(f"{r['pid']:>3}{('Y' if r['scales'] else '.'):>4}{r['nc']:>4}{r['h1']:>3}"
              f"{r['erank']:>7.2f}{r['ans_entropy']:>6.2f}{r['n_unique']:>5}{r['ncd']:>6.3f}")

    # ----- binary AUC on actually_scales (caveat: tiny positive set)
    labels = np.array([r["scales"] for r in rows])
    print(f"\n=== AUC vs actually_scales ({labels.sum()}/{len(labels)} positive) "
          f"-- HIGHER score predicts scalable ===")
    print("    (caveat: only 3 positives in full set; treat as directional, not significant)")
    for key, label, flip in [("erank", "effective_rank", False),
                             ("h1", "H1_features", False),
                             ("ans_entropy", "answer_entropy", False),
                             ("n_unique", "unique_answers", False),
                             ("ncd", "NCD_mean", False)]:
        x = np.array([r[key] for r in rows], float)
        a = auc(x, labels)
        print(f"  {label:>16}: AUC = {a:.3f}")

    out = d / "spectral_reanalysis.json"
    json.dump(rows, open(out, "w"), indent=2)
    print(f"\nSaved per-problem signals to {out}")


if __name__ == "__main__":
    main()
