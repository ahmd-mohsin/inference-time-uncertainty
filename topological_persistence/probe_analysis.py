# Probe + multi-layer analysis (Direction 12 + Direction 2), offline / CPU.
#
# Tests the central intuition from the Qwen3-8B run: on hard problems the model is
# CONFIDENTLY WRONG (8 chains agree on a wrong answer), so answer-space signals are blind
# (AUC ~0.5). The only thing that worked was hidden-state effective rank (AUC 0.80, last
# layer). This script asks two precise questions on the SAVED hidden states:
#
#   Q-A (multi-layer effective rank, D2): which layer's effective rank best predicts
#        whether a problem actually_scales? (mid / 3-quarter / last)
#   Q-B (internal-correctness probe, D12): can a linear probe on a chain's pooled hidden
#        state predict whether THAT chain is correct? If yes -> the model "knows"
#        internally even when it emits wrong tokens -> sampling/selection can recover it.
#        Then: does mean probe-confidence on a problem's chains predict actually_scales?
#
# Inputs (merged): hidden_states.npz (with `{pid}_{tag}_{j}` last-layer seq AND
# `{pid}_{tag}_{j}__ml` = (3,hidden) mid/3q/last pooled), chains_raw.json, validation.json.
#
# Usage: python -m topological_persistence.probe_analysis --data-dir <dir>

import argparse, json
from pathlib import Path
import numpy as np

LAYER_NAMES = ["mid", "three_quarter", "last"]


def eff_rank(M):
    if M.ndim != 2 or M.shape[0] < 2: return 0.0
    M = M - M.mean(0, keepdims=True)
    s = np.linalg.svd(M, compute_uv=False); s2 = s ** 2
    if s2.sum() <= 0: return 0.0
    p = s2 / s2.sum()
    return float(np.exp(-(p * np.log(p + 1e-12)).sum()))


def auc(scores, labels):
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    return float(sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg)))


def answers_match(pred, gold):
    if not pred or not gold: return False
    pred, gold = pred.strip().strip("$").rstrip(".,"), gold.strip().strip("$").rstrip(".,")
    if pred == gold: return True
    try: return abs(float(pred) - float(gold)) < 1e-6
    except: return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/topological_outputs_aime2026_qwen8b")
    args = ap.parse_args()
    d = Path(args.data_dir)
    H = np.load(d / "hidden_states.npz")
    raw = json.load(open(d / "chains_raw.json"))
    val = {v["problem_id"]: v for v in json.load(open(d / "validation.json"))}
    has_ml = any(k.endswith("__ml") for k in H.files)
    pids = sorted(int(k) for k in raw if int(k) in val)
    print(f"problems: {len(pids)} | multilayer states present: {has_ml}")

    # ---------- Q-A: multi-layer effective rank vs actually_scales ----------
    # per problem, per layer: eff rank of the 8 IID pooled vectors
    scales = []
    er_by_layer = {ln: [] for ln in LAYER_NAMES}
    er_last_tokseq = []  # the original last-layer-token-seq effective rank (what scored 0.80)
    for pid in pids:
        scales.append(bool(val[pid]["actually_scales"]))
        n = len(raw[str(pid)]["iid"])
        if has_ml:
            ml = np.stack([H[f"{pid}_iid_{j}__ml"] for j in range(n)])  # (n, 3, hidden)
            for li, ln in enumerate(LAYER_NAMES):
                er_by_layer[ln].append(eff_rank(ml[:, li, :]))
        # last-layer token-seq pooled to one vector/chain (matches prior run)
        pooled_last = np.stack([H[f"{pid}_iid_{j}"].mean(0) for j in range(n)])
        er_last_tokseq.append(eff_rank(pooled_last))

    print("\n=== Q-A: effective rank vs actually_scales (LOW rank -> scalable) ===")
    print(f"  {'last_tokseq(prev signal)':>26}: AUC={auc([-x for x in er_last_tokseq], scales):.3f}")
    if has_ml:
        for ln in LAYER_NAMES:
            print(f"  {('pooled_'+ln):>26}: AUC={auc([-x for x in er_by_layer[ln]], scales):.3f}")

    # ---------- Q-B: internal-correctness probe (D12) ----------
    if has_ml:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        # build per-chain dataset: x = pooled hidden (concat 3 layers), y = chain correct?
        X, y, grp = [], [], []
        for pid in pids:
            gold = raw[str(pid)]["gold_answer"]
            for j, c in enumerate(raw[str(pid)]["iid"]):
                X.append(H[f"{pid}_iid_{j}__ml"].reshape(-1))
                y.append(answers_match(c["answer"], gold))
                grp.append(pid)
        X, y, grp = np.array(X), np.array(y), np.array(grp)
        print(f"\n=== Q-B: per-chain internal-correctness probe (D12) ===")
        print(f"  chains: {len(y)} | correct: {y.sum()} | wrong: {(~y).sum()}")
        if 0 < y.sum() < len(y):
            # leave-one-PROBLEM-out CV (no train/test leakage across a problem's chains)
            preds = np.full(len(y), np.nan)
            for pid in pids:
                te = grp == pid; tr = ~te
                if y[tr].sum() == 0 or y[tr].sum() == tr.sum():  # need both classes to train
                    continue
                sc = StandardScaler().fit(X[tr])
                clf = LogisticRegression(max_iter=2000, C=0.1).fit(sc.transform(X[tr]), y[tr])
                preds[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
            ok = ~np.isnan(preds)
            print(f"  probe AUC (per-chain correctness, LOPO-CV): {auc(preds[ok], y[ok]):.3f}")
            # KEY: does mean probe-confidence on WRONG chains predict that the problem scales?
            conf_wrong, lab = [], []
            for pid in pids:
                m = (grp == pid) & (~y) & ok
                if m.sum() == 0: continue
                conf_wrong.append(float(np.mean(preds[m])))   # how 'right-looking' the wrong chains are
                lab.append(bool(val[pid]["actually_scales"]))
            if len(set(lab)) > 1:
                print(f"  mean probe-conf on WRONG chains vs actually_scales: AUC={auc(conf_wrong, lab):.3f}")
                print(f"    (high -> wrong chains internally 'know' the answer -> recoverable -> scalable)")
        else:
            print("  (degenerate: all chains same correctness class)")

    out = d / "probe_analysis.json"
    json.dump({"pids": pids, "scales": [bool(s) for s in scales],
               "eff_rank_last_tokseq": er_last_tokseq,
               "eff_rank_by_layer": er_by_layer if has_ml else {}}, open(out, "w"), indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
