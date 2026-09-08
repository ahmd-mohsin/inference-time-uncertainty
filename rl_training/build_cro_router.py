# CRO experiment #1 — the ROUTER. Offline/local. Answers: "is there a LEARNABLE failure->recovery
# decision problem?" i.e. can a failure-conditioned gate g_phi(z|s_F) predict, from the failed state
# (question + failed code + error signature), WHICH recovery option to switch to — and thereby close a
# meaningful fraction of the oracle-minus-best-fixed headroom the Atlas found (+0.22 MBPP, +0.05-0.13 TACO)?
#
# Data = pooled default-FAILED problems from code_recover (rec_*.json) and taco_recover (taco_*.json),
# both of which carry per-strategy recovery outcomes (switch:[{correct,strategy}]) + router features
# (question, fail_code, fail_err) over the SAME 8 engineering recovery options.
#
# Metric (over failures): best_fixed = max_z mean_i r_iz (best SINGLE strategy for everyone);
# oracle = mean_i max_z r_iz (per-problem best); iid = mean_i any(iid_retry);
# router = CROSS-VALIDATED routed recovery (predict z from features on held-out folds, take r_{i,z_hat}).
# gap_closed = (router - best_fixed) / (oracle - best_fixed).  GO if >= 0.5-0.6.
import json, glob, os, sys, numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

PULLED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs_pulled")

def load_cells(globs):
    cells = []
    for g in globs:
        for fp in glob.glob(g):
            try:
                d = json.load(open(fp))
            except Exception:
                continue
            if "per_problem" in d:
                cells.append((os.path.basename(fp), d))
    return cells

def per_strategy_recovery(p, strat_names):
    """r_z in {0,1} per strategy = did ANY switch sample with strategy z pass; plus iid recovery."""
    by = {m: 0 for m in strat_names}
    for x in p.get("switch", []):
        if x.get("correct"):
            by[x["strategy"]] = 1
    iid = 1 if any(p.get("iid_retry", [])) else 0
    return by, iid

def matched_budget_portfolio(SWP, IIP, trials=300, seed=0):
    """Fair test: at EQUAL budget B (=min pool size), does a diversified portfolio (sample B from the
    across-strategy switch pool) recover more failures than iid-retry (sample B from the iid pool)?"""
    rng = np.random.default_rng(seed)
    ports, iids, Bs = [], [], []
    for sw, ii in zip(SWP, IIP):
        if not sw or not ii:
            continue
        B = min(len(sw), len(ii))
        if B == 0:
            continue
        Bs.append(B)
        ph = ih = 0
        for _ in range(trials):
            ph += 1 if any(rng.choice(sw, B, replace=False)) else 0
            ih += 1 if any(rng.choice(ii, B, replace=False)) else 0
        ports.append(ph/trials); iids.append(ih/trials)
    if not ports:
        return None
    port = float(np.mean(ports)); iid = float(np.mean(iids))
    return {"B_median": int(np.median(Bs)), "portfolio_matched": port, "iid_matched": iid,
            "portfolio_minus_iid": port-iid, "n": len(ports)}

def build(globs, label, strat_names):
    cells = load_cells(globs)
    X_text, R, IID, srcs, SWP, IIP = [], [], [], [], [], []
    for name, d in cells:
        for p in d["per_problem"]:
            if any(p.get("default", [])):      # only DEFAULT-FAILED problems
                continue
            if not p.get("fail_code") and not p.get("question"):
                continue
            by, iid = per_strategy_recovery(p, strat_names)
            feat = f"{p.get('question','')}\n<ERR>{p.get('fail_err','')}\n<CODE>{p.get('fail_code','')}"
            X_text.append(feat); R.append([by[m] for m in strat_names]); IID.append(iid); srcs.append(name)
            SWP.append([bool(x.get("correct")) for x in p.get("switch", [])])
            IIP.append([bool(x) for x in p.get("iid_retry", [])])
    R = np.array(R); IID = np.array(IID); n = len(R)
    if n == 0:
        print(f"[{label}] no failed problems"); return None
    mbp = matched_budget_portfolio(SWP, IIP)
    best_fixed_vec = R.mean(0)                          # mean recovery per strategy
    best_fixed = best_fixed_vec.max(); bf_arg = strat_names[int(best_fixed_vec.argmax())]
    oracle = R.max(1).mean()                            # per-problem best
    iid_mean = IID.mean()
    routable = (R.max(1) > 0)                           # a switch option recovers
    # ---- learned router: features -> argmax strategy, cross-validated ----
    y = R.argmax(1)                                     # label = best recovery option (ties -> first)
    vec = TfidfVectorizer(max_features=4000, ngram_range=(1,2), sublinear_tf=True, min_df=2)
    Xtf = vec.fit_transform(X_text)
    routed_hits = np.zeros(n)
    # stratify only on classes with >=2 members; fall back to plain KFold-ish via label grouping
    cls_ok = np.array([c for c,ct in Counter(y).items() if ct >= 5])
    mask = np.isin(y, cls_ok)
    routed_recovery = None; per_fold = []
    if mask.sum() >= 30 and len(cls_ok) >= 2:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        idx_all = np.arange(n)
        for tr, te in skf.split(Xtf[mask], y[mask]):
            tr_i = idx_all[mask][tr]; te_i = idx_all[mask][te]
            clf = LogisticRegression(max_iter=2000, C=2.0, class_weight="balanced")
            clf.fit(Xtf[tr_i], y[tr_i])
            pred = clf.predict(Xtf[te_i])
            routed_hits[te_i] = R[te_i, pred]
            per_fold.append(float(R[te_i, pred].mean()))
        # routed recovery over the routable-eval population (held-out only)
        routed_recovery = float(routed_hits[mask].mean())
        eval_oracle = float(R[mask].max(1).mean())
        eval_bestfixed = float(R[mask].mean(0).max())
        eval_iid = float(IID[mask].mean())
    res = {"label": label, "n_failures": int(n), "n_routable": int(routable.sum()),
           "frac_routable": float(routable.mean()), "matched_budget_portfolio": mbp, "strategies": strat_names,
           "best_fixed": float(best_fixed), "best_fixed_strategy": bf_arg,
           "oracle": float(oracle), "iid_retry": float(iid_mean),
           "strategy_win_counts": {strat_names[i]: int((y==i).sum()) for i in range(len(strat_names))},
           "routed_recovery_cv": routed_recovery,
           "eval_population": None if routed_recovery is None else {
               "n": int(mask.sum()), "oracle": eval_oracle, "best_fixed": eval_bestfixed,
               "iid_retry": eval_iid, "routed_cv": routed_recovery,
               "gap_closed_over_bestfixed": (routed_recovery-eval_bestfixed)/(eval_oracle-eval_bestfixed) if eval_oracle>eval_bestfixed else None,
               "router_minus_iid": routed_recovery-eval_iid,
               "per_fold": per_fold}}
    return res

def report(res):
    if not res: return
    print(f"\n===== {res['label']} =====")
    print(f"failures={res['n_failures']} routable={res['n_routable']} ({res['frac_routable']:.2%})")
    mbp = res.get("matched_budget_portfolio")
    if mbp: print(f"MATCHED-BUDGET PORTFOLIO (B={mbp['B_median']}, n={mbp['n']}): portfolio={mbp['portfolio_matched']:.3f} "
                  f"iid={mbp['iid_matched']:.3f}  PORTFOLIO−IID={mbp['portfolio_minus_iid']:+.3f}")
    print(f"iid_retry={res['iid_retry']:.3f}  best_fixed={res['best_fixed']:.3f} ({res['best_fixed_strategy']})  oracle={res['oracle']:.3f}")
    print(f"strategy win-counts: {sorted(res['strategy_win_counts'].items(), key=lambda x:-x[1])}")
    ep = res.get("eval_population")
    if ep:
        print(f"[held-out CV over routable classes, n={ep['n']}] iid={ep['iid_retry']:.3f} best_fixed={ep['best_fixed']:.3f} "
              f"ROUTED={ep['routed_cv']:.3f} oracle={ep['oracle']:.3f}")
        gc = ep['gap_closed_over_bestfixed']
        print(f"  router-iid={ep['router_minus_iid']:+.3f}  GAP CLOSED (router vs bestfixed, / oracle-bestfix) = "
              f"{'n/a' if gc is None else f'{gc:.1%}'}  per_fold={[f'{x:.2f}' for x in ep['per_fold']]}")
    else:
        print("  (too few routable samples for CV router)")

if __name__ == "__main__":
    from rl_training.code_passk import STRAT_NAMES
    out = {}
    out["CODE_mbpp+he"] = build([f"{PULLED}/cro_router/rec_*.json"], "CODE MBPP+HumanEval (router families)", STRAT_NAMES)
    out["CODE_taco"]    = build([f"{PULLED}/taco_atlas/taco_*.json"], "CODE TACO (competitive prog)", STRAT_NAMES)
    out["CODE_all"]     = build([f"{PULLED}/cro_router/rec_*.json", f"{PULLED}/taco_atlas/taco_*.json"], "CODE ALL pooled", STRAT_NAMES)
    out["MATH_negctrl"] = build([f"{PULLED}/math_negctrl/mrec_*.json"], "MATH (negative control)", STRAT_NAMES)
    for k in out: report(out[k])
    json.dump({k:v for k,v in out.items()}, open(f"{PULLED}/cro_router_result.json","w"), indent=2)
    print(f"\nsaved -> {PULLED}/cro_router_result.json")
