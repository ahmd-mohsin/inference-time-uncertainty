# Failure-Surface analysis — offline core of the "active recovery diagnosis" thesis. Uses ONLY the
# per-problem x per-option recovery matrix already collected (code_recover/taco_recover/math_recover),
# no GPU. Produces the award-target results that are computable now:
#
#  R1  Failure-Surface Diversity  D_fail = 1 - mean_{i<j} Corr(E_i, E_j)   (E_z = 1{option z FAILED})
#      -> does D_fail PREDICT portfolio gain across cells (math, HE, MBPP, TACO tiers x families),
#         while semantic proxies (option-entropy, #distinct-winning-options) do NOT?
#  R3  Adaptive vs static portfolio at matched budget B: greedy CONDITIONAL-coverage policy that picks
#      the next option using the population failure-correlation structure + outcomes observed so far,
#      vs a fixed uniform portfolio. (Ground-truth outcomes known per problem => exact simulation.)
#  R4  Marginal recovery coverage: greedy set-cover ordering of options; coverage curve F(|S|).
import json, glob, os, numpy as np
from itertools import combinations
PULLED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs_pulled")
from rl_training.code_passk import STRAT_NAMES

def cell_matrix(d):
    """failures x n_options binary recovery R (options auto-detected from THIS cell's switch strategies,
    so math's 14 strategies and code's 8 options each use their own set), plus iid recovery vector."""
    S = sorted({x["strategy"] for p in d.get("per_problem", []) for x in p.get("switch", [])})
    if not S: return np.zeros((0,0)), np.zeros(0)
    idx={m:j for j,m in enumerate(S)}; R=[]; IID=[]
    for p in d.get("per_problem", []):
        if any(p.get("default", [])): continue
        row=[0]*len(S)
        for x in p.get("switch", []):
            if x.get("correct"): row[idx[x["strategy"]]]=1
        R.append(row); IID.append(1 if any(p.get("iid_retry",[])) else 0)
    return np.array(R), np.array(IID)

def d_fail(R):
    """1 - mean pairwise error-correlation across options (only options with variance)."""
    E = 1 - R                                   # failure indicators
    keep = [j for j in range(E.shape[1]) if 0 < E[:,j].mean() < 1]
    if len(keep) < 2: return None, None
    C = np.corrcoef(E[:,keep].T)
    iu = np.triu_indices(len(keep),1)
    mean_corr = float(np.nanmean(C[iu]))
    return 1-mean_corr, mean_corr

def portfolio_gain(R, IID):
    """oracle (per-problem any-option) minus best single fixed option — the routable headroom;
    and 'any-option' minus iid (full-pool, optimistic)."""
    if len(R)==0: return None
    oracle = float((R.max(1)>0).mean())
    best_fixed = float(R.mean(0).max())
    return {"oracle_any": oracle, "best_fixed": best_fixed,
            "oracle_minus_bestfixed": oracle-best_fixed, "any_minus_iid": oracle-float(IID.mean()),
            "n": int(len(R))}

def option_entropy(R):
    """semantic-diversity proxy: entropy of which option is the (tie-broken) winner + # distinct winners."""
    if len(R)==0: return None, None
    win = R.argmax(1)[R.max(1)>0]
    if len(win)==0: return 0.0, 0
    _,cnt = np.unique(win, return_counts=True); p=cnt/cnt.sum()
    return float(-(p*np.log(p+1e-12)).sum()), int(len(cnt))

def adaptive_vs_static(R, B, trials=60, seed=0):
    """Both get B option-trials on default-failed problems, EQUAL budget. STATIC: B random distinct
    options. ADAPTIVE (active diagnosis): pick next option maximizing conditional recovery among the
    population still-unsolved by the options tried so far (uses the failure-correlation structure +
    outcomes observed on THIS problem). Population stats from R itself (LOO bias negligible at n>>1).
    Exact per-problem outcomes known => count solves. Vectorised over problems."""
    rng=np.random.default_rng(seed); n,M=R.shape
    if n<20 or B>=M: return None
    stat=ada=0.0
    for _ in range(trials):
        order=rng.permutation(M)[:B]
        stat += float((R[:,order].max(1)>0).mean())
        solved=np.zeros(n,bool); tried=np.full((n,M),False)
        # greedy adaptive: recompute a conditional score table each step from population still-unsolved
        for t in range(B):
            # per-problem: which options already tried (as a signature) -> but approximate with a single
            # population conditional given the *global* tried set is heterogeneous; use per-problem argmax
            # over options not yet tried, scored by population recovery among problems matching this
            # problem's observed outcome pattern on tried options.
            for i in np.where(~solved)[0]:
                cand=np.where(~tried[i])[0]
                if len(cand)==0: continue
                tr=np.where(tried[i])[0]
                if len(tr):
                    mask=(R[:,tr].max(1)==0)
                    sub=R[mask] if mask.sum()>=10 else R
                else: sub=R
                z=cand[int(np.argmax(sub[:,cand].mean(0)))]
                tried[i,z]=True
                if R[i,z]==1: solved[i]=True
        ada += float(solved.mean())
    return {"B":B,"static_portfolio":stat/trials,"adaptive_diag":ada/trials,
            "adaptive_minus_static":(ada-stat)/trials}

def marginal_coverage(R):
    """greedy set-cover ordering; coverage after k options."""
    if len(R)==0: return None
    n,M=R.shape; chosen=[]; covered=np.zeros(n,bool); curve=[]
    for _ in range(M):
        best=-1;bz=None
        for z in range(M):
            if z in chosen: continue
            gain=((R[:,z]==1)&(~covered)).sum()
            if gain>best: best=gain; bz=z
        chosen.append(bz); covered|=(R[:,bz]==1); curve.append(float(covered.mean()))
    return {"greedy_order":[STRAT_NAMES[z] for z in chosen],"coverage_curve":curve}

def load(glb):
    out=[]
    for fp in glob.glob(glb):
        try: d=json.load(open(fp))
        except Exception: continue
        if "per_problem" in d: out.append((os.path.basename(fp)[:-5], d))
    return out

if __name__=="__main__":
    cells=[]
    for glb,dom in [(f"{PULLED}/cro_router/rec_*.json","code"),(f"{PULLED}/taco_atlas/taco_*.json","code"),
                    (f"{PULLED}/math_negctrl/mrec_*.json","math")]:
        for name,d in load(glb):
            R,IID=cell_matrix(d)
            if len(R)<20: continue
            df,mc=d_fail(R); pg=portfolio_gain(R,IID); ent,ndist=option_entropy(R)
            cells.append({"cell":name,"domain":dom,"n":int(len(R)),"D_fail":df,"mean_err_corr":mc,
                          "opt_entropy":ent,"n_distinct_winners":ndist,**(pg or {})})
    # R1: correlate D_fail vs gain across cells; compare with entropy
    import numpy as np
    valid=[c for c in cells if c.get("D_fail") is not None and c.get("oracle_minus_bestfixed") is not None]
    def corr(xs,ys):
        if len(xs)<4: return None
        return float(np.corrcoef(xs,ys)[0,1])
    df=[c["D_fail"] for c in valid]; gain=[c["any_minus_iid"] for c in valid]   # gain = portfolio(any)-iid
    ent=[c["opt_entropy"] for c in valid]
    r_df=corr(df,gain); r_ent=corr(ent,gain)
    # spearman rank corr too (robust)
    def spear(xs,ys):
        if len(xs)<4: return None
        rx=np.argsort(np.argsort(xs)); ry=np.argsort(np.argsort(ys)); return corr(rx,ry)
    sp_df=spear(df,gain)
    print("==== per-cell failure-surface table ====")
    for c in sorted(valid,key=lambda x:-(x["D_fail"] or 0)):
        print(f"{c['cell']:<22} dom={c['domain']:<4} n={c['n']:<5} D_fail={c['D_fail']:.3f} "
              f"errCorr={c['mean_err_corr']:+.3f} portf-iid={c['any_minus_iid']:+.3f} "
              f"oracle-bestfix={c['oracle_minus_bestfixed']:+.3f} opt_H={c['opt_entropy']:.2f}")
    # domain means
    for dom in ("code","math"):
        ds=[c for c in valid if c["domain"]==dom]
        if ds: print(f"  [{dom}] mean D_fail={np.mean([c['D_fail'] for c in ds]):.3f} "
                     f"mean portf-iid={np.mean([c['any_minus_iid'] for c in ds]):+.3f} (n_cells={len(ds)})")
    print(f"\nR1  corr(D_fail, portfolio-iid) = {r_df}  spearman = {sp_df}  |  corr(opt_entropy, gain) = {r_ent}  (n_cells={len(valid)})")
    # R3/R4 on the big pooled code matrix
    codeR=[];
    for glb in [f"{PULLED}/cro_router/rec_*.json",f"{PULLED}/taco_atlas/taco_*.json"]:
        for _,d in load(glb):
            R,_=cell_matrix(d)
            if len(R): codeR.append(R)
    Rall=np.vstack(codeR)
    print(f"\n==== pooled CODE matrix: {Rall.shape[0]} failures x {Rall.shape[1]} options ====")
    mcov=marginal_coverage(Rall); print("R4 greedy coverage curve:", [f"{x:.3f}" for x in mcov["coverage_curve"]])
    print("   greedy option order:", mcov["greedy_order"])
    rng=np.random.default_rng(1); Rsim = Rall[rng.permutation(len(Rall))[:800]]  # subsample for speed
    for B in [2,3,4]:
        a=adaptive_vs_static(Rsim,B)
        if a: print(f"R3 B={B}: static={a['static_portfolio']:.3f} adaptive={a['adaptive_diag']:.3f} "
                    f"adaptive-static={a['adaptive_minus_static']:+.3f}")
    json.dump({"cells":cells,"R1_corr_Dfail_gain":r_df,"R1_corr_entropy_gain":r_ent,
               "R4_coverage":mcov},open(f"{PULLED}/failure_surface_result.json","w"),indent=2)
    print(f"\nsaved -> {PULLED}/failure_surface_result.json")
