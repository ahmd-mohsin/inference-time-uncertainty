#!/usr/bin/env python3
"""Cost-reliability frontier (P0.1): compare RVP@1 to the BASE model's inference-time selection
baselines (self-consistency majority-vote@n, and oracle best-of-n) at matched sample budget n.
Reads SAVE_SAMPLES eval jsons (per_problem has 'ans'/'ok'). Answers the reviewer's
"why not just best-of-n / self-consistency?" objection: shows RVP bakes the selection gain into
n=1 (a single forward pass), matching base@self-consistency-n for some n>1.

Usage: python3 -m rl_training.rvp_scripts.selectbench --base ev_base.json --rvp ev_rvp.json [--draws 200] [--out frontier.json]
"""
import argparse, json, random, math
from collections import Counter

def load(p): return json.load(open(p))

def maj_at_n(per, n, draws, rng):
    """Self-consistency: for each problem, draw n samples, majority-vote the answer string,
    correct iff the voted answer is a correct one. Averaged over `draws` random subsets."""
    accs=[]
    for pr in per:
        ans=pr.get("ans"); ok=pr.get("ok")
        if not ans: accs.append(pr["c"]/pr["k"]); continue
        k=len(ans); idx=list(range(k))
        if n>=k:  # deterministic: vote over all k
            reps=[(_vote(ans,ok,idx))]
        else:
            reps=[_vote(ans,ok,rng.sample(idx,n)) for _ in range(draws)]
        accs.append(sum(reps)/len(reps))
    return sum(accs)/len(accs)

def _vote(ans, ok, sel):
    cnt=Counter(ans[i] for i in sel)
    top=cnt.most_common(1)[0][0]  # ties -> first-most-common (Counter is insertion-stable)
    # voted answer is correct iff any selected sample with that answer string was verified correct
    return 1.0 if any(ans[i]==top and ok[i]==1 for i in sel) else 0.0

def bestofn_oracle(per, n):
    """Perfect-verifier best-of-n = P(at least one correct in n draws) = 1 - C(k-c,n)/C(k,n)."""
    accs=[]
    for pr in per:
        c,k=pr["c"],pr["k"]
        if n>=k: accs.append(1.0 if c>0 else 0.0); continue
        accs.append(1.0 - (math.comb(k-c,n)/math.comb(k,n) if k-c>=n else 0.0))
    return sum(accs)/len(accs)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base",required=True); ap.add_argument("--rvp",required=True)
    ap.add_argument("--draws",type=int,default=200); ap.add_argument("--out",default=None)
    ap.add_argument("--tag",default=""); a=ap.parse_args()
    rng=random.Random(0)
    b=load(a.base); r=load(a.rvp); per=b["per_problem"]; k=b["k"]
    if not per or "ans" not in per[0]:
        print("[selectbench] base json lacks per-sample 'ans' (re-eval with SAVE_SAMPLES=1)"); return
    ns=[n for n in (1,2,4,8,16,32) if n<=k]
    maj={n:maj_at_n(per,n,a.draws,rng) for n in ns}
    boN={n:bestofn_oracle(per,n) for n in ns}
    rvp1=r["pass1"]; base1=b["pass1"]
    # smallest n where base self-consistency reaches RVP@1
    n_star=next((n for n in ns if maj[n]>=rvp1), None)
    res={"tag":a.tag,"k":k,"base_pass1":base1,"rvp_pass1":rvp1,
         "base_majvote_at_n":maj,"base_bestofn_oracle_at_n":boN,
         "rvp_at_1_equals_base_selfconsistency_at_n":n_star}
    print(f"[selectbench {a.tag}] base@1={base1:.4f}  RVP@1={rvp1:.4f}")
    for n in ns: print(f"   n={n:>2}: base maj@n={maj[n]:.4f}  base bestof-n(oracle)={boN[n]:.4f}")
    print(f"   => RVP@1 matches base self-consistency@n for n={n_star} (None=RVP@1 exceeds base maj@{k})")
    if a.out: json.dump(res,open(a.out,"w"),indent=2)

if __name__=="__main__": main()
