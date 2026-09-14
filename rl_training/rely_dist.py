"""Reliability-distribution sharpening: from per-problem c/k eval jsons, show how the per-prompt success prob p_hat
shifts base->RFT->RVP. RVP should move mass low->high (concentration). Fields: mean p1, frac(p_hat>0.5), entropy."""
import json,sys,math,glob
def stats(f):
    d=json.load(open(f)); pp=d.get("per_problem",[])
    ph=[(x.get("c",0)/x.get("k",1)) for x in pp if "c" in x]
    if not ph: return None
    n=len(ph); m=sum(ph)/n; hi=sum(1 for p in ph if p>0.5)/n; lo=sum(1 for p in ph if p<0.1)/n
    # binary entropy averaged (uncertainty per prompt); RVP should lower it (more decisive)
    H=sum(-(p*math.log(p+1e-9)+(1-p)*math.log(1-p+1e-9)) for p in ph)/n
    return dict(n=n,pass1=round(m,3),frac_hi=round(hi,3),frac_lo=round(lo,3),entropy=round(H,3))
for f in sys.argv[1:]:
    s=stats(f); tag=f.split("comp_")[-1].split(".json")[0] if "comp_" in f else f
    print(tag, s)
