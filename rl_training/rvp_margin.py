"""Mechanistic-interp M1/M3: teacher-forced completion logprob of chosen(y+) vs rejected(y-) under a checkpoint.
Reports mean logp(y+), logp(y-), margin m=logp(y+)-logp(y-). Run on the SAME held-out pairs for RFT vs RVP vs shuf
-> Delta-margin shows mass reallocation (RVP raises m; decompose into +logp(y+) vs -logp(y-))."""
import argparse, json, os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
def seq_logp(model, tok, prompt, completion, device):
    try:
        pre = tok.apply_chat_template([{"role":"user","content":prompt}], tokenize=True, add_generation_prompt=True)
    except Exception:
        pre = tok(prompt+"\n", add_special_tokens=False)["input_ids"]
    comp = tok(completion, add_special_tokens=False)["input_ids"]
    ids = (pre+comp)[:1024]; n_comp = min(len(comp), max(0,1024-len(pre)))
    if n_comp<=0: return None
    x = torch.tensor([ids], device=device)
    with torch.no_grad():
        lg = model(x).logits[0].float().log_softmax(-1)
    tot=0.0
    for i in range(len(pre), len(ids)):
        tot += lg[i-1, ids[i]].item()
    return tot/ max(1,(len(ids)-len(pre)))  # per-token mean logp
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",required=True); ap.add_argument("--pairs",required=True); ap.add_argument("--n",type=int,default=150); ap.add_argument("--tag",default="m")
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok=AutoTokenizer.from_pretrained(a.model,trust_remote_code=True)
    model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.bfloat16,device_map="cuda",trust_remote_code=True).eval()
    dev=next(model.parameters()).device
    rows=[json.loads(l) for l in open(a.pairs) if l.strip()][:a.n]
    lp_pos=[]; lp_neg=[]; marg=[]
    for r in rows:
        lp=seq_logp(model,tok,r["prompt"],r["chosen"],dev); ln=seq_logp(model,tok,r["prompt"],r["rejected"],dev)
        if lp is None or ln is None: continue
        lp_pos.append(lp); lp_neg.append(ln); marg.append(lp-ln)
    res={"tag":a.tag,"model":a.model,"n":len(marg),
         "logp_pos":sum(lp_pos)/max(1,len(lp_pos)),"logp_neg":sum(lp_neg)/max(1,len(lp_neg)),
         "margin":sum(marg)/max(1,len(marg))}
    json.dump(res,open(a.out,"w"),indent=2)
    print(f"[margin {a.tag}] n={res['n']} logp+={res['logp_pos']:.3f} logp-={res['logp_neg']:.3f} margin={res['margin']:.3f}")
if __name__=="__main__": main()
