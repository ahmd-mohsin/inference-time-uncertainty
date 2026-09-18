#!/usr/bin/env python3
"""Probability-mass accounting (addresses the "margin growth != total correct-mass increase" critique).
For each checkpoint (base/rft/rvp/rft2), teacher-force the POOLED verified candidate set per prompt and
compute the AGGREGATE correct-vs-incorrect margin over that support:
    M(x) = logsumexp_{y in correct} logpi(y|x)  -  logsumexp_{y in incorrect} logpi(y|x)
    restricted correct-mass  s(x) = sigma(M(x))   (fraction of pooled-support mass on correct answers)
Reports mean M and mean s per checkpoint. Tests: (1) does RVP raise aggregate correct-mass vs RFT
(not just the per-pair margin)? (2) does ReST (rft2) reach the same aggregate margin by a different
per-mode trajectory? Uses pairs.jsonl (chosen=verified-correct, rejected=verified-incorrect).

Usage: python3 -m rl_training.mass_accounting --models base=<hf>,rft=<dir>,rvp=<dir>,rft2=<dir> \
    --data pairs.jsonl --n 80 --out mass.json
"""
import argparse, json, math, os, sys
import torch
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer

def seq_logprob(model, tok, prompt, completion, device):
    """sum log p(completion | prompt) under teacher forcing."""
    pids = tok(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(device)
    cids = tok(completion, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    if cids.shape[1] == 0: return -1e9
    ids = torch.cat([pids, cids], dim=1)
    with torch.no_grad():
        logits = model(ids).logits.float()
    logp = torch.log_softmax(logits[0, :-1], dim=-1)
    tgt = ids[0, 1:]
    tok_lp = logp[torch.arange(tgt.shape[0]), tgt]
    comp_lp = tok_lp[pids.shape[1]-1:]  # completion tokens only
    return float(comp_lp.sum().item())

def logsumexp(xs):
    if not xs: return -1e9
    m = max(xs); return m + math.log(sum(math.exp(x-m) for x in xs))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="comma list name=path")
    ap.add_argument("--data", required=True); ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    # pool candidates by prompt
    pos = defaultdict(set); neg = defaultdict(set)
    for line in open(a.data):
        try: r = json.loads(line)
        except: continue
        p = r.get("prompt"); c = r.get("chosen"); j = r.get("rejected")
        if p is None: continue
        if c: pos[p].add(c)
        if j: neg[p].add(j)
    prompts = [p for p in pos if pos[p] and neg[p]][:a.n]
    print(f"[mass] {len(prompts)} prompts with both correct & incorrect pooled candidates")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    res = {}
    for spec in a.models.split(","):
        name, path = spec.split("=", 1)
        if name != "base" and not os.path.exists(os.path.join(path, "config.json")):
            print(f"[mass] {name}: missing {path}, skip"); continue
        print(f"[mass] loading {name} <- {path}")
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device).eval()
        Ms = []; Ss = []; POS=[]; NEG=[]
        for p in prompts:
            cp = [seq_logprob(model, tok, p, y, device) for y in pos[p]]
            cn = [seq_logprob(model, tok, p, y, device) for y in neg[p]]
            lp = logsumexp(cp); ln = logsumexp(cn); M = lp - ln
            Ms.append(M); Ss.append(1.0/(1.0+math.exp(-max(min(M,30),-30))))
            POS.append(lp); NEG.append(ln)
        res[name] = {"mean_aggregate_margin": sum(Ms)/len(Ms),
                     "mean_restricted_correct_mass": sum(Ss)/len(Ss),
                     "mean_logsumexp_correct": sum(POS)/len(POS),
                     "mean_logsumexp_incorrect": sum(NEG)/len(NEG), "n": len(Ms)}
        print(f"[mass] {name}: M={res[name]['mean_aggregate_margin']:.3f}  correct-mass={res[name]['mean_restricted_correct_mass']:.3f}  "
              f"lse+={res[name]['mean_logsumexp_correct']:.2f} lse-={res[name]['mean_logsumexp_incorrect']:.2f}")
        del model; torch.cuda.empty_cache()
    json.dump(res, open(a.out, "w"), indent=2)
    print(f"[mass] wrote {a.out}")

if __name__ == "__main__": main()
