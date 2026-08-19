#!/usr/bin/env python3
# E3 GPU scorer — teacher-force a checkpoint over a (clustered) bank and write per-trace logprobs.
# NO generation (cheap forward pass). Usage:
#   python score_bank_logprobs.py --model <ckpt> --bank bank_modes.jsonl --out scored_<arm>.jsonl
# bank lines need: {problem_id, mode_id, prompt, completion}. Emits {problem_id,mode_id,logp_theta,logp_ref?}.
import argparse, json, os, sys, torch
os.environ.setdefault("HF_HUB_DISABLE_XET","1")
from transformers import AutoModelForCausalLM, AutoTokenizer
ap=argparse.ArgumentParser()
ap.add_argument("--model",required=True); ap.add_argument("--bank",required=True); ap.add_argument("--out",required=True)
ap.add_argument("--field",default="logp_theta")   # name for this model's column
a=ap.parse_args()
tok=AutoTokenizer.from_pretrained(a.model); model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.bfloat16,device_map="cuda")
model.eval()
def seq_logprob(prompt, completion):
    pids=tok(prompt, return_tensors="pt").input_ids.cuda()
    full=tok(prompt+completion, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        logits=model(full).logits[:, :-1, :].log_softmax(-1)
    tgt=full[:,1:]
    tok_lp=logits.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)[0]
    # sum logprob over completion tokens only
    start=pids.shape[1]-1
    return float(tok_lp[start:].sum().item())
with open(a.bank) as f, open(a.out,"w") as w:
    for line in f:
        line=line.strip()
        if not line: continue
        d=json.loads(line)
        lp=seq_logprob(d["prompt"], d["completion"])
        rec={"problem_id":d["problem_id"],"mode_id":d.get("mode_id",0), a.field:lp}
        if "ref_logprob" in d: rec["logp_ref"]=d["ref_logprob"]
        w.write(json.dumps(rec)+"\n")
print("scored ->", a.out)
