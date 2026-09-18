"""Math-domain (GSM8K) RVP: bank/pairs/eval via answer-match verify. Mirrors comp RVP for a 2nd dataset."""
import argparse, json, os, sys, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.dataset import (load_gsm8k, load_math500, load_math_full, load_competition_math,
    load_deepmath, load_aime_all, load_amc, load_omni_math, load_olympiad_bench,
    format_prompt, extract_numeric_answer, extract_boxed_answer, answers_match)
import signal as _signal
class _VerifyTimeout(Exception): pass
def _verify_alarm(signum, frame): raise _VerifyTimeout()
def verify(text, gold):
    # try boxed-latex extraction first (MATH/Olympiad), then numeric (GSM8K/AIME/AMC).
    # answers_match uses sympy, which can HANG forever on pathological Olympiad expressions;
    # guard every match with a hard SIGALRM timeout so one bad problem can't stall the whole eval.
    for extract in (extract_boxed_answer, extract_numeric_answer):
        try:
            old = _signal.signal(_signal.SIGALRM, _verify_alarm)
            _signal.setitimer(_signal.ITIMER_REAL, 2.0)
            try:
                if answers_match(extract(text), gold): return True
            finally:
                _signal.setitimer(_signal.ITIMER_REAL, 0)
                _signal.signal(_signal.SIGALRM, old)
        except Exception: pass
    return False
LOADERS = {
    "gsm8k":           lambda a: load_gsm8k(split=a.split, n_problems=a.n, seed=a.seed),
    "math500":         lambda a: load_math500(split="test", n_problems=a.n),
    "math_full":       lambda a: load_math_full(split="train", n_problems=a.n, seed=a.seed),
    "competition_math":lambda a: load_competition_math(n_problems=a.n, seed=a.seed),
    "deepmath":        lambda a: load_deepmath(split="train", n_problems=a.n, seed=a.seed),
    "aime":            lambda a: load_aime_all(n_problems=a.n),
    "amc":             lambda a: load_amc(n_problems=a.n),
    "omni_math":       lambda a: load_omni_math(n_problems=a.n, seed=a.seed, min_difficulty=float(os.environ.get("OMNI_MINDIFF","0"))),
    "olympiad_bench":  lambda a: load_olympiad_bench(n_problems=a.n, seed=a.seed),
}
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",required=True,choices=["bank","pairs","eval","margin"])
    ap.add_argument("--model",required=True); ap.add_argument("--split",default="train")
    ap.add_argument("--dataset",default="gsm8k",choices=list(LOADERS.keys()))
    ap.add_argument("--n",type=int,default=500); ap.add_argument("--k",type=int,default=8)
    ap.add_argument("--temperature",type=float,default=0.8); ap.add_argument("--max-pairs-per",type=int,default=2)
    ap.add_argument("--shuffle",action="store_true"); ap.add_argument("--out",required=True); ap.add_argument("--seed",type=int,default=1)
    ap.add_argument("--data",default=None)  # margin mode: pairs.jsonl with prompt/chosen/rejected
    ap.add_argument("--hard-neg",action="store_true")  # pairs: pick hardest (highest model-logprob) y-, best y+
    ap.add_argument("--num-shards",type=int,default=1); ap.add_argument("--shard-index",type=int,default=0)  # data-parallel gen
    a=ap.parse_args()
    # --- margin mode: teacher-force y+/y- under the model, report mean per-token logp + logit margin ---
    # (mechanism panel: Prop 2 signature = margin up via logp(y-) down, no generation, HF not vLLM)
    if a.mode=="margin":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok=AutoTokenizer.from_pretrained(a.model); tok.pad_token=tok.pad_token or tok.eos_token
        model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.bfloat16,device_map="cuda").eval()
        def comp_logp(prompt,completion):
            pids=tok(prompt,return_tensors="pt",add_special_tokens=True).input_ids
            fids=tok(prompt+completion,return_tensors="pt",add_special_tokens=True).input_ids[:, :int(os.environ.get("DPO_MAXLEN","1024"))]
            with torch.no_grad(): logits=model(fids.to("cuda")).logits[0,:-1].float()
            lp=torch.log_softmax(logits,-1); tgt=fids[0,1:].to("cuda")
            tl=lp.gather(-1,tgt.unsqueeze(-1)).squeeze(-1); cs=max(pids.shape[1]-1,0)
            seg=tl[cs:]; return seg.mean().item() if seg.numel() else 0.0
        pairs=[json.loads(l) for l in open(a.data)][:a.n]
        sp=sn=sm=0.0; nn=0
        for r in pairs:
            lp_pos=comp_logp(r["prompt"],r["chosen"]); lp_neg=comp_logp(r["prompt"],r["rejected"])
            sp+=lp_pos; sn+=lp_neg; sm+=(lp_pos-lp_neg); nn+=1
        res={"model":a.model,"n_pairs":nn,"logp_pos":sp/max(nn,1),"logp_neg":sn/max(nn,1),"margin":sm/max(nn,1)}
        json.dump(res,open(a.out,"w"),indent=2)
        print(f"[margin] n={nn} logp_pos={res['logp_pos']:.4f} logp_neg={res['logp_neg']:.4f} margin={res['margin']:.4f}")
        return
    rows=LOADERS[a.dataset](a)
    if a.num_shards>1: rows=rows[a.shard_index::a.num_shards]  # data-parallel gen: this GPU's slice
    from vllm import LLM, SamplingParams
    llm=LLM(model=a.model,trust_remote_code=True,dtype="bfloat16",gpu_memory_utilization=float(os.environ.get("GEN_GPU_MEM","0.5")),max_model_len=int(os.environ.get("MAXLEN","2048")),tensor_parallel_size=int(os.environ.get("VLLM_TP","1")),enforce_eager=True)
    sp=SamplingParams(n=a.k,temperature=(0.0 if a.mode=="eval" and a.k==1 else a.temperature),top_p=0.95,max_tokens=int(os.environ.get("MAXTOK","640")),seed=a.seed,logprobs=(1 if a.hard_neg else None))
    prompts=[format_prompt(r,a.model) for r in rows]
    outs=llm.generate(prompts,sp)
    if a.mode=="eval":
        save_samples=os.environ.get("SAVE_SAMPLES")=="1"  # also store per-sample answer+correctness (self-consistency / best-of-n analysis)
        per=[]; cov=0
        for r,o in zip(rows,outs):
            g=r["gold_answer"]; oks=[1 if verify(x.text,g) else 0 for x in o.outputs]
            c=sum(oks); kk=len(o.outputs); cov+=int(c>0)
            rec={"c":c,"k":kk}
            if save_samples:
                anss=[]
                for x in o.outputs:
                    ans=extract_boxed_answer(x.text)
                    if ans is None or str(ans).strip()=="": ans=extract_numeric_answer(x.text)
                    anss.append(("" if ans is None else str(ans).strip()))
                rec["ans"]=anss; rec["ok"]=oks; rec["gold"]=str(g)
            per.append(rec)
        n=len(rows); p1=sum(p["c"]/p["k"] for p in per)/n
        json.dump({"model":a.model,"n":n,"k":a.k,"pass1":p1,"coverage_passk":cov/n,"per_problem":per},open(a.out,"w"),indent=2)
        print(f"[math_eval] n={n} k={a.k} pass1={p1:.4f} cov={cov/n:.4f}")
        return
    rng=random.Random(a.seed); npairs=0; nbank=0
    with open(a.out,"w") as f:
        for r,o in zip(rows,outs):
            P=format_prompt(r,a.model); texts=[x.text for x in o.outputs]
            corr=[t for t in texts if verify(t,r["gold_answer"])]; inc=[t for t in texts if not verify(t,r["gold_answer"])]
            if a.mode=="bank":
                for t in corr[:a.max_pairs_per]: f.write(json.dumps({"prompt":P,"completion":t})+"\n"); nbank+=1
            else:  # pairs
                if a.shuffle:
                    if len(texts)>=2:
                        for _ in range(min(a.max_pairs_per,len(texts)//2)):
                            x,y=rng.sample(texts,2); f.write(json.dumps({"prompt":P,"chosen":x,"rejected":y})+"\n"); npairs+=1
                elif corr and inc:
                    if a.hard_neg:
                        # hardest negatives = highest-logprob wrong answers (steal the most single-attempt mass);
                        # best positives = highest-logprob correct. Deterministic top-k. Guard None logprob.
                        score={x.text: ((x.cumulative_logprob or 0.0)/max(len(x.token_ids),1)) for x in o.outputs}
                        ci=sorted(corr,key=lambda t:score.get(t,-1e9),reverse=True)
                        ii=sorted(inc, key=lambda t:score.get(t,-1e9),reverse=True)
                        for j in range(min(a.max_pairs_per,len(ci),len(ii))):
                            f.write(json.dumps({"prompt":P,"chosen":ci[j%len(ci)],"rejected":ii[j]})+"\n"); npairs+=1
                    else:
                        for _ in range(min(a.max_pairs_per,len(corr),len(inc))):
                            f.write(json.dumps({"prompt":P,"chosen":rng.choice(corr),"rejected":rng.choice(inc)})+"\n"); npairs+=1
    print(f"[math_{a.mode}] {'bank='+str(nbank) if a.mode=='bank' else 'pairs='+str(npairs)} -> {a.out}")
if __name__=="__main__": main()
