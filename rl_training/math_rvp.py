"""Math-domain (GSM8K) RVP: bank/pairs/eval via answer-match verify. Mirrors comp RVP for a 2nd dataset."""
import argparse, json, os, sys, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.dataset import (load_gsm8k, load_math500, load_math_full, load_competition_math,
    load_deepmath, load_aime_all, load_amc, load_omni_math, load_olympiad_bench,
    format_prompt, extract_numeric_answer, extract_boxed_answer, answers_match)
def verify(text, gold):
    # try boxed-latex extraction first (MATH/Olympiad), then numeric (GSM8K/AIME/AMC)
    for extract in (extract_boxed_answer, extract_numeric_answer):
        try:
            if answers_match(extract(text), gold): return True
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
    ap.add_argument("--mode",required=True,choices=["bank","pairs","eval"])
    ap.add_argument("--model",required=True); ap.add_argument("--split",default="train")
    ap.add_argument("--dataset",default="gsm8k",choices=list(LOADERS.keys()))
    ap.add_argument("--n",type=int,default=500); ap.add_argument("--k",type=int,default=8)
    ap.add_argument("--temperature",type=float,default=0.8); ap.add_argument("--max-pairs-per",type=int,default=2)
    ap.add_argument("--shuffle",action="store_true"); ap.add_argument("--out",required=True); ap.add_argument("--seed",type=int,default=1)
    a=ap.parse_args()
    rows=LOADERS[a.dataset](a)
    from vllm import LLM, SamplingParams
    llm=LLM(model=a.model,trust_remote_code=True,dtype="bfloat16",gpu_memory_utilization=float(os.environ.get("GEN_GPU_MEM","0.5")),max_model_len=int(os.environ.get("MAXLEN","2048")),tensor_parallel_size=int(os.environ.get("VLLM_TP","1")),enforce_eager=True)
    sp=SamplingParams(n=a.k,temperature=(0.0 if a.mode=="eval" and a.k==1 else a.temperature),top_p=0.95,max_tokens=int(os.environ.get("MAXTOK","640")),seed=a.seed)
    prompts=[format_prompt(r,a.model) for r in rows]
    outs=llm.generate(prompts,sp)
    if a.mode=="eval":
        per=[]; cov=0
        for r,o in zip(rows,outs):
            c=sum(1 for x in o.outputs if verify(x.text,r["gold_answer"])); kk=len(o.outputs); cov+=int(c>0)
            per.append({"c":c,"k":kk})
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
                    for _ in range(min(a.max_pairs_per,len(corr),len(inc))):
                        f.write(json.dumps({"prompt":P,"chosen":rng.choice(corr),"rejected":rng.choice(inc)})+"\n"); npairs+=1
    print(f"[math_{a.mode}] {'bank='+str(nbank) if a.mode=='bank' else 'pairs='+str(npairs)} -> {a.out}")
if __name__=="__main__": main()
