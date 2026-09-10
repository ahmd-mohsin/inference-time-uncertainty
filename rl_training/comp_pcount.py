# Measure per-prompt success probability p at a policy (k samples/prompt), for the advantage-density theorem:
# GRPO zero-advantage ("dead") group fraction (binary reward, group size G) = E_prompt[p^G + (1-p)^G].
# Emits the p-distribution + predicted dead-fraction for given G. Usage: python -m rl_training.comp_pcount --model M --pool P --k 16 --G 8
import argparse,json,os,sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl_training.comp_tasks import verify_solution
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--model",required=True); ap.add_argument("--pool",required=True)
    ap.add_argument("--n",type=int,default=200); ap.add_argument("--k",type=int,default=16); ap.add_argument("--G",type=int,default=8)
    ap.add_argument("--tag",default="pcount"); a=ap.parse_args()
    rows=[json.loads(l) for l in open(a.pool) if l.strip()][:a.n]
    from vllm import LLM,SamplingParams; from transformers import AutoTokenizer
    if os.path.isdir(a.model) and os.path.exists(a.model+"/adapter_config.json") and not os.path.exists(a.model+"/config.json"):
        from rl_training.model_utils import merge_adapter_if_needed; a.model=merge_adapter_if_needed(a.model)
    tok=AutoTokenizer.from_pretrained(a.model,trust_remote_code=True)
    llm=LLM(model=a.model,trust_remote_code=True,dtype="bfloat16",gpu_memory_utilization=float(os.environ.get("EVAL_GPU_MEM","0.5")),max_model_len=2048,enforce_eager=True)
    sp=SamplingParams(n=a.k,temperature=1.0,top_p=0.95,max_tokens=640)
    def chat(p):
        try: return tok.apply_chat_template([{"role":"user","content":p}],tokenize=False,add_generation_prompt=True)
        except: return p+"\n"
    outs=llm.generate([chat(r["prompt"]) for r in rows],sp)
    ps=[]
    for r,o in zip(rows,outs):
        t={"prog":r["prog"],"test_inputs":r["test_inputs"]}; s=sum(verify_solution(c.text,t) for c in o.outputs); ps.append(s/a.k)
    import statistics as st
    from math import comb
    G=a.G; K=a.k
    counts=[round(p*K) for p in ps]
    def unbiasedD(c): return (comb(c,G)+comb(K-c,G))/comb(K,G) if K>=G else float('nan')
    dead=st.mean([unbiasedD(c) for c in counts])           # UNBIASED dead-group fraction
    d_fail=st.mean([ (comb(K-c,G)/comb(K,G)) for c in counts])
    d_succ=st.mean([ (comb(c,G)/comb(K,G)) for c in counts])
    succ_mass=st.mean([p*G for p in ps])
    hist={f"{lo/10:.1f}-{(lo+1)/10:.1f}":sum(1 for p in ps if lo/10<=p<(lo+1)/10 or (lo==9 and p==1.0)) for lo in range(10)}
    res={"tag":a.tag,"n":len(ps),"k":a.k,"G":G,"mean_p":st.mean(ps),"dead_frac_unbiased":dead,"d_fail":d_fail,"d_success":d_succ,"success_mass_per_group":succ_mass,"counts":counts,"p_hist":hist}
    os.makedirs("/tmp/instance_storage/gu/eval_out",exist_ok=True)
    json.dump(res,open(f"/tmp/instance_storage/gu/eval_out/pcount_{a.tag}.json","w"),indent=2)
    print(f"[pcount {a.tag}] mean_p={st.mean(ps):.3f} dead_unbiased={dead:.3f} d_fail={d_fail:.3f} d_succ={d_succ:.3f} p_hist={hist}")
if __name__=="__main__": main()
