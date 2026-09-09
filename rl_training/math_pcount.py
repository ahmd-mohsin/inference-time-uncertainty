# MATH-domain advantage-density check: per-prompt success prob p on a math panel (k samples), predicted GRPO
# dead-group fraction D(G)=E[p^G+(1-p)^G]. Tests Theorem-1 generalization beyond the compositional domain.
import argparse,os,sys,json,statistics as st
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--model",required=True); ap.add_argument("--panel",default="test")
    ap.add_argument("--n",type=int,default=150); ap.add_argument("--k",type=int,default=16); ap.add_argument("--G",type=int,default=8); ap.add_argument("--tag",default="math")
    a=ap.parse_args()
    from rl_training.panel_eval import load_panel, extract, match, PROMPT
    from vllm import LLM,SamplingParams; from transformers import AutoTokenizer
    items=load_panel(a.panel,a.n)
    tok=AutoTokenizer.from_pretrained(a.model,trust_remote_code=True)
    llm=LLM(model=a.model,trust_remote_code=True,dtype="bfloat16",gpu_memory_utilization=float(os.environ.get("EVAL_GPU_MEM","0.5")),max_model_len=2048,enforce_eager=True)
    sp=SamplingParams(n=a.k,temperature=1.0,top_p=0.95,max_tokens=1024)
    def chat(q):
        m=[{"role":"user","content":PROMPT.replace("{q}",q)}]
        try: return tok.apply_chat_template(m,tokenize=False,add_generation_prompt=True)
        except: return PROMPT.replace("{q}",q)+"\n"
    outs=llm.generate([chat(it["q"]) for it in items],sp)
    ps=[]
    for it,o in zip(items,outs):
        s=sum(1 for c in o.outputs if match(extract(c.text),it["gold"])); ps.append(s/a.k)
    G=a.G; D=st.mean([p**G+(1-p)**G for p in ps])
    h={f"{lo/10:.1f}-{(lo+1)/10:.1f}":sum(1 for p in ps if lo/10<=p<(lo+1)/10 or (lo==9 and p==1.0)) for lo in range(10)}
    res={"tag":a.tag,"panel":a.panel,"n":len(ps),"k":a.k,"G":G,"mean_p":st.mean(ps),"predicted_dead_frac":D,"p_hist":h,"extreme_mass":(h.get("0.0-0.1",0)+h.get("0.9-1.0",0))/len(ps)}
    os.makedirs("/tmp/instance_storage/gu/eval_out",exist_ok=True)
    json.dump(res,open(f"/tmp/instance_storage/gu/eval_out/mathpcount_{a.tag}.json","w"),indent=2)
    print(f"[mathpcount {a.tag}] panel={a.panel} mean_p={st.mean(ps):.3f} D(8)={D:.3f} extreme_mass={res['extreme_mass']:.2f} hist={h}")
if __name__=="__main__": main()
