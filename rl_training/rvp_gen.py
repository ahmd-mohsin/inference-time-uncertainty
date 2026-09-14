"""RVP pair generator: from a checkpoint, sample K completions per prompt, verify each, emit verified preference pairs
(prompt, chosen=verified-correct, rejected=verified-incorrect) for prompts the model COVERS-but-is-unreliable-on.
--shuffle: label control (chosen/rejected assigned ignoring verification)."""
import argparse, json, os, sys, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl_training.comp_tasks import verify_solution, _rand_records
def _ti(r):
    ti=r.get("test_inputs")
    if ti: return ti
    rng=random.Random(abs(hash(r["prompt"]))%10**7); return [_rand_records(rng,rng.randint(6,10)) for _ in range(6)]
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",required=True); ap.add_argument("--pool",required=True)
    ap.add_argument("--n",type=int,default=400); ap.add_argument("--k",type=int,default=12)
    ap.add_argument("--temperature",type=float,default=1.0); ap.add_argument("--max-pairs-per",type=int,default=2)
    ap.add_argument("--shuffle",action="store_true"); ap.add_argument("--out",required=True)
    a=ap.parse_args()
    rows=[json.loads(l) for l in open(a.pool) if l.strip()][:a.n]
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(a.model,trust_remote_code=True)
    llm=LLM(model=a.model,trust_remote_code=True,dtype="bfloat16",gpu_memory_utilization=float(os.environ.get("GEN_GPU_MEM","0.5")),max_model_len=2048,enforce_eager=True)
    sp=SamplingParams(n=a.k,temperature=a.temperature,top_p=0.95,max_tokens=640)
    def chat(p):
        try: return tok.apply_chat_template([{"role":"user","content":p}],tokenize=False,add_generation_prompt=True)
        except Exception: return p+"\n"
    outs=llm.generate([chat(r["prompt"]) for r in rows],sp)
    rng=random.Random(0); pairs=0
    with open(a.out,"w") as f:
        for r,o in zip(rows,outs):
            task={"prog":r["prog"],"test_inputs":_ti(r)}
            texts=[c.text for c in o.outputs]
            fmt=lambda t: t if "```" in t else "```python\n"+t+"\n```"
            corr=[t for t in texts if verify_solution(t,task)]; inc=[t for t in texts if not verify_solution(t,task)]
            if a.shuffle:
                if len(texts)>=2:
                    for _ in range(min(a.max_pairs_per,len(texts)//2)):
                        x,y=rng.sample(texts,2); f.write(json.dumps({"prompt":r["prompt"],"chosen":fmt(x),"rejected":fmt(y)})+"\n"); pairs+=1
            else:
                if corr and inc:  # covers-but-unreliable prompt
                    for _ in range(min(a.max_pairs_per,len(corr),len(inc))):
                        f.write(json.dumps({"prompt":r["prompt"],"chosen":fmt(rng.choice(corr)),"rejected":fmt(rng.choice(inc))})+"\n"); pairs+=1
    print(f"[rvp_gen] {'SHUFFLE ' if a.shuffle else ''}pairs={pairs} from {len(rows)} prompts -> {a.out}")
if __name__=="__main__": main()
