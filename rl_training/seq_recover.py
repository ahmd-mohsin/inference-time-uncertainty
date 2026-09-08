# Sequential ACTIVE-DIAGNOSIS recovery (award-thesis R2/R3). On default-FAILED code problems, compare THREE
# arms at matched budget T (T recovery attempts each):
#   IID       : T independent retries from the ORIGINAL prompt (no feedback).
#   STATIC    : T independent attempts, each conditioned ONLY on the original failure (code+error),
#               spread across the 8 recovery strategies (portfolio, no inter-attempt feedback).
#   SEQUENTIAL: T attempts where attempt t sees the FULL history of prior attempts' code + execution
#               errors (true diagnosis — the model accumulates evidence about the latent failure).
# Round-batched so vLLM stays efficient: at each round we build one prompt per still-unsolved problem and
# generate 1 sample each. Records recovery@t per arm -> tests whether (a) SEQUENTIAL > STATIC > IID, and
# (b) recovery rate keeps rising across rounds (predictability-after-observation proxy).
#
# Usage: python -m rl_training.seq_recover --model-path <dir> --bench mbpp --tag seq_qi_mbpp \
#   --shard-index S --num-shards 8 --T 6
import argparse, json, os, sys, random
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl_training.code_passk import load_bench, extract_code, run_tests, build_prompt, STRATEGIES, STRAT_NAMES

SYS = "You are an expert Python programmer."

def chat(model_path, user, history=None):
    """history = list of (assistant_code, user_feedback). Build a multi-turn prompt in the model's format."""
    ml = model_path.lower(); turns = []
    if any(k in ml for k in ["qwen","deepseek"]):
        s=f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n"
        for code,fb in (history or []):
            s+=f"<|im_start|>assistant\n```python\n{code}\n```<|im_end|>\n<|im_start|>user\n{fb}<|im_end|>\n"
        return s+"<|im_start|>assistant\n"
    if "llama" in ml:
        s=(f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{SYS}<|eot_id|>"
           f"<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>")
        for code,fb in (history or []):
            s+=(f"<|start_header_id|>assistant<|end_header_id|>\n\n```python\n{code}\n```<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n{fb}<|eot_id|>")
        return s+"<|start_header_id|>assistant<|end_header_id|>\n\n"
    return chat("qwen", user, history)

def base_task(it):
    if it.get("mbpp"):
        return (f"Write a Python function for the following task. Return ONLY a ```python code block.\n\n"
                f"Task: {it['prompt']}\nMust satisfy tests like:\n{it['test'].splitlines()[0]}")
    return f"Complete the following Python function. Return the FULL function in a ```python code block.\n\n{it['prompt']}"

def classify_err(err):
    e=(err or "").lower()
    if "timeout" in e: return "TIME-LIMIT / infinite-loop or too-slow algorithm"
    if "syntaxerror" in e or "indentation" in e: return "SYNTAX error (malformed code)"
    if "assertion" in e or "wrong-output" in e: return "WRONG-OUTPUT (logic/spec bug, code runs but answer is wrong)"
    if "indexerror" in e or "keyerror" in e: return "INDEXING / boundary error (off-by-one, empty/edge input)"
    if "typeerror" in e or "valueerror" in e or "attributeerror" in e: return "TYPE/VALUE error (wrong types or unhandled case)"
    if "recursion" in e: return "RECURSION-depth error"
    return "runtime error"

def fb_msg(err, diagnostic, label=False):
    base=f"That attempt failed. Error:\n{err}\n"
    if label:
        base+=f"Failure class: {classify_err(err)}.\n"
    if diagnostic:
        return base+("Diagnose WHY it failed (what class of bug: wrong algorithm, edge case, wrong "
                     "invariant, complexity, spec misread?), then give a corrected full solution in a "
                     "```python block.")
    return base+"Provide a corrected full solution in a ```python block."

def run(a):
    from vllm import LLM, SamplingParams
    from rl_training.model_utils import merge_adapter_if_needed
    mp = merge_adapter_if_needed(a.model_path)
    items = load_bench(a.bench)
    if a.max_problems>0: items=items[:a.max_problems]
    if a.num_shards>1: items=items[a.shard_index::a.num_shards]
    _gm=float(os.environ.get("EVAL_GPU_MEM",0.85)); _eager=os.environ.get("EVAL_ENFORCE_EAGER","1")=="1"
    llm=LLM(model=mp,dtype="bfloat16",trust_remote_code=True,tensor_parallel_size=1,max_model_len=6144,
            gpu_memory_utilization=_gm,enable_prefix_caching=True,enforce_eager=_eager)
    def gen(prompts,n=1,temp=1.0):
        sp=SamplingParams(n=n,temperature=temp,top_p=0.95,max_tokens=1024,stop=["<|im_end|>","<|eot_id|>","<|endoftext|>"])
        return llm.generate(prompts,sp)
    # ---- round 0: default attempt (1 sample); keep only FAILED problems ----
    d0=gen([chat(mp, base_task(it)) for it in items],1)
    failed=[]
    for it,o in zip(items,d0):
        code=extract_code(o.outputs[0].text); ok,err=run_tests(code,it,return_err=True)
        if not ok: failed.append({"it":it,"code0":code,"err0":err})
    per=[]
    for f in failed:
        it=f["it"]
        rec={"id":it["id"],"code0":f["code0"][:1500],"err0":f["err0"],
             "iid":[],"static":[],"seq":[],"seq_hist":[]}
        per.append(rec)
    # ---- IID arm: T independent retries from ORIGINAL prompt ----
    if per:
        iid_out=gen([chat(mp, base_task(f["it"])) for f in failed], a.T, temp=1.0)
        for rec,f,o in zip(per,failed,iid_out):
            for s in o.outputs:
                rec["iid"].append(bool(run_tests(extract_code(s.text), f["it"])))
    # ---- STATIC portfolio arm: T attempts conditioned only on ORIGINAL failure, across strategies ----
    if per:
        sp_prompts=[]; sp_map=[]
        for pi,f in enumerate(failed):
            for t in range(a.T):
                strat=STRATEGIES[STRAT_NAMES[t%len(STRAT_NAMES)]]
                user=base_task(f["it"])+f"\n\nA previous attempt failed with error:\n{f['err0']}\n{strat} Provide a corrected full solution in a ```python block."
                sp_prompts.append(chat(mp,user)); sp_map.append(pi)
        sp_out=gen(sp_prompts,1)
        for (pi),o in zip(sp_map,sp_out):
            per[pi]["static"].append(bool(run_tests(extract_code(o.outputs[0].text), failed[pi]["it"])))
    # ---- SEQUENTIAL arm: round-batched, attempt t sees history of prior attempts.
    #   diag-mode 'full'       : show prior CODE + errors (may ANCHOR on buggy code).
    #   diag-mode 'error_only' : HIDE the buggy code, show only the error signatures (anchoring antidote).
    # diag-modes: full | error_only | error_label | certificate | cert_memory
    #   certificate  = FORGET-TO-REPAIR (E_rich, ∅): feedback = concrete counterexample, code HIDDEN,
    #                  only the LATEST certificate kept (no accumulation of proposals).
    #   cert_memory  = accumulate ALL failure certificates (code hidden) — cumulative learning w/o anchoring.
    from rl_training.certify import make_certificate
    CERT = a.diag_mode in ("certificate","cert_memory")
    ERR_ONLY = a.diag_mode in ("error_only","error_label") or CERT
    LABEL = (a.diag_mode=="error_label")
    ACCUM = (a.diag_mode != "certificate")   # 'certificate' keeps only the latest cert
    def hide(code): return "# (previous attempt hidden)" if ERR_ONLY else code
    def fb_for(code, item, err):
        if CERT:
            c = make_certificate(code, item)
            return c if c else fb_msg(err, True)   # fallback if no counterexample isolable
        return fb_msg(err, True, LABEL)
    if per:
        hist=[[(hide(f["code0"]), fb_for(f["code0"], f["it"], f["err0"]))] for f in failed]
        solved=[False]*len(failed); solved_at=[None]*len(failed)
        for t in range(a.T):
            idx=[i for i in range(len(failed)) if not solved[i]]
            if not idx: break
            prompts=[chat(mp, base_task(failed[i]["it"]), hist[i]) for i in idx]
            outs=gen(prompts,1)
            for i,o in zip(idx,outs):
                code=extract_code(o.outputs[0].text); ok,err=run_tests(code,failed[i]["it"],return_err=True)
                per[i]["seq"].append(bool(ok))
                if ok: solved[i]=True; solved_at[i]=t
                else:
                    entry=(hide(code), fb_for(code, failed[i]["it"], err))
                    hist[i]=(hist[i]+[entry]) if ACCUM else [entry]
        for i in range(len(failed)): per[i]["seq_solved_at"]=solved_at[i]
    out={"tag":a.tag,"model":mp,"bench":a.bench,"T":a.T,"n_failed":len(per),"per_problem":per}
    Path(a.output_dir).mkdir(parents=True,exist_ok=True)
    fp=Path(a.output_dir)/(f"seq_{a.tag}.shard{a.shard_index}-of-{a.num_shards}.json" if a.num_shards>1 else f"seq_{a.tag}.json")
    json.dump(out,open(fp,"w")); print(f"[{a.tag} shard {a.shard_index}/{a.num_shards}] failed={len(per)} -> {fp}")

def merge(a):
    parts=[]
    for s in range(a.num_shards):
        fp=Path(a.output_dir)/f"seq_{a.tag}.shard{s}-of-{a.num_shards}.json"
        if not fp.exists():
            print(f"[warn] missing shard {s} — skipping"); continue
        parts.append(json.load(open(fp)))
    per=[pp for part in parts for pp in part["per_problem"]]; T=parts[0]["T"]; n=len(per)
    def anyk(key,k): return sum(1 for p in per if any(p.get(key,[])[:k]))/max(n,1)
    # recovery@t curves (cumulative any-solved by round t) for each arm
    iid=[anyk("iid",k) for k in range(1,T+1)]
    static=[anyk("static",k) for k in range(1,T+1)]
    seq=[anyk("seq",k) for k in range(1,T+1)]
    out={"tag":a.tag,"model":parts[0]["model"],"bench":parts[0]["bench"],"T":T,"n_failed":n,
         "recovery_at_t":{"iid":iid,"static":static,"seq":seq},
         "final":{"iid":iid[-1],"static":static[-1],"seq":seq[-1],
                  "seq_minus_static":seq[-1]-static[-1],"seq_minus_iid":seq[-1]-iid[-1]},
         "per_problem":per}
    fp=Path(a.output_dir)/f"seq_{a.tag}.json"; json.dump(out,open(fp,"w"))
    print(f"[{a.tag}] failed={n} T={T}")
    print(f"  recovery@T  iid={iid[-1]:.3f}  static={static[-1]:.3f}  SEQ={seq[-1]:.3f}  "
          f"seq-static={seq[-1]-static[-1]:+.3f}  seq-iid={seq[-1]-iid[-1]:+.3f}")
    print(f"  seq curve: {[f'{x:.3f}' for x in seq]}")
    print(f"saved -> {fp}")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model-path"); ap.add_argument("--bench",default="mbpp")
    ap.add_argument("--max-problems",type=int,default=-1); ap.add_argument("--T",type=int,default=6)
    ap.add_argument("--output-dir",default="/tmp/instance_storage/gu/eval_out"); ap.add_argument("--tag",default="seq")
    ap.add_argument("--diag-mode",default="full",choices=["full","error_only","error_label","certificate","cert_memory"])
    ap.add_argument("--shard-index",type=int,default=0); ap.add_argument("--num-shards",type=int,default=1); ap.add_argument("--merge",action="store_true")
    a=ap.parse_args(); merge(a) if a.merge else run(a)

if __name__=="__main__":
    main()
