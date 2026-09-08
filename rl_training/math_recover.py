# Base-vs-GRPO failure-conditioned switching (mechanism link for reasoning-monoculture recovery).
# Tests: does RL routing compression make post-failure strategy-switching MORE valuable?
#   G_switch = switch_recovery - iid_retry_recovery (on default-failed problems, matched budget)
#   prediction: G_switch(grpo) > G_switch(base)  [monoculture => iid retry repeats the same failed route]
# Math analog of code_recover: default (n_def iid) + iid-retry (n_rec) + strategy-SWITCH (prefix-forced,
# n_rec spread over 14 strategies). safe_is_correct verifier. 8-GPU DP + merge.
import argparse, json, os, sys, statistics
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.dataset import get_inference_dataset, format_prompt
from rl_training.safe_match import safe_is_correct
from rl_training.strategy_probe import STRAT_NAMES, STRATEGY_PREFIX


def run(a):
    from vllm import LLM, SamplingParams
    from transformers import AutoConfig
    from rl_training.model_utils import merge_adapter_if_needed
    mp = merge_adapter_if_needed(a.model_path)
    probs = get_inference_dataset({"dataset": {"name": a.dataset, "split": "test", "n_problems": -1, "seed": 42}})
    if a.max_problems > 0: probs = probs[:a.max_problems]
    if a.num_shards > 1: probs = probs[a.shard_index::a.num_shards]
    try: cap = int(getattr(AutoConfig.from_pretrained(mp, trust_remote_code=True), "max_position_embeddings", 4096))
    except Exception: cap = 4096
    mml = min(4096, cap)
    _gm = float(os.environ.get("EVAL_GPU_MEM", 0.85)); _eager = os.environ.get("EVAL_ENFORCE_EAGER","1")=="1"
    llm = LLM(model=mp, dtype="bfloat16", trust_remote_code=True, tensor_parallel_size=1, max_model_len=mml,
              gpu_memory_utilization=_gm, enable_prefix_caching=True, enforce_eager=_eager)
    def gen(prompts, n):
        sp = SamplingParams(n=n, temperature=1.0, top_p=1.0, max_tokens=mml-1024, stop=["<|im_end|>","<|eot_id|>","<|endoftext|>"])
        return llm.generate(prompts, sp)
    plain = gen([format_prompt(p, mp) for p in probs], a.n_def + a.n_rec)   # default + iid-retry
    per_strat = max(1, a.n_rec // len(STRAT_NAMES) + 1)
    sp2, sidx, sseed = [], [], []
    for pi, p in enumerate(probs):
        for m in STRAT_NAMES:
            sp2.append(format_prompt(p, mp) + STRATEGY_PREFIX[m]); sidx.append((pi, m)); sseed.append(STRATEGY_PREFIX[m])
    sout = gen(sp2, per_strat)
    per = []
    for p, o in zip(probs, plain):
        g = str(p.get("gold_answer","")); outs = list(o.outputs)
        deff = [bool(safe_is_correct(s.text, g)[0]) for s in outs[:a.n_def]]
        iidr = [bool(safe_is_correct(s.text, g)[0]) for s in outs[a.n_def:a.n_def+a.n_rec]]
        per.append({"problem_id": p["problem_id"], "default": deff, "iid_retry": iidr, "switch": []})
    for (pi, m), seed, o in zip(sidx, sseed, sout):
        g = str(probs[pi].get("gold_answer",""))
        for s in o.outputs:
            per[pi]["switch"].append({"correct": bool(safe_is_correct(seed+(s.text or ""), g)[0]), "strategy": m})
    out = {"tag": a.tag, "model": mp, "dataset": a.dataset, "n_problems": len(probs), "n_def": a.n_def,
           "n_rec": a.n_rec, "strategies": STRAT_NAMES, "per_problem": per}
    Path(a.output_dir).mkdir(parents=True, exist_ok=True)
    fp = (Path(a.output_dir)/(f"mrec_{a.tag}.shard{a.shard_index}-of-{a.num_shards}.json" if a.num_shards>1 else f"mrec_{a.tag}.json"))
    json.dump(out, open(fp,"w")); print(f"[{a.tag} shard {a.shard_index}/{a.num_shards}] {len(probs)} -> {fp}")


def merge(a):
    import random
    parts=[]
    for s in range(a.num_shards):
        fp=Path(a.output_dir)/f"mrec_{a.tag}.shard{s}-of-{a.num_shards}.json"
        if not fp.exists(): raise FileNotFoundError(f"missing shard {s}")
        parts.append(json.load(open(fp)))
    per=[pp for part in parts for pp in part["per_problem"]]; nrec=parts[0]["n_rec"]; strat=parts[0]["strategies"]
    fails=[p for p in per if not any(p["default"])]
    nf=len(fails); iid=sw=0; oracle=[]; bestfix={m:[] for m in strat}
    for p in fails:
        if any(p["iid_retry"]): iid+=1
        pool=[x["correct"] for x in p["switch"]]
        idx=random.sample(range(len(pool)), min(nrec,len(pool))) if pool else []
        if any(pool[i] for i in idx): sw+=1
        by={m:[] for m in strat}
        for x in p["switch"]: by[x["strategy"]].append(x["correct"])
        Rm={m:(1 if any(by[m]) else 0) for m in strat}
        oracle.append(max(Rm.values()) if Rm else 0)
        for m in strat: bestfix[m].append(Rm[m])
    out={"tag":a.tag,"model":parts[0]["model"],"dataset":parts[0]["dataset"],"n_problems":len(per),
         "n_default_failed":nf,"iid_retry_recovery":iid/max(nf,1),"switch_recovery":sw/max(nf,1),
         "G_switch":(sw-iid)/max(nf,1),"oracle_recovery":(statistics.mean(oracle) if oracle else 0),
         "best_fixed_recovery":(max(statistics.mean(bestfix[m]) for m in strat) if nf else 0),"per_problem":per}
    fp=Path(a.output_dir)/f"mrec_{a.tag}.json"; json.dump(out,open(fp,"w"))
    print(f"[{a.tag}] nfail={nf} iid={out['iid_retry_recovery']:.3f} switch={out['switch_recovery']:.3f} "
          f"G_switch={out['G_switch']:+.3f} oracle={out['oracle_recovery']:.3f} bestfix={out['best_fixed_recovery']:.3f}")
    print(f"saved -> {fp}")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model-path"); ap.add_argument("--dataset",default="math500")
    ap.add_argument("--max-problems",type=int,default=200); ap.add_argument("--n-def",type=int,default=8); ap.add_argument("--n-rec",type=int,default=12)
    ap.add_argument("--output-dir",default="/tmp/instance_storage/gu/eval_out"); ap.add_argument("--tag",default="mrec")
    ap.add_argument("--shard-index",type=int,default=0); ap.add_argument("--num-shards",type=int,default=1); ap.add_argument("--merge",action="store_true")
    a=ap.parse_args(); merge(a) if a.merge else run(a)


if __name__=="__main__":
    main()
