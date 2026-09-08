# Diagnostic-diversity experiment: Counterfactual Strategy Disagreement (D_CF) as an ERROR/uncertainty
# signal. Tests "diversity for falsification, not coverage": strategies may fail DIFFERENTLY (disagree
# on the answer) even when they don't SOLVE different problems (v_coverage≈0). Prediction: D_CF predicts
# errors better than iid self-consistency, and the gap GROWS after GRPO (routing collapse makes iid
# overconfident). Saves EXTRACTED ANSWERS (not just correctness) so disagreement is measurable.
#
# Per problem: iid pool (N_iid free) + forced pool (N_forced per strategy, prefix-seeded). For each
# sample store (answer_string, correct). Merge computes: default A0 = iid majority answer; error label
# E=1[A0 != gold]; iid uncertainty (1-agreement, entropy); D_CF (strategy-level answer disagreement);
# and error-AUROC + risk-coverage for each signal. 8-GPU DP + merge.
#
# Usage: python -m rl_training.cf_disagree --model-path <dir> --dataset math500 --tag cf_qm_grpo \
#   --shard-index S --num-shards 8 --n-iid 16 --n-forced 3

import argparse, json, os, sys, math
from collections import Counter
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.dataset import get_inference_dataset, format_prompt, extract_numeric_answer, answers_match
from rl_training.safe_match import safe_is_correct
from rl_training.strategy_probe import STRAT_NAMES, STRATEGY_PREFIX


def _ans(text):
    a = extract_numeric_answer(text or "")
    return None if a is None else str(a).strip()


def run(a):
    from vllm import LLM, SamplingParams
    from transformers import AutoConfig
    from rl_training.model_utils import merge_adapter_if_needed
    mp = merge_adapter_if_needed(a.model_path)
    problems = get_inference_dataset({"dataset": {"name": a.dataset, "split": "test", "n_problems": -1, "seed": 42}})
    if a.max_problems > 0: problems = problems[:a.max_problems]
    if a.num_shards > 1: problems = problems[a.shard_index::a.num_shards]
    try: cap = int(getattr(AutoConfig.from_pretrained(mp, trust_remote_code=True), "max_position_embeddings", 4096))
    except Exception: cap = 4096
    mml = min(4096, cap)
    _gm = float(os.environ.get("EVAL_GPU_MEM", 0.85)); _eager = os.environ.get("EVAL_ENFORCE_EAGER","1")=="1"
    llm = LLM(model=mp, dtype="bfloat16", trust_remote_code=True, tensor_parallel_size=1, max_model_len=mml,
              gpu_memory_utilization=_gm, enable_prefix_caching=True, enforce_eager=_eager)
    def gen(prompts, n):
        sp = SamplingParams(n=n, temperature=1.0, top_p=1.0, max_tokens=mml-1024, stop=["<|im_end|>","<|eot_id|>","<|endoftext|>"])
        return llm.generate(prompts, sp)
    iid_out = gen([format_prompt(p, mp) for p in problems], a.n_iid)
    fpr, fidx, fseed = [], [], []
    for pi, p in enumerate(problems):
        for m in STRAT_NAMES:
            fpr.append(format_prompt(p, mp) + STRATEGY_PREFIX[m]); fidx.append((pi, m)); fseed.append(STRATEGY_PREFIX[m])
    f_out = gen(fpr, a.n_forced) if fpr else []
    per = [{"problem_id": p["problem_id"], "gold": str(p.get("gold_answer","")),
            "iid": [], "forced": {m: [] for m in STRAT_NAMES}} for p in problems]
    for p, o in zip(per, iid_out):
        for s in o.outputs:
            per_ok = bool(safe_is_correct(s.text, p["gold"])[0])
            p["iid"].append({"ans": _ans(s.text), "correct": per_ok})
    for (pi, m), seed, o in zip(fidx, fseed, f_out):
        g = per[pi]["gold"]
        for s in o.outputs:
            full = seed + (s.text or "")
            per[pi]["forced"][m].append({"ans": _ans(full), "correct": bool(safe_is_correct(full, g)[0])})
    out = {"tag": a.tag, "model": mp, "dataset": a.dataset, "n_problems": len(problems),
           "n_iid": a.n_iid, "n_forced": a.n_forced, "strategies": STRAT_NAMES, "per_problem": per}
    Path(a.output_dir).mkdir(parents=True, exist_ok=True)
    fp = (Path(a.output_dir)/(f"cf_{a.tag}.shard{a.shard_index}-of-{a.num_shards}.json" if a.num_shards>1 else f"cf_{a.tag}.json"))
    json.dump(out, open(fp,"w"))
    print(f"[{a.tag} shard {a.shard_index}/{a.num_shards}] {len(problems)} problems -> {fp}")


def _auroc(scores, labels):
    pos=sum(labels); neg=len(labels)-pos
    if pos==0 or neg==0: return float('nan')
    order=sorted(range(len(scores)), key=lambda i: scores[i]); ranks=[0.0]*len(scores); k=0; r=1
    while k<len(order):
        j=k
        while j+1<len(order) and scores[order[j+1]]==scores[order[k]]: j+=1
        avg=(r+(r+(j-k)))/2
        for t in range(k,j+1): ranks[order[t]]=avg
        r+=(j-k+1); k=j+1
    sp=sum(ranks[i] for i in range(len(labels)) if labels[i]==1)
    return (sp-pos*(pos+1)/2)/(pos*neg)


def _entropy(counter, total):
    return -sum((c/total)*math.log(c/total) for c in counter.values() if c>0) if total else 0.0


def merge(a):
    parts=[]
    for s in range(a.num_shards):
        fp=Path(a.output_dir)/f"cf_{a.tag}.shard{s}-of-{a.num_shards}.json"
        if not fp.exists(): raise FileNotFoundError(f"missing shard {s}: {fp}")
        parts.append(json.load(open(fp)))
    per=[pp for part in parts for pp in part["per_problem"]]
    S=parts[0]["strategies"]
    rows=[]  # (iid_unc, iid_ent, D_cf, D_cf_ent, E, A0_correct)
    for p in per:
        ans=[x["ans"] for x in p["iid"] if x["ans"] is not None]
        if len(ans)<2: continue
        cnt=Counter(ans); A0,_=cnt.most_common(1)[0]; tot=len(ans)
        iid_agree=cnt[A0]/tot; iid_unc=1-iid_agree; iid_ent=_entropy(cnt,tot)
        # per-strategy majority answer
        strat_ans=[]
        for m in S:
            fa=[x["ans"] for x in p["forced"].get(m,[]) if x["ans"] is not None]
            if fa: strat_ans.append(Counter(fa).most_common(1)[0][0])
        if len(strat_ans)<2: continue
        scnt=Counter(strat_ans); k=len(strat_ans)
        D_cf=sum(1 for x in strat_ans if x!=A0)/k          # fraction of strategies disagreeing with default
        D_cf_ent=_entropy(scnt,k)                          # entropy of strategy-answer distribution
        # error label: default A0 wrong vs gold
        E=0 if answers_match(A0, p["gold"]) else 1
        rows.append((iid_unc, iid_ent, D_cf, D_cf_ent, E))
    if len(rows)<20: print(f"[{a.tag}] too few rows ({len(rows)})"); return
    lab=[r[4] for r in rows]
    sig={"iid_1-agree":[r[0] for r in rows], "iid_entropy":[r[1] for r in rows],
         "D_CF(disagree)":[r[2] for r in rows], "D_CF_entropy":[r[3] for r in rows],
         "iid+D_CF":[r[0]+r[2] for r in rows]}
    aur={k:_auroc(v,lab) for k,v in sig.items()}
    out={"tag":a.tag,"model":parts[0]["model"],"dataset":parts[0]["dataset"],"n":len(rows),
         "err_rate":sum(lab)/len(lab),"auroc":aur,"per_problem":per}
    fp=Path(a.output_dir)/f"cf_{a.tag}.json"; json.dump(out,open(fp,"w"))
    print(f"[{a.tag}] n={len(rows)} err_rate={sum(lab)/len(lab):.3f}  error-AUROC:")
    for k in ["iid_1-agree","iid_entropy","D_CF(disagree)","D_CF_entropy","iid+D_CF"]:
        print(f"    {k:16} {aur[k]:.3f}")
    print(f"saved -> {fp}")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model-path"); ap.add_argument("--dataset",default="math500")
    ap.add_argument("--max-problems",type=int,default=200); ap.add_argument("--n-iid",type=int,default=16)
    ap.add_argument("--n-forced",type=int,default=3); ap.add_argument("--output-dir",default="/tmp/instance_storage/gu/eval_out")
    ap.add_argument("--tag",default="cf"); ap.add_argument("--shard-index",type=int,default=0)
    ap.add_argument("--num-shards",type=int,default=1); ap.add_argument("--merge",action="store_true")
    a=ap.parse_args(); merge(a) if a.merge else run(a)


if __name__=="__main__":
    main()
