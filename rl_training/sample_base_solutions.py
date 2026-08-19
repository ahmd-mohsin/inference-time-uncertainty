#!/usr/bin/env python3
# E2 GPU sampler — generate N base solutions/problem WITH TEXT + verify -> bank jsonl.
# Run on a cluster (needs vllm). Usage: python sample_base_solutions.py --model <base> --n 512 \
#   --difficulty-json <diff> --subset hard --out bank_modes.jsonl
import argparse, json, os, sys
os.environ.setdefault("HF_HUB_DISABLE_XET","1"); os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER","0")
sys.path.insert(0, os.getcwd())
from vllm import LLM, SamplingParams
from src.data.dataset import get_inference_dataset, format_prompt
from rl_training.safe_match import answers_match   # existing verifier
ap=argparse.ArgumentParser()
ap.add_argument("--model",required=True); ap.add_argument("--n",type=int,default=512)
ap.add_argument("--dataset",default="olympiad_bench"); ap.add_argument("--difficulty-json",default="")
ap.add_argument("--subset",default="hard"); ap.add_argument("--out",required=True)
ap.add_argument("--max-new-tokens",type=int,default=4096); ap.add_argument("--temperature",type=float,default=1.0)
a=ap.parse_args()
probs=get_inference_dataset({"dataset":{"name":a.dataset,"split":"test","n_problems":-1,"seed":42}})
# subset by difficulty label if given
if a.difficulty_json and os.path.exists(a.difficulty_json):
    diff=json.load(open(a.difficulty_json)); keep={d["problem_id"] for d in diff.get("per_problem",[]) if d.get("label")==a.subset}
    probs=[p for p in probs if int(p["problem_id"]) in keep]
llm=LLM(model=a.model, tensor_parallel_size=1, max_model_len=4096, gpu_memory_utilization=0.9)
sp=SamplingParams(n=a.n, temperature=a.temperature, max_tokens=a.max_new_tokens)
with open(a.out,"w") as w:
    for p in probs:
        prompt=format_prompt(p, a.model); gold=str(p.get("gold_answer",""))
        outs=llm.generate([prompt], sp)[0].outputs
        for o in outs:
            if answers_match(o.text, gold):   # keep only base-CORRECT witnesses
                w.write(json.dumps({"problem_id":int(p["problem_id"]),"prompt":prompt,
                                    "completion":o.text,"gold":gold})+"\n")
print("bank written ->", a.out)
