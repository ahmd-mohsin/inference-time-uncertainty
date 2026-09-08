# Generate the SELF-REPAIR GRPO training set: for each code problem, sample one attempt; on FAILURE record
# the repair prompt (problem + the real error, CODE HIDDEN — the error_only recipe) plus the unit-test
# metadata the GRPO reward needs to execute-verify completions. 8-GPU DP + merge to a jsonl.
# Usage: python -m rl_training.dump_repair_data --model-path <dir> --bench mbpp --tag rep_qc \
#   --shard-index S --num-shards 8
import argparse, json, os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl_training.code_passk import load_bench, extract_code, run_tests
from rl_training.seq_recover import base_task, chat

REPAIR_INSTR = ("\n\nA previous attempt at this problem FAILED with this error:\n{err}\n"
                "(the failed code is hidden on purpose). Diagnose the likely cause and write a correct, "
                "complete solution in a ```python block.")
# RAW-state (Exp1): the failed proposal is RETAINED in the repair state (CEGIS-style anchored context).
REPAIR_INSTR_RAW = ("\n\nYour previous attempt (below) FAILED with this error:\n{err}\n"
                    "```python\n{code}\n```\nFix it and write a correct, complete solution in a ```python block.")

def run(a):
    from vllm import LLM, SamplingParams
    from rl_training.model_utils import merge_adapter_if_needed
    mp = merge_adapter_if_needed(a.model_path)
    items = load_bench(a.bench)
    if a.num_shards > 1: items = items[a.shard_index::a.num_shards]
    llm = LLM(model=mp, dtype="bfloat16", trust_remote_code=True, tensor_parallel_size=1,
              max_model_len=4096, gpu_memory_utilization=float(os.environ.get("EVAL_GPU_MEM",0.85)),
              enable_prefix_caching=True, enforce_eager=True)
    sp = SamplingParams(n=a.n_probe, temperature=1.0, top_p=0.95, max_tokens=1024,
                        stop=["<|im_end|>","<|eot_id|>","<|endoftext|>"])
    outs = llm.generate([chat(mp, base_task(it)) for it in items], sp)
    from rl_training.rewards import _passvec
    recs = []
    for it, o in zip(items, outs):
        # take the first FAILED sample's code+error as the repair condition; skip problems solved every time
        err = None; fcode = None
        for s in o.outputs:
            c = extract_code(s.text); ok, e = run_tests(c, it, return_err=True)
            if not ok: err = e; fcode = c; break
        if err is None:   # model always solves it -> not a repair case
            continue
        # pfail = parent per-test pass vector (for the residual reward; identical across RAW/EVID arms)
        pfail = _passvec(fcode, it["test"], it.get("entry"), bool(it.get("mbpp")))
        if a.keep_code:
            prompt = base_task(it) + REPAIR_INSTR_RAW.format(err=err, code=fcode[:1500])
        else:
            prompt = base_task(it) + REPAIR_INSTR.format(err=err)
        recs.append({"problem_id": it["id"], "prompt": chat(mp, prompt),
                     "bench": a.bench, "mbpp": bool(it.get("mbpp")),
                     "test": it["test"], "entry": it.get("entry"),
                     "pfail": pfail, "code": fcode[:1500]})
    Path(a.output_dir).mkdir(parents=True, exist_ok=True)
    fp = Path(a.output_dir)/f"repair_{a.tag}.shard{a.shard_index}-of-{a.num_shards}.jsonl"
    with open(fp,"w") as f:
        for r in recs: f.write(json.dumps(r)+"\n")
    print(f"[{a.tag} shard {a.shard_index}] {len(recs)} repair prompts -> {fp}")

def merge(a):
    recs=[]
    for s in range(a.num_shards):
        fp=Path(a.output_dir)/f"repair_{a.tag}.shard{s}-of-{a.num_shards}.jsonl"
        if fp.exists():
            recs+=[json.loads(l) for l in open(fp)]
    fp=Path(a.output_dir)/f"repair_{a.tag}.jsonl"
    with open(fp,"w") as f:
        for r in recs: f.write(json.dumps(r)+"\n")
    print(f"[{a.tag}] merged {len(recs)} repair prompts -> {fp}")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model-path"); ap.add_argument("--bench",default="mbpp")
    ap.add_argument("--n-probe",type=int,default=3,help="samples/problem to elicit a failure")
    ap.add_argument("--output-dir",default="/tmp/instance_storage/gu/repair_data"); ap.add_argument("--tag",default="rep")
    ap.add_argument("--shard-index",type=int,default=0); ap.add_argument("--num-shards",type=int,default=1); ap.add_argument("--merge",action="store_true")
    ap.add_argument("--keep-code",action="store_true",help="Exp1 RAW-state: retain the failed proposal in the repair prompt")
    a=ap.parse_args(); merge(a) if a.merge else run(a)

if __name__=="__main__":
    main()
