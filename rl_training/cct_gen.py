"""CCT C0 rollout+audit on ONE GPU: sample candidates from an RFT checkpoint on a shard of DAG tasks,
run the whole/observed/conditional-contract audit (cct_c0), aggregate the C0 statistics, dump JSON.

Fan across the fleet: one process per GPU, each a disjoint seed shard. Aggregate the JSONs for the C0 verdict:
does conditional-contract checking recover correct components that whole-solution (and observed) masking discards,
on the RFT model's OWN failed rollouts?
"""
import argparse, json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl_training.comp_tasks import _rand_records, _extract
from rl_training.comp_dag import gen_dag, render
from rl_training.cct_c0 import audit_candidate

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--n-tasks", type=int, default=64)
    ap.add_argument("--n-nodes", type=int, default=6, help="DAG size (blocks per task)")
    ap.add_argument("--k", type=int, default=6, help="candidates per task")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    # build DAG task shard: each task = (dag, fixed eval records)
    tasks = []
    rr = random.Random(a.seed0 * 991 + 7)
    for s in range(a.seed0, a.seed0 + a.n_tasks):
        dag = gen_dag(s, n_nodes=a.n_nodes)
        records = _rand_records(rr, rr.randint(6, 10))
        tasks.append({"seed": s, "dag": dag, "records": records, "prompt": render(dag)})

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    llm = LLM(model=a.model, trust_remote_code=True, dtype="bfloat16",
              gpu_memory_utilization=float(os.environ.get("GEN_GPU_MEM", "0.5")),
              max_model_len=2048, enforce_eager=True)
    sp = SamplingParams(n=a.k, temperature=a.temperature, top_p=0.95, max_tokens=a.max_tokens)

    def chat(p):
        try:
            return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        except Exception:
            return p + "\n"

    outs = llm.generate([chat(t["prompt"]) for t in tasks], sp)
    rng = random.Random(a.seed0 + 12345)
    agg = {"model": a.model, "seed0": a.seed0, "n_tasks": a.n_tasks, "k": a.k, "n_nodes": a.n_nodes,
           "candidates": 0, "excluded_task": 0, "blocks_parsed": 0, "blocks_unparsed": 0,
           "whole_pass": 0, "whole_fail": 0,
           # among whole-FAIL candidates only (the CCT regime):
           "wf_blocks": 0, "wf_observed_pass": 0, "wf_conditional_pass": 0,
           "wf_cond_recovered": 0,      # whole-fail block: observed FAIL + conditional PASS + binding OK (recovered, discarded by whole&observed)
           "wf_accidental_rej": 0,      # observed PASS but conditional FAIL (accidental agreement, correctly rejected)
           "wf_binding_fail": 0,        # conditional-correct transform but wrong dependency binding
           "recovered_ops": {}}
    for t, o in zip(tasks, outs):
        for c in o.outputs:
            code = _extract(c.text)
            if not code:
                agg["blocks_unparsed"] += 1; continue
            blocks = audit_candidate(t["records"], t["dag"], code, rng)
            if blocks is None:
                agg["excluded_task"] += 1; continue
            agg["candidates"] += 1
            parsed = [b for b in blocks if b["parsed"]]
            agg["blocks_parsed"] += len(parsed); agg["blocks_unparsed"] += len(blocks) - len(parsed)
            whole = blocks[0]["whole"] if blocks else False
            agg["whole_pass" if whole else "whole_fail"] += 1
            if whole:
                continue  # CCT regime is the FAILED whole-solutions
            for b in parsed:
                agg["wf_blocks"] += 1
                if b["observed"]: agg["wf_observed_pass"] += 1
                if b["conditional"]: agg["wf_conditional_pass"] += 1
                if (not b["observed"]) and b["conditional"] and b["binding"]:
                    agg["wf_cond_recovered"] += 1
                    agg["recovered_ops"][b["op"]] = agg["recovered_ops"].get(b["op"], 0) + 1
                if b["observed"] and (not b["conditional"]):
                    agg["wf_accidental_rej"] += 1
                if b["conditional"] and (not b["binding"]):
                    agg["wf_binding_fail"] += 1
    with open(a.out, "w") as f:
        json.dump(agg, f, indent=2)
    rr = agg["wf_cond_recovered"]; ob = agg["wf_observed_pass"]
    print(f"[cct_gen] cand={agg['candidates']} whole_fail={agg['whole_fail']} wf_blocks={agg['wf_blocks']} "
          f"cond_recovered={rr} observed_pass={ob} accidental_rej={agg['wf_accidental_rej']} -> {a.out}")

if __name__ == "__main__":
    main()
