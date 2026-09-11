# M1 Coverage-Targeted Harvesting: instead of uniform k per prompt, sample until first verified success, then
# REDIRECT the saved budget to zero-success prompts at higher k + higher temperature. Matched TOTAL sample budget
# to uniform-k. Emits a verified bank {prompt, completion}. Derived from: frozen saturation (§82b), ~70% dead groups
# (§80b), higher-temp reaches more prompts (§71b). Compare vs uniform comp_gen at matched samples (+ uniform @1.5x control).
import argparse, json, os
from vllm import LLM, SamplingParams
from rl_training.comp_tasks import verify_solution

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--pool", required=True)
    ap.add_argument("--n", type=int, default=400); ap.add_argument("--budget-mult", type=int, default=8,
                    help="total sample budget = n * budget_mult (matched to uniform k=budget_mult)")
    ap.add_argument("--k-base", type=int, default=4); ap.add_argument("--out", required=True)
    ap.add_argument("--stats-out", default=""); a = ap.parse_args()
    tasks = [json.loads(l) for l in open(a.pool)][:a.n]
    llm = LLM(model=a.model, dtype="bfloat16", trust_remote_code=True, max_model_len=2048,
              gpu_memory_utilization=float(os.environ.get("GEN_GPU_MEM", 0.5)), enforce_eager=True)
    tok = llm.get_tokenizer()
    def wrap(p): return tok.apply_chat_template([{"role":"user","content":p}], tokenize=False, add_generation_prompt=True)
    total_budget = a.n * a.budget_mult
    solved = {}  # idx -> completion
    used = 0
    # pass 1: k_base per prompt @ temp 1.0
    sp1 = SamplingParams(n=a.k_base, temperature=1.0, top_p=0.95, max_tokens=1024, stop=["<|im_end|>","<|endoftext|>"])
    outs = llm.generate([wrap(t["prompt"]) for t in tasks], sp1); used += a.n * a.k_base
    for i, (t, o) in enumerate(zip(tasks, outs)):
        for s in o.outputs:
            if verify_solution(s.text, t): solved[i] = s.text.strip(); break
    # pass 2: redistribute remaining budget across UNSOLVED prompts at higher k + temp 1.1
    unsolved = [i for i in range(len(tasks)) if i not in solved]
    if unsolved and used < total_budget:
        k2 = max(1, (total_budget - used) // len(unsolved))
        sp2 = SamplingParams(n=k2, temperature=1.1, top_p=0.97, max_tokens=1024, stop=["<|im_end|>","<|endoftext|>"])
        outs2 = llm.generate([wrap(tasks[i]["prompt"]) for i in unsolved], sp2); used += len(unsolved) * k2
        for i, o in zip(unsolved, outs2):
            for s in o.outputs:
                if verify_solution(s.text, tasks[i]): solved[i] = s.text.strip(); break
    with open(a.out, "w") as f:
        for i, c in solved.items(): f.write(json.dumps({"prompt": wrap(tasks[i]["prompt"]), "completion": c}) + "\n")
    stats = {"n": len(tasks), "budget_mult": a.budget_mult, "total_budget": total_budget, "samples_used": used,
             "N_dist": len(solved), "pass1_solved": sum(1 for i in solved if i not in unsolved) if False else None,
             "unsolved_after_pass1": len(unsolved), "recovered_pass2": len([i for i in unsolved if i in solved])}
    print("[cth]", json.dumps(stats))
    if a.stats_out: json.dump(stats, open(a.stats_out, "w"))

if __name__ == "__main__": main()
