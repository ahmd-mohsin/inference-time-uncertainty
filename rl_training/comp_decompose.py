# Decomposition Bootstrapping for the compositional domain: recover base pass@k=0 "frontier" problems by
# STRUCTURED DECOMPOSITION (implement each pipeline step as a helper, then compose) — converts one hard N-step
# problem into N easy 1-step problems the model reliably handles — then DISTILL the recovered solution as a clean
# single-shot (direct-prompt) target. Aims to place mass on the depth ceiling that iid sampling / RFT cannot reach.
import argparse, json, os
from vllm import LLM, SamplingParams
from rl_training.comp_tasks import verify_solution, _extract

DECOMP_SUFFIX = ("\n\nSolve this by DECOMPOSITION: implement EACH pipeline step as its own small helper function "
                 "(step1(records), step2(records), ...), one per step in the exact order listed, each doing ONLY that "
                 "one operation; then define `solve(records)` that calls them in order and returns the result. "
                 "Think step by step, but return ONLY a ```python code block defining all helpers and `solve`.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--pool", required=True)
    ap.add_argument("--k", type=int, default=8); ap.add_argument("--k-decomp", type=int, default=8)
    ap.add_argument("--n", type=int, default=400); ap.add_argument("--out", required=True)
    ap.add_argument("--stats-out", default=""); a = ap.parse_args()
    tasks = [json.loads(l) for l in open(a.pool)][:a.n]
    llm = LLM(model=a.model, dtype="bfloat16", trust_remote_code=True, max_model_len=4096,
              gpu_memory_utilization=float(os.environ.get("GEN_GPU_MEM", 0.45)), enforce_eager=True)
    tok = llm.get_tokenizer()
    def wrap(p): return tok.apply_chat_template([{"role":"user","content":p}], tokenize=False, add_generation_prompt=True)
    # ---- direct single-shot pass@k (baseline coverage) ----
    outs = llm.generate([wrap(t["prompt"]) for t in tasks],
                        SamplingParams(n=a.k, temperature=1.0, top_p=0.95, max_tokens=1024, stop=["<|im_end|>","<|endoftext|>"]))
    bank = []; solved = set()
    for i, (t, o) in enumerate(zip(tasks, outs)):
        for s in o.outputs:
            if verify_solution(s.text, t): solved.add(i); bank.append({"prompt": wrap(t["prompt"]), "completion": s.text.strip()}); break
    frontier = [i for i in range(len(tasks)) if i not in solved]
    # ---- decomposition prompting on pass@k=0 frontier ----
    dprompts = [wrap(tasks[i]["prompt"] + DECOMP_SUFFIX) for i in frontier]
    recovered = set()
    if dprompts:
        douts = llm.generate(dprompts, SamplingParams(n=a.k_decomp, temperature=1.0, top_p=0.95, max_tokens=2048, stop=["<|im_end|>","<|endoftext|>"]))
        for i, o in zip(frontier, douts):
            for s in o.outputs:
                if verify_solution(s.text, tasks[i]):
                    recovered.add(i)
                    # DISTILL under the DIRECT prompt (teach single-shot to produce the working solution)
                    bank.append({"prompt": wrap(tasks[i]["prompt"]), "completion": s.text.strip()}); break
    with open(a.out, "w") as f:
        for r in bank: f.write(json.dumps(r) + "\n")
    stats = {"n": len(tasks), "direct_cov": len(solved), "frontier_pk0": len(frontier),
             "decomp_recovered": len(recovered), "total_cov": len(solved) + len(recovered),
             "ceiling_break": len(recovered)}
    print("[comp_decompose]", json.dumps(stats))
    if a.stats_out: json.dump(stats, open(a.stats_out, "w"))

if __name__ == "__main__": main()
