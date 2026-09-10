# E1 oracle diagnostic (reviewer priority #3): on the base's OBSERVED-ZERO-SUCCESS cohort (budget k), does giving the
# correct per-primitive code (ORACLE COMPONENTS; wiring/order is already in the prompt) recover the problem? Distinguishes
# "local execution/component" bottleneck (components-given recovers -> recomposition won't help; fix local solving) from
# "composition/assembly" bottleneck (still fails with correct components -> interface/recomposition intervention warranted).
import argparse, json, os
from vllm import LLM, SamplingParams
from rl_training.comp_tasks import verify_solution, PRIMS

def comp_block(prog):
    lines = []
    for j, name in enumerate(prog, 1):
        code = PRIMS[name]["code"]
        lines.append(f"# step {j} ({name}):\n" + "\n".join("    " + l for l in code.split("\n")))
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--pool", required=True)
    ap.add_argument("--k", type=int, default=8); ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--stats-out", default=""); a = ap.parse_args()
    tasks = [json.loads(l) for l in open(a.pool)][:a.n]
    llm = LLM(model=a.model, dtype="bfloat16", trust_remote_code=True, max_model_len=4096,
              gpu_memory_utilization=float(os.environ.get("GEN_GPU_MEM", 0.5)), enforce_eager=True)
    tok = llm.get_tokenizer()
    def wrap(p): return tok.apply_chat_template([{"role":"user","content":p}], tokenize=False, add_generation_prompt=True)
    sp = SamplingParams(n=a.k, temperature=1.0, top_p=0.95, max_tokens=1024, stop=["<|im_end|>","<|endoftext|>"])
    # 1) direct pass@k -> observed-zero-success cohort (frontier)
    outs = llm.generate([wrap(t["prompt"]) for t in tasks], sp)
    frontier = []
    for t, o in zip(tasks, outs):
        if not any(verify_solution(s.text, t) for s in o.outputs): frontier.append(t)
    # 2) ORACLE-COMPONENTS arm on the frontier: give correct per-step code, ask to compose
    comp_prompts = [wrap(t["prompt"] +
        "\n\nHere is CORRECT reference code for each pipeline step (in order):\n```python\n" + comp_block(t["prog"]) +
        "\n```\nAssemble these steps IN ORDER into a single `def solve(records):` that applies them and returns the result. "
        "Return ONLY a ```python code block defining `solve`.") for t in frontier]
    comp_rec = 0
    if comp_prompts:
        couts = llm.generate(comp_prompts, sp)
        for t, o in zip(frontier, couts):
            if any(verify_solution(s.text, t) for s in o.outputs): comp_rec += 1
    # 3) positive control: reproduce full reference (harness sanity) on a sample of frontier
    stats = {"n": len(tasks), "frontier_zero_success_at_k": len(frontier), "budget_k": a.k,
             "oracle_components_recovered": comp_rec,
             "oracle_components_recovery_rate": round(comp_rec / max(1, len(frontier)), 3)}
    print("[oracle_diag]", json.dumps(stats))
    if a.stats_out: json.dump(stats, open(a.stats_out, "w"))

if __name__ == "__main__": main()
