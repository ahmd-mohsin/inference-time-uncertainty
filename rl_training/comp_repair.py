# Self-Repair Distillation for the compositional domain: recover base pass@k=0 "frontier" problems
# via execute->feedback->repair, then emit a SINGLE-SHOT distillation bank (breaks the sampling ceiling
# that plain verified-RFT/ReST-EM is capped at). One GPU, batched multi-turn.
import argparse, json, os
from copy import deepcopy
from vllm import LLM, SamplingParams
from rl_training.comp_tasks import run_program, verify_solution, _extract

def feedback(code, task):
    inp = task["test_inputs"][0]; exp = run_program(inp, task["prog"])
    try:
        ns = {}; exec(_extract(code), ns); got = ns["solve"](deepcopy(inp))
        got = str(got)[:600]
    except Exception as e:
        got = f"raised {type(e).__name__}: {e}"
    return json.dumps(inp), got, str(exp)[:600]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--pool", required=True)
    ap.add_argument("--k", type=int, default=8); ap.add_argument("--repair-turns", type=int, default=4)
    ap.add_argument("--n", type=int, default=400); ap.add_argument("--out", required=True)
    ap.add_argument("--stats-out", default=""); a = ap.parse_args()
    tasks = [json.loads(l) for l in open(a.pool)][:a.n]
    llm = LLM(model=a.model, dtype="bfloat16", trust_remote_code=True, max_model_len=4096,
              gpu_memory_utilization=float(os.environ.get("GEN_GPU_MEM", 0.45)), enforce_eager=True)
    tok = llm.get_tokenizer()
    def wrap(p): return tok.apply_chat_template([{"role":"user","content":p}], tokenize=False, add_generation_prompt=True)
    # ---- single-shot pass@k ----
    sp = SamplingParams(n=a.k, temperature=1.0, top_p=0.95, max_tokens=1024, stop=["<|im_end|>","<|endoftext|>"])
    outs = llm.generate([wrap(t["prompt"]) for t in tasks], sp)
    bank = []; solved_ss = set(); last_fail = {}
    for i, (t, o) in enumerate(zip(tasks, outs)):
        hit = None
        for s in o.outputs:
            if verify_solution(s.text, t): hit = s.text.strip(); break
        if hit is not None:
            solved_ss.add(i); bank.append({"prompt": wrap(t["prompt"]), "completion": hit})
        else:
            last_fail[i] = o.outputs[0].text  # seed the repair loop with a failed attempt
    # ---- repair loop on pass@k=0 frontier problems ----
    frontier = [i for i in range(len(tasks)) if i not in solved_ss]
    repaired = set()
    for turn in range(a.repair_turns):
        pend = [i for i in frontier if i not in repaired]
        if not pend: break
        prompts = []
        for i in pend:
            inp, got, exp = feedback(last_fail[i], tasks[i])
            rp = (tasks[i]["prompt"] + f"\n\nYour previous attempt:\n```python\n{_extract(last_fail[i])}\n```\n"
                  f"When run on records={inp}, it returned {got}, but the expected result is {exp}. "
                  "The bug is in the order or logic of the pipeline steps. Fix `solve`. Return ONLY a ```python code block.")
            prompts.append(wrap(rp))
        ro = llm.generate(prompts, SamplingParams(n=1, temperature=0.8, top_p=0.95, max_tokens=1024, stop=["<|im_end|>","<|endoftext|>"]))
        for i, o in zip(pend, ro):
            txt = o.outputs[0].text; last_fail[i] = txt
            if verify_solution(txt, tasks[i]):
                repaired.add(i)
                # distill the RECOVERED solution as a clean single-shot target
                bank.append({"prompt": wrap(tasks[i]["prompt"]), "completion": txt.strip()})
    with open(a.out, "w") as f:
        for r in bank: f.write(json.dumps(r) + "\n")
    stats = {"n": len(tasks), "single_shot_cov": len(solved_ss), "frontier_pk0": len(frontier),
             "repair_recovered": len(repaired), "total_cov": len(solved_ss) + len(repaired),
             "ceiling_break": len(repaired)}
    print("[comp_repair]", json.dumps(stats))
    if a.stats_out: json.dump(stats, open(a.stats_out, "w"))

if __name__ == "__main__": main()
