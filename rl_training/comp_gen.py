# One rejection-FT round for the compositional domain (§64): sample K solutions/prompt from the current model via
# vLLM, VERIFY each with the executable checker, and write ACCEPTED {prompt, completion} for the next SFT step.
# The RFT loop (rft_comp.sh) alternates: comp_gen (this) -> sft_train on accepted -> comp_gen ...
# Usage: python -m rl_training.comp_gen --model <merged dir|id> --pool pool.jsonl --k 4 --out accepted.jsonl
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl_training.comp_tasks import verify_solution


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.pool) if l.strip()][: a.n]
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

    outs = llm.generate([chat(r["prompt"]) for r in rows], sp)
    accepted = []
    n_solved = 0
    for r, o in zip(rows, outs):
        task = {"prog": r["prog"], "test_inputs": r["test_inputs"]}
        best = None
        for c in o.outputs:
            if verify_solution(c.text, task):
                best = c.text; break
        if best is not None:
            n_solved += 1
            # store as a clean ```python block completion for SFT
            accepted.append({"prompt": r["prompt"], "completion": best if "```" in best else "```python\n" + best + "\n```"})
    with open(a.out, "w") as f:
        for r in accepted:
            f.write(json.dumps(r) + "\n")
    print(f"[comp_gen] {n_solved}/{len(rows)} prompts solved (any-of-{a.k}); wrote {len(accepted)} accepted -> {a.out}")


if __name__ == "__main__":
    main()
