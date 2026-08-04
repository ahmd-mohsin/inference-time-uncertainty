#!/usr/bin/env python3
"""Option B: dense-bank certificate validation.

Empirically show that the set-form p_ref (sum of pi_ref over sampled DISTINCT correct traces)
CONVERGES to the measured base pass@1 as the number of samples grows -- validating that the
certificate's p_ref = base pass@1 (Option A) is the right quantity, not the vacuous single-trace value.

For a small set of fragile-band problems: sample base N times each, verify each, and for the CORRECT
samples compute teacher-forced pi_base(y|q). Then report, as a function of #samples m:
  - pass@1 estimate = (#correct so far)/m
  - set_mass(m)     = sum over DISTINCT correct traces of exp(logprob)   [the certificate's p_ref]
Show set_mass -> pass@1 and the resulting certificate 1-(1-alpha*set_mass)^k stabilizes non-vacuously.
Single GPU (vLLM sample + HF logprob). Small problem set so it finishes fast.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.dataset import get_inference_dataset, format_prompt
from rl_training.safe_match import safe_is_correct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="Qwen/Qwen2.5-Math-7B")
    ap.add_argument("--dataset", default="olympiad_bench")
    ap.add_argument("--difficulty-json", required=True)
    ap.add_argument("--n-problems", type=int, default=20, help="small set to keep it fast")
    ap.add_argument("--pass1-min", type=float, default=0.05, help="pick problems with measurable pass@1")
    ap.add_argument("--pass1-max", type=float, default=0.5)
    ap.add_argument("--samples", type=int, default=256, help="base samples per problem (dense)")
    ap.add_argument("--max-new-tokens", type=int, default=3072)
    ap.add_argument("--out", default="rl_training/runs/dense_cert.json")
    a = ap.parse_args()

    import torch
    from vllm import LLM, SamplingParams
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from rl_training.support_ratchet import sequence_logprob

    d = json.load(open(a.difficulty_json))
    lab = {p["problem_id"]: p for p in d["per_problem"]}
    probs = get_inference_dataset({"dataset": {"name": a.dataset, "split": "test",
                                               "n_problems": -1, "seed": 42}})
    # pick problems in the measurable-pass1 window (so set_mass has something to converge to)
    picked = [p for p in probs if a.pass1_min <= lab.get(p["problem_id"], {}).get("pass1", 0) <= a.pass1_max]
    picked = picked[:a.n_problems]
    print(f"[dense] {len(picked)} problems, pass1 in [{a.pass1_min},{a.pass1_max}], {a.samples} samples each")

    llm = LLM(model=a.model_path, dtype="bfloat16", trust_remote_code=True, tensor_parallel_size=1,
              max_model_len=a.max_new_tokens + 1024, gpu_memory_utilization=0.85,
              enable_prefix_caching=True)
    sp = SamplingParams(n=a.samples, max_tokens=a.max_new_tokens, temperature=1.0, top_p=1.0,
                        stop=["<|im_end|>", "<|endoftext|>"])
    outs = llm.generate([format_prompt(p, a.model_path) for p in picked], sp)

    # collect correct completions per problem
    correct_texts = {}
    pass1 = {}
    for p, o in zip(picked, outs):
        gold = str(p.get("gold_answer", ""))
        cs = [s.text for s in o.outputs if safe_is_correct(s.text, gold)[0]]
        correct_texts[p["problem_id"]] = (p, cs)
        pass1[p["problem_id"]] = len(cs) / a.samples
    del llm; torch.cuda.empty_cache()

    # teacher-forced base logprob for DISTINCT correct traces
    tok = AutoTokenizer.from_pretrained(a.model_path)
    model = AutoModelForCausalLM.from_pretrained(a.model_path, torch_dtype=torch.bfloat16).to("cuda").eval()
    import math
    res = []
    with torch.no_grad():
        for pid, (p, cs) in correct_texts.items():
            distinct = list(dict.fromkeys(cs))  # dedup identical strings
            lps = []
            prompt = format_prompt(p, a.model_path)
            for y in distinct:
                pids = tok(prompt, add_special_tokens=False)["input_ids"]
                cids = tok(y, add_special_tokens=False)["input_ids"]
                if not cids: continue
                ids = torch.tensor([pids + cids], device="cuda")
                m = torch.tensor([[0]*len(pids) + [1]*len(cids)], dtype=torch.float32, device="cuda")
                lps.append(float(sequence_logprob(model(ids).logits[:, :-1, :], ids[:, 1:], m[:, 1:])[0]))
            # set_mass = sum exp(lp) over distinct correct traces
            set_mass = sum(math.exp(x) for x in lps) if lps else 0.0
            res.append({"problem_id": pid, "pass1": pass1[pid], "n_correct": len(cs),
                        "n_distinct": len(distinct), "set_mass": set_mass,
                        "set_logmass": (math.log(set_mass) if set_mass > 0 else None)})
            print(f"  q{pid}: pass1={pass1[pid]:.3f} n_correct={len(cs)} distinct={len(distinct)} "
                  f"set_mass={set_mass:.4g}  (ratio set_mass/pass1={set_mass/max(pass1[pid],1e-9):.2f})")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"samples": a.samples, "per_problem": res}, open(a.out, "w"), indent=2)
    # summary: does set_mass track pass1? and certificate at alpha=0.5
    import statistics as st
    ratios = [r["set_mass"]/max(r["pass1"], 1e-9) for r in res if r["pass1"] > 0]
    def cert(p, al, k): return 1 - (1 - al*p)**k
    print(f"\n[dense] set_mass/pass1 ratio: median={st.median(ratios):.2f} (→1.0 means set_mass recovers pass@1)")
    for K in [64, 256]:
        c = [cert(r["set_mass"], 0.5, K) for r in res]
        cp = [cert(r["pass1"], 0.5, K) for r in res]
        print(f"  certified pass@{K}: from set_mass={sum(c)/len(c):.4f} | from pass1(OptionA)={sum(cp)/len(cp):.4f}")
    print(f"[dense] wrote {a.out}")


if __name__ == "__main__":
    main()
