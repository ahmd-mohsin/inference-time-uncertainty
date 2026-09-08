# Eval a checkpoint on a controlled-task split (§50). Raw prompts (already fully formed by
# controlled_tasks.py) + executable integer verifier. Reports mean_p (pass rate) + per-problem.
# Usage: python -m rl_training.ctrl_eval --model-path <dir> --data <split.jsonl> --k 4 --tag ct_x
import argparse, json, os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl_training.controlled_tasks import verify
from rl_training.seq_recover import chat

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--k", type=int, default=4); ap.add_argument("--n", type=int, default=-1)
    ap.add_argument("--tag", default="ct"); ap.add_argument("--output-dir", default="/tmp/instance_storage/gu/eval_out")
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    from rl_training.model_utils import merge_adapter_if_needed
    mp = merge_adapter_if_needed(a.model_path)
    rows = [json.loads(l) for l in open(a.data) if l.strip()]
    if a.n > 0: rows = rows[:a.n]
    llm = LLM(model=mp, dtype="bfloat16", trust_remote_code=True, tensor_parallel_size=1, max_model_len=2048,
              gpu_memory_utilization=float(os.environ.get("EVAL_GPU_MEM", 0.85)), enforce_eager=True)
    sp = SamplingParams(n=a.k, temperature=1.0, top_p=0.95, max_tokens=1024,
                        stop=["<|im_end|>", "<|eot_id|>", "<|endoftext|>"])
    outs = llm.generate([chat(mp, r["prompt"]) for r in rows], sp)
    per = []
    for r, o in zip(rows, outs):
        c = sum(int(verify(s.text, r["gold"])) for s in o.outputs)
        per.append({"gold": r["gold"], "k": len(o.outputs), "correct": c, "p": c / max(len(o.outputs), 1),
                    "family": r.get("family"), "split": r.get("split")})
    out = {"tag": a.tag, "data": a.data, "n": len(per), "k": a.k,
           "mean_p": sum(x["p"] for x in per) / max(len(per), 1),
           "solved_any": sum(1 for x in per if x["correct"] > 0) / max(len(per), 1), "per": per}
    Path(a.output_dir).mkdir(parents=True, exist_ok=True)
    json.dump(out, open(Path(a.output_dir) / f"ct_{a.tag}.json", "w"))
    print(f"[ct {a.tag}] n={out['n']} mean_p={out['mean_p']:.4f} solved_any={out['solved_any']:.4f}")

if __name__ == "__main__":
    main()
