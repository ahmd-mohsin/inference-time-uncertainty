# CODE eval PHASE 2 (offline scoring, NO vLLM/GPU) — read code_gen jsonl, run _passvec per completion, report pass@1/solve@k.
# Usage: python -m rl_training.code_score --gen /tmp/instance_storage/gu/code_out/cg_X.jsonl --tag X
import argparse, json, os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl_training.rewards import _passvec, _extract_code

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True); ap.add_argument("--tag", default="cs")
    ap.add_argument("--output-dir", default="/tmp/instance_storage/gu/eval_out")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.gen) if l.strip()]
    p1_sum = 0.0; solved = 0
    for r in rows:
        comps = r["completions"]; passes = 0
        for c in comps:
            pv = _passvec(_extract_code(c), r["test"], r.get("entry"), bool(r.get("mbpp")))
            if pv and all(pv): passes += 1
        p1_sum += passes / max(len(comps), 1); solved += int(passes > 0)
    n = len(rows)
    out = {"tag": a.tag, "n": n, "pass@1": p1_sum/max(n,1), "solve@k": solved/max(n,1)}
    Path(a.output_dir).mkdir(parents=True, exist_ok=True)
    json.dump(out, open(Path(a.output_dir)/f"cs_{a.tag}.json", "w"))
    print(f"[code_score {a.tag}] pass@1={out['pass@1']:.4f} solve@k={out['solve@k']:.4f} n={n}")

if __name__ == "__main__":
    main()
