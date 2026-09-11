# CODE eval PHASE 2 (offline, SEQUENTIAL — guaranteed correct; executor-based pools returned 0 in the shared-PID pod).
# run_tests confirmed working per-call; sequential is slow (~0.5-1s/completion) but correct. Prints progress.
import argparse, json, os, sys, time
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl_training.code_passk import run_tests, extract_code

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True); ap.add_argument("--tag", default="cs")
    ap.add_argument("--timeout", type=int, default=6); ap.add_argument("--max-comp", type=int, default=4)
    ap.add_argument("--output-dir", default="/tmp/instance_storage/gu/eval_out")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.gen) if l.strip()]
    p1_sum = 0.0; solved = 0; t0 = time.time()
    for i, r in enumerate(rows):
        item = {"test": r["test"], "entry": r.get("entry"), "mbpp": bool(r.get("mbpp")), "plus_test": r.get("plus_test")}
        comps = r["completions"][:a.max_comp]
        passes = sum(int(bool(run_tests(extract_code(c), item, timeout=a.timeout))) for c in comps)
        p1_sum += passes/max(len(comps),1); solved += int(passes>0)
        if i % 25 == 0: print(f"[code_score {a.tag}] {i}/{len(rows)} running_p@1={p1_sum/(i+1):.3f} ({time.time()-t0:.0f}s)", flush=True)
    n = len(rows)
    out = {"tag": a.tag, "n": n, "pass@1": p1_sum/max(n,1), "solve@k": solved/max(n,1)}
    Path(a.output_dir).mkdir(parents=True, exist_ok=True); json.dump(out, open(Path(a.output_dir)/f"cs_{a.tag}.json","w"))
    print(f"[code_score {a.tag}] DONE pass@1={out['pass@1']:.4f} solve@k={out['solve@k']:.4f} n={n}", flush=True)
if __name__ == "__main__": main()
