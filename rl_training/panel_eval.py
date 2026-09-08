# BOTTLENECK-LOCATION: estimate p_t(q)=Pr_pi[fully correct | q] on FIXED problem panels, per checkpoint.
# Panels (seed-fixed, identical across all checkpoints — the whole point is a controlled comparison):
#   train  : GSM8K train[:N] — problems the GRPO policy was trained on
#   test   : GSM8K test[:N]  — related UNSEEN instances, same distribution (transfer to new instances)
#   math   : MATH-500[:N]    — held-out harder/OOD (compositional transfer)
# For each problem, K fresh samples -> p(q)=correct/K. Mean over panel = panel accuracy at this checkpoint.
# Comparing trajectories across checkpoints separates: TRAIN rises + TEST flat => generalization bottleneck;
# both rise => learning fine; rise-then-fall => retention. Self-contained GSM8K/MATH answer match; no src dep.
# Usage: python -m rl_training.panel_eval --model-path <adapter|hf-id> --panel test --tag pe_s1_ck200 \
#   --n 200 --k 8 --shard-index S --num-shards 8 [--merge]
import argparse, json, os, re, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl_training.seq_recover import chat

PROMPT = "Solve the problem. Show brief reasoning and put the final answer in \\boxed{}.\n\nProblem: {q}"

def load_panel(panel, n):
    from datasets import load_dataset
    items = []
    if panel in ("train", "test"):
        d = load_dataset("openai/gsm8k", "main")[panel]
        for r in d: items.append({"q": r["question"], "gold": r["answer"].split("####")[-1].strip().replace(",", "")})
    elif panel == "math":
        try: d = load_dataset("HuggingFaceH4/MATH-500")["test"]
        except Exception: d = load_dataset("qq8933/MATH500")["test"]
        for r in d: items.append({"q": r.get("problem") or r.get("question"), "gold": str(r.get("answer") or "")})
    elif panel == "svamp":
        # closer-OOD: grade-school arithmetic word problems, different source than GSM8K
        d = load_dataset("ChilleD/SVAMP")["test"]
        for r in d:
            body = (r.get("Body") or "").strip(); q = (r.get("Question") or "").strip()
            items.append({"q": (body + " " + q).strip(), "gold": str(r.get("Answer"))})
    elif panel == "asdiv":
        # near-OOD: ASDiv diverse arithmetic word problems (different source/distribution than GSM8K)
        d = None
        for did in ("yimingzhang/asdiv", "MU-NLPC/Calc-asdiv_a", "nguyen-brat/asdiv"):
            try: d = load_dataset(did)["test" if "test" in load_dataset(did) else "train"]; break
            except Exception: continue
        if d is None: raise SystemExit("asdiv unavailable")
        for r in d:
            # yimingzhang/asdiv schema: text=question ("Question: ..."), label=gold answer, target=formula
            q = r.get("text") or r.get("question") or r.get("body") or r.get("Body") or ""
            g = r.get("label") or r.get("answer") or r.get("Answer") or r.get("result") or ""
            q = str(q).replace("Question:", "").strip()
            g = str(g).split("####")[-1].strip()
            items.append({"q": q, "gold": g})
    elif panel.startswith("aime"):
        # year-versioned AIME (contamination control: 2025/2026 post-date most pretraining). panel="aime24|aime25|aime26"
        yr = panel[-2:]
        cands = {"24": ["HuggingFaceH4/aime_2024","Maxwell-Jia/AIME_2024","AI-MO/aimo-validation-aime"],
                 "25": ["yentinglin/aime_2025","opencompass/AIME2025","MathArena/aime_2025"],
                 "26": ["MathArena/aime_2026","opencompass/AIME2026"]}.get(yr, [])
        d = None
        for did in cands:
            try:
                dd = load_dataset(did); sp = "test" if "test" in dd else list(dd.keys())[0]; d = dd[sp]; break
            except Exception: continue
        if d is None: raise SystemExit(f"aime{yr} unavailable")
        for r in d:
            q = r.get("problem") or r.get("question") or r.get("Problem") or ""
            g = r.get("answer") or r.get("Answer") or r.get("solution") or ""
            items.append({"q": str(q).strip(), "gold": str(g).strip()})
    elif panel == "amc":
        # far-OOD, harder than MATH: AMC competition problems
        d = None
        for did in ("AI-MO/aimo-validation-amc", "math-ai/amc23"):
            try: d = load_dataset(did)["train" if "train" in load_dataset(did) else "test"]; break
            except Exception: continue
        if d is None: raise SystemExit("amc unavailable")
        for r in d:
            items.append({"q": (r.get("problem") or r.get("question") or "").strip(), "gold": str(r.get("answer") or "")})
    else:
        raise SystemExit("bad panel " + panel)
    return items[:n] if n > 0 else items

def norm(x):
    x = str(x).strip().strip("$").replace(",", "").replace(" ", "").rstrip(".")
    try: return str(int(float(x)))
    except Exception: return x

def extract(t):
    if not t: return None
    m = re.findall(r"\\boxed\{([^}]*)\}", t)
    if m: return norm(m[-1])
    m = re.findall(r"####\s*(.+)", t)
    if m: return norm(m[-1])
    m = re.findall(r"(?:final answer|answer)\s*(?:is|:|=)\s*\$?(-?[\d,\./]+)", t, re.I)
    if m: return norm(m[-1])
    m = re.findall(r"-?\d[\d,]*\.?\d*", t.replace(",", ""))
    return norm(m[-1]) if m else None

def match(pred, gold):
    if pred is None: return False
    p, g = norm(pred), norm(gold)
    if p == g: return True
    try: return abs(float(p) - float(g)) < 1e-6
    except Exception: return False

def run(a):
    from vllm import LLM, SamplingParams
    from rl_training.model_utils import merge_adapter_if_needed
    mp = merge_adapter_if_needed(a.model_path)
    items = load_panel(a.panel, a.n)
    if a.num_shards > 1: items = items[a.shard_index::a.num_shards]
    llm = LLM(model=mp, dtype="bfloat16", trust_remote_code=True, tensor_parallel_size=1, max_model_len=2048,
              gpu_memory_utilization=float(os.environ.get("EVAL_GPU_MEM", 0.85)), enable_prefix_caching=True, enforce_eager=True)
    sp = SamplingParams(n=a.k, temperature=1.0, top_p=0.95, max_tokens=1024, stop=["<|im_end|>", "<|eot_id|>", "<|endoftext|>"])
    outs = llm.generate([chat(mp, PROMPT.replace("{q}", it["q"])) for it in items], sp)
    per = []
    for it, o in zip(items, outs):
        c = sum(int(match(extract(s.text), it["gold"])) for s in o.outputs)
        per.append({"gold": it["gold"], "k": len(o.outputs), "correct": c, "p": c/max(len(o.outputs), 1)})
    out = {"tag": a.tag, "panel": a.panel, "model": mp.split("/")[-1], "n": len(items),
           "mean_p": sum(x["p"] for x in per)/max(len(per), 1),
           "solved_any": sum(1 for x in per if x["correct"] > 0)/max(len(per), 1), "per": per}
    Path(a.output_dir).mkdir(parents=True, exist_ok=True)
    fp = Path(a.output_dir)/(f"pe_{a.tag}.shard{a.shard_index}-of-{a.num_shards}.json" if a.num_shards > 1 else f"pe_{a.tag}.json")
    json.dump(out, open(fp, "w")); print(f"[pe {a.tag} {a.panel} s{a.shard_index}] n={len(items)} mean_p={out['mean_p']:.3f}")

def merge(a):
    per = []
    for s in range(a.num_shards):
        fp = Path(a.output_dir)/f"pe_{a.tag}.shard{s}-of-{a.num_shards}.json"
        if fp.exists():
            d = json.load(open(fp)); per += d["per"]
    out = {"tag": a.tag, "panel": a.panel, "n": len(per),
           "mean_p": sum(x["p"] for x in per)/max(len(per), 1),
           "solved_any": sum(1 for x in per if x["correct"] > 0)/max(len(per), 1)}
    json.dump({**out, "per": per}, open(Path(a.output_dir)/f"pe_{a.tag}.json", "w"))
    print(f"[pe {a.tag}] panel={a.panel} n={out['n']} mean_p={out['mean_p']:.3f} solved_any={out['solved_any']:.3f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True); ap.add_argument("--panel", default="test", choices=["train", "test", "math", "svamp", "asdiv", "amc", "aime24", "aime25", "aime26"])
    ap.add_argument("--n", type=int, default=200); ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--output-dir", default="/tmp/instance_storage/gu/eval_out"); ap.add_argument("--tag", default="pe")
    ap.add_argument("--shard-index", type=int, default=0); ap.add_argument("--num-shards", type=int, default=1); ap.add_argument("--merge", action="store_true")
    a = ap.parse_args(); merge(a) if a.merge else run(a)

if __name__ == "__main__":
    main()
