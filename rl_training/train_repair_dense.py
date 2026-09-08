# Exp7 DENSE / RESIDUAL-reward self-repair GRPO. Fixes the sparse-binary reward that made v1/v2 null.
# Dataset rows (jsonl): prompt (certificate-style repair prompt, CODE HIDDEN), test, entry, mbpp,
# and optionally pfail = list[bool] parent's per-assert outcome (for residual). Reward arms:
#   binary   : 1 if all tests pass
#   fraction : mean per-assert pass fraction (+ α·all-pass)          [dense]
#   residual : (fixed among parent-FAILING)/|F-| − λ·(regressed among parent-PASSING)/|S-| + α·all-pass
# Per-assert verification runs in a subprocess with a timeout (safe vs infinite loops).
import argparse, json, os, sys, tempfile, subprocess, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def extract_code(t):
    m = re.findall(r"```(?:python)?\n(.*?)```", t or "", re.DOTALL); return m[0] if m else (t or "")

_HARNESS = r"""
import json,sys
{code}
__ok=[]
{checks}
print("PASSVEC:"+json.dumps(__ok))
"""
def per_assert_vec(code, test, entry, mbpp, timeout=8):
    """Return list[bool] per assert (True=pass). Subprocess with timeout."""
    import ast
    try: tree = ast.parse(test)
    except Exception: return []
    checks = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            seg = ast.get_source_segment(test, node.test)
            if seg: checks.append(seg)
    if not checks: return []
    cand = "" if mbpp else f"candidate={entry}\n"
    body = "\n".join(f"try:\n    __ok.append(bool({c}))\nexcept Exception:\n    __ok.append(False)" for c in checks)
    prog = _HARNESS.format(code=cand+code, checks=body)
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f: f.write(prog); path=f.name
        r = subprocess.run(["python3", path], capture_output=True, text=True, timeout=timeout)
        for line in (r.stdout or "").splitlines():
            if line.startswith("PASSVEC:"): return json.loads(line[8:])
    except Exception: pass
    finally:
        try: os.unlink(path)
        except Exception: pass
    return [False]*len(checks)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--output-dir", required=True); ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--reward", default="residual", choices=["binary","fraction","residual"])
    ap.add_argument("--num-generations", type=int, default=8); ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lora-r", type=int, default=16); ap.add_argument("--alpha", type=float, default=0.5); ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--max-completion-length", type=int, default=1024); ap.add_argument("--vllm-mode", default="server")
    a = ap.parse_args()
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer
    from peft import LoraConfig
    rows = [json.loads(l) for l in open(a.data) if l.strip()]
    ds = Dataset.from_list([{"prompt": r["prompt"], "test": r["test"], "entry": r.get("entry"),
                             "mbpp": bool(r.get("mbpp")), "pfail": r.get("pfail")} for r in rows])
    print(f"[dense-grpo:{a.reward}] {len(ds)} prompts")
    MODE = a.reward
    def reward_fn(completions, test, entry, mbpp, pfail=None, **kw):
        out = []
        pfail = pfail or [None]*len(completions)
        for c, t, e, m, pf in zip(completions, test, entry, mbpp, pfail):
            code = extract_code(c if isinstance(c, str) else c[-1]["content"])
            vec = per_assert_vec(code, t, e, m)
            if not vec: out.append(0.0); continue
            allp = all(vec)
            if MODE == "binary": out.append(1.0 if allp else 0.0); continue
            if MODE == "fraction": out.append(sum(vec)/len(vec) + (a.alpha if allp else 0.0)); continue
            # residual
            if pf and len(pf) == len(vec):
                Fm = [i for i in range(len(vec)) if not pf[i]]; Sm = [i for i in range(len(vec)) if pf[i]]
                fixed = sum(1 for i in Fm if vec[i])/max(len(Fm),1)
                regr = sum(1 for i in Sm if not vec[i])/max(len(Sm),1)
                out.append(fixed - a.lam*regr + (a.alpha if allp else 0.0))
            else:
                out.append(sum(vec)/len(vec) + (a.alpha if allp else 0.0))
        return out
    cfg = GRPOConfig(output_dir=a.output_dir, learning_rate=a.lr, per_device_train_batch_size=a.num_generations,
        num_generations=a.num_generations, max_completion_length=a.max_completion_length, max_steps=a.steps,
        save_steps=max(a.steps//3,1), save_total_limit=2, logging_steps=1, gradient_accumulation_steps=1,
        bf16=True, beta=0.0, use_vllm=True, vllm_mode=a.vllm_mode, report_to="none", temperature=1.0, top_p=0.95)
    peft_cfg = LoraConfig(r=a.lora_r, lora_alpha=a.lora_r*2, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
    tr = GRPOTrainer(model=a.model, reward_funcs=[reward_fn], args=cfg, train_dataset=ds, peft_config=peft_cfg)
    tr.train(); tr.save_model(a.output_dir); open(os.path.join(a.output_dir,"DENSE_DONE"),"w").write(a.reward)
    print(">> dense-grpo done ->", a.output_dir)

if __name__ == "__main__":
    main()
