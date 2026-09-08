# MECHANISTIC PROBE (Angle B): discriminate GRPO="reweight known outcomes" from SFT="rewrite the reasoning
# computation". Run on base / A-GRPO-adapter / C-SFT-adapter, all evaluated on HELD-OUT MATH-500.
#   M1  teacher-forced NLL on held-out CORRECT MATH solutions (reusable-computation acquisition)  [key graph]
#   M2  mean token entropy of the policy on its own MATH generations (sharpening / entropy collapse)
#   M3  mean completion length + explicit step-count on MATH (does structured multi-step process transfer?)
#   M4  LoRA-delta Frobenius norm by layer (WHERE the update writes) — offline, no GPU inference
# Usage:
#   python -m rl_training.mech_probe --model-path <hf-id|adapter> --tag base_math   --n 200         # M1/M2/M3
#   python -m rl_training.mech_probe --lora-delta <adapter_dir> --tag A_layers                        # M4 only
import argparse, json, math, os, re, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl_training.panel_eval import PROMPT, extract, match

def load_math(n):
    from datasets import load_dataset
    try: d = load_dataset("HuggingFaceH4/MATH-500")["test"]
    except Exception: d = load_dataset("qq8933/MATH500")["test"]
    items = []
    for r in d:
        q = r.get("problem") or r.get("question")
        items.append({"q": q, "gold": str(r.get("answer") or ""), "sol": (r.get("solution") or "").strip()})
    return items[:n] if n > 0 else items

def m4_lora_delta(adapter_dir, tag, out_dir):
    """M4: per-layer Frobenius norm of B@A LoRA delta. No model load — reads adapter safetensors directly."""
    import torch
    from safetensors.torch import load_file
    fp = None
    for c in ("adapter_model.safetensors", "adapter_model.bin"):
        if (Path(adapter_dir)/c).exists(): fp = Path(adapter_dir)/c; break
    if fp is None: raise SystemExit(f"no adapter weights in {adapter_dir}")
    sd = load_file(str(fp)) if str(fp).endswith(".safetensors") else torch.load(str(fp), map_location="cpu")
    # pair lora_A / lora_B by base module name; delta = B @ A ; group norm by layer index and proj type
    mods = {}
    for k, v in sd.items():
        if "lora_A" in k: mods.setdefault(k.replace("lora_A", "*"), {})["A"] = v.float()
        elif "lora_B" in k: mods.setdefault(k.replace("lora_B", "*"), {})["B"] = v.float()
    by_layer, by_proj = {}, {}
    for name, ab in mods.items():
        if "A" not in ab or "B" not in ab: continue
        delta = ab["B"] @ ab["A"]
        fn = float(delta.norm().item())
        lm = re.search(r"layers\.(\d+)\.", name); li = int(lm.group(1)) if lm else -1
        pm = re.search(r"(q|k|v|o|gate|up|down)_proj", name); pj = pm.group(1) if pm else "other"
        by_layer[li] = by_layer.get(li, 0.0) + fn
        by_proj[pj] = by_proj.get(pj, 0.0) + fn
    out = {"tag": tag, "by_layer": {str(k): by_layer[k] for k in sorted(by_layer)}, "by_proj": by_proj}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    json.dump(out, open(Path(out_dir)/f"mech_{tag}_lora.json", "w"))
    nl = len(by_layer); tot = sum(by_layer.values())
    early = sum(v for k, v in by_layer.items() if 0 <= k < nl/2); late = tot - early
    print(f"[M4 {tag}] layers={nl} total_delta={tot:.2f} early_half={early:.2f} late_half={late:.2f} by_proj={ {k:round(v,2) for k,v in by_proj.items()} }")

def gen_math_traces(a):
    """CONFOUND-FIX prep: sample a model on held-out MATH, keep verifier-correct completions as a STYLE-NEUTRAL
    shared reference set {prompt, completion}. Score all of base/C/A against the SAME set (default: base's own
    correct traces) so M1 measures 'does the update raise prob of reachable correct reasoning' — not style-match
    to human prose. Usage: --gen-traces --model-path <base> --tag q7b_reftraces --n 200 --k 8"""
    from vllm import LLM, SamplingParams
    from rl_training.model_utils import merge_adapter_if_needed
    from rl_training.seq_recover import chat
    mp = merge_adapter_if_needed(a.model_path)
    items = load_math(a.n)
    llm = LLM(model=mp, dtype="bfloat16", trust_remote_code=True, tensor_parallel_size=1, max_model_len=2048,
              gpu_memory_utilization=float(os.environ.get("EVAL_GPU_MEM", 0.85)), enable_prefix_caching=True, enforce_eager=True)
    sp = SamplingParams(n=a.k, temperature=1.0, top_p=0.95, max_tokens=1024, stop=["<|im_end|>", "<|eot_id|>", "<|endoftext|>"])
    outs = llm.generate([chat(mp, PROMPT.replace("{q}", it["q"])) for it in items], sp)
    recs = []
    for it, o in zip(items, outs):
        p = chat(mp, PROMPT.replace("{q}", it["q"]))
        for s in o.outputs:
            if match(extract(s.text), it["gold"]):
                recs.append({"prompt": p, "completion": s.text.strip()}); break
    Path(a.output_dir).mkdir(parents=True, exist_ok=True)
    fp = Path(a.output_dir)/f"reftraces_{a.tag}.jsonl"
    with open(fp, "w") as f:
        for r in recs: f.write(json.dumps(r) + "\n")
    print(f"[gen-traces {a.tag}] {len(recs)}/{len(items)} verifier-correct MATH traces -> {fp}")

def run_infer(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from rl_training.model_utils import merge_adapter_if_needed
    from rl_training.seq_recover import chat
    mp = merge_adapter_if_needed(a.model_path)
    tok = AutoTokenizer.from_pretrained(mp, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(mp, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="cuda")
    model.eval()
    items = load_math(a.n)

    # M1: teacher-forced NLL on (prompt, CORRECT reference solution). Mask the prompt; average over solution tokens.
    # --ref-traces: score a SHARED jsonl of {prompt, completion} (confound-fixed: self-generated correct MATH traces,
    # style-neutral) instead of human MATH-500 prose. This is the clean 'reusable computation' test.
    if a.ref_traces:
        pairs = [json.loads(l) for l in open(a.ref_traces) if l.strip()]
        src = [(r["prompt"], r["completion"]) for r in pairs]
        m1_tag = "M1_reftrace_nll"
    else:
        src = [(chat(mp, PROMPT.replace("{q}", it["q"])), it["sol"]) for it in items if it["sol"]]
        m1_tag = "M1_math_trace_nll"
    nlls = []
    for p, comp in src:
        if not comp: continue
        pids = tok(p, return_tensors="pt").input_ids.to("cuda")
        full = tok(p + comp, return_tensors="pt").input_ids.to("cuda")
        if full.shape[1] <= pids.shape[1] or full.shape[1] > 2048: continue
        with torch.no_grad():
            logits = model(full).logits[:, :-1, :]
        tgt = full[:, 1:]
        lp = torch.log_softmax(logits.float(), dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)[0]
        sol_lp = lp[pids.shape[1]-1:]  # only completion-token log-probs
        nlls.append(float(-sol_lp.mean().item()))
    m1 = sum(nlls)/max(len(nlls), 1)

    # M2/M3: greedy generation on MATH; token entropy proxy via output scores; length + step-line count.
    lens, steps, ents, corr = [], [], [], 0
    for it in items[:min(len(items), a.gen_n)]:
        p = chat(mp, PROMPT.replace("{q}", it["q"]))
        ids = tok(p, return_tensors="pt").input_ids.to("cuda")
        with torch.no_grad():
            go = model.generate(ids, max_new_tokens=512, do_sample=False, output_scores=True,
                                return_dict_in_generate=True, pad_token_id=tok.pad_token_id)
        gen = go.sequences[0, ids.shape[1]:]
        txt = tok.decode(gen, skip_special_tokens=True)
        lens.append(int(gen.shape[0]))
        steps.append(txt.count("\n") + txt.count("=") )  # explicit derivation-step proxy
        if go.scores:
            e = 0.0
            for s in go.scores:
                pr = torch.softmax(s[0].float(), dim=-1); e += float(-(pr*torch.log(pr+1e-12)).sum().item())
            ents.append(e/len(go.scores))
        corr += int(match(extract(txt), it["gold"]))
    out = {"tag": a.tag, "model": mp.split("/")[-1], "n_nll": len(nlls),
           m1_tag: m1,
           "M2_mean_token_entropy": sum(ents)/max(len(ents), 1),
           "M3_mean_gen_len": sum(lens)/max(len(lens), 1),
           "M3_mean_step_count": sum(steps)/max(len(steps), 1),
           "math_acc_greedy": corr/max(len(lens), 1)}
    Path(a.output_dir).mkdir(parents=True, exist_ok=True)
    json.dump(out, open(Path(a.output_dir)/f"mech_{a.tag}.json", "w"))
    print(f"[mech {a.tag}] M1_nll={m1:.4f} ({m1_tag}) M2_ent={out['M2_mean_token_entropy']:.3f} "
          f"M3_len={out['M3_mean_gen_len']:.0f} M3_steps={out['M3_mean_step_count']:.1f} acc={out['math_acc_greedy']:.3f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path"); ap.add_argument("--lora-delta")
    ap.add_argument("--tag", required=True); ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--gen-n", type=int, default=60); ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--ref-traces", default=""); ap.add_argument("--gen-traces", action="store_true")
    ap.add_argument("--output-dir", default="/tmp/instance_storage/gu/mech_out")
    a = ap.parse_args()
    if a.lora_delta: m4_lora_delta(a.lora_delta, a.tag, a.output_dir)
    elif a.gen_traces: gen_math_traces(a)
    else: run_infer(a)

if __name__ == "__main__":
    main()
