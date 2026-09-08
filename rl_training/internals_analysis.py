#!/usr/bin/env python3
"""Mechanistic 'why the model forgets' analysis — internal signatures of coverage collapse.

Teacher-forces a set of BASE-CORRECT traces through a model and records, per trace:
  1. per-token log-prob of the completion  -> logp collapse map (localize WHERE mass drops)
  2. per-layer attention entropy (mean over heads, query positions) -> attention narrowing
  3. per-layer mean-pooled hidden state (completion tokens) -> representation drift vs base

Run once per model/checkpoint (base + RL checkpoint-10/30/.../100). A separate plotting step
diffs each RL output against the base output to build the 4 figures:
  - per-token Delta(logp) along the trace (fork-token localization)
  - attention-entropy shift (base vs RL, per layer)
  - forgetting-over-training curve (mean Delta logp vs checkpoint step)
  - hidden-state drift (cosine distance base<->RL, per layer, lost vs retained)

Usage:
  python internals_analysis.py --model <path|hf> --traces traces.jsonl --out out.npz \
      [--max-traces 150] [--tag base|ckpt-30|...]
traces.jsonl rows: {"problem_id":int, "prompt":str, "completion":str, "retained":bool(optional)}
"""
import argparse, json, os, sys
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--max-traces", type=int, default=150)
    ap.add_argument("--max-len", type=int, default=3072)
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="eager",  # eager needed to return attentions
        output_attentions=True, output_hidden_states=True)
    model.eval()

    rows = []
    for i, line in enumerate(open(a.traces)):
        line = line.strip()
        if not line: continue
        rows.append(json.loads(line))
        if len(rows) >= a.max_traces: break

    def pick(r, keys):
        for k in keys:
            if k in r and r[k]: return r[k]
        return None
    per_trace = []
    for r in rows:
        prompt = pick(r, ["prompt", "question", "problem"])
        completion = pick(r, ["completion", "solution", "text", "response"])
        if prompt is None or completion is None:
            continue
        # tokenize prompt and full separately to find the completion span
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        c_ids = tok(completion, add_special_tokens=False)["input_ids"]
        ids = (p_ids + c_ids)[: a.max_len]
        n_p = min(len(p_ids), a.max_len)
        if len(ids) <= n_p + 1:  # no completion tokens left
            continue
        inp = torch.tensor([ids], device="cuda")
        with torch.no_grad():
            out = model(inp, output_attentions=True, output_hidden_states=True)
        logits = out.logits[0].float()                       # [T, V]
        logp = torch.log_softmax(logits[:-1], dim=-1)        # predict token t+1 from t
        tgt = inp[0, 1:]                                      # [T-1]
        tok_logp = logp.gather(1, tgt[:, None]).squeeze(1)   # [T-1]
        comp_logp = tok_logp[n_p - 1:].cpu().numpy()         # completion-token logps

        # attention entropy per layer: mean over heads and query positions (completion region)
        attn = out.attentions                                # tuple[L] each [1,H,T,T]
        L = len(attn); ent = np.zeros(L, dtype=np.float32)
        for li, at in enumerate(attn):
            aw = at[0, :, n_p:, :].float()                   # [H, Tc, T]
            aw = aw.clamp_min(1e-12)
            e = -(aw * aw.log()).sum(-1)                     # [H, Tc] entropy per query
            ent[li] = e.mean().item()
        # per-layer mean-pooled hidden state over completion tokens
        hs = out.hidden_states                               # tuple[L+1] each [1,T,D]
        pooled = np.stack([h[0, n_p:, :].float().mean(0).cpu().numpy() for h in hs])  # [L+1, D]

        per_trace.append(dict(
            problem_id=r.get("problem_id", -1),
            retained=bool(r.get("retained", False)),
            mean_logp=float(comp_logp.mean()),
            comp_logp=comp_logp.astype(np.float32),
            attn_entropy=ent,
            hidden=pooled.astype(np.float32),
            n_comp=int(len(comp_logp)),
        ))
    # save (object arrays for ragged comp_logp)
    np.savez_compressed(
        a.out,
        tag=a.tag,
        problem_id=np.array([t["problem_id"] for t in per_trace]),
        retained=np.array([t["retained"] for t in per_trace]),
        mean_logp=np.array([t["mean_logp"] for t in per_trace], dtype=np.float32),
        attn_entropy=np.stack([t["attn_entropy"] for t in per_trace]) if per_trace else np.zeros((0,)),
        hidden=np.stack([t["hidden"] for t in per_trace]) if per_trace else np.zeros((0,)),
        comp_logp=np.array([t["comp_logp"] for t in per_trace], dtype=object),
        n_comp=np.array([t["n_comp"] for t in per_trace]),
    )
    print(f"[internals] {a.tag or a.model}: {len(per_trace)} traces -> {a.out} "
          f"(mean_logp={np.mean([t['mean_logp'] for t in per_trace]):.2f})")

if __name__ == "__main__":
    main()
