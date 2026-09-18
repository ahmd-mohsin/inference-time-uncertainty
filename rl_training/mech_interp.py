#!/usr/bin/env python3
"""Mechanistic-interpretability panel for RVP (base vs RVP checkpoint), on the model's own
verified-correct(y+)/verified-incorrect(y-) pairs. Three rigorous, forward-pass-only probes:

  (1) LAYER-WISE LOGIT-LENS MARGIN: apply the final norm + lm_head (logit lens) to EACH layer's
      residual stream, measure the mean per-token log-prob it assigns to the actual completion
      tokens -> per-layer "support" for y+ vs y-. margin_L = support(y+)_L - support(y-)_L.
      Shows the DEPTH at which the correct-vs-incorrect margin is built, base vs RVP.
  (2) DECISIVENESS (entropy): mean final-layer predictive entropy over completion positions.
      RVP's selection story predicts LOWER entropy (sharper next-token distribution).
  (3) HIDDEN-STATE DRIFT: per-layer mean cosine distance between RVP and base residual streams,
      localising WHERE in the network RVP moved mass.

Usage: python3 -m rl_training.mech_interp --base <hf> --rvp <dir> --data pairs.jsonl --n 80 --out mech.json
Memory-safe: batch=1, layer-by-layer lens (gather target logprob only; full softmax only for entropy).
"""
import argparse, json, os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load(m):
    tok = AutoTokenizer.from_pretrained(m); tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(m, torch_dtype=torch.bfloat16, device_map="cuda",
                                                 output_hidden_states=True).eval()
    return tok, model

def final_norm(model):
    # locate the final RMSNorm/LayerNorm before lm_head (Qwen/Llama: model.model.norm)
    for attr in ("norm", "final_layernorm", "ln_f"):
        n = getattr(getattr(model, "model", model), attr, None)
        if n is not None: return n
    return torch.nn.Identity()

@torch.no_grad()
def probe(tok, model, prompt, completion, maxlen):
    norm = final_norm(model); lm = model.get_output_embeddings()
    pids = tok(prompt, return_tensors="pt", add_special_tokens=True).input_ids
    fids = tok(prompt + completion, return_tensors="pt", add_special_tokens=True).input_ids[:, :maxlen].to("cuda")
    cs = min(pids.shape[1] - 1, fids.shape[1] - 2)  # completion tokens start here (predict-next indexing)
    if fids.shape[1] - 1 <= cs: return None
    out = model(fids)
    hs = out.hidden_states  # tuple len L+1 ([emb], layer1..layerL); each [1,T,H]
    tgt = fids[0, cs + 1:]                       # actual completion token ids to predict
    per_layer = []
    for h in hs[1:]:                             # skip embedding layer
        logits = lm(norm(h[0, cs:-1])).float()   # [Tc, V] logit-lens (keep model dtype into matmul, float logits)
        lp = torch.log_softmax(logits, -1)
        per_layer.append(lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).mean().item())
    # decisiveness: entropy of the FINAL-layer next-token dist over completion positions
    flog = torch.log_softmax(lm(norm(hs[-1][0, cs:-1])).float(), -1)
    ent = (-(flog.exp() * flog).sum(-1)).mean().item()
    return per_layer, ent, [h[0, cs:-1].float().mean(0) for h in hs[1:]]  # + per-layer mean resid (for drift)

def run(model_tag, tok, model, pairs, maxlen):
    L = model.config.num_hidden_layers
    mpos = [0.0]*L; mneg = [0.0]*L; epos = eneg = 0.0; nn = 0; resid_pos = None
    for r in pairs:
        a = probe(tok, model, r["prompt"], r["chosen"], maxlen)
        b = probe(tok, model, r["prompt"], r["rejected"], maxlen)
        if a is None or b is None: continue
        for i in range(L): mpos[i] += a[0][i]; mneg[i] += b[0][i]
        epos += a[1]; eneg += b[1]; nn += 1
        rp = a[2]
        resid_pos = rp if resid_pos is None else [resid_pos[i] + rp[i] for i in range(L)]
    if nn == 0: return None
    return {"model": model_tag, "n": nn, "L": L,
            "lens_margin_by_layer": [(mpos[i]-mneg[i])/nn for i in range(L)],
            "lens_pos_by_layer": [mpos[i]/nn for i in range(L)],
            "lens_neg_by_layer": [mneg[i]/nn for i in range(L)],
            "entropy_pos": epos/nn, "entropy_neg": eneg/nn,
            "_resid_pos": [(resid_pos[i]/nn) for i in range(L)]}  # mean y+ resid per layer (for drift)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--rvp", required=True)
    ap.add_argument("--data", required=True); ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--out", required=True); ap.add_argument("--maxlen", type=int, default=640)
    a = ap.parse_args()
    pairs = [json.loads(l) for l in open(a.data)][:a.n]
    res = {}
    for tag, m in (("base", a.base), ("rvp", a.rvp)):
        tok, model = load(m); r = run(tag, tok, model, pairs, a.maxlen)
        res[tag] = r
        del model; torch.cuda.empty_cache()
    # hidden-state drift: cosine distance between RVP and base mean y+ resid, per layer
    if res.get("base") and res.get("rvp"):
        bp, rp = res["base"].pop("_resid_pos"), res["rvp"].pop("_resid_pos")
        drift = []
        for i in range(min(len(bp), len(rp))):
            u, v = bp[i], rp[i]
            drift.append(1.0 - float(torch.nn.functional.cosine_similarity(u.unsqueeze(0), v.unsqueeze(0)).item()))
        res["resid_drift_by_layer"] = drift
    else:
        for k in ("base","rvp"):
            if res.get(k): res[k].pop("_resid_pos", None)
    json.dump(res, open(a.out, "w"), indent=2)
    b, r = res.get("base"), res.get("rvp")
    if b and r:
        print(f"[mech] base margin final={b['lens_margin_by_layer'][-1]:.4f} entropy={b['entropy_pos']:.4f} | "
              f"rvp margin final={r['lens_margin_by_layer'][-1]:.4f} entropy={r['entropy_pos']:.4f}")
        print(f"[mech] layer of max margin gain: {max(range(r['L']), key=lambda i: r['lens_margin_by_layer'][i]-b['lens_margin_by_layer'][i])}")
if __name__ == "__main__": main()
