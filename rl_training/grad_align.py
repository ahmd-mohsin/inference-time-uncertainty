# Gradient-alignment probe for the rho-down/c-up mechanism. Theory: Delta c_m ~= eta * sum_j rho_j
# <grad c_m, grad c_j>, so competence can RISE for a strategy whose routing FALLS, via shared-gradient
# transfer between strategies. We estimate strategy-conditioned gradients and their cosine-alignment.
#
# g_m = gradient (w.r.t. the last N decoder layers) of the mean NLL of the strategy-m prefix seed given
# the problem, averaged over K problems. This is the direction that increases entry+execution of
# strategy m. G_ij = cos(g_i, g_j); T_m = sum_j rho_j G_mj (rho uniform here, or supplied). High
# positive off-diagonal alignment => strategies reinforce each other => benign collapse (c survives).
#
# Single-GPU HF (last-N-layer grads only, bf16). Usage:
#   python -m rl_training.grad_align --model-path <dir> --dataset math500 --tag ga_qm_base --k 60 --last-layers 2
import argparse, json, os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.dataset import get_inference_dataset, format_prompt
from rl_training.strategy_probe import STRAT_NAMES, STRATEGY_PREFIX


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path"); ap.add_argument("--dataset", default="math500")
    ap.add_argument("--k", type=int, default=60); ap.add_argument("--last-layers", type=int, default=2)
    ap.add_argument("--output-dir", default="/tmp/instance_storage/gu/eval_out"); ap.add_argument("--tag", default="ga")
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from rl_training.model_utils import merge_adapter_if_needed
    mp = merge_adapter_if_needed(a.model_path)
    tok = AutoTokenizer.from_pretrained(mp)
    model = AutoModelForCausalLM.from_pretrained(mp, torch_dtype=torch.bfloat16, device_map="cuda")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()  # cut activation memory (was OOMing on full-seq backward)
    model.eval()
    # select last-N decoder layers' params as the gradient subspace
    layers = model.model.layers
    keep = set(id(p) for l in layers[-a.last_layers:] for p in l.parameters())
    for p in model.parameters(): p.requires_grad_(id(p) in keep)
    gparams = [p for p in model.parameters() if p.requires_grad]
    probs = get_inference_dataset({"dataset": {"name": a.dataset, "split": "test", "n_problems": -1, "seed": 42}})[:a.k]

    def strat_grad(m):
        seed = STRATEGY_PREFIX[m]
        model.zero_grad(set_to_none=True); tot = 0.0; nb = 0
        for p in probs:
            prompt = format_prompt(p, mp)
            full = tok(prompt + seed, return_tensors="pt").input_ids.cuda()
            plen = tok(prompt, return_tensors="pt").input_ids.shape[1]
            out = model(full)
            logits = out.logits[:, :-1, :]; tgt = full[:, 1:]
            lp = torch.log_softmax(logits.float(), -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)[0]
            loss = -lp[plen-1:].mean()          # NLL of the seed continuation
            (loss / len(probs)).backward()
            tot += loss.item(); nb += 1
        g = torch.cat([p.grad.detach().flatten().float().cpu() for p in gparams])  # keep on CPU (avoid GPU OOM across 14 strategies)
        return g / (g.norm() + 1e-8), tot / max(nb, 1)

    gs = {}; nll = {}
    for m in STRAT_NAMES:
        gs[m], nll[m] = strat_grad(m)
        print(f"  {m}: nll={nll[m]:.3f}")
    import numpy as np
    G = {i: {j: float(torch.dot(gs[i], gs[j]).item()) for j in STRAT_NAMES} for i in STRAT_NAMES}
    # T_m under uniform routing = mean_j G_mj ; off-diagonal alignment
    T = {m: sum(G[m][j] for j in STRAT_NAMES if j != m) / (len(STRAT_NAMES)-1) for m in STRAT_NAMES}
    offdiag = [G[i][j] for i in STRAT_NAMES for j in STRAT_NAMES if i != j]
    out = {"tag": a.tag, "model": mp, "dataset": a.dataset, "k": len(probs), "last_layers": a.last_layers,
           "strategies": STRAT_NAMES, "G": G, "T_uniform": T, "nll": nll,
           "mean_offdiag_cos": sum(offdiag)/len(offdiag), "min_offdiag": min(offdiag), "max_offdiag": max(offdiag)}
    os.makedirs(a.output_dir, exist_ok=True)
    fp = os.path.join(a.output_dir, f"ga_{a.tag}.json"); json.dump(out, open(fp, "w"), indent=2)
    print(f"[{a.tag}] mean off-diagonal strategy-gradient cosine = {out['mean_offdiag_cos']:+.3f} "
          f"(range {out['min_offdiag']:+.3f}..{out['max_offdiag']:+.3f})")
    print(f"  (positive => strategies share gradient direction => training one helps others => benign collapse)")
    print(f"saved -> {fp}")


if __name__ == "__main__":
    main()
