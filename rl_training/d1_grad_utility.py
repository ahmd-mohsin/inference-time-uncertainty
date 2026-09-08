# DIRECTION 1 GATE — gradient-utility diagnostic (reviewer's decisive first experiment).
# Question: does per-constraint learning-utility u_j = <g_all^probe, g_j> predict which update raises FULL
# correctness — better than pass-frequency? If not, STOP the method.
#   g_i      = REINFORCE gradient of log pi(a_i|s) wrt LoRA params (per sampled repair a_i)   [stored per sample]
#   g_all    = sum_i (R_all,i - b) g_i           (full-correctness gradient; R_all = 1[all tests pass])
#   g_j      = sum_i (v_{j,i} - b_j) g_i          (per-constraint j gradient; v_j = 1[test j passes])
#   u_j      = <g_all^PROBE, g_j^TRAIN>           (probe batch disjoint from the batch defining g_j)
#   f_j      = mean_i v_{j,i}                      (pass frequency of constraint j)
# Falsifiable checks: (1) some HIGH-f constraints have <=0 utility; (2) equal-f constraints differ in utility;
# (3) BRANCH PREDICTION — a one-step update along the high-utility direction raises fresh full-correctness MORE
# than random or high-pass-frequency directions. Single GPU, LoRA-wrapped. Grads stored per-sample on CPU.
# Usage: python -m rl_training.d1_grad_utility --model-path Qwen/Qwen2.5-Coder-7B-Instruct \
#   --data /tmp/instance_storage/gu/repair_data/repair_rep_qc.jsonl --tag d1_qc \
#   --n-probe 8 --n-train 14 --n-eval 24 --k 6 --max-tests 8
import argparse, json, os, sys, random
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from rl_training.code_passk import extract_code, run_tests
from rl_training.rewards import _passvec

def load_states(data, tok):
    recs = [json.loads(l) for l in open(data)]
    # each rec: prompt (chat string), test, entry, mbpp, pfail
    return recs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--data", required=True)
    ap.add_argument("--n-probe", type=int, default=8); ap.add_argument("--n-train", type=int, default=14)
    ap.add_argument("--n-eval", type=int, default=24); ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--max-tests", type=int, default=8); ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", default="/tmp/instance_storage/gu/eval_out"); ap.add_argument("--tag", default="d1")
    a = ap.parse_args()
    random.seed(a.seed); torch.manual_seed(a.seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    tok = AutoTokenizer.from_pretrained(a.model_path, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
                                                 attn_implementation="eager").to("cuda")
    lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=["q_proj","k_proj","v_proj","o_proj"], task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    model.eval()  # LoRA params still have grad; base frozen
    tparams = [p for p in model.parameters() if p.requires_grad]
    ndim = sum(p.numel() for p in tparams)
    print(f"[d1] trainable dim={ndim/1e6:.1f}M")

    recs = [json.loads(l) for l in open(a.data)]
    random.shuffle(recs)
    probe = recs[:a.n_probe]; train = recs[a.n_probe:a.n_probe+a.n_train]; ev = recs[a.n_probe+a.n_train:a.n_probe+a.n_train+a.n_eval]

    @torch.no_grad()
    def sample(prompt, k):
        ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
        out = model.generate(ids, do_sample=True, temperature=1.0, top_p=0.95, max_new_tokens=768,
                             num_return_sequences=k, pad_token_id=tok.pad_token_id)
        comps = [o[ids.shape[1]:] for o in out]
        return ids[0], comps

    def logp_sum(prompt_ids, comp_ids):  # differentiable scalar sum log pi(comp|prompt)
        ids = torch.cat([prompt_ids, comp_ids])[None].to("cuda")
        logits = model(ids).logits[0][:-1]
        targets = ids[0, 1:]
        lp = torch.log_softmax(logits.float(), -1)[range(targets.shape[0]), targets]
        L = prompt_ids.shape[0]
        return lp[L-1:L-1+comp_ids.shape[0]].sum()

    def grad_vec(scalar_loss):
        for p in tparams:
            if p.grad is not None: p.grad = None
        scalar_loss.backward()
        return torch.cat([(p.grad.detach().flatten().float().cpu() if p.grad is not None
                           else torch.zeros(p.numel())) for p in tparams])

    def collect(problems):
        """For each problem: sample k, exec-verify -> R_all + per-test v; store per-sample grad g_i (CPU)."""
        out = []
        for r in problems:
            it = {"test": r["test"], "entry": r.get("entry"), "mbpp": r.get("mbpp"), "id": r.get("problem_id")}
            prompt_ids, comps = sample(r["prompt"], a.k)
            gis = []; Ralls = []; vmat = []
            for c in comps:
                text = tok.decode(c, skip_special_tokens=True)
                vec = _passvec(extract_code(text), r["test"], r.get("entry"), bool(r.get("mbpp")))
                if not vec: continue
                gi = grad_vec(-logp_sum(prompt_ids, c))  # d(-logp)/dθ ; g_i = -this
                gis.append(-gi); Ralls.append(1.0 if all(vec) else 0.0); vmat.append(vec)
            if len(gis) < 2: continue
            m = min(min(len(v) for v in vmat), a.max_tests)
            vmat = [v[:m] for v in vmat]
            out.append({"gis": gis, "Rall": Ralls, "vmat": vmat, "m": m, "id": r.get("problem_id")})
        return out

    def gall(coll):
        G = torch.zeros(ndim)
        for pb in coll:
            b = sum(pb["Rall"])/len(pb["Rall"])
            for gi, R in zip(pb["gis"], pb["Rall"]): G += (R-b)*gi
        return G

    print("[d1] collecting probe grads..."); Cp = collect(probe); Gp = gall(Cp)
    print("[d1] collecting train grads..."); Ct = collect(train)
    # per-constraint utility
    rows = []
    for pb in Ct:
        for j in range(pb["m"]):
            fj = sum(v[j] for v in pb["vmat"])/len(pb["vmat"])
            bj = fj
            gj = torch.zeros(ndim)
            for gi, v in zip(pb["gis"], pb["vmat"]): gj += ((1.0 if v[j] else 0.0)-bj)*gi
            uj = float(torch.dot(Gp, gj))
            rows.append({"f": fj, "u": uj, "id": pb["id"], "j": j, "gj": gj})
    import numpy as np
    f = np.array([r["f"] for r in rows]); u = np.array([r["u"] for r in rows])
    # checks 1-2
    corr = float(np.corrcoef(f, u)[0,1]) if len(f) > 2 else 0.0
    highf = [(r["f"], r["u"]) for r in rows if r["f"] >= 0.8]
    highf_lowu = sum(1 for _, uu in highf if uu <= 0)
    # spread at fixed frequency (bin f, measure utility std within bins)
    spreads = []
    for lo in (0.0, 0.34, 0.67):
        us = [r["u"] for r in rows if lo <= r["f"] < lo+0.33]
        if len(us) >= 3: spreads.append((lo, float(np.std(us)), float(np.mean(us))))

    # BRANCH PREDICTION: 3 directions from train constraints; one-step update; Δ fresh full-correctness
    def direction(weights):
        d = torch.zeros(ndim)
        for r, w in zip(rows, weights): d += w*r["gj"]
        n = d.norm(); return d/n if n > 0 else d
    d_util = direction([1.0 if r["u"] > 0 else 0.0 for r in rows])
    rng = np.random.default_rng(a.seed); d_rand = direction(list(rng.choice([0.0,1.0], size=len(rows))))
    d_freq = direction([r["f"] for r in rows])

    @torch.no_grad()
    def eval_fullcorrect(problems):
        n = tot = 0
        for r in problems:
            prompt_ids, comps = sample(r["prompt"], a.k)
            for c in comps:
                vec = _passvec(extract_code(tok.decode(c, skip_special_tokens=True)), r["test"], r.get("entry"), bool(r.get("mbpp")))
                if not vec: continue
                n += int(all(vec)); tot += 1
        return n/max(tot,1)

    base_fc = eval_fullcorrect(ev)
    def apply_and_eval(direc):
        # theta += lr*direc ; eval ; revert
        i = 0; saved = []
        with torch.no_grad():
            for p in tparams:
                num = p.numel(); saved.append(p.detach().clone())
                p.add_((a.lr*direc[i:i+num]).to(p.device, p.dtype).view_as(p)); i += num
        fc = eval_fullcorrect(ev)
        with torch.no_grad():
            for p, s in zip(tparams, saved): p.copy_(s)
        return fc
    fc_util = apply_and_eval(d_util); fc_rand = apply_and_eval(d_rand); fc_freq = apply_and_eval(d_freq)

    res = {"tag": a.tag, "n_constraints": len(rows), "corr_f_u": corr, "highf_count": len(highf),
           "highf_lowORneg_u": highf_lowu, "util_spread_at_fixed_f": spreads,
           "base_fullcorrect": base_fc, "fc_high_utility": fc_util, "fc_random": fc_rand, "fc_high_passfreq": fc_freq,
           "delta_util": fc_util-base_fc, "delta_rand": fc_rand-base_fc, "delta_freq": fc_freq-base_fc}
    Path(a.output_dir).mkdir(parents=True, exist_ok=True)
    json.dump(res, open(Path(a.output_dir)/f"d1_{a.tag}.json", "w"), indent=2)
    print(json.dumps(res, indent=2))
    verdict = "PASS (utility predicts; >random and >passfreq)" if (fc_util > fc_rand and fc_util >= fc_freq and fc_util > base_fc) else "FAIL/INCONCLUSIVE — do NOT build the controller"
    print(f"[d1 GATE] {verdict}")

if __name__ == "__main__":
    main()
