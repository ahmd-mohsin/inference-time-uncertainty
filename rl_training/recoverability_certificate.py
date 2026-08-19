#!/usr/bin/env python3
"""E3 — Mode-mass preservation *certificate* (Recoverability-Constrained RLVR pivot).

Jensen certificate (turns teacher-forced replay into a guarantee, not a regularizer):

  log p_θ(M|q) >= log p_0(M|q) + E_{y~ν_m}[ log π_θ(y|q) - log π_0(y|q) ]      (ν_m = base | y∈M,q)

so if the teacher-forced quantity   Δ_qm = E_{y~ν_m}[ log π_θ(y|q) - log π_0(y|q) ]  >= log α
then  p_θ(M|q) >= α · p_0(M|q)  (sufficient, one-sided).  Combined with E1's closed form this gives a
finite-budget recoverability guarantee:  R_K(M;θ) >= 1 - (1 - α·p_0(M|q))^K.

This module estimates Δ_qm from J witness traces per mode and reports a **lower confidence bound**
LCB(Δ_qm) (so the certificate is statistically calibrated, not a point estimate). It is INPUT-agnostic:
it consumes per-trace logprobs {log π_θ(y|q), log π_0(y|q)} that a GPU scoring pass produces
(teacher-forced forward pass over banked traces — cheap: no generation). Two entry points:

  score-template : emit a GPU script that teacher-forces a checkpoint over a bank and writes
                   {problem_id, mode_id, logp_theta, logp_ref} per trace.
  certify        : (CPU, runs now) read such a scored file (or the base bank's ref_logprob as a
                   self-check), compute Δ_qm + LCB per mode, and report the certified-mode fraction.

Self-check without GPU: with only base ref_logprob available, θ==base ⇒ Δ=0 ⇒ certified iff logα≤0
(α≤1), which must hold for every mode. `certify --selfcheck` verifies the estimator/LCB plumbing.
"""
import argparse, json, math, os
import numpy as np


def lcb_mean(xs, conf=0.95):
    """One-sided lower confidence bound on the mean via Student-t (small J)."""
    x = np.asarray(xs, dtype=float)
    n = len(x)
    if n == 0:
        return float("-inf")
    if n == 1:
        return float(x[0])  # no spread info; point value (flag low-confidence upstream)
    m = x.mean()
    s = x.std(ddof=1)
    try:
        from scipy.stats import t as tdist
        tcrit = tdist.ppf(conf, df=n - 1)
    except Exception:
        tcrit = 1.833 if n - 1 <= 9 else 1.645  # ~t_0.95 fallback
    return float(m - tcrit * s / math.sqrt(n))


def certify(records, log_alpha, conf, min_witness):
    """records: list of {problem_id, mode_id, logp_theta, logp_ref}. Group by (pid,mode)."""
    modes = {}
    for r in records:
        key = (r["problem_id"], r.get("mode_id", 0))
        modes.setdefault(key, []).append(r["logp_theta"] - r["logp_ref"])
    rows = []
    for (pid, mid), deltas in modes.items():
        d = np.asarray(deltas, dtype=float)
        row = dict(problem_id=pid, mode_id=mid, n_witness=len(d),
                   delta_mean=float(d.mean()),
                   delta_lcb=lcb_mean(d, conf),
                   low_confidence=len(d) < min_witness)
        row["certified"] = (row["delta_lcb"] >= log_alpha) and not row["low_confidence"]
        # implied guaranteed recoverability multiplier: p_theta(M) >= alpha_eff * p_0(M),
        # alpha_eff = exp(min(delta_lcb, 0)) capped at 1 (one-sided; gains above ref don't count)
        row["alpha_eff_lb"] = float(math.exp(min(row["delta_lcb"], 0.0)))
        rows.append(row)
    return rows


def do_certify(a):
    if a.selfcheck:
        # build synthetic records from the base bank: theta==base -> delta identically 0
        recs = []
        with open(a.bank) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                rl = d.get("ref_logprob")
                if rl is None:
                    continue
                recs.append(dict(problem_id=d.get("problem_id"), mode_id=0,
                                 logp_theta=rl, logp_ref=rl))
        rows = certify(recs, math.log(a.alpha), a.conf, a.min_witness)
        cert = sum(r["certified"] for r in rows)
        print(f"[selfcheck θ=base] modes={len(rows)} Δ≡0  certified(α={a.alpha})={cert}/{len(rows)} "
              f"(expect all, since logα={math.log(a.alpha):.3f} ≤ 0)")
        return
    recs = [json.loads(l) for l in open(a.scored) if l.strip()]
    rows = certify(recs, math.log(a.alpha), a.conf, a.min_witness)
    cert = sum(r["certified"] for r in rows)
    lowc = sum(r["low_confidence"] for r in rows)
    dmean = np.mean([r["delta_mean"] for r in rows]) if rows else 0.0
    print(f"=== E3 mode-mass certificate | scored={os.path.basename(a.scored)} | α={a.alpha} "
          f"(logα={math.log(a.alpha):.3f}) conf={a.conf} min_witness={a.min_witness} ===")
    print(f"modes={len(rows)} | mean Δ={dmean:+.2f} nats | "
          f"CERTIFIED (LCB≥logα) = {cert}/{len(rows)} ({100*cert/max(len(rows),1):.1f}%) | "
          f"low-confidence(<{a.min_witness} witnesses)={lowc}")
    # endangered modes = below floor (candidates for preservation pressure in E4 primal-dual)
    endangered = [r for r in rows if not r["certified"] and not r["low_confidence"]]
    endangered.sort(key=lambda r: r["delta_lcb"])
    print(f"endangered modes (LCB<logα): {len(endangered)}; worst 5:")
    for r in endangered[:5]:
        print(f"  q{r['problem_id']} m{r['mode_id']}: Δ_mean={r['delta_mean']:+.1f} "
              f"LCB={r['delta_lcb']:+.1f} (n={r['n_witness']})")
    if a.out:
        json.dump({"alpha": a.alpha, "conf": a.conf, "rows": rows,
                   "certified": cert, "n_modes": len(rows)}, open(a.out, "w"), indent=2)
        print(f"saved -> {a.out}")


SCORE_TEMPLATE = '''#!/usr/bin/env python3
# E3 GPU scorer — teacher-force a checkpoint over a (clustered) bank and write per-trace logprobs.
# NO generation (cheap forward pass). Usage:
#   python score_bank_logprobs.py --model <ckpt> --bank bank_modes.jsonl --out scored_<arm>.jsonl
# bank lines need: {problem_id, mode_id, prompt, completion}. Emits {problem_id,mode_id,logp_theta,logp_ref?}.
import argparse, json, os, sys, torch
os.environ.setdefault("HF_HUB_DISABLE_XET","1")
from transformers import AutoModelForCausalLM, AutoTokenizer
ap=argparse.ArgumentParser()
ap.add_argument("--model",required=True); ap.add_argument("--bank",required=True); ap.add_argument("--out",required=True)
ap.add_argument("--field",default="logp_theta")   # name for this model's column
a=ap.parse_args()
tok=AutoTokenizer.from_pretrained(a.model); model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.bfloat16,device_map="cuda")
model.eval()
def seq_logprob(prompt, completion):
    pids=tok(prompt, return_tensors="pt").input_ids.cuda()
    full=tok(prompt+completion, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        logits=model(full).logits[:, :-1, :].log_softmax(-1)
    tgt=full[:,1:]
    tok_lp=logits.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)[0]
    # sum logprob over completion tokens only
    start=pids.shape[1]-1
    return float(tok_lp[start:].sum().item())
with open(a.bank) as f, open(a.out,"w") as w:
    for line in f:
        line=line.strip()
        if not line: continue
        d=json.loads(line)
        lp=seq_logprob(d["prompt"], d["completion"])
        rec={"problem_id":d["problem_id"],"mode_id":d.get("mode_id",0), a.field:lp}
        if "ref_logprob" in d: rec["logp_ref"]=d["ref_logprob"]
        w.write(json.dumps(rec)+"\\n")
print("scored ->", a.out)
'''


def do_score_template(a):
    open(a.out, "w").write(SCORE_TEMPLATE)
    print(f"emitted GPU scorer -> {a.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("certify")
    c.add_argument("--scored", default="", help="jsonl with {problem_id,mode_id,logp_theta,logp_ref}")
    c.add_argument("--bank", default="", help="base bank (for --selfcheck)")
    c.add_argument("--selfcheck", action="store_true")
    c.add_argument("--alpha", type=float, default=0.5)
    c.add_argument("--conf", type=float, default=0.95)
    c.add_argument("--min-witness", type=int, default=8)
    c.add_argument("--out", default="")
    c.set_defaults(fn=do_certify)
    s = sub.add_parser("score-template"); s.add_argument("--out", default="rl_training/score_bank_logprobs.py")
    s.set_defaults(fn=do_score_template)
    a = ap.parse_args(); a.fn(a)


if __name__ == "__main__":
    main()
