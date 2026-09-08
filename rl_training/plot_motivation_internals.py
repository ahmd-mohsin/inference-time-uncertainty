#!/usr/bin/env python3
"""Motivation-section figures: how RLVR (Oat-Zero-7B) forgets base-recoverable solutions.

Reads the two teacher-forced internals npz (base = Qwen2.5-Math-7B, RL = sail Oat-Zero-7B) and produces
a 4-panel figure for the paper's motivation section:
  (a) trace-level log-prob collapse: base vs RL total logp (scatter, below-diagonal = suppressed)
  (b) suppression distribution: histogram of Delta = logp_RL - logp_base per trace
  (c) per-token collapse map: an example base-correct trace, base vs RL per-token logp (localized drop)
  (d) internal signatures per layer: hidden-state drift (1-cos) rises in top layers; attention entropy flat

Usage: python plot_motivation_internals.py --dir rl_training/runs_pulled/round2_eval/internals_oat \
         --base int_qmbase.npz --rl int_oat.npz --out fig_motivation
Outputs <out>.pdf and <out>.png.
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="rl_training/runs_pulled/round2_eval/internals_oat")
    ap.add_argument("--base", default="int_qmbase.npz")
    ap.add_argument("--rl", default="int_oat.npz")
    ap.add_argument("--rl-name", default="Oat-Zero-7B (RLVR)")
    ap.add_argument("--base-name", default="Qwen2.5-Math-7B (base)")
    ap.add_argument("--out", default="rl_training/runs_pulled/DELIVERABLES/report/fig_motivation")
    a = ap.parse_args()

    zb = np.load(os.path.join(a.dir, a.base), allow_pickle=True)
    zr = np.load(os.path.join(a.dir, a.rl), allow_pickle=True)
    n = min(len(zb["mean_logp"]), len(zr["mean_logp"]))
    ml_b, ml_r = zb["mean_logp"][:n], zr["mean_logp"][:n]
    nc_b, nc_r = zb["n_comp"][:n], zr["n_comp"][:n]
    tot_b, tot_r = ml_b * nc_b, ml_r * nc_r          # total trace logp
    dlt = tot_r - tot_b                               # suppression (negative = RL forgot)
    supp_frac = float((dlt < 0).mean())
    ae_b, ae_r = zb["attn_entropy"][:n], zr["attn_entropy"][:n]   # [n, L]
    hb, hr = zb["hidden"][:n], zr["hidden"][:n]                   # [n, L+1, D]
    cos = (hb * hr).sum(-1) / (np.linalg.norm(hb, axis=-1) * np.linalg.norm(hr, axis=-1) + 1e-9)
    drift = 1.0 - cos.mean(0)                          # per-layer mean drift [L+1]

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(2, 2, figsize=(11, 8.2))

    # (a) trace-level scatter
    lo = min(tot_b.min(), tot_r.min()); hi = max(tot_b.max(), tot_r.max())
    ax[0,0].scatter(tot_b, tot_r, s=14, alpha=0.5, color="#c0392b", edgecolor="none")
    ax[0,0].plot([lo,hi],[lo,hi], "k--", lw=1, label="no change")
    ax[0,0].set_xlabel(f"base log $\\pi$(trace)"); ax[0,0].set_ylabel(f"RLVR log $\\pi$(trace)")
    ax[0,0].set_title(f"(a) RLVR suppresses base-correct solutions\n"
                      f"{100*supp_frac:.0f}% below diagonal, mean $\\Delta$={dlt.mean():.0f} nats")
    ax[0,0].legend(loc="upper left", frameon=False)

    # (b) suppression histogram
    ax[0,1].hist(dlt, bins=30, color="#c0392b", alpha=0.8)
    ax[0,1].axvline(0, color="k", ls="--", lw=1)
    ax[0,1].axvline(dlt.mean(), color="#2c3e50", lw=2, label=f"mean {dlt.mean():.0f}")
    ax[0,1].set_xlabel("$\\Delta$ log $\\pi$(trace) = RLVR $-$ base (nats)")
    ax[0,1].set_ylabel("# base-correct traces")
    ax[0,1].set_title("(b) Distribution of forgetting\n(left of 0 = solution driven below base)")
    ax[0,1].legend(frameon=False)

    # (c) per-token collapse map for the most-collapsed trace
    j = int(np.argmin(dlt))
    cb = np.asarray(zb["comp_logp"][j], dtype=float)
    cr = np.asarray(zr["comp_logp"][j], dtype=float)
    m = min(len(cb), len(cr))
    cbc, crc = np.cumsum(cb[:m]), np.cumsum(cr[:m])   # cumulative logp along the trace
    x = np.arange(m)
    ax[1,0].plot(x, cbc, color="#2980b9", lw=1.6, label=a.base_name)
    ax[1,0].plot(x, crc, color="#c0392b", lw=1.6, label=a.rl_name)
    ax[1,0].fill_between(x, cbc, crc, where=(crc<cbc), color="#c0392b", alpha=0.12)
    ax[1,0].set_xlabel("completion token position")
    ax[1,0].set_ylabel("cumulative log $\\pi$")
    ax[1,0].set_title(f"(c) Where the collapse happens (example trace)\n"
                      f"gap opens at strategy tokens; final $\\Delta$={crc[-1]-cbc[-1]:.0f} nats")
    ax[1,0].legend(loc="lower left", frameon=False)

    # (d) per-layer internal signatures
    L = len(drift); xl = np.arange(L)
    ax2 = ax[1,1]
    ax2.plot(xl, drift, color="#8e44ad", lw=2, marker="o", ms=3, label="hidden-state drift (1$-$cos)")
    ax2.set_xlabel("layer"); ax2.set_ylabel("representation drift", color="#8e44ad")
    ax2.tick_params(axis="y", labelcolor="#8e44ad")
    axt = ax2.twinx(); axt.spines["top"].set_visible(False)
    dae = (ae_r.mean(0) - ae_b.mean(0))               # per-layer attn-entropy shift [L]
    axt.plot(np.arange(len(dae)), dae, color="#16a085", lw=1.5, ls="--", marker="s", ms=3,
             label="attn-entropy shift (RLVR$-$base)")
    axt.axhline(0, color="#16a085", lw=0.6, alpha=0.5)
    axt.set_ylabel("$\\Delta$ attention entropy", color="#16a085")
    axt.tick_params(axis="y", labelcolor="#16a085")
    ax2.set_title("(d) Internal signature: top-layer drift rises,\nattention entropy ~unchanged (collapse is in the output)")
    l1,lab1 = ax2.get_legend_handles_labels(); l2,lab2 = axt.get_legend_handles_labels()
    ax2.legend(l1+l2, lab1+lab2, loc="upper left", frameon=False, fontsize=9)

    fig.suptitle("RLVR drives base-recoverable solutions below the base model's probability "
                 f"({100*supp_frac:.0f}% of traces, mean {dlt.mean():.0f} nats)", fontsize=12, y=1.00)
    fig.tight_layout(rect=[0,0,1,0.98])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out + ".pdf", bbox_inches="tight"); fig.savefig(a.out + ".png", dpi=160, bbox_inches="tight")
    print(f"n_traces={n}  suppressed={100*supp_frac:.0f}%  meanΔ={dlt.mean():.1f} nats  "
          f"top-layer drift={drift[-1]:.3f}  attn-shift(mean)={dae.mean():+.4f}")
    print(f"saved -> {a.out}.pdf / .png")

if __name__ == "__main__":
    main()
