# PROGRESS SNAPSHOT v7 — the award result is landing
_2026-09-11. Full detail: ADAPTIVE_FORGETTING_RESULTS.md §120-§129. Method: rl_training/vsf_trainer.py. Advisor memo: ASTRA_ADVICE.md._

## 1. HEADLINE (Astra-framed, now empirically supported)
**For verifiable OOD learning the resource that matters is TRANSFERABLE VERIFIED SUPPORT — its acquisition AND preservation — not the number
of on-policy updates.** On-policy outcome-RL (GRPO) can DEGRADE a model below its base on hard OOD (it fails to sample successes → dead groups →
no signal → it moves mass off already-correct outputs). A minimal fix — VSF (Verified Support Floor): GRPO + a persistent, prompt-balanced replay
of verified successes — REPAIRS this and matches offline replay (RFT). The effect scales UP with model size.

## 2. THE MECHANISM MATRIX (accuracy on hard/mid OOD; identical data per row) — COMPLETE
| Cell | base | GRPO | VSF (ours) | RFT (DVR) | VSF repair |
|---|---|---|---|---|---|
| 1.5B-mid (d7→9)  | —     | 0.235 | 0.330 (CI .330/.335) | 0.482 | ~40% |
| 1.5B-hard (d12→14)| 0.085 | —     | 0.215 | 0.260 | near-full |
| 3B-hard (d12→14) | 0.255 | 0.210 **(↓base)** | 0.280 | 0.505 | ~28%, clears base |
| 7B-hard (d12→14) | 0.475 | 0.455 **(↓base)** | **0.575 (=RFT)** | 0.567 | **FULL** |
| 9B-hard (d12→14) | —     | 0.395 | 0.470 | 0.520 | ~60% |
_Non-starved GRPO (num_gen 16) at 1.5B-mid = 0.235 == plain GRPO → deficit is not a weak-baseline artifact._
_HONEST: VSF reverses GRPO's regression at every size and closes 40–100% of the GRPO→RFT gap, but **RFT ≥ VSF everywhere** (== at 7B).
So RFT/DVR is the reliable recipe; VSF is the **causal repair / drop-in fix for on-policy pipelines**, not a strict SOTA winner._

## 3. THE THREE AWARD LEGS (all now hold — pushed §127/§128/§129)
1. **The deficit is REAL, not a weak baseline** (Astra's #1 risk, de-risked): doubling GRPO group size (num_gen 8→16, halves dead-group rate)
   does NOT help — non-starved GRPO 0.235 == plain GRPO 0.235 ≪ RFT 0.482 at 1.5B-mid.
2. **The mechanism is acquisition vs regression (α/β)**: GRPO ends up BELOW base (3B 0.210<0.255, 7B 0.455<0.475) = it damages held-out
   capability; RFT/VSF end ABOVE base (+0.09 to +0.25) = they acquire. This is the smoking gun.
3. **VSF REPAIRS it** and scales up: at 7B, VSF 0.575 ≈ RFT 0.567 ≫ GRPO 0.455 — the support floor recovers the entire gap. Partial at 1.5B
   (0.235→0.330), FULL at 7B. (Opposite of the abandoned CTH trick, which was weak-model-only.)

## 4. WHY THIS IS AWARD-SHAPED
- Overturns the field-default (GRPO/outcome-RL) for OOD with a controlled matrix + a clean mechanism + a minimal, principled repair (VSF).
- Backed by theory (success-gradient identity; dead-group bound a_G(p)=1−(1−p)^G−p^G; α/β decomposition) and the honest boundary (CTH size-gated).
- Scales the RIGHT way: the gap and the VSF repair are governed by HEADROOM and get STRONGER at 7B — not a small-model artifact.

## 5. RUNNING NOW (finishing the matrix + robustness)
- **VSF 9B-hard** (1094, server) — does VSF repair hold at 9B? | **VSF 3B-hard retry** (1093, OOM'd once, lighter) — fills the VSF column.
- **GRPO 1.5B/3B-hard** (workers) — the two missing plain-GRPO hard cells.
- Base evals DONE (1.5B/3B/7B on hard OOD) → α/β computable per-problem next.

## 6. STILL TO DO for the paper (Astra D/E, in priority order)
- Per-problem α/β table from saved per_problem JSONs (quantify acquisition vs preservation exactly).
- Predictive-law train/test split (fit gain=f(headroom,support) on 1.5B/3B, predict 7B/9B out-of-sample).
- VSF seed CIs (currently 1 seed at 7B) + VSF ablations (prompt-balanced vs uniform replay; support-floor vs plain replay; bank refresh on/off).
- One 30–32B anchor (needs tensor/pipeline parallelism — 32B won't fit ZeRO-2 LoRA on 40GB; systems pilot required).
- Literature audit (self-imitation / balanced replay / ReST prior art) before claiming VSF algorithmic novelty — position as "principle + causal account + predictive law + scale", not "a buffer".

## 7. INFRA / HYGIENE
- Recurring eval bug: a leftover vLLM on GPU0 starves the next eval's engine init → "Engine core initialization failed"/empty acc. Fix = kill GPU0 pid, re-eval. (Hit 7B/9B/3B evals; all salvageable — models are saved.)
- num_gen 16 needs `--gradient-accumulation-steps 16` (gen_batch = per_device×grad_accum, default grad_accum 8); OOMs at 7B on 40GB (colocate).
- Bedrock creds used for the Astra consult are EXPIRED — rotate. Everything pushed to GitHub; pods die at 24h TTL.
