# RVP — Next steps to award-caliber (dense, prioritized)

Assessment of what we have, what a top-tier (NeurIPS best-paper-caliber) committee will demand next,
and exactly how to deliver it on the 72-GPU fleet. Priorities: **P0 = must-have to survive review**,
**P1 = elevates to award contention**, **P2 = polish / stretch**. Each item states the *reviewer
objection it closes*, the *method*, the *cost*, and the *risk*.

---

## Where we stand (strengths already banked)
Clear thesis (reliability via **selection**, headroom-gated) · 5-theorem theory · dense scale×family×domain
validation grid · GRPO/RFT/RVP ladder · mechanism (margin trajectory + late-layer logit-lens) ·
final-24 3×6×4 generalization matrix (72 cells) · controls (shuf/xrft) · ablations (β, data-efficiency,
one-shot, hard-neg, RVP-from-base) · honest scope boundaries (general-base mixed, RLHF-chat collapse,
Mathstral). This is already a strong paper. The gap to *award* is **baseline breadth, theory→practice
tightness, external-domain validity, and writing**.

---

## P0 — must-have (close the obvious reviewer objections)

### P0.1 The "why not best-of-n / self-consistency?" head-to-head  ← THE killer experiment
- **Objection closed:** "You're just amortizing inference-time selection (best-of-n, self-consistency,
  verifier reranking) into the weights — show it head-to-head at *matched inference cost*."
- **Method:** on the winning cells (Qwen2.5-Math-1.5B + 7B, MATH-500/Olympiad/Omni/GSM8K), plot a
  **cost–reliability frontier**: x = samples/inference-cost `n`, y = pass@1-equivalent accuracy, for
  (a) base + majority-vote@n (self-consistency), (b) base + best-of-n under the *same* verifier used
  for banking, (c) **RVP at n=1**. Claim to establish: **RVP@1 matches or beats base@best-of-n for
  n≈4–8** — i.e. RVP bakes the selection gain into a single forward pass. Also report RVP@n (RVP + SC)
  to show they compose.
- **How:** reuse `math_rvp --mode eval` (already samples k=16 → majority-vote & best-of-n are free
  post-hoc from `ev_*.json` per-problem records); add a `--mode selectbench` that reads the saved
  per-problem samples and computes maj@n / bestof@n / pass@1 curves. New analysis only, **no retrain**.
- **Cost:** ~0 GPU (post-hoc on existing eval JSONs) + a small script. **Risk:** low. **Do first.**

### P0.2 Empirically validate Prop 2: p(success) = σ(m + c)  ← the theory→practice money figure
- **Objection closed:** "Your central proposition (pass@1 is σ of the margin) is asserted, not shown."
- **Method:** per problem, compute the RVP model's teacher-forced margin `m` and its empirical success
  rate `p̂ = c/k`; scatter `p̂` vs `m` across all problems/cells and fit `σ(am+b)`. A tight logistic
  fit *across cells* is a single, decisive figure that ties mechanism → outcome.
- **How:** extend `mech_interp.py` (already computes per-layer margins) to dump per-problem final-layer
  `m` alongside the eval's per-problem `p̂`; join + fit in a plotting script.
- **Cost:** ~1 GPU-hr per cell (teacher-forcing pass) on 3–4 cells. **Risk:** low; even a moderate fit
  is publishable ("monotone, well-calibrated").

### P0.3 Strong same-family baselines beyond GRPO
- **Objection closed:** "RFT and GRPO aren't the only post-training baselines."
- **Method:** add, on 2–3 headroom cells, **(a) iterative RFT / ReST-style** (2–3 rounds of
  reject-sample→SFT — the natural "just do more RFT" competitor; we predict it plateaus while RVP adds
  the margin), **(b) plain DPO on generic (non-verified) preference pairs** (isolates "verified" vs
  "any preference"), **(c) a trained verifier/reward-model reranker** at matched inference cost (ties
  to P0.1). shuf already covers randomized-sign; these cover the *realistic* alternatives.
- **How:** iterative RFT = loop `sft_train`; generic-DPO = `dpo_train` on unverified pairs; reranker =
  small classifier head or LM-as-verifier over the k=16 samples.
- **Cost:** ~3–4 cells × sharded/1-GPU. **Risk:** medium (iterative-RFT is a few runs). **Highest-value
  P0 after P0.1.**

---

## P1 — elevates to award contention

### P1.1 A real external code domain (MBPP / HumanEval+ / BigCodeBench)
- **Objection closed:** the paper claims code+math but the strong new evidence is all math; "does the
  selection principle transfer to a *different* verifiable domain with an external, standard benchmark?"
- **Method:** run the full RFT→RVP pipeline on a code base (Qwen2.5-Coder-1.5B/7B) with unit-test
  verification; report pass@1 on MBPP + HumanEval(+). We already have `code_gen*.py`, `code_passk.py`,
  `code_eval_min.py`. A clean +Δ here makes the generality claim external, not just synthetic CompDAG.
- **Cost:** 2–4 cells. **Risk:** medium (verifier/sandbox plumbing).

### P1.2 One larger-scale point (14B / 32B via sharded full-param)
- **Objection closed:** "scales" currently tops out at 7–9B; a 14B/32B point removes the ceiling doubt.
- **Method:** `launch_sharded7b.sh` pattern with ZeRO-3 CPU-offload on Qwen2.5-Math-14B (or 32B across
  a node). Expect a smaller-but-positive Δ (headroom shrinks) — consistent with the scale trend.
- **Cost:** 1–2 nodes, sharded, pod-death-prone → per-stage S3 sync mandatory. **Risk:** medium-high.

### P1.3 Statistical rigor on the headline matrix
- **Objection closed:** "seed SD isn't a significance test."
- **Method:** paired **bootstrap** CIs (resample problems) + a paired sign/permutation test per cell on
  the final-24 matrix; report p-values / BCa intervals. Post-hoc on existing per-problem JSONs.
- **Cost:** ~0 GPU. **Risk:** low. Bundle with P0.1.

### P1.4 Downstream utility
- **Objection closed:** "does pass@1 reliability translate to something users feel?"
- **Method:** a compute-savings statement (RVP@1 accuracy = base@ best-of-n → n× fewer samples) from
  P0.1, and/or a short agentic/tool-use task where single-shot correctness matters. The frontier plot
  from P0.1 is likely enough.

---

## P2 — polish / stretch (writing is where awards are won or lost)

### P2.1 Framing & front matter (do regardless — this is high-leverage)
- **Title + abstract:** lead with the *counterintuitive* claim (reliability without new capability;
  selection, not acquisition; headroom-gated). One-sentence result: "converts coverage into pass@1 at
  constant coverage, +.04–+.21, via a single verified-preference pass."
- **Figure 1:** the concept figure — coverage≫pass@1 gap → RVP reallocates mass I→C → pass@1 rises at
  fixed coverage. Pair with the σ(m+c) money figure (P0.2) and the cost–reliability frontier (P0.1) as
  the three "hero" figures.
- **Related work:** position crisply vs RLHF/DPO, RFT/STaR/ReST, self-consistency/best-of-n, process &
  outcome reward models, verifier reranking. Our novelty = *verified-self-label decoupled preference
  run after RFT, headroom-gated, margin-mechanistic* — not the DPO loss.
- **Limitations + broader impact + reproducibility checklist** (NeurIPS requires these): scope
  boundaries (already honest — an asset), compute, seeds, code/data release, verifier details.

### P2.2 Consistency & camera-ready pass
- Reconcile the two 1.5B MATH-500 numbers in the paper (sharded seed-robustness `+.172` vs LoRA-1GPU
  final-24 `+.100`): state explicitly they are **full-param sharded** vs **LoRA-1GPU** configs so a
  reader doesn't read a contradiction. (Currently implicit.)
- Fix the two `Overfull \hbox` wide tables (`tab:main`, and the `\running` grid) — shrink or wrap.
- Unify "\running{} in-flight" language: several cells marked in-flight are now **done** (final-24
  covers them) — promote or remove the `\running` tags so the paper doesn't read as unfinished.
- Make sure every table's config (LoRA vs sharded, β, steps, k, n) is in its caption.

### P2.3 Stretch mechanistic depth
- Causal patching: patch the late-layer (26/28) RVP residual direction into the base and show pass@1
  rises — turns the *correlational* logit-lens localization into a *causal* claim. High-effort, high-reward.

---

## Recommended execution order (fastest path to award-ready)
1. **P0.1 + P1.3 + P0.2** — all mostly post-hoc on existing eval JSONs → the cost–reliability frontier,
   bootstrap CIs, and the σ(m+c) figure. ~1 day, low risk, closes the biggest objections. **Start here.**
2. **P0.3** iterative-RFT / generic-DPO / reranker baselines — a few fleet cells.
3. **P1.1** real code domain (MBPP/HumanEval+) — the external-validity win.
4. **P2.1 + P2.2** writing: front matter, three hero figures, consistency/camera-ready pass.
5. **P1.2 / P2.3** larger-scale point and causal patching — if fleet time allows.

Everything runs detached + per-stage S3-synced (nodes cycle); harvest immediately; report honestly
(nulls stay). The paper is already solid — items 1–3 are what turn "solid" into "award contender."
