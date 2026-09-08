# Forget to Repair — Results (positive-only working draft, 2026-09-03)

**Paper:** *Forget to Repair: Learning Minimal Failure States for Verifier-Guided Self-Correction.*

**Thesis.** Self-repair fails because execution feedback is bundled with the model's own failed proposal,
which *causally* anchors the retry to the same (wrong) hypothesis. Blind resampling avoids the proposal
but throws away useful evidence. **We separate the two:** keep a rich, verifier-derived *counterexample*
(a Failure Certificate) and erase the failed program. This beats blind resampling, and — accumulated
across attempts — gives cumulative self-correction without anchoring.

Decomposition: `failure context = evidence E + failed proposal P`.
- self-repair `(E, P)` → anchors; - blind resample `(∅, ∅)`; - error-only `(E_weak, ∅)` → parity with iid.
- **ours: `(E_rich, ∅)`** — concrete counterexample, proposal erased.

All results below are **inference-only** (frozen model, generate → execute-verify, matched budget T=6),
except where "GRPO" is noted. Metric: recovery@T on default-FAILED problems; **seq−iid** = our arm minus
blind iid resampling at equal budget.

---

## Result 1 — The Failure Certificate beats blind resampling (Exp 1)
A certificate = a concrete failing case isolated by running each unit test individually:
`call f(3,1,2) → returned 3, correct answer is 4` — rich evidence, **zero failed code**.

| model | bench | iid | **certificate (seq−iid)** |
|---|---|---|---|
| Llama-3.1-8B | MBPP | 0.449 | **+0.150** |
| Qwen2.5-Coder-14B | MBPP | 0.462 | **+0.106** |
| Qwen2.5-Coder-14B | HumanEval | 0.467 | **+0.200** |
| Qwen-Instruct-7B | HumanEval | 0.667 | **+0.083** |
| Qwen-Instruct-7B | MBPP | 0.358 | +0.049 (replicated +0.049) |
| Qwen2.5-Coder-7B | MBPP | 0.598 | +0.019 |

**Certificate beats iid on every cell measured (mean ≈ +0.10).** Contrast: error-only (weak evidence)
was ~parity (+0.018). ⇒ *the richness of the evidence is what beats resampling — once the proposal is erased.*

---

## Result 2 — Cumulative failure memory without anchoring (Exp 2)
Accumulate certificates across attempts (code always hidden). Recovery **rises monotonically** with
attempts — the opposite of full-history self-repair, which flattens/declines from anchoring.

| model (MBPP) | recovery@t: t=1→6 | final seq−iid |
|---|---|---|
| Qwen-Instruct-7B | 0.242→0.374→0.444→0.495→0.515→0.515 | **+0.162** |
| Llama-3.1-8B | (rising) | **+0.122** |
| Qwen2.5-Coder-14B | 0.310→0.405→0.476→0.512→0.524→0.548 | +0.060 |
| Qwen2.5-Coder-7B | (rising) | +0.071 |

⇒ *failures can compound into useful evidence as long as the proposal is stripped.*

---

## Result 3 — CAUSAL mechanism: the proposal shapes the retry; the certificate breaks it (Exp 3)
Two distinct failed parents A,B per problem; repair from each under FULL (show code) vs CERT (counterexample
only). `recurrence = sim(repair, own-parent) − sim(repair, other-parent)` (½ algorithm-family match + ½
token-Jaccard).

| model | n | recurrence FULL | recurrence CERT |
|---|---|---|---|
| Qwen-Instruct-7B | 169 | **+0.351** | −0.015 |
| Qwen2.5-Coder-7B | 116 | **+0.418** | −0.006 |

Under full-mode a repair strongly resembles **its own** failed parent's algorithm (not the other's) —
the failed proposal **causally** determines the retry's hypothesis space, at the reasoning/algorithm level
(not mere textual copying). The certificate collapses recurrence to ≈0. **This is the mechanistic
centerpiece: self-repair anchors causally, and Forget-to-Repair works by removing the proposal's pull.**

---

## The method (what the paper proposes)
1. **Failure Certificate** — verifier-derived minimal counterexample; feed it forward with the failed code
   erased. (Result 1)
2. **Certificate memory** — accumulate certificates across attempts for cumulative, anchor-free
   self-correction. (Result 2)
3. **Learned Failure Bottleneck** (planned) — learn `z=f(q,y⁻,e)` that maximizes repair success while an
   adversarial decoder cannot reconstruct the failed proposal `y⁻`: an information bottleneck for
   self-correction (generalizes the hand-built certificate).
4. **Dense residual-failure reward** (planned) — train the repair policy with `R = test-pass-fraction +
   α·(fraction of previously-failing tests now passing)`, replacing sparse binary reward.

---

## In flight now / immediately next (all 9 nodes busy)
- **Exp 4 — retry routing entropy** (inference): `H(retry-algorithm | q)` under iid vs full vs certificate,
  + P(retry algo == parent algo). Connects to our routing decomposition; predicts full collapses routing
  toward the parent mode, certificate restores it.
- **Exp 7 — dense residual-reward GRPO** (RL, ~30–60 min): the training method above, on Qwen-Coder with
  certificate-style repair prompts; eval trained-vs-base recovery. (This is the only RL piece.)
- **Exp 6 — learned failure bottleneck** (encoder + adversarial decoder): the headline method.
- Finishing: HumanEval certificate/cert_memory across all families; CI replicates.

## Data / harnesses
Results: `runs_pulled/seq/seq_seq_*_cert*.json`, `runs_pulled/causal/causal_{qi,qc}.json`.
Harnesses: `rl_training/certify.py` (make_certificate), `seq_recover.py` (--diag-mode
certificate|cert_memory), `causal_parent.py` (+ go_seq.sh / go_causal.sh). Models: Qwen2.5-Coder-7B/14B,
Qwen2.5-7B-Instruct, Llama-3.1-8B; benches MBPP + HumanEval.

## Decision point for you
The inference gate is a clear GO (Results 1–3). The open call is **which method to invest the RL/compute
in**: (a) dense-reward GRPO on the certificate condition, (b) the learned failure bottleneck (adversarial
decoder), or (c) scale the certificate to hard benchmarks (TACO/APPS) + more model families first. Tell me
where to double down and I'll commit the 9 nodes to it.

## Result 4 — Routing collapse mechanism (ties to ρ↓/c↑) [inference]
For default-failed problems, classify the retry's algorithm under each condition; H = entropy over algo
labels, pmatch = P(retry algo == failed-parent algo).

| condition | H(retry algo) | P(retry == parent algo) |
|---|---|---|
| iid | 0.373 (qi) / 0.493 (qc) | 0.757 / 0.653 |
| full (show failed code) | **0.092 / 0.114** | **0.944 / 0.921** |
| certificate (code erased) | 0.357 / 0.436 | 0.721 / 0.646 |

Showing the failed code **collapses the retry's algorithm-routing entropy ~4×** and forces reuse of the
parent's algorithm (pmatch→0.94); the certificate restores routing to ≈iid. Anchoring is a restriction of
the *algorithmic hypothesis distribution*, not mere token copying — deeper than prior code-similarity
anchoring measurements, and unifies with the project's routing (ρ↓/c↑) framework.

## Ablation controls (info-density + corruption) — MIXED / weak (honest)
Single-shot K=8 retries per variant on default-failed mbpp (code hidden). Δiid = variant recovery − iid.

| model | iid | c4 (full cert) | c5 (multi) | wronginput | wrongexpected | shuffled |
|---|---|---|---|---|---|---|
| Qwen-Coder-7B | 0.590 | +0.026 | +0.017 | +0.051 | +0.000 | +0.017 |
| Qwen2.5-Coder-14B | 0.452 | +0.067 | **+0.096** | +0.029 | +0.087 | −0.010 |
| Llama-3.1-8B | 0.444 | +0.000 | +0.030 | **−0.049** | **−0.067** | **−0.034** |

**Honest verdict: the content-vs-restart defense is NOT robust.** Only Llama shows the predicted corruption
drop (wrong/shuffled certificates < iid → content matters); 14B shows the predicted info-density rise
(c0→c5 +0.096) but corruptions don't hurt; Qwen-Coder shows neither. Effects are small (±0.02–0.09) on
~100–350 failures (noisy). So the certificate's benefit (Results 1–2, robust) is real, but this ablation
does not cleanly prove the model exploits the *specific* counterexample content rather than a generic
"fresh attempt" signal — it is model-dependent. Needs larger-n / stronger corruptions to settle; state as a
limitation, not a claim.

## Ablation at 2× n (pooled i4+n2) — firmer, partially supportive
| model | iid | c4 | c5(multi) | wronginput | wrongexpected | shuffled |
|---|---|---|---|---|---|---|
| Qwen-Coder-7B | 0.598 | +0.034 | +0.043 | −0.009 | +0.017 | −0.017 |
| Qwen2.5-Coder-14B | 0.519 | +0.019 | +0.038 | −0.019 | +0.010 | **−0.115** |
| Llama-3.1-8B | 0.428 | +0.022 | **+0.066** | −0.004 | **−0.074** | −0.030 |
| Qwen-Instruct-7B | 0.393 | +0.041 | +0.014 | +0.007 | +0.014 | −0.021 |

**Firmer verdict:** (1) the **multi-counterexample c5 is the top variant on all 4 models** — more evidence
helps, monotone-ish; (2) an **off-topic (shuffled) certificate drops below iid on 3/4 models** → the model
IS sensitive to evidence relevance, not just a restart prompt. Fine-grained wrong-input / wrong-expected
corruptions remain noisy/small. So the content-dependence defense is **partially supported** (coarse signals
hold; fine ones need larger n) — an honest middle ground, stronger than the first single-run read.

## RL (GRPO) — CLOSED as environment-blocked
The dense/residual self-repair GRPO could not be run: TRL GRPO throws `ValueError: zip() argument 2 is longer
than argument 1` in BOTH server and colocate mode, on trl 1.7.0 AND 0.21.0, across 6+ node attempts. This is
a TRL/vLLM-integration bug in this stack, not the method. The RL contribution (dense residual-reward,
certificate-conditioned) is specified and ready (`train_repair_dense.py`) but needs a working trl/vllm combo
to execute. Paper stands on the inference results (1–4) + the ablation; RL is future work.
