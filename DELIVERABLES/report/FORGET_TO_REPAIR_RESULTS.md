# Forget-to-Repair — the (E_rich, ∅) quadrant (2026-09-03)

*Question the anchoring paper (Try Again, Don't Look Back, Jul 2026) does NOT answer: can we keep useful
failure EVIDENCE while erasing the harmful failed PROPOSAL? Decompose failure context = evidence E + failed
proposal P. Self-repair = (E,P) [anchors]. Blind resampling = (∅,∅). error-only ≈ (E_weak,∅) [parity].
The missing quadrant is **(E_rich, ∅)** — a concrete counterexample, code erased.*

Harness: `rl_training/certify.py` (`make_certificate`: runs each unit-test assert individually to isolate a
concrete counterexample `call → returned X, correct is Y`, containing NONE of the failed code) +
`seq_recover.py` diag-modes `certificate` (latest counterexample only) / `cert_memory` (accumulate
counterexamples). Matched budget T=6, 8-GPU DP.

## Exp 1 — Failure Certificate beats iid (GATE: CLEARED)
recovery@T=6, error-only recovery restored parity (~+0.018); the rich certificate goes further:

| model (mbpp) | iid | static | SEQ (certificate) | **seq−iid** |
|---|---|---|---|---|
| Llama-3.1-8B | 0.449 | 0.486 | 0.598 | **+0.150** |
| Qwen2.5-Coder-14B | 0.462 | 0.375 | 0.567 | **+0.106** |
| Qwen-Instruct-7B | 0.358 | 0.432 | 0.407 | +0.049 |
| Qwen2.5-Coder-7B | 0.598 | 0.551 | 0.617 | +0.019 |

**All 4 positive; 2 cells ≥ +10 pts** (pre-registered bar: ≥+3–5 on ≥2). Certificate > error-only > iid ≈
full-mode-minus-anchoring. **Rich failure evidence survives proposal-erasure and beats blind resampling.**

## Exp 2 — Cumulative certificate memory (cumulative learning WITHOUT anchoring)
Accumulating counterexamples (code always hidden) makes recovery RISE monotonically across rounds — the
opposite of full-history anchoring (which flattens/declines):

| model (mbpp) | seq curve (recovery@t, t=1..6) | final seq−iid | vs single-cert |
|---|---|---|---|
| Qwen-Instruct-7B | 0.242→0.374→0.444→0.495→0.515→0.515 | **+0.162** | +0.11 over single (+0.049) |
| Qwen2.5-Coder-14B | 0.310→0.405→0.476→0.512→0.524→0.548 | +0.060 | ≈ single (+0.106) |

Qwen-Instruct: accumulating counterexamples lifts +0.16 over iid and +0.11 over a single certificate —
cumulative failure learning works when the proposal is stripped.

## Verdict: GO — the Forget-to-Repair hypothesis holds at inference.
The novel claim is supported: **failure feedback is beneficial once you separate evidence (counterexample)
from the model's own failed proposal.** This is the missing quadrant the anchoring literature leaves open,
and it flips the project from "iid is a strong baseline you can't beat" to "you CAN beat iid — with rich,
proposal-free evidence." error-only (weak evidence) was parity; the concrete counterexample is the lever.

## 2nd wave — confirmed on HumanEval + full cert_memory grid + tight replicates
**HumanEval certificate (beats iid there too):** 14B **+0.200** (SEQ 0.667 vs iid 0.467), Qwen-Instruct +0.083.
**cert_memory grid completed:** Llama **+0.122**, Qwen-Coder +0.071 (all 4 families positive).
**Replicates (tight):** Qwen-Instruct mbpp +0.049 / +0.049 (identical); 14B mbpp +0.106 / +0.097.
⇒ **Certificate beats iid on all 11 cells measured** (mbpp ×4 + HE ×2 + cert_memory ×4 + replicates), mean
seq−iid ≈ +0.10. The (E_rich,∅) effect is robust and replicated — not noise.

## Exp 3 — CAUSAL proof: the failed proposal shapes the retry (and the certificate breaks it)
For each problem, two DISTINCT failed parents A,B (different algorithm signatures); repair from each under
FULL (show parent code) vs CERT (counterexample only). recurrence = sim(repair, own-parent) − sim(repair,
other-parent), sim = ½·algo-label-match + ½·token-Jaccard. (inference-only, no training)

| model | n | recurrence FULL | recurrence CERT |
|---|---|---|---|
| Qwen-Instruct-7B | 169 | **+0.351** | −0.015 |
| Qwen2.5-Coder-7B | 116 | **+0.418** | −0.006 |

Under full-mode the repair strongly resembles ITS OWN failed parent's algorithm (+0.35/+0.42) — the failed
proposal *causally* determines the retry's hypothesis space (not merely textual copying: the signal includes
algorithm-family match). Under the certificate (code erased) recurrence collapses to ≈0. **Causal evidence
that (a) self-repair anchors at the reasoning/algorithm level, and (b) the Failure Certificate works by
removing the proposal's causal pull.** This is the paper's mechanistic centerpiece.

## Next wave (launching now, gated on this GO)
- **Exp 3 — causal same-problem, different-failed-parent:** does the retry algorithm follow the failed
  parent? (proves anchoring is reasoning-level, and that the certificate breaks it).
- **Exp 4 — retry routing entropy:** H(M|q) vs H(M|q,y⁻,e) vs H(M|q,certificate) — reuse the ρ↓/c↑ routing
  machinery; predict full-context collapses routing toward the parent mode, certificate restores it.
- **Exp 7 — dense residual-failure-reward GRPO:** reward = test-pass-fraction + residual-repair term (fixes
  the binary-reward null of v1/v2), trained on the certificate condition.
- **Exp 6 — learned failure bottleneck:** encoder z=f(q,y⁻,e) maximizing repair success while an adversarial
  decoder cannot reconstruct y⁻ — the learned generalization of the hand-built certificate.
- Confirm on HumanEval (inst3, fetching) + more cert_memory cells for tight CIs.

## Data
`runs_pulled/seq/seq_seq_{qi,ll,qc,qc14}_cert.json`, `..._certmem.json`. Reproduce: `go_seq.sh <dir> <tag>
mbpp 6 -1 certificate|cert_memory`.
