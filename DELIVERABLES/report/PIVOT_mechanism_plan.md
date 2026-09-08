# Pivot Plan — the Corrective + Mechanism paper (2026-09-01)

## Why we pivoted (evidence, not vibes)

Three "performance win" angles were tested cheaply and **all came back negative**:
- **Adaptation / optionality** (math→math shifts): collapsed adapts as well or better → no penalty.
  (`ADAPTATION_OPTIONALITY_RESULTS.md`)
- **Downstream routing exploit** (stratified pass@k, oracle router, functional rank): Δ(stratified−iid)
  ≈0 across Qwen/Llama/DeepSeek; pass@32 identical; oracle-strategy < iid pass@16; oracle-competence
  ceiling ~5 pts. → no exploitable performance value. (`ROUTING_VS_COMPETENCE_RESULTS.md`, go/no-go)
- **Preservation → accuracy**: floor/DPH-F restore marginal diversity but give no competence or pass@k
  edge over plain GRPO.

So the paper is **not** a new regularizer, **not** a router, **not** weeks of large-scale RL for a
method. It is a **scientific correction with a mechanistic backbone**:

> **"Mode collapse" under RLVR is routing compression, not competence erasure. Marginal reasoning
> diversity and reasoning capability are different objects; diversity-preservation RL (SetPO / DPH-RL
> / DMPO / support-floor) optimizes a marginal quantity that carries no downstream value.**

The award-lever is the **mechanism**: show the suppressed strategies are still *represented* and
*causally recoverable* — "knowing vs choosing" proven at the representation level, not just behaviorally.

---

## What is already DONE (the corrective evidence — reuse, don't rerun)

| # | Claim | Evidence | Status |
|---|---|---|---|
| C1 | Decomposition π=Σρ·c is measurable | strategy_probe (c) + route_logprob (ρ) + free_route (behavioral ρ) | ✅ |
| C2 | Competence survives/rises under RL | forced-c grpo ~1.7× base, all 14 strategies +, high-adherence (~82%) | ✅ Qwen, Llama |
| C3 | Routing compresses under RL | logp Δ −0.34 (qm) / −0.67 (llama); behavioral H(ρ) 3.51→2.71 eff strategies; floor preserved | ✅ |
| C4 | 90% of "collapsed" (q,m) keep/raise competence | Δlogρ-vs-Δc quadrant map: 1877/2087 in QII | ✅ |
| C5 | Effect is training-time, established early | qm base→r1→r2 dynamics: ρ↓/c↑ in first 100 steps then plateau | ✅ |
| C6 | Cross-family (not Qwen-specific) | Llama replicates; DeepSeek routing pending pull | ✅ Qwen+Llama |
| C7 | Modes functionally redundant; small basis | 8–9/14 solve avg problem; minimal basis 7–9/14 | ✅ |
| C8 | No downstream win from diversity | stratified≈iid, oracle<iid, adaptation null | ✅ (the honest negative) |

**Only hardening needed on the corrective side:** replace the coarse regex strategy classifier
(~36% unclassified) with a **strong instruction-model judge** (e.g. Qwen2.5-7B-Instruct / Llama-3.1-8B-
Instruct as a separate frozen classifier), validate F1>0.9 on 300–500 hand-checked traces, and re-derive
ρ (behavioral) + the quadrant map. This is the one methodological gap a reviewer will attack.

---

## The NEW core: MECHANISM experiments (to run — this is the paper's spine)

### M1 — Layer-wise strategy decodability probe  (small scale, ~1–2 nodes)
**Question:** after RL suppresses a strategy's routing, is the strategy still *represented*?
- Build a strategy-labeled trace set: for N problems, generate traces, label each with the strong
  judge (above). Keep balanced classes over the 14 strategies.
- Extract per-layer hidden states `h_ℓ(q)` at a fixed position (last token of the strategy-prefix, or
  mean-pooled over the plan span) with an HF forward (`output_hidden_states=True`) — vLLM won't expose
  these, use `transformers` batched forward.
- Train a linear/logistic probe `h_ℓ → M` per layer, for **base / grpo / floor / dphf**.
- **Metric:** strategy-decodability accuracy vs layer. **Hypothesis (C-mech-1):** for strategies whose
  output routing dropped >X nats, layer decodability is ≈unchanged base→grpo (representation retained).
- Harness to build: `strategy_hidden.py` (extract + probe), `go_probe_hidden.sh`.

### M2 — Activation steering: causal recovery of a suppressed strategy  (small scale, ~2–3 nodes)
**Question:** can we *causally* bring a suppressed strategy back — and does it still execute correctly?
- Learn a mode direction per strategy: `v_m = E[h_ℓ | M=m] − E[h_ℓ | M≠m]` (from M1's hidden states).
- Steer generation: `h_ℓ' = h_ℓ + α·v_m` (hook on the residual stream, sweep α), for the grpo model.
- **Metrics:** (a) does strategy m reappear in free generation (behavioral ρ_m rises under steering)?
  (b) is the steered output still correct (competence preserved under causal intervention)?
- **Hypothesis (C-mech-2):** steering the grpo model along `v_m` restores ρ_m ≈ base level with
  competence intact → suppression, not erasure, proven causally.
- Harness to build: `steer_recover.py` (register forward hooks, α-sweep, measure ρ_m + correctness).
- Note: activation steering itself isn't novel (RISER); the novelty is using it as a **causal probe of
  what RLVR collapse is**, tied to the ρ/c decomposition.

### M3 — Mechanism generality (small scale)
- Repeat M1 (+ M2 on the cleanest strategy) on a 2nd family (Llama) and 2 checkpoints (r1, r2) to show
  the representation-retention is not model- or step-specific.

### M4 — Strong-classifier re-derivation (small scale, gates M1/M2 quality)
- Stand up the frozen instruct-judge; re-label the free_route + probe traces; re-emit behavioral ρ,
  H(ρ), and the Δρ-vs-Δc quadrant map with the strong classifier. Report classifier F1 + agreement
  with regex as an appendix. This closes the last measurement objection.

---

## Experiment ordering (all SMALL-scale; no weeks-long RL)

1. **M4 strong judge** — fetch an instruct model, re-label, re-derive ρ + quadrant map. (gates the rest)
2. **M1 layer probe** — base/grpo/floor/dphf decodability-vs-layer. (the key mechanism figure)
3. **M2 steering** — causal recovery on grpo for the top suppressed strategies. (the causal clincher)
4. **M3 generality** — M1 on Llama + r1/r2 checkpoints.
5. **(optional) compute-efficiency check** — same-accuracy-fewer-samples was the only unclosed weak
   downstream angle; a quick pass@k-per-token comparison of stratified vs iid to formally close it.

**Go/no-go for the paper's strength:**
- If M1 shows decodability retained AND M2 steering recovers the mode with competence → the mechanism
  triad (behavioral C3/C4 + representation M1 + causal M2) is a genuinely strong, arguably award-
  contending "knowing vs choosing" result. **No large-scale RL needed.**
- If M1 shows decodability *also* collapses for the suppressed strategies → then it's partial erasure,
  not pure suppression; the paper becomes "collapse is a mix," still a correction but weaker — report
  honestly.

## The paper's contributions (final framing)
1. **Decomposition** of reasoning-mode collapse into routing ρ and conditional competence c (causal).
2. **Empirical correction**: RLVR compresses ρ while preserving/raising c; 90% of "collapsed" modes
   keep competence; cross-family + training-dynamics.
3. **Mechanism**: suppressed strategies remain linearly decodable (M1) and are causally recoverable by
   activation steering with competence intact (M2) — knowing ≠ choosing at the representation level.
4. **Negative result with teeth**: marginal diversity ≠ functional capability; diversity-preservation
   RL and repertoire routing yield no downstream accuracy (stratified≈iid, oracle<iid, adaptation null)
   — the field is optimizing the wrong object.

## Explicitly NOT doing
- No new preservation regularizer. No repertoire router as a "method". No weeks-long large-scale RL for
  a performance method. (All three are unsupported by the go/no-go evidence.)

## Harnesses (built vs to-build)
- Built + validated: `strategy_probe.py` (c, prefix-forced), `route_logprob.py` (ρ logp),
  `free_route.py` (behavioral ρ, regex classifier), `stratified_passk.py` (downstream gate),
  `analyze_probe.py` (quadrant map, oracle, repertoire, rank), launchers `go_probe/go_route/go_free/go_sp/go_ff/go_pr.sh`.
- To build: `strategy_hidden.py` + `go_probe_hidden.sh` (M1), `steer_recover.py` (M2), instruct-judge
  classifier module (M4).
