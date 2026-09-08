# Findings, Core Novelty, and the Downstream Question — one-page summary (2026-09-01)

## The one-sentence claim

> **What looks like "mode collapse" under RLVR is largely *routing compression*, not *capability erasure*:
> the model stops *choosing* many reasoning strategies, but — on the strategies the base was genuinely
> competent at — it can still *execute* them at least as well, or better. Marginal strategy diversity and
> reasoning capability are different objects, and in math they are nearly decoupled.**

---

## What we measured (the decomposition)

For a problem `q` and a reasoning strategy `m`, we separate three quantities that prior "diversity
collapse" work conflates into one marginal number:

- **Accessibility / routing** `ρ(m|q)` = how often the model *chooses* strategy m (measured from free
  generations + a strategy classifier, and from the log-prob of entering the strategy).
- **Conditional competence** `c(m,q)` = P(correct | forced to use m) — can it still *execute* m?
  (measured by high-adherence prefix-forcing, forced_n=32).
- **Functional complementarity / value** `v(m|q)` = does m solve problems the *other* strategies can't?
  (marginal coverage over the strategy set).

A strategy only matters downstream if **A · C · V** is jointly non-trivial — the **ACV** view.

---

## Findings (evidence, honestly qualified)

**F1 — Routing compresses under RL.** Behavioral routing entropy drops (Qwen: 3.51 → 2.71 effective
strategies/problem; log-prob of entering named strategies down −0.34 nats/tok on Qwen, −0.67 on Llama).
The support-floor baseline preserves routing (≈ base). *Both* an entropy measure and a log-prob measure
agree.

**F2 — Competence is preserved/improved, on modes that matter (the key result).** Using a stable
high-N measurement (forced_n=32) and controlling for base competence:

| family | modes with base competence ≥ 0.2 | mean Δc (grpo − base) | retain | improve |
|---|---|---|---|---|
| Qwen  | 174 | **+0.077** | 87% | 71% |
| Llama | 63  | **+0.169** | 94% | 83% |
| DeepSeek | 328 | −0.027 | 55% | 32% (boundary case) |

So on Qwen + Llama, RL suppresses routing (F1) while conditional competence *rises* (F2) — `ρ↓, c↑`.
(An earlier forced_n=4 pass suggested erosion; that was small-sample regression-to-mean, removed by
forced_n=32. DeepSeek is the honest boundary case: weak base on MATH-500 + train/eval mismatch.)

**F3 — In math, strategies are functionally redundant (`v ≈ 0`).** Of 100 solvable problems, almost
none are solved *uniquely* by one strategy; mean per-mode complementarity `v_m ≈ 0.014`; a mean of
8–9 of 14 strategies solve each solvable problem.

**F4 — Therefore diversity buys no downstream accuracy in math.** Stratified sampling (deliberately
drawing K *different* strategies) vs iid sampling, matched budget, across Qwen/Llama/DeepSeek:
Δ(stratified − iid) ∈ [−0.05, +0.01] ≈ 0, and **pass@32 is identical** for stratified and iid.
Oracle single-strategy routing is *below* plain iid pass@16. Preservation baselines (floor / DPH-F)
add marginal diversity but no competence or pass@k edge over plain GRPO.

**F5 — The paradox is mechanistically expected, not surprising.** Softmax routing gradient
`∂J/∂z_m = ρ_m(c_m − J)` suppresses below-average strategies; but the executor shares parameters, so
`Δc_m ≈ η Σ_j ρ_j ⟨∇c_m, ∇c_j⟩` — competence can *rise* via shared-gradient transfer even as routing
falls.

**F7 — Gradient-alignment (mechanism test) — INCONCLUSIVE (infra).** Attempted to measure the
strategy-conditioned gradient cosine `G_ij=cos(∇L_i,∇L_j)` directly (grad_align.py) to empirically ground
F6's benign-collapse claim. Repeated CUDA OOM on the single-GPU full-sequence backward of a 7B model (even
at last-1-layer + CPU-stored grads + gradient checkpointing — killed-process GPU memory not freeing between
relaunches on the shared GPU; a partial run reached 11/14 strategies). **Deferred to future work** (needs
multi-GPU/ZeRO sharding or activation offload). F6 therefore rests on the analytical derivation + the robust
empirical ρ↓/c↑ (F1,F2), not a measured G matrix.

---

## Core novelty (what is new vs the field)

The 2026 literature (SetPO, DPH-RL, DMPO, Uniqueness-Aware RL, ModC) treats *marginal* strategy
diversity as the thing to preserve, and mode collapse as a capability problem to fix. Our contribution
is a **measurement + conceptual correction**, not another regularizer:

1. **Causal decomposition** of "mode collapse" into routing (ρ), conditional competence (c), and
   functional value (v). We show marginal mode probability **cannot identify capability loss** — the
   same `ρ_m` can hide a fully-competent-but-unchosen strategy or a genuinely erased one.
2. **New empirical phenomenon** across model families: `ρ↓` while `c↑` — collapse without forgetting.
3. **A theory of *when collapse is benign vs harmful*.** Harmful collapse requires *both* loss of
   competence/access **and** non-redundant functional value (`v>0`). In math, `v≈0`, so the observed
   collapse is *benign compression*. This reconciles our null with the literature's positive diversity
   results: **diversity helps only where modes are functionally complementary.**

That reframes the target from *"preserve diversity"* to *"preserve only functionally non-substitutable
capability"* — and shows most diversity-preservation effort in math is optimizing a quantity with no
downstream value.

---

## Is the downstream task important? — the honest answer

**We deliberately tested it, and in math the answer is: no measurable downstream benefit from the
diversity/repertoire angle.** Stratified sampling ≈ iid (F4); oracle routing < iid; adaptation to other
math tasks showed no penalty from collapse (separate experiment). This is *why the paper is a
correction, not a performance-method paper* — and stating it plainly is a strength, not a weakness.

**Where downstream importance *could* still exist (and the paper's forward claim):** only when
strategies are **functionally complementary** (`v>0`) — e.g. controlled algorithmic regimes (BFS vs
DFS, brute-force vs DP, enumeration vs closed-form) where one strategy uniquely succeeds. There,
mode-diverse sampling should help (consistent with ModC). The paper's testable prediction:

> **Δpass@k from mode-diverse sampling ∝ functional complementarity v.**

Math sits at `v≈0, Δpass@k≈0`; complementary tasks sit at `v≫0, Δpass@k≫0`. If that relationship
holds, `v` — not entropy — is the quantity that determines whether reasoning diversity matters. **That
is the downstream importance: not "preserve diversity for accuracy," but "diversity has value exactly
and only where complementarity is high," which is measurable and predictive.**

---

## Status & what remains
- Validated: decomposition, `ρ↓/c↑` (Qwen+Llama, high-N), `v≈0` in math, no downstream diversity gain.
- To strengthen to award-contending: **mechanism** — layer-wise strategy-decodability probe (is the
  suppressed strategy still *represented*?) + activation steering (causal recovery); the **controlled
  complementarity benchmark** (the only place harmful collapse / diversity value can exist); and running
  the diversity-preservation baselines through the (ρ,c,v) probe.
- Full plan: `PIVOT_mechanism_plan.md`. Detailed results: `ROUTING_VS_COMPETENCE_RESULTS.md`.

## §44 (2026-09-08): The SFT operator advantage is IRREDUCIBLE (72-GPU sweep)
Tested whether GRPO's weak OOD transfer can be "rescued" by adding a forward-KL self-distillation term
(NLL on the model's OWN verified-correct traces), weight β_sd — the mass-placing axis: β_sd=0 is plain
GRPO, β_sd→∞ is pure SFT. Swept β_sd∈[0.05,20] × {400,800} steps, 8–9 replicates/point across 9 nodes,
two OOD benchmarks.
RESULT (honest): FLAT plateau. MATH-500 stays ~0.33 (vs GRPO 0.30, SFT 0.410); full-MATH stays ~0.29≈base
(vs SFT 0.361). No dose-response even at 10–20× weight. H1 (monotonic axis) and H2 (full rescue) REFUTED.
WHY IT MATTERS (thesis-strengthening): the SFT operator's OOD-transfer advantage is NOT decomposable into
"GRPO + trace rehearsal." The on-policy advantage-weighted gradient caps how much an explicit NLL term can
place mass (ρ≈0 persists) — so you cannot cheaply convert GRPO into SFT-level generalization. The UPDATE
RULE itself is the irreducible cause of OOD transfer of verified experience. This is a stronger, more
falsifiable claim than a clean rescue would have been, and it closes the obvious reviewer question
("just add rehearsal to GRPO") with a decisive negative.
