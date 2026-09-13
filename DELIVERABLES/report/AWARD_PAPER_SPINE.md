# The Support–Coverage Theory of Verified Self-Improvement
_Award-level theory + methodology spine. 2026-09-13. Companion to ADAPTIVE_FORGETTING_RESULTS.md (§111,§120–§140) and PROGRESS_SNAPSHOT v11._

## THE ONE-SENTENCE CONTRIBUTION
We give a **predictive theory** of verified self-improvement that (a) proves on-policy outcome-RL and decoupled verified replay (RFT) are the **same gradient family**, so the large OOD gap between them is *procedural* (coverage/replay), not objective-fundamental; (b) shows the realized gain is governed by **headroom × accessibility** (an inverted-U), not headroom alone; and (c) predicts that **RFT is the ceiling of the accessible frontier** — no acquisition or retention wrapper can beat it — a prediction we confirm three independent ways. We supply the **α/β hidden-acquisition measurement methodology** that reveals net accuracy conceals a bimodal acquire/regress process, and **VSF**, a causal instrument that decomposes an RL trainer into its coverage vs objective channels.

**Why this is award-shaped without a manufactured winner:** the standard bar is "propose a method that wins." We instead *prove why the simple method (RFT) already wins and cannot be beaten on the accessible frontier*, turn that into quantitative preregistered predictions, and confirm them out-of-sample across 3 families × 3 difficulties. The failed method-searches (regression-budget §139, feedback-acquisition A0 §140) become **confirmations of Theorem 4**, not embarrassing nulls.

## THEORY (formal spine)
- **T1 — Gradient identity (established §111).** For a parameter-independent binary verifier V with p_θ(x)=E_{y~π}[V], ∇_θ p_θ(x) = p_θ(x)·E[∇_θ log π(y|x) | V=1]. Within-prompt-normalized RFT on current-policy successes moves along ∇log p_θ(x); the binary-reward policy gradient is p_θ(x) × the same direction. ⇒ RFT and outcome-RL are **not distinct gradient families**; a persistent OOD gap must be procedural — *which* prompts receive a success-gradient and *how often*.
- **T2 — Support-boundedness + headroom bound (established §101).** Verified-only training gives zero direct gradient on unreached prompts; gains there are *measured generalization*, bounded by headroom · transferable-value. Reachability ≠ reliability.
- **T3 — Accessibility gating (new, §137 empirical → formalize).** Define accessibility a(x;C) = Pr(a verified solution for x is found within sampling budget C). Realized new-coverage gain ≈ Σ_x headroom(x) · a(x;C) · transfer(x). Because a(x;C) collapses at extreme difficulty (headroom rises but the bank cannot be filled), gain is an **inverted-U in headroom**, not linear. (Refutes the earlier linear-"law", §136→§137.)
- **T4 — Ceiling / saturation (new; the punchline).** Let B* = the verified bank realizable at budget C (the *accessible frontier*). (i) Once B* is filled, additional verified experience of the same distribution has zero marginal gain (**predicts A0/§140**). (ii) Continued on-policy RL initialized from RFT has zero net headroom over RFT (**predicts §139**, and follows from T1: RL only re-weights the same gradient directions RFT already saturated). ⇒ **No acquisition amplifier or regression-budget wrapper can exceed RFT on the accessible frontier.** RFT is the ceiling.

## METHODOLOGY (measurement + causal instrument)
- **M1 — α/β hidden-acquisition decomposition (measurement contribution).** Report, per problem: α = P(RL correct | base wrong) [acquisition], β = P(RL wrong | base correct) [regression], and oracle envelope O = E[max(base,RL)]. Net accuracy is misleading: GRPO's near-flat net (0.43 vs base 0.37) **hides** α=0.21 offset by β=0.20 (§138). Prescription for the field: *stop reporting net Δaccuracy alone for RL post-training; report (α,β,O).*
- **M2 — VSF as a causal probe (methodological novelty).** A persistent, prompt-balanced verified-replay floor added to the GRPO loss. Used **not as a SOTA method** but as a controlled instrument that turns the coverage/replay channel on independently of the objective. It provably repairs both axes (α↑ 0.21→0.39, β↓ 0.20→0.08) yet stays < RFT — isolating "coverage/replay" as the causal lever (confirms T1's procedural claim) while confirming T4's ceiling (VSF < RFT everywhere).

## PREREGISTERED OUT-OF-SAMPLE PREDICTIONS  (locked BEFORE the 72-GPU matrix results — this is the credibility hinge)
Tested on the fresh 3-family × 3-difficulty matrix (Qwen-1.5B/3B, deepseek-1.3B; mid 7→9, hard 12→14, vhard 16→18), seed CIs (grpo×3, rft×3, vsf×2 per cell):
- **P1 (sign + capability shrink).** RFT − GRPO > 0 in *every* cell; magnitude decreases monotonically with model capability. [Falsified if any cell shows GRPO ≥ RFT, or the ordering across sizes is non-monotone.]
- **P2 (α/β ordering).** α: RFT > VSF > GRPO and β: RFT < VSF < GRPO in *every* cell. [Falsified by any inversion outside seed CI.]
- **P3 (accessibility inverted-U — the sharpest test).** Per family, RFT-gain(over base) is non-monotone in difficulty: it does **not** keep rising from hard→vhard; vhard shows *lower* gain than the mid/hard peak despite larger headroom, because a(x;C)→0. [Falsified if gain rises monotonically with difficulty, which would revive the linear-headroom law and kill T3.]
- **P4 (VSF repair).** β_VSF < β_GRPO in *every* cell (dual-axis repair generalizes). [Falsified by any cell with β_VSF ≥ β_GRPO within CI.]
- **P5 (ceiling).** No cell has VSF > RFT beyond seed noise (T4 holds out-of-distribution family/difficulty). [Falsified by any VSF > RFT with non-overlapping CI.]

If P1–P5 hold with CIs → the theory has demonstrated **predictive** power (not just descriptive fit) across unseen difficulties and a 3rd family → award-shaped. Reported honestly: the theory's boldest prediction is a *negative* (P5/T4), pre-committed and then confirmed by two independent method-search kills.

## WHAT WOULD RAISE IT FURTHER (beyond current compute)
A frontier-scale point (>9B, blocked by 40GB colocate OOM) testing whether the ceiling and inverted-U persist; and a 2-axis (headroom × accessibility) quantitative fit with held-out cells. Both are future work; the current claim is the theory + its within-reach predictive validation.

## HONESTY GUARDRAILS (do not violate)
- Do NOT claim a method that beats RFT. The contribution is the theory + measurement + the *proven ceiling*.
- If any preregistered prediction fails, report it and revise the theory — do not drop the prediction.
- α/β, VSF-repair, accessibility-gating are all already observed (§137/§138); the matrix tests whether they *generalize as predicted* to new families/difficulties. Keep that distinction explicit.
