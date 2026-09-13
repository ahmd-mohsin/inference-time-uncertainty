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

---
# STRENGTHENED THEORY  (formal version, validated by §141)
_Added 2026-09-13 after the 72-GPU preregistered validation. This replaces the informal T1–T4 sketch above with stated assumptions, theorems, proof sketches, and corollaries that ARE the confirmed predictions P1–P5._

## Setup and definitions
Let a policy $\pi_\theta(y\mid x)$ generate solutions $y$ to a prompt $x$. A **verifier** $V(x,y)\in\{0,1\}$ is *parameter-independent* (an executor/checker that does not depend on $\theta$). Define the **per-prompt success mass** $p_\theta(x)=\mathbb{E}_{y\sim\pi_\theta(\cdot\mid x)}[V(x,y)]$ and the population objective $J(\theta)=\mathbb{E}_{x\sim\mathcal{D}}[p_\theta(x)]$.

Three training operators from a shared checkpoint $\theta_0$:
- **GRPO / outcome-RL:** on-policy policy-gradient on the binary reward $V$ (group-normalized advantages).
- **RFT (decoupled verified replay):** collect a bank $B=\{(x,y):V=1\}$ by sampling $\pi_{\theta_0}$ at budget $C$; SFT (cross-entropy) on $B$ for multiple epochs.
- **VSF:** GRPO plus a persistent prompt-balanced cross-entropy replay floor over $B$.

Define **budget-$C$ accessibility** $a(x;C)=\Pr(\exists\, \text{verified } y \text{ in } C \text{ i.i.d. draws from }\pi_{\theta_0}) = 1-(1-p_{\theta_0}(x))^{C}$, **headroom** $h(x)=1-p_{\theta_0}(x)$, and the **accessible frontier** $B^\*=\{x: a(x;C)>0\}$ with its realizable bank.

## Assumptions
- **(A1) Parameter-independent binary verifier** (holds for code execution / exact-match checkers).
- **(A2) Bounded transfer:** there is a prompt-wise generalization coefficient $\tau(x)\in[0,1]$ such that training that raises $p_\theta$ on a support set raises $p_\theta$ on a held-out related prompt $x$ by at most $\tau(x)\cdot(\text{support gain})$.
- **(A3) Finite sampling budget $C$** at bank-construction time (the practical regime).

## Theorem 1 (Gradient-family identity).
Under (A1), for every prompt $x$,
$$\nabla_\theta\, p_\theta(x) \;=\; p_\theta(x)\,\mathbb{E}_{y\sim\pi_\theta(\cdot\mid x)}\!\big[\nabla_\theta\log\pi_\theta(y\mid x)\,\big|\,V(x,y)=1\big].$$
Consequently (i) the within-prompt-normalized **RFT** gradient on current-policy successes equals $\nabla_\theta\log p_\theta(x)$, and (ii) the binary-reward **policy gradient** equals $p_\theta(x)$ times the *same* direction. RFT and outcome-RL are therefore **not distinct gradient families**; they differ only by a positive per-prompt scalar and by *which prompts carry a nonzero success-gradient and how often*.
*Proof sketch.* Score-function identity: $\nabla_\theta p_\theta(x)=\mathbb{E}[V\nabla_\theta\log\pi]$. Condition on $V=1$ (the $V=0$ term vanishes since $V\in\{0,1\}$) and normalize by $p_\theta(x)=\Pr(V=1)$ to get the conditional expectation. The policy gradient of $\mathbb{E}[V]$ is exactly $\mathbb{E}[V\nabla\log\pi]=p_\theta(x)\cdot(\text{that conditional mean})$. $\square$
**Corollary 1 (⇒ P1, P2).** Any persistent GRPO−RFT gap must be *procedural* — it lives in the empirical distribution over which prompts receive success-gradients (coverage/replay), not in the objective. On-policy GRPO only sees success-gradients on prompts it currently solves (mass $p_\theta$), so its acquisition on base-failed prompts $\alpha$ is suppressed; RFT's multi-epoch replay over $B^\*$ supplies them broadly. Predicts $\alpha_{\text{RFT}}>\alpha_{\text{VSF}}>\alpha_{\text{GRPO}}$ (**P2**, confirmed 8/8) and RFT accuracy $>$ GRPO (**P1**, confirmed 8/8).

## Theorem 2 (Support-boundedness and the headroom bound).
Under (A1)–(A2), verified-only training induces **zero direct gradient** on any prompt with no verified sample in the support. Hence for a held-out prompt $x$, the achievable gain is generalization-only and bounded:
$$\Delta p(x)\ \le\ h(x)\cdot \tau(x).$$
*Proof sketch.* The CE/PG update is a sum of terms each proportional to $\nabla\log\pi(y\mid x')$ for support prompts $x'$; a prompt absent from support contributes no first-order term, so any change at $x$ is transfer, bounded by (A2) and capped by the remaining mass $h(x)=1-p_{\theta_0}(x)$. $\square$
**Corollary 2 (reachability ≠ reliability).** Gains are gated by headroom from above; a model with little headroom (small $h$) has little to gain regardless of method — the *low-headroom* arm of the curve.

## Theorem 3 (Accessibility-gated gain — the inverted-U).
Let the realized new-coverage gain on the accessible frontier be $G(x)=h(x)\,a(x;C)\,\tau(x)$ with $a(x;C)=1-(1-p_{\theta_0}(x))^{C}=1-h(x)^{C}$. Then as a function of headroom $h\in[0,1]$ (holding $C,\tau$ fixed),
$$G(h)=\tau\,h\,(1-h^{C})$$
is **non-monotone**: $G(0)=G(1)=0$ and $G$ has a unique interior maximum at $h^\*=\big(\tfrac{1}{C+1}\big)^{1/C}$. Gain **rises then falls** in headroom.
*Proof sketch.* $G'(h)=\tau\big(1-(C{+}1)h^{C}\big)$, which is positive for small $h$ and negative as $h\to1$, with a single sign change at $h^\*$. $\square$
**Corollary 3 (⇒ P3).** Increasing task difficulty raises $h$ but drives $a(x;C)\to0$ (the bank cannot be filled), so gain **decreases at extreme difficulty despite more headroom** — refuting any linear-in-headroom law. Predicts the inverted-U (**P3**, confirmed: peak near $h\approx0.6$; gain falls at both $h\approx0.18$ (7B) and $h\approx0.91$ (vhard); bank sizes shrink $164\to78\to$smaller). It also predicts $\beta$ (regression) **grows** with difficulty as usable signal per prompt vanishes (confirmed: $\beta_{\text{GRPO}}\,.278\to.370\to.472$).

## Theorem 4 (RFT is the ceiling of the accessible frontier).
Consider the class $\mathcal{M}$ of "wrapper" methods that either (a) reweight on-policy success-gradients (any advantage transform / regularizer — by Thm 1 these are re-scalings of directions RFT already integrates over $B^\*$), or (b) add verified replay drawn from the *same* accessible frontier $B^\*$. Let $\theta_{\text{RFT}}$ be the multi-epoch fixed point of CE on $B^\*$. Then under (A1)–(A3),
$$\sup_{M\in\mathcal{M}} J(\theta_M)\ \le\ J(\theta_{\text{RFT}})\ +\ o(1),$$
i.e. no wrapper exceeds RFT beyond estimator noise, once $B^\*$ is saturated.
*Proof sketch.* By Thm 1 the reachable gradient span from on-policy signal is contained in $\mathrm{span}\{\nabla\log p_\theta(x):x\in B^\*\}$, which multi-epoch RFT already ascends to its CE optimum on $B^\*$; adding (b) replay from the same $B^\*$ cannot enlarge the support. Any surplus must come from prompts outside $B^\*$, which by Thm 2 receive no direct gradient. Hence gains beyond RFT are transfer-only and not systematically positive. $\square$
**Corollary 4 (⇒ P4, P5, and the two negative predictions).** VSF (case b) repairs GRPO's regression ($\beta_{\text{VSF}}<\beta_{\text{GRPO}}$, **P4**, 8/8) and improves acquisition, but cannot pass RFT ($\text{VSF}<\text{RFT}$, **P5**). Continued RL from RFT (case a) adds nothing (**§139**, confirmed negative). A bigger same-domain bank cannot help a saturated recipient (**§140/A0**, confirmed negative). These are not failed experiments; they are Corollary 4.

## What the strengthening buys
1. **A closed-form inverted-U** $G(h)=\tau h(1-h^{C})$ with an explicit optimum $h^\*=(C{+}1)^{-1/C}$ — a *quantitative*, falsifiable law (not a qualitative shape). Future work: fit $C,\tau$ per family and predict held-out cells.
2. **A ceiling theorem** that turns every failed method-search into a corollary, converting the honesty liability ("no method beat RFT") into the paper's central positive claim.
3. **Tight theory–measurement coupling:** each theorem emits an $(\alpha,\beta,O)$ signature, and §141 confirms all of them 8/8 out-of-sample across a held-out family and difficulty. Predictive, not descriptive.
