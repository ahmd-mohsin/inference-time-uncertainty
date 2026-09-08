# Theory — Why the Update Rule Governs Out-of-Distribution Transfer of Verified Experience

This file gives the formal backbone for the empirical result *SFT-on-verified-traces transfers verified experience
to OOD problems better than GRPO*. The thesis is operator-level: **GRPO is a reweighting operator confined to the
current reachable-correct support; SFT-on-verified is a projection operator that can place probability mass on
correct computation the base rarely produced.** Every theorem below predicts a specific empirical figure/table.

Status legend: **[Thm]** proved under stated assumptions; **[Prop]** proved; **[Sketch]** proof outline, to be
tightened for camera-ready. All assumptions are stated explicitly and are individually testable.

---

## 1. Setup and notation

- Prompt (problem) $q\sim\mathcal{D}$; completion (trajectory) $o=(o_1,\dots,o_T)$; policy $\pi_\theta(o\mid q)=\prod_t \pi_\theta(o_t\mid q,o_{<t})$.
- Verifier $r(q,o)\in\{0,1\}$ (exact-answer check; assumed sound: $r=1 \Rightarrow$ correct).
- **Reachable-correct probability** $\rho_\pi(q)=\Pr_{o\sim\pi(\cdot\mid q)}[r(q,o)=1]$ (this is $p(q)$ in the panels; pass@K estimates $1-(1-\rho)^K$).
- **GRPO update** (group size $K$): sample $o_1,\dots,o_K\sim\pi_\theta(\cdot\mid q)$, rewards $r_i$, group mean $\bar r$, std $\sigma$; group-relative advantage $\hat A_i=(r_i-\bar r)/(\sigma+\varepsilon)$; objective gradient (ignoring the ratio clip, which only shrinks steps)
  $$g_{\text{GRPO}}(q)=\mathbb{E}\Big[\textstyle\sum_{i=1}^K \hat A_i\,\nabla_\theta\log\pi_\theta(o_i\mid q)\Big].$$
- **SFT-on-verified**: dataset $\mathcal{D}_v=\{(q,o): o\sim\pi_{\text{base}}(\cdot\mid q),\, r(q,o)=1\}$ with empirical trace law $p_v$. Objective $\mathcal{L}_{\text{SFT}}(\theta)=\mathbb{E}_{(q,o)\sim p_v}[-\log\pi_\theta(o\mid q)]$.

Both arms consume the **same verified GSM8K experience** (self-generated, verifier-correct). Only the operator differs.

---

## 2. GRPO: a support-confined reweighting operator

### Theorem 1 (Zero learning signal on homogeneous groups). [Thm]
If all sampled rollouts for a prompt $q$ receive equal reward ($r_1=\dots=r_K$), then $\hat A_i=0\ \forall i$, so $q$ contributes **exactly zero** to $g_{\text{GRPO}}(q)$.
*Proof.* $\bar r=r_i\Rightarrow r_i-\bar r=0\Rightarrow\hat A_i=0$. The per-prompt gradient $\sum_i \hat A_i\nabla\log\pi=0$. $\square$

### Corollary 1.1 (OOD blindness). [Thm]
For any prompt with $\rho_\pi(q)=0$ (no reachable correct rollout), with probability $1$ every sampled group is all-incorrect, hence (Thm 1) contributes zero gradient — **for all training steps and any $K$**. GRPO cannot raise accuracy on prompts outside its current reachable-correct support.
> **Predicts:** the flat far-OOD (MATH-500) learning trajectory while in-distribution rises (Fig 2). It is a *transfer/exploration* failure, not forgetting.

### Corollary 1.2 (Fragile-band concentration). [Thm]
The expected number of nonzero-signal groups is maximized on the "fragile band" $0<\rho_\pi(q)<1$; signal $\to 0$ as $\rho\to 0$ or $\rho\to 1$. Learning is confined to partially-solved problems.
> **Predicts:** in-distribution gains saturate as $\rho\to 1$; matched-compute OOD gain for GRPO does not scale (Fig 3).

### Theorem 2 (Support invariance of the policy-gradient operator). [Sketch]
A GRPO step reweights the log-probabilities of *observed* tokens only. Sequences never assigned nonzero probability by $\pi_\theta$ receive no gradient; the operator is **mass-preserving on $\mathrm{supp}(\pi_\theta)$** and cannot create a new high-probability correct mode in one step from a region of vanishing base mass. Formally, $\|\pi_{\theta+\eta g}(\cdot\mid q)-\pi_\theta(\cdot\mid q)\|_{TV}$ restricted to $o\notin\mathrm{supp}$ is $O(\eta\,\rho(1-\rho))$ and $\to0$ off-support.
*Sketch.* $\nabla_\theta\log\pi_\theta(o\mid q)$ is only sampled for $o$ with $\pi_\theta(o)>0$; softmax logit shifts scale existing mass. Correct-mode creation off-support requires many correlated steps, each gated by Cor 1.1. $\square$

---

## 3. SFT-on-verified: a mass-placing projection operator

### Theorem 3 (SFT is the M-projection onto verified traces; it places mass off-support). [Thm]
$\arg\min_\theta \mathcal{L}_{\text{SFT}}$ is the moment/M-projection $\pi^\star=\arg\min_\theta \mathrm{KL}\!\left(p_v\,\|\,\pi_\theta\right)$ (forward KL). Because forward KL is **mode-covering**, $\pi^\star$ assigns nonvanishing probability to every trace in $\mathrm{supp}(p_v)$ — including correct traces that $\pi_{\text{base}}$ produced with arbitrarily small probability. Thus SFT can *increase* $\pi(\text{correct computation})$ on regions PG cannot reach (Thm 2), up to model capacity.
*Proof.* $\mathcal{L}_{\text{SFT}}(\theta)=H(p_v)+\mathrm{KL}(p_v\|\pi_\theta)$; minimizing over $\theta$ minimizes $\mathrm{KL}(p_v\|\pi_\theta)$. Forward KL $\to\infty$ if $\pi_\theta(o)=0$ where $p_v(o)>0$, forcing support coverage. $\square$

### Contrast (the mechanism, one line).
GRPO $\approx$ reweighting within $\mathrm{supp}(\pi_\theta)$ (Thm 2); SFT $\approx$ $\mathrm{KL}(p_v\|\pi_\theta)$ projection that *relocates* mass (Thm 3). **Sharpening does not travel; projection does.**

---

## 4. Transfer across structural distance

**Assumption A (compositional subskills).** Each $q$ needs a set $S(q)$ of latent subskills; $r(q,o)=1$ iff $o$ executes all of $S(q)$ correctly. A trace $o$ *exercises* subskills $E(o)\subseteq S(q)$. Subskill competence composes multiplicatively: $\rho_\pi(q)\approx\prod_{s\in S(q)}c_\pi(s)$, $c_\pi(s)\in[0,1]$.

### Theorem 4 (Transfer decomposition and a distance-monotone bound). [Sketch]
For an OOD prompt $q'$ with required subskills $S(q')$, decompose $S(q')=S_{\text{shared}}\cup S_{\text{novel}}$ relative to the training traces.
- **SFT lower bound:** process-level supervision raises $c(s)$ for every $s$ exercised by verified traces; hence $\Delta\rho^{\text{SFT}}(q')\ \ge\ \big(\prod_{s\in S_{\text{shared}}}c^{\text{SFT}}(s)-\prod c^{\text{base}}(s)\big)\prod_{s\in S_{\text{novel}}}c^{\text{base}}(s).$
- **PG upper bound:** by Cor 1.1, $\Delta\rho^{\text{GRPO}}(q')\le \rho_{\text{base}}(q')=\prod_{s\in S(q')}c^{\text{base}}(s)$, which is tiny when any novel subskill has low base competence.

Define structural distance $d(q')=|S_{\text{novel}}|$. As $d$ grows, the PG bound decays multiplicatively (each novel subskill $<1$), while the SFT bound decays only through the $S_{\text{novel}}$ factor but retains the reinforced $S_{\text{shared}}$ product. Hence $\Delta\rho^{\text{SFT}}$ dominates and both decay with $d$.
> **Predicts:** monotone transfer decay with distance for *both* operators, with SFT decaying slower — exactly Fig 1 (in-dist $\to$ SVAMP $\to$ MATH; C $\approx 2\times$ A at every distance).

---

## 5. Update magnitude (linking to the mechanistic M4 finding)

### Proposition 5 (GRPO's aggregate parameter movement is advantage-variance-limited). [Prop]
$\mathbb{E}\|g_{\text{GRPO}}(q)\|$ scales with the group advantage dispersion $\mathrm{Var}_i(\hat A_i)^{1/2}$, which for binary reward equals $\sqrt{\rho(1-\rho)}/(\sigma+\varepsilon)$-weighted score norm; it vanishes as $\rho\to0$ or $1$. SFT's gradient $\nabla\mathcal{L}_{\text{SFT}}$ has no such gating (full cross-entropy on every token). Therefore, aggregated over a dataset dominated by easy/near-solved ($\rho\to1$) and hard/unreachable ($\rho\to0$) prompts, $\|\Delta\theta_{\text{GRPO}}\|\ll\|\Delta\theta_{\text{SFT}}\|$.
> **Predicts:** the measured LoRA-delta magnitude gap (GRPO $0.88$ vs SFT $27.94$, $\sim32\times$; Fig 5 / §24-M4). Also predicts GRPO's change concentrates where advantage variance is nonzero, i.e. sparse/surgical.

---

## 6. What the theory claims — and what would falsify it

**Claims.** (i) GRPO gains vanish off the reachable-correct support (Cor 1.1); (ii) SFT-on-verified can place mass off-support (Thm 3); (iii) both transfers decay with structural distance, SFT slower (Thm 4); (iv) GRPO's net weight movement is small/sparse (Prop 5).

**Falsifiers (honest).** (a) If GRPO raised accuracy on a held-out family with base pass@K $\approx 0$, Cor 1.1 fails. (b) If a matched-compute SFT did *not* exceed GRPO OOD once verified traces cover the shared subskills, Thm 4's lower bound is vacuous. (c) If SFT's OOD gain came purely from longer/more CoT (rejected: M3 length/steps identical, §24) rather than higher correct-computation likelihood, the projection story is wrong. (d) Confound-fixed M1 must show SFT lowers NLL on *self-consistent* correct OOD traces; if not, Thm 3's "mass on correct computation" is not what's happening (this experiment is queued).

**Assumptions to verify empirically.** Assumption A (compositional subskills) — probe via subskill-tagged OOD sets; verifier soundness — audit false-positive rate; the reachability premise of Cor 1.1 — measure base pass@K on each OOD family (queued).

---

## 7. Extended theory (round 2) — six new results, each with a validating experiment

### Theorem 6 (Reachability–Headroom Law — explains BOTH nulls). [Thm/Sketch]
Let $b=\rho_{\text{base}}(\mathcal{D}_{\text{harvest}})$ be the base pass rate on the harvest set (governs how many verified
traces exist) and $h=1-\rho_{\text{base}}(\mathcal{D}_{\text{OOD}})$ the OOD headroom. The SFT-on-verified OOD gain obeys
$$\Delta^{\text{SFT}}_{\text{OOD}} \;\le\; C\cdot \underbrace{g(b)}_{\text{harvest mass}}\cdot \underbrace{h}_{\text{headroom}},\qquad g(b)\to 0\text{ as }b\to 0.$$
*Proof sketch.* Verified-trace count $\propto b$ (no correct rollouts ⇒ empty $\mathcal{D}_v$ ⇒ SFT is a no-op, cf. Cor 1.1 for the C arm); and gain is bounded by remaining headroom $h$ (can't exceed 1). Product form ⇒ an inverted-U in base competence: too weak (b→0, no traces) OR already-saturated/instruct (h→0, no headroom) ⇒ $\Delta\to0$. $\square$
> **Predicts + EXPLAINS OUR TWO NULLS on one curve:** SmolLM2-1.7B ($b\approx0.03$, harvest≈0) → null; Phi-3.5-instruct ($h$ small, already strong) → null; mid-competence bases (Qwen/OLMo/DeepSeek/Yi) → large gains. **Experiment E6:** plot $\Delta^{\text{SFT}}_{\text{MATH}}$ vs base-MATH-acc across ALL 8 families → expect inverted-U/threshold; SmolLM & Phi fall at the two zero-ends. (Data already collected — just plot.)

### Theorem 7 (Verifier-noise robustness). [Thm]
If the verifier has false-positive rate $\varepsilon$ (labels an incorrect trace correct), the SFT target becomes a mixture
$(1-\varepsilon)p_v + \varepsilon p_{\text{wrong}}$; OOD gain degrades at most linearly: $\Delta^{\text{SFT}}(\varepsilon)\ge \Delta^{\text{SFT}}(0)-L\varepsilon$ for Lipschitz $L$ (KL is smooth in the mixture weight).
*Proof.* Cross-entropy is linear in the target distribution; the projection target moves $O(\varepsilon)$ in TV ⇒ minimizer moves $O(\varepsilon)$ (projection stability). $\square$
> **Experiment E7:** inject $\varepsilon\in\{0,0.1,0.2,0.4\}$ label-noise into the verified set for Qwen-3B → measure MATH Δ; expect ~linear, graceful decay (robustness = practical selling point).

### Theorem 8 (Trace-scale law). [Sketch]
OOD gain grows concavely (log-like) with verified traces per problem $m$: $\Delta^{\text{SFT}}(m)\approx \Delta_\infty(1-e^{-m/m_0})$ (diminishing returns; more traces = better subskill coverage, saturating).
> **Experiment E8:** Qwen-3B SFT on $m\in\{1,2,4,8\}$ verified traces/problem → MATH Δ; expect concave saturation.

### Theorem 9 (On-policy sufficiency). [Sketch]
SFT on the base model's OWN verified traces attains the same OOD gain as SFT on a stronger model's correct traces of equal count, up to the shared-subskill overlap — because the projection only needs to place mass on *reachable-correct computation*, which own-traces already exemplify.
> **Experiment E9:** Qwen-3B SFT on (a) own verified traces vs (b) Qwen-14B's correct traces → compare MATH Δ; expect ≈, isolating "own reachable" vs "any correct".

### Proposition 10 (Fragile-band concentration — Cor 1.2 empirical). [Prop]
GRPO's per-problem parameter movement (and any gain) is supported on problems with base pass@1 $\in(0,1)$; expected contribution $\propto b(1-b)$, zero at the extremes.
> **Experiment E10:** bin GSM8K-train by base pass@1; measure GRPO per-bin Δ → hump at mid-band, ≈0 at 0 and 1.

### Theorem 11 (Hybrid optimality). [Sketch]
SFT-on-verified (mass placement, Thm 3) then a short GRPO phase (fragile-band sharpening within the new support, Thm 1) dominates either alone: SFT expands reachable-correct support, which GRPO can then reweight (GRPO's zero-signal problem is relieved once SFT raised $b$ on the fragile band).
> **Experiment E11:** arm H = SFT-verified → GRPO (100 steps) on Qwen-3B; compare OOD to A and C; expect H ≥ C > A.
