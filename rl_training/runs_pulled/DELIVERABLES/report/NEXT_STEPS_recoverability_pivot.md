# What's Next for the Paper — Pivot to *Recoverability-Constrained RLVR*

Status as of 2026-08-19. Based on a NeurIPS-level review of the current manuscript
(`coverage_methodology_neurips.tex`) against the literature through Aug 19, 2026.

**Verdict on the current draft:** good instinct + a real phenomenon, but **borderline/reject** as-is
because (1) a foundational object mismatch (theory analyzes *modes*, implementation constrains *one
exact sequence*), (2) the novelty is now largely occupied by very recent work, (3) the theory
overclaims "irreversibility," and (4) the headline gains are tiny and near-saturation.

**The move:** reframe from "base-anchored *trace* preservation / coverage" to
**Recoverability-Constrained RLVR** — preserve *reasoning-mode mass* (and ideally *routes*), and
prove a finite-rollout **recoverability certificate**. Keep current results as *pilot evidence*, do
not build the final paper on them.

---

## 1. The foundational bug to fix first (object mismatch)

- Theory: a reasoning **mode** `M` is a *set of trajectories*; extinction argued when
  `p_θ(M|q) ≪ 1/K` (won't appear in an on-policy group of `K`). **Sensible.**
- Implementation + evidence: constrains/measures the probability of **one exact sequence**
  `π_θ(y_q|q)`, e.g. `log π₀ = −254 → log π_RL = −320`.
- **Problem:** an exact 300–1000-token trace has astronomically small prob even when its *strategy*
  is likely — `e^{−254} ≪ 1/256` already for the base. So "pushed below the 1/K sampling floor" is
  **mathematically wrong**: the 1/K floor applies to the *mass of an event/mode*, not one verbatim
  sequence.
- **Reviewer attack (valid):** "lower likelihood of one exact trajectory does not establish loss of
  the corresponding reasoning mode." The 49/49 and 150/150 results show **distributional drift /
  witness-likelihood contraction**, NOT 100% mode suppression.
- **Fix now:** relabel the 49/49 result as *"witness likelihood contraction"* everywhere; stop
  claiming mode extinction from single-trace likelihoods.

---

## 2. Novelty is now crowded — what to delete / soften

Recent work already occupies pieces of the current story. **Delete** the broad claim that "all
competing methods are just bounded reward shaping `wA⁺` and therefore cannot solve this." Some are
substantially more sophisticated:

| Work | What it already does | Implication for us |
|---|---|---|
| **Yue et al.** (2504.13837) | establishes the pass@k inversion (RL ↑pass@1, base catches at large k) | our motivating phenomenon is *known* — not a contribution |
| **BBG** (2606.15455, Jun'26) | diversity collapse = boundary/overtraining; Bayesian Boundary Gating redirects away from saturated prompts | prompt-level mitigation baseline |
| **PBA** (2607.20543, Jul'26) | **Per-Problem Base Anchoring** — anchors risky prompts to frozen base; notes base-replay/KL as natural stronger variant; **+4.7 pt pass@256 over GRPO, 3 seeds** | *very close to our narrative*; sets the bar (our +0.3–0.7 is far below) |
| **DPH-RL** (2509.07430) | initial-policy samples + mass-covering divergences as rehearsal for coverage | baseline |
| **DyJR** (2603.16157) | historical trajectories as distributional anchor, JS regularization for diversity | baseline |
| **UCPO** (2605.00365) | conditional uniformity penalty among correct solutions | baseline |
| **PKPO** (2505.15201) | direct pass@k policy gradients | baseline |
| **RiskPO / Poly-EPO** | rare-path/entropy targeting; set-level diverse-strategy objectives | baseline |
| **Verifier-Induced Support Reshaping** (2608.00220, Aug'26) | defines *effective rewardable support* under fixed rollout budget; shows one verifier's RL can make later-verifier successes too rare for subsequent on-policy RL; **causal evidence it concentrates at response openings** | **"coverage necessary for continued RL" is no longer ours** — but points to the route-level method |
| **Pass@k diagnostic** (2511.16231) | direct pass@k optimization still yields vanishing signal where exploration matters | *supports* our support-blindness theorem |
| (2506.14245) | questions whether raw pass@k faithfully measures reasoning | motivates separating support/pass@k/correctness |

**Consequence:** "coverage is necessary for continued RL" (our R4) can no longer be the *primary*
conceptual contribution. It becomes supporting evidence for the recoverability framing.

---

## 3. The new conceptual object — Recoverability

Define a mode `M_{qm} ⊆ Y` (a *set* of trajectories = a strategy family), not a sequence. Its mass:

```
p_{θ,qm} = Pr_{y~π_θ(·|q)}[ y ∈ M_{qm} ]
```

**Recoverability** under a future budget of `K` rollouts:

```
R_K(M_{qm}; π_θ) = 1 − (1 − p_{θ,qm})^K
```

Operational meaning (the whole point):
- `p = 1e−2, K=256 → R ≈ 0.924` (discoverable)
- `p = 1e−4, K=256 → R ≈ 0.025` (effectively extinct for on-policy RL, even though support > 0)

This replaces vague "coverage." Explicitly separate five distinct quantities the paper currently
blurs: **support**, **effective (recoverable) support**, **correct-answer probability**,
**strategy diversity**, **pass@k**.

---

## 4. The new constraint — on mode mass / recoverability (not a sequence)

Replace `π_θ(y_q|q) ≥ α·π₀(y_q|q)` with a **mode-mass** floor:

```
p_θ(M_{qm}|q) ≥ α · p₀(M_{qm}|q)
```

or directly the deployment/training quantity (stronger):

```
R_K(M_{qm}; π_θ) ≥ γ · R_K(M_{qm}; π₀)
```

Meaning: *RL may sharpen whatever modes it wants, but cannot make a verified strategy effectively
undiscoverable under the target compute budget.* This is qualitatively distinct from:
- PBA → protects **prompts**
- global KL → protects **distributions**
- UCPO → **uniformity among sampled correct solutions**
- **Ours → identified rewardable modes at an explicit future sampling budget.**

---

## 5. New theory to prove (all more defensible than current claims)

### 5a. Recoverability guarantee (the headline theorem)
If the floor holds `p_θ(M|q) ≥ α·p_ref(M|q)`, then after `K` rollouts:
```
Pr[observe M] ≥ 1 − (1 − α·p_ref(M|q))^K
```
and after `N` independent RL groups of size `K`:
```
Pr[rediscover M within N groups] ≥ 1 − (1 − α·p_ref(M|q))^{KN}
```
→ finite-compute recoverability guarantee (not just narrative).

### 5b. Jensen certificate — turns replay into a *certificate*, not "replay"
Let `ν_m(y) = π₀(y | y∈M_m, q)` (base conditioned on mode). By Jensen:
```
log p_θ(M_m|q) ≥ log p₀(M_m|q) + E_{y~ν_m}[ log( π_θ(y|q) / π₀(y|q) ) ]
```
So maintaining the **teacher-forced** quantity
```
E_{y~ν_m}[ log π_θ(y|q) − log π₀(y|q) ] ≥ log α    ⟹    p_θ(M_m|q) ≥ α·p₀(M_m|q)
```
Teacher-forcing banked traces becomes a **Monte-Carlo certificate for strategy-level mass
preservation** (with multiple traces/mode + a confidence LCB). Candidate name:
**Certified Mode-Support Policy Optimization.**

### 5c. Fix the extinction theorem — drop "irreversible"
Current `‖E[g_y]‖ = O(π_θ(y|q))` + "irreversible/cannot be restored" is too strong (GRPO group
coupling, parameter sharing can indirectly move a missing mode, `‖∇log π‖` not bounded). Replace with
a **sample-supported estimator** class:
```
ĝ = Σ_{i=1..K} c_i(Y_{1:K}) ∇_θ log π_θ(Y_i|q),   |c_i| ≤ C
Pr[∃ i: Y_i ∈ M] = 1 − (1−p_M)^K ≤ K·p_M
```
Claim (defensible): when `p_M ≪ 1/K`, the probability of getting **any direct mode-specific gradient
observation** is ≤ `K·p_M` → finite-rollout on-policy algorithms are **support-blind** to
sufficiently rare modes. Does NOT claim the net never accidentally raises `M` via parameter sharing.

---

## 6. Better method — preserve **routes**, not full solutions
Support-reshaping (2608.00220) shows critical distributional change concentrates in the **first
tokens**, and forced openings causally change downstream searchability. So split a solution
`y = (z, c)`: `z` = early **reasoning route / strategy prefix**, `c` = free continuation. Protect
`p_θ(z|q)` instead of `p_θ(y|q)`.
- RL stays free to improve reasoning *inside* a preserved route.
- A base substitution-strategy and an RL improved-substitution derivation count as *preserved* even
  with completely different tokens — the current exact-trace objective cannot tell these apart.
- Slogan: **preserve entry into valuable reasoning basins; do not preserve the trajectory inside.**

---

## 7. Experiments to run, in priority order

> Overarching: **move the headline off the saturated Olympiad fragile band** (0.9832 vs 0.9900 is too
> close to 1.0 and too small). Need a regime where GRPO destroys *multiple points* of reachable
> support and the method recovers *multiple points*. Omni-MATH / harder boundary subsets. PBA already
> shows multi-point Omni-MATH gains, so that's the bar.

**E0 — Freeze current work as pilot.** Stop spending compute on exact-trace `expSR`/`expPROJ` as the
final method. Relabel 49/49 as "witness likelihood contraction." Keep round-2 R4 as preliminary.

**E1 — Operationalize modes + recoverability (definitions + measurement).**
Implement `p_{qm}(θ)`, `R_K(q,m;θ)`. Report support / effective support / correct-prob / diversity /
pass@k as *separate* curves. Guard against pass@k rewarding accidental correct answers.

**E2 — Multi-witness strategy bank.** For each fragile problem, sample 128–1024 verified base
solutions offline; **cluster by strategy** (not surface text): e.g. substitution vs geometric,
direct algebra vs induction, decomposition. Start with a controlled subset where strategies are
reliably identifiable. Report cluster stability + human/LLM labeling agreement. Bank becomes
`B = {q, M_1..M_J, Y_{q1}..Y_{qJ}}`.

**E3 — Mode-mass certificate.** Estimate `Δ_{qm} = E_{y~ν_{qm}}[log π_θ − log π₀]` from multiple
traces; enforce a statistically calibrated **lower confidence bound** `LCB(Δ_{qm}) ≥ log α`. This is
the "certificate," not a regularizer.

**E4 — Adaptive primal-dual (replace fixed μ=0.5).**
```
max_θ J_RLVR(θ)   s.t.   Δ_{qm}(θ) ≥ log α
λ_{qm} ← [ λ_{qm} + η_λ (log α − Δ_{qm}) ]_+
```
Only *endangered* modes get preservation pressure — cleaner than a global hand-set penalty.

**E5 — Route-level method as the main algorithm.** Compare three preservation granularities:
(a) exact-trace (current), (b) full strategy-cluster mass, (c) **prefix/route**. Hypothesis: route
preservation gives the best pass@1 / coverage Pareto frontier (protects exploration without
behavior-cloning). Motivated directly by the causal opening-token evidence.

**E6 — Baselines reviewers will demand.** GRPO; global KL; DPH-RL (forward/JS ref reg); **PBA**;
**BBG**; **UCPO**; PKPO; DyJR/replay. If the full set is infeasible, prioritize **PBA, DPH-RL, UCPO,
BBG** (closest to our claims). Vanilla-GRPO-only will not survive review.

**E7 — Seeds + stats.** ≥3 independent *training* seeds for the main comparison (prompt bootstrap
only captures eval-set uncertainty, not RL training stochasticity). Report seed-level uncertainty +
paired per-problem analysis. Stop calling sub-1-point effects "stable" when CIs cross 0.

**E8 — The killer *causal* experiment (mechanism figure).** Take a known correct strategy of
controlled mass `p`. Artificially suppress its route probability across checkpoints. Measure the
probability that GRPO / PKPO / UCPO / RiskPO-like objectives rediscover it as `K·p` crosses below 1.
Then add the off-policy constraint and show it *stays trainable* below that boundary. If a **phase
transition at `K·p ≈ 1`** appears, that's the mechanism figure — validates the theorem causally
instead of endpoint-likelihood correlation.

**E9 — Pareto headline.** Show the method **strictly shifts the frontier outward** on
`pass@1 vs recoverable-mode-count vs future-RL-improvement`.

---

## 8. Target framing

- **Old title:** *Coverage Preservation in RLVR as an Off-Policy Constraint*
- **New title (pick one):**
  - *Recoverability-Constrained RLVR: Preserving Rewardable Reasoning Modes Under Finite Compute*
  - *Never Lose a Solvable Mode: Certified Recoverability for RLVR*

**Thesis chain:**
1. The relevant resource in RLVR is not entropy but **finite-budget rewardable support**.
2. On-policy estimators become **support-blind** below `p ~ 1/K`.
3. We identify verified reasoning **modes/routes** and impose **one-sided mode-mass constraints**.
4. `p_θ(M) ≥ α·p_ref(M) ⟹ R_K(M;θ) ≥ 1 − (1 − α·p_ref(M))^K` (certified recoverability).
5. Empirics: GRPO → pass@1 ↑, recoverable-mode-count ↓; PKPO/UCPO/BBG/PBA mitigate pieces but still
   lose rare modes; **ours** → pass@1 ↑, high-k ↑, certified mode survival ↑, continued-RL sample
   efficiency ↑; and the Pareto frontier shifts out.

---

## 9. What to reuse from existing work (not thrown away)
- Round-1/round-2 full-FT pipeline, HF-checkpoint death-proofing, sharded pass@k eval harness.
- Oat-Zero R1/R2 difficulty-resonance result (crossover is intermediate-difficulty) — good motivation.
- 49/49 & 150/150 → reframed as *witness likelihood contraction* (drift evidence).
- Round-2 R4 fork result → *preliminary* support for "coverage convertible by continued RL."
- expSR/expPROJ → *pilot* implementations; route/cluster/certificate versions become the main method.
