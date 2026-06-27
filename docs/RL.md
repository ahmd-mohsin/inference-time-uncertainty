# Making RL Generalize, Not Just Sharpen: Expanding the Reasoning Boundary

## 0. One-line thesis

Standard RLVR improves **pass@1** by concentrating probability on reasoning paths the base
model already samples, but it **shrinks pass@k** (the reachable solution set). We aim to
make the RL gradient *expand support* — sharpen the dominant correct mode **and** protect /
grow the rare-but-valid tail — using **inter-rollout diversity (semantic distance between
trajectory embeddings)** as the signal and **off-policy harvesting of the model's own
rare high-k successes** as the support-expansion mechanism, without ProRL-scale brute-force
training.

---

## 1. The problem, precisely (why RL sharpens)

Grounding paper: **Yue et al. 2025, "Does RL Really Incentivize Reasoning Capacity in LLMs
Beyond the Base Model?"** (arXiv:2504.13837). Established findings:

1. **RL sharpens but doesn't expand.** On pass@k curves, RLVR beats base at small k (k=1)
   but the **base overtakes at large k** (k=128–1024) across every benchmark/model family.
   RL improves sampling efficiency on already-solvable problems while *narrowing* the set of
   solvable problems (e.g. Minerva-32B: base solves ~9% more at k=128).
2. **The paths RL finds were already in the base.** Perplexity analysis: RL outputs sit in
   the *low-perplexity (easy-to-sample) region of the base model's own distribution*. RL
   redistributes mass onto pre-existing paths; it does not create new ones.
3. **Distillation expands the boundary; RL doesn't.** Distillation pushes pass@k *above*
   base because it injects **off-policy** paths from a stronger teacher.

### Why — the mechanism (from §2.1, §4.1, §4.5 of the paper)

- Objective `J(θ)=E_{y∼π_θ}[r]`, `r∈{0,1}` (exact match). Policy gradient *maximizes
  log-prob of correct rollouts, minimizes incorrect.*
- **On-policy ⇒ redistribution only.** PPO/GRPO learn solely from the current policy's own
  samples. A path the base emits with prob ≈ 0 is never sampled → never gets gradient → RL
  *cannot* add it. RL is structurally a mass-redistribution operator over existing support.
- **Binary reward ⇒ novelty is uncredited.** Every correct rollout gets r=1 regardless of
  *how* it solved it, so gradient flows to the **already-most-probable** correct path,
  collapsing onto one mode and starving rare valid alternatives. (Fig 5: RL piles frequency
  at accuracy≈1.0, *and* raises frequency at accuracy 0 — some problems become unsolvable.)
- **It is NOT just lowered entropy.** §4.5 ablation: raising the RL model's temperature to
  match the base model's entropy **still** underperforms base at large k. The paths are
  *pruned*, not merely down-weighted — you cannot reheat them back. This is the crucial
  result: the fix must *expand support*, not just *preserve entropy*.

**Fix conditions implied by the diagnosis** — to expand rather than sharpen we must break at
least one of:
(a) on-policy-ness (get gradient on paths not currently sampled),
(b) binary reward (credit novelty/coverage, not just correctness),
(c) winner-take-all mode collapse (protect rare correct modes from the dominant one).

---

## 2. Complete literature: what's been tried to make RL generalize

### 2.1 Entropy / exploration preservation (slows collapse; stays on-policy)

- **Cui et al. 2025, "The Entropy Mechanism of RL for Reasoning LMs"** (arXiv:2505.22617).
  Derives `R = -a·e^H + b` — downstream performance is *traded against* policy entropy.
  Entropy collapse is driven by the **covariance between action probability and logit
  change** (∝ advantage under PG); the covariance stays positive → entropy decreases
  monotonically. Fix: **Clip-Cov / KL-Cov** — clip or KL-penalize the high-covariance
  tokens to keep exploration alive.
- **Wang et al. 2025, "Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective
  RL"** (arXiv:2506.01939). Only ~20% of tokens — high-entropy **"forking tokens"** — drive
  reasoning; restricting policy gradient to them matches/beats full-gradient (Qwen3-32B:
  +11.04 AIME'25). **This is branch-point credit assignment** (cf. our Direction 14A): the
  decisions that matter are the contested forks.
- **Takeaway:** these *slow* collapse and concentrate gradient where it matters, but remain
  **on-policy** — they cannot add paths outside current support. They mitigate (c), touch
  (b), but never (a).

### 2.2 Prolonged training + KL control + task diversity — the rebuttal to "RL only sharpens"

- **Liu et al. 2025, "ProRL: Prolonged RL Expands Reasoning Boundaries"**
  (arXiv:2505.24864). Strongest claim that RL *does* expand: "novel reasoning strategies
  inaccessible to base models, even under extensive sampling… scenarios where base models
  fail entirely." Mechanism: **KL divergence control + reference-policy resetting + diverse
  task suite + very long training.** Key condition: expansion correlates with **base-model
  competence and training duration** — i.e. it works only when the base is competent enough
  and you train long enough.
- **This is the elephant.** Our contribution cannot be "RL can expand" (ProRL claimed it).
  It must be **"RL expands *efficiently and reliably* via novelty-reward + off-policy
  tail-harvesting, without ProRL's brute-force prolonged-training + reference-reset
  machinery."** Position against ProRL as the efficient alternative.

### 2.3 Exploration-budget allocation

- **Knapsack RL** (arXiv:2509.25849). Not a diversity reward — an *allocation* method:
  frame per-task rollout budget as a knapsack, give more rollouts to high-learning-signal
  (hard) tasks, fixing GRPO's **zero-gradient problem** on always-pass / always-fail tasks
  (+20–40% non-zero gradients; 2× compute efficiency). Relevant because the support
  expansion we want happens precisely on the **hard problems where the gradient is currently
  zero** — allocation is complementary to our reward change.

### 2.4 Representation-space diversity (the closest prior to our measurement)

- **Sun et al. 2025, "Representation-Based Exploration..."** (arXiv:2510.11686). A diversity
  bonus **derived from the pre-trained model's hidden states** → +50% verifier efficiency at
  inference; **pass@80 = GRPO's pass@256 (3× sample efficiency)**; argues "deliberate
  exploration with the right notion of diversity is a path to discovery beyond sharpening."
  **This essentially already did representation-space diversity** — so it is our *tool*, not
  our novelty. We differentiate on *how it is used* (group-relative, applied to correct
  rollouts, fused with off-policy harvesting), not *whether* representation diversity helps.

### 2.5 Off-policy / distillation to inject new paths

- The Yue paper's own contrast: distillation expands pass@k because it is **off-policy**
  (teacher tokens carry paths the base couldn't sample). RL+SFT hybrids, success replay,
  STaR / ReST-style self-training, and teacher-trace injection all live here — but mostly
  studied as *separate SFT*, **not as an RL objective that expands support.** This is the
  least-explored quadrant and our strongest opening.

---

## 3. Our ideas and honest positioning

| Idea | Closest prior work | Novelty |
|---|---|---|
| **#1 Vector reward with a novelty term** — reward a correct rollout *more* when it is distinct from the other correct rollouts in its group | Entropy preservation (Cui); rep-exploration (Sun) | **Partial.** Entropy/representation bonuses exist, but a **per-group novelty reward that specifically protects rare *correct* modes** ("reward correct-AND-distinct") is sharper than generic entropy and not standard. |
| **#2 Semantic distance between trajectory embeddings** as the diversity metric (group-relative pairwise distance in representation space) | Sun et al. (hidden-state diversity bonus → 3× pass@k) | **Weakest / most done.** Use it as the **measurement tool** inside #1/#3, *not* as the headline novelty. Differentiator is *group-relative, correct-rollouts-only*, not *whether* representation diversity helps. |
| **#3 Off-policy self-distillation of the model's own high-k tail** — sample at large k, harvest the rare correct paths the base finds, train on them off-policy to permanently widen support | Distillation (Yue); STaR / ReST self-training | **Strongest / least crowded.** "Harvest the model's own rare high-k successes and train off-policy to widen support" is not a well-trodden RL objective. Clean mechanism story below. |

**Strategic read:**
- **#2 alone is not a paper** — fold it into #1/#3 as the metric.
- **#3 is the sharpest**, because Yue's perplexity analysis *proves* the correct rare path is
  in-support but low-probability. Harvesting those and training off-policy moves a low-prob
  correct path to high-prob **without pruning the others** — directly attacking the cause of
  sharpening (on-policy redistribution).
- **Frame against ProRL:** same goal (boundary expansion), but efficient and mechanistic
  rather than brute-force prolonged training.

---

## 4. Methodology — how we build on them

The unifying objective: **sharpen the dominant correct mode while protecting and growing the
rare-but-valid tail.** Three components, each tied to a fix-condition from §1.

### 4.1 Component A — Group-relative novelty reward (fixes b + c)

Augment the scalar verifiable reward with a diversity term computed *within the GRPO group*
and *only among correct rollouts*:

```
r_i = correct_i · ( 1 + λ · novelty_i )
novelty_i = mean pairwise semantic distance of rollout i to the OTHER correct rollouts,
            in representation space (trajectory embedding; §4.4)
```

- Restricting the bonus to **correct** rollouts is the key difference from entropy bonuses:
  we are not rewarding random diversity (which entropy does, including diverse *wrong*
  paths), we are rewarding *correct paths that are distinct from the consensus correct
  path* — i.e. protecting the rare valid mode the standard gradient would crush.
- `λ` controls the sharpen↔spread tradeoff. Connects to Cui's entropy mechanism: instead of
  penalizing high-covariance tokens post-hoc, we make the *reward itself* resist collapse.

### 4.2 Component B — Off-policy tail harvesting (fixes a — the support ceiling)

Standard RL never gets gradient on unsampled paths. So once per K steps:
1. Sample the **base / current policy at large k** (e.g. k=64–256) on the hard problems
   (those with low pass@1 but pass@k > 0 — exactly Yue's "in-support but low-prob" set).
2. **Harvest** the rare correct rollouts (the tail the on-policy gradient would never
   reinforce).
3. Add them as **off-policy targets** (SFT-style log-likelihood, or importance-weighted into
   the RL batch). This injects the path *into the high-probability region* the way
   distillation does — but the teacher is the model's *own* tail, so no external model.
- This is the mechanism that turns "redistribution within support" into "support
  expansion," because the harvested path was effectively at p≈0 under the on-policy
  distribution and now receives direct gradient.

### 4.3 Component C — Hard-problem budget targeting (fixes the zero-gradient problem)

Borrow Knapsack-RL's insight: concentrate the large-k harvesting (B) and the novelty bonus
(A) on the **hard problems where the boundary actually needs expanding** (pass@1 ≈ 0,
pass@k > 0). Easy problems are already solved; expansion there is wasted. This makes the
method *efficient* — the answer to ProRL's brute force.

### 4.4 Measuring "semantic distance between trajectories" (the §2.4 tool, used carefully)

- Embed each rollout's reasoning (mean-pooled hidden states, or a sentence-embedding of the
  reasoning text) → one vector per rollout.
- novelty = mean cosine / Euclidean distance to the other correct rollouts in the group.
- **Caveat from our own prior negative result** (topological diversity, see
  `topological_persistence/METHODOLOGY.md`): raw high-dim hidden-state distances *concentrate*
  and can be near-uninformative; normalize per-dim and validate the metric separates
  genuinely different approaches before trusting it. Use approach-level embeddings, not raw
  last-layer means, if concentration shows up.

---

## 5. The falsification experiment (cheap, decisive, before scale)

**Claim to test:** novelty-reward + tail-harvesting **preserves base-level large-k pass@k
while keeping the small-k gain** — i.e. it sharpens without shrinking the boundary.

Setup (one small model, e.g. Qwen2.5-7B / Qwen3-4B, a verifiable-reward math set):
1. **Baselines:** base model; standard GRPO. Reproduce Yue's crossover (GRPO wins pass@1,
   base wins pass@256) — this is the control that must replicate.
2. **Ours-A:** GRPO + group-relative novelty reward (Component A only).
3. **Ours-AB:** + off-policy tail harvesting (Component B).
4. **Metric:** the **whole pass@k curve, k=1…256**, not just pass@1. Plus boundary-coverage
   (count of problems solved at any k).

> **Success = the pass@k crossover disappears: our model matches or beats GRPO at small k
> AND matches or beats the *base model* at large k.** That is, literally, "sharpen the mode
> and keep the tail." If novelty reward alone (Ours-A) closes the large-k gap, A is the
> lever; if only Ours-AB does, support expansion genuinely requires off-policy injection
> (the stronger, more interesting result).

**Ablations that make it publishable:** novelty on all-rollouts vs correct-only; harvested
tail off-policy vs on-policy reuse; λ sweep (the sharpen↔spread frontier); hard-problem
targeting on/off.

**Failure modes to watch:** (i) novelty reward rewards diverse *wrong* paths → guard by
gating on correctness; (ii) representation distance concentrates → §4.4 caveat; (iii)
off-policy harvest destabilizes RL → importance-weight or interleave SFT steps.

---

## 6. Positioning paragraph (for the proposal / advisor)

> Entropy-preservation methods (Clip-Cov/KL-Cov, forking-token gradients) slow policy
> collapse but stay on-policy — they cannot add paths outside the base support. ProRL shows
> RL *can* expand the boundary, but via prolonged training + reference-policy reset + a
> diverse task suite (brute force). Sun et al. show a representation-space diversity bonus
> yields ~3× pass@k. Our angle unifies and sharpens these: a **group-relative novelty reward
> that protects rare *correct* modes** (not generic entropy), fused with **off-policy
> harvesting of the model's own rare high-k successes** to widen support the way distillation
> does — but self-supervised and targeted at the hard problems where the boundary is
> actually stuck. The goal is to make the RL gradient *sharpen the dominant mode and grow the
> tail at once*, efficiently, without ProRL-scale training. Success criterion: erase the
> pass@k crossover — beat GRPO at small k and the base model at large k simultaneously.

---

## 7. Reference list (arXiv ids)

- **2504.13837** — Yue et al., RL sharpens not expands (the problem).
- **2505.22617** — Cui et al., entropy mechanism (Clip-Cov/KL-Cov).
- **2506.01939** — Wang et al., 80/20 forking tokens (branch-point gradient).
- **2505.24864** — Liu et al., ProRL (prolonged RL expands boundary).
- **2509.25849** — Knapsack RL (exploration-budget allocation).
- **2510.11686** — Sun et al., representation-based exploration (hidden-state diversity, 3× pass@k).
- **2601.16175** — TTT-Discover (test-time RL for discovery; reuse/horizon, cf. Directions 14–15).
- **2407.21787** — Large Language Monkeys (coverage scales, selection plateaus).
