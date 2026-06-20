# Core Intuitions: Papers Related to Trajectory Diversity & Model Ceilings

---

## 1. DRA-GRPO: Diversity-Aware Reward Adjustment for GRPO

**Paper:** https://arxiv.org/abs/2505.09655

**The problem it solves:** Standard GRPO gives identical scalar rewards to all correct
trajectories regardless of how different their reasoning paths are. This causes the
policy to collapse into a narrow set of dominant strategies, killing diversity.

**Core mechanism:**

1. During GRPO training, the model samples K solutions per problem. Normally, all
   correct ones get reward=1.
2. DRA measures the *semantic density* around each solution using Submodular Mutual
   Information (SMI) — how similar is this solution to its neighbors in the batch?
3. Solutions in dense clusters (over-represented strategies) get their reward
   *downweighted*. Solutions in sparse regions (novel strategies) get *upweighted*.
   This is Inverse Propensity Scoring (IPS) — rare strategies receive higher
   effective reward.
4. Effect: a repulsive force against redundancy. The gradient pushes the policy
   toward covering the full high-reward landscape rather than collapsing to one mode.

**Key intuition in one sentence:** If you reward a model equally for finding the
same answer five different ways that are all basically identical, you're teaching it
to be repetitive. Reward the *rare* correct approach more, and the model retains
strategic diversity.

**Why it matters for our work:** This is the *training-time* version of our
test-time problem. We want to detect diversity collapse at inference; they prevent
it during RL training. The SMI + IPS machinery is directly applicable as a
*diversity scoring function* for our trajectory ensembles — not to shape gradients,
but to measure whether the model's output distribution is healthy or collapsed.

---

## 2. Geometry Score (Khrulkov & Oseledets, ICML 2018)

**Paper:** https://arxiv.org/abs/1802.02664

**The problem it solves:** How do you detect mode collapse in a generative model
without knowing the ground truth distribution? Existing metrics (FID, IS) compare
feature statistics but miss *structural/topological* properties — whether the
generator covers all the "holes" and "loops" in the real data manifold.

**Core mechanism:**

1. Sample points from both the real distribution and the generator.
2. Build a Vietoris-Rips simplicial complex at increasing distance thresholds.
3. Track which topological features (connected components = H_0, loops = H_1,
   voids = H_2) appear and disappear as the threshold grows → persistence diagrams.
4. Compare the persistence diagrams of real vs. generated samples. A generator
   with mode collapse will have *fewer persistent features* — missing loops/holes
   that exist in the real data's topology.

**Key intuition in one sentence:** If you can measure the "shape" of what a
generator produces (not just its statistics), you can detect when entire regions
of the solution space are missing — which is exactly what mode collapse is.

**Why it matters for our work:** Replace "GAN outputs" with "reasoning trajectories"
and "mode collapse" with "diversity collapse / model ceiling." If 8 chains form a
point cloud whose persistent homology has converged (no new topological features
appearing), the model has exhausted its representational capacity for this problem.

---

## 3. TTRL-CoCoV: Test-Time Reinforcement Learning via Verification-Generation Gap

**Paper:** https://arxiv.org/abs/2606.03608

**The problem it solves:** Can a model improve itself *during inference* on unseen
problems, without any ground-truth labels? Standard test-time scaling (best-of-N,
majority vote) selects among existing samples but doesn't *change the model*.

**Core mechanism:**

1. Key asymmetry: **verification is easier than generation.** A model that can't
   reliably produce the correct answer can often reliably judge whether a proposed
   answer is correct.
2. Generate K samples → use the model's own verification ability to assign
   pseudo-labels (self-supervised signal).
3. Apply RL (PPO/GRPO) using these pseudo-labels as reward → the model literally
   trains on the fly, using its own verification as the supervisor.
4. Confidence-conditioned strategy:
   - High-confidence samples: add exploration bonus to prevent diversity collapse
   - Low-confidence samples: delegate to verifier to filter wrong pseudo-labels
   - Medium-confidence: skip (signal too unreliable)

**Key intuition in one sentence:** A model can bootstrap its own improvement at test
time by exploiting the gap between what it can *judge* and what it can *produce* —
using its judgment to teach its generation.

**Result:** +9.8% Pass@1, and the label-free method *outperforms* fully supervised RL
by +5% on some benchmarks. Test-time RL converts additional compute into genuine
capability improvement, not just more attempts.

**Why it matters for our work:** This is the mechanism by which a "ceiling" under
pure sampling becomes breakable. If DAD detects diversity collapse (the model can't
find new answers by sampling), TTRL-style intervention can shift the weights
specifically to open up new generation paths. The verification-generation gap tells
you *whether such an intervention would help*: if the model can verify the correct
answer but can't generate it, there's recoverable capacity. If it can't even verify,
that's a hard ceiling.

---

## 4. The Depth Ceiling: Limits of LLMs in Discovering Latent Planning

**Paper:** https://arxiv.org/abs/2604.06427

**The problem it solves:** Is there a fundamental depth limit on the reasoning
strategies a model can *discover on its own* (from outcome supervision alone)?
And does test-time compute extend that limit?

**Core mechanism:**

1. Use graph path-finding as a controlled testbed where the required number of
   latent planning steps is precisely controllable.
2. Train models with ONLY final-answer supervision (no chain-of-thought labels).
3. Measure: how many coordinated latent steps can the model *discover* and execute?

**Key finding — the dissociation:**
- **Discovery ceiling:** Models cap out at discovering strategies requiring ~5
  coordinated latent steps, *regardless of scale* (tiny transformers: 3, GPT-4o: 5,
  GPT-5.4: 7).
- **Execution ceiling (higher):** Once a strategy is discovered, it generalizes to
  ~8 steps at test time.

**Key intuition in one sentence:** There's a hard limit on what models can *figure
out* from outcome feedback alone, but a softer limit on how far they can *extend* a
strategy they've already learned — test-time compute helps execution, not discovery.

**Why it matters for our work:** This formalizes exactly the ceiling we're trying to
detect. Our diversity metrics should distinguish:
- "The model hasn't discovered the right strategy yet" (diversity collapse because it
  keeps trying the same wrong approach) → needs training signal / RL intervention
- "The model discovered the strategy but can't execute it deeply enough in one pass"
  (diversity exists in early steps but chains truncate/degrade) → more test-time
  compute helps

---

## 5. Representation-Based Exploration for Language Models (Test-Time to Post-Training)

**Paper:** https://arxiv.org/abs/2510.11686

**The problem it solves:** Does standard RL actually *discover* new behaviors, or
does it just *sharpen* what the base model already does? And can you make test-time
sampling genuinely exploratory rather than redundant?

**Core mechanism:**

1. Measure diversity in the model's *hidden representation space* (not surface text).
   Two outputs that look different as text but have similar hidden states are NOT
   truly diverse.
2. Add an exploration bonus at test time: each new sample is incentivized to be
   representationally distant from previously generated samples.
3. At training time: RL with exploration bonus → the model *learns to produce diverse
   solutions*, not just to concentrate on the single best strategy.

**Key finding — sharpening vs. discovery:**
- Standard GRPO/RL: concentrates mass on already-reachable solutions (sharpening).
  Pass@k improves only because P(best solution) increases, not because new solutions
  appear.
- RL with exploration bonus: genuinely discovers new behaviors the base model couldn't
  reach. Pass@80 with exploration matches Pass@256 without it (3x sample efficiency).

**Key intuition in one sentence:** If you want to find the best trajectory among K
attempts, you need K *different* attempts, not K copies of your best guess — and
the model's own representation space defines what "different" means.

**Max-trajectory vs. average reward:** Standard RL optimizes E[R] (average quality),
which encourages convergence to one reliable strategy. But with verifiers or best-of-K
selection, what matters is max_i R(x_i). Optimizing for max-over-samples fundamentally
requires *high variance* in solution strategies. Exploration bonuses align the
objective with this max-trajectory view.

**Why it matters for our work:**
1. The representation-based diversity metric is a natural candidate for our ceiling
   detector — measure effective dimensionality in the model's own hidden space.
2. The sharpening/discovery distinction maps directly onto our question: does DAD
   conditioning *discover* new solutions (expand the reachable set) or just
   *sharpen* selection among existing ones?
3. The 3x sample efficiency result quantifies exactly what proper diversity
   measurement buys you at test time.

---

## Connecting Thread: The Three Regimes

These papers collectively reveal a three-regime picture:

| Regime | Signal | Intervention |
|--------|--------|--------------|
| **Scalable** (more compute helps) | High representational diversity, persistent topology still revealing new features, strategy discovered but execution limited | More sampling, best-of-N, majority vote |
| **Conditioning-scalable** (structured compute helps) | Moderate diversity, competing strategies exist but sampling is redundant | DAD-style workspace conditioning, exploration bonuses, test-time diversity steering |
| **Ceiling** (compute alone won't help) | Diversity collapsed, topology frozen, strategy not discovered, no verification-generation gap | RL intervention (TTRL), max-trajectory reward training, new supervision signal needed |

The ceiling detector's job is to classify problems into these regimes cheaply (from 8
chains), so compute is allocated to problems that will actually benefit from it.
