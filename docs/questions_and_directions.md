# Questions and Directions: Model Ceiling Detection via Trajectory Diversity

## Core Problem Statement

Given a small initial sample of reasoning chains (e.g., 8 trajectories), can we predict
whether additional test-time compute will yield new correct solutions — or whether the
model has hit a representational ceiling that only weight updates can overcome? And
once we condition on disagreements (breaking IID), does the resulting trajectory space
genuinely expand, or does it merely rephrase the same failure modes?

---

## Direction 1: Topological Persistence of the Solution Manifold

**Idea:** Treat each trajectory as a point in a high-dimensional semantic space (e.g.,
final-layer hidden states at reasoning steps). Compute the persistent homology of the
point cloud formed by K chains. If the Betti numbers stabilize rapidly (the topology
"freezes" with few samples), the model's representational capacity for this problem is
exhausted — adding more samples cannot create new topological features (new solution
strategies). If Betti numbers keep changing, the manifold is still being revealed and
more compute helps.

**Representation caveat:** Collapsing a full trajectory to a single point (e.g.,
mean-pooled hidden state) is an under-representation. It discards path structure —
where the chain branches, step ordering, whether two chains diverge early vs. late.
Two chains arriving at the same answer via completely different reasoning map to
nearby points despite being strategically distinct. Better representations:
- Treat each chain as a *curve* (sequence of step-level embeddings) and use Fréchet
  distance or DTW between curves, then do persistent homology on the resulting
  distance matrix between curves
- Compute a persistence diagram *per trajectory* (from its own step-level point
  cloud), then use Wasserstein distance between persistence diagrams as the
  inter-trajectory metric
- Time-delay embedding (Takens-style) on the token-level entropy signal along the
  chain — preserves dynamical structure without full hidden states

The point representation is a cheap first pass. The real signal lives in path geometry.

**Why it's different:** Submodular information gain tells you about marginal value of
the *next* sample. Persistent homology tells you about the *shape* of what remains
unexplored — holes in the solution space the model will never fill regardless of
sampling density, because they don't exist in the model's representational support.

**Concrete signal:** Compute the Vietoris-Rips complex at increasing radii for
trajectory embeddings. If H_1 (loops) dies at the same radius as H_0 (connected
components merge), the model sees one cluster of solutions. If H_1 persists
significantly beyond H_0 stabilization, there are distinct strategic "corridors" still
being discovered.

**Related work:** "Geometry Score: A Method For Comparing Generative Adversarial
Networks" (Khrulkov & Oseledets, ICML 2018) — https://arxiv.org/abs/1802.02664.
Uses persistent homology on point clouds of generated samples to detect mode collapse
in GANs. Directly analogous: replace "GAN outputs" with "reasoning trajectories" and
"mode collapse" with "diversity collapse / model ceiling."

**Question:** Can the death time of the longest-lived H_1 feature in 8 chains predict
whether the 1000th chain will find a fundamentally new approach?

---

## Direction 2: Spectral Decay of the Trajectory Gram Matrix

**Idea:** Arrange K trajectory embeddings as rows of a matrix. Compute the singular
value spectrum. A sharp spectral cutoff (few dominant singular values capturing >95% of
variance) means the model is generating in a low-dimensional subspace — its
"effective diversity dimension" is small. This is the ceiling: you cannot sample your
way out of a low-rank generative manifold.

**The DAD connection:** After conditioning on disagreements, recompute the spectrum.
If conditioning *increases* the effective rank (pushes more mass into smaller singular
values), it genuinely expanded the solution space. If the rank doesn't change but
values just rotate, conditioning is reshuffling, not expanding.

**Concrete experiment:**
- Sample 8 IID chains → compute rank-k approximation threshold at 95% energy
- Sample 8 chains conditioned on DAD workspace → recompute
- Define "diversity gain" = (effective_rank_conditioned - effective_rank_iid) / effective_rank_iid
- If diversity gain ≈ 0 for a problem, flag it as compute-unscalable

**Question:** Is there a critical effective-rank threshold below which majority-vote
accuracy is empirically flat regardless of sample budget?

---

## Direction 3: Mode Connectivity via Linear Interpolation in Logit Space

**Idea:** Pick two trajectories that disagree on the final answer. At their divergence
point (first step where they disagree), interpolate between the two logit distributions
(alpha * logits_A + (1-alpha) * logits_B for alpha in [0,1]). If every interpolant
is high-entropy or leads to a new valid continuation, there is a *connected* region of
solution space between them — the model "knows about" intermediate strategies but
sampling happened to miss them. If interpolation immediately falls into degenerate
(incoherent) territory, the two solutions are *disconnected modes* — the model has no
path between them, and more sampling will just revisit the same modes.

**Why it matters for ceilings:** Disconnected modes mean the generative process is
stuck in discrete basins. You cannot get to the correct answer by sampling more if the
correct basin doesn't exist in the model's representation. Connected modes mean the
correct answer might sit "between" observed chains — more varied temperature/top-p
might reach it.

**Question:** Does DAD's workspace conditioning create new interpolation paths (mode
connectivity) that IID sampling lacks?

---

## Direction 4: Causal Intervention on the Disagreement Graph

**Idea:** DAD extracts a claim DAG. Instead of just picking the top-leverage claim to
condition on, perform counterfactual interventions: forcibly set a disputed claim to each
of its competing values and observe whether downstream claims *actually change* in
subsequent generations. If fixing claim X to value V₁ vs V₂ produces the same
downstream trajectory distribution, then X is a spurious disagreement (surface
variation, not causal). The "true ceiling" is defined by the number of
causally-effective disagreements: if all disagreements are spurious (downstream is
invariant to intervention), more compute is wasted.

**Connection to identifiability:** This is essentially testing whether the DAG is
identifiable from interventional data. If it is, the model has structured uncertainty
that additional compute can resolve. If interventions don't propagate, the model lacks
the mechanism to convert more samples into better answers.

**Question:** What fraction of high-leverage disputed claims (as identified by
Gauss-Southwell) are actually causally downstream-effective vs. epiphenomenal?

---

## Direction 5: Lyapunov Exponents of the Reasoning Chain

**Idea:** Treat the sequence of hidden states in a reasoning chain as a discrete
dynamical system: h_{t+1} = f(h_t, x_t). Compute the maximal Lyapunov exponent
(sensitivity to perturbation) at each reasoning step. If the Lyapunov exponent is high
at a step, small perturbations (different sampled tokens) lead to exponentially
diverging trajectories — the model is at a "reasoning bifurcation." If the exponent is
low everywhere, the model is on a stable attractor regardless of sampling noise.

**Ceiling interpretation:**
- High Lyapunov exponents + wrong final answers = the model *can* explore but
  hasn't found the right basin yet → more compute helps
- Low Lyapunov exponents + wrong final answers = the model is locked onto a wrong
  attractor → weight update needed
- High Lyapunov exponents + right answers = fragile correctness, might benefit from
  verification/voting rather than more exploration

**Practical approximation:** Perturb the hidden state at step t by a small epsilon
(or equivalently, resample one token from the softmax at step t). Measure how much the
subsequent K steps diverge (L2 distance in hidden space). The growth rate of this
divergence approximates the local Lyapunov exponent.

**Question:** Is there a correlation between per-problem Lyapunov exponent (measured at
the first disagreement point) and the scaling curve of majority-vote accuracy?

---

## Direction 6: The Adversarial Witness Problem

**Idea:** Instead of asking "will more samples help?" directly, ask the *dual* question:
"can I construct a prompt perturbation that makes the model find a different answer?" If
a tiny rephrasing or hint makes the model switch to the correct answer, the knowledge
exists in the weights but the sampling process can't reach it without guidance — a
diversity problem. If no prompt perturbation (within a bounded norm) changes the output,
the model genuinely doesn't have the solution in its representational capacity.

**Implementation:**
- Use gradient-based soft-prompt search to find perturbations that flip the final answer
- If such perturbations exist and are small, report "ceiling NOT reached — latent
  knowledge exists, sampling is inadequate"
- If no perturbation flips the answer (or only very large/degenerate ones do), report
  "ceiling reached — weight update required"

**Why this is complementary:** Diversity measures tell you about the *sampling process*.
The adversarial witness tells you about the *model's latent capacity*. A model might
have low trajectory diversity but high latent capacity (bad sampling strategy). Or
high trajectory diversity but zero capacity to reach the correct answer (all its
diverse attempts are wrong in the same fundamental way).

**Question:** For problems where DAD fails after max rounds, does an adversarial
witness (prompt perturbation that produces the correct answer) typically exist?

---

## Direction 7: Information Bottleneck on the Trajectory Ensemble

**Idea:** Apply the information bottleneck principle to the collection of trajectories.
Define three random variables: X = the problem, T = the collection of trajectories
(compressed representation), Y = the correct answer. Find the optimal compression of
T that maximizes I(T;Y) while minimizing I(X;T). The *rate* at which you need to
increase I(X;T) to get more I(T;Y) is the scaling law of test-time compute for that
problem.

**Ceiling detection:** If the I(T;Y) curve saturates while I(X;T) is still growing
(you're adding trajectories that carry information about the problem but NOT about the
answer), that's the ceiling: the model is generating problem-relevant but
answer-irrelevant reasoning.

**Practical proxy:** Use a learned compressor (small MLP) that takes trajectory
embeddings and predicts the answer. Track its loss as you feed it 1, 2, 4, 8, ...
trajectories. If the loss plateaus at 4 trajectories, those first 4 carry all the
information the model has about the answer — the rest is redundant.

**Question:** At what trajectory count does the information bottleneck saturate for
problems where best-of-N eventually succeeds vs. permanently fails?

---

## Direction 8: Thermodynamic Free Energy of the Answer Distribution

**Idea:** Frame the answer distribution across chains as a Boltzmann distribution with
the "energy" being the negative log-likelihood of each answer. Define the free energy
F = U - T*S where U is the average energy (negative average confidence) and S is the
entropy of the answer distribution. The "temperature" T here is a parameter you
sweep.

**Ceiling signal:** Compute F at different effective temperatures (by reweighting
chains). A phase transition in F (discontinuity in dF/dT) indicates a structural
change in the energy landscape. If the phase transition temperature is very low (the
landscape is smooth, all answers coexist at almost any temperature), the model has
no strong preference — it's sampling noise all the way down. If there's a clear
phase transition at a moderate temperature, the model has genuine competing hypotheses
and more samples + selection can resolve them.

**Analogy:** In protein folding, a clear folding temperature means the sequence
*encodes* the correct structure. A model with a clear "reasoning phase transition"
*encodes* the correct solution — it just needs enough sampling to crystallize.

**Question:** Do problems with clear thermodynamic phase transitions have higher
best-of-N scaling rates?

---

## Direction 9: Trajectory Genealogy via Token-Level Phylogenetics

**Idea:** Build a phylogenetic tree of reasoning chains. Align chains at the token
level (like sequence alignment in bioinformatics). Compute a distance matrix and
infer the tree topology. The tree structure reveals:
- Star topology (all chains diverge from a common ancestor early) → the model makes
  one irrevocable decision early and explores cosmetically from there
- Deep branching (chains diverge at many different points) → genuine exploration of
  different reasoning paths
- Comb topology (each new chain is just slightly modified from the last) → no
  real diversity, just stochastic perturbation

**After DAD conditioning:** Does the tree topology change from star → deep? If yes,
conditioning truly opened new strategic branches. If it stays star-shaped (just a
different star point), conditioning redirected but didn't diversify.

**Ceiling signal:** Star topology + wrong majority answer → ceiling (the early
irrevocable decision is wrong and sampling can't undo it). Deep branching + mixed
answers → scalable (genuine strategic exploration in progress).

**Question:** What's the phylogenetic tree depth distribution for problems that
eventually benefit from >100 samples?

---

## Direction 10: Compressibility as a Diversity Proxy

**Idea:** Concatenate all K reasoning chains. Compute the compression ratio
(length after gzip / original length). If K chains compress almost as well as
one chain repeated K times, they carry minimal unique information — the model is
saying the same thing K different ways. If the compression ratio grows close to
K (nearly incompressible), each chain adds genuinely new information.

**The normalized compression distance (NCD) variant:** For each pair of chains (i,j),
compute NCD(i,j) = (C(ij) - min(C(i),C(j))) / max(C(i),C(j)) where C(x) is the
compressed length of x. The average NCD across all pairs is a cheap, model-free
diversity score. Plot NCD vs. sample count: if NCD drops as you add more chains
(later chains are more compressible given earlier ones), diversity is collapsing.

**Why this might beat embedding-based diversity:** It captures *surface-level*
diversity (different words, different orderings, different equation formulations)
which embedding similarity misses. A model might produce semantically identical
chains with very different surface forms (high embedding similarity, high NCD) —
that's useless diversity. Or semantically different chains with similar surface
patterns (low embedding similarity, low NCD) — that's genuine strategic diversity.

**Question:** Is NCD-decay-rate (across 8 chains) predictive of whether the problem
is compute-scalable?

---

## Direction 11: Escape Velocity from Wrong Attractors

**Idea:** For problems where the majority of chains give wrong answer A, define the
"escape velocity" as the minimum perturbation strength (temperature, prompt
modification, conditioning strength) required to shift even one chain to a different
answer. Low escape velocity means the wrong attractor is shallow — more diverse
sampling or mild conditioning can escape it. High escape velocity means it's a deep
basin — the model is structurally committed to the wrong reasoning.

**Measurement:** Start from the standard sampling configuration. Gradually increase
temperature. At what temperature does the answer distribution first become
non-degenerate (entropy > 0)? Call this T_escape.

**Ceiling interpretation:**
- T_escape very high (>1.5) → deep wrong attractor → weight update needed
- T_escape moderate (0.7–1.2) → shallow attractor → strategic conditioning (DAD)
  or more samples will help
- T_escape very low (<0.5) → no real attractor → the model is already exploring,
  just needs more budget

**Connection to DAD:** The workspace conditioning in DAD effectively provides a
"directed perturbation" — compare its effectiveness against undirected temperature
increases. If DAD achieves what T=1.5 achieves at T=0.7, it's providing an efficient
escape direction, not just brute force.

**Question:** Is T_escape computable from 4 chains at T=0.7, or do you need to
actually sample at elevated temperatures to estimate it?

---

## Direction 12: Representational Capacity via Probing the Residual Stream

**Idea:** Train a linear probe on the model's residual stream (intermediate hidden
states) to predict the correct answer — NOT from the generated tokens, but from the
internal representations *before* the model commits to output tokens. If the probe
can predict the correct answer from the hidden states of a wrong chain, the model
"knows" the answer internally but its generation process fails to surface it. This is
a clear signal that test-time compute (better search/selection) can help. If the
probe *cannot* predict the correct answer from any chain's hidden states, the knowledge
is absent from the weights.

**Practical version:** For each problem, generate 8 chains. Collect hidden states at
the point of first disagreement. Train a simple classifier: "given hidden state at
step t, is the chain heading toward the correct answer?" If this classifier has low
accuracy (not much better than majority-baseline), the correct-answer signal was
never present internally. If it's high, the signal exists but gets lost in generation.

**Why this is complementary to diversity:** Diversity measures tell you whether the
model *explores* different paths. Probing tells you whether the *right* path exists
in the model's representational space at all, even if it never gets selected.

**Question:** For problems where DAD fails, how often is the correct answer linearly
decodable from the residual stream of wrong chains?

---

## Meta-Question: The Conditioning Paradox

You observe that conditioning on disagreements breaks IID. Here's the deeper question:

**Does conditioning-induced diversity *actually* expand the reachable set, or does it
merely redistribute probability mass within the same set?**

Consider: if the model's generative support (the set of all sequences it can produce
with p > epsilon) is fixed by the weights, then conditioning can only *redistribute*
mass within that support — it cannot create new reachable points. Diversity increases
locally (the conditioned distribution looks more spread out) but the *reachable set*
hasn't changed.

The counter-argument: conditioning changes the *effective* temperature landscape. A
sequence that was reachable but with probability 10^{-20} (requiring 10^20 samples)
becomes reachable with probability 10^{-2} (requiring 100 samples). This is a
meaningful expansion of the *practically* reachable set even if the theoretical
support hasn't changed.

**The real ceiling question:** Is the correct solution in the model's generative
support at all? If yes, conditioning makes it practically reachable (compute helps).
If no, nothing short of weight updates will work.

**Proposed test:** Generate at extremely high temperature (T=2.0+) with very long
chains. If the correct answer *ever* appears, even once in 1000 chains, the support
includes it — and the question becomes pure search efficiency. If it never appears
even at extreme temperatures, that's strong evidence of a hard ceiling.

---

## Synthesis: A Practical Ceiling Detector

Combining the cheapest signals from above, a practical early detector might look like:

1. **8 chains, standard temperature** → compute effective spectral rank, NCD, and
   answer entropy
2. **8 chains, conditioned on DAD workspace** → recompute same metrics
3. **Decision rule:**
   - If spectral rank doesn't increase AND NCD is low AND entropy doesn't increase
     after conditioning → predict ceiling (weight update needed)
   - If spectral rank increases OR NCD is maintained OR new answers appear after
     conditioning → predict scalable (more compute helps)
4. **Calibration:** Validate against ground truth by running 1000+ chains on a
   benchmark and checking which problems actually benefit from scale

The key insight: you don't need 1000 chains to *detect* the ceiling. You need the
right 8 chains + the right diversity metrics + the right comparison (IID vs.
conditioned) to infer it.

---

## Direction 13: Test-Time Discovery via Max-Trajectory Reward (Learning to Discover)

**The shift in objective:** Standard test-time scaling optimizes expected reward across
trajectories (majority vote, best-of-N by average). "Learning to Discover at Test Time"
inverts this: instead of maximizing E[R], maximize max_i R(trajectory_i) — the reward
of the single best trajectory. The insight is that if you optimize for the *ceiling*
of your trajectory distribution rather than its *center*, you train the model to
produce at least one SOTA-level solution among its K samples, even if most samples are
garbage.

**Why this matters for our ceiling question:** This reframes the problem entirely.
A model might have a "ceiling" under average-reward optimization (majority vote
plateaus) while having NO ceiling under max-trajectory optimization (the single best
chain keeps improving with more samples + RL shaping). The ceiling isn't a property
of the model alone — it's a property of the (model × objective × selection) triple.

**The RL interference angle:** The key mechanism is that an external RL reward signal
reshapes the sampling distribution at test time — not by retraining the full weights,
but by learning a lightweight policy (e.g., a value head or a steering vector) that
biases generation toward trajectories with high *peak* reward. This is a middle ground
between pure sampling (no weight update, diversity-limited) and full retraining
(expensive, changes the model). It's a *targeted* weight update that specifically
expands the practically-reachable set in the direction of high reward.

**Connection to DAD and the conditioning paradox:**

DAD conditions on disagreements to redirect sampling — but the objective is still
implicitly average-reward (majority vote selects the final answer). What if instead:

1. Use DAD's disagreement extraction to *identify* the frontier of what the model
   can produce (the diverse set of competing strategies)
2. Apply a reward signal (PRM, ORM, or ground truth on a few examples) to identify
   which *direction* in trajectory space leads to SOTA
3. Use a small RL update (LoRA, steering vector, or even just a reweighting of the
   softmax via a learned value function) to bias future sampling toward that direction

This converts DAD from a "find disagreements → resolve them by voting" system into a
"find disagreements → use them as exploration signal → RL-steer toward the best
frontier." The ceiling moves because you're not just redistributing mass within the
existing support — you're *shaping* the support via gradient signal.

**The deeper question this raises:**

If we can detect (via directions 1–12 above) that a problem is at a hard ceiling
under pure sampling — can a small RL intervention (trained on the disagreement
structure) break through that ceiling? In other words:

- Pure sampling ceiling = "the correct answer has probability < epsilon in the model's
  distribution, no amount of IID or conditioned sampling will find it"
- RL-assisted ceiling = "after a lightweight reward-driven update, is the correct
  answer now reachable?"

The gap between these two ceilings is the "discovery gap" — what the model *could*
find with external interference that it *cannot* find through internal search alone.

**Concrete experimental framing:**

1. Identify problems where DAD hits the ceiling (spectral rank frozen, NCD collapsed,
   no new answers after conditioning) — these are "pure-sampling-hard"
2. Apply a reward model (PRM) to score all generated chains. Check: does the PRM
   assign higher reward to chains that are closer (in embedding space) to the correct
   solution, even when no chain actually reaches it?
3. If yes: train a small LoRA or prefix that maximizes *max-trajectory PRM reward*
   rather than average reward. This should specifically expand the model's reach
   toward the high-reward frontier.
4. Re-evaluate: does the previously-ceiling'd problem now get solved?

**The meta-insight:** Diversity collapse (our ceiling signal) might be the *trigger*
for when to apply RL interference. The two-phase system becomes:

- Phase 1 (cheap): Pure sampling + DAD conditioning. Monitor diversity metrics.
- Phase 2 (expensive, triggered by diversity collapse): Small RL update targeting
  max-trajectory reward in the direction indicated by the disagreement structure.

This makes the ceiling detection problem directly actionable: you're not just
diagnosing "this needs retraining" — you're identifying the minimal, targeted
intervention (a few gradient steps on a reward signal derived from the trajectory
diversity structure) that can break through.

**Open questions:**

- How many gradient steps (on the max-trajectory RL objective) are needed to break
  through a detected ceiling? Is it proportional to the "hardness" of the ceiling
  (e.g., inverse spectral rank)?
- Can the disagreement structure itself serve as the reward signal? E.g., reward
  trajectories that resolve the highest-leverage disputed claim in a *novel* way
  (not just picking one of the existing competing values, but finding a third option)
- Does this create a "discovery curriculum"? Easy problems → pure sampling. Medium
  problems → DAD conditioning. Hard problems → RL-steered discovery. The boundaries
  between these regimes are exactly what the ceiling detector identifies.
- Is there a formal relationship between the max-trajectory reward objective and the
  submodular information gain of the trajectory set? Maximizing max_i R(x_i) over a
  diverse set might be equivalent to maximizing a submodular facility-location
  objective where the "facilities" are solution strategies and the "coverage" is
  reward.

---

## Direction 14: Inter-Rollout Disagreement as the Dense Learning Signal for Discovery

**Grounding paper:** "Learning to Discover at Test Time" (TTT-Discover), arXiv:2601.16175.
Test-time RL on a *single* problem; 512 rollouts/step (8 groups × 64) sharing a reused
initial state; entropic objective `J_β = E_s[log E_a[e^{βR(s,a)}]]` to chase the *max*
(not mean) reward; PUCT reuse with `Q =` max-reward-of-descendants and `P(s) =`
reward-rank; LoRA r32, 50 steps, ~$500/problem.

### The core observation that motivates everything below

TTT-Discover compresses each of its ~25,600 rollouts into a **single scalar** `R(s)`.
But for a discovery problem the reward is **flat almost everywhere**: near the state of
the art, nearly every rollout returns `≈ r_sota` or `0` (invalid). The paper itself
reports the symptom — late in training "even smaller improvements vanish," and the
temperature `β` is "challenging" to set because reward *differences* disappear. So the
extrinsic signal carries almost no gradient exactly when discovery is hardest.

Yet the 64 rollouts in a group still **disagree richly** — on the high-level approach, on
intermediate decisions, on which sub-structure to exploit — even when their final rewards
are identical. That disagreement is:
- **Dense** — it exists at every decision point, and it persists when reward is flat;
- **Localizing** — the decisions rollouts *contest* are precisely where the solution space
  still branches, i.e. the frontier of what is unexplored;
- **Free** — it is already computed; TTT-Discover discards 100% of it.

**Meta-thesis:** Use inter-rollout disagreement as the dense learning signal the flat
scalar reward cannot provide — for credit assignment, exploration, ceiling detection, and
recombination. This is the DAD disagreement thesis (this repo) carried from *answering* to
*discovery*: there the disputed claim is the unit of uncertainty; here the contested
decision is the unit of unexplored solution space.

Below, six concrete mechanisms (A–F), ordered roughly from "drop-in reward shaping" to
"new action type." They are additive to TTT-Discover's Algorithm 1, not replacements.

### 14A. Branch-point credit assignment (disagreement → dense advantage)

A scalar `R(s)` gives **no credit assignment**: it cannot say *which* token/decision made
the kernel fast. But siblings in a rollout group that share a prefix and then diverge form
a natural **counterfactual**: align the 64 rollouts, find the token positions where high-
and low-reward siblings split (the *pivotal* decisions), and concentrate the policy
gradient there. A decision is important iff sibling rollouts that chose differently ended
with different reward — an empirical, per-decision advantage estimated from the group, not
a hand-designed shaping. Turns 64 noisy scalars into a dense, localized signal.

**Question:** Does pivotal-token-weighted RL beat uniform per-token GRPO at the same rollout
budget on a discovery task, *and* does the set of pivotal decisions concentrate (few high-
leverage forks) or spread (many)?

### 14B. Consensus / novelty decomposition (train on the disagreement residual)

Decompose every rollout into a **consensus part** (what nearly all rollouts agree on — the
boilerplate, the already-known approach) and a **novelty residual** (where it departs from
the pack). Discovery lives entirely in the residual that *also* raised reward. Up-weight
the gradient on novelty residuals of above-consensus-reward rollouts; down-weight the
consensus (the model already knows it). This is a principled antidote to mode collapse:
the policy is pushed to reinforce *what was new and worked*, not to re-memorize the shared
prefix that every rollout already contains.

**Question:** Is the "reward gain attributable to the novelty residual" a better predictor
of a genuine discovery (a basin jump, e.g. AlphaEvolve's symmetric → TTT-Discover's
asymmetric step function) than total reward?

### 14C. Disagreement-shaped advantage — a tuning-free replacement for β

TTT-Discover's `β` is fragile because it must convert vanishing reward gaps into usable
advantages with a hand-set temperature. Replace it: shape the advantage by **how far a
rollout's trajectory departs from the group consensus, scaled by its reward delta**. A
rollout that is *both* high-reward *and* far-from-consensus gets the strongest push — that
is the discovery direction by definition. The effective temperature now comes from the
*empirical disagreement spread of the batch*, not a hyperparameter: when the group is
spread out, exploration is cheap and the shaping is gentle; when it has collapsed, a small
reward gain at large representational distance is amplified. Self-annealing, no β.

**Question:** Does disagreement-shaped advantage match or beat the adaptive-β entropic
objective on the paper's own benchmarks (Erdős bound, TriMul kernel) without per-task
tuning?

### 14D. Disagreement collapse as a basin-exhaustion detector → forced jump

As test-time RL proceeds, the policy *converges* — rollouts collapse onto the one trick it
found. Track inter-rollout disagreement (entropy over approaches / mean pairwise distance
in an "approach embedding"). **When disagreement falls below threshold while reward
plateaus, the current basin is exhausted** — more rollouts here are wasted. That is the
trigger to act: re-seed `reuse` from a *low-reward-but-high-disagreement* buffered state,
inject a perturbation, or force a contrastive approach ("solve unlike the last K"). This
gives the search an explicit, measured fuel gauge for *when to stop refining and jump* —
which AlphaEvolve and TTT-Discover lack (they rely on the PUCT exploration bonus, which is
content-blind reward-rank). Connects directly to this repo's ceiling-detection work, now
as an *online* trigger inside a discovery loop rather than an offline verdict.

**Question:** Across a run, does the *rate of SOTA improvement* correlate with inter-rollout
disagreement rather than mean reward? And does a forced re-spread (when disagreement
collapses) re-accelerate discovery vs. letting naive RL continue?

### 14E. Cross-rollout synthesis as a first-class action (directed crossover)

Disagreeing rollouts often hold *complementary partial strengths*: rollout A has a clever
data layout, B a clever loop fusion, C a better numerical trick. TTT-Discover only ever
*selects* (max) or *reuses* a single state. Add an action type conditioned on a **set of
disagreeing rollouts**, explicitly tasked to *recombine their distinct ideas* into one
candidate — and let the reward of the synthesis train the model to be a good recombiner.
This is evolutionary crossover, but (i) the parents are chosen by *measured disagreement*
(maximize complementary coverage, not just top-reward), and (ii) the crossover operator is
the learned policy itself, improving over the run — exactly the DAD "disagreement
workspace," repurposed from resolving answers to fusing solution fragments.

**Question:** Does disagreement-selected synthesis (pick parents that disagree most while
both being valid) find higher-reward children than reward-greedy parent selection or iid
best-of-N at equal execution budget?

### 14F. Free process-verifier from resolved disagreements (prune before you execute)

When rollouts disagree on an *intermediate* claim ("this bound is tight" / "this fusion is
legal") and the downstream reward later reveals who was right, you get a **labeled example
for free**: contested-claim → correct-resolution, supervised by ground-truth reward. Train
a lightweight process verifier on the model's *own* resolved disagreements; then use it to
**prune rollouts at the branch point before the expensive transition** (math actions get a
10-minute code execution each — pruning duds pre-execution is the dominant cost lever).
Disagreement becomes self-generated process supervision, and the verifier compounds within
the run.

**Question:** Can a verifier trained only on intra-run resolved disagreements prune
≥50% of rollouts pre-execution with negligible loss of best-found reward — i.e. buy the
same discovery for a fraction of TTT-Discover's $500?

### Why disagreement is the *right* signal for discovery (the unifying argument)

1. Discovery reward is sparse and flat near SOTA → the scalar has ~no gradient.
2. The generative distribution nonetheless stays structured: rollouts disagree.
3. Disagreement is therefore a **dense signal that survives a flat reward**.
4. Disagreement **localizes the frontier**: contested decisions = where solution space
   still branches = where the next discovery can come from.
5. Hence: credit-assign on the contested decisions (14A/B), explore along the
   disagreement gradient (14C), detect exhaustion when it collapses (14D), recombine
   across it (14E), and distill a verifier from how it resolves (14F).

### The cheap decisive test (no test-time RL, one small model)

Pick a domain with clean continuous reward and public harness (TriMul-style GPU kernels:
reward = 1/runtime). With a small model, sample one group of ~64 candidate solutions.
Measure, per batch: (a) **mean pairwise disagreement** (in an approach-embedding or via
branch-point analysis) and (b) **best reward found**. Then the single question the whole
direction rides on:

> **Across rollout groups, does higher inter-rollout disagreement predict a higher best
> reward (and the appearance of structurally novel solutions), more than mean reward
> does?** And: do high-reward rollouts share *identifiable pivotal decisions* that
> low-reward siblings lack?

If yes → disagreement carries the discovery signal and 14A–F are justified; build the
RL loop around it. If no → disagreement near SOTA is just noise on this domain, and we
pivot before spending a single test-time-training dollar. One small model, an afternoon,
zero RL — the opposite of an expensive commitment.

---

## Direction 15: Composition, Not Gradient — Discovery as Horizon Extension Driven by Disagreement

*(This direction is grounded in a literature synthesis; the papers are mapped at the end.)*

### The tension the literature refuses to resolve

Two findings, both well-evidenced, are in direct contradiction if read naively:

- **"RL only sharpens"** (Yue et al., arXiv:2504.13837): RLVR-trained models beat the base
  model at small `k` but the **base model overtakes them at large pass@k**; coverage is
  "bounded by the base model"; RL "does not elicit fundamentally new reasoning patterns" —
  it re-weights mass onto paths already in the base support. Distillation adds new patterns;
  RL does not.
- **TTT-Discover** (arXiv:2601.16175): test-time *RL* on a single problem produces genuinely
  new SOTA (a 600-piece asymmetric step function no human or prior AI found).

Both cannot be true unless **the discovery in TTT-Discover does not come from the RL
gradient at all.** Resolution: TTT-Discover's real engine is `state reuse` — it feeds a
solution back as the *initial state* of the next attempt (`s_{i+1} ∼ reuse(H_i)`,
"adds an extra timestep to its trajectory"). That **extends the effective horizon** so the
policy *composes* improvements across attempts, reaching solutions no single base-model
rollout could express in one shot. The entropic-`β` RL is just local polishing on top.

**Thesis (Direction 15):** Discovery at test time is **composition across attempts**, not a
better gradient. The base model is a fixed library of *moves*; what's "new" is a *path*
through that library longer than any single rollout. Therefore the lever that matters is
**which solutions you reuse/compose, and where you splice them** — and *disagreement is the
correct operator for choosing both*. This subsumes Direction 14 under a cleaner principle
and yields a falsifiable claim the cited papers never test.

### Why disagreement is the composition operator (the logic)

1. RL can't add moves to the library (2504.13837). So gains must come from *recombining*
   existing moves over a longer horizon.
2. Recombination is only useful between attempts that **differ** — two near-identical
   rollouts compose into nothing new. The value of composing A and B is exactly their
   *complementary disagreement*: A solves a sub-part B doesn't, and vice-versa.
3. The right splice point is a **contested decision** — a token/step where rollouts
   genuinely branch. Splicing at a consensus point changes nothing; splicing at a branch
   point swaps a real sub-strategy.
4. So "disagreement-directed composition" = pick maximally-complementary parents (where to
   get moves) and splice at their branch points (where to join them). This is a *measured*,
   model-derived replacement for AlphaEvolve's hand-crafted crossover/diversity heuristics —
   which TTT-Discover dropped and never replaced.

### The three sub-claims, each falsifiable and cheap before any RL

**15A. Horizon, not gradient, is the discovery lever.**
Ablate TTT-Discover into: (i) RL + reuse (full), (ii) RL, no reuse (every attempt from
`<empty>`), (iii) **no RL, reuse only** — i.e. iteratively re-prompt the *frozen* base model
with its own best-so-far as the initial state, no weight update. If (iii) recovers most of
the SOTA gain, the discovery is **composition/horizon**, and the expensive RL is doing
little. *Prediction from 2504.13837:* (iii) should be surprisingly strong.

**15B. Disagreement-directed splicing beats reward-greedy reuse.**
Replace TTT-Discover's reward-rank reuse with: select the *pair* (or set) of buffered
solutions that **maximizes complementary disagreement** subject to both being valid, and
prompt the model to compose them at their contested decisions. Compare best-found reward
vs. (a) reuse-the-single-best and (b) iid best-of-N, at equal execution budget. This is
Direction 14E sharpened into the *primary* search operator rather than an add-on.

**15C. Selection is the hidden ceiling, and it scales with disagreement.**
From Large Language Monkeys (arXiv:2407.21787): coverage scales 4 orders of magnitude but
**selection (majority vote / reward model) plateaus after a few hundred samples** — the
answer is present yet unpickable. From Sample-Scrutinize-Scale (arXiv:2502.01839):
**self-verification accuracy itself improves with more samples** ("implicit scaling").
Synthesis claim: the quantity that makes a candidate *selectable* is whether its winning
decisions **survive cross-examination by disagreeing siblings** — i.e. a candidate whose
contested choices are confirmed correct by the resolved disagreements (Direction 14F) is
both more likely right *and* identifiable without ground truth. So disagreement is not only
the *generation* operator (15B) but the *selection* operator that breaks the pass@k →
pass@1 gap these papers leave open.

### Why this is more fundamental than Directions 13–14

- Direction 13 framed discovery as a *max-reward RL objective*. The literature
  (2504.13837) says that objective **cannot exceed the base model's reach** — so 13's
  premise is partly wrong, and Direction 15 explains *why TTT-Discover works anyway*
  (composition, not gradient).
- Direction 14 used disagreement as a *learning signal inside RL*. Direction 15 says the
  RL may be largely unnecessary: disagreement is the **search-and-selection** operator over
  a frozen base library, and the whole thing may not need gradients at all — which makes it
  ~100× cheaper than TTT-Discover's \$500/problem if 15A holds.
- It connects to **representation-based exploration** (Sun et al., arXiv:2510.11686): their
  result that a *hidden-state* diversity bonus buys 3× sample efficiency is evidence that
  the useful notion of "different attempt" lives in representation space — exactly where
  15B should measure complementary disagreement (approach-embedding distance), not in
  surface tokens.
- It respects **compute-optimal scaling** (Snell et al., arXiv:2408.03314): the
  sample-vs-revise-vs-reuse choice should switch by problem difficulty — 15A's ablation
  directly measures *for which problems* reuse/composition is the active ingredient.

### The single decisive experiment (frozen model, no RL, cheap)

On a clean-reward public harness (GPU kernels: reward = 1/runtime; or the Erdős
step-function with the released validator):

1. **Baseline:** iid best-of-N with a small frozen model (the Large-Language-Monkeys curve).
2. **Reuse-only (15A):** iteratively re-prompt the frozen model with its best-so-far as
   initial state, N total rollouts — *no weight updates*.
3. **Disagreement-composition (15B):** same budget, but each new attempt is seeded by the
   max-complementary-disagreement pair from the buffer, spliced at contested decisions.

Measure best-found reward vs. compute for all three.

> **The bet: (2) ≫ (1) — composition beats iid sampling with zero training — and (3) ≫ (2)
> — disagreement-directed composition beats reward-greedy reuse.** If both hold, we have a
> training-free discovery method that explains TTT-Discover's result as horizon extension,
> beats naive reuse via disagreement, and costs a small fraction of \$500/problem. If (2) ≈
> (1), composition isn't the engine and the gradient really matters — pivot back to 14.

### Literature map (for grounding)

- **2407.21787 Large Language Monkeys** — coverage scales, *selection* plateaus → the
  pass@k→pass@1 gap (motivates 15C).
- **2502.01839 Sample, Scrutinize & Scale** — verification *scales* with samples, weak
  out-of-box → selection is improvable (15C).
- **2504.13837 RL only sharpens** — RL bounded by base support, base wins at large pass@k →
  discovery must be composition, not gradient (15A, core tension).
- **2510.11686 Representation-based exploration** — hidden-state diversity bonus = 3×
  efficiency → measure disagreement in representation space (15B).
- **2408.03314 Compute-optimal TTS** — best strategy is difficulty-dependent → reuse is the
  active ingredient only for some problems (15A ablation).
- **AlphaEvolve / 2506.13131** — evolutionary crossover works but needs hand-crafted
  diversity/fitness → replace with measured disagreement (15B).
- **2601.16175 TTT-Discover** — the artifact to explain; reuse extends horizon, RL polishes.
