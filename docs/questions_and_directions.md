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

**Why it's different:** Submodular information gain tells you about marginal value of
the *next* sample. Persistent homology tells you about the *shape* of what remains
unexplored — holes in the solution space the model will never fill regardless of
sampling density, because they don't exist in the model's representational support.

**Concrete signal:** Compute the Vietoris-Rips complex at increasing radii for
trajectory embeddings. If H_1 (loops) dies at the same radius as H_0 (connected
components merge), the model sees one cluster of solutions. If H_1 persists
significantly beyond H_0 stabilization, there are distinct strategic "corridors" still
being discovered.

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
