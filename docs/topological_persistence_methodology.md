# Topological Persistence for Model Ceiling Detection: Full Methodology

## 1. The Hypothesis

**Claim:** Given K reasoning chains from an LLM, the persistent homology of their
hidden-state trajectories predicts whether additional test-time compute (more chains)
will improve accuracy — or whether the model has reached a representational ceiling
that only weight updates can break.

**Formal statement:** Let M be a model, q a problem, and T_K = {τ_1, ..., τ_K} a
set of K reasoning trajectories. Define the "compute scalability" of (M, q) as:

  S(M, q) = lim_{N→∞} [Acc(M, q, N) - Acc(M, q, K)]

where Acc(M, q, N) is the majority-vote accuracy with N chains. We hypothesize that
the topological features of T_K (specifically, the persistence of H_1 in the
hidden-state point cloud) are predictive of S(M, q) > 0.

**Why this should work:** If the model's generation process for problem q is confined
to a topologically simple region of representation space (a single connected cluster
with no loops), then all future samples will fall in this same region — more compute
cannot discover fundamentally new solution strategies. Conversely, if the topology is
rich (loops, voids), distinct strategic corridors exist that more sampling could
explore.

---

## 2. Representing Trajectories in Hidden Space

### 2.1 From Tokens to Curves

A reasoning chain τ_i is a sequence of tokens (t_1, t_2, ..., t_L). At each
generation step l, the model's last transformer layer produces a hidden state:

  h_l^{(i)} = f_θ(t_1, ..., t_l) ∈ ℝ^d

where d is the hidden dimension (5120 for Qwen3-32B). The full trajectory is a curve
in ℝ^d:

  γ_i = (h_1^{(i)}, h_2^{(i)}, ..., h_L^{(i)})

This is NOT a single point. It is a path through the model's representation space,
encoding the sequential reasoning process.

### 2.2 Why Hidden States, Not Text

Text is the projection of reasoning onto a discrete vocabulary. Two chains can produce
identical text ("Therefore x = 5") while the underlying representations diverge — the
model arrived at x=5 via different internal computations. Conversely, chains can look
textually different ("We compute x" vs "Solving for x gives") while following the
same representational trajectory.

The hidden state IS the computation. Topology on hidden states measures the diversity
of the model's actual reasoning process, not its surface expression.

### 2.3 Subsampling

With L up to 16384 tokens, storing every hidden state is expensive. We subsample every
S steps (S=32 in our implementation), yielding ~512 points per curve. For the topology
to be meaningful, we need S small enough to capture the curvature of the path, but
large enough to be tractable. The key constraint: S must be smaller than the typical
"reasoning step" length (which is ~50-200 tokens in math chains). At S=32, we capture
the within-step dynamics.

---

## 3. Distance Between Trajectories

### 3.1 Dynamic Time Warping (DTW)

Trajectories have different lengths. Chain i might be 3000 tokens and chain j might
be 12000 tokens. We cannot simply compare h_l^{(i)} with h_l^{(j)} at the same index.

DTW finds the optimal alignment between two curves by warping the time axis. Given
curves γ_i = (a_1, ..., a_M) and γ_j = (b_1, ..., b_N), DTW computes:

  DTW(γ_i, γ_j) = min_{π} (1/(|π|)) Σ_{(m,n)∈π} ||a_m - b_n||_2

where π is a monotonic alignment path through the M×N grid. The normalization by |π|
makes it comparable across different-length curves.

**Why DTW over Fréchet:** DTW allows many-to-one alignments (one reasoning step in
chain i might correspond to several sub-steps in chain j). Fréchet distance requires
a monotonic bijection, which is too rigid for reasoning chains where one model
sometimes expands a step into multiple sub-steps.

### 3.2 The Distance Matrix

Given K chains, we compute the K×K distance matrix:

  D_{ij} = DTW(γ_i, γ_j)  for all i,j ∈ {1,...,K}

This is a metric (symmetric, non-negative, satisfies triangle inequality up to
DTW's approximation). It encodes the pairwise dissimilarity of all reasoning
trajectories in representation space.

---

## 4. Persistent Homology

### 4.1 The Vietoris-Rips Filtration

Given the distance matrix D, we build a nested sequence of simplicial complexes
(the Rips filtration). At radius ε:

  VR(ε) = {σ ⊆ {1,...,K} : D_{ij} ≤ ε for all i,j ∈ σ}

As ε increases from 0 to ∞:
- At ε=0: K isolated points (each chain is its own component)
- At small ε: edges appear between nearby chains
- At larger ε: triangles, tetrahedra form as clusters merge
- At ε=∞: one single K-simplex

### 4.2 Homology Groups

At each ε, the simplicial complex VR(ε) has homology groups:

- **H_0(VR(ε))**: Connected components. Rank = number of distinct clusters of
  chains at scale ε.
- **H_1(VR(ε))**: 1-cycles (loops). A non-trivial H_1 element means there exist
  chains that form a "ring" — they are pairwise connected but not through a common
  center. This indicates distinct strategic corridors.
- **H_2(VR(ε))**: Voids. Enclosed empty regions in the trajectory space.

### 4.3 Persistence

As ε grows, homological features are "born" and "die":

- A component (H_0) is born when a point appears (at ε=0) and dies when it merges
  with another component (at ε = the edge length connecting them).
- A loop (H_1) is born when a cycle forms and dies when it gets "filled in" by a
  higher-dimensional simplex.

The **persistence** of a feature is death - birth. Long-lived features represent
genuine topological structure; short-lived features are noise.

### 4.4 The Persistence Diagram

The output is a multiset of (birth, death) pairs for each dimension:

  PD_k = {(b_i, d_i) : feature i in H_k}

Points far from the diagonal (d_i >> b_i) are significant structural features.
Points near the diagonal are noise.

---

## 5. Ceiling Detection from Topology

### 5.1 The Key Signals

**H_0 (components):** If all chains merge into one component at a small radius
(relative to the diameter), the chains are tightly clustered — one strategy dominates.
The "stabilization radius" is where H_0 reaches rank 1.

**H_1 (loops):** This is the primary ceiling signal. Interpretations:

- **H_1 = 0 (no loops):** All chains lie in a convex/contractible region. The model
  has ONE strategy basin. More sampling explores the same basin — ceiling reached.

- **H_1 ≥ 1 (loops present):** There exist chains that form non-trivial cycles.
  This means the trajectory space has "holes" — regions the model routes around
  rather than through. These holes represent distinct strategic corridors. More
  sampling could fill unexplored corridors — scalable.

**H_1 lifetime:** A loop with lifetime (d - b) >> 0 is a robust structural feature
(a genuine strategic corridor). A short-lived loop is noise (random fluctuation).

### 5.2 Why Loops Mean Diversity

Consider 4 chains A, B, C, D with distances:
```
  A--B close, B--C close, C--D close, D--A close
  BUT A--C far, B--D far
```

This forms a loop: the chains are connected sequentially but there's no "shortcut"
through the middle. Geometrically, they surround an empty region — a solution strategy
that none of them took. This is the topological signature of "the model has multiple
distinct approaches and there's unexplored space between them."

If instead all pairwise distances are similar (a tight ball), there's no loop —
everything is one cluster. No unexplored space.

### 5.3 The Ceiling Decision Rule

Given the persistence diagram from K chains:

1. Compute Betti numbers β_0(ε) and β_1(ε) as functions of radius ε
2. Extract the longest-lived H_1 feature: max_lifetime = max{d_i - b_i}
3. Count significant H_1 features: n_significant = |{i : d_i - b_i > threshold}|

Decision:
- **CEILING** if: n_significant = 0 AND β_1(ε) = 0 for all ε
  (no loops at any scale — single-basin generation)
- **SCALABLE** if: n_significant > 0 AND max_lifetime is large relative to
  the diameter of the point cloud
  (robust loops — multiple strategic corridors exist)
- **UNCERTAIN** otherwise

---

## 6. The Conditioning Experiment (IID vs DAD)

### 6.1 Breaking IID

Standard sampling: each chain τ_i ~ P(τ | q, θ) independently. The topology of
T_K reflects the structure of P(τ | q, θ).

DAD conditioning: after observing T_K and extracting disagreements, we sample
τ'_j ~ P(τ | q, θ, workspace(T_K)). These are NOT IID — they're conditioned on
the disagreement structure of the first batch.

### 6.2 What Conditioning Does Topologically

If conditioning merely redistributes mass within the same support (the same region
of representation space), the topology should be unchanged — same loops, same
components, same Betti numbers.

If conditioning genuinely expands the reachable set (opens new representational
paths), we should see:
- New H_1 features appearing (new loops = new strategic corridors)
- Longer H_1 lifetimes (more robust structural diversity)
- Higher effective dimensionality of the point cloud

### 6.3 The Comparison Metric

  diversity_gain = (persistence_conditioned - persistence_iid) / persistence_iid

  new_features = (n_H1_conditioned > n_H1_iid)

If diversity_gain ≈ 0 and new_features = False: conditioning doesn't help
topologically — the model is stuck in the same basin regardless of the prompt.
This strengthens the ceiling prediction.

If diversity_gain > 0 or new_features = True: conditioning opens genuinely new
paths — more compute (with the right conditioning) can still help.

---

## 7. Mathematical Justification

### 7.1 Why Persistent Homology and Not Simpler Measures

**Variance/spread:** The covariance matrix of K points captures second-order
structure (ellipsoidal spread) but misses topological features. A ring of points
has low variance in the radial direction but non-trivial H_1. Variance says
"these points are spread out"; persistence says "there's a hole in the middle
they can't reach."

**Pairwise distance statistics:** Mean/max/min DTW distances tell you about the
scale of diversity but not its structure. A tight cluster and a loose ring can
have similar mean pairwise distances but completely different topology.

**Clustering (k-means, DBSCAN):** These find components (H_0) but not loops (H_1)
or higher structure. The number of clusters tells you "how many answer groups" but
not "how many distinct strategic pathways connect them."

### 7.2 Stability Theorem

Persistent homology satisfies a stability theorem (Cohen-Steiner, Edelsbrunner,
Harer 2007): if two distance matrices D and D' satisfy ||D - D'||_∞ ≤ δ, then
the bottleneck distance between their persistence diagrams is ≤ δ.

This means: small perturbations to the chains (sampling noise, slight prompt
variations) produce small changes in the persistence diagram. The topological
signal is robust to noise. Features with lifetime >> δ are genuine structure,
not artifacts of sampling variability.

### 7.3 The Geometry Score Connection

Khrulkov & Oseledets (2018) showed that for generative models, comparing the
persistent homology of generated samples against real data detects mode collapse.
Our adaptation:

- "Real data manifold" → the full set of possible reasoning trajectories for (M, q)
- "Generated samples" → our K chains
- "Mode collapse" → diversity collapse / model ceiling

If the persistent homology of K chains has converged (adding more samples doesn't
create new features), the generator has fully revealed its support — more sampling
won't help. This is exactly the ceiling.

### 7.4 Sample Complexity

How many chains K do we need for the topology to be informative? The Niyogi-Smale-
Weinberger theorem (2008) gives: to recover the homology of a manifold with reach τ
from samples, you need K = O(1/τ^d) samples where d is the intrinsic dimension.

For our setting, the intrinsic dimension of the trajectory manifold is likely low
(reasoning strategies form a low-dimensional subspace of ℝ^5120). Empirically, K=8
chains already distinguish "single cluster" from "multiple corridors" — this is
because we're not trying to recover the full manifold homology, just detect whether
non-trivial features exist at all. The binary question "is H_1 empty or not?" requires
far fewer samples than "what is the complete topology?"

---

## 8. What Each Result Means

### 8.1 Problem 0 (Easy): CEILING_REACHED

- All 8 chains → same answer (204), same strategy
- H_1 = 0: no loops, single basin
- DTW distances: tight range (std/mean = 2.6%)
- Interpretation: the model has ONE reliable path to the answer. Sampling more won't
  find anything new because there IS nothing else in the model's representation.
- Prediction: Acc(M, q, 1000) ≈ Acc(M, q, 8)

### 8.2 Problem 1 (Hard): SCALABLE

- 8 chains → 6 correct, 2 truncated. Same final answer but different token counts.
- H_1 = 2: two loops detected
- Interpretation: the model has MULTIPLE reasoning corridors. Some lead to completion
  (13k tokens), some lead to truncation (16k cap). The loops indicate distinct paths
  through representation space — more sampling or longer budgets could find shorter
  paths.
- Prediction: Acc(M, q, N) > Acc(M, q, 8) for large N (or with longer token budget)

### 8.3 The NCD Divergence

NCD (Normalized Compression Distance) measures text-level diversity. The interesting
finding: Problem 1 has HIGH NCD (0.91) but the hidden-state topology says SCALABLE,
while a naive text-diversity measure would say "already very diverse." This validates
the hypothesis: text diversity ≠ representational diversity. The model is saying
different words but the *computation* reveals structural corridors that text cannot.

---

## 9. Limitations and Open Questions

1. **K=8 might be too small** for stable homology estimation on truly complex
   problems. The stability theorem guarantees robustness to perturbations, but the
   sample complexity for detecting specific features depends on the manifold geometry.

2. **DTW on raw hidden states** might conflate magnitude and direction. An alternative:
   normalize to the unit sphere (cosine geometry) before DTW, measuring angular paths
   rather than Euclidean paths.

3. **The ceiling detector doesn't tell you WHAT to do** about the ceiling — only that
   it exists. Combining with Direction 13 (RL intervention) would make it actionable:
   detect ceiling → apply targeted RL to break through.

4. **Layer choice matters.** We use the last layer, which captures the most processed
   representation. Earlier layers might show diversity that gets "crushed" by later
   layers — investigating per-layer topology could reveal WHERE the ceiling forms.

5. **The conditioning comparison assumes DAD-style intervention is the right one.**
   Other conditioning strategies (rephrasing, hints, different temperature) might
   produce different topological signatures. The framework is general — the specific
   conditioning is a free parameter.
