# Astra (GPT-6) advice — converting DVR/coverage findings into an award-winning, scale-robust methodology
_Queried 2026-09-11 via Bedrock openai.gpt-6-astra, reasoning_effort=medium, 9149 output tokens. Context = PROGRESS_SNAPSHOT v6 + §120-§125. Prompt archived at /tmp/astra_prompt.txt._

---

The strongest version of this paper is not “RFT beats GRPO.” It is:

**For verifiable OOD learning, the relevant resource is transferable verified support—not the number of on-policy updates. Standard group-relative RL can fail to acquire or retain that support, and the resulting disadvantage persists at larger model sizes when the evaluation distribution has genuine headroom.**

That could support a major paper. But three gaps currently separate your evidence from that claim:

1. **Your success-gradient identity does not yet establish that the empirical gap is “not the objective.”** It establishes a local directional relationship under specific sampling and normalization conditions. Stale replay, token normalization, clipping, KL, prompt weighting, and GRPO normalization still matter.
2. **Your 7B hard result is currently stronger evidence of avoiding RL degradation than of learning from replay.** If base performance is approximately 0.57 and RFT is 0.567, the result is important, but it is not a large positive adaptation result.
3. **“Coverage” needs a precise operational definition.** Distinct prompts, distinct verified solutions, latent compositional mechanisms, and coverage of the evaluation distribution are different objects.

My recommendation is to build the paper around a **verified-support principle**, introduce a minimal replay mechanism that intervenes directly on the proposed failure, and make the decisive large-model experiment demonstrate both:

- a substantial advantage over a strong, non-starved RL baseline; and
- an absolute improvement over the starting checkpoint.

Below is the concrete path.

---

# A. What constitutes a convincing scaling story?

## A1. Headroom-gated scaling is scientifically sufficient—but your present evidence supports a narrower statement

A paper does not need a frontier model to be exceptional. A discovery that changes how we understand a widely used training procedure can be more important than a frontier benchmark improvement.

But distinguish these claims:

1. **Cross-size robustness:** DVR wins at the tested sizes.
2. **A size–difficulty interaction:** the gap shrinks on easy distributions and reappears on a harder distribution at the same size.
3. **A predictive scaling principle:** independently measured headroom predicts the gap on unseen size–difficulty combinations.
4. **Frontier relevance:** the mechanism matters for models and workloads representative of current expensive post-training.

You have evidence for 1 and an especially useful instance of 2. You do not yet have 3. Models up to 9B alone do not establish 4.

Do not write “returns at every scale where headroom exists” until you have tested that quantifier. Write:

> Across the tested model sizes, the replay advantage is suppressed near saturation and reappears on harder distributions; a headroom-and-support model predicts where this occurs.

The crucial upgrade is not necessarily 70B. It is **out-of-sample prediction plus a larger-model positive adaptation result**.

A coherent size × difficulty surface, including a 30–32B-class point, would often be more persuasive than a single expensive 70B result with weak controls. A 70B-class validation is valuable if affordable, but it should validate an established mechanism, not carry the entire paper.

## A2. Your 7B hard result needs an immediate accounting check

You report:

- base coverage approximately 57%;
- RFT approximately 56.7%;
- GRPO approximately 45.5%;
- GRPO training reward approximately 5%.

These numbers may be perfectly consistent, but only under distinctions that must be explicit.

Possible explanations include:

- 57% is pass@K, while 5% is pass@1;
- training and evaluation are different distributions;
- decoding budgets or temperatures differ;
- “coverage” means prompts with at least one success in a large collection budget;
- the floor occurs after training-induced degradation rather than at initialization.

If a per-sample success probability is 0.05, then an independent K=16 collection succeeds at least once with probability

\[
1-(1-0.05)^{16}\approx 0.56.
\]

That would almost exactly reconcile the numbers. But it changes the interpretation: **57% coverage is not 57% pass@1 capability**.

Moreover, with \(p=0.05\), the all-failure probability is:

\[
P_{\rm fail}(G)=(1-p)^G.
\]

This is about 0.66 at \(G=8\), but about 0.44 at \(G=16\). Those are substantial dead-group rates, not necessarily universal starvation. Heterogeneity across prompts can make the situation worse: many prompts may have \(p_x\approx0\), while a minority accounts for nearly all reward.

Before writing another headline, produce a single table with:

- train versus OOD split;
- pass@1 and fixed-budget pass@K;
- sampling temperature and token limits;
- average success probability;
- per-prompt success-probability distribution;
- observed all-fail, mixed, and all-success group rates;
- reward-gradient norm and KL-gradient norm;
- absolute change from base.

Also, **zero reward advantage does not itself explain degradation**. With a frozen reference and no other update, an all-fail group provides no reward update; it does not inherently damage the policy. Degradation must come from updates on other prompts, KL/reference choices, optimization, distributional interference, formatting shifts, or implementation issues.

The stronger mechanistic claim is:

> Dead groups remove restorative reward signals on low-success prompts, allowing updates elsewhere to degrade those capabilities; replay restores a persistent learning signal on previously verified support.

That claim is testable.

## A3. Do not manufacture failure; manufacture a controlled capability frontier

The right way to create headroom is to specify a difficulty-generating process before method comparison and evaluate all methods on it.

Your executable domain is unusually well suited to this.

### Primary large-model setup: compositional programs with interpreters supplied in context

Generate task families with independently adjustable axes:

- composition depth;
- number of interacting constraints;
- branching factor;
- state size;
- dependency distance;
- number of API calls;
- partial observability, if relevant;
- semantic novelty of operator combinations.

Use an explicit DSL or API whose semantics are provided to all methods. Randomize symbol names and selected semantics across independently generated “worlds,” while holding the underlying reasoning demands controlled.

The aim is not to make the model ignorant of undocumented syntax. It is to create **new compositional problems with executable ground truth**.

Use three separate generalization axes:

1. **Combinatorial OOD:** unseen combinations of primitives seen during training.
2. **Depth OOD:** longer compositions than training.
3. **World OOD:** new symbol mappings or semantic parameterizations, with their definitions supplied.

Do not combine all three into one uninterpretably hard benchmark. Report them separately.

Calibrate difficulty using base-model development measurements only. Freeze the evaluation distributions before running the post-training methods. Include both:

- **common distributions**, shared across model sizes;
- **capability-matched distributions**, chosen to place each base model in comparable pass@1 or pass@K bands.

The first prevents “each size gets a different benchmark” objections. The second separates size from saturation.

### Secondary large-model setup: executable multi-step tool use

Use a deterministic local environment with:

- a database, filesystem, or program state;
- 10–30 required dependent operations;
- an outcome verifier;
- held-out dependency structures;
- identical tool interfaces and inference budgets across methods.

This creates genuine long-horizon headroom without requiring subjective grading.

However, do not make agentic training the first large-scale expansion. It adds credit assignment, tool errors, trajectory-length variance, and interaction costs that can obscure the original mechanism.

### Math is a validation domain, not necessarily your main scaling vehicle

Use independently generated instances with derivations or robust exact-answer verification. Audit answer-only shortcuts and verifier exploits.

Math can broaden the result, but a strong executable-domain mechanism paper is better than a weakly controlled multi-domain paper.

## A4. Headroom is necessary, not sufficient

High headroom can arise because the model lacks the primitives needed to learn anything from your bank. In that regime, neither RFT nor GRPO will succeed.

Your real prediction should be:

\[
\text{replay benefit}
\approx
\text{evaluation headroom}
\times
\text{transferable verified support}
-
\text{regression}.
\]

A useful phase diagram has two principal axes:

- **evaluation headroom**;
- **train-side success accessibility**, including bank collection cost.

The four regimes are:

- low headroom, high accessibility: little measurable gain;
- high headroom, high accessibility: strongest expected replay gains;
- high headroom, low but nonzero accessibility: replay may help if successes can be amortized;
- high headroom, zero accessible support: replay cannot solve discovery.

That last boundary is essential. Replay does not create successes ex nihilo.

---

# B. The method: the best minimal delta

Do not invent a complicated acquisition policy to make an established procedure look novel. Your negative CTH result is strong evidence against making prompt-targeting the central algorithm.

I would prioritize one main candidate and keep a second conditional candidate.

## B1. Main candidate: Verified Support Floor

Call it **Verified Support Floor, VSF**.

The principle is:

> Once a training prompt has yielded a verified success, its contribution to learning should not subsequently disappear merely because current on-policy groups contain no successful sample.

This directly addresses your proposed mechanism and applies at any model size.

The method is a coverage-balanced, persistent replay channel attached to a competent on-policy learner. It is not “replace RL with RFT”; it is **prevent on-policy optimization from repeatedly discarding already-acquired verified support**.

### State

Maintain a bank

\[
\mathcal B_t=\{(x,\mathcal B_t(x))\},
\]

where \(\mathcal B_t(x)\) contains a bounded number of diverse verified solutions for prompt \(x\).

Also maintain:

- first discovery time;
- number of replay exposures;
- collection-policy version;
- current-model log likelihood or another staleness diagnostic;
- optional structural metadata from the task generator.

The central unit of coverage is the **prompt**, not the successful trajectory.

### Initialization

From the common starting checkpoint, collect an initial bank under a fixed, explicitly charged sampling budget.

Every comparison method receives one of two experimental treatments:

1. an identical initialization/checkpoint with bank collection included in end-to-end cost;
2. an identical precollected bank in an intervention study where access to that bank is the randomized variable.

Do not mix these two comparisons.

### Training loop

At each iteration:

1. **Fresh collection.** Sample prompts from the prescribed training distribution and generate standard on-policy groups.
2. **Verification.** Compute binary outcomes and group advantages.
3. **Archive update.** Add verified successes to the bank. Deduplicate and cap solutions per prompt.
4. **Live update.** Apply the chosen GRPO-style update on the fresh groups.
5. **Support-floor update.** Traverse covered prompts using a shuffled, prompt-balanced replay schedule. For each selected prompt, sample a verified solution uniformly or by a prespecified diversity rule.
6. **Bank refresh.** Replace stale or redundant solutions with fresh verified solutions when available, without deleting the last success for a prompt solely because it is old.

Use a combined objective such as

\[
L_t(\theta)
=
L_{\rm live}(\theta)
+
\lambda_t L_{\rm replay}(\theta)
+
\beta L_{\rm KL}(\theta),
\]

with

\[
L_{\rm replay}(\theta)
=
-\frac{1}{|\mathcal C_t|}
\sum_{x\in\mathcal C_t}
\mathbb E_{y\sim\mathcal B_t(x)}
[\log \pi_\theta(y\mid x)],
\]

where \(\mathcal C_t\) is the covered prompt set.

For the exact theoretical connection, \(\log\pi(y\mid x)\) is sequence log probability. Length normalization changes the objective; treat it as an explicit ablation, not a harmless implementation detail.

### What makes this more than replay appended to GRPO?

The actual algorithmic content is the **support constraint**:

> Every covered prompt receives at least \(m\) successful-sequence training exposures during each replay cycle.

Implement that by stratified traversal, not importance weights with high variance.

A simple fixed replay fraction should be the primary version. An adaptive version can increase replay when measured verified-support exposure falls below a target, but do not make an elaborate controller necessary to reproduce the result.

The key measurements are:

- fraction of covered prompts receiving successful updates in a window;
- minimum and lower quantiles of successful-update counts;
- concentration of total successful training mass;
- forgetting of previously solved prompts;
- OOD transfer per unit collection and training compute.

### Why it could beat GRPO

It converts a success discovered once into a reusable training signal, rather than requiring rediscovery inside future groups.

For a prompt with current success probability \(p_x\), fresh sampling needs order \(1/p_x\) attempts per successful sample. A verified bank removes that repeated acquisition cost for future replay updates, although not the initial discovery cost.

It also prevents the training distribution from implicitly concentrating on prompts that remain easy enough to generate mixed groups.

### Why it could beat vanilla RFT

Only one part can plausibly do this: **continued acquisition and refresh from an improving policy**, while maintaining balanced support on already-covered prompts.

That is a hypothesis, not a theorem. Fixed-bank DVR may remain best. Standard iterative ReST already performs recollection, so the fair comparator is strong iterative ReST, not only one-shot RFT.

If VSF fails to improve on iterative ReST, do not conceal that. The conclusion could still be:

> The support floor is sufficient to repair much of GRPO’s OOD deficit, and replay-only training is the more compute-efficient implementation in this regime.

That would be mechanistically powerful even without a new SOTA optimizer.

### The decisive VSF ablations

Use identical banks and total budgets:

- GRPO;
- GRPO plus random trajectory replay;
- GRPO plus prompt-balanced replay;
- GRPO plus the explicit support floor;
- one-shot prompt-balanced RFT;
- iterative prompt-balanced ReST;
- VSF with and without bank refresh.

The distinction between ordinary replay and a support floor may be small. If it is, your novel method is not the floor; your contribution is the identified principle. Accept that outcome.

### Novelty warning

RL with successful-trajectory replay, self-imitation, balanced replay, and iterative rejection sampling all have substantial prior art. A literature audit is mandatory before calling VSF algorithmically novel.

The defensible novelty would be:

- a specific support constraint;
- a causal account of its OOD effect;
- a predictive regime map;
- demonstrated scale transfer.

Not the mere presence of a buffer.

## B2. Conditional second candidate: structure-aware verified-bank compression

If you want a cleaner algorithmic improvement over vanilla RFT, target **which verified examples receive training compute**, not which unsolved prompts receive sampling compute.

This does not repeat your failed CTH intervention.

Call this **Compositional Support Replay, CSR**.

Each verified example covers structural features known from the generator:

- operator pairs or triples;
- control-flow motifs;
- dependency patterns;
- semantic interactions;
- depth–primitive combinations.

For a selected bank \(S\), define

\[
F(S)=
\sum_{j=1}^{d}
w_j
\left[
1-\exp\left(
-\frac{\sum_{e\in S}a_{ej}}{\tau_j}
\right)
\right],
\]

where \(a_{ej}\ge0\) records feature coverage.

This is a monotone submodular function. Under a cardinality constraint, standard greedy selection has a \(1-1/e\) approximation guarantee. With token-cost constraints, use a proper knapsack-constrained submodular algorithm; a casual gain-per-cost heuristic alone does not justify the same guarantee.

Train on the selected bank with prompt balancing.

### Why it might scale

As models improve, they may generate more redundant verified successes. Compression can preserve transferable structural support while reducing training tokens.

The scaling target becomes:

> Equal or better OOD performance with fewer replay tokens, with benefits increasing as bank redundancy grows.

That is different from “large models need more targeted collection.”

### The limitation

The theorem guarantees coverage of your declared features—not OOD generalization. The hard scientific question is whether those features explain transfer.

Run a cheap pilot:

- random bank subset;
- prompt-balanced subset;
- structural-coverage subset;
- loss/difficulty-matched subset;
- equal tokens, equal distinct prompts, equal success counts.

If CSR does not beat prompt-balanced sampling at 3B and 7B, kill it. Do not build the paper around a submodular guarantee that optimizes an irrelevant surrogate.

**Recommendation:** VSF first, CSR only if the pilot works. Do not prioritize off-policy correction or headroom curricula yet.

---

# C. Theory: precise statements worth proving

You want a short chain of results that explains the intervention, not a collection of generic bounds.

## C1. Correct the success-gradient interpretation

For a fixed prompt \(x\), define

\[
p_\theta(x)
=
\sum_y \pi_\theta(y\mid x)V(x,y),
\]

with a parameter-independent binary verifier. Under ordinary differentiation conditions,

\[
\nabla_\theta p_\theta(x)
=
p_\theta(x)
\mathbb E_{\pi_\theta(\cdot\mid x,V=1)}
[\nabla_\theta\log\pi_\theta(y\mid x)].
\]

Thus,

\[
\mathbb E_{\pi_\theta(\cdot\mid x,V=1)}
[\nabla_\theta\log\pi_\theta(y\mid x)]
=
\nabla_\theta\log p_\theta(x).
\]

This gives two different aggregate objectives:

\[
J_{\rm RL}(\theta)=\mathbb E_x[p_\theta(x)]
\]

versus the idealized fresh, prompt-normalized success objective

\[
J_{\rm success}(\theta)=\mathbb E_x[\log p_\theta(x)].
\]

Their within-prompt directions coincide, but their cross-prompt weighting differs. The latter puts relatively more weight on low-success prompts.

More importantly, a frozen bank drawn from policy \(\mu\) uses

\[
\mathbb E_{\mu(\cdot\mid x,V=1)}
[\nabla_\theta\log\pi_\theta(y\mid x)],
\]

which generally is not \(\nabla\log p_\theta(x)\) once \(\theta\) moves away from \(\mu\).

Your correct statement is:

> At the collection policy, normalized success imitation and binary policy gradient share a within-prompt direction. Their practical differences arise from prompt weighting, sampling support, replay age, and optimization—not from an inherent opposition between imitation and reward maximization.

Avoid “the gap must be procedural.” The identity alone does not prove that.

## C2. Dead-group probability and productive-update bounds

For independent binary outcomes within a group of size \(G\), the probability of a nonconstant reward group is

\[
a_G(p)=1-(1-p)^G-p^G.
\]

For group-centered binary advantages, both all-failure and all-success groups have zero reward advantage.

For small \(p\),

\[
a_G(p)=Gp+O(p^2)
\]

for fixed \(G\ge2\).

If \(n\) independent groups are sampled for the prompt, then the number of nonconstant groups is binomial:

\[
M\sim {\rm Binomial}(n,a_G(p)).
\]

This yields exact concentration and waiting-time statements. In particular,

\[
\Pr(M=0)=(1-a_G(p))^n.
\]

To obtain at least one nonconstant group with probability \(1-\delta\), one needs

\[
n\ge
\frac{\log\delta}{\log(1-a_G(p))},
\]

which scales as \(\log(1/\delta)/(Gp)\) for small \(p\).

Do not call \(M\) the statistical effective sample size without additional variance analysis. Call it the number of **reward-active groups**.

An important implication:

> Increasing group size reduces the frequency of dead groups, but at fixed generated-sample budget it does not remove the fundamental scarcity of successes.

That directly motivates a dynamic-resampling baseline.

## C3. Expected unnormalized group-centered gradient

For samples \(y_i\sim\pi_\theta\), rewards \(R_i\), scores \(s_i=\nabla\log\pi_\theta(y_i\mid x)\), and group mean \(\bar R\),

\[
\widehat g=
\frac1G\sum_{i=1}^{G}(R_i-\bar R)s_i
\]

satisfies

\[
\mathbb E[\widehat g]
=
\left(1-\frac1G\right)\nabla p_\theta(x).
\]

This is useful precisely because it is modest: group centering alone does not change the expected within-prompt direction.

Standard-deviation normalization, clipping, multi-epoch updates, and KL must be analyzed separately. Do not silently extend this identity to the full implementation.

Together, C2 and C3 say:

> A directionally sensible estimator can still allocate its finite update opportunities very poorly.

## C4. A verified-support occupancy theorem

Suppose prompt \(i\) has fixed success probability \(p_i\), and receives \(K_i\) independent collection attempts.

Its probability of entering the verified bank is

\[
q_i=1-(1-p_i)^{K_i}.
\]

Hence expected distinct verified prompt coverage is

\[
\mathbb E[C]=\sum_i q_i.
\]

Now condition on a bank of \(C\) covered prompts and \(R\) replay draws.

If replay samples trajectories uniformly and prompt \(i\) has bank mass \(w_i\), expected distinct prompts exposed during replay is

\[
\sum_{i=1}^{C}[1-(1-w_i)^R].
\]

Uniform prompt weights maximize this quantity over \(w\), by concavity. Stratified replay visits every covered prompt once after exactly \(C\) draws.

This cleanly separates:

1. **acquisition coverage**: which prompts ever yield a success;
2. **training exposure coverage**: which covered prompts actually contribute updates.

Your current claims appear to combine them. The paper should not.

There is also an allocation result: with known independent \(p_i\), the marginal expected distinct-prompt gain from one more sample is

\[
p_i(1-p_i)^{K_i}.
\]

Selecting the largest marginal gain is optimal for the separable integer-budget problem. But this is not a general OOD theorem, and unknown probabilities make exploration relevant. Given your CTH negative, put this in supporting theory rather than making it the main method.

## C5. Replay amortization: discovery versus reuse

A useful theorem or proposition should quantify the difference between:

- collecting new independent successful trajectories;
- repeatedly training on a previously discovered successful trajectory.

For fixed success probability \(p_i\), collecting \(m\) successes requires expected \(m/p_i\) attempts. Discovering one success requires expected \(1/p_i\) attempts, after which \(m\) replay exposures require no further verification sampling.

However:

> Repeated exposures are not \(m\) independent observations and cannot be assigned the same statistical information value.

The result supports compute amortization, not a claim that replay creates information. Your experiments must determine when repeated optimization of one or several successes transfers.

## C6. An honest OOD coverage bound

There is no distribution-free theorem saying that more distinct successful training prompts improve arbitrary OOD performance.

You need assumptions linking training support to evaluation tasks.

A workable model is a finite collection of transferable mechanisms \(z\in\{1,\ldots,M\}\), with OOD weights \(q_z\). Suppose:

- an example exposes one or more identifiable mechanisms;
- \(n_z\) is the effective number of informative training examples for mechanism \(z\);
- conditional error satisfies a learning bound
  \[
  e_z(n_z)\le \epsilon_z+c_z/\sqrt{n_z}
  \]
  when covered;
- uncovered mechanisms have error bounded by 1.

Then

\[
R_{\rm OOD}
\le
\sum_{z:n_z=0}q_z
+
\sum_{z:n_z>0}
q_z\min\left\{1,\epsilon_z+\frac{c_z}{\sqrt{n_z}}\right\}.
\]

This formalizes the support-versus-redundancy tradeoff. It does not establish that your chosen mechanisms are correct.

The empirical contribution is to show that structural coverage predicts transfer better than:

- bank size;
- total successful tokens;
- average training difficulty;
- base likelihood;
- average ID reward.

For tasks requiring several mechanisms jointly, use explicit interaction features or a compositional error assumption. Primitive coverage alone will not explain compositional generalization.

## C7. A headroom decomposition that cannot hide regression

Let \(B(x)\) indicate base success and \(M(x)\) post-training success under a fixed evaluation protocol. Define

\[
H=\Pr(B=0),
\]

\[
\alpha_M=\Pr(M=1\mid B=0),
\qquad
\beta_M=\Pr(M=0\mid B=1).
\]

Then exactly,

\[
{\rm Acc}(M)-{\rm Acc}(B)
=
H\alpha_M-(1-H)\beta_M.
\]

For replay \(R\) versus GRPO \(G\),

\[
{\rm Acc}(R)-{\rm Acc}(G)
=
H(\alpha_R-\alpha_G)
-
(1-H)(\beta_R-\beta_G).
\]

This is the right decomposition for your 7B hard result.

It tells you whether replay wins by:

- solving previously unsolved tasks;
- preserving previously solved tasks;
- both.

With stochastic decoding, estimate a consistent joint protocol, or use a probability-weighted decomposition; do not classify tasks using one noisy base sample and treat that label as truth.

A **predictive** headroom law requires modeling \(\alpha\) and \(\beta\) from quantities available before the target experiment. Defining transferable value as “observed gain divided by headroom” is tautological.

For example, fit a low-complexity model of acquisition and regression using:

- base pass-rate distribution;
- initial bank coverage;
- structural overlap with the target distribution;
- reward-active group rate;
- replay exposure concentration.

Fit on smaller models and some difficulty levels, then predict held-out larger-model cells without refitting.

That would be a genuine methodological finding.

---

# D. The killer experiment matrix

I would stage this rather than launch a giant undirected grid.

## D0. Establish three distinct comparison questions

Every figure should say which question it answers:

1. **Same data:** what changes when optimizer, normalization, or replay schedule changes?
2. **Same end-to-end compute:** what procedure produces the best OOD model?
3. **Same support access:** does giving GRPO the replay bank repair the deficit?

No single matching rule answers all three.

Count:

- generated tokens, including failed samples;
- verifier cost where material;
- forward/backward training tokens;
- collection and training GPU-hours;
- wall time;
- peak memory;
- bank storage.

Match token budgets as well as FLOPs where lengths differ. Offline collection is not free.

## D1. Priority zero: audit and repair the hard-task GRPO baseline

Use the existing 7B hard setting first.

Run:

- current GRPO;
- independently validated GRPO implementation/configuration;
- tuned lower learning rate and KL choices;
- larger groups at matched rollout-token budget;
- dynamic resampling or filtering that constructs informative groups, charging rejected samples;
- a reward-gradient diagnostic run with carefully separated KL effects;
- frozen base and continued-SFT controls where appropriate.

Use at least three independent training seeds for the key comparisons.

Record, by prompt:

- initial success probability;
- live group status over time;
- first verified success;
- bank membership;
- successful update exposures;
- change in held-out behavior.

**Go/no-go:** if a modest correction to GRPO eliminates the result, rebuild the claim around that fact. Do not scale a broken baseline.

## D2. Mechanism matrix: the highest-value causal experiment

At 1.5B and 7B, on a hard but learnable distribution, run these eight arms:

1. tuned GRPO;
2. reward-active-group resampling GRPO;
3. one-shot prompt-balanced RFT;
4. iterative prompt-balanced ReST;
5. GRPO plus trajectory-uniform replay;
6. GRPO plus prompt-balanced replay / VSF;
7. VSF with coverage-blocked bank;
8. VSF with random-removal bank matched to arm 7.

Use the same verified bank for bank-conditioned comparisons. Match replay tokens and control for solution length, collection likelihood, and per-prompt success multiplicity.

Three seeds gives 48 runs across the two sizes.

This is more valuable than another dozen small-model trick ablations.

The key prediction is not simply “VSF wins.” It is:

> Repairing verified-support exposure closes a substantial fraction of the OOD gap, and selectively removing transferable support reopens it.

Dynamic resampling is especially important. If it closes the gap at comparable compute, the result is about **informative sampling and support acquisition**, not a unique advantage of offline replay.

## D3. Size × difficulty surface

Choose one main model family with four sizes spanning your current range, for example 1.5B, 3B, 7B, and 9B.

Use three shared difficulty distributions:

- near-saturated;
- intermediate;
- hard but learnable.

Calibrate these by a pilot, not by assuming depth corresponds to equal difficulty across sizes.

Run three main methods:

- strongest tuned on-policy baseline;
- strongest iterative DVR/ReST baseline;
- VSF.

With three seeds:

\[
4\times3\times3\times3=108
\]

runs.

That is a substantial matrix. If compute forces cuts, preserve:

- all sizes at intermediate and hard difficulty;
- easy cells at 1.5B and 7B/9B;
- three seeds at the decisive larger-model cells.

Use your existing valid runs instead of rerunning them.

For a second family, test two sizes on intermediate and hard distributions, primarily to validate the sign and mechanism. Do not double the entire grid.

### What to plot

Not accuracy versus parameter count alone.

Plot:

1. replay–RL gap versus headroom;
2. gap versus reward-active group fraction;
3. gap versus transferable bank coverage;
4. absolute acquisition and regression rates;
5. quality–compute frontiers;
6. predicted versus observed held-out gaps.

A headroom-normalized gain can be supplementary, but it becomes unstable near zero headroom. Do not use it as the main metric.

## D4. Larger-model anchor: one family, two distributions, strong comparators

Choose a 30–32B-class model first.

Use:

- one common hard distribution shared with 7B/9B;
- one prespecified capability-matched harder distribution.

Run:

- frozen base;
- tuned GRPO;
- reward-active-group resampling GRPO;
- iterative DVR/ReST;
- VSF.

The base requires evaluation, not training. Four trainable methods × two distributions × three seeds gives 24 training runs.

The ideal result is:

- DVR or VSF improves materially above base;
- the replay advantage persists against non-starved RL;
- the predicted effect size is approximately right;
- the effect survives compute accounting.

A roughly ten-point gap would be compelling in your context, but do not turn “ten points” into a scientific threshold. A smaller, clean, accurately predicted improvement can be more important than a larger failure-induced gap.

### Hardware warning

Your current 7B/9B ZeRO-2 LoRA setup does not automatically extend to 32B.

ZeRO-2 shards optimizer states and gradients, but generally does not shard the frozen base parameters in the way needed here. A BF16 32B base alone is about 64 GB before activations and overhead, so it cannot simply be replicated on a 40 GB GPU.

You will need a tested combination of:

- parameter sharding, tensor parallelism, or pipeline parallelism;
- or quantized adapter training, applied consistently across methods.

Inference can use tensor parallelism separately.

Budget a systems pilot before committing the full matrix. Do not let different quantization or sharding regimes become an uncontrolled method comparison.

If the 32B result is clean and throughput allows it, add one 70B-class validation cell. Otherwise, stop at a strong 32B result and state the limit accurately.

## D5. Causal removal experiments that go beyond prompt identity

Your existing coverage removal is valuable, but a reviewer will ask whether the blocked bank removed unusually informative or difficult examples.

Construct banks matched on:

- verified trajectory count;
- token count;
- distinct prompt count, where relevant;
- base success probability;
- solution likelihood;
- composition depth;
- primitive frequency.

Then vary **structural relationship to the OOD target**.

For example:

- bank A covers many prompts but omits a target-relevant operator interaction;
- bank B has the same volume and prompt count but includes that interaction;
- bank C substitutes an interaction irrelevant to the target.

The critical interaction is:

> Removing support hurts most on the OOD subsets predicted to require that support.

This turns “some examples matter more” into a transfer mechanism.

Use multiple independently constructed matched banks. One hand-designed removal is not enough.

## D6. Falsification tests

Pre-register these expectations:

### High success, low headroom

The replay advantage should shrink. If it does not, headroom is not the main explanation.

### High headroom, no accessible successes

Replay should not help substantially without a new source of support. If it does, identify whether another mechanism or data source is responsible.

### High headroom, poor structural overlap

Large bank coverage should not guarantee OOD improvement.

### GRPO given persistent verified support

The gap should shrink. If it does not, the current procedural account is incomplete.

### Group-size changes at fixed generation cost

Reduced dead-group frequency should not automatically yield proportionate OOD gains. This distinguishes dead-group rate from total useful support.

### Equal covered prompts, different multiplicities

Beyond some point, redundant successes should have diminishing value relative to new transferable support.

### Stale versus refreshed bank

If stale replay is the key failure at large scale, VSF should benefit from refresh; if refresh has no effect, simplify the method.

### Same checkpoint, different starting policies

Include at least one stronger instruction- or reasoning-tuned starting checkpoint if feasible. A failure restricted to a particular base checkpoint is much less general.

## D7. Predictive-law protocol

Use a genuine train/test split over experimental conditions.

For example:

- fit on 1.5B/3B and easy/intermediate conditions;
- validate model specification on one 7B development condition;
- freeze coefficients;
- predict 7B/9B hard and 32B conditions.

Compare against simple baselines:

- size only;
- headroom only;
- mean train success only;
- bank size only;
- headroom plus transferable support.

Report uncertainty and prediction error, not only \(R^2\). Avoid a flexible regressor with more explanatory variables than independent experimental cells.

Treat task templates or generated worlds as statistical clusters. Thousands of examples derived from a few templates do not provide thousands of independent pieces of evidence.

## D8. Allocation of the 72 GPUs

I would allocate GPU-hours approximately as follows, revising after measured throughput:

- 10%: baseline audit, systems pilots, evaluator validation;
- 30%: causal mechanism matrix;
- 25%: size–difficulty surface;
- 25%: larger-model anchor;
- 10%: replication, math validation, unexpected falsifications.

For scheduling, separate generation and training pools for the larger models, but size them from measured tokens/second and replay demand. A fixed server split chosen before profiling can waste a large fraction of the cluster.

Do not spend the last 10% until the main results are analyzed. You will need it for the reviewer-obvious control you did not initially anticipate.

---

# E. Positioning, contributions, and the main failure risk

## E1. Recommended title and claim

**Title: “Verified Support, Not Just On-Policy Updates: Scaling OOD Post-Training with Persistent Replay”**

**One-sentence claim:**

> In controlled verifiable tasks, OOD post-training is governed by the acquisition and preservation of transferable verified support; replay-based training outperforms group-relative outcome RL when sufficient headroom and accessible support coexist, and a support-preserving hybrid tests and repairs the underlying failure.

After successful larger-scale and predictive experiments, you can sharpen “tests” to “repairs” without qualification.

Avoid “RFT dominates RL” in the title. Your results concern specific algorithms, budgets, distributions, and starting checkpoints. Broad RL claims invite easy counterexamples.

## E2. Three contributions

1. **A scale-resolved phenomenon.**  
   A controlled size–difficulty study showing that apparent diminishing returns to verified replay are partly saturation effects, with the advantage reappearing on harder distributions and validated by held-out predictions.

2. **A causal support mechanism.**  
   Matched-bank removal, replay-support interventions, and group-activity measurements distinguish acquisition coverage, training exposure, and transferable structural support from bank volume and nominal objective choice.

3. **A minimal algorithmic repair and regime theory.**  
   A persistent, prompt-balanced verified-support floor repairs a substantial part of the on-policy OOD deficit, with exact signal-availability results and a headroom/acquisition/regression model that predicts both successes and failures.

If VSF does not beat strong iterative ReST, contribution 3 should say so. You can still present the hybrid as a mechanistic intervention rather than a superior final recipe.

## E3. The single biggest risk

**The paper is actually demonstrating a disadvantaged GRPO baseline against a replay method that received a more favorable success-discovery budget.**

That risk includes several versions:

- GRPO groups are unnecessarily starved;
- offline collection uses substantially more generation compute;
- the bank comes from a stronger or different policy;
- decoding budgets differ;
- hyperparameters cause avoidable collapse;
- the replay result merely preserves base capability while GRPO damages it.

These are not peripheral objections. Any one can collapse the central interpretation.

## E4. The experiment that de-risks it

At the 7B hard setting and one larger-model hard setting, run a **compute-accounted support intervention**:

1. Start every method from the identical checkpoint.
2. Collect one verified bank with an explicit, charged budget.
3. Compare:
   - tuned GRPO without replay;
   - dynamically resampled, reward-active GRPO without replay;
   - GRPO with access to the identical bank through VSF;
   - iterative DVR/ReST using that bank.
4. Equalize end-to-end resource budgets and report full quality–compute curves.
5. Evaluate:
   - absolute improvement over base;
   - acquisition of previously unsolved tasks;
   - regression on previously solved tasks;
   - support exposure and structural transfer.
6. Repeat with coverage-blocked and matched random-removal banks.

The strongest possible pattern is:

- competent GRPO remains worse than replay;
- access to persistent verified support closes much of the difference;
- selectively removing transferable support reopens it;
- the effect appears at larger size with positive improvement over base;
- its magnitude is predicted from measurements made before training.

That is the paper.

---

# Final recommendation

Do not chase another small algorithmic trick. Your abandoned interventions already suggest that the stable phenomenon is more important than those tricks.

For the next experimental cycle, prioritize exactly this sequence:

1. **Reconcile the 57% coverage and 5% reward metrics.**
2. **Show the 7B result survives a strong, non-starved GRPO baseline and honest compute accounting.**
3. **Give GRPO the identical verified support and test whether the gap closes.**
4. **Separate acquisition from preservation using the headroom decomposition.**
5. **Predict held-out size–difficulty cells, rather than retrospectively fitting the curve.**
6. **Obtain one larger-model result with a real positive gain over base.**
7. **Only then decide whether VSF is a new winning method or a causal intervention supporting DVR as the recommended procedure.**

The award-level opportunity is not a new name for rejection sampling. It is a quantitatively predictive answer to a consequential question:

**When does expensive on-policy post-training fail to turn verifiable experience into generalization, and what minimal change makes that experience persist and transfer?**
