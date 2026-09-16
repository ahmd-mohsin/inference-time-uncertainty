# EXPLAIN — paste-and-understand scratchpad

**How to use:** paste any paragraph from the paper (or any math) under the next `## PASTE` heading.
I'll fill in a plain-English explanation right below it, spelling out every math symbol in
words, with a one-line "so what" at the end.

Legend I'll use when decoding symbols:
- I write each symbol, then `=` , then what it means in words.
- `→` means "leads to" / "becomes". `≈` means "about equal". `∝` means "grows with".
- Greek letters get named (e.g. `θ = theta = the model's weights`).

---

## PASTE (intro paragraph)

> Large language models are increasingly post-trained on tasks with an automatic verifier—an executor or exact-match checker V(x, y) ∈ {0, 1} that decides whether a sampled solution y to a problem x is correct without human labels. This regime (code that must pass unit tests, math whose final answer must match) has produced the two dominant recipes for verified self-improvement. The first is on-policy outcome reinforcement learning (GRPO and relatives), which optimises the binary reward directly on the model's own rollouts [8]. The second is decoupled verified replay—rejection-sampling fine-tuning (RFT/STaR/RAFT), which samples a bank of verified-correct solutions and fine-tunes on them by supervised cross-entropy [10, 4].
>
> A subtle but decisive measurement issue governs how these recipes are judged. The community almost always reports coverage, or pass@k—the probability that at least one of k sampled attempts is correct [2]. But a deployed model answers once: the quantity that matters is single-attempt reliability, pass@1 = E_x[p_θ(x)] with p_θ(x) = E_{y∼π_θ}[V(x, y)]. Coverage = E_x[1 − (1 − p_θ(x))^k] upper-bounds pass@1 and can be arbitrarily larger. Re-analysing all of our verified-training runs through the honest single-attempt lens exposes three facts that jointly motivate the theory and method developed in the following sections.

## EXPLANATION

**The big picture:** People train LLMs on problems where a computer can automatically check
if the answer is right (run the code, compare the final number). There are two popular ways to
do this. The paper's point in the second half: everyone *measures success the wrong way* — they
report a number that flatters the model. Here's the whole thing decoded.

### Symbols, one at a time
- `x` = a problem (a coding task, a math question).
- `y` = a solution the model wrote (one attempt / one sample).
- `V(x, y)` = the **verifier**: a function that takes a problem `x` and an answer `y` and returns
  a verdict. `∈ {0, 1}` = "its output is either 0 or 1" — `1` = correct, `0` = wrong. No human
  needed; it's just "did the code pass the tests" or "does the final answer match."
- `θ` = theta = the model's **weights** (its trained parameters).
- `π_θ` = pi-sub-theta = the **model itself** viewed as a sampler — "the thing that, given a
  problem, produces answers." The subscript `θ` just means "with the current weights."
- `y ∼ π_θ` = "an answer `y` drawn (sampled) from the model." The `∼` means "is sampled from."
- `E` = **expected value** = average. `E_x[...]` = "average over problems `x`."
  `E_{y∼π_θ}[...]` = "average over the answers the model samples."
- `k` = how many attempts you let the model make at each problem.

### The two training recipes (first paragraph)
- **On-policy outcome RL (GRPO):** let the model try problems, and directly reward the tries
  that the verifier marks correct. It learns from its *own* live attempts. `[8]` is a citation.
- **Decoupled verified replay (RFT / STaR / RAFT):** first collect a pile ("bank") of the
  model's answers that the verifier said were correct, then just do ordinary supervised
  training (imitate those correct answers). "Cross-entropy" = the standard next-word training
  loss. "Decoupled" = collecting and training are separate steps, not live.

### The measurement problem (second paragraph — the important part)
Two ways to score a trained model:

- **pass@k / coverage** = "give the model `k` tries; did *at least one* come out correct?"
  This is what almost everyone reports.
- **pass@1** = "the model answers **once** — is that single answer right?" This is what actually
  matters, because a deployed model doesn't get 256 tries; it answers the user once.

Now the formulas:
- `p_θ(x) = E_{y∼π_θ}[V(x, y)]` = for one problem `x`, the model's **per-problem success rate** =
  "if I sample one answer, what's the probability it's correct?" (Average the verifier's 0/1
  verdict over the model's sampled answers → a number between 0 and 1.)
- `pass@1 = E_x[p_θ(x)]` = average that per-problem success rate over all problems. Plain: **the
  chance a single attempt is right, averaged across problems.**
- `coverage = E_x[1 − (1 − p_θ(x))^k]` = the pass@k formula. Read the inside:
  - `(1 − p_θ(x))` = chance one attempt is **wrong**.
  - `(1 − p_θ(x))^k` = chance **all `k`** attempts are wrong (multiply the wrong-chance by itself
    `k` times — the `^k` means "to the power k").
  - `1 − (that)` = chance **at least one** of the `k` is right.
  - `E_x[...]` = average that over problems.

**The punchline:** `coverage ≥ pass@1`, and "can be arbitrarily larger." Why: with more tries
(`k` big) you almost always stumble onto a correct answer eventually, so coverage creeps toward
1 — even if the model's *single* shot is usually wrong. Example: if a single try is right only
10% of the time (`p_θ = 0.1`), then pass@1 = 0.10, but pass@256 ≈ 1.0. Coverage looks like a
triumph; the model is still wrong 9 times out of 10 when it counts.

### So what (why this paragraph exists)
The authors re-scored **all** their training runs using the honest one-shot number (pass@1)
instead of the flattering many-tries number (coverage). Doing that reveals **three facts** (in
the next section) that motivate the paper's whole theory and method (RVP). In one line: *the
field has been grading on coverage, which hides that single-attempt reliability barely moved —
and closing that specific gap is what this paper is about.*

---

## PASTE (Fact 1 — GRPO ≈ null, RFT wins)

> We trained matched base / GRPO / RFT checkpoints (and VSF, a verified-replay floor added to the GRPO loss as a causal probe) from identical initialisations and evaluated pass@1 with an unbiased per-attempt estimator (k samples per problem, temperature 0.8; the same held-out problems for every arm). Across domains and model sizes the ranking is stark and repeatable (Table 1, Fig. 1): GRPO moves single-attempt reliability almost nothing off the base model (+0.001 on compositional code, +0.02 on GSM8K, +0.002 on the weak deepseek model), whereas decoupled verified replay (RFT) delivers a large, genuine acquisition gain (2.4× base on the code cell). The near-null of outcome RL is not a tuning artefact—it holds under doubled group size and across three independent recipients—and it is the first motivating puzzle: why does replaying verified successes acquire reliability that rewarding the very same successes on-policy does not?

## EXPLANATION

**One-line takeaway:** in a fair head-to-head, **on-policy RL (GRPO) barely improves the
one-shot answer, while copying verified-correct solutions (RFT) improves it a lot** — and the
paper's whole puzzle is *why*.

### The setup, decoded
- **"matched checkpoints from identical initialisations"** = they took one starting model and
  made several copies, then trained each copy a different way, so any difference is caused by the
  *method*, not by luck in where they started. Fair fight.
- The arms (the different training methods being compared):
  - **base** = the untrained starting model (the control).
  - **GRPO** = on-policy outcome RL: let the model attempt problems live and reward the attempts
    the verifier marks correct.
  - **RFT** = decoupled verified replay: collect a pile of the model's *already-correct* answers,
    then plain supervised-train on them (copy them).
  - **VSF** = "verified-replay floor" — a diagnostic arm: they bolt the replay signal onto the
    GRPO loss to test *causally* whether adding replay is what helps. It's a probe, not a product.
- **"unbiased per-attempt estimator (k samples per problem, temperature 0.8)"** = how they
  measure pass@1 honestly. For each problem they draw `k` sample answers at sampling temperature
  0.8 (0.8 = moderately random, so samples vary), and estimate the true single-attempt success
  probability from those `k` draws without bias. `k` here is just for *measuring* reliability
  accurately — not giving the model `k` tries to win (that would be coverage, the flattering
  number from the last paragraph).
- **"the same held-out problems for every arm"** = all methods graded on the identical unseen
  test set → apples-to-apples.

### The result (the numbers)
- **GRPO ≈ does nothing to pass@1:** +0.001 on compositional code, +0.02 on GSM8K, +0.002 on
  the weak DeepSeek model. Those are essentially zero — rewarding correct rollouts on-policy did
  **not** make the single shot more reliable.
- **RFT wins big:** "2.4× base on the code cell" = single-attempt success rate is **2.4 times**
  the base model's. Copying verified-correct solutions genuinely *acquired* reliability.

### "Not a tuning artefact" — why they trust it
Two robustness checks so nobody can dismiss the GRPO null as "you just tuned it badly":
- **"holds under doubled group size"** — GRPO compares a *group* of sampled answers per problem
  to judge which are better; they doubled that group size (a key GRPO knob) and GRPO still did
  nothing. So it's not an under-powered-group issue.
- **"across three independent recipients"** — they saw the same null on three different base
  models. So it's not one quirky model.

### The puzzle this sets up (why the paragraph exists)
Here's the strange part: **GRPO and RFT learn from the *same* correct solutions.** GRPO rewards
them on-policy; RFT replays them by imitation. Same successes, opposite outcomes — RFT acquires
reliability, GRPO doesn't. **Why does *replaying* a verified success teach the model something
that *rewarding* the very same success does not?** That "first motivating puzzle" is what the
theory section goes on to answer (the gradient-identity results: RL's gradient can be near-zero
exactly where it should be learning, while replay's cross-entropy gradient is not).

---

## PASTE (Fact 2 — the coverage−pass@1 gap, "reachability is not reliability")

> The second, and central, motivating fact is that even the winning recipe leaves most of the achievable reliability on the table. After RFT the model can almost always produce a correct solution within k samples (high coverage), yet a single sample is correct far less often (low pass@1). Table 2 and Fig. 2 report this coverage−pass@1 gap across nine cells spanning three model families (Qwen-Coder, Qwen, deepseek-coder), sizes from 1.3B to 7B, and two domains. The gap is large and systematic (often 0.20–0.50), and it tracks the reliability headroom: it is widest where the base model is weak and shrinks toward zero only when the model is already near-ceiling (GSM8K-3B, base 0.83). In other words, the correct solution is reachable but the post-RFT model still spreads probability mass across incorrect completions, so it will not be selected on the single attempt that matters. Reachability is not reliability.
>
> Table 2: The reliability gap after decoupled verified replay: coverage (pass@k, k=8–16) greatly exceeds single-attempt reliability (pass@1) for the same RFT checkpoint, across families, sizes, and domains. The gap tracks headroom—large for weak/hard cells, near zero only at ceiling. This residual is the opportunity our method targets.

## EXPLANATION

**One-line takeaway:** even after the *winning* method (RFT), the model **knows** the right answer
(it produces it within a few tries) but **won't say it first** (one shot is often wrong). That
leftover gap — "reachable but not reliable" — is exactly what this paper's method goes after.

### The core idea in everyday terms
Imagine a student who, given 8 attempts, almost always gets the problem right somewhere in those
8 — but if you demand a single answer, they're right less than half the time. They *have* the
knowledge; they just don't reliably put it first. That's the post-RFT model.

- **coverage (pass@k)** = "within `k` tries, at least one correct" — here **high**.
- **pass@1** = "the single answer is correct" — here **much lower**.
- **coverage − pass@1 gap** = the difference between the two. A big gap means: the correct
  solution is *in there* (reachable) but the model spreads its probability across many wrong
  completions too, so a single draw often lands on a wrong one.

### The evidence (what Table 2 / Fig. 2 show)
- **9 cells** = every combination of {3 model families: Qwen-Coder, Qwen, deepseek-coder} ×
  {sizes 1.3B → 7B} × {2 domains: code, math}. "Cell" = one such combination.
- **Gap is large and systematic: often 0.20–0.50.** I.e. pass@1 is 20–50 percentage points below
  coverage — not a fluke, it shows up everywhere.
- **"It tracks the reliability headroom"** = the gap is *biggest* where the base model is *weak*
  (lots of room to improve) and *shrinks to ~zero* only when the model is already almost perfect
  — their example **GSM8K-3B, base 0.83**: base already gets 83% single-shot right, so there's
  little headroom and almost no gap left to close.

### Why this is the "central" fact (why the paragraph exists)
It reframes the problem. The bottleneck after RFT is **not** "the model can't solve it"
(it can — coverage is high). The bottleneck is **selection**: the model doesn't concentrate its
probability on the correct solution, so the one attempt that actually gets deployed is often
wrong. **"Reachability is not reliability."**

This is the precise opening for the paper's method (RVP): don't try to *acquire* new solutions
(RFT already made them reachable) — instead **sharpen selection** so the reachable-correct answer
becomes the *single most likely* answer, converting that 0.20–0.50 headroom into pass@1. And note
the honest boundary already baked in: where there's **no gap** (near-ceiling cells like
GSM8K-3B), there's nothing to convert — which is exactly why the method's wins are headroom-gated
(and why our bigger-model matrix deliberately targets medium-difficulty cells that still have this
gap). This connects straight to the [[rl-focus-moderate-difficulty-benchmarks]] framing.

---

## PASTE (Fact 3 — mechanistic cause: neither recipe suppresses verified-WRONG modes)

> The gap has a mechanistic cause: existing training does not suppress verified-incorrect modes. Why does verified replay stop short? We answer this mechanistically by teacher-forcing each checkpoint over the same held-out set of the model's own verified-correct (y+) and verified-incorrect (y−) samples and measuring the average per-token log-probability of each, together with the correct-vs-incorrect logit margin m = log π_θ(y+) − log π_θ(y−) (Table 3, Fig. 3). Two clean signatures emerge. First, GRPO does not move the margin at all: its log π(y+), log π(y−), and m are identical to the base model—the distributional face of its ≈ 0 pass@1 gain, and consistent with the fact that on-policy signal only reaches prompts the model already solves. Second, RFT is mode-covering: it raises the log-probability of the correct solution (−0.110 → −0.092) but raises the log-probability of the incorrect solution by essentially the same amount (−0.139 → −0.119), so the margin barely changes (0.029 → 0.027). Positive-only imitation lifts correct and incorrect modes together; it structurally cannot down-weight the verified-wrong completions that steal single-attempt mass. This is precisely why the reliability gap in §2 persists: neither dominant recipe touches the selection axis—the reallocation of probability from verified-incorrect to verified-correct on the same prompt.

## EXPLANATION

**One-line takeaway:** they open the hood and find *why* the gap survives — RFT makes the model
like the correct answer more, but it likes the **wrong** answer *equally* more, so the model is no
better at *choosing* the right one. GRPO doesn't move anything at all. Neither touches "selection."

### How they measured it (the method)
- **teacher-forcing** = feed the model a *fixed* answer token-by-token and read off how probable
  the model thinks that exact answer is. You're not letting it generate — you're asking "how much
  do you believe THIS string?" Done for two kinds of strings on the same prompts:
  - **y+ = verified-correct** samples (answers the checker marked right).
  - **y− = verified-incorrect** samples (answers the checker marked wrong) — the model's *own*
    past outputs, so it's a fair internal comparison.
- **log π_θ(y)** = log-probability the model (weights θ) assigns to answer `y`. Higher (closer to
  0, since logs of probabilities are negative) = the model believes it more. "per-token average"
  just normalizes for length.
- **the margin `m = log π_θ(y+) − log π_θ(y−)`** = how much more the model prefers the correct
  answer over the wrong one. **This is the selection signal.** `m` big and positive → the model
  reliably puts the right answer first (high pass@1). `m ≈ 0` → correct and wrong are neck-and-neck
  → a single draw is a coin-flip → low pass@1. **pass@1 lives or dies on the margin.**

### The two findings
1. **GRPO moves nothing.** Its `log π(y+)`, `log π(y−)`, and `m` are *identical to the base
   model*. That's the distribution-level explanation of its ≈0 pass@1 gain from Fact 1. Why?
   On-policy reward only reaches prompts the model *already* solves — so there's no gradient where
   it matters, and the margin never changes.
2. **RFT is "mode-covering" — the key finding.** It lifts *both* peaks by nearly the same amount:
   - correct answer: log-prob **−0.110 → −0.092** (up a bit).
   - wrong answer:  log-prob **−0.139 → −0.119** (up by *almost the same* amount).
   - margin: **0.029 → 0.027** (essentially unchanged — even slightly down).

   So RFT raised the model's confidence in the right answer **and in the wrong answer together**.
   Net effect on *choosing*: zero. "Mode-covering" = it spreads mass to cover all the modes it
   was trained on (it only trains on positives, but generation still visits nearby wrong modes and
   they ride up too).

### Why this nails the puzzle (why the paragraph exists)
- **RFT trains only on positives** → by construction it can *raise* the correct mode but has **no
  mechanism to push the verified-wrong mode down**. Those wrong completions are what "steal
  single-attempt mass" (grab probability that should go to the correct answer).
- Both dominant recipes ignore the **selection axis** = *reallocating probability from
  verified-incorrect to verified-correct on the same prompt*. GRPO doesn't reach it; RFT lifts
  both sides equally.
- **This is the exact hole RVP fills:** RVP does preference training on (y+, y−) pairs, whose
  gradient *subtracts* the wrong-answer's gradient — it **pushes log π(y−) down** while pulling
  log π(y+) up, so the **margin `m` grows**. Growing `m` is what finally converts the reachable
  correct answer into the single most-likely one → the coverage−pass@1 gap closes. The whole
  method is "move the margin," and this section proves nothing else in the field does. Connects to
  the RVP gradient-identity theory (Δm = η·β·σ(−u)·‖∇log π(y+) − ∇log π(y−)‖² ≥ 0).

---

## PASTE (The Opportunity — the three facts converge into the method's thesis)

> The opportunity. These three facts define the gap our work targets. (i) On-policy outcome RL acquires almost no single-attempt reliability (Table 1); (ii) even the best recipe, decoupled verified replay, leaves a large coverage−pass@1 gap that scales with headroom (Table 2); and (iii) the mechanistic reason is that both recipes fail to suppress the verified-incorrect modes—they optimise coverage (reachability) and leave selection (reliability) untouched (Table 3). Existing preference methods use verifier-labelled pairs for alignment [7, 6, 9] but have not been framed against, or evaluated on, this post-replay reliability residual. This motivates the theory and methodology developed next: a decoupled, verified-preference objective that directly reallocates probability mass from verified-incorrect to verified-correct solutions the model can already reach, targeting the single axis—selection—that positive-only imitation and on-policy reward provably leave open.

## EXPLANATION

**One-line takeaway:** this paragraph is the "so here's the plan" — it stacks the three findings
into one argument (there's leftover reliability, and nobody's method goes after it the right way)
and names the fix: a method that **pushes wrong answers down and right answers up on the same
prompt** — the one thing the field has left untouched.

### The three facts, restated as a chain
- **(i)** On-policy RL (GRPO) → almost no pass@1 gain. *Rewarding* correct rollouts doesn't build
  single-attempt reliability. (Fact 1 / Table 1.)
- **(ii)** The winning method (RFT) → still leaves a big **coverage − pass@1 gap**, and that gap
  is *bigger where there's more room to improve* ("scales with headroom"). The model can reach the
  answer but doesn't reliably pick it. (Fact 2 / Table 2.)
- **(iii)** The reason, mechanistically: both methods **optimise coverage (reachability) but leave
  selection (reliability) untouched** — neither pushes the verified-*wrong* answers down. (Fact 3 /
  Table 3.)

Read together: **there is a real, measurable pile of reliability sitting unclaimed after RFT, and
it exists specifically because no standard recipe moves the selection axis.**

### The gap in the literature (why this is novel)
- **"Existing preference methods use verifier-labelled pairs for alignment [7,6,9]"** = yes, DPO
  and friends already train on (good, bad) pairs — but for *alignment* (helpfulness, safety,
  human-preference tuning).
- **"…have not been framed against, or evaluated on, this post-replay reliability residual"** =
  nobody has pointed those preference tools *at this specific leftover* — the coverage−pass@1 gap
  that remains **after** an RFT replay stage, measured in **pass@1**. That framing (and eval) is
  the paper's claim to newness: same tool, unclaimed target.

### The method it sets up (RVP, in one breath)
- **"decoupled"** = a separate stage after RFT (not fused into on-policy RL). RFT first makes the
  correct answer *reachable*; then this stage runs.
- **"verified-preference objective"** = preference training (DPO-style) on **verifier-labelled**
  pairs from the model's *own* samples: y+ = verified-correct, y− = verified-incorrect.
- **"directly reallocates probability mass from verified-incorrect to verified-correct solutions
  the model can already reach"** = the exact mechanism from Fact 3 — push `log π(y−)` **down**,
  pull `log π(y+)` **up**, so the margin grows and the reachable-correct answer becomes the single
  most-likely one. "Can already reach" is the crucial guard: it only re-ranks solutions RFT made
  available — it isn't trying to acquire new ones.
- **"targeting the single axis—selection—that positive-only imitation and on-policy reward
  provably leave open"** = it aims at the one lever the other two methods structurally can't move
  (RFT lifts both modes equally; GRPO moves nothing), and the next section *proves* this (the
  gradient-identity theory).

### Why the paragraph matters
It's the hinge of the paper: motivation → method. It also bakes in the honesty that runs through
this whole project — the opportunity is **headroom-gated** (fact ii says the gap scales with
headroom; at ceiling there's nothing to claim), which is exactly why our 72-GPU matrix targets
medium-difficulty cells with a live gap, and why SKIP-RFT / near-ceiling runs are *expected* to
show little. RVP is a selection method for the reachable-but-not-reliable regime — no more, and
no less. See [[rl-focus-moderate-difficulty-benchmarks]].

---

## PASTE (Theory setup — definitions, symbols, the three operators)

> A policy π_θ(y | x) generates a solution y to a problem x. A verifier V(x, y) ∈ {0, 1} is parameter-independent: an executor or exact-match checker that does not depend on θ (A1). Define the per-prompt success mass and the population objective (single-attempt reliability), p_θ(x) = E_{y∼π_θ(·|x)}[V(x, y)], J(θ) = E_{x∼D}[p_θ(x)] = pass@1, and coverage cov_k(θ) = E_x[1 − (1 − p_θ(x))^k]. Write C(x) = {y : V(x, y) = 1} (verified-correct) and I(x) (verified-incorrect). From a base checkpoint θ0 define headroom h(x) = 1 − p_θ0(x) and, at sampling budget C, accessibility a(x; C) = 1 − (1 − p_θ0(x))^C. For a self-generated pair (y+ ∈ C, y− ∈ I) the logit margin is m_θ(x) = log π_θ(y+ | x) − log π_θ(y− | x). Three operators are trained from the shared θ0: GRPO (on-policy policy-gradient on V), RFT (multi-epoch cross-entropy on a verified bank B = {(x, y) : V = 1} sampled from π_θ0), and VSF (GRPO plus a persistent prompt-balanced replay floor over B, used as a causal probe).

## EXPLANATION

**One-line takeaway:** this is the paper's dictionary — it defines every symbol precisely so the
theorems can be stated. Nothing surprising happens here; it's naming the pieces. Below, each
symbol in words.

### The actors
- **π_θ(y | x)** = the model ("policy"), with weights **θ** (theta). Given a problem **x**, it
  produces a solution **y**. `π_θ(y|x)` = the probability the model assigns to producing solution
  `y` for problem `x`.
- **V(x, y) ∈ {0, 1}** = the **verifier**: checks solution `y` to problem `x`, returns `1`
  (correct) or `0` (wrong). An executor (run the code) or exact-match checker (does the final
  answer match).
- **(A1) "parameter-independent … does not depend on θ"** = Assumption 1: the verifier is a fixed
  external judge — it does **not** change as the model trains. This matters for the theory: the
  reward signal is a constant function, not something the model can game by shifting θ. (It's what
  makes the gradient identities clean.)
- **x ∼ D** = problems `x` are drawn from a distribution/dataset **D**.

### The quantities being optimized
- **p_θ(x) = E_{y∼π_θ(·|x)}[V(x, y)]** = **per-prompt success mass**: for one problem `x`, the
  probability a single sampled answer is correct. (Average the verifier's 0/1 over the answers the
  model samples → a number in [0,1].) `E[...]` = expected value = average.
- **J(θ) = E_{x∼D}[p_θ(x)] = pass@1** = the **objective**: average per-prompt success over all
  problems = single-attempt reliability. This is what the paper wants to raise. `J` is the score
  the whole method is trying to maximize.
- **cov_k(θ) = E_x[1 − (1 − p_θ(x))^k]** = **coverage** = pass@k = chance at least one of `k`
  samples is correct, averaged over problems. (Same formula as before: `(1−p)^k` = all k wrong;
  `1 −` that = at least one right.) This is the flattering number; `J` (pass@1) is the honest one.

### Sets of solutions
- **C(x) = {y : V(x, y) = 1}** = the set of all **verified-correct** solutions to `x`.
- **I(x)** = the set of **verified-incorrect** solutions to `x`. (C for Correct, I for Incorrect.)

### Two headroom notions (measured at the *base* model θ0)
- **h(x) = 1 − p_θ0(x)** = **headroom** = how much room there is to improve on problem `x` at the
  start. If the base already nails it (p_θ0 ≈ 1), headroom ≈ 0; if the base is weak, headroom is
  large. (This is the "gap scales with headroom" quantity from Fact 2.)
- **a(x; C) = 1 − (1 − p_θ0(x))^C** = **accessibility** at sampling budget **C** = the chance the
  base model produces *at least one* correct solution if you let it sample `C` times. In words:
  "can we even *find* a correct y+ to train on, within C tries?" A problem the base never solves
  in C samples has accessibility ≈ 0 → no verified-correct example → nothing for replay/preference
  to use. (Note: here **C** is a sampling budget number; **C(x)** is the correct-set — same letter,
  different role.)

### The selection signal (the star of the theory)
- **m_θ(x) = log π_θ(y+ | x) − log π_θ(y− | x)** = the **logit margin** for a self-generated pair,
  where **y+ ∈ C(x)** is a verified-correct sample and **y− ∈ I(x)** is a verified-incorrect one.
  It's how much more log-probability the model gives the right answer than the wrong one — the
  same margin from Fact 3. Growing `m_θ` is exactly what RVP does and what pass@1 needs.
  "Self-generated pair" = both y+ and y− come from the model's *own* samples (not an external
  dataset).

### The three training methods being compared (all start from the SAME base θ0)
- **GRPO** = on-policy policy-gradient on `V`: attempt problems live, use the verifier's reward to
  nudge θ. (The near-null arm.)
- **RFT** = multi-epoch cross-entropy on a **verified bank B = {(x, y) : V = 1}** sampled from the
  base π_θ0: collect the base model's own correct answers, imitate them for several passes. (The
  winning-but-incomplete arm; mode-covering.)
- **VSF** = GRPO **plus** a "persistent prompt-balanced replay floor" over B — i.e. keep mixing in
  the verified replay signal during GRPO, balanced across prompts. It's a **causal probe**, not a
  product: it isolates whether *adding replay* is the active ingredient. ("Floor" = a baseline
  signal always present under the RL loss.)

### Why this paragraph matters
It's scaffolding, but load-bearing: by nailing down `J` (pass@1) vs `cov_k`, the sets `C/I`,
`headroom` vs `accessibility`, and the `margin m_θ`, the paper can now state its theorems precisely
— e.g. "RFT raises both modes so `m_θ` barely moves," "gains are gated by headroom×accessibility,"
and "RVP's gradient increases `m_θ`." Every later claim is phrased in these symbols.

---

## PASTE (Theorem 1 — the gradient-family identity: why on-policy reward acquires little)

> Theorem 1 (Gradient-family identity). Under (A1), for every prompt x, ∇_θ p_θ(x) = p_θ(x) · E_{y∼π_θ(·|x)}[∇_θ log π_θ(y | x) | V(x, y) = 1]. Consequently (i) the within-prompt-normalised RFT gradient on current-policy successes equals ∇_θ log p_θ(x), and (ii) the binary-reward policy gradient equals p_θ(x) times the same direction. RFT and outcome-RL are therefore not distinct gradient families; they differ only by a positive per-prompt scalar and by which prompts carry a nonzero success-gradient.
> Proof. Score-function identity ∇_θ p_θ(x) = E[V ∇ log π]; condition on V = 1 (the V = 0 term vanishes as V ∈ {0,1}) and normalise by p_θ(x) = Pr(V=1). The policy gradient of E[V] is E[V ∇ log π] = p_θ(x)·(that conditional mean).
> Corollary 1.1 (coverage throttling). Because the on-policy update weights each prompt's success-gradient by p_θ(x), a prompt the base model almost never solves (p_θ0(x) ≈ 0) receives ≈ 0 update: outcome-RL cannot acquire base-failed prompts. RFT's decoupled multi-epoch replay over B supplies those gradients regardless of current success mass. This predicts GRPO pass@1 ≈ base while RFT acquires broadly.
> [Confirmed: GRPO turnover-robust acquisition A=0.010 (net +0.001) vs RFT A=0.132 (net +0.126), Table 4; GRPO leaves the per-prompt histogram identical — 79% of prompts stay p̂<0.1 — while RFT rescues a third, Fig. 4.]
> Scope (honest). Eq.(1) is an exact per-prompt identity for current-policy expectations. A stale multi-epoch bank, clipped/group-normalised GRPO, and cross-prompt reweighting each perturb the aggregate direction; we do not claim "all RFT–GRPO gaps are procedural" in general—only the identity and the coverage-throttling mechanism it exposes.

## EXPLANATION

**One-line takeaway:** GRPO and RFT are secretly pushing the model in the **same direction** — but
GRPO multiplies that push by `p_θ(x)` (how often the model already solves the prompt), so on hard
prompts (where it solves ≈ never) the push is ≈ **zero**. RFT has no such multiplier, so it learns
the hard prompts too. That's the mathematical reason for Fact 1's GRPO null.

### First, the notation
- **∇_θ** ("nabla-theta") = the **gradient** = "the direction in weight-space that increases this
  quantity fastest." Training = take a step along a gradient. `∇_θ (something)` = "how to change
  the weights θ to raise `something`."
- **∇_θ p_θ(x)** = the direction that raises the per-prompt success `p_θ(x)` (getting problem `x`
  right more often).
- **∇_θ log π_θ(y|x)** = the direction that raises the model's log-probability of a *specific*
  answer `y`. (The "score function" — the basic building block of policy gradients.)
- **E[ … | V=1]** = average **conditioned on** the answer being correct — i.e. average only over
  the answers that pass the verifier.

### What the theorem says (Eq. 1), in words
> ∇_θ p_θ(x) = p_θ(x) · E[ ∇_θ log π_θ(y|x) | V=1 ]

"The direction that makes the model solve problem `x` more often = **(how often it currently
solves it)** × **(the average direction that boosts its correct answers)**."

Two consequences fall out:
- **(i) RFT's gradient** (imitate current-policy correct answers, normalized per prompt) = the
  average direction that boosts correct answers = **∇_θ log p_θ(x)** (the *un-scaled* improvement
  direction).
- **(ii) GRPO's gradient** (binary-reward policy gradient) = **p_θ(x) × the same direction**.

So **they point the same way**; they differ only by a **positive scalar `p_θ(x)`** (per prompt)
and by **which prompts get a nonzero push**. "Not distinct gradient families" = they're the same
underlying update, just weighted differently. This is a genuinely clarifying result — it says the
GRPO-vs-RFT difference isn't about *direction*, it's about *weighting*.

### The proof, in one line
It's just the **score-function identity** (a standard calculus fact: ∇E[f] = E[f ∇log π]) applied
to `f = V`. Since `V` is 0 or 1, the "wrong" answers (V=0) contribute nothing to the sum, so the
whole gradient comes from the correct answers, scaled by how many there are, `p_θ(x) = Pr(V=1)`.
Nothing exotic — it's an exact rearrangement.

### The killer consequence — Corollary 1.1 ("coverage throttling")
Because GRPO's push is **multiplied by `p_θ(x)`**:
- On a prompt the base **almost never solves** (`p_θ0(x) ≈ 0`) → push ≈ 0 × direction = **≈ 0**.
  GRPO **cannot learn the prompts the model is currently failing** — the exact prompts you'd most
  want to fix. It only sharpens what the model already gets right.
- **RFT** replays a **fixed bank** of verified-correct answers over multiple epochs, so it delivers
  the improvement gradient **regardless** of current success mass — including for hard prompts.
- Prediction: **GRPO pass@1 ≈ base** (stuck), **RFT acquires broadly** (improves across the board).

"Coverage throttling" = the update is throttled (choked) in proportion to how little coverage the
model has on a prompt — precisely backwards from what you want.

### The confirmation (numbers)
- **"turnover-robust acquisition A"** = a robust measure of how many prompts genuinely moved from
  failing→solving (robust to noisy prompts that flip by luck — "turnover"). GRPO **A=0.010** (net
  pass@1 +0.001) vs RFT **A=0.132** (net +0.126). RFT acquires ~13× more.
- **Distributionally (Fig. 4):** GRPO leaves the per-prompt reliability histogram **identical** to
  base — **79%** of prompts stay in the "almost never solved" pile (estimated success `p̂ < 0.1`)
  — while **RFT rescues about a third** of them. `p̂` = the empirical (measured) per-prompt
  success rate.

### The honesty caveat (Scope) — important
The authors explicitly **do not overclaim**. Eq.(1) is exact only for *current-policy* expectations
(sampling from the model right now). Real GRPO/RFT differ from the clean identity because of:
- a **stale multi-epoch bank** (RFT replays old samples, not current-policy),
- **clipped / group-normalised GRPO** (real GRPO isn't the raw policy gradient),
- **cross-prompt reweighting** (aggregating across prompts shifts the direction).

So they claim **only** the identity + the coverage-throttling *mechanism* — **not** that "every
RFT–GRPO gap in the wild is just this scalar." That restraint is the same honesty running through
the project: prove the mechanism, don't inflate it. Connects to [[rl-routing-vs-competence]] (RL
sharpens what's already there rather than acquiring new mass) and the broader
[[rl-award-paper-master-plan]] theory.

---

## FOLLOW-UP (How exactly do GRPO and RFT differ, if they point the same direction?)

They point the same *direction* per prompt, so the difference is entirely in **two weighting
choices** — and those choices flip the outcome on hard prompts.

### The two differences

**1. The per-prompt multiplier `p_θ(x)`**
- **GRPO** scales each prompt's improvement push by `p_θ(x)` = how often the model *currently*
  solves it.
- **RFT** normalizes it away (the "÷ p_θ(x)" in `∇log p_θ = (1/p_θ)∇p_θ`), so every prompt gets a
  **full-strength** push regardless of current success rate.

**2. Which prompts even get a push (nonzero gradient)**
- **GRPO is on-policy:** to get *any* signal on a prompt, it must actually sample a correct answer
  during training. On a prompt it almost never solves, it almost never draws a `V=1`, so it sees
  ≈ no gradient there.
- **RFT is decoupled + multi-epoch:** the correct answers were collected once into a fixed bank
  `B`, then replayed every epoch. So a hard prompt that got even one banked correct answer is
  trained on every pass, forever, no matter what the current model does.

### Concrete example
Two prompts: easy (`p_θ=0.8`) and hard (`p_θ=0.02`).

| | easy prompt push | hard prompt push |
|---|---|---|
| **GRPO** (× p_θ) | 0.8 · dir | 0.02 · dir ≈ nothing |
| **RFT** (normalized) | 1 · dir | 1 · dir (full) |

Same `dir`, opposite treatment of the hard prompt. GRPO pours its effort into prompts it already
gets right (0.8) and starves the ones it's failing (0.02). RFT spends equally on both.

### Why this is the whole story
- **Same direction** ⇒ neither method changes what "getting better" means — they both climb the
  same hill.
- **Different weighting** ⇒ they climb it on different prompts. GRPO only sharpens the
  already-solved (→ pass@1 barely moves, since those were already near-solved). RFT lifts the
  unsolved (→ broad acquisition).

So "not distinct gradient families" doesn't mean "identical" — it means the gap is **not** because
RL and imitation optimize different things. It's a pure **coverage-weighting** artifact: GRPO's
`p_θ(x)` multiplier chokes off exactly the prompts with the most headroom. That's the mechanism
behind GRPO ≈ base and RFT acquiring broadly.

---

## PASTE (Theorem 2 + Prop 1 — positive-only replay is mode-covering; selection untouched)

> Theorem 2 (support-boundedness, scoped). Under (A1), the verified-only cross-entropy objective contributes zero example-specific gradient on any prompt absent from the bank; any change there is transfer through shared parameters. (We state this as an identity about the example-specific term, not as an upper bound on out-of-distribution improvement: shared-parameter updates can move off-support predictions, so no generalisation ceiling is claimed.)
> Proposition 1 (no margin control). The RFT gradient ∇_θ Σ_{y∈C(x)} log π_θ(y | x) contains no term that decreases log π_θ(y− | x) for a verified-incorrect y−; it therefore does not directly increase the margin m_θ(x). Positive-only imitation is mode-covering: it can raise the probability of correct completions without suppressing the incorrect ones the model also samples.
> This is the mechanistic origin of the reliability gap (Motivation §2). On the same held-out verified pairs, RFT raises log π(y+) from −0.110 to −0.092 but raises log π(y−) almost identically (−0.139 → −0.119), leaving the margin flat (0.029 → 0.027); GRPO moves neither. Coverage rises, single-attempt selection does not.

## EXPLANATION

**One-line takeaway:** RFT only ever trains on *correct* answers, so its gradient has **no term
that pushes wrong answers down**. It can lift the correct answer's probability, but the wrong one
often rides up with it — the margin (the selection signal) doesn't grow. That's the *proof* that
positive-only replay structurally can't close the reliability gap.

### Theorem 2 (support-boundedness) — decoded, carefully
- **"verified-only cross-entropy objective"** = RFT's loss: imitate the banked correct answers
  (standard next-token / cross-entropy training on `y+` only).
- **"zero example-specific gradient on any prompt absent from the bank"** = for a prompt that
  **isn't in the training bank**, RFT's loss has literally no term about it → no *direct* gradient
  for that prompt. Makes sense: you can't imitate an example you never included.
- **"any change there is transfer through shared parameters"** = the model *can* still change its
  behavior on unseen prompts, but only as a **side effect** — because all prompts share the same
  weights θ, so updating θ for banked prompts spills over. That spillover is "transfer," not a
  direct push.
- **The parenthetical is the honesty guard (important):** they call it an **identity about the
  example-specific term**, *not* an upper bound on out-of-distribution gains. In plain words:
  "we are NOT claiming RFT can't generalize to unseen prompts — shared-parameter transfer can
  genuinely help off-support. We only claim there's no *direct* per-example gradient there." This
  deliberately avoids the overclaim "RFT has a generalization ceiling." ("Scoped" in the title =
  narrowly stated on purpose.)

### Proposition 1 (no margin control) — the core structural fact
- RFT's gradient is **∇_θ Σ_{y∈C(x)} log π_θ(y|x)** = "raise the log-probability of every
  **correct** solution `y ∈ C(x)`." Every term in that sum *pushes a correct answer up*.
- **What's missing:** there is **no term of the form "decrease `log π_θ(y−|x)`"** for a
  verified-wrong `y−`. RFT never sees wrong answers in its loss, so it has no lever to push them
  down.
- Recall the margin **m_θ(x) = log π_θ(y+|x) − log π_θ(y−|x)**. To grow the margin you must raise
  `y+` **and/or** lower `y−`. RFT only does the first, and only for the specific banked correct
  strings — it **does not directly increase `m_θ`**.
- **"Mode-covering"** = the precise term for this behavior: imitation spreads probability to
  *cover* the modes it's trained on (the correct completions), but because generation still visits
  nearby wrong completions and they share parameters, those wrong modes get lifted too. It covers;
  it doesn't *separate*.

### The evidence (why we believe it, not just assert it)
Measured on the same held-out verified pairs (Motivation Table 3 / Fig. 3):
- RFT: `log π(y+)` −0.110 → **−0.092** (correct answer up a bit).
- RFT: `log π(y−)` −0.139 → **−0.119** (wrong answer up by *almost the same amount*).
- margin: 0.029 → **0.027** (essentially flat — even a hair down).
- GRPO: moves **neither** (frozen, per Theorem 1's throttling).

So numerically: **coverage rises** (both modes more probable → more likely to reach a correct one
in k tries) but **single-attempt selection does not** (the model is no better at putting the
correct one *first*). That is the mechanistic origin of the coverage−pass@1 gap from §2.

### How Theorem 2 and Prop 1 fit together
- **Theorem 2:** RFT's *direct* influence is confined to the banked (correct) examples — it has no
  direct handle on anything else, including wrong completions.
- **Proposition 1:** even *within* the prompts it does train, it only lifts correct answers and
  never suppresses wrong ones, so the *margin* — the thing pass@1 depends on — doesn't move.
- **Together:** positive-only imitation is structurally a **coverage/acquisition** tool, blind to
  the **selection** axis. This is the theoretical complement to Theorem 1 (which killed GRPO):
  neither dominant recipe touches selection — exactly the opening RVP is built to fill (its
  preference gradient *does* contain the missing "push `y−` down" term, so `m_θ` grows). Connects
  to [[rl-operator-mass-placing-result]] (RFT/SFT place mass; only a contrastive step separates
  it) and the RVP margin-ascent proposition.
