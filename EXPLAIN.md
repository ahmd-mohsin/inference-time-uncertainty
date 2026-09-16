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
