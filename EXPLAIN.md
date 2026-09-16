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
