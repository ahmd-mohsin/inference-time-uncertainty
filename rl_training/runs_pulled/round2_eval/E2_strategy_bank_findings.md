# E2 — Multi-witness strategy bank (pipeline + local validation)

Built `rl_training/strategy_bank.py` (`cluster` + `emit-sampler`). Ran `cluster` on the existing
base-correct bank `coverage_7b_mi036f/banks/bank_olympiad_7b.jsonl` (no GPU). Output:
`E2_modes_olympiad_bank.json`.

## What E2 does
Groups base-correct solution **texts** per problem into **reasoning modes** (strategy clusters) so
recoverability becomes mode-level (fixes the theory/implementation object mismatch). Strategy features
= a curated taxonomy of math-strategy signals (substitution, induction, contradiction, coordinate vs
synthetic geometry, trig, casework, factoring, calculus, number theory, inequality/AM-GM, counting,
generating functions) + structural counts; agglomerative clustering on cosine distance.

## Local validation result
- 305 problems, mean **3.46 traces/problem** (thin — bank was built with ≤4 witnesses).
- Mean **2.21 strategy modes/problem** (median 2, max 4); **233/305 problems are multi-mode**.
- Splits are interpretable, e.g. problem 0: {synthetic-geo}×2, {substitution+synthetic-geo}, {trig};
  problem 2: {algebraic-manip}×2, {coordinate-geo+trig}, {counting}.

**Takeaway:** even from ~3–4 base-correct solutions, fragile problems already exhibit **multiple
distinct solution strategies**. This is direct evidence the paper's protected object should be
**mode mass** `p_θ(M|q)`, not one exact sequence — a single-trace floor conflates a specific
`(strategy, wording)` with the strategy family it represents.

## Status vs the real E2
- ✅ Clustering pipeline built + validated locally (CPU).
- ✅ GPU sampler emitted: `rl_training/sample_base_solutions.py` — generates N (128–1024) base
  solutions/problem *with text*, verifies against gold (keeps only base-correct witnesses), writes a
  bank jsonl. **Launch-ready; needs a cluster.**
- ⏭ Full E2 = run the sampler at N=512 on the Olympiad hard band (and Omni-MATH for the headline),
  then re-cluster. With hundreds of witnesses/problem the mode count and per-mode mass estimates
  become reliable (the ≤4-trace bank undercounts modes).
- ⏭ Faithful labeling: current strategy features are an LLM-free *proxy* (better than raw TF-IDF, but
  heuristic). The final tables should add an **LLM strategy judge** (anthropic, available on-cluster)
  and report cluster stability + human/LLM agreement, as the review requires.

## Feeds
E2 modes are the input to **E3** (mode-mass certificate `Δ_qm = E_{y~ν_m}[log π_θ − log π₀] ≥ log α`)
and to mode-level **recoverability** (swap E1's `p̂_q` for per-mode mass `p_θ(M|q)`).
