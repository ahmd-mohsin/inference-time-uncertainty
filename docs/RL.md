# Making RL Generalize, Not Just Sharpen

## TL;DR

Reinforcement learning with verifiable rewards (RLVR) makes a model **more reliable at
answers it could already find** (pass@1 ↑) but **worse at finding answers it rarely finds**
(pass@k ↓ at large k). It sharpens the peak and shaves off the tail. We are building an RL
recipe that **sharpens the peak AND keeps/grows the tail** — by rewarding *correct-and-novel*
reasoning and by feeding the model its own rare successes off-policy. Success is concrete:
**erase the pass@k crossover** — beat plain GRPO at small k *and* match/beat the base model
at large k, at small compute (not ProRL-scale brute force).

Code: `rl_training/` (TRL GRPOTrainer). Running now: a 4-arm study on Qwen3-8B / AIME across
4×8 A100 nodes (`base`, `grpo`, `oursA`, `oursAB`).

---

## 1. What we are doing

We post-train a small reasoning model (Qwen3-8B) with GRPO on competition math (AIME
2024+2025+2026), and we measure the **full pass@k curve (k = 1 … 256)**, not just pass@1.
The whole study is one controlled ladder:

| Arm | What it is | Purpose |
|---|---|---|
| **base** | the untrained model | the pass@k ceiling RL must not fall below |
| **grpo** | standard GRPO, full data | reproduces the "RL sharpens" crossover (control) |
| **oursA** | GRPO + group-relative **novelty reward** (Component A) + hard-targeting (C) | does rewarding *correct-and-different* protect the tail? |
| **oursAB** | oursA + **off-policy tail harvesting** (Component B) | does feeding the model its own rare successes expand support? |

Each arm trains, then evaluates pass@k. The comparison of the four curves is the result.

---

## 2. The gap (why standard RL sharpens but does not expand)

Grounding paper: **Yue et al. 2025, "Does RL Really Incentivize Reasoning Capacity Beyond the
Base Model?"** (arXiv:2504.13837). What they established, and *why* it happens:

- **The finding.** On pass@k curves, RLVR beats the base model at small k but the **base
  overtakes it at large k** (k = 128–1024), across every model family and benchmark. So RL
  *narrows* the set of solvable problems even as it raises average accuracy.
- **The cause is structural, three layers deep:**
  1. **On-policy ⇒ redistribution only.** PPO/GRPO learn only from the policy's *own*
     samples. A reasoning path the model emits with probability ≈ 0 is never sampled, never
     gets gradient, and so can never be reinforced. RL can only move probability *among paths
     already in the support* — it cannot add new ones.
  2. **Binary reward ⇒ novelty is uncredited.** Every correct rollout gets reward 1
     regardless of *how* it solved the problem. The gradient therefore flows to whichever
     correct path is *already most probable*, collapsing the distribution onto one mode and
     starving rare-but-valid alternatives.
  3. **It is not just low entropy.** Their key ablation: reheating the RL model's temperature
     to match the base model's entropy *still* loses at large k. The rare paths are
     **pruned**, not merely down-weighted — you cannot get them back by sampling hotter.
- **The contrast that points to the fix.** Distillation *does* expand pass@k beyond the
  base, because it injects **off-policy** paths from a teacher. The lever that works is
  off-policy signal — exactly what on-policy RL lacks.

**So the gap is:** to *expand* rather than *sharpen*, the recipe must break at least one of
— (a) on-policy-only gradient, (b) correctness-only reward, (c) winner-take-all collapse.

---

## 3. What's been tried (and why it's not enough)

| Line of work | Papers | What it does | Why it's insufficient alone |
|---|---|---|---|
| Entropy / exploration preservation | Cui et al. 2505.22617 (Clip-Cov/KL-Cov); Wang et al. 2506.01939 (forking tokens) | slow the entropy collapse; concentrate gradient on high-entropy decisions | **stays on-policy** — slows the tail loss, can't add paths (touches c, not a) |
| Prolonged RL | ProRL, Liu et al. 2505.24864 | KL control + reference reset + diverse tasks + *very long* training → does expand the boundary | works by **brute force**; expensive; our efficient alternative is the contribution |
| Exploration budget | Knapsack-RL 2509.25849 | give more rollouts to hard tasks (fix zero-gradient problems) | an *allocation* trick, not a reward change — complementary, we reuse the idea |
| Representation diversity | Sun et al. 2510.11686 | hidden-state diversity bonus → ~3× pass@k efficiency | proves diversity-in-representation works; it is our **measurement tool**, not our novelty |
| Off-policy / self-distillation | distillation (Yue); STaR / ReST | inject new paths via SFT on teacher/self traces | studied as *separate SFT*, **not as an RL objective that expands support** — our opening |

**Where we sit:** ProRL already showed *RL can expand*. So our claim is not "RL can expand"
— it is **"RL can expand efficiently and reliably, via a correct-and-novel reward + the
model's own off-policy tail, without ProRL-scale training."**

---

## 4. The method (three components)

Unifying objective: **keep the dominant correct mode sharp while protecting and growing the
rare-but-valid tail.** Implemented in `rl_training/`, built on TRL's `GRPOTrainer` with LoRA
and colocated vLLM.

**Component A — group-relative novelty reward** (`rewards.py`, `semantic.py`). Within each
GRPO group, give a correct rollout a bonus for being *semantically distant from the other
correct rollouts*:
```
reward_i = correct_i · (1 + λ · novelty_i)
novelty_i = mean embedding distance of rollout i to the OTHER correct rollouts in the group
```
The bonus applies **only to correct rollouts** — this is the crux that separates us from a
generic entropy bonus (which would also reward diverse *wrong* answers). We reward
*correct paths that differ from the consensus correct path*, i.e. we protect the rare valid
mode the standard gradient crushes. Distance is measured with a pretrained sentence-embedding
model (`all-MiniLM`), in approach space, not raw logits. (Fixes b + c.)

**Component B — off-policy tail harvesting** (`harvest.py`). Periodically: sample the current
policy at large k on the hard problems, **harvest the rare correct rollouts** (the tail
on-policy RL never reinforces), and SFT on them off-policy. This injects a previously-≈0-prob
correct path into the high-probability region — distillation's mechanism, but the teacher is
*the model's own tail*, so no external model. (Fixes a — the support ceiling.)

**Component C — hard-problem targeting** (`difficulty_prepass.py`). Pre-label every problem by
sampling the base model k times: *solved* (pass@1 high), *hard* (low pass@1, pass@k > 0 — the
in-support-but-low-prob set), *stuck* (pass@k = 0). Focus A and B on the **hard** set, where
the boundary actually needs expanding. Easy problems are already solved; expansion there is
wasted. (This is what makes it efficient vs ProRL.)

---

## 5. What we expect

The decisive plot is the four pass@k curves overlaid. Concretely we expect:

- **base**: high at large k, low at small k (broad but unreliable).
- **grpo**: high at small k, **drops below base at large k** — reproduces the Yue crossover
  (this must replicate, or our premise is wrong).
- **oursA**: keeps grpo's small-k gain but the large-k drop is *reduced* — the novelty reward
  alone slows the tail loss.
- **oursAB**: the crossover **disappears** — matches/beats grpo at small k *and* matches/beats
  base at large k. This is the win: sharpen the mode and keep the tail.

**The single success criterion:** the pass@k crossover is erased by oursAB (and ideally
already softened by oursA). Two informative outcomes:
- If **oursA alone** closes the large-k gap → the novelty reward is the lever (cheap, no
  off-policy machinery needed).
- If only **oursAB** closes it → support expansion genuinely requires off-policy injection
  (the stronger, more mechanistically interesting result, and the one Yue's analysis predicts).

A clean **negative** result is also valuable: if neither closes the gap, it is direct evidence
that on-policy reward shaping *cannot* expand support and only off-policy data can — sharpening
the field's understanding of the boundary.

---

## 6. Practical decisions (so results are trustworthy)

- **16k context, no truncation.** Reasoning models need room to think; truncation
  contaminates correctness labels (a prior topological run had 41% of chains cut off → garbage
  signal). Context window = 16384, generation budget ~14336. This makes runs ~10× slower but
  the labels are clean.
- **Answer extraction is `src/data/dataset.py:extract_numeric_answer`** — a 7-strategy
  extractor (boxed-in-full-text, strip `<think>`, answer markers, bold, "= X", bare number,
  trailing LaTeX) validated 18/18 on realistic reasoning output (incl. truncated-`<think>`,
  multi-boxed, fractions, units). The earlier weak boxed-only extractor produced false
  pass@k = 0.
- **Clean ablation.** `grpo` runs on full data with **no** novelty and **no** hard-targeting,
  so it is a true standard-GRPO control; "ours" arms add A (+C) and B.
- **Validated end-to-end before scale.** A real eval produced a non-zero, rising pass@k curve
  (0.44 → 0.50) with correctly extracted answers and a visible easy/hard spread — proof the
  pipeline measures what it should.

---

## 7. References (arXiv)

- **2504.13837** — Yue et al., RL sharpens, doesn't expand (*the gap*).
- **2505.22617** — Cui et al., entropy mechanism (Clip-Cov/KL-Cov).
- **2506.01939** — Wang et al., 80/20 forking tokens.
- **2505.24864** — Liu et al., ProRL (prolonged RL expands boundary).
- **2509.25849** — Knapsack-RL (exploration-budget allocation).
- **2510.11686** — Sun et al., representation-based exploration (3× pass@k).
- **2407.21787** — Large Language Monkeys (coverage scales, selection plateaus).
- **2601.16175** — TTT-Discover (test-time RL; composition/horizon, cf. questions_and_directions Dir 14–15).
