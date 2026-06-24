# Methodology: What We're Testing, What Went Wrong, How We Fixed It

This is the single source of truth for the ceiling-detection experiment. Read it before
interpreting any run. It supersedes the framing in `README.md` (which describes the
original topology-only design that did not work).

---

## 1. The scientific question (unchanged)

> Given a *small* sample of reasoning chains (K=8) for a problem, can we predict whether
> spending more test-time compute (more samples) will yield a correct answer the small
> sample missed — **before** paying for that compute?

Three regimes we want to separate (from `core_intuitions.md`):

| Regime | Meaning | Right intervention |
|---|---|---|
| **SCALABLE** | correct answer is reachable; more sampling will surface it | best-of-N / majority vote |
| **CEILING (stuck)** | correct answer not in the model's reach at any sample count | weight update / new supervision |
| **CEILING (solved)** | already solved at K=8; no headroom either way | stop sampling |

The *secondary* question (the "conditioning paradox"): does conditioning on disagreements
(DAD) **expand** the reachable set, or merely **reshuffle** probability within it?

---

## 2. What went wrong in the previous runs (Qwen3-32B, AIME 2026)

The original pipeline predicted the regime from the **persistent homology (H₁ loops)** of
the 8-chain hidden-state point cloud. It reported "87% accuracy." That number was an
artifact. Five distinct failures, from deepest to shallowest:

### (A) Distance concentration made H₁ pure noise — the core bug
8 mean-pooled chains live in ~5120-D. With curve+DTW embeddings, pairwise distances
concentrated to a **CV of ~2–5%** (all 8 chains ~equidistant → a near-regular simplex).
Persistent homology on an equidistant cloud produces H₁ "loops" whose lifetimes are
**<2% of the H₀ scale** — i.e. determined by 5th-significant-figure jitter. Every H₁
feature was noise. Measured: H₁ had **AUC 0.33** for predicting actual scaling — *worse
than chance*.

### (B) DTW over curves was the worst possible representation
On the *same* data, simple mean-pooled points had distance CV ~14–62% (real variation),
but DTW over 5120-D curves crushed it to ~3%. The README's "path geometry is the real
signal" intuition backfired; curve+DTW destroyed the signal mean-pooling preserved.

### (C) K=8 is too few points for homology
H₁ (loops) and especially H₂ (voids, which the config requested) are not estimable from
8 points. The 64-chain validation set was generated but **never fed to the topology** —
the verdict was locked from 8 noisy points.

### (D) The detector thresholds didn't match the geometry
`h1_lifetime < 0.3` was applied to an *unnormalized* distance scale (~110), and
`betti_convergence_rate` was **0.0 for all 30 problems** (dead signal). The verdict
collapsed to "H₁_count == 0 → ceiling," i.e. thresholding noise.

### (E) The validation couldn't falsify the hypothesis — and the benchmark had no headroom
- Scoring rule was `prediction_correct = scales OR already_solved`. 10 of 11 "correct"
  SCALABLE calls had **not** actually scaled — they were rescued by `already_solved`.
  SCALABLE was nearly unfalsifiable.
- AIME 2026 is **saturated** for Qwen3-32B: 25/30 solved at pass@8. Only **3/30**
  problems actually had scaling headroom. With 3 positives, no signal is statistically
  testable — "87%" mostly measured "this benchmark is easy."

**One-line summary:** noise geometry → arbitrary thresholds → rubber-stamped by a
permissive metric on a saturated benchmark.

---

## 3. The honest reframing (important — read this)

When we re-tested every candidate signal offline against ground truth, the ranking was:

| Signal | AUC vs actually_scales |
|---|---|
| answer entropy (trivial) | 0.98 |
| NCD (surface compressibility) | 0.94 |
| unique-answer count (trivial) | 0.91 |
| spectral effective rank | 0.44 |
| H₁ features (old core) | 0.33 |

**The hidden states lost to simply counting distinct answers.** But there is a catch we
must be honest about, and it shapes the whole experiment:

> **Answer entropy is near-tautological with `actually_scales`.** High answer diversity at
> K=8 means "the 8 chains haven't converged on one answer," and `actually_scales` (pass@k
> gain) means "more samples surface a correct answer the small sample missed." These are
> almost the same quantity by construction. So answer entropy's 0.98 is *not* evidence of
> deep structure — it is a coarse restatement of the target.

This reframes the real contribution. The question is **not** "does answer entropy predict
scaling" (it trivially does). The question is:

> **Does any signal — hidden-state spectral rank, or the IID→conditioned diversity gain —
> add predictive power BEYOND the answer-distribution baseline?**

That is the only result worth claiming. Everything is now built to answer it.

---

## 4. What we changed in the code

| File | Change | Why |
|---|---|---|
| `config.py` | `Qwen/Qwen3-8B`, `TP=1` | single-GPU stability + real AIME headroom so scaling is measurable |
| `config.py` | `representation: curve→point`, `normalize=True` | mean-pooling preserves variation DTW destroyed; z-score fights concentration |
| `config.py` | `max_homology_dim: 2→1` | H₂ meaningless on 8 points |
| `embeddings.py` | `normalize_points()` (z-score per dim) | attacks distance concentration at the source |
| `spectral.py` (new) | effective rank, spectral gain, answer entropy/unique/majority | the validated signals + the difficulty covariate |
| `ceiling_detector.py` | `detect_ceiling_v2`: verdict from answer entropy; spectral rank + gain as covariates; **H₁ demoted to reference-only** | stop deciding on noise |
| `analyze_results.py` | dropped `scales OR already_solved`; no-headroom problems scored `None` | make SCALABLE falsifiable |
| `run_robust.py` | computes/saves new signals via v2; normalized point embeddings | wire the new path |
| `plot_metrics.py` | `08_primary_signals.png` | visualize what now drives the verdict |
| `run_overnight.sh` | Qwen3-8B, fresh dir, added spectral re-analysis stage | end-to-end on the new model |

Current `detect_ceiling_v2` decision rule (answer entropy in nats over K chains):
`ent ≥ 0.4 → SCALABLE`, `ent ≤ 0.05 → CEILING_REACHED`, else `UNCERTAIN`.

---

## 5. How we tackle it — the experimental protocol

**Run:** Qwen3-8B, AIME 2026, 30 problems, 8 IID + 8 conditioned chains; validate with 64
chains/problem; spectral re-analysis; plots. (`scripts/run_overnight.sh`)

**Why Qwen3-8B is the fix, not just convenience:** an 8B model fails or partially solves
far more AIME problems, turning `actually_scales` from 3/30 (untestable) into an expected
10–15/30 — enough to actually evaluate signals.

**The analysis that matters** (in `spectral_reanalysis.py`, run on the new data):
1. Establish the **baseline**: AUC of answer entropy / unique-count vs `actually_scales`.
2. Ask the real question: does **spectral effective rank** or **spectral gain
   (IID→conditioned)** improve prediction *conditioned on* the answer-entropy baseline?
   Concretely — among problems with the *same* answer-diversity bucket, does the spectral
   signal separate scalable from stuck?
3. Report the **conditioning result** honestly: is `spectral_gain` > 0 systematically, and
   does it correlate with anything? (Previous run: ~0, a null result.)

**Success criterion (what would make this publishable):**
- A signal that beats the answer-entropy baseline at separating **stuck-ceiling** from
  **scalable** *among the hard (low-coverage) problems* — because that is exactly where
  answer entropy is ambiguous (a hard problem can have low entropy because all 8 chains
  agree on a *wrong* answer, vs low entropy because they agree on the *right* one;
  answer entropy alone cannot tell these apart, but hidden-state geometry might).

**Failure criterion (also a valid result):**
- If nothing beats answer entropy, the honest finding is: "for this model/benchmark,
  cheap answer-distribution statistics are sufficient; hidden-state topology/geometry adds
  nothing." That is publishable as a negative result and saves everyone the 1.5 GB of
  hidden states.

---

## 6. Known caveats to carry forward
- **Confident-but-wrong problems** (all 8 chains agree on a wrong answer) read as CEILING
  by design. That is the intended meaning ("sampling won't help"), but verify against
  ground truth that these are genuinely stuck, not mislabeled.
- **maj@k vs pass@k divergence** (e.g. previous prob 9: pass@64→1.0 but maj@64→0.0). A
  problem can be "scalable" under pass@k yet get *worse* under majority vote. Report both.
- **Entropy thresholds (0.4 / 0.05)** are provisional; calibrate them on the Qwen3-8B
  validation curve rather than trusting the defaults.
- **Small N.** 30 problems is small; treat all AUCs as directional and report confidence
  intervals or at least the raw contingency tables.
