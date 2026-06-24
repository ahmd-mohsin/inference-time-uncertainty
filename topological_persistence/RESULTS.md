# Results & Validation — AIME 2026 (Qwen3-32B)

> Analysis of the overnight run (2026-06-23). Generation: 30 problems × 8 IID + 8
> conditioned chains; validation: 64 chains/problem. Data in
> `data/topological_outputs_aime2026/` (`summary.json`, `validation.json`,
> `problem_*.json`). This file is the honest read on whether the results support the
> claims in `README.md`. Short version: **the pipeline runs end-to-end and the headline
> 87% reproduces, but that number is confounded and does not establish the README's
> core hypothesis.**

## Run summary

| Verdict | Count |
|---|---|
| CEILING_REACHED | 12 |
| SCALABLE | 11 |
| UNCERTAIN | 7 |
| **Total** | **30** |

Headline validation number (`analyze_results.py`): **20/23 = 87%** prediction accuracy
(UNCERTAIN excluded by design). Breakdown: CEILING 10/12, SCALABLE 10/11.

## The headline number is real but confounded

Three structural problems mean 87% mostly does **not** measure the topological hypothesis
(README: "H₁ loops ⇒ scalable; frozen/no-H₁ ⇒ ceiling").

### 1. SCALABLE is scored with an unfalsifiable disjunction
`analyze_results.py:297`: `prediction_correct = scales or already_solved`.
Of 11 SCALABLE problems, **only 1 (prob 9) actually scaled** (`actually_scales=true`).
The other 10 are scored "correct" purely because they were already solved at 8 chains —
i.e. we predicted "more compute helps," it did **not**, and it still counts as a hit.
That contradicts the README's definition of SCALABLE.

### 2. pass@k is saturated ⇒ CEILING is right almost by construction
~25/30 problems already have `pass@8 = 1.0`. CEILING is judged correct when `not scales`,
and nothing can scale past 1.0. So CEILING accuracy mostly reflects "AIME 2026 is easy for
Qwen3-32B at 8 samples," not topology.

### 3. On problems with actual headroom, the topology signal is mostly wrong
The only 3 problems that genuinely scaled (`actually_scales=true`):

| pid | H1 | verdict | pass_gain(8→64) | signal correct? |
|---|---|---|---|---|
| 8  | 0 | CEILING_REACHED | +0.16 | ✗ (said ceiling, but it scaled) |
| 9  | 1 | SCALABLE        | +0.46 | ✓ |
| 27 | 0 | CEILING_REACHED | +0.25 | ✗ (said ceiling, but it scaled) |

H₁ caught **1 of 3** real scaling cases. Meanwhile the highest-H₁ problems (prob 2: H1=4;
prob 0: H1=2) did **not** scale at all. The signal is, if anything, anti-correlated with
real scaling here.

## De-confounded view: H₁ vs. actually-scaled

Stripping the `already_solved` rescue and looking only at whether more compute genuinely
helped:

- Problems that actually scaled: **3** (pids 8, 9, 27).
- Of those, H₁ > 0: **1** (pid 9). H₁ = 0: **2** (pids 8, 27).
- Problems with H₁ > 0 that did **not** scale: many (pids 0, 2, 10, 12, 15, 17, 19, 24, 26…).

⇒ **H₁ features do not predict scalability in this dataset.** The contingency is close to
noise, and the two clearest scaling cases were missed.

## Conditioning (README objective 3): null result
Across problems, `diversity_gain ≈ 0` (mostly slightly negative) and
`new_topological_features` is mostly False. DAD conditioning is **not** expanding the
solution manifold in this data — the comparison runs correctly, but the measured effect is
~zero. The README presents conditioning as a working capability; the data shows no effect.

## Notable single cases
- **prob 9** — cautionary: `pass@64 → 1.0` but `maj@64 → 0.0`. The correct answer exists
  but majority vote picks it *less* often as samples grow. "Scaling helps" via pass@k is
  misleading when selection accuracy degrades.
- **prob 13** — SCALABLE verdict, but `n_correct_of_N = 0` (never solved, even at 64).
  Counted as a wrong prediction. A genuinely hard/unsolved problem mislabeled scalable.
- **stuck ceilings** (pids 14, 28, 29) — `pass@64 = 0`, correctly called CEILING. These are
  the cleanest true positives: no solution exists in the model's distribution.

## Methodology pivot (applied in code — to be run on Qwen3-8B)

The offline re-analysis (`spectral_reanalysis.py`, run on the surviving
`hidden_states.npz`) settled the direction. AUC for predicting `actually_scales`:

| Signal | source | AUC |
|---|---|---|
| answer entropy | trivial baseline | **0.98** |
| NCD mean | Direction 10 | 0.94 |
| unique answers | trivial baseline | 0.91 |
| effective rank | Direction 2 | 0.44 |
| H₁ features | Direction 1 (old core) | **0.33** |

**Hidden-state geometry (both H₁ and spectral rank) loses to simply counting distinct
answers.** Effective rank tracks *difficulty* (Spearman +0.69 vs coverage) but not
*scalability*. Caveat: only 3 scaling positives on this saturated benchmark, so these
AUCs are directional, not significant — which is itself the point (the benchmark can't
test the hypothesis).

Changes made to the pipeline:
1. **`spectral.py`** (new) — effective rank, spectral gain (IID→conditioned), answer
   entropy / unique-count / majority-fraction.
2. **`ceiling_detector.detect_ceiling_v2`** — verdict now driven by answer entropy;
   spectral rank + gain recorded as covariates; H₁ demoted to reference-only.
3. **`config.py`** — Qwen3-8B, TP=1; representation `curve`→`point` (DTW concentrated
   distances ~10× worse than mean-pooling); `normalize=True` (z-score per dim);
   `max_homology_dim` 2→1.
4. **`embeddings.normalize_points`** — fights distance concentration.
5. **`analyze_results.py`** — dropped the `scales OR already_solved` rescue; already-solved
   problems with no headroom are now scored `None` (excluded), not auto-correct.
6. **`run_overnight.sh`** — Qwen3-8B, fresh dir, added spectral re-analysis stage.

The Qwen3-8B run is the actual experiment: an 8B model has real headroom on AIME, so
`actually_scales` becomes a measurable variable and we can finally ask whether *any*
hidden-state / conditioning signal beats the answer-entropy baseline.

## What would make the claim defensible
1. Report SCALABLE accuracy **only on problems unsolved at 8** (remove the `already_solved`
   disjunction in `analyze_results.py:297`).
2. Report the H₁-vs-`actually_scales` contingency table directly, instead of folding it
   into a single accuracy number.
3. State the conditioning null result explicitly.
4. Use a harder benchmark (or restrict to low-`pass@8` problems) so pass@k isn't saturated
   and CEILING isn't trivially correct.

## Data integrity notes (important)
- The committed `summary.json` was previously stale (`n_errors: 30`, all verdicts 0);
  it has been corrected to `12/11/7` matching the actual Phase-C log.
- `validation.json` (all 30 problems) was restored from the cluster read before the
  instance terminated.
- **`problem_22.json … problem_29.json` (per-problem signals + distance matrices) were
  NOT pulled before the Greenland instance went down** — only `summary.json` and
  `validation.json` were captured for those 8. The local tree has `problem_0..21.json`.
  Aggregate verdicts/validation for all 30 are intact; the per-problem detail JSON for
  22–29 must be regenerated if needed.
- `hidden_states.npz` (1.5 GB) is intentionally git-ignored (exceeds GitHub's 100 MB
  limit) and was never pushed.
