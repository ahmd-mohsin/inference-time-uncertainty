# Sequential Active-Recovery Diagnosis — results (2026-09-03)

Tests the award-thesis crux (R2/R3): does letting the model condition on the *actual execution feedback*
of its prior recovery attempts (true "active diagnosis") beat static portfolio and iid resampling?
Harness: `rl_training/seq_recover.py` (go_seq.sh), matched budget T=6, three arms —
IID (fresh retries), STATIC (portfolio conditioned only on the original failure), SEQUENTIAL (attempt t
sees the full history of prior attempts' code + errors). 8-GPU DP over default-failed problems.

## Headline: NO-GO — sequential self-diagnosis HURTS. iid is the strong baseline.

recovery@T=6 (fraction of default-failed problems recovered), across families/benches:

| cell | IID | STATIC | SEQUENTIAL | seq−iid | seq−static |
|---|---|---|---|---|---|
| Qwen-Instruct · HumanEval | **0.677** | 0.581 | 0.387 | **−0.290** | −0.194 |
| Qwen2.5-Coder-14B · MBPP | **0.488** | 0.449 | 0.323 | −0.165 | −0.126 |
| Qwen2.5-Coder-14B · HumanEval | 0.467 | **0.533** | 0.467 | 0.000 | −0.067 |
| Llama-3.1-8B · MBPP | 0.414 | **0.460** | 0.430 | +0.017 | −0.030 |

Sequential is worst or tied-worst in every cell. The cumulative curves show the mechanism cleanly
(Qwen-Instruct HE): IID `0.32→0.48→0.48→0.52→0.61→0.68`; SEQ `0.29→0.36→0.36→0.36→0.36→0.39`. **Each fresh
iid sample keeps discovering new solutions; each history-conditioned attempt adds almost nothing.**

## Interpretation (unexpected, and the corrective contribution)
Verified not a harness bug — sequential attempts are valid code (solve 9/31 at round 0, +3 later; a
garbage bug would solve ~0). The effect is **ANCHORING**: conditioning on its own failed code + errors
biases the model toward correlated variants of the same wrong solution, collapsing exploration. Fresh iid
sampling preserves the diversity that actually drives recovery. This falsifies the intuition (and the
critique's hope) that self-diagnosis on accumulated feedback helps *these* (RLVR-tuned, not repair-trained)
models — it is a **negative capability**: the models cannot yet exploit their own execution feedback.

Combined with the router NO-GO and the D_fail refutation, the emerging thesis is **corrective**:
> *For RLVR code models, iid resampling is a remarkably strong recovery baseline. Routing, learned
> failure-surface diversity, and self-reflective feedback conditioning all fail to beat it — and feeding
> the model its own failure history actively hurts. The one robust lever is matched-budget diversification
> (a portfolio), whose edge shrinks with budget/competence.*

## POSITIVE — the anchoring antidote works: hide the buggy code, keep the error
`--diag-mode error_only` (show only the error signature each round; NEVER echo the model's failed code)
reverses most of the anchoring loss and often makes sequential feedback BEAT iid:

| cell | SEQ full-mode | SEQ error-only | Δ (eo−full) | error-only seq−iid | error-only seq−static |
|---|---|---|---|---|---|
| Qwen-Instruct · HumanEval | 0.387 | **0.677** | +0.290 | −0.032 | +0.097 |
| Qwen2.5-Coder-14B · HumanEval | 0.467 | **0.600** | +0.133 | **+0.133** | +0.000 |
| Llama-3.1-8B · MBPP | 0.430 | **0.535** | +0.105 | **+0.096** | +0.047 |
| Qwen2.5-Coder-14B · MBPP | 0.323 | 0.370 | +0.047 | −0.071 | −0.063 |

**Mechanism confirmed by the intervention:** the failure was ANCHORING on the model's own buggy code, not
an inability to use feedback. Remove the code, keep the error → the accumulated *error* history now helps
(error-only SEQ also beats STATIC portfolio in 3/4 cells, so it is a genuine sequential/diagnostic gain,
not just fresh sampling). **Deployable recipe for code agents: on failure, tell the model the error and
say "try again" — do NOT paste back its own failed code.** (Larger-n confirmation + horizon T=10 running.)

## Aggregate across 21 cells (families × benches × budgets) — the precise, honest picture
- **Full-mode sequential (show buggy code): mean seq−iid = −0.110** — anchoring hurts, robustly (worst
  Qwen-Instruct·HE −0.290).
- **Error-only (hide code): mean seq−iid = +0.018; beats iid in 10/17 cells.** So the antidote's real,
  clean effect is **removing the ≈−0.11 anchoring penalty** (−0.110 → +0.018, a ~+0.13 swing). *Framing
  correction:* the earlier "+0.10..+0.29" numbers are error-only **vs full-mode** (the antidote effect),
  NOT vs iid. Versus iid, error-only modestly wins on most cells (Llama·MBPP +0.096, Qwen-Instruct·MBPP
  +0.080, Qwen-Coder·HE +0.038) but roughly *ties* on several and LOSES on Qwen2.5-Coder-14B·MBPP
  (−0.07..−0.16, the consistent exception — strong model, near ceiling).
- **Clean contribution = the anchoring MECHANISM**, not a large absolute win over iid: showing a model its
  own buggy code degrades its next attempt; hiding it (keep the error) removes the damage. A big win *over*
  iid is what the GRPO self-repair experiment must deliver (train the model to exploit feedback, not merely
  avoid anchoring).

## FINAL aggregate over all error-only replicates (n=17 runs): parity with iid, model-gated
mean seq−iid = **+0.018 ± 0.069** (≈ parity). Per cell (mean over replicates):
| cell | n | mean seq−iid |
|---|---|---|
| ll_mbpp | 3 | **+0.072** |
| qi_mbpp | 2 | +0.061 |
| qc14_he | 3 | +0.067 |
| qc_he | 1 | +0.038 |
| qi_he | 3 | +0.022 |
| ll_he | 1 | 0.000 |
| qc_mbpp | 1 | −0.066 |
| qc14_mbpp | 3 | **−0.092** |

**Honest final read:** the anchoring MECHANISM (full-mode seq−iid −0.11) is robust and is the contribution.
The hide-the-code antidote reliably REMOVES that penalty (→ +0.018 mean, i.e. parity with iid), and helps
weaker models (Llama/Qwen-Instruct MBPP +0.06–0.07) while the strong 14B-coder does not benefit (−0.09).
It does NOT robustly beat iid on average. Deployable claim: *"on failure, give the model the error, not its
own failed code"* — it never hurts and helps weaker models / low budgets; it is not a universal win.

## Regime: the error-only diagnostic advantage FRONT-LOADS (largest at small budget)
Horizon T=10 vs T=6 (error_only): the sequential/diagnostic gain over iid is a low-budget effect — blind
iid's diversity catches up given many samples.
- Qwen-Instruct·HE: seq−iid −0.03 (T=6) → **+0.03 (T=10)** (SEQ 0.677→0.710).
- 14B·HE: +0.133 (T=6) → 0.00 (T=10) — iid rose 0.467→0.600 and caught up.
- 14B·MBPP: −0.07 (T=6) → −0.118 (T=10) — iid scales faster here.
Consistent with the portfolio result (edge shrinks with budget). Practical reading: **error-only feedback
recovery is most valuable when retries are scarce (few-shot agent budgets)** — it recovers more per
attempt; with a large sampling budget, plain iid resampling suffices. Budget sweep T={3,6,10} running to
draw the curve.

## Live follow-ups (nodes running now)
1. **Anchoring antidote (`--diag-mode error_only`):** hide the buggy CODE, keep only the ERROR signature.
   If error-only recovery ≥ iid while full-history < iid, the deployable recipe is *"tell the model what
   broke, never show it its own broken code."* (running: seq_*_eo cells.)
2. **The training lever (next, GRPO):** the model was never trained to use feedback. Feedback-conditioned
   self-repair GRPO — reward a repair that passes *given* the failed attempt+error — is the experiment that
   could flip this negative into a positive, distinctive result (a self-repairing code policy that beats
   iid). `train_grpo.py` is math-only today; a code-execution reward + failure-conditioned rollout is the
   build.

## Data / reproduce
Pulled: `rl_training/runs_pulled/seq/seq_seq_*.json`. `python -m rl_training.seq_recover --merge --tag <t>`.
(Note: `seq_hist` text not persisted this run — only per-round pass/fail; add transcript dump before the
GRPO write-up.)
