# Failure-Surface & Active-Recovery-Diagnosis — offline results (2026-09-03)

Testing the new "active recovery diagnosis" thesis on the recovery matrix already collected
(code_recover MBPP+HE × 4 families incl. denser n_rec=24; complete TACO Atlas 4×{easy,med,hard};
math negative control × 3). Analyzer: `rl_training/analyze_failure_surface.py`. **Reported honestly —
two of the four target signatures do NOT hold offline as formulated.**

## Locked positive (independent of the new thesis)
**Matched-budget diversified portfolio beats iid-retry** at equal budget B. Fair B=12 subsample from each
pool: MBPP+HE portfolio 0.264 vs iid 0.192 = **+0.072** (n=834); pooled code +0.028. Denser B=24 per family:
Qwen-Instruct **+0.170** (portf−iid), Llama +0.084, Qwen-Coder +0.080. Domain-gated: math **negative**
(portf−iid −0.145 Llama, −0.061 Qwen-Coder). The single-option router remains NO-GO (`CRO_ROUTER_RESULTS.md`).

## R1 — Failure-Surface Diversity `D_fail = 1 − mean_{i<j}Corr(E_i,E_j)`: **REFUTED**
`corr(D_fail, portfolio−iid) = −0.35` (Spearman −0.13, n=26 cells) — the WRONG sign.

| domain | mean D_fail | mean portfolio−iid |
|---|---|---|
| code (23 cells) | 0.784 | **+0.054** |
| math (3 cells) | **0.900** | **−0.058** |

Math has the *most* orthogonal failure surfaces (lowest error correlation) yet portfolio recovery *hurts*.
Diagnosis: raw failure-decorrelation conflates "complementary" with "both individually weak and failing on
random disjoint problems." **Useful diversity requires competence × complementarity, not error
decorrelation.** `D_fail` as defined is not the predictor; a competence-gated complementarity measure is
needed (future: restrict to options with recovery > iid, then measure their marginal coverage). Option
entropy also fails (corr −0.28). *This is a genuine negative that reshapes the metric.*

## R3 — Adaptive vs static portfolio (offline simulation): **WEAK (+0.012)**
Equal budget B, static = B random options, adaptive = greedy conditional-coverage using the population
failure structure + prior binary outcomes: adaptive−static = +0.012 (B=2), +0.012 (B=3), +0.010 (B=4).
Only ~1 point — below the award bar. **BUT** this offline sim conditions only on *binary pass/fail* of
prior attempts; it discards the observed **error text**, which is the actual diagnostic signal. So +0.012
is a lower bound. The real test needs attempt 2 to condition on attempt 1's stderr/output → GPU harness
`seq_recover.py` (R2/R3 proper).

## R4 — Marginal recovery coverage: **REAL complementarity**
Pooled code (3128 failures × 8 options), greedy set-cover coverage curve:
`0.048 → 0.079 → 0.101 → 0.118 → 0.134 → 0.144 → 0.152 → 0.160`
Order: defensive → boundary → rewrite → builtins → root_cause → algo_replace → complexity → spec_reread.
No single option dominates; each adds 1–3 pts of new coverage. Supports a *marginal-coverage* training
objective `Δ_i(z|S)=F_i(S∪{z})−F_i(S)` — but the gains are modest at this option granularity.

## R2 — Does recovery become more predictable after an intervention? **UNTESTED (needs GPU)**
This is the load-bearing award-level claim and the one still open. Requires sequential rollouts where the
model observes the *actual execution feedback* of each recovery attempt. Harness: `seq_recover.py`
(fail → attempt a₁ conditioned on the failed code+error → observe o₁ → attempt a₂ conditioned on a₁'s
outcome+error → …). Target: prediction accuracy of the winning next-action rises with observations
(e.g. 30%→48%→65%) and adaptive beats static by ≫ the offline +0.012.

## Honest bottom line
Of the four signatures the critique demanded, the offline data supports only marginal-coverage
complementarity (R4); refutes the D_fail metric (R1); shows weak adaptive gain from binary-only
conditioning (R3); and leaves the crux (R2, error-text-conditioned diagnosis) to the GPU harness now
queued. The robust, publishable positive today remains **matched-budget diversified portfolio recovery,
domain-gated to code**. Whether "active diagnosis" clears the award bar hinges entirely on R2/R3 from
`seq_recover.py`.
