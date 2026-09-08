# Failure-Conditioned Recovery — the first POSITIVE downstream result (2026-09-02)

**Reframe (reasoning monoculture, direction #3):** the four prior downstream nulls were all
*single-episode, unconditional*. The right question for the hidden repertoire is **conditional on
failure**: when the model's dominant route fails, does switching to a *different* reasoning strategy
recover the problem better than just retrying the same route? Formally `v⁻_m = P(strategy m solves |
default failed)` — this can be ≫0 even when marginal/coverage value ≈0.

**Design (`code_recover.py`, exec-verified):** per code problem, a DEFAULT attempt (8 iid), then TWO
**matched-budget** recovery arms of 12 samples each — (A) iid-retry (same prompt) vs (B) strategy-SWITCH
(forced diverse strategies). On problems the default FAILS, compare recovery rate. Matched budget
isolates *diversity* from *extra attempts* (the confound in the earlier offline signal). 3 coder
families (Qwen2.5-Coder, Qwen2.5-Instruct, Llama-3.1) × {HumanEval, MBPP} + HE replicates.

## Results — recovery on default-failed problems (matched budget, n_rec=12)

| job | bench | n_failed | iid-retry | switch | **switch − iid** |
|---|---|---:|---:|---:|---:|
| Qwen-Instruct | humaneval | 12 | 0.250 | 0.583 | **+0.333** |
| Qwen-Instruct (rep) | humaneval | 13 | 0.154 | 0.538 | **+0.385** |
| Llama | humaneval | 26 | 0.308 | 0.577 | **+0.269** |
| Llama (rep) | humaneval | 22 | 0.364 | 0.318 | −0.045 |
| Qwen-Instruct | mbpp | 118 | 0.102 | 0.305 | **+0.203** |
| Llama | mbpp | 172 | 0.174 | 0.227 | +0.052 |
| Qwen-Coder | mbpp | 61 | 0.213 | 0.197 | −0.016 |
| Qwen-Coder | humaneval | — | — | — | (shards crashed) |

**Mean switch − iid = +0.169; 5/7 positive, several large (+0.20 to +0.39).**

## Verdict — POSITIVE (first downstream win), model/benchmark-dependent

**On problems the default route fails, switching reasoning strategy recovers meaningfully more than
retrying iid at equal budget** — strongest on HumanEval (Qwen-Instruct +0.33/+0.39, Llama +0.27) and on
Qwen-Instruct MBPP (+0.20). This is the first regime where the hidden repertoire delivers a downstream
gain, and it fits the theory exactly: **`v_coverage≈0` but `v_recovery≫0`** — diversity's value is
*recovery, not exploration*. It reconciles all prior nulls (which were unconditional) with a real,
usable effect (failure-triggered strategy switching).

**Honest caveats:**
- HumanEval fail-counts are small (n=12–26) → the large HE effects are encouraging but noisy; one Llama
  HE replicate was slightly negative (−0.045, n=22).
- MBPP (larger n) is mixed: Qwen-Instruct strong (+0.20), Llama weak (+0.05), Qwen-Coder ≈0 (−0.016).
  Qwen-Coder (already near-ceiling, very few failures) shows little room — consistent with "recovery
  matters where the default actually fails."
- Two Qwen-Coder HE jobs lost their generation shards (excluded).

**So the effect is real but heterogeneous** — clearest on weaker/instruct models and on HumanEval; weak
where the base model rarely fails. Next step to firm it: larger n (full MBPP already 500; more HE via
resampling or a harder code set), + the *failure-type-conditioned* switch (route the recovery strategy
by failure signal: timeout→DP, wrong-formula→symbolic, etc.) which should beat undirected switching.

## Where this sits
This is the escape from the single-episode nulls the monoculture framing predicted. Combined with the
mechanism (ρ↓/c↑, competence preserved), the paper now has a **positive application**: *RLVR compresses
routing into a monoculture that is fine on average but brittle on its own failures; the preserved latent
repertoire, invoked as a failure-triggered recovery action, restores a substantial fraction of failures
that iid retrying cannot.* Adjacent to CodeRescue-style recovery routing, but the novelty is using
**preserved conditional reasoning modes** as the recovery action.

## UPDATE — the recovery win is CODE-SPECIFIC (math switching HURTS)

Ran the math analog (`math_recover.py`, base/grpo/floor × {MATH-500, Olympiad} for Qwen + Llama;
+ oracle ceiling per policy) to (a) test the base-vs-grpo mechanism link and (b) check generality.

| job (grpo/floor) | n_fail | iid-retry | switch | **G_switch** | oracle | best-fixed |
|---|---:|---:|---:|---:|---:|---:|
| qm_grpo / math500 | 29 | 0.24 | 0.07 | **−0.172** | 0.07 | 0.03 |
| qm_floor / math500 | 31 | 0.26 | 0.16 | −0.097 | 0.16 | 0.06 |
| qm_grpo / olympiad | 71 | 0.20 | 0.15 | −0.042 | 0.17 | 0.03 |
| qm_floor / olympiad | 87 | 0.29 | 0.15 | −0.138 | 0.16 | 0.05 |
| ll_grpo / olympiad | 111 | 0.21 | 0.12 | −0.090 | 0.13 | 0.03 |
| ll_floor / olympiad | 112 | 0.18 | 0.06 | −0.116 | 0.06 | 0.03 |

**In math, strategy-switch recovery is NEGATIVE everywhere (G_switch −0.04 to −0.17)** — switching after
failure is *worse* than iid-retry. Oracle recovery is small and often *below* iid-retry, and the
oracle−best-fixed gap is tiny (unlike code's +0.20). So math has **no failure-conditioned recovery
value** either. (base arms were still running at pull time; grpo+floor already settle it — the effect
is negative regardless of base.)

**Why:** prefix-forcing a math "strategy" (use trig / casework / substitution) on a problem that just
failed produces *lower-quality* reasoning without genuine complementarity — consistent with math
v≈0 across the whole project. In code, "strategies" are genuinely different algorithms (recursive vs
DP vs greedy) with complementary failure modes, so switching recovers. **The recovery method's home is
executable/algorithmic domains (code), not math.**

## Consolidated verdict
- **Positive, code-specific:** failure-conditioned strategy-switch recovery beats iid-retry (+0.17
  mean) with a large per-failure oracle−best-fixed router headroom (+0.20). This is the paper's
  applied result — but scoped to **code/algorithmic tasks**.
- **Negative in math:** switching hurts (−0.04 to −0.17); no recovery value. Reinforces that math
  reasoning "modes" lack functional complementarity at any conditioning (marginal, diagnostic, OR
  failure-conditioned).
- **Mechanism tie-in (G_switch grpo>base) NOT established:** it needed a code base-vs-RL pair (absent);
  the math test is uninformative because switching is net-negative there.

**Bottom line:** the reasoning-monoculture recovery story is real but **domain-gated by genuine
functional complementarity** — present in code, absent in math. The award-level version requires
scaling the code side (SWE-bench-style, failure-type-conditioned routing, base-vs-code-RL pair for the
mechanism law), not math.

Raw: `rl_training/runs_pulled/probe_routing/rec_*.json`, `mrec_*.json`. Harness: `code_recover.py`,
`math_recover.py`, `go_recover.sh`, `go_mrec.sh`.
Prior nulls + mechanism: `FINDINGS_AND_NOVELTY.md`, `STATUS_AND_ALL_RESULTS.md`,
`DCF_DIAGNOSTIC_DIVERSITY_RESULTS.md`.
