# Counterfactual Recovery Atlas — GO (2026-09-02)

First experiment of the **CRO (Counterfactual Recovery Optimization)** direction. Builds the
failure × recovery-option landscape at scale on executable code, and tests whether an intelligent
per-failure recovery router is worth building.

**Setup:** `code_recover.py` with 8 engineering recovery options (root_cause, boundary, algo_replace,
complexity, defensive, rewrite, spec_reread, builtins). Per problem: DEFAULT (4 iid) → on failure,
matched-budget IID-RETRY (12) vs each recovery STRATEGY (switch pool). Exec-verified. 6 completed jobs:
Qwen2.5-Instruct + Llama-3.1 × {MBPP ×2, HumanEval}. (3 Qwen-Coder jobs crashed shards — recurring
Qwen-Coder-on-main issue; excluded.) Aggregated over default-FAILED problems.

## Results (per default-failed problem)

| model×bench | n_fail | iid-retry b | oracle | best-fixed | **oracle−best-fixed** | frac A^rec>0.2 | #options-best |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen-Instruct MBPP | 134 | 0.19 | 0.33 | 0.15 | +0.18 | 0.17 | 7/8 |
| Llama MBPP | 193 | 0.27 | 0.31 | 0.10 | +0.21 | 0.10 | 8/8 |
| Qwen-Instruct MBPP(rep) | 135 | 0.19 | 0.34 | 0.16 | +0.19 | 0.18 | 8/8 |
| Llama MBPP(rep) | 203 | 0.30 | 0.35 | 0.16 | +0.19 | 0.14 | 8/8 |
| Qwen-Instruct HE | 15 | 0.27 | 0.73 | 0.47 | +0.27 | 0.53 | 5/8 |
| Llama HE | 37 | 0.46 | 0.46 | 0.19 | +0.27 | 0.14 | 7/8 |

**Aggregate:** mean oracle − best-fixed = **+0.217**; mean oracle − iid-retry = +0.141; mean fraction of
failures with max_z A^rec > 0.2 = **0.21**.

**Specialization (which recovery option is best per failure):** highly heterogeneous — 7–8 of 8 options
are the *best* recovery for some subset of failures. `algo_replace` is the modal winner (≈use a
different algorithm/data-structure), but `boundary` (edge cases), `defensive`, `root_cause`,
`complexity`, and `builtins` each uniquely recover different failures.

## Verdict — GO (build the full CRO method)

All three go-criteria are met:
1. **oracle − best-fixed = +0.22 ≥ 0.15 ✓** — a per-failure recovery router has ~22 pts of learnable
   headroom *over the single best fixed strategy* (and +14 over iid-retry). The routing is the value,
   not any one strategy.
2. **frac(max A^rec > 0.2) = 0.21 ✓** (borderline; 0.10–0.53) — a meaningful mass of failures has a
   recovery option that beats iid-retry by >0.2.
3. **Best recovery option is strongly heterogeneous across failures ✓** — 7–8/8 options each win on
   some failures → *different failures require different recovery strategies*, the core CRO premise.

This is the first result in the project that justifies a **method**, and it is grounded in the one
robust positive (code recovery), not generic diversity. It is also domain-gated exactly as predicted:
strong in executable code (real algorithmic complementarity), negative in math (`MONOCULTURE_RECOVERY_RESULTS.md`).

## Next step (deliberate compute commitment — needs user go-ahead)
Build **CRO** proper:
1. **Failure-conditioned gate** `g_φ(z | s_F)` with **RETRY as an explicit action** (math taught: sometimes
   retry beats switch) — trained toward the counterfactual-optimal `q*(z) ∝ exp(A^rec/τ)`.
2. **Learned latent recovery options** (soft prompts / LoRA adapters) replacing hand-written strategies.
3. **A^rec-weighted consolidation:** distill high-A^rec recovered solutions back into the base policy
   (failure → discover alternate solution → verify → amortize into pass@1) — the self-improving loop.
4. Evaluate on competitive-programming (CodeContests/TACO/APPS — needs a stdin/stdout verifier, not yet
   built) + NP/planning; math as negative control.
Baselines: GRPO, DPH-RL, SetPO, CARE, Fission-GRPO, BPO, BAPO. Headline equation:
`A^rec(s,z) = Q(s,z) − Q(s,retry)`.

Raw: `rl_training/runs_pulled/probe_routing/atl_*.json`. Harness: `code_recover.py` (8 eng. options),
`go_recover.sh`. Recurring infra note: Qwen-Coder code jobs on the main node crash shards (3/9 here).
