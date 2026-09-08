# CRO Experiment #1 — The Learned Recovery Router (preliminary, 2026-09-03)

*Question (CRO §7, component 1): can a failure-conditioned gate `g_φ(z | s_F)` predict, from the failed
state (question + failed code + error signature), WHICH recovery option to switch to — and thereby close
a meaningful fraction of the oracle−best-fixed headroom the Atlas reported (+0.22 MBPP, +0.05–0.13 TACO)?*

**GO criterion (pre-registered):** routed recovery closes ≥50–60% of the oracle−best-fixed gap.

## Verdict: NO-GO for the learned single-option router (as specified). Honest null.

Data: pooled default-FAILED problems from `code_recover` (MBPP+HumanEval × {Qwen-Instruct, Llama,
Qwen-Coder, Qwen2.5-Coder-14B}) and `taco_recover` (TACO easy/med/hard × families), all over the same 8
engineering recovery options with router features. Router = TF-IDF(question+err+failed-code) → logistic
regression, 5-fold stratified CV, held-out routed recovery.

| pool | failures | iid-retry | best-fixed | routed (CV) | oracle | gap closed | router − iid |
|---|---|---|---|---|---|---|---|
| MBPP+HE (4 families) | 462 | 0.184 | 0.093 (`algo_replace`) | **0.078** | 0.297 | **−7.4%** | **−0.106** |
| TACO | 1922 | 0.082 | 0.035 (`defensive`) | 0.058 | 0.111 | +30.1% | −0.024 |
| CODE pooled | 2384 | 0.102 | 0.044 (`defensive`) | 0.065 | 0.147 | +19.9% | −0.037 |
| MATH (neg. control) | — | — | — | — | — | — | no failed-state features |

The learned router closes well under the 50–60% bar and, more damningly, **underperforms plain iid-retry
in every pool** (router − iid < 0). The single best recovery option is not reliably predictable from the
failed state with these features.

## Why it fails (two findings, both honest)

1. **The oracle gap is mostly ties, not specialization.** Only ~12–30% of failures are "routable" (any
   switch option recovers). Among those, most are recovered by MANY options simultaneously — the argmax
   label collapses onto `root_cause` (348/462 MBPP, 1764/1922 TACO). So per-problem "best strategy" is
   largely non-unique. The Atlas's heterogeneous-specialization picture is thinner than it looked once
   you condition on a *unique* winner.
2. **Budget confound (being fixed by the denser reruns).** Each strategy arm gets only ~2 samples
   (`n_rec/8`), while iid-retry gets the full 12. So per-strategy recovery `r_z` is under-powered vs iid,
   which mechanically makes best-fixed < iid (0.093 < 0.184). The denser runs (`fa_*_mbppD`, n_rec=24 →
   ~3–4 samples/strategy) are in flight to give a fair per-budget test.

## The reframe that survives: portfolio recovery, not routing

The earlier positive result — "strategy-switch beats iid-retry +0.04 to +0.27 at matched budget"
(`MONOCULTURE_RECOVERY_RESULTS.md`, `code_recover` merge) — was **matched-budget DIVERSE sampling**:
spread the 12-sample retry budget across the 8 strategies vs 12 iid samples. That win is real. Routing to
a *single* predicted option is what fails. So the value of the latent repertoire on failure is:

> **diversify the retry distribution (a portfolio), NOT learn a gate that picks one option.**

This flips CRO §7 component 1. The defensible method is not `g_φ(z|s_F)` selecting an option; it is a
**failure-conditioned diversification** of the retry policy (increase entropy over recovery options once a
failure is detected), consolidated by the matched-budget recovery advantage. The RETRY-vs-diversify gate
(when to keep retrying iid vs when to diversify) may still be learnable and is the salvageable piece —
but option-level routing is not supported by the data.

## What this means for the paper
- Keep the mechanism (ρ↓/c↑) and the four nulls — unchanged and solid.
- Downgrade "learned recovery router" from a headline method to a **reported negative** (honest; strengthens
  credibility). The Atlas oracle gap is real but dominated by ties and not option-routable.
- Promote **matched-budget portfolio recovery** to the positive result, with the domain-gating
  (code positive, math negative: G_switch Llama −0.174 / Qwen-Coder −0.091 / Qwen-Instruct +0.000).
- Open question worth one more experiment: the **retry-vs-diversify gate** (binary, not 8-way) — does a
  learned "should I diversify now?" signal beat always-diversify? That is the last learnable object.

## Reproduce
`python -m rl_training.build_cro_router` → `rl_training/runs_pulled/cro_router_result.json`.
Pulled data: `rl_training/runs_pulled/{cro_router,taco_atlas,math_negctrl}/`.
