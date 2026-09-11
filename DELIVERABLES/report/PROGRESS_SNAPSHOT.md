# Progress Snapshot v2 — Verified Self-Improvement: the Coverage Bound, and the hunt for a composition mechanism
_Updated 2026-09-10. Canonical detail log: `ADAPTIVE_FORGETTING_RESULTS.md` §79–§107b. All results pushed to git. Feedback-ready._

## 0. Honest one-paragraph verdict
Verified self-improvement (RFT/ReST-EM/GRPO) **amplifies reachable competence but does not expand it**. We formalize this coverage
bound, mechanize it (RFT ≫ GRPO because RFT expands coverage while RL only reweights), quantify it (gain ∝ headroom, across model
sizes and domains), and show it **resists five natural interventions** (repair, archive, delayed-value selection, curriculum
escalation, decomposition-recomposition). This is a **strong, honest, rigorously-controlled paper** — but NOT yet a clear award: the
attempted *positive methods* are modest or fail. One last positive-mechanism swing (process-level wiring supervision) is running now.

## 1. What is SOLID (survives scrutiny + the reviewer's corrections)
- **RFT ≫ GRPO for OOD** — shared checkpoint, depth-7 OOD: RFT+ 0.745/0.750/0.760 (3 seeds) vs GRPO 0.555/0.530/0.535 (+0.21). GRPO's
  train reward rises 0.375→0.6 (it LEARNS) but OOD stays flat ⇒ *reweighting, not coverage expansion*. Replicates directionally in math.
- **Coverage is causal + headroom-gated** — 1000-panel: full 0.790 vs coverage-blocked 0.625 (+0.165), ≈ random-removal 0.795 (not
  volume). Effect vs size: 1.5B +0.165 → 3B ~0 → 7B ~0 → 14B ~0 at fixed difficulty; **partially returns at 7B/depth-14 (+0.02) once base
  coverage drops to 58%** ⇒ headroom-gated AND weak-model-amplified.
- **A robust ceiling null-battery** — repair (recovers only 3%/16%/40% of frontier at 1.5B/7B/14B), archive-preservation, delayed-value
  selection, curriculum escalation (nulls vs fair fixed-hard control), decomposition-recomposition (see §3) — none expands competence.
- **A clean DAG testbed** — a new domain (`comp_dag.py`) where composition (data-flow wiring) is PROVABLY the bottleneck: oracle-component
  recovery 9.9% at 1.5B (vs 100% in linear pipelines), rising to 52.7% at 7B (capability-gated). Reusable instrument.

## 2. Theory — the Headroom Bound (corrected, §101; supersedes the buggy §98)
- **Operational reachability** `R_{C,τ}(G)={x : s_G(x;C) ≥ τ}`, s = Pr(verified solution within budget C). (Fixes the §98 bug that
  `R_k={x:p>0}` is budget-independent, `R_k=R_1`.)
- **Zero-success is a cohort, not p=0** — one-sided bound `p ≤ 1−0.05^{1/n}`; report n.
- **Thm 1 (support-boundedness):** verified-only training gives no direct gradient on unreached x; gains there = *measured* generalization,
  not a free set.
- **Cor 2 (headroom-gating):** new-coverage gain ≤ headroom · transferable-value → predicts the size/domain curve. (Reachability ≠ reliability.)
- **Thm 3 (acquisition-cost factorization):** decomposition lowers acquisition cost only under explicit assumptions (valid decomposition,
  sound local verification, reusable components, compatible interfaces) — not a free lunch.
- Dropped: "RFT cannot acquire outside support" (softmax support is huge), headroom-independent absolute-gain target (metric ceiling `Acc−a≤1−a`).

## 3. What FAILED or is MODEST (honest)
- **Decomposition-distillation:** small OOD gain (+0.06/+0.02/+0.015 at 1.5B/7B/14B, shrinks with headroom) AND **not efficient** —
  token-instrumented E0: DEC round = 2.1× RFT tokens, ~1.1–1.5× cost-to-accuracy. It is a *local-solving scaffold*, not a composition teacher
  (E1: composition trivial in linear domain), and not a speedup.
- **Self-expanding loop:** does NOT diverge (depth-8 dec 0.810 vs rft 0.795, +0.015).
- **E2 recomposition (post-hoc rewiring):** FALSIFIED — on held-out DAG wiring, B (plain refs) 0.83 > C (random-aug) 0.81 > F (recomposition)
  0.77. Rewiring interventions HURT.

## 4. RUNNING NOW — the two award-relevant swings
- **Portfolio (1094-main, GPU0-4):** composition-teaching hypotheses vs held-out DAG wiring, all vs B=0.83 baseline:
  - **Tvalue (Route B):** verified intermediate VALUES inlined in targets → dense data-flow/wiring supervision.
  - **Tplan:** explicit dependency-graph plan before code.
  - **cur:** curriculum (train 3–5-node DAGs → transfer up to 6-node).
  - **C:** random-augmentation control.
  - Award criterion: any arm beats B AND the gain PERSISTS at 7B ⇒ positive composition mechanism whose benefit doesn't vanish with scale.
- **W2:** 14B depth-14 headroom (completes the return-with-headroom curve at scale).

## 5. The two routes to a paper (decision)
- **Route A (safe, strong):** the **coverage-bound LAW paper**. Thesis: *self-improvement amplifies, it doesn't expand.* The null battery
  becomes the EVIDENCE for a limit; RFT≫GRPO is the mechanism; the headroom curve is the quantification. Strong main-track, shortlist-plausible
  if framing/rigor are excellent. Does NOT depend on any pending result.
- **Route B (high-risk, high-reward):** if the portfolio's Tvalue/Tplan/cur beats B and holds at 7B → lead with the positive composition
  mechanism (validated bottleneck via E1-DAG + matched controls). This is the clear-award version.

## 6. What I recommend / need feedback on
1. Do you want to COMMIT to Route A as the paper now (I start a clean paper-shaped draft + harden RFT≫GRPO with a fixed-labeled-bank
   analysis), while Route B runs as the swing?
2. Is the coverage-bound *law* framing the thesis you want, or do you want to keep chasing a positive method until one lands?
3. Scope: 1.5B+7B is enough for the law; 14B adds cost. Keep 14B in or drop to save compute?
4. Any hypothesis you want added to the portfolio (e.g., process-level RL from the B-trained checkpoint, contrastive wiring)?

## 7. Infra
72 GPUs = 3 SSM-main nodes (ports 1093/1094/1095, profile greenlandw) + 6 workers (via `sshpass -p '' ssh -p 2222`, auth "none").
New code this cycle: `comp_repair.py`, `comp_decompose.py` (token-instrumented), `comp_oracle_diag.py`, `comp_dag.py` (DAG domain +
bank/eval/target-format modes). Death-proofed: report + code in git after every result; pods ~24h TTL; pools regenerate deterministically.

---
# v3 UPDATE (2026-09-10, later) — Route B resolved + the unifying law
## Route B result (dense data-flow supervision on the DAG domain)
- **Seed-robust positive at 1.5B (§109):** Tvalue 0.857 vs plain-distill B 0.787 = **+0.07, non-overlapping 3-seed ranges**, on held-out DAG wiring.
- **Placebo PASSES (§110):** scrambled-value targets = 0.780 ≈ B 0.787 << Tvalue 0.857 → the gain is the **data-flow signal**, not target length.
- **But capability-gated (§110):** gain shrinks +0.07 (1.5B) → +0.015 (7B), tracking the composition bottleneck (E1-DAG oracle-recovery 9.9%→52.7%).
  The clean "scale-invariant unlock" award criterion is NOT met.

## THE THESIS (recommend building the paper on this)
**Self-improvement amplifies the binding bottleneck; the amplifiable gap closes with capability.** Unifies every result:
coverage-expansion (RFT, headroom-gated) · composition-teaching (Route B, wiring-gated, PLACEBO-VALIDATED designed confirmation) ·
repair (feedback-usage-gated) · RFT≫GRPO (expand vs reweight). One law, one mechanism per bottleneck, all capability-gated.

## Honest award read
Strong, shortlist-PLAUSIBLE main-track paper IF framed on the unifying law with Route B as the keystone (designed intervention that
hits a specific bottleneck and helps in proportion to it, placebo-controlled). NOT a guaranteed award (no scale-invariant SOTA method).

## FEEDBACK NEEDED
1. Build the paper on the unifying "amplify the binding bottleneck" law (recommended), or insist on chasing a scale-invariant method?
2. Route B is the keystone positive — keep hunting for a variant that persists at scale (e.g., process-level RL, harder DAGs at 7B), or lock the story?
3. OK to start the actual paper draft now (intro/thesis/figures), or more experiments first?
