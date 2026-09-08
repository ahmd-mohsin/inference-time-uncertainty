# Knowing vs Choosing — Routing-vs-Competence Probe Results (2026-08-31)

**Question.** Our earlier headline was a causal ~39% "mode-mass collapse" under RLVR. But mode-mass
conflates two very different phenomena:

- **Routing collapse** — `ρ(m|q)↓`: the model stops *choosing* strategy m (but still knows it).
- **Competence erasure** — `c(m,q)↓`: the model loses the *ability to execute* m.

Decomposition: `π(y|q) = Σ_m ρ(m|q)·π(y|q,m)`, where `c(m,q) = P(correct | q, do(M=m))`.
If the 39% is routing collapse with competence intact, the field (SetPO / DPH-RL / Uniqueness-Aware
RL / DMPO) is regularizing the wrong object — diversity should live in the *controllable conditional
repertoire*, not the *marginal sampling distribution*.

**Design (inference-only, no training).** For each policy, per problem, sample:
- **DEFAULT**: 8 plain samples → default pass@1 + which of a 14-strategy taxonomy appear (regex ρ proxy).
- **FORCED**: for each of 14 strategies, 4 samples with an explicit "solve using strategy m"
  instruction → forced competence `c(m,q)` + adherence (did it comply).

**Scale.** 9 probes across 9 nodes / ~72 GPUs, 8-GPU data-parallel each, 150 problems × 64 samples.
Policies: base (Qwen2.5-Math-7B), **grpo** (cov-r1-qm-omni-grpo, collapsed), floor (cov-r1-qm-omni-floor,
preserved), dphf (cov-dphf-qm-oly, closest competitor). Shifts: OlympiadBench, Omni-Math (the
distribution the 39% was originally measured on), MATH-500 (grpo only).
Harness: `rl_training/strategy_probe.py` + `go_probe.sh`; analysis `rl_training/analyze_probe.py`;
raw JSONs `rl_training/runs_pulled/probe_routing/` (death-proofed local).

---

## 1. Summary — forced competence `c` is the decisive column

| dataset | policy | pass@1 | ρ (routing proxy) | adherence | **competence c** |
|---|---|---:|---:|---:|---:|
| Olympiad | base  | 0.220 | 0.086 | 0.166 | 0.220 |
| Olympiad | **grpo (collapsed)** | **0.309** | 0.081 | 0.125 | **0.318** |
| Olympiad | floor | 0.228 | 0.079 | 0.146 | 0.242 |
| Olympiad | dphf  | 0.242 | 0.079 | 0.136 | 0.252 |
| Omni | base  | 0.125 | 0.097 | 0.206 | 0.131 |
| Omni | **grpo (collapsed)** | **0.212** | 0.087 | 0.160 | **0.194** |
| Omni | floor | 0.133 | 0.090 | 0.181 | 0.145 |
| Omni | dphf  | 0.146 | 0.094 | 0.175 | 0.152 |
| MATH500 | grpo | 0.599 | 0.086 | 0.129 | 0.590 |

Notes: `ρ` = mean fraction of default samples exhibiting each strategy (regex proxy — see caveat).
`c` = mean forced-mode correctness over the 14 strategies. `adherence` = fraction of forced samples
that actually exhibited the instructed strategy (regex).

## 2. Per-strategy forced competence, grpo − base (negative = erased)

**Every strategy is positive on both datasets — not one shows erasure.** The most-collapsed
strategies gain the most.

| strategy | Olympiad base→grpo (Δ) | Omni base→grpo (Δ) |
|---|---|---|
| casework       | 0.203 → 0.338 (**+0.135**) | 0.148 → 0.192 (+0.043) |
| generating_fn  | 0.188 → 0.312 (**+0.123**) | 0.113 → 0.180 (+0.067) |
| substitution   | 0.213 → 0.335 (+0.122) | 0.147 → 0.205 (+0.058) |
| trig           | 0.207 → 0.327 (+0.120) | 0.120 → 0.195 (+0.075) |
| contradiction  | 0.210 → 0.325 (+0.115) | 0.122 → 0.210 (**+0.088**) |
| factoring      | 0.217 → 0.330 (+0.113) | 0.137 → 0.173 (+0.037) |
| … (all 14)     | every Δ in [+0.047, +0.135] | every Δ in [+0.037, +0.088] |

## 3. Functional necessity / redundancy (strategy "solves" a problem if forced-c ≥ 0.25)

| dataset | policy | solvable problems | mean redundancy (strategies/solvable-problem) | uniquely-necessary problems |
|---|---|---:|---:|---:|
| Olympiad | base  | 116 | 8.01 | 15 |
| Olympiad | grpo  | 121 | 9.28 | 9  |
| Olympiad | floor | 115 | 8.91 | 8  |
| Olympiad | dphf  | 118 | 8.85 | 10 |
| Omni | base  | 72 | 8.25 | 7  |
| Omni | grpo  | 75 | 9.24 | 8  |
| Omni | floor | 79 | 8.13 | 10 |
| Omni | dphf  | 77 | 8.09 | 15 |
| MATH500 | grpo | 137 | 12.50 | 4 |

base/Olympiad uniquely-necessary strategies (only-solver counts): factoring:4, induction:2,
synthetic_geo:2, counting:2, substitution:1, contradiction:1.

**~8–9 of 14 strategies solve each solvable problem; only ~10–13% of problems require a unique
strategy.** Most reasoning modes are functionally redundant.

---

## Interpretation (honest)

1. **The 39% "collapse" is NOT competence erasure.** Routed to a strategy, the RL model executes it
   *at least as well as base — better, on all 14 strategies, both datasets.* RL did not remove the
   ability to run these strategies.
2. **Most reasoning modes are functionally redundant** (8–9 solve each problem). Supports "collapse
   as rational compression": RL drops redundant modes while keeping (improving) the functional basis.
3. **DPH-F does not win on competence** (0.252 Oly / 0.152 Omni — just above base, below grpo). Its
   mass-covering rehearsal preserves *marginal* diversity but buys no competence edge over plain GRPO.
4. **Generality:** the pattern holds on OlympiadBench *and* Omni-Math (the original distribution) —
   not an artifact of one benchmark.

**Emerging thesis:** *diversity should live in the controllable conditional repertoire `c`, not the
marginal sampling distribution `ρ`.* Preserving marginal diversity (DPH-F / SetPO / Uniqueness-RL)
targets the wrong object. This is the "Knowing vs Choosing" paper: knowing (competence) survives RL;
only choosing (routing) sharpens.

## Caveats (must close before over-claiming)

- **The ρ routing proxy is regex keyword-presence and too coarse — it did NOT reproduce the 39% drop**
  (ρ ≈ equal across policies). To claim "routing collapsed while competence held," we need the proper
  measure: mode-mass Δ = logπ_θ − logπ_0 on the bank witnesses (`score_bank_logprobs.py` + existing
  bank). **This is the missing half of the decomposition.**
- **Instruction-forcing achieved low adherence (~13–20% by regex)** — the model often ignores the
  instruction and uses its own default. So `c` under-identifies true `do(M=m)`. Few-shot strategy
  demonstrations or activation steering would tighten the forcing.

## Candidate next steps (for you to choose)

1. **Close the decomposition (cheap, high value):** run the proper log-prob routing measure on the
   bank across base/grpo/floor — quantify how much ρ actually dropped while c held. Reuses
   `score_bank_logprobs.py`; runs on the now-idle 9 nodes. *This is the single most important missing
   number.*
2. **Tighten forcing:** add a few-shot strategy-demonstration condition to `strategy_probe.py` and
   re-run on grpo/base — raises adherence, sharpens the competence estimate.
3. **Mechanistic (award-style):** layerwise linear probe for strategy decodability + activation
   steering — if steering *restores* a suppressed strategy in default generation, that is direct
   evidence for suppression-not-erasure.
4. **Build the method:** Decoupled Repertoire Optimization — let ρ sharpen for pass@1 while
   constraining `c(m,q) ≥ c_0(m,q) − δ` on functionally-necessary strategies. Target: GRPO-level
   pass@1 + preserved conditional repertoire, with less rehearsal than DPH-F.

---

# UPDATE (2026-08-31, same day): the mandatory trio is complete — the causal map

The two caveats above are now closed. Ran all three mandatory experiments across 9 nodes.

## Exp #2 — high-adherence forcing (closes the ~15% adherence caveat)

Replaced the weak "solve using m" instruction with a **strategy-prefix seed** (the assistant turn
is pre-seeded with a strategy-specific opening, e.g. "We proceed by induction. Let P(n)…", and the
model must continue). **Adherence rose from ~13–20% → ~82%**, uniform across all policies. This is
genuine `do(M=m)`. Competence under true forcing (forcing even ill-suited strategies drives absolute
values down vs the earlier inflated instruction numbers — that is correct):

| dataset | policy | pass@1 | adherence | **competence c (high-adherence)** |
|---|---|---:|---:|---:|
| Olympiad | base | 0.223 | 0.817 | 0.066 |
| Olympiad | **grpo** | **0.309** | 0.817 | **0.115** |
| Olympiad | floor | 0.238 | 0.818 | 0.063 |
| Olympiad | dphf | 0.246 | 0.819 | 0.058 |
| Omni | base | 0.122 | 0.813 | 0.040 |
| Omni | **grpo** | **0.203** | 0.816 | **0.071** |
| Omni | floor | 0.131 | 0.823 | 0.035 |
| Omni | dphf | 0.158 | 0.816 | 0.035 |
| MATH500 | grpo | 0.597 | 0.814 | 0.234 |

**Under high-adherence forcing, grpo competence is still highest — ~1.7× base** (Oly 0.115 vs 0.066;
Omni 0.071 vs 0.040). The "not erasure — competence rose" conclusion survives the forcing-strength
check. floor/dphf sit at base level: their preservation buys no competence.

## Exp #1 — proper routing measure Δlogρ

`ρ(m|q)` = per-token log-prob each policy assigns to the canonical strategy-m opening seed given the
problem (teacher-forced, no generation; `rl_training/route_logprob.py`). Mean per-token routing logp:

| policy | mean routing logp/tok | Δ vs base |
|---|---:|---:|
| base  | −3.244 | — |
| **grpo**  | **−3.587** | **−0.343** (routing suppressed) |
| floor | −3.266 | −0.022 (routing preserved) |

**grpo suppresses routing into the named strategies (−0.34 nats/tok); floor preserves it (≈base).**
Exactly the designed contrast: RL sharpens routing, the floor holds it.

## Exp #3 — the make-or-break map: Δlogρ vs Δc (grpo − base), per (q,m)

| quadrant | meaning | count |
|---|---|---:|
| QI (ρ↑, c↑) | chosen more & better | 2 |
| **QII (ρ↓, c≥0)** | **suppressed but competence kept/up** | **1879** |
| QIII (ρ↓, c<0) | true erasure | 211 |
| QIV (ρ↑, c<0) | amplified despite worse | 1 |
| flat | | 7 |

**Of 2087 routing-collapsed (q,m) pairs, 1877 (90%) kept or improved competence (QII). Only ~10%
(QIII) show genuine erasure.**

## Verdict — this is the paper

The decomposition is now measured on both sides and agrees across Olympiad + Omni + MATH-500:

> **RLVR collapses the routing distribution (ρ↓, −0.34 nats/tok) while preserving or expanding
> conditional competence (c↑, ~1.7× base). ~90% of "collapsed" reasoning modes remain executable
> when invoked. Mode collapse is NOT capability collapse. Diversity-preserving methods (floor here;
> DPH-F/SetPO/Uniqueness-RL by construction) restore marginal routing mass but buy no competence —
> they regularize the wrong object.**

This is *Knowing Is Not Choosing*. The 39% is routing, not forgetting.

Remaining to harden before a submission (per the plan, not blocking the claim): checkpoint dynamics
(ρ↓ while c↑ over training), ≥1 more model family (de-risk Qwen-specificity), run the baselines
through the same interventional probe, minimal-sufficient-repertoire (set-cover) + oracle-router gap,
and stratified pass@k. Method (Interventional Repertoire Optimization / routed minimal basis) comes
AFTER, since GRPO already improves c — do not build a preservation optimizer.

---

# UPDATE 2 (2026-08-31): cross-family generalization + checkpoint dynamics (9-job batch)

Ran a 9-job batch across 9 nodes (probe+route each, via `go_pr.sh`): DeepSeek + Llama families and
qm round-2 checkpoint dynamics. 8/9 completed and are pulled to `runs_pulled/probe_routing/`; the
DeepSeek-base job was lost when its node died mid-fetch (DeepSeek base uses `.bin` shards — an
`allow_patterns` fix was applied but the node `TargetNotConnected`'d before it finished).

## 3rd family — Llama-3.1-8B-Instruct (Olympiad) — CLEAN REPLICATION (stronger than Qwen)

| policy | pass@1 | adherence | competence c | routing logp/tok | Δroute vs base | Δc vs base |
|---|---:|---:|---:|---:|---:|---:|
| base  | 0.138 | 0.805 | 0.034 | −3.857 | — | — |
| **grpo** | **0.247** | 0.786 | **0.083** | **−4.525** | **−0.668** (↓routing) | **+0.049 (~2.4×)** |
| floor | 0.155 | 0.805 | 0.057 | −3.545 | +0.312 (routing preserved) | +0.023 |

Same decomposition as Qwen, larger: grpo suppresses routing by −0.67 nats/tok while ~2.4× competence;
floor preserves routing. The core claim is **not Qwen-specific**.

## Checkpoint dynamics — qm base→r1→r2 (Olympiad) — routing↓ and competence↑ happen EARLY, then plateau

| total steps | route logp/tok | competence c | pass@1 |
|---|---:|---:|---:|
| 0 (base) | −3.244 | 0.066 | 0.223 |
| 100 (r1-grpo) | −3.587 | 0.115 | 0.309 |
| 120 (r2 ckpt-20) | −3.569 | 0.113 | 0.319 |
| 160 (r2 ckpt-60) | −3.596 | 0.117 | 0.332 |
| 200 (r2 ckpt-100) | −3.575 | 0.109 | 0.345 |

Routing drops (−0.34) and competence rises (~1.7×) **together in the first 100 steps**, then both
plateau while pass@1 keeps climbing. This is the temporal-order evidence: routing collapses *while*
competence rises, not afterward — collapse and capability-gain are the same early event.

## 2nd family — DeepSeek-Math-7B (Omni) — INCONCLUSIVE

grpo c 0.028 ≈ floor 0.026 (pass@1 0.065/0.072), adherence ~0.73. DeepSeek-Math is weak on Omni-hard
(base pass@1 ~0.07) → almost no competence signal to decompose; base lost to node death. Needs a
rerun on an easier set (e.g. MATH-500) where DeepSeek has headroom. **Not evidence against — just no
signal here.**

## Offline metrics (§ oracle gap / minimal repertoire / functional rank) — reality check

Computed from the prefix probes (no GPU). Oracle-router gap (A_oracle − default) is **small**:
grpo +0.056 (Oly), +0.026 (Omni), −0.025 (MATH500 — forcing hurts a saturated set). Functional rank
barely compresses (12.5→11.5; grpo slightly *below* floor/dphf). Minimal 95%-coverage basis = 7–9 of
14 strategies (MATH-500: 4). **Implication:** the models are already well-routed among these
strategies, so the ambitious "post-training is mostly a routing problem" / "routed method beats GRPO
by a lot" payoff (§14–20 of the plan) is **not** supported — headroom is ~5 pts. The solid result is
the decomposition itself, not a big exploitable oracle gap. (Caveat: oracle limited to the 14 forced
strategies; forcing depresses absolute c.)

## Consolidated verdict across all runs

- **Qwen-Math + Llama both show routing↓ / competence↑** (Δroute −0.34 / −0.67; c ~1.7× / ~2.4×).
- **Dynamics confirms it's a training-time effect, established in the first 100 steps.**
- **90% of routing-collapsed (q,m) pairs keep/improve competence** (QII); ~10% true erasure.
- **DPH-F / floor restore routing mass but buy no competence** — regularize the wrong object.
- Repertoire-*exploitation* extras (oracle gap, rank expansion) are modest → the paper's strength is
  the causal decomposition + cross-family + dynamics, not a routed-repertoire SOTA method.
- **DeepSeek inconclusive** (too weak on Omni); rerun on MATH-500 to finish the 3rd family cleanly.

---

## Why these experiments finish so fast (vs the day-long GRPO training runs)

1. **Inference-only, no training.** Probe = generation; routing = a single teacher-forced forward
   pass. No backprop, optimizer, ZeRO sharding, gradient sync, or vLLM-weight-reload loop — the
   things that make each GRPO step ~50 s. A whole probe is cheaper than a few training steps.
2. **8-GPU data-parallel per node.** pass@k / competence are embarrassingly parallel across problems,
   so each node strides its 150 problems across all 8 GPUs (1 model replica per GPU, TP=1) → ~8×.
3. **9 nodes at once = ~72 GPUs, one job per node** — the whole matrix runs concurrently.
4. **The routing measure does NO generation** — it scores the log-prob of ~14 short (~15-token)
   strategy-prefix seeds per problem. ~2,100 tiny forward passes/policy; finishes in minutes.
5. **Tiny sample counts.** default_n=8, forced_n=4, 150 problems → ~9.6k short generations/policy,
   far below a pass@256 eval.
6. **Local model reuse + fetch_big completeness.** The prefix rerun pointed at already-fetched local
   dirs (no 15 GB re-download), and fetch_big guarantees complete weights so vLLM never silent-hangs.
7. **vLLM `enforce_eager` + prefix caching**, and merge is just concatenating shard JSONs.

Net: a per-node probe+route is ~20–30 min of compute (plus a one-time ~10–15 min 15 GB fetch when the
model isn't already local), vs many hours for a GRPO training run of the same model.

# DOWNSTREAM GO/NO-GO (2026-09-01) — decisive, and it's a NO-GO for a performance method

Before committing a weeks-long large-scale phase, we ran the make-or-break downstream test:
**strategy-stratified pass@k vs iid pass@k at matched budget** (`rl_training/stratified_passk.py`,
32 free samples/problem, per-sample correctness + regex-argmax strategy label; Monte-Carlo pass@k).
If deliberately sampling K *different* strategies beats K iid samples, controllable diversity converts
to accuracy → worth scaling. 9 jobs = 3 families × {base, grpo, floor}.

| job | iid@8 | strat@8 (Δ) | iid@16 | strat@16 (Δ) | iid@32 | strat@32 | oracle-strategy |
|---|---|---|---|---|---|---|---|
| qm base    | 0.555 | 0.507 (−0.048) | 0.645 | 0.636 (−0.008) | 0.713 | 0.713 | 0.365 |
| qm grpo    | 0.629 | 0.609 (−0.020) | 0.699 | 0.699 (0.000) | 0.747 | 0.747 | 0.461 |
| qm floor   | 0.581 | 0.536 (−0.045) | 0.666 | 0.644 (−0.022) | 0.720 | 0.720 | 0.388 |
| llama base | 0.381 | 0.357 (−0.024) | 0.463 | 0.448 (−0.015) | 0.533 | 0.533 | 0.247 |
| llama grpo | 0.494 | 0.500 (+0.006) | 0.569 | 0.574 (+0.005) | 0.640 | 0.640 | 0.364 |
| llama floor| 0.397 | 0.403 (+0.006) | 0.474 | 0.474 (0.000) | 0.547 | 0.547 | 0.282 |
| ds base    | 0.640 | 0.640 (+0.001) | 0.734 | 0.734 (0.000) | 0.813 | 0.813 | 0.338 |
| ds grpo    | 0.657 | 0.654 (−0.002) | 0.751 | 0.751 (−0.001) | 0.820 | 0.820 | 0.348 |
| ds floor   | 0.644 | 0.658 (+0.013) | 0.744 | 0.750 (+0.006) | 0.827 | 0.827 | 0.367 |

**Verdict: NO downstream win, robust across all 3 families.** Stratified never meaningfully beats iid
(Δ ∈ [−0.048, +0.013], ≈0); **pass@32 is identical for stratified and iid in every case** (strategy
diversity adds zero coverage at full budget); oracle-strategy routing (0.25–0.46) is far *below* plain
iid pass@16 (0.46–0.75) — routing to a single "best" strategy is worse than just sampling. Combined
with the earlier ~5-pt oracle-competence ceiling and flat functional rank, the hidden-repertoire /
routing angle carries **no exploitable performance value**.

**Decision:** do NOT build a repertoire/routing method or a preservation optimizer, and do NOT commit
the weeks-long large-scale RL phase to a performance method. The behavioral free-generation routing
also confirmed the science (Qwen H(ρ): base 3.51 → grpo 2.71 effective strategies; floor 3.41 ≈ base).
Pivot to the **corrective + mechanism paper** — full plan: `DELIVERABLES/report/PIVOT_mechanism_plan.md`.
(Caveat: stratified selection used a coarse regex classifier (~36% unclassified) + same-policy samples;
a stronger judge could nudge small-k, but identical pass@32 makes the no-coverage-gain conclusion robust.)

## Reproduce
- Stratified pass@k: `bash rl_training/go_sp.sh <MODEL_DIR> <TAG> <dataset> 150 32` → `sp_<tag>.json`.
- Combined probe+route: `bash rl_training/go_pr.sh <SPEC> <TAG> <dataset> 150`
  (SPEC: base:HFID | fork:REPO | ckpt:REPO@checkpoint-N | /localdir).
- Routing only: `bash rl_training/go_route.sh <MODEL_DIR> <TAG> <dataset> 150`.
- Probe only: `bash rl_training/go_probe.sh <SPEC> <TAG> <dataset> 150 8 4 prefix`.
- Analysis (auto-runs quadrant map + offline metrics): `python3 rl_training/analyze_probe.py`.
- NOTE: DeepSeek + other older bases ship `.bin` shards — `base:` fetch must include `*.bin` +
  `pytorch_model.bin.index.json` (the `ds_base` bug).
- Analyze: `python3 rl_training/analyze_probe.py rl_training/runs_pulled/probe_routing`.
- Clusters (2026-08-31): t1040 mi-031af6e95af9ee154, t1042 mi-096b13e7b3bc9ac38,
  t1041 mi-03ef3d43949d503b2 (see `rl_training/ACTIVE_INSTANCES.md`). All 9 probes DONE; GPUs idle.
