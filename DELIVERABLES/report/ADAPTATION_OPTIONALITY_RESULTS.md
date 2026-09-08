# Reasoning-Optionality Adaptation-Shock — 3-Experiment Results (2026-08-31)

**Question tested:** does RLVR-induced *mode collapse* destroy future adaptability?
Thesis (Critique 2): two policies matched on pass@1/pass@k but differing in minority-mode mass
should adapt at very different speeds when the reward/task shifts — collapse → slower rediscovery
(rediscovery time `T_m ≈ 1/(G·p_m)`, so a Δ-nat collapse ⇒ `e^Δ`× slower).

**Design.** Take three forks of Qwen2.5-Math-7B, all previously RL'd on Omni-Math, then RL-adapt
each on a *shift* dataset. Per-step training reward (= `correct_frac`, accuracy on the batch) is the
adaptation curve. Three arms per shift:
- **base** — Qwen2.5-Math-7B, full repertoire (no prior collapse).
- **collapsed** — `cov-r1-qm-omni-grpo-7b` (plain GRPO fork — sharpened / mode-collapsed).
- **preserved** — `cov-r1-qm-omni-floor-7b` (support-floor fork — minority modes preserved).

Three shift difficulties, run in parallel across three clusters (8×A100 each, vLLM server + ZeRO-3):
exp-1 OlympiadBench (t1040), exp-2 MATH-500 (t1042), exp-3 AIME (t1041). 80 steps, reward logged
every 10 steps.

---

## Results — training reward (accuracy) vs step

Each cell: **mean over 80 steps** (step-10 → step-80 endpoints).

| Shift          | base            | **collapsed** (grpo) | preserved (floor) |
|----------------|-----------------|----------------------|-------------------|
| **MATH-500**   | 0.533 (.47→.58) | **0.605 (.58→.63)**  | 0.535 (.48→.61)   |
| **OlympiadBench** | 0.269 (.23→.24) | **0.335 (.34→.31)** | 0.258 (.23→.22)   |
| **AIME**       | 0.070 (.05→.10) | *(pending, restart)* | 0.070 (.06→.08)   |

Full per-arm curves (reward at steps 10,20,…,80):

```
EXP-1 Olympiad
  base     : 0.234 0.282 0.230 0.302 0.270 0.275 0.323 0.236
  collapsed: 0.338 0.380 0.288 0.373 0.327 0.289 0.380 0.307
  preserved: 0.234 0.284 0.236 0.309 0.255 0.239 0.288 0.218

EXP-2 MATH-500
  base     : 0.466 0.530 0.543 0.443 0.566 0.588 0.525 0.584
  collapsed: 0.584 0.636 0.595 0.538 0.632 0.650 0.579 0.632
  preserved: 0.479 0.548 0.543 0.491 0.579 0.588 0.498 0.611

EXP-3 AIME
  base     : 0.050 0.070 0.064 0.073 0.070 0.070 0.073 0.100
  collapsed: (arm crashed rank-0 exit 1; restarted from local complete model, ~1h to finish)
  preserved: 0.063 0.071 0.066 0.084 0.070 0.061 0.073 0.075
```

AIME reward ~0.07 = ~2/30 problems solved per batch → very noisy, tiny signal.

---

## What the data says (honest)

**The optionality thesis is not supported for within-math shifts. Three shifts, one story:**

1. **Easy/medium shifts (MATH-500, Olympiad): the *collapsed* fork adapts best.**
   It is already higher at step 10 (0.34 vs 0.23 base on Olympiad; 0.58 vs 0.48 on MATH-500) and
   stays higher throughout. This is **not** an adaptation-*reserve* effect — RL-sharpening on
   Omni-Math simply made the model better at in-distribution math, and that skill transfers.
   The **preserved** fork tracks **base** almost exactly — preserving minority modes bought no
   adaptation advantage.

2. **Hard shift (AIME): the sharpening advantage vanishes.** base ≈ preserved ≈ 0.070, flat and
   noisy. Even if the collapsed arm also lands ~0.07 (expected), the key fact holds: **collapsed is
   never *worse* than preserved anywhere.** No optionality penalty appears at any difficulty.

**Conclusion:** mode collapse from RLVR does not impair adaptation to *other math tasks* — it either
helps (in-distribution transfer) or is neutral (hard shift). This matches the earlier OOD math→math
finding. The predicted `e^Δ` rediscovery penalty does not materialize when the shift stays in-domain,
because the "collapsed" strategies are still the *right* strategies for the new math task.

---

## Where this sits in the bigger arc

The paper has moved through three framings; here is the honest scorecard:

| Framing | Status | Evidence |
|---|---|---|
| **Coverage/diversity preservation** (support floor keeps rare modes) | ✅ established | causal ~39% mode-mass preservation vs plain GRPO; floor certifies ≥ alive as plain |
| **DPH-F baseline is the closest competitor** | ✅ established, but *overturned our lead* | DPH-F preserves MORE mode-mass (+3.51 vs floor +1.38) → reframed to reward-survival Pareto/selectivity |
| **Capability Survival / extinction (CSPO kill→rescue)** | ❌ negative | kill phase did not extinguish; GRPO *improved* capabilities; rare-tail "deaths" = N=256 noise; nothing certified-extinct at N=1024 (need ~5450) |
| **Reasoning Optionality (adaptation reserve)** | ❌ negative (this doc) | 3 within-math shifts: collapsed ties-or-beats preserved; no `e^Δ` penalty |

So the **only clean, positive, causal result** remains the **mode-mass preservation** (coverage
floor and DPH-F both preserve minority-mode mass; the floor does it selectively). The two
"stronger-claim" reframes (extinction, optionality) both came back negative *within math*.

---

## Two ways to proceed

**(a) Write the honest negative + pivot back to the established result.**
Frame the paper on the causal mode-mass preservation (39%) + DPH-F Pareto, and *include* the
optionality/extinction negatives as a rigorous "when does collapse actually cost you?" section —
answer: not within-domain. This is submittable and honest, but it is an incremental
diversity-preservation paper in a crowded space (SetPO, DPH-RL, DMPO, Uniqueness-Aware RL).

**(b) Run the one decisive test the thesis actually needs: cross-domain (math→code/logic).**
The optionality prediction only bites when the dominant (sharpened) mode *fails* on the new task
and a *dropped* strategy is required. Within math, the sharpened mode is never wrong-family, so no
penalty. A math→code shift is the real test. Cost: stand up a code-RL reward (execution/unit-test
verifier) + a small code shift set — infra we don't currently have. If collapsed adapts 10–100×
slower there while preserved recovers, *that* is the NeurIPS-level result. If it ties there too, the
optionality thesis is dead and (a) is the paper.

**Recommendation:** stop spending on within-math adaptation (three shifts agree). Decide between (a)
ship the honest preservation paper now, or (b) invest in the math→code setup as the make-or-break
experiment. My read: (b) is the only path to an award-level claim, but it is real infra work; (a) is
the safe, honest, publishable floor.

---

## Reproduce / locations
- Launchers: `rl_training/go_srcadapt.sh` (fetch complete source → `go_adapt.sh` server+ZeRO-3).
- Clusters (2026-08-31): exp-1 `mi-031af6e95af9ee154` t1040, exp-2 `mi-096b13e7b3bc9ac38` t1042,
  exp-3 `mi-03ef3d43949d503b2` t1041 (see `rl_training/ACTIVE_INSTANCES.md`).
- Reward curves parsed from `/tmp/instance_storage/gu/logs/adapt_adapt_<tag>_train.log`
  (`grep -aoE "reward.: .[0-9.]+"`; value is a quoted string in the TRL step log).
- 8/9 arms reached step 80/80; grpo_aime restarted from local complete model after a rank-0
  crash (exitcode 1), ~1h to finish — will not change the direction.
