# Recoverability-Constrained RLVR — Consolidated Research Document

*Master doc: problem, motivation, methodology, results-to-date, and the experiment list.*
Supersedes the scattered motivation/results notes. Working title:
**"Recoverability-Constrained RLVR: Preserving Rewardable Reasoning Modes Under Finite Compute."**

---

## 1. What we are trying to solve

RL with verifiable rewards (RLVR, e.g. GRPO) reliably raises a model's single-sample accuracy
(pass@1) but is observed to **narrow its reasoning coverage**: at a large sampling budget `k`, the
*base* model solves problems the RL'd model no longer can (Yue et al. 2504.13837). RLVR trades
breadth for sharpness — it concentrates probability on a dominant correct mode and lets rare-correct
strategies decay until they are effectively unreachable.

**Core question.** *What must an RL policy preserve so that a verified reasoning capability stays
discoverable — and therefore still learnable — under a bounded future sampling/compute budget?*

We answer with a new object (**recoverability**), a new constraint (**a base-anchored, off-policy,
one-sided floor on reasoning-mode mass**), and a finite-budget **certificate**. The payoff framing:
coverage is the *substrate for continued RL* — a mode pruned in round 1 cannot be exploited by round 2.

---

## 2. Motivation (only the parts that still hold)

- **The phenomenon is real and is a *difficulty resonance*.** On a released full-RL model
  (Oat-Zero-7B vs base): AMC (easy) — both saturate; **OlympiadBench (intermediate) — base overtakes
  at k≥128 (a 27-problem lost set), widening to +0.023 at k=1024**; AIME (too hard) — RL leads
  everywhere (no samplable base tail). The crossover exists only where a wide, *unsaturated* fragile
  band exists.
- **It requires full-parameter, heavily-trained RL.** LoRA / light training compresses the crossover
  (loses 3–14 problems vs Oat-Zero's 27). The phenomenon — and thus the method's target — needs
  full-FT, many-step RL. (Methods lesson, not a negative.)
- **Mechanism.** For a correct mode `M` of policy mass `p_θ(M|q)`, the expected on-policy
  policy-gradient routed to it scales as `‖E[g_M]‖ = O(p_θ(M|q))`, because `M` contributes to the
  objective only when sampled (prob `p_θ`). Once RL sharpening pushes `p_θ(M) ≪ 1/K` (group size K),
  `M` is essentially never sampled → receives ~no gradient → is pruned. **Any estimator whose gradient
  support is a finite set of K on-policy samples is support-blind to modes rarer than ~1/K.** No
  reward/advantage reshaping (UCPO, PKPO, RiskPO, Polychromic) escapes the `O(p_θ)` factor — you
  cannot push on a mode you never sample.
- **Why it matters beyond diversity:** the pruned modes are lost *for continued RL* — later rounds
  can't rediscover them. Coverage preservation is about keeping future trainability, not entropy.

---

## 3. Methodology — Recoverability-Constrained RLVR

### 3.1 The object: recoverability (not "coverage", not one trace)
A **reasoning mode** `M_{qm} ⊆ Y` is a *set* of trajectories = a strategy family (not one exact
sequence — a 300–1000-token trace has astronomically small prob even when its strategy is likely).
Its mass `p_{θ,qm} = Pr_{y∼π_θ(·|q)}[y ∈ M_{qm}]`. Its **recoverability** under `K` future rollouts:

```
R_K(M; π_θ) = 1 − (1 − p_θ(M|q))^K
```

Operationally: `p=1e-2, K=256 → R≈0.92` (discoverable); `p=1e-4 → R≈0.025` (effectively extinct for
on-policy RL, even though support > 0). We keep **support / effective(recoverable) support /
correct-answer prob / strategy diversity / pass@k** as *separate* axes (the review flagged we blurred
them).

### 3.2 The constraint: a base-anchored, off-policy, one-sided floor on mode mass
```
p_θ(M_{qm}|q) ≥ α · p_0(M_{qm}|q)        (α=0.5: mode mass may not fall below half the base's)
```
or directly on the deployment quantity `R_K(M;π_θ) ≥ γ · R_K(M;π_0)`. Three properties that make it
categorically different from KL / reward-shaping / replay:
1. **Off-policy / teacher-forced** — evaluated on banked base traces, so the correction gradient's
   magnitude is *independent* of the current `p_θ(M)` → it stays alive as `p_θ→0`, exactly where the
   reward gradient vanishes.
2. **A constraint, not a reward** — acts on `log π_θ` directly; immune to GRPO's group-relative
   "constant advantage cancels" no-op.
3. **One-sided ratchet** — only penalizes dropping *below* the floor → near-zero pass@1 tax.

Contrast: PBA protects *prompts*; global KL protects *distributions*; UCPO enforces *uniformity among
sampled correct solutions*; **ours protects identified rewardable modes at an explicit future budget.**

### 3.3 The certificate (turns replay into a guarantee)
Let `ν_m(y)=π_0(y|y∈M_m,q)`. By Jensen:
```
log p_θ(M|q) ≥ log p_0(M|q) + E_{y∼ν_m}[ log π_θ(y|q) − log π_0(y|q) ]
```
So maintaining the teacher-forced `Δ_qm = E_{y∼ν_m}[log π_θ − log π_0] ≥ log α` **implies**
`p_θ(M) ≥ α·p_0(M)`, hence `R_K(M;θ) ≥ 1 − (1 − α·p_0(M))^K` — a **finite-compute recoverability
guarantee**. Estimate `Δ_qm` from J witness traces with a lower confidence bound `LCB(Δ_qm) ≥ log α`
→ statistically calibrated **"Certified Mode-Support Policy Optimization."**

### 3.4 Implementations
- **expSR** — soft one-sided Lagrangian penalty added to the GRPO loss (`μ·relu(logα + logπ_0 − logπ_θ)`).
- **expPROJ** — hard projected-gradient correction restoring feasibility after each GRPO step.
- **E4 (planned): adaptive primal-dual** — dual `λ_{qm} ← [λ_{qm} + η(logα − Δ_qm)]_+` so only
  *endangered* modes get pressure.
- **E5 (planned): route-level** — protect the strategy *prefix* `p_θ(z|q)`, not the full trace, so RL
  is free to improve reasoning inside a preserved basin ("preserve entry into valuable reasoning
  basins; don't preserve the trajectory").

### 3.5 The theorem to keep (softened, defensible)
For a sample-supported estimator `ĝ = Σ_{i≤K} c_i(Y_{1:K}) ∇log π_θ(Y_i|q)`, `|c_i|≤C`:
`Pr[∃i: Y_i∈M] = 1−(1−p_M)^K ≤ K·p_M`. So when `p_M ≪ 1/K`, the probability of *any* direct
mode-specific gradient observation → 0: **finite-rollout on-policy RL is support-blind below `p∼1/K`.**
(We do NOT claim "irreversible" — parameter sharing can indirectly move `M`; the defensible claim is
support-blindness.)

---

## 4. Results to date (all committed; see `runs_pulled/round2_eval/`)

| # | Result | Evidence |
|---|---|---|
| **Phenomenon** | Difficulty-resonant crossover on a released full-RL model | Oat-Zero-7B vs base: Olympiad base 0.911 vs Oat 0.888 @k=1024; AMC/AIME no crossover |
| **Mechanism (drift)** | RL suppresses the exact modes it loses | 49/49 (Olympiad) & 150/150 (Omni-MATH) base-correct lost traces driven 75–84 nats below base — *reframed as witness-likelihood contraction, not mode extinction* |
| **Exploitability (R3)** | preserved mass is usable by continued RL | 66% of protected Olympiad problems yield a nonzero-advantage GRPO group |
| **Round-2 ceiling** | continued RL from the floor fork reaches a higher large-k ceiling | fragile band n=329, n=1024: r2-floor **overtakes** r2-grpo at k≥32 (+0.0037@64→+0.0068@256, 0.990@256); plain wins small k |
| **E1 extinction** | plain continued-RL extinguishes recoverable modes; floor doesn't | grpo drove **2** problems (#168,#237) to zero recoverable mass (unrecoverable at any K); floor **0** |
| **E3-real** | mode-mass certificate (floor's own bank) | Δ=logπ−logπ₀ on 1055 mode-witnesses: floor +4.40 nats (84% ≥ α-floor, 0.4% collapse) vs grpo −9.99 (25%, 40% collapse); certified 59.6% vs 13.5% |
| **E3-unbiased** | same, on a **held-out** 11,664-witness bank (floor never trained on it) | floor −1.68 (46% ≥ α-floor, **2.3% collapse**) vs grpo −10.11 (23%, **39% collapse**); paired floor>grpo 77%; **grpo collapses ~17× more base modes** |
| **E8 observational** | the `K·p≈1` boundary *predicts* the extinctions | at K=1024 grpo leaves exactly 2 problems below K·p=1 (=#168,#237, extinct), floor 0; grpo killed a mode base solved 33/1024 times, floor preserved it |
| **E8 interventional (25-step)** | honest NULL — underpowered | plain vs floor from base, 25 steps: both barely move (median Δ=0), no K·p gap. 25 steps too short to cross the boundary (r1 forks needed 400; r2 used 100). The interventional signal at proper scale *is* E3-unbiased. |

**One-line synthesis:** identical RL that from the plain fork collapses ~39% of base reasoning modes
(driving 2 to extinction) preserves them (≈2%) from the coverage-floor fork; the `K·p≈1` support-blind
boundary explains exactly which modes die; and the preserved mass converts into a higher continued-RL
large-k ceiling.

### Tooling built (reusable, committed)
`recoverability.py` (R_K, effective support) · `strategy_bank.py` (mode clustering + GPU sampler) ·
`recoverability_certificate.py` (Jensen LCB certificate + endangered-mode detection) ·
`score_bank_logprobs.py` (teacher-forced scorer) · `phase_transition.py` (E8) ·
`go_e8_arm.sh` (full-FT plain/floor arm) · `bootstrap_fast.sh` (hardened node setup).

---

## 5. Experiments to perform (prioritized; ✅ done / ⏳ todo)

**Foundations (mostly done)**
- ✅ E0 Freeze pilot; relabel single-trace suppression as *witness-likelihood contraction*.
- ✅ E1 Operationalize recoverability `R_K` + effective support.
- ✅ E2 Multi-witness strategy bank (pipeline + 11,664-witness held-out Olympiad bank).
- ✅ E3 Mode-mass certificate — real **and** held-out/unbiased.
- ✅ E8 Observational `K·p≈1` phase-transition figure.

**Next up (todo, no rush)**
1. ⏳ **E8-interventional (long horizon).** Both arms from base at **150–200 steps**, checkpoints
   every 10; score the trajectory → watch fragile modes cross `K·p=1` over training (plain crosses,
   floor holds). The clean standalone causal figure. (~3–4h GPU; 25-step run was underpowered.)
2. ⏳ **E2-final labeling.** Replace the heuristic strategy proxy with an **LLM strategy judge**
   (anthropic, on-cluster); report cluster stability + human/LLM agreement.
3. ⏳ **E4 Adaptive primal-dual** — dual ascent on `λ_{qm}`; only endangered modes get pressure.
4. ⏳ **E5 Route-level preservation** — protect strategy prefix `p_θ(z|q)`; compare exact-trace vs
   cluster vs route on the pass@1/coverage Pareto frontier (hypothesis: route wins).
5. ⏳ **E6 Baselines reviewers will demand** — GRPO, global KL, **PBA**, **DPH-RL**, **UCPO**, **BBG**,
   PKPO, DyJR/replay. Prioritize PBA/DPH-RL/UCPO/BBG (closest to our claims). Vanilla-GRPO-only won't survive review.
6. ⏳ **E7 Statistical strength** — ≥3 independent *training* seeds for the main comparison; seed-level
   uncertainty + paired per-problem CIs. Stop calling sub-1-pt effects "stable."
7. ⏳ **Headline off the saturated band** — rerun on **Omni-MATH / harder boundary subsets** where GRPO
   destroys *multiple points* of reachable support (PBA shows multi-pt Omni-MATH gains → that's the bar).
8. ⏳ **Scaling-law test** — grid {7B, 14B, 32B} × {Olympiad, Omni-MATH, AIME} at k=1024; report the
   `expPROJ − GRPO` coverage gain vs scale & benchmark hardness (theory predicts the effect grows with
   an unsaturated fragile band).
9. ⏳ **Pareto headline figure** — method strictly shifts `pass@1 vs recoverable-mode-count vs
   future-RL-improvement` frontier outward.

**Standing infra notes:** clusters die ~1–2h in (ephemeral-storage eviction fixed by running on nvme;
pip fixed by `PIP_CONFIG_FILE=/dev/null` bypass of the base image's dead nvidia extra-index; xet-fast
downloads for sharded models; dual DNS for c10d; nvtx≥0.2.11 for deepspeed). Always push
checkpoints/banks off-node (HF for 15GB ckpts; incremental laptop pull for banks) — a bank was lost
once to a mid-run death.

---

## 10. Two-phase plan to a NeurIPS best-paper-level submission

**Principle:** lock the technique on ONE cheap, decisive setting first (Phase A); only then spend
compute on the dense matrix (Phase B). Do not scale a method that isn't yet frozen.

### Phase A — "the technique is established" GATE (finish before scaling)
Minimal bar, on Qwen2.5-Math-7B / Olympiad-fragile (our current setting):
- [x] Mechanism, three independent angles: E3 held-out certificate (39% vs 2.3% collapse), E8
      observational (K·p≈1 predicts extinctions), E8 interventional-150 (causal, +3.34 nats, 0% vs 4%).
- [x] Baselines beaten under identical setup: plain, global-KL, PBA/base-anchor, **UCPO** all done.
      Final 5-way ordering (mode-mass Δ): UCPO −3.44 < plain −1.96 < global-KL −0.73 < PBA −0.28 <
      **floor +1.38**. UCPO (reward-shaping) is the WORST — sharpest confirmation of O(p_θ) blindness.
- [~] **Final method frozen** (method-freeze head-to-head on Olympiad-fragile, 150 steps, mode-mass Δ):
      - expSR (soft one-sided floor) = the floor arm, **DONE (+1.38)** — the incumbent to beat.
      - **expPROJ (hard projected-gradient): DROPPED at 7B.** Architecturally incompatible with the
        required full-FT DeepSpeed **ZeRO-3** setup: `ProjectionGRPOTrainer` builds a second (side) SGD
        optimizer over the model params, which corrupts ZeRO-3's parameter-partition hooks → crash on
        the first normal step (deepspeed `_partition_param`). Would only run at ZeRO-2 / small scale.
      - **E4 primal-dual (global dual ascent on μ): RUNNING** on node mi-0cc49a5556549b330 (2026-08-22).
      - **E5 route-level: RUNNING (queued after E4)** — implemented cleanly as expSR on a *route bank*
        (each witness truncated to its 64-token strategy prefix; ref = base logp over the prefix), so
        the floor protects entry into the reasoning basin p_θ(z|q), not the trajectory. No trainer
        change (route-ness lives in the bank). Scored over both the full bank (comparable) and the
        route bank (native). Tooling: `rl_training/build_route_bank.py`.
      Winner (best pass@1/coverage Pareto among expSR / E4 / E5) becomes THE method.
- [ ] **≥3 seeds** on the frozen method vs best baseline (E7) — turn "directional" into "significant".
- [ ] **One unsaturated headline**: reproduce the pass@1↑ + coverage↑ on a harder/wider band
      (Omni-MATH boundary subset) where the gain is multiple points, not 0.7%.
Exit criterion: frozen method + a significant multi-point win on ≥1 hard benchmark, 3 seeds, all
baselines. That is the paper's spine.

### Phase B — dense best-paper campaign (only after the Phase-A gate)
The matrix, each cell = pass@1 / large-k coverage / recoverable-mode-count / certified-mode-% /
continued-RL ceiling, ≥3 seeds, paired CIs:
- **Models (scale axis):** Qwen2.5-Math-{1.5B, 7B}, Qwen2.5-Math-**14B**, and a 2nd family
  (Llama-3.1-8B or DeepSeek-Math) for generality — the theory predicts the effect grows with an
  unsaturated fragile band, so 14B/32B should widen it (scaling-law figure).
- **Benchmarks (hardness axis):** MATH-500, OlympiadBench, Omni-MATH-Rule, AIME — span
  saturated→resonant→too-hard to trace the difficulty resonance.
- **Method ablations:** α (floor slack) sweep; μ / primal-dual vs fixed; expSR vs expPROJ vs route-level;
  bank size (witnesses/mode) & witness-selection; on- vs off-policy; one-sided vs symmetric (vs PBA).
- **Baselines (full):** GRPO, global-KL, PBA, DPH-RL, UCPO, BBG, PKPO, RiskPO — all at matched budget.
- **Mechanism at scale:** the K·p phase-transition figure with a wide-p bank + checkpoint trajectory
  (raise save_total_limit) so the transition gradient shows, across model sizes.
- **Continued-RL headline:** the fork→round-2 ceiling (R4) repeated across models/benchmarks/seeds.
Est: a few hundred GPU-hours; only justified once Phase A is frozen. Infra is ready (bootstrap_fast,
nvme, HF-checkpoint death-proofing, sharded eval, recoverability/certificate/phase-transition tooling).
