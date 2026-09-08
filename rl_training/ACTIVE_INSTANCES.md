## WORKING GRPO RECIPE on new clusters (2026-09-06) — hard-won, torch 2.6 container
Fresh pods need FULL stack; verl unusable (needs torch 2.7). Use trl+vllm:
1. bootstrap_fast.sh (vllm==0.23.0 trl==1.7.0 deepspeed) — installs the stack (~15min, slow).
2. trl 1.7.0 pulls transformers 4.57.6 which needs torch.float8_e8m0fnu (torch 2.7) + TransformGetItemToIndex.
   SHIM both in /tmp/instance_storage/gu/shim/sitecustomize.py (float8->float8_e4m3fn alias; TransformGetItemToIndex
   from torch._higher_order_ops.flex_attention). go_math.sh exports PYTHONPATH=/tmp/instance_storage/gu/shim (shim ONLY, no append).
3. UNINSTALL wandb (pip uninstall -y wandb) — TRL auto-reports to wandb -> forces login -> crashes GRPO. THIS was the blocker.
4. Keep transformers==4.57.6 (do NOT downgrade to 4.47 — vllm 0.23.0/trl 1.7.0 need is_trackio_available).
5. Launch via on-node reset_launch.sh (kills stale go_math ghosts by PID from a SCRIPT FILE to avoid pkill self-match;
   ensures no-wandb + tf 4.57.6; then go_math.sh). go_math.sh: set -o pipefail (NOT -u), vLLM-serve GPU0 + accelerate ZeRO-2 GPU1-7,
   train_grpo --reward-mode math --dataset gsm8k --no-novelty. LAUNCH SUCCEEDED on nA: 8/8 GPUs, VLLMUP.
Gotcha: early failed launches leave go_math health-loop ghosts (720s) that rewrite fixed log paths -> read STALE logs.
reset_launch uses unique timestamped log per run. Shared PID namespace: kill by PID, never pkill -f a pattern in your ssh cmdline.

# NEW CLUSTERS (2026-09-06) — RESET plan (veRL positive control + common eval)
Profile greenlandw; creds: ada credentials update --account 703671891219 --role Intern --provider isengard --once.
SSH aliases in ~/.ssh/config (nA/nB/nC mains via tunnel; nX1/nX2 workers via ProxyJump). Reconnect tunnel:
  pkill -f "localPortNumber.*<PORT>"; aws ssm start-session --target <SSM> --document-name AWS-StartPortForwardingSession \
    --parameters '{"portNumber":["2222"],"localPortNumber":["<PORT>"]}' --profile greenlandw --region us-west-2 &
NVME /tmp/instance_storage root-owned on fresh pods -> sudo chown -R greenland-user:greenland-users /tmp/instance_storage/gu (DONE all 9).
HF token deployed to mains. 72 GPUs (3 mains + 6 workers, 8 each).
| alias | port | SSM | main IP | workers |
|---|---|---|---|---|
| nA (nA1/nA2) | 1060 | mi-0604bc34ab7889fc8 | 10.2.134.148 | 10.2.144.34, 10.2.87.155 |
| nB (nB1/nB2) | 1061 | mi-08d6abad1b4b532ec | 10.2.249.136 | 10.2.155.219, 10.2.42.225 |
| nC (nC1/nC2) | 1062 | mi-090371afcc5a37670 | 10.2.227.43 | 10.2.105.66, 10.2.3.247 |
Jobs: nA = veRL TinyZero Qwen2.5-3B Countdown positive control (bootstrapping). nB/nC = common-eval + intervention (pending).
Old clusters (instH/instJ/instK) all DIED 2026-09-05.

# Active us-west-2 instances — profile greenlandw, account 144991380388

## APPLIED CODE EXPERIMENT (2026-09-01) — high-v domain downstream win test
Goal: show stratified(mode-diverse) vs iid pass@k + complementarity v in CODE (HumanEval/MBPP, unit-test
verified) — the regime ACV theory predicts diversity helps (math was v≈0, 3× downstream null).
Harness: code_passk.py (gen iid + 6 strategy-forced, exec-verify subprocess, regex strategy-label) +
go_code.sh (base: snapshot coder model, 8-GPU DP + merge). Deploy set: code_passk.py, go_code.sh (+repo
via bootstrap_fast.sh). fetch_big NOT needed (coder models sharded → base: snapshot).
- **INSTANCE 1** SSM mi-010ce5a51cda819ce t1040 · main 10.2.102.225, w 10.2.240.176, 10.2.183.118
  · Start 2026-09-01T23:04Z → dies ~2026-09-02T23:04Z. Jobs: main=Qwen2.5-Coder-7B-Instruct×humaneval(code_qc_he),
  w1=Qwen2.5-Coder×mbpp(code_qc_mbpp), w2=Qwen2.5-7B-Instruct×humaneval(code_qi_he).
- INSTANCE 2: pending — DeepSeek-Coder-6.7b + Qwen2.5-7B-Instruct × {humaneval,mbpp}.
- INSTANCE 3: pending — Llama-3.1-8B-Instruct + Qwen2.5-Coder × {humaneval,mbpp}.
Fresh-node gotchas: sudo chown /tmp/instance_storage/gu (root-owned); go_probe/go_code NOT on GitHub → cat-deploy.
Prior findings (math, done): FINDINGS_AND_NOVELTY.md, ROUTING_VS_COMPETENCE_RESULTS.md, [[rl-routing-vs-competence]].


## FREE-GENERATION ROUTING batch (2026-08-31) — conclusive behavioral rho via LLM-judge
Goal: measure rho(m|q) from FREE generations + LLM-judge classifier (NOT regex/logp) → routing
entropy H(rho); combine with existing competence c → conclusive Δrho-vs-Δc across 3 families.
Launcher: go_ff.sh <SPEC> <TAG> <DATASET> 150 64  (fetch base:/fork: + go_free.sh → free_route.py,
8-GPU DP + merge). New scripts NOT on GitHub → cat-deploy after bootstrap_fast.sh.
- **INSTANCE 1 (Qwen)** SSM mi-0a788a6fdebaa5e80 t1040 · main 10.2.164.24, w 10.2.96.2, 10.2.192.247
  · StartTime 2026-08-31T22:46Z → dies ~2026-09-01T22:46Z (24h). Jobs: free_qm_base (base:Qwen/Qwen2.5-Math-7B),
  free_qm_grpo (fork:muahmed7338/cov-r1-qm-omni-grpo-7b), free_qm_floor (fork:...-floor-7b) on olympiad_bench.
- INSTANCE 2 (Llama): pending — free_llama_{base,grpo,floor} on olympiad_bench.
- INSTANCE 3 (DeepSeek): pending — free_ds_{base,grpo,floor} on math500 (DS headroom).
Results already local (routing-vs-competence): DELIVERABLES/report/ROUTING_VS_COMPETENCE_RESULTS.md +
runs_pulled/probe_routing/. [[rl-routing-vs-competence]].


## ROUTING-vs-COMPETENCE PROBE (2026-08-31) — "Knowing vs Choosing" pivot
The decomposition experiment: is the 39% "mode collapse" ROUTING suppression (rho_m down, competence
c_m intact) or COMPETENCE erasure (c_m down)? Inference-only (no training). Launcher: go_probe.sh
<SPEC> <TAG> olympiad_bench 150 8 4  → strategy_probe.py, 8-GPU data-parallel + merge. Per problem:
DEFAULT (8 samples → rho + pass@1) + FORCED per-strategy (4 samples × 14 taxonomy strategies →
forced correctness c(m,q) + adherence). Output: $NV/eval_out/probe_<TAG>.json (summary has
default_mode_mass, routing_active_strategies, forced_adherence, forced_competence).
- **t1040 main (mi-031af6e95af9ee154):** probe_base  = base:Qwen/Qwen2.5-Math-7B
- **t1042 main (mi-096b13e7b3bc9ac38):** probe_grpo  = fork:muahmed7338/cov-r1-qm-omni-grpo-7b (COLLAPSED)
- **t1041 main (mi-03ef3d43949d503b2):** probe_floor = fork:muahmed7338/cov-r1-qm-omni-floor-7b (PRESERVED)
- **t1040 w1 10.2.202.71:** probe_dphf = fork:muahmed7338/cov-dphf-qm-oly-7b (DPH-F, closest competitor)
Headline test: if GRPO default_mode_mass << base BUT forced_competence ~= base → ROUTING suppression
(the field is solving the wrong problem; pursue Decoupled Repertoire Optimization). If forced_competence
also collapses → genuine erasure. Deploy gotcha: scp-hop to workers FAILS silently → use `ssh cat`-pipe
from main. Launch gotcha: aggressive pkill in shared-PID ns tears the ssh before the launch line runs →
kill only the specific GPU0 leftover pid (nvidia-smi compute-apps), then setsid nohup launch (no pkill).
Adaptation-shock campaign (below) is DONE (8/9 arms step 80; within-math optionality = clean negative):
see DELIVERABLES/report/ADAPTATION_OPTIONALITY_RESULTS.md.



## CSPO WAVE-2 (2026-08-29, after 2 cluster deaths) — 6 kills/certs, 2 clusters
NODE-1: SSM mi-0ea7cad18645042df t1040 · main 10.2.51.59, w 10.2.167.67, 10.2.122.6
  main=kill G8-s0, w1(167.67)=kill G4, w2(122.6)=base-cert(Oly N2048)
NODE-2: SSM mi-0c55feb238b1705df t1042 · main 10.2.64.183, w 10.2.195.5, 10.2.112.98
  main=kill G14, w1(195.5)=kill G8-s1, w2(112.98)=DPH-F pass@k (fetch_big cov-dphf + go_eval2 local)
Gp sweep = G4,G8×2,G14. Launchers: go_kill.sh STEPS SEED G (self-setup). Kill ckpts->HF cspo-kill-g{G}-s{SEED}-7b.
LESSON (prior deaths): 15GB ckpt push too slow to survive short-lived nodes; TODO build in-train p̂-trajectory
logger (tiny durable file). ada creds expire ~hourly (ada credentials update --account 703671891219 --role Intern --provider isengard --once).


## CSPO WAVE (2026-08-29) — 6 parallel exps across both clusters (capability-survival reframe)
Plan: DELIVERABLES/report/CSPO_plan.md. Core object: rl_training/capability_survival.py. Kill launcher:
go_kill.sh (self-setup: hostname/prewarm/base-dl; args STEPS SEED G; ckpt every 10, keep all; pushes to
cspo-kill-g{G}-s{SEED}-7b). New: train_grpo --dph-forward-kl, --save-steps/--save-total-limit.
- **C2 main 10.2.57.100 (t1042):** kill G=8 seed0 → e8_kill_s0 (primary Gp trajectory)
- **C2 w1 10.2.203.98:** kill **G=16** → e8_kill_g16_s0 (Gp boundary shifts up)
- **C2 w2 10.2.204.144:** kill **G=4** → e8_kill_g4_s0 (boundary shifts down)
- **C1 main 10.2.53.7 (t1040):** kill G=8 seed1 → e8_kill_g8_s1 (variance)
- **C1 w1 10.2.143.64:** base capability cert — go_eval2 hf:Qwen2.5-Math-7B N_SAMPLES=2048 on Oly fragile band → A_0 (eval_out/passk_base_cert.json)
- **C1 w2 10.2.162.141:** DPH-F pass@k — go_eval2 hf:cov-dphf-qm-oly-7b N_SAMPLES=1024 → the Pareto point (eval_out/passk_dphf_passk.json)
NEXT after this: certify_checkpoints.py (offline p̂ sweep over kill ckpts → Gp transition Fig1), CSPOTrainer
(per-q λ from CP-LCB + reservoir rescue) → kill→rescue (Fig2) + acquisition-survival Pareto (Fig3).
GOTCHAS baked into go_kill: fresh-node hostname→loopback (c10d), prewarm olympiad (offline train), hf-xet removed.
Wave-1 (obsolete floor evidence): ESTABLISH_wave1_results.md. DPH-F row: reframe_dphf/DPHF_findings.md.


## CLUSTER-2 (2026-08-28) — REFRAME wave-1 (capability survival). SSM `mi-0b09cbd375f1a39f1` · tunnel **1042**
main 10.2.57.100, workers 10.2.203.98, 10.2.204.144. All bootstrapped, hf-xet removed. Code deployed
(support_ratchet/coverage_trainer/train_grpo with DPH-F; go_ood/fetch_big/go_eval2/go_dphf).
- **Node A (main): S3 DPH-F baseline** — `go_dphf.sh 150 0.02` (forward-KL rehearsal = NLL on 1055-witness
  bank, --dph-forward-kl mode=forward_kl). Interventional 150-step full-FT from Qwen2.5-Math-7B. Score Δ vs
  same witnesses → drops into baseline table (closest competitor). Bank=ratchet_bank_e8.jsonl(1055), diff=difficulty_olympiad_7b.json.
- **Node B (w1 203.98): S2 cert-floor** — high-k (2048) recoverability sampling of cov-r1-qm-omni-floor-7b on
  omni_math_hard (N_PROBLEMS=200) → Clopper-Pearson Alive/Uncertain/Extinct. tag qmomni_cert_floor.
- **Node C (w2 204.144): S2 cert-grpo** — same for cov-r1-qm-omni-grpo-7b. tag qmomni_cert_grpo.
Cert re-analysis (laptop, N=1024) already done: NOTHING certified-extinct at 1024 samples; floor certifies
MORE alive (329-band: 325 vs 322; qm-omni: 1000 vs 994). High-k runs push toward certifying extinction.
Reframe plan: DELIVERABLES/report/REFRAME_capability_survival.md. FULL story: FULL_STORY.md.


## OOD/TRANSFER RETENTION campaign (2026-08-28) — the strong downstream angle
Design: take round-1 forks (plain vs floor, both on HF at ckpt-150 + top-level model.safetensors),
eval coverage + EXTINCTION (n_correct=0 count) on a DIFFERENT distribution than each was RL'd on.
Thesis: forgotten rare modes matter most OOD → floor retains capability, plain catastrophically forgets.
Metric: pass@k curve + coverage (frac n_correct>0) + extinction count. N_SAMPLES=256 (env), full OOD set.
Launcher: go_eval2.sh hf:<REPO|BASEID> <tag> <dataset> "" ; N_SAMPLES=256 env. NO diff-json (full set;
omni_math_hard self-filters difficulty>=5). Round-1 forks: cov-r1-{qm-omni,ds-omni,llama-oly}-{grpo,floor}-7b.
GOTCHA: hf-xet hangs on 15GB single-file safetensors (DISABLE_XET ignored) -> `pip uninstall -y hf-xet`
on every fresh node (done on cluster below). go_eval2 hf: uses allow_patterns (top-level model only).

### Cluster-1 = new N1 (2026-08-28): Qwen-Math (RL'd Omni) -> OlympiadBench 3-way OOD
SSM `mi-0b6ffaf67a7e53c92` · tunnel **1040** · main 10.2.53.7, workers 10.2.143.64, 10.2.162.141.
- main:        qmomni_ood_oly_floor  (hf:muahmed7338/cov-r1-qm-omni-floor-7b)
- w1 143.64:   qmomni_ood_oly_grpo   (hf:muahmed7338/cov-r1-qm-omni-grpo-7b)
- w2 162.141:  qmomni_ood_oly_base   (hf:Qwen/Qwen2.5-Math-7B)
All 3 bootstrapped (bootstrap_fast.sh -> BOOTSTRAP_FAST_DONE), hf-xet removed. Results -> eval_out/passk_<tag>.json.
Cluster-2 (pending 2nd instance): Llama (RL'd Oly) -> Omni-MATH 3-way + DeepSeek(Omni)->Oly.


Reconnect a tunnel:
  pkill -f "localPortNumber.*<PORT>"; sleep 2
  nohup aws ssm start-session --target <SSM> --document-name AWS-StartPortForwardingSession \
    --parameters '{"portNumber":["2222"],"localPortNumber":["<PORT>"]}' --profile greenlandw --region us-west-2 &
  ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p <PORT> greenland-user@localhost
Creds expired -> ada credentials update --account 703671891219 --role Intern --provider isengard --once
HF token (user muahmed7338): <HF_TOKEN — set locally in ~/.hf_token, NOT committed>  (deploy to node ~/.hf_token)

## N2 (2026-08-26) — 2nd continued-RL ceiling cell (after Instruct×Omni ceiling DONE)
SSM `mi-067b3dd70b065b165` · tunnel **1041** · main 10.2.39.91, workers 10.2.17.190, 10.2.112.51.
- **Instruct×Omni 3-way ceiling: DONE** → `runs_pulled/round2_eval/ino_ceiling/` + INO_CEILING_RESULT.md
  + tab:round2omni in the .tex (headline: floor stays above base at all k, plain drops below base, +1.2pt pass@1 tax).
- **NOW: Qwen2.5-Math-7B × Omni-MATH continued-RL** (2nd ceiling cell). Stage1 (N2 main):
  `rl_training/go_r1push_qm.sh` = go_cell round-1 (prepass→bank→plain+expSR forks + mode-mass Δ) THEN
  pushes both forks to PUBLIC HF `muahmed7338/cov-r1-qm-omni-{grpo,floor}-7b` (death-proof).
  Stage2 (next): round-2 GRPO from each fork → `cov-r2-qm-omni-{grpo,floor}-7b`. Stage3: pass@k eval.
  Cell dir: `/tmp/instance_storage/gu/cell_qwenmath7b_omni_math_hard/`.

## NEW N2 (2026-08-27) — continued-RL round-2 for Qwen-Math + DeepSeek
SSM `mi-0cd3663216f968a58` · tunnel **1041** · main 10.2.159.192, workers 10.2.86.194, 10.2.32.247.
Both cells' round-1 forks are on HF: cov-r1-{qm,ds}-omni-{grpo,floor}-7b. Round-2 running (all death-proofed):
- main: Qwen-Math×Omni r2-from-grpo → cov-r2-qm-omni-grpo-7b
- w1:   Qwen-Math×Omni r2-from-floor → cov-r2-qm-omni-floor-7b
- w2:   DeepSeek×Omni  r2-from-grpo → cov-r2-ds-omni-grpo-7b
Queue: ds-floor round-2 launches on the first freed N2 node.
Difficulties on nodes: qm=/tmp/instance_storage/gu/qm_omni_diff.json, ds=/tmp/instance_storage/gu/ds_omni_diff.json.
Launcher: rl_training/go_r2_hf.sh (fork from HF → 100 steps → hf daemon push).

## NEW N1 (2026-08-27) — continued-RL round-2 phase (after tunnel deaths)
SSM `mi-0aa98dc64a291004d` · tunnel **1040** · main 10.2.138.113, workers 10.2.129.50, 10.2.18.99.
ALL round-1 forks survived on HF (auto-pushed): cov-r1-{llama-oly,qm-omni,ds-omni}-{grpo,floor}-7b + cov-r1-llama-omni-grpo-7b.
So NO round-1 re-runs for qm/ds/llama-oly — jump to continued-RL round-2. Launcher: `rl_training/go_r2_hf.sh`
<grpo|floor> <R1REPO> <dataset> <diffpath> <R2REPO> <tag> <maxlen> (pulls fork from HF, identical GRPO 100 steps,
hf daemon death-proofs to R2REPO). Running now:
- main: Llama×Oly r2-from-grpo → cov-r2-llama-oly-grpo-7b
- w1:   Llama×Oly r2-from-floor → cov-r2-llama-oly-floor-7b
- w2:   Llama-3.2-3B×Omni diversity cell RE-RUN (was lost, no HF push)
Llama-Oly difficulty on nodes at /tmp/instance_storage/gu/llama_oly_diff.json (also local runs_pulled).
Continued-RL cells done/ready: Instruct×Omni DONE (INO_CEILING_RESULT.md); Llama-Oly running; qm-omni + ds-omni
forks on HF → round-2 ready to launch on the next instance.

## PRIOR — N1 (2026-08-26), eval phase
| field | value |
|---|---|
| SSM | `mi-079380dd186f0ec00` |
| tunnel port | **1040** |
| region / account | us-west-2 / 144991380388 (greenlandw) |
| main IP | 10.2.51.236 |
| worker IPs | 10.2.58.38, 10.2.243.129 |

**N1 now runs 3 jobs (2026-08-26, after difficulty prepass finished):**
- **main (10.2.51.236):** internals analysis `rl_training/go_internals.sh` — samples Instruct base-correct
  traces on the Omni fragile band, teacher-forces through base + r2-grpo ckpt-{10,30,60,100} via
  `internals_analysis.py` → `$NV/internals/int_*.npz` (per-token logp / attn-entropy / hidden-state for
  the 4-panel "why it forgets" figure: logp-collapse map, attention narrowing, forgetting-over-training,
  representation drift). Pull int_*.npz → laptop, plot 4 figures.
- **w1 (10.2.58.38):** Llama-3.1-8B-Instruct × Omni-MATH fragile-band cell (go_cell.sh, maxlen 2048).
- **w2 (10.2.243.129):** Llama-3.1-8B-Instruct × OlympiadBench cell. Both extend the diversity table to
  the Llama family (wider fragile band expected) + become round-1 forks for Llama continued-RL.
  NOTE: 8B expSR arm may need the offload path (Qwen3-8B pattern); plain arm + fragile band run regardless.

## (N1 orig note) Instruct×Omni round-2 pass@k CEILING EVAL — DONE on N2 (see below)
- **Instruct×Omni round-2 pass@k CEILING EVAL** (the continued-RL payoff).
  - Round-2 TRAINING already DONE + durable on HF: `muahmed7338/cov-r2-ino-grpo-7b` and
    `cov-r2-ino-floor-7b`, both **checkpoint-100** (grpo=ctrl fork, floor=ours fork).
  - Step 1 (running now): `rl_training/go_prep_ino.sh` — downloaded Qwen2.5-7B-Instruct base,
    prewarmed Omni-MATH, 8-shard difficulty prepass → regenerate
    `/tmp/instance_storage/gu/cell_qwen25_7b_instruct_omni_math_hard/difficulty.json` (orig lost).
  - Step 2 (next): `rl_training/go_eval_ino.sh <base|grpo|floor>` — 8-shard pass@k, k=1024, Omni
    fragile band → `/tmp/instance_storage/gu/eval_out/passk_r2_ino_<fork>_frag.json`. Pull to laptop
    `runs_pulled/round2_eval/` then assemble the 3-way ceiling (base / r2-grpo / r2-floor).
  - Workers (10.2.58.38, 10.2.243.129): idle/unbootstrapped — can parallelize the 3 evals across nodes.

## PRIOR (DEAD) — diversity campaign instances (2026-08-25)
Both DIED. Diversity results (7 clean cells) are saved: DIVERSITY_findings.md + tab:diversity in the
.tex. Continued-RL forks survived on HF (above). Nothing to recover from these.
| inst | SSM | tunnel | node | cell (result) |
|---|---|---|---|---|
| I1 | mi-05eb80e2290593395 | 1030 | main/w1/w2 | Qwen3-8B×Oly (excluded), DeepSeek×Omni (+2.28), Instruct×Oly (+2.28) |
| I2 | mi-0a0a489493371c215 | 1031 | main/w1/w2 | Qwen3-8B×Omni (provisional), Instruct×Omni (+7.64), 1.5B×Oly (+0.13) |

Per-node: /tmp/instance_storage/gu/cell_<name>_<dataset>/ + logs/. Workers via ssh -p 2222 greenland-user@<workerIP>.

## ADAPTATION-SHOCK trio (2026-08-30) — cluster mi-031af6e95af9ee154 t1040
main 10.2.70.162, w1 10.2.202.71, w2 10.2.201.227. Optionality experiment: RL-adapt 3 sources on
OlympiadBench (shift; forks trained on Omni); per-step reward = adaptation curve.
- main: adapt_base_oly (base Qwen2.5-Math-7B, full repertoire)
- w1:   adapt_grpo_oly (fork cov-r1-qm-omni-grpo, COLLAPSED)
- w2:   adapt_floor_oly (fork cov-r1-qm-omni-floor, PRESERVED)
Launcher go_srcadapt.sh (fetch complete via fetch_big -> go_adapt server+ZeRO-3, 8 GPUs/node).
ALL fixes applied: fetch_big completeness (vLLM hangs on incomplete model — see [[rl-vllm-hang-incomplete-model]]),
train_grpo scp'd (GitHub push blocked -> nodes lacked --save-steps flag -> argparse crash), hostname self-fix.
Test: does PRESERVED adapt faster than COLLAPSED at matched pre-shift acc? Reward-vs-step in adapt_<tag>_train.log.

## ADAPTATION exp-2 (2026-08-30) — cluster mi-096b13e7b3bc9ac38 t1042 — MATH-500 shift
main 10.2.172.110, w1 10.2.240.81, w2 10.2.253.145. Same trio as exp-1 but shift=math500:
- main adapt_base_m500 (base), w1 adapt_grpo_m500 (collapsed), w2 adapt_floor_m500 (preserved).
exp-1 (t1040 mi-031af6e95af9ee154, olympiad shift) base TRAINING confirmed. Both = optionality on 2 shifts.

## ADAPTATION exp-3 (2026-08-30) — cluster mi-03ef3d43949d503b2 t1041 — AIME shift (hardest)
main 10.2.151.240, w1 10.2.32.26, w2 10.2.158.63. Trio on aime_all:
- main adapt_base_aime, w1 adapt_grpo_aime (collapsed), w2 adapt_floor_aime (preserved).
FULL CAMPAIGN: optionality across 3 shift difficulties — exp1 olympiad(t1040), exp2 math500(t1042), exp3 aime(t1041).
Readout: reward-vs-step per arm (adapt_<tag>_train.log). Predict preserved adapts faster than collapsed; gap grows with shift difficulty.

## NEW 72-GPU CLUSTERS (2026-09-07, KiroScienceInterns, p4d.24xlarge, acct 144991380388)
3 jobs × 3 nodes = 9 nodes / 72 GPU. LIVE cluster-3: SSM mi-0ea4a6d03d45a31f0 → tunnel port 1061 (main
10.2.134.136 + workers 10.2.66.242, 10.2.132.147). Clusters 1&2 (mains i-06f3bb57, i-0ed5f4e1) need SsmManagedInstanceId (not in JSON yet).
Env recipe: see memory rl-container-recipe-torch211-vllm023 (torch 2.11 + vllm 0.23 + rm flash_attn + nccl≥2.31 + shim). jsonlines needed for GRPO.
LAUNCH MECHANICS (critical): ssh-hop nohup gets SIGHUP'd → use rl_training/detach_run.sh (setsid). NEVER pkill -f a
pattern that appears literally in your own command (self-kill) → use BRACKET trick: pkill -9 -f "vllm[_]serve",
"train[_]grpo". GRPO port via PORT= env (avoid leftover-vllm clash). SSH: -o UserKnownHostsFile=/dev/null (pods reuse port, host key rotates).
Launchers: go_sft_m.sh <seed> <MODEL> <mtag> [GENGPUS] (BSZ env for 14B); go_math_m.sh <seed> <MODEL> <mtag> (8-GPU node).
RUNNING cluster-3: main=Qwen C {1.5B,3B,7B,14B}; w2=C {Phi-3.5,OLMo-2-7B,Yi-1.5-9B,Qwen2.5-Math-7B}; w1=GRPO-A q3b.

## FULL 72-GPU FLEET (2026-09-07) — all 3 clusters reachable
| cluster | SSM mi- | port | main | workers |
|---|---|---|---|---|
| C3 (live) | mi-0ea4a6d03d45a31f0 | 1061 | 10.2.134.136 | 10.2.66.242, 10.2.132.147 |
| C1 (boot) | mi-09711a00f46317000 | 1062 | 10.2.178.35 | 10.2.176.190, 10.2.248.195 |
| C2 (boot) | mi-0f55e42bcfcb1438a | 1063 | 10.2.247.251 | 10.2.210.65, 10.2.25.167 |
bootstrap_deps.sh now includes ALL fixes (sequential vllm/trl, rm flash_attn, nccl, jsonlines, nvtx, shim). 72 GPU total.
