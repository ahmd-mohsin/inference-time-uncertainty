# RVP — Reliability via Verified Preference

Research code, experiments, theory, and paper for **RVP (Reliability via Verified Preference)**: a
lightweight, headroom-gated post-training step that raises a language model's **single-attempt
reliability (pass@1)** on math reasoning **at essentially constant coverage (pass@k)** — by
*selecting better among what the model can already produce*, not by teaching it new skills.

- **Paper:** `Template_for_the_Research_Proposal_for_HLTH460_and_HLTH462_at_UC_SHSS/` (LaTeX; `paper.tex` is the root, `experiments.tex` holds all result tables).
- **Theory:** `DELIVERABLES/report/THEORY.md` (5 theorems/props) + `.../theory.tex`.
- **Canonical experiment ledger (every run, honest, incl. nulls & infra losses):** `DELIVERABLES/report/ADAPTIVE_FORGETTING_RESULTS.md` (see §168–§180).
- **Plain-English decode of every paper paragraph/theorem (every math symbol spelled out):** `EXPLAIN.md`.
- **Upstream:** `github.com/ahmd-mohsin/inference-time-uncertainty`.

> This repo grew out of a larger RL-forgetting line of work, so it contains legacy modules
> (routing, recoverability, CRO, code-repair, topological persistence). **The RVP paper touches only
> the files listed under [Core RVP code](#core-rvp-code) and [Orchestration](#orchestration-rvp_scripts).**
> Everything else is prior exploration, kept for provenance.

---

## 1. What RVP is (the one-paragraph version)

Given a math-pretrained base model, RVP runs **after** a rejection-sampling fine-tune (RFT):

```
BASE ──RFT──> RFT model ──RVP (decoupled DPO on self-labeled y+/y-)──> RVP model
      (imitate                (raise the logit margin
       verified-correct        m = logπ(y+) − logπ(y-) by
       self-samples)           suppressing incorrect modes)
```

- **RFT** = *Rejection-sampling Fine-Tuning*: sample k solutions from the base, keep only
  verifier-**correct** ones, SFT on them. Creates coverage headroom (mode-covering; no margin control).
- **RVP** = *Reliability via Verified Preference*: build preference pairs from the RFT model's own
  samples — a verified-**correct** `y+` and a verified-**incorrect** `y-` for the same problem — and
  run a **DPO** loss over them. This raises the margin `m = logπ(y+) − logπ(y-)`, mostly by
  **suppressing `logπ(y-)`**, so the correct answer wins at greedy/low-temperature decoding.
- **Why it works (Prop 2):** the per-problem success probability is `p_θ = σ(m + c)`, strictly
  increasing in the margin. Push the margin up → pass@1 up, while the *set* of solvable problems
  (coverage/pass@k) is unchanged. **Wins by selection, not by new capability.**
- **Scope (headroom-gated):** the gain scales with the *addressable RFT gap*. Big on
  math-specialized **base** models; **near-flat** once instruction-tuning has already consumed that
  gap; **destructive** on general RLHF-chat models (suppressing `logπ(y-)` pushes mass off the tuned
  distribution). This is a clean boundary, reported honestly — not hidden.

RVP's contribution is the **recipe and framing** (verified self-labels + decoupled preference,
run after RFT, headroom-gated), not the DPO loss itself.

---

## 2. Headline results (all pass@1, k=16 coverage held ~constant)

**Final-24 matrix — 3 models × 6 datasets × 4 seeds (§180, LoRA-1GPU, fresh fleet):**

| Model | Verdict | Representative Δ pass@1 |
|---|---|---|
| **Qwen2.5-Math-1.5B (base)** | RVP wins **all 6** datasets | GSM8K **+.206**, MATH-500 +.100, AMC +.089, DeepMath +.074, Olympiad +.039, Omni +.035 (per-seed spread ≤.006) |
| Qwen2.5-Math-1.5B-**Instruct** | flat (gap consumed by tuning) | ≈0 across sets; residual only on high-headroom Olympiad |
| Qwen2.5-Math-7B-**Instruct** | small positive, RVP top arm | Olympiad +.025, AMC +.016, Omni +.014, DeepMath +.014, MATH-500 +.010, GSM8K +.002 (ceiling) |

**7B base headline (full-param ZeRO-3 sharded DPO, in paper):** Qwen2.5-Math-7B `+.137 / +.058 / +.055` (MATH-500 / Olympiad / Omni).

**Scope discriminator:** RVP wins on math-specialized **base *and* instruct** (Qwen2.5-Math-7B-Instruct wins all 3), but **collapses** on general RLHF-chat (Yi-1.5-9B-Chat, −.125 to −.164). ⇒ the deciding axis is **math-specialized pretraining**, not instruct-tuning.

**Mechanism (logit-lens panels):** the margin-gain concentrates in **late layers** (26/28 for 1.5B, 21/28 for 7B) — RVP sharpens the *final decision*, not early features. Not longer chains-of-thought.

**Method characterization:** data-efficient (~25 verified pairs capture the full gain); **single-pass** (iterating RVP collapses it, −.118/−.154); **hard-negative mining = tie** with random verified negatives (simplest construction wins).

See `ADAPTIVE_FORGETTING_RESULTS.md` §180 for the full table + caveats (2 of 24 A-cells landed 2 seeds not 4 due to HF 429 rate-limiting).

---

## 3. Theory (`DELIVERABLES/report/THEORY.md`, `theory.tex`)

| # | Statement | What it buys the paper |
|---|---|---|
| **Thm 1** | Gradient-family identity: GRPO ≈ `p_θ(x)·∇logp` (coverage-throttled); RFT is the unscaled imitation gradient | Separates on-policy RL from imitation |
| **Cor 1.1** | Coverage throttling: GRPO's update shrinks as `p_θ`→0 or →1 | Explains why GRPO can't fix rare/hard modes |
| **Thm 2** | Support-boundedness: imitation can't place mass outside the sampled support | RFT is mode-covering only |
| **Prop 1** | RFT is mode-covering with **no margin control** | Motivates a *separate* margin step (RVP) |
| **Prop 2** | `p_θ = σ(m + c)`, strictly increasing in the margin `m = logπ(y+) − logπ(y-)` | The core RVP lever: raise margin ⇒ raise pass@1 |
| **Thm 3** | No-ceiling: CE-minimizing θ (2/3) ≠ accuracy-maximizing θ (0.9); `J(2/3)=.583 < J(.9)=.613` | Selection headroom exists beyond likelihood training |

---

## 4. The RVP pipeline, stage by stage

Single-GPU reference implementation: **`rl_training/rvp_scripts/math_cell_1gpu.sh`** (one cell = one
GPU; pack 8/node × 9 nodes = 72 concurrent cells). Env: `GPU BASE EVAL TAG [SEED NEVAL NBANK GEN_GPU_MEM HARDNEG]`.

| Stage | Command (module) | Output | Purpose |
|---|---|---|---|
| 1. **Bank** | `math_rvp --mode bank` | `bank.jsonl` | Sample k=8, keep verifier-correct self-samples |
| 2. **RFT** | `sft_train` → `model_utils.merge_adapter_if_needed` | `rft/merged_full` | LoRA SFT on the bank; create coverage headroom |
| 3. **Pairs** | `math_rvp --mode pairs` (`--hard-neg` optional) | `pairs.jsonl` | Verified `y+`/`y-` pairs from the RFT model |
| 4. **RVP** | `dpo_train` (β=0.1, 300 steps) → merge | `rvp/merged_full` | Decoupled DPO; raise the margin |
| 5. **Eval** | `math_rvp --mode eval` for `base/rft/rvp` | `ev_*.json` | pass@1 + coverage(pass@k), k=16 |

After **every stage** the cell runs `rvp_scripts/s3_sync.py <tag>` so a cycling pod loses nothing.
For **7B+**, single-GPU LoRA-DPO hits a TRL meta-tensor error → use **full-parameter ZeRO-3 sharded
DPO** instead (`dpo_train --full` via `accelerate` launch; see `launch_sharded7b.sh` + the
[sharded-DPO recipe](#7-fleet--infrastructure)). The cell auto-shrinks to bsz1/maxlen512 for ≥7B, but
sharded is the reliable 7B path.

**Mechanism panel:** `python -m rl_training.mech_interp --base <hf> --rvp <dir> --data pairs.jsonl --n 80 --out mech.json`
— per-layer logit-lens margin, decisiveness entropy, residual drift (base vs RVP).

---

## 5. Repository structure — where everything lives

```
inference-time-uncertainty/
├── README.md                    ← this file
├── EXPLAIN.md                   ← plain-English decode of every paper paragraph/theorem
├── LICENSE, environment.yml
│
├── Template_..._UC_SHSS/        ← THE PAPER (LaTeX)
│   ├── paper.tex                  root (\input the rest)
│   ├── motivation.tex methodology.tex theory.tex experiments.tex
│   └── main.tex howto.tex
│
├── DELIVERABLES/report/         ← all research write-ups
│   ├── ADAPTIVE_FORGETTING_RESULTS.md   ← CANONICAL LEDGER (§168–§180: every run, honest)
│   ├── THEORY.md                        ← 5 theorems/props, proofs
│   ├── AWARD_PAPER_SPINE.md, PAPER_DRAFT.md, FINDINGS_AND_NOVELTY.md, …
│   └── (legacy: CRO/routing/repair result docs from the earlier line)
│
├── rl_training/                 ← ALL training/eval/analysis code
│   ├── rvp_scripts/               ← RVP orchestration (see §6)
│   ├── queue/                     ← node bootstrap (FULL_BOOTSTRAP.sh, flashbuild_cu126.sh, …)
│   ├── math_rvp.py                ← RVP driver: modes bank | pairs | eval | margin (+ --hard-neg, --num-shards)
│   ├── sft_train.py               ← RFT (LoRA SFT on verified-correct self-samples)
│   ├── dpo_train.py               ← DPO (RVP); --full = full-param ZeRO-3 for 7B+
│   ├── mech_interp.py             ← layer-wise logit-lens mechanism panel (base vs RVP)
│   ├── model_utils.py             ← merge_adapter_if_needed (LoRA→merged_full for vLLM eval)
│   ├── safe_match.py, rewards.py  ← math verification / reward
│   ├── flywheel.py                ← multi-stage sharded orchestrator (bank→RFT→pairs→DPO→eval + S3)
│   └── (legacy: train_grpo, routing, recoverability, code_*, comp_*, taco_*, cct_*, …)
│
├── configs/  data/  scripts/  src/  tests/  docs/   ← support + legacy
├── checkpoints_pulled/ runs_pulled/ recovered/       ← harvested artifacts
└── external/ topological_persistence/ verification_gap/  ← legacy sub-lines
```

---

## 6. Orchestration (`rl_training/rvp_scripts/`)

The launchers dispatch cells across the 3-cluster / 72-GPU fleet, **detached on-node** (`setsid nohup`)
so the laptop/tunnel can drop, with **per-stage S3 sync**.

| Script | Role |
|---|---|
| **`math_cell_1gpu.sh`** | one full RVP cell on one GPU (bank→RFT→pairs→DPO→eval), per-stage S3 sync; auto bsz1/maxlen512 for ≥7B |
| **`launch_final24.sh`** | the 24h "important experiments" plan: 3 models × 6 datasets × 4 seeds = 72 cells; clones + `FULL_BOOTSTRAP`s fresh pods, runs payload detached (§180) |
| **`launch_pack.sh`** | generic 72-cell packer (8/node × 9), base64 payloads, stagger/mem knobs |
| **`launch_sharded7b.sh`** | reliable **7B** path: 1 cell/node = 8-GPU full-param ZeRO-3 sharded DPO |
| **`math_hard_shard.sh`** | data-parallel bank-gen (8 shards) + sharded pipeline, 9-point S3 sync |
| **`launch_math_matrix.sh`** | 9-cell base×dataset generalization matrix (sharded) |
| **`s3_sync.py`** | push a tag's artifacts to S3 (`SYNC_CKPT=1` also syncs checkpoints) |
| **`run_mech.sh`** | waiter: fires `mech_interp` once the RVP ckpt exists and the node is idle |
| **`data_eff.sh` `iter_rvp.sh` `seed_ci_node.sh` `beta_point.sh` `beta_sweep_node.sh`** | ablation helpers (data-efficiency, one-shot iteration, seed CIs, β-robustness) |
| **`boot_and_run.sh` `reset_and_run.sh` `salvage_eval.sh`** | node boot / reset / result-salvage utilities |

---

## 7. Fleet & infrastructure

- **Fleet:** 3 SDB `p4d.24xlarge` clusters = 9 nodes × 8×A100-40GB = **72 GPUs**, `us-west-2`, AWS profile `greenlandw`.
- **Access:** container sshd on port **2222** (host sshd on 22 is publickey-only). SSM port-forward tunnels map local `4210/4211/4212` → remote `2222`; workers reached via nested empty-password `sshpass -p '' ssh -p 2222`. `mi-*` IDs change on every pod restart (homes wiped); IPs stable within a job. Current fleet map lives in memory `rl-active-instances-uw2.md`.
- **Bootstrap:** fresh pods run `rl_training/queue/FULL_BOOTSTRAP.sh` (vLLM, flash-attn build ~15–50 min) + `nvtx/math_verify/deepspeed`.
- **Death-proofing:** pods cycle every few hours. Mitigations — per-stage S3 sync (bank/pairs early, ckpt before eval, per-arm results), `EVAL_CONC=1`, detached `setsid` runs.
- **S3:** `s3://greenland-intern-artifacts-703671891219-us-east-2-an/cmohsinm-rvp/<tag>/` (nodes write via IRSA Intern-role creds; the laptop `greenlandw` account gets a 403 cross-account on read — pull from a node/S3-role context).
- **Sharded-DPO recipe (7B+):** full-param ZeRO-3 CPU-offload via `python -m accelerate.commands.launch` (not LoRA — LoRA+ZeRO-3 crashes); `nvtx` upgrade + reset self-kill gotchas documented in memory `rl-sharded-dpo-recipe.md`.

---

## 8. Experiment log (ledger § map)

`ADAPTIVE_FORGETTING_RESULTS.md` records every wave honestly (wins, nulls, infra losses):

- **§170** 9-cell base×dataset generalization matrix (sharded, β=0.1/300).
- **§172–§174** instruct-base discriminator resolved: math-instruct wins (boundary = math-specialization).
- **§173** seed-CIs (in paper) + Mathstral cross-family collapse (honest limitation; requires a successful RFT gap).
- **§175→§176** hard-neg "HURTS" was an **unmatched artifact**; the **matched** result is a **tie** — corrected in the ledger.
- **§177** method characterization: data-efficiency, one-shot collapse, hard-neg=tie.
- **§178** breadth (AMC +.110, DeepMath +.077 on base) + `competition_math` loader dead (dropped, not guessed) + 7B LoRA-DPO meta-tensor blocker (→ route sharded).
- **§179** detached-run + S3 handoff protocol.
- **§180** **final-24 matrix complete** (3×6×4, 71/72 cells, 0 failures) + late-layer mech panels + the 7B `GEN_GPU_MEM 0.30→0.85` fix.

---

## 9. Reproduce

```bash
# One RVP cell on GPU 0 (base model, math500 eval, 4-seed run uses SEED=1..4):
export HOME=/home/greenland-user
cd ~/inference-time-uncertainty
GPU=0 BASE=Qwen/Qwen2.5-Math-1.5B EVAL=math500 NEVAL=200 SEED=1 \
  TAG=demo_m15_m500_s1 GEN_GPU_MEM=0.45 \
  bash rl_training/rvp_scripts/math_cell_1gpu.sh
# → ~/gu/demo_m15_m500_s1/RES.md  (base / rft / rvp pass@1 + coverage), S3-synced per stage.

# Full 72-cell final-24 matrix across the fleet (after tunnels 4210/4211/4212 are up):
bash rl_training/rvp_scripts/launch_final24.sh

# 7B via sharded full-param DPO:
bash rl_training/rvp_scripts/launch_sharded7b.sh

# Mechanism panel:
python -m rl_training.mech_interp --base <hf> --rvp ~/gu/<tag>/rvp/merged_full \
  --data ~/gu/<tag>/pairs.jsonl --n 80 --out ~/gu/<tag>/mech_<tag>.json
```

Datasets: `math500 · olympiad_bench · omni_math · gsm8k · amc · deepmath` (via `math_rvp` loaders).
Verification is exact-match + `math_verify` (`safe_match.py`), with a per-sample alarm to avoid sympy hangs.

---

## 10. Principles this repo is run under

- **Report honestly** — nulls are findings; no manufactured wins. Scope boundaries (instruct-flat,
  general-chat collapse, Mathstral) are stated, not hidden.
- **pass@1 is the primary metric** for the paper (single-attempt reliability), at held coverage.
- **Harvest immediately** — nodes self-wipe on restart; only S3 + git survive.
- **Never commit secrets**; use least-privilege AWS credentials; assume production when uncertain.
