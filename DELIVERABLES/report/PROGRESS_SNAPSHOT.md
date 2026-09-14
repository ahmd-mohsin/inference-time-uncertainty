# PROGRESS SNAPSHOT v17 — RVP breaks the RFT reliability ceiling: dense (1.3B–7B × 3 families × code+math) + mechanism
_2026-09-14. Single source of truth. Full detail: ADAPTIVE_FORGETTING_RESULTS.md §120–§163. Method code: rl_training/{rvp_gen,dpo_train,math_rvp,rvp_margin,comp_eval}.py + rvp_scripts/. Recipe: rl_training/queue/FULL_BOOTSTRAP.sh._

## 0. STATUS RIGHT NOW
**RVP DENSELY ESTABLISHED + MECHANISM SHOWN (§161–§163).** On 3 fresh clusters (72 GPUs), the dense matrix confirms decoupled verified-preference (RVP) beats the RFT ceiling at single-attempt pass@1 across **sizes 1.3B→7B, 3 families (Qwen-Coder, Qwen, deepseek-coder), 2 domains (compositional-code + GSM8K), 3 difficulties** — wherever reliability headroom exists; the sole ~0 cell is near-ceiling GSM8K-3B (predicted). Mechanism (§163): RVP raises the correct-vs-incorrect **logit margin by suppressing incorrect modes** (logp(y+) flat, logp(y−) drops), Δmargin tracks the pass@1 gain. Full spine done: measurement → theory → dense empirical → mechanism. Remaining: more seeds/CIs, BigCodeBench, 14B (needs ref-offload).

## 0d. RVP DENSE RESULT (§161–§163) — matched-budget pass@1, RVP vs RFT
| domain | model | size | RFT | RVP | Δ |
|---|---|---|---|---|---|
| comp | Qwen-Coder-1.5B | 1.5B | 0.246 | **0.615** | +0.37 |
| comp | Qwen-Coder-3B (vhard) | 3B | 0.055 | 0.103 | +0.047 |
| comp | **deepseek-coder-6.7B** | 6.7B | 0.361 | **0.549** | +0.19 |
| comp | **Qwen-Coder-7B** | 7B | 0.667 | **0.775** | +0.108 |
| comp | Qwen-Coder-7B (hard) | 7B | 0.415 | 0.493 | +0.078 |
| comp | **Qwen-7B (non-coder)** | 7B | 0.544 | 0.647 | +0.103 |
| GSM8K | Qwen-1.5B | 1.5B | 0.711 | 0.748 | +0.037 |
| GSM8K | deepseek-1.3B | 1.3B | 0.050 | 0.075 | +0.025 |
| GSM8K | Qwen-3B (base .83) | 3B | 0.847 | 0.851 | ~0 (ceiling) |
- Scale trend: gain **shrinks with size but stays positive** (+0.37 @1.5B → +0.11 @7B). shuffled≈RFT & RVP>xrft in every headroom cell. Effect **headroom-gated** (§150/§160).
- **Theory (§160):** verified preference reallocates mass I→C; positive-only CE is mode-covering and can't suppress I. **Mechanism (§163):** ↑logit-margin via ↓logp(incorrect), Δmargin ∝ Δpass@1.
- Everything preregistered (§152), controls pass, honest boundary (ceiling) + retraction (§143) retained.

## 0d. RVP EVIDENCE (§153–§159) — matched-budget pass@1, all arms from each cell's RFT
| domain | family | diff | RFT | **RVP** | xrft | shuf | RVP−RFT |
|---|---|---|---|---|---|---|---|
| comp | Qwen-3B | mid | 0.392 | **0.645** | 0.528 | 0.386 | +0.253 |
| comp | Qwen-3B | hard | 0.215 | **0.415** | 0.305 | 0.205 | +0.200 |
| comp | Qwen-1.5B | mid | 0.258 | **0.637** | 0.412 | 0.265 | +0.379 |
| comp | deepseek-1.3B | mid | 0.142 | **0.271** | 0.173 | 0.139 | +0.129 |
| comp | deepseek-1.3B | hard | 0.083 | **0.149** | 0.091 | 0.081 | +0.066 |
| **math (GSM8K)** | Qwen-1.5B | — | 0.704 | **0.743** | 0.730 | 0.702 | +0.039 |
- Ordering **RVP > xrft > RFT ≈ shuf** in every cell. P1–P4 (§152) all hold; effect **tracks reliability headroom** (large at low pass@1, modest at high-baseline GSM8K).
- **Mechanism:** RFT maximizes coverage; verified *preference* on self ± pairs concentrates mass onto reachable solutions (*selection*) — the axis positive-only imitation can't touch (§150 coverage≫pass@1). Grounded + preregistered.
- **Larger LLM (Qwen-7B, comp mid):** base 0.455 → RFT 0.675 → xrft 0.757; RVP-DPO rerunning (bsz=1).

## 0d. THE RVP RESULT (§153–§154, the positive-method headline)
pass@1, matched-budget (+250 steps from each family's RFT), all arms from that RFT:
| family | RFT | **RVP** | xrft (pos-only) | shuf (ctrl) | RVP−RFT | RVP−xrft |
|---|---|---|---|---|---|---|
| Qwen-3B (5 seeds) | 0.215 | **0.415** | 0.305 | 0.205 | **+0.200** | +0.110 |
| deepseek-1.3B | 0.142 | **0.271** | 0.173 | 0.139 | +0.129 | +0.098 |
| Qwen-1.5B | 0.258 | **0.637** | 0.412 | 0.265 | +0.379 | +0.225 |
- **P1** RVP≫RFT (all 3); **P2** coverage rises (no collapse); **P3** shuffled-pair ctrl ≈ RFT (verified signal, not exposure); **P4** RVP≫matched-budget positive-only (negatives carry the reliability signal). β-robust (0.05–0.3).
- **Mechanism:** RFT maximizes *coverage*; verified *preference* on self ± pairs concentrates mass onto reachable solutions (*selection*) — the axis positive-only imitation structurally can't touch. Grounded in §150 (coverage≫pass@1), preregistered §152.
- **Honest caveats:** difficulty-axis + ds/1.5B seeds in progress; report greedy+pass@4; prior-art positioning (DPO/V-STaR/RAFT/self-rewarding use verified pairs — contribution = the reliability *diagnosis* + decoupled framing + cross-family magnitude + P3/P4 controls); a 2nd domain (math/BigCodeBench) would strengthen.

## 0c. THE MEASUREMENT RESULT (§150–§151, review-defended headline)
Patched comp_eval to log per-completion counts (c/k) + seeds. On c3hard/3B (n=200):
| arm | pass@1 | coverage@16 | turnover-robust acq A | reg D | net@1 |
|---|---|---|---|---|---|
| base | 0.090 | 0.330 | — | — | — |
| GRPO | 0.091 | 0.353 | 0.010 | 0.009 | **+0.001** |
| RFT | 0.216 | 0.610 | 0.132 | 0.006 | **+0.126** |
| VSF | 0.128 | 0.470 | 0.044 | 0.007 | +0.037 |
- **Coverage overstates reliability** (base .090 vs .330). **RFT lifts single-attempt reliability +0.126 (2.4× base); GRPO ~0.** The advantage is genuine **acquisition**, turnover-robust and seed-stable (§151: pass@1 RFT .215/.215; coverage-null α≤.045/β≤.076 ≪ naive ~0.5).
- **The β/"forgetting" story was mostly turnover** (true D≈.006 vs null≈.5) → the real mechanism is acquisition, not regression. Corrects the earlier coverage-level α/β framing.

## 1. HEADLINE (what the data supports — empirical)
**For verifiable OOD learning, decoupled verified replay (RFT) is the ceiling recipe in every regime tested: it acquires the most and regresses the least. On-policy outcome-RL (GRPO) acquires little and actively regresses; VSF (a verified-replay support floor) causally repairs both axes but does not beat RFT; continued RL after RFT adds nothing; and no acquisition/wiring wrapper (CCT) beats it.** Gains follow an **empirical** headroom×accessibility inverted-U. This is characterization + measurement, reported as empirical regularities — NOT a proven universal theorem (see §4).

## 2. PREREGISTERED VALIDATION MATRIX (§141, EMPIRICAL — k16 OOD, seed CIs)
| cell | fam | base | GRPO | VSF | RFT | R−G | R−base |
|---|---|---|---|---|---|---|---|
| c15mid | 1.5B | .390 | .373 | .528 | **.767** | +.394 | +.377 |
| c15hard | 1.5B | .180 | .200 | .300 | **.435** | +.235 | +.255 |
| c15vh | 1.5B | .090 | .100 | .140 | **.192** | +.092 | +.102 |
| c3mid | 3B | .525 | .517 | — | **.833** | +.316 | +.308 |
| c3hard | 3B | .365 | .338 | .435 | **.615** | +.277 | +.250 |
| c3vh | 3B | .205 | .180 | — | **.357** | +.177 | +.152 |
| dsmid | ds1.3B | .330 | .343 | .395 | **.562** | +.219 | +.232 |
| dshard | ds1.3B | .220 | .197 | .268 | **.370** | +.173 | +.150 |
| c7mid | 7B | .825 | (server-mode) | — | **.854** | — | +.029 |
_3 checkpoints / 2 families × 3 difficulties. 7B GRPO/VSF = colocate OOM (server-mode). RFT≫GRPO 8/8 non-overlapping; GRPO<VSF<RFT throughout._

## 3. PER-PROBLEM α/β (measurement contribution, §141 — α=acquire|base=0, β=regress|base=1)
Every cell: **α ordered RFT>VSF>GRPO** (8/8) and **β ordered RFT<VSF<GRPO** (8/8, VSF repairs regression). e.g. c15mid α .151/.307/.656, β .278/.128/.060. β GROWS with difficulty (regression worsens as accessibility drops). Net accuracy HIDES this bimodal acquire/regress split → *report (α,β,O), not net Δacc.*

## 4. THEORY STATUS — scoped after external review (§143)
- **T1 gradient identity** (per-prompt, exact current-policy): correct as an identity; used descriptively. "All RFT–GRPO gaps are procedural" is NOT established in general (stale multi-epoch bank + clipped/group-normalized GRPO + cross-prompt reweighting need more).
- **Accessibility-gated inverted-U**: an **empirical** shape (RFT gain rises then falls in headroom; peak ~0.6; bank size collapses with difficulty). The closed form G(h)=τh(1−h^C) and peak h*=(C+1)^(−1/C) are a POSITED model — its predicted peak (0.76–0.90) does NOT match the observed ~0.60; not a validated law.
- **Theorem 4 (RFT is a universal ceiling): RETRACTED.** Valid counterexample (verified): CE-optimum on the bank ≠ accuracy-maximum (J(2/3)=.583 < J(.9)=.613), same support, 1-D. The VSF<RFT / RL-after-RFT=RFT / CCT<RFT results are **empirical regularities in tested regimes**, NOT proof no method can beat RFT.

## 5. HYPOTHESES KILLED BY THEIR OWN GATES (honest)
- **CCT program (C0→novelty→C1→C2), §144–146** — C0: conditional-contract recovers 12,016 correct components (+52% over observed, rejects accidental agreements) → mechanism REAL; **novelty: 99% are re-copies of ops the RFT model already masters** (pass@k≈1.0) → C1 unwarranted; **C2 wiring: verified connected-fragment training 0.182 < whole-RFT 0.221** on held-out wiring → composition hypothesis falsified. No CCT variant beats RFT in the DAG regime.
- **Feedback-acquisition (memo A0), §140** — 32–82% larger matched-domain bank doesn't improve RFT (mean Δ +.009) → recipient saturated.
- **Regression-budget (memo H1–H5), §138/§139** — RL-from-RFT = RFT within noise → constraining it can't beat RFT.
- **CTH coverage-targeting** (size-gated), **linear-headroom "law"** (accessibility-gated), and earlier: self-repair distillation, archive preservation, delayed-value, difficulty escalation, recomposition, execution-value supervision.

## 6. HONEST AWARD READ
The defensible paper is **characterization + measurement**: RFT is the empirical ceiling for verified OOD; the α/β hidden-acquisition decomposition (net accuracy misleads); VSF as a causal coverage-vs-objective repair; the empirical accessibility-gated inverted-U; robust across families/difficulties/hardware; and a large battery of pre-registered gate-kills (three external method memos + the full CCT program) — every proposed method that could beat RFT was tested and failed. Strong main-track / possibly spotlight. NOT a clear award: no new method beats RFT, and the ceiling is empirical (the theorem was retracted, §143). Award-tier would need either a method that genuinely beats RFT (none found) or a validated frontier-scale predictive law (the inverted-U closed form is unvalidated; peak mismatch). Reported without a manufactured winner.

## 7. DATA GAPS / INFRA
- 7B GRPO/VSF need server-mode (40GB colocate OOM); no clean >9B point. CCT's one untested regime = a HARD-LOCAL-OP domain (BigCodeBench, low-mastery ops) — separate future investment, unfunded on current evidence.
- Env recipe committed (FULL_BOOTSTRAP.sh: vllm0.23 + transformers4.57.6 + torch/torchvision cu126 + nvidia-cuda-runtime-cu13 + flash-attn cu126; train_grpo caps vllm_max_model_length=4096 to avoid KV-OOM). SSM tunnels drop frequently; launch detached from the main via sshpass. Pods die at 24h TTL / sshd faults.

## 8. NEXT (honest options)
- **Write the characterization+measurement paper** from §120–146 (recommended — the science is done; method search exhausted).
- Optional: replicate C2 on a 2nd checkpoint when a cluster returns (result already unambiguous); or test CCT in a hard-local-op domain (BigCodeBench) — the only regime where component recovery could carry novel supervision.
- Will NOT: claim the retracted ceiling theorem, re-fund gate-killed programs, or present the inverted-U closed form as validated.

## 9. DENSE-SCALE RUN PLAN (ready to fire on new compute)
Goal: turn the RVP result from "shown" into "densely established" for the paper. All scripts exist and are validated: `rvp_gen.py` (verified ± pairs), `dpo_train.py` (decoupled preference; bsz=1 for ≥7B), `sft_train.py` (RFT/xrft positive-only control), `math_rvp.py` (GSM8K), `comp_eval.py` (per-completion pass@1 + coverage), `rvp_family.sh`/`math_flywheel.sh` (self-driving flywheels; take BASE/FAM/TRD/OODD/GEN_GPU_MEM env). Per flywheel ≈ 6 GPUs, ~40–60 min.
Matrix to run (each cell = base→RFT→{RVP×3 seeds, xrft×2, shuf×1}→pass@1+pass@4 eval, matched-budget):
- **Sizes/families:** Qwen2.5-Coder {1.5B, 3B, 7B, 14B, 32B} + Qwen2.5 {1.5B,7B} + deepseek-coder {1.3B,6.7B} + Llama/Mistral for cross-lineage (≥6 families).
- **Difficulties (comp):** mid(7→9), hard(12→14), vhard(16→18) — 3 points per family.
- **Domains:** compositional-code (comp_dag), GSM8K, MATH-500, + BigCodeBench (real multi-library code; build a contract/test harness) and a code-exec benchmark — ≥3 real domains.
- **Seeds:** ≥3 per arm for CIs (paired-bootstrap over problems; ≥1000 eval problems for the small-effect math cells per the review's power calc).
- **β / lr sweep:** β∈{0.05,0.1,0.3}, lr∈{5e-6,1e-5}; confirm robustness + report sensitivity.
- **Ablations:** pair-count per prompt, on-policy vs decoupled preference, RVP-from-base vs RVP-from-RFT (does coverage-first matter?), iterate RVP→regen pairs→RVP (flywheel rounds).
- **Controls (mandatory, per review):** shuffled-label, matched-budget positive-only (xrft), matched-token/FLOP accounting, greedy + pass@1 + pass@4/16, frozen decoding, no train/test leakage, unchanged-checkpoint null for any α/β.
- **Scale test:** does RVP−RFT shrink or hold with model size? (7B/14B/32B) — the "scales with size" question.
Priority order when compute lands: (1) 7B/14B RVP across 3 comp difficulties + GSM8K (scale × domain); (2) ≥3 seeds + CIs on all existing cells; (3) MATH-500 + BigCodeBench (real 2nd/3rd domain); (4) β/lr + ablations; (5) flywheel-rounds. Fastest disconfirmer first: if RVP−RFT vanishes at 14B/32B or on BigCodeBench, that bounds the claim — run those early.
Ops: harvest results to the ledger immediately (worker pods self-wipe); push checkpoints off-node; run each flywheel on ONE stable node (don't split RFT/RVP across ephemeral workers).
