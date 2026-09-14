# PROGRESS SNAPSHOT v14 — verified self-improvement: characterization + measurement (headline defended at pass@1)
_2026-09-14. Single source of truth. Full detail: ADAPTIVE_FORGETTING_RESULTS.md §120–§151. Method code: rl_training/{vsf_trainer,cct_*,comp_eval}.py. Recipe: rl_training/queue/FULL_BOOTSTRAP.sh._

## 0. STATUS RIGHT NOW
**Measurement reanalysis complete + headline strengthened (§150–§151).** After the external review, re-evaluated at single-attempt **pass@1** with a turnover-robust estimator: the RFT≫GRPO advantage **survives and sharpens** as a genuine reliability/acquisition gain, and the "regression/forgetting" story was largely a coverage/turnover artifact. Method search remains **exhausted** — no method beats RFT (CCT §144–146, VRT §149 all null). Deliverable: **characterization + measurement** paper, now defended against the review. Infra: 2101/2103 pods dead (48 GPUs); 2102 = 24 GPUs (MAIN running k=32 confirm; workers re-bootstrapping for deepseek reliability replication).

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
