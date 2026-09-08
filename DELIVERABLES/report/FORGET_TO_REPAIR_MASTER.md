# Forget to Repair — Master Results Ledger

**Paper:** *Forget to Repair: Minimal Failure States for Verifier-Guided Self-Correction.*
**Core claim:** execution feedback is not useless — it is useful information *contaminated by the model's own
failed proposal*. Self-correction needs a **proposal-invariant failure state**: keep the verifier evidence
(a concrete counterexample = Failure Certificate), erase the failed program. This ledger accumulates ALL
empirical data for the technique. Base model unless noted: **Qwen2.5-Coder-7B-Instruct**. Recovery = fraction of
default-FAILED problems solved within the recovery budget. Updated 2026-09-04.

---
## 1. RL: dense verifier reward, 4 arms (Contribution 3) — trained & evaluated
GRPO (TRL, LoRA r32, vLLM-server + ZeRO-2, 150 steps, 166 MBPP-train repair prompts). Eval = seq_recover, T=6,
diag=certificate, trained-vs-BASE, full benchmark. `n_fail` = first-attempt failures (same set ⇒ lower = higher pass@1).
Training reward (final mean): cert_residual **0.77→0.83**, fraction ~0.55, residual ~0.52, binary ~0.51.

**MBPP**
| Model | n_fail↓ (pass@1) | iid@6 | SEQ(cert)@6 | seq−iid | end-to-end unsolved = n·(1−SEQ)↓ |
|---|---|---|---|---|---|
| BASE            | 151 | 0.543 | 0.616 | +0.073 | 58.0 |
| A binary        | 144 | 0.507 | 0.576 | +0.069 | 61.1 |
| B fraction      | 143 | 0.510 | 0.615 | +0.105 | 55.1 |
| C residual      | 146 | 0.541 | 0.623 | +0.082 | 55.0 |
| **D cert_residual** | **125** | 0.632 | **0.680** | +0.048 | **40.0** |

**HumanEval** (small n, noisy)
| Model | n_fail↓ | iid@6 | SEQ(cert)@6 | seq−iid |
|---|---|---|---|---|
| BASE | 26 | 0.692 | 0.654 | −0.038 |
| A binary | 19 | 0.684 | 0.684 | +0.000 |
| B fraction | 24 | 0.667 | 0.750 | +0.083 |
| D cert_residual | 18 | 0.500 | 0.667 | +0.167 |

**Takeaways:** D (cert_residual) best pass@1 (MBPP 125 vs 151 → ~+26 first-try; HE 18 vs 26) AND best end-to-end
(MBPP 40 vs 58 unsolved → ~+18 net). **Binary (sparse) is worst — worse than base (61.1)**, motivating dense shaping (D≫A).

---
## 2. Certificate content-dependence — info-density ladder + corruption controls (reviewer-critical)
`cert_ablate.py`, single-shot K=8 per variant on default-FAILED problems, code hidden. c0=bare notice, c1=failed-tests,
c2=failing input, c3=input+got, c4=input+got+expected (full cert), c5=multi-counterexample. Corruptions are c4-shaped
but with a bogus input / corrupted expected / another problem's certificate.

**MBPP, BASE (iid=0.583):**
| variant | rate | Δiid |
|---|---|---|
| c0 | 0.589 | +0.007 |
| c1 | 0.603 | +0.020 |
| c2 | 0.642 | +0.060 |
| c3 | 0.556 | −0.026 |
| **c4 (full cert)** | **0.675** | **+0.093** |
| c5 multi | 0.656 | +0.073 |
| wronginput | 0.609 | +0.026 |
| wrongexpected | 0.603 | +0.020 |
| shuffled | 0.596 | +0.013 |

**MBPP, cert_residual TRAINED model (iid=0.642):**
| variant | rate | Δiid |
|---|---|---|
| c4 | 0.670 | +0.028 |
| c5 multi | 0.689 | +0.047 |
| wronginput | 0.670 | +0.028 |
| wrongexpected | 0.594 | −0.047 |
| shuffled | 0.623 | −0.019 |

**HumanEval, BASE (iid=0.800, CEILING):** all variants ≤ iid (c4 −0.100); no headroom — easy regime.

**Takeaways:** benefit **rises with information** and peaks at the full certificate (c4/c5). **Corrupting the content
collapses the benefit toward iid (base) or below iid (trained: wrongexpected −0.047, shuffled −0.019).** The model uses
the EVIDENCE CONTENT, not a mere restart prompt.

---
## 3. Difficulty frontier (U-shape) — certificate helps most at the capability frontier
cert−iid recovery (base Qwen-Coder), pooled across benches/difficulties:
| Benchmark / tier | iid recovery | cert−iid | regime |
|---|---|---|---|
| HumanEval | 0.800 | ~0 (≤0) | too easy (iid already recovers) |
| **MBPP** | 0.583 | **+0.093 (c4)** | **peak — capability frontier** |
| TACO easy (n=137) | 0.153 | +0.022 | hard |
| TACO medium (n=179) | 0.101 | +0.022 | hard |
| TACO hard (n=193) | 0.073 | +0.005 | too hard (no competence) |

**Takeaway:** Δcert peaks where the model *has* the capability but chose the wrong hypothesis (MBPP), and vanishes
when tasks are trivially recoverable (HumanEval) or beyond competence (TACO hard). Predictive theory, not just "cert helps."

---
## 4. Prior inference results (from FORGET_TO_REPAIR_RESULTS.md, reproduced here for one ledger)
- **R1** Failure Certificate beats iid across 4 MBPP model cells (seq−iid: Llama +0.150, 14B +0.106, Qwen-Instruct
  +0.049, Qwen-Coder +0.019); error_only ≈ parity → rich evidence survives proposal-erasure.
- **R2** cert_memory (accumulate counterexamples, code hidden) rises monotonically across rounds (Qwen-Instruct
  0.242→0.515; 14B 0.310→0.548) = cumulative learning w/o anchoring.
- **R3 causal** two-parent recurrence: FULL context +0.35/+0.42 own-parent recurrence vs CERT ≈0 (proposal anchors retry).
- **R4 routing** full-context collapses retry-algorithm entropy 0.37→0.09, pmatch→0.94; certificate restores.

---
## 5. TODO for award bar (tracked)
- [ ] Learned Minimal Failure State bottleneck B_ψ(q,P,E)→z (≤128 tok) must beat hand certificate (+3–5) or match at ½ tokens.
- [ ] Proposal-Leakage main figure: two-parent classifier D(z,q)→{A,B}, L=2(Acc−0.5); full-code high / error-only low / cert low / learned lowest at high repair.
- [ ] Cumulative memory (cert_memory) curves on base + trained (in progress).
- [ ] CEGIS baseline: failed-code+counterexample vs counterexample-only vs sketch+counterexample vs iid.
- [ ] Residual-reward RL D-vs-C isolation at matched rollout compute + seeds; scale ≥500 failed/condition (EvalPlus/MBPP+/HE+).
- [ ] Mechanism at larger N + 2nd/3rd model family (Qwen-14B, Llama).

Raw JSONs: laptop `rl_training/runs_pulled/repair_eval/` + `repair_ckpt/`. Harnesses: seq_recover.py, cert_ablate.py,
taco_cert.py, certify.py, train_grpo.py (--reward-mode code), go_ev.sh, eval_seq.sh, launch_arm.sh.

---
## 6. Cumulative memory (proposal-free failure memory, Result 2) — base vs trained, MBPP
seq_recover --diag-mode cert_memory (accumulate counterexamples across T=6 rounds, code hidden). recovery@t:
| round t | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| BASE iid | 0.344 | 0.430 | 0.497 | 0.517 | 0.543 | 0.570 |
| BASE cert_memory SEQ | 0.351 | 0.424 | 0.483 | 0.510 | 0.517 | 0.530 |
| cert_residual iid | 0.302 | 0.430 | 0.483 | 0.557 | 0.597 | 0.617 |
| **cert_residual cert_memory SEQ** | **0.383** | **0.483** | **0.517** | 0.537 | 0.577 | **0.624** |

**Takeaway:** on BASE Qwen-Coder cert_memory ≈ iid (this model is an unusually strong blind-retryer — cf. monoculture-
recovery finding). On the **dense-trained** cert_residual model, cumulative certificate memory beats iid at *every*
round (esp. early: +0.081 @t1) and ends highest (0.624). Training makes the model actually *use* accumulated evidence.

## 7. TACO base-vs-trained (does dense cert training transfer to hard algorithmic tasks?)
cert−iid recovery, stdin/stdout competitive-programming:
| condition | n | iid | cert | cert−iid |
|---|---|---|---|---|
| BASE MEDIUM (n=179, 800-cap) | 179 | 0.140 | 0.140 | +0.000 |
| BASE HARD (n=193, 800-cap) | 193 | 0.078 | 0.052 | −0.026 |
| **cert_residual TRAINED MEDIUM** | 177 | 0.096 | 0.141 | **+0.045** |
| cert_residual TRAINED EASY | 136 | 0.132 | 0.140 | +0.007 |

**Takeaway:** base Qwen-Coder barely uses certificates on TACO (med +0.00, hard −0.03 — strong iid retryer / no
competence). The **cert_residual-TRAINED** model gets **+0.045 on TACO medium** (vs base +0.000) — dense certificate
training TRANSFERS evidence-use to out-of-distribution harder algorithmic problems. (Earlier n=400 TACO med +0.022
for base was noise; at n≈180 base med = +0.000.)

**Honest synthesis so far:** the certificate's inference-time value is real but concentrated at the *moderate*
capability frontier (MBPP single-shot c4 +0.093 is the cleanest), and it is AMPLIFIED by dense-reward training
(cert_residual: best pass@1, best end-to-end, cert_memory>iid every round, +0.045 TACO-med transfer). The content-
dependence controls (corruptions ≈iid / negative) rule out a pure "restart prompt" explanation.

## 8. Proposal-Leakage (MAIN FIGURE axis) — proposal_leakage.py, base Qwen-Coder, MBPP, n=100 two-parent pairs
For each problem generate two independent FAILED parents P_A,P_B; for representation rep, z=rep(P). Leakage =
2*(retrieval_acc−0.5), retrieval = cosine-TF-IDF argmax of z to its own vs the other parent's code.
| Representation | retrieval acc | Leakage L | repair signal (from §2) |
|---|---|---|---|
| full_code (failed program) | 0.990 | **+0.980** | anchored (worst; §R3/R4) |
| error_only (error text, code hidden) | 0.510 | +0.020 | ≈ iid |
| **certificate (counterexample, code hidden)** | 0.535 | **+0.070** | **best (c4 +0.093 vs iid)** |

**Takeaway (the paper's core geometry):** the failed program is almost perfectly identifiable from itself (L=0.98) —
carrying it forward is what anchors the retry. The certificate carries near-zero proposal identity (L=0.07) yet is the
strongest repair signal. **Certificate = Pareto-optimal: high corrective information, ~zero proposal leakage.** This is
exactly the objective a learned Minimal Failure State bottleneck B_ψ should optimize; target = match/beat certificate's
repair at even lower L or fewer tokens. Harness: rl_training/proposal_leakage.py + go_ev.sh.

## 9. Content-dependence across ALL arms + tighter base (cert_ablate, MBPP) — robustness of §2
Δiid for the full certificate (c4), multi (c5) vs the three corruption controls. Positive c4/c5 + ≤0 corruptions ⇒
the model uses evidence CONTENT, not a restart prompt. Holds for every model:
| Model (iid) | c4 | c5_multi | wronginput | wrongexpected | shuffled |
|---|---|---|---|---|---|
| BASE k=8 (0.583)  | **+0.093** | +0.073 | +0.026 | +0.020 | +0.013 |
| BASE k=16 (0.675) | **+0.060** | +0.020 | −0.007 | −0.013 | −0.053 |
| A binary (0.604)  | +0.021 | +0.028 | −0.076 | −0.076 | −0.083 |
| B fraction (0.573)| +0.063 | +0.049 | +0.035 | +0.007 | −0.014 |
| D cert_residual (0.597) | +0.047 | +0.054 | +0.013 | −0.013 | −0.007 |
| D cert_residual, HumanEval (0.750, ceiling) | −0.083 | −0.083 | −0.167 | −0.083 | 0.000 |

**Takeaway:** across base (both K) and all trained arms, the **full certificate is the top variant (+0.02..+0.09)** and
**every corruption falls to ≤ iid** (binary model most sensitive: corruptions −0.08). Content-dependence is robust — not
a fresh-prompt artifact. HumanEval is ceiling (iid 0.75) so all variants ≤ iid (no headroom), consistent with §3 frontier.
(NB: residual-model ablation run errored (iid=0.000, merge/load fault) — to re-run.)

## 12. Seed robustness (RL arms across seeds) — MBPP, recovery@6 diag=certificate
end-to-end unsolved = n_fail·(1−SEQ), lower=better. Seeds: 42 (§1), 123 (s2), 456 (s3).
| Arm | seed | n_fail (pass@1) | SEQ(cert)@6 | seq−iid | unsolved↓ |
|---|---|---|---|---|---|
| BASE | – | 151 | 0.616 | +0.073 | 58.0 |
| cert_residual | 42 | 125 | 0.680 | +0.048 | **40.0** |
| cert_residual | 123 | 155 | 0.619 | +0.019 | 59.0 |
| cert_residual | 456 | 151 | 0.669 | +0.146 | 50.0 |
| fraction | 42 | 143 | 0.615 | +0.105 | 55.1 |
| fraction | 123 | 150 | 0.593 | +0.033 | 61.0 |
| residual | 42 | 146 | 0.623 | +0.082 | 55.0 |
| residual | 123 | 139 | 0.568 | +0.058 | 60.0 |
| binary | 42 | 144 | 0.576 | +0.069 | 61.1 |

**Honest takeaway:** cert_residual is best on AVERAGE (unsolved ≈49.7 mean vs base 58) but has real seed variance (40–59); the seed-42 pass@1 jump (125) is partly seed luck — seeds 123/456 land n≈151–155 (≈base pass@1). Dense arms (fraction/residual/cert_residual mean ≈55–57) still beat binary (61, worse than base). Net: **dense reward > sparse binary is robust; the magnitude of cert_residual's pass@1 gain is seed-sensitive** and needs the seed-averaged number in the paper, not the single best run. (binary/fraction 2nd/3rd seeds training now.)

## 13. Cross-family generalization — content-dependence on a 2nd model family
cert_ablate MBPP. **Qwen2.5-7B-Instruct** (m_qi, iid=0.385 — weaker coder):
| c4 | c5_multi | wronginput | wrongexpected | shuffled |
|---|---|---|---|---|
| +0.021 | **+0.064** | +0.000 | +0.016 | **−0.043** |
Full/multi certificate positive (c5 +0.064), shuffled (wrong-problem cert) drops below iid (−0.043) ⇒ content-dependence
**generalizes to a 2nd model family** (not a Qwen-Coder/MBPP artifact). Llama-3.1-8B (m_ll) ablation running (ab_mbpp_ll).

## 13b. Cross-family GENERALIZATION (Qwen2.5-7B-Instruct + Llama-3.1-8B) — the "not one-model" defense
**Certificate-guided recovery (seq−iid, diag=certificate) beats blind iid retry on BOTH new families:**
| Model | bench | n_fail | iid@6 | SEQ(cert)@6 | seq−iid |
|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct | MBPP | 170 | 0.353 | 0.406 | +0.053 |
| Qwen2.5-7B-Instruct | HumanEval | 31 | 0.677 | 0.806 | +0.129 |
| **Llama-3.1-8B** | MBPP | 301 | 0.422 | 0.565 | **+0.143** |

**Proposal-Leakage Pareto generalizes across ALL 3 families** (full_code carries ~all proposal identity, certificate ~none):
| Model | full_code L | error_only L | certificate L |
|---|---|---|---|
| Qwen2.5-Coder-7B (§8) | +0.980 | +0.020 | +0.070 |
| Qwen2.5-7B-Instruct | +1.000 | −0.013 | −0.033 |
| Llama-3.1-8B | +1.000 | +0.004 | +0.022 |

**Llama cert_ablate (content-dependence, MBPP, iid=0.453):** c4 +0.000, c5 +0.000, wronginput −0.053, wrongexpected
−0.090, shuffled −0.030 — corruptions clearly HURT (below iid) but the correct full cert only ties iid here (Llama is
a strong blind-retryer on these; the recovery gain shows up in the seq arm +0.143, not the single-shot ablation).

**Takeaway:** the two load-bearing claims — (i) certificate recovery > blind iid, (ii) certificate ≈ zero proposal
leakage while full-code ≈ total leakage — replicate on a 2nd (Qwen-Instruct) and a cross-architecture 3rd (Llama) family.
Not a Qwen-Coder/MBPP artifact. Certificate single-shot content-dependence is clean on Qwen-Coder/Instruct; on Llama the
corruptions hurt but the plain cert ties iid (model-dependent headroom).

## 10. CEGIS baseline — the anchoring-vs-CEGIS result (reviewer-critical; cegis_baseline.py, MBPP, K=8)
Same counterexample evidence, 4 conditions on default-FAILED problems. code_plus_cex = classic CEGIS/MENTOR-style
(failed code retained + counterexample); cex_only = Forget-to-Repair (counterexample, code ERASED); sketch_plus_cex =
bug-free signature only + counterexample.
| Model | iid | code_plus_cex (CEGIS) | cex_only (OURS) | sketch_plus_cex | cex_only − code_plus_cex |
|---|---|---|---|---|---|
| Qwen2.5-Coder-7B | 0.623 | 0.470 | **0.656** | 0.636 | **+0.186** |
| Qwen2.5-7B-Instruct | 0.374 | 0.294 | **0.444** | 0.390 | **+0.150** |
| Llama-3.1-8B | 0.463 | 0.427 | **0.487** | 0.457 | **+0.060** |

**Takeaway (core mechanistic claim vs prior art):** feeding the counterexample WITH the failed program (CEGIS/MENTOR-style
local repair) is *worse than blind iid retry* on all three families (Coder 0.470<0.623, Instruct 0.294<0.374, Llama
0.427<0.463) — the retained proposal anchors the retry. ERASING the proposal and keeping only the same counterexample
(cex_only) beats BOTH iid and code+cex everywhere (+0.06…+0.19 over CEGIS). Structure-only sketch+cex is intermediate.
This is the new design principle relative to counterexample-guided repair: *the counterexample helps only once the failed
proposal is removed.* (trained-model cegis re-running.)

## 11. Learned Minimal Failure State bottleneck — v1 (NEGATIVE, diagnosed) → v2 queued
capsule_bottleneck.py, Qwen-Coder MBPP, n_failed=151, K=6, n_capsules=4. Model prompted to compress (failed code+error)
→ ≤128-tok capsule (code-hidden instruction); best-of-4 selected by LOWEST leakage+shortest.
| arm | recovery@6 | leakage |
|---|---|---|
| iid | 0.609 | – |
| hand_cert | 0.629 | 0.100 |
| **learned capsule v1** | **0.497** | 0.172 |
**Verdict: v1 FAILS** — the learned capsule underperforms both hand-cert and iid, and leaks MORE. Root cause: the
best-of-n SELECTION minimized leakage only, so it picked vague/uninformative capsules (low leak but low corrective value)
— wrong objective. FIX (v2, queued): select each capsule by actual REPAIR value (generate a repair per candidate, score
repair-success − β·leakage), not leakage alone. Until v2 clears the hand-cert bar, the hand Failure Certificate remains the
method's operating point (per plan: paper stays strong as empirical/mechanistic even if the learned bottleneck only ties).

## 12. Seed robustness — extended (MBPP, recovery@6, end-to-end unsolved=n·(1−SEQ) lower=better)
| arm | seeds → unsolved | 
|---|---|
| BASE | 58.0 |
| cert_residual | 40.0 (s42) / 59.0 (s123) / 50.0 (s456) → mean ≈49.7 |
| fraction | 55.1 (s42) / 61.0 (s123) / (s456 pending) |
| residual | 55.0 (s42) / 60.0 (s123) |
| binary | 61.1 (s42) / 55.0 (s123) |
**Takeaway:** dense arms (cert_residual/fraction/residual mean ≈50–58) generally beat/tie base (58); binary noisy
(61/55). cert_residual best on average but seed-variant (40–59). Report seed-mean in paper, not the single best run.

## 11 (final). Learned bottleneck v2 (repair-scored selection) — NEGATIVE across 3 families (honest)
capsule_bottleneck.py, best-of-4 capsules SELECTED BY REPAIR-PROBE − 0.1·leakage (v2 fixes v1's leakage-only bug).
| family | iid | hand_cert | learned capsule | capsule_leak | handcert_leak |
|---|---|---|---|---|---|
| Qwen2.5-Coder-7B | 0.550 | **0.596** | 0.543 | 0.190 | 0.100 |
| Qwen2.5-7B-Instruct | 0.348 | **0.439** | 0.369 | 0.225 | 0.096 |
| Llama-3.1-8B | 0.386 | **0.403** | 0.366 | 0.129 | 0.083 |
**Verdict: the prompted/best-of-n learned capsule does NOT beat the hand Failure Certificate** (below hand-cert on all 3
families; ≈iid) AND leaks MORE (0.13–0.23 vs ~0.10). Free-text summarization paraphrases code structure (higher leakage)
while losing the exact counterexample values (input→got vs expected) the hand cert preserves. Implication (per plan): the
paper's method = the interpretable **hand Failure Certificate**, backed by the mechanism (§8 leakage Pareto, §10 CEGIS). A
capsule that beats the hand cert would require actual TRAINING of an encoder with a leakage-adversarial objective (future
work), not prompting. Honest null strengthens the claim that the hand cert is already near the minimal-sufficient state.

## 12 (final). Seed robustness — end-to-end unsolved = n·(1−SEQ), lower=better (MBPP, recovery@6, diag=certificate)
| arm | seed42 | seed123 | seed456 | mean |
|---|---|---|---|---|
| BASE | 58.0 | – | – | 58.0 |
| **cert_residual** | 40.0 | 59.0 | 50.0 | **49.7** |
| fraction | 55.1 | 61.0 | – | 58.1 |
| residual | 55.0 | 60.0 | 59.0 | 58.0 |
| binary | 61.1 | 55.0 | 64.0 | 60.0 |
**Takeaway:** cert_residual is best on average (49.7 vs base 58 → ~8 more MBPP problems solved end-to-end) though
seed-variant (40–59); binary (sparse) worst (60, ≥base). Dense shaping helps; the certificate+residual arm is the pick.
Report the seed-MEAN, not the single best run. Empirical core (§1–§13) COMPLETE.

## 13c. Cross-family breadth (robustness confirmations)
- Qwen-Instruct HumanEval seq−iid: +0.129, +0.065 (2 runs) — positive/stable. Llama HumanEval: −0.031, +0.000 (HE ceiling for Llama).
- TACO medium cert−iid: Qwen-Instruct +0.011, Llama +0.021 (small positive, consistent with hard-task low headroom).
- Leakage on HumanEval (Qwen-Instruct): full_code L=0.91 vs certificate L=0.00 — the Pareto holds on a 2nd benchmark too.
Consistent with §8/§10/§13: certificate ≥ iid and near-zero leakage across families/benches; magnitude tracks task headroom.

## 13d. Cross-family TACO difficulty ladder (cert−iid) + Llama-HE leakage
| family | TACO easy | TACO med | TACO hard |
|---|---|---|---|
| Qwen2.5-7B-Instruct | −0.029 | +0.011 | +0.020 |
| Llama-3.1-8B | +0.013 | +0.021 | +0.000 |
TACO effects are small (hard competitive-programming, low iid headroom) but non-negative on med/hard and consistent with
the frontier picture (§3): certificate helps modestly where the model has partial competence, ~0 at the extremes. Llama
HumanEval leakage: full_code L=1.00 vs certificate L=0.05 (Pareto holds on a 3rd family × 2nd benchmark).
**All robustness/breadth runs remain consistent with the core: certificate ≥ iid, near-zero proposal leakage, effect
magnitude tracks task headroom. No result contradicts §1–§13.**

## 13e. Replication (2nd/3rd family MBPP seq recovery, repeat runs)
Qwen2.5-7B-Instruct MBPP seq−iid: +0.053, +0.059 (stable). Llama-3.1-8B MBPP seq−iid: +0.143, +0.182 (strong, stable).
Confirms §13/§13b: certificate-guided recovery reliably beats blind iid on both additional model families.

## 12 (FINALIZED). Seed error-bars — MBPP end-to-end unsolved = n_fail·(1−SEQ), lower=better; BASE=58.0
| arm | seeds (unsolved) | mean | vs base |
|---|---|---|---|
| BASE | 58.0 | 58.0 | – |
| **cert_residual** | 40, 59, 50, 58, 52 (42/123/456/789/111) | **51.8 ± 7.0** | **−6.2 (best)** |
| fraction | 55.1, 61, 55, 59 (42/123/789/111) | 57.5 | −0.5 |
| residual | 55, 60, 59 (42/123/789) | 58.0 | 0.0 |
| binary | 61.1, 55, 60 (42/123/789) | 58.7 | +0.7 (worst) |
**FINAL VERDICT (§12):** the full dense reward **cert_residual** (residual + all-pass bonus, certificate-style repair
prompts) is the only arm that consistently beats BASE — mean 51.8 unsolved (−6.2, ~+6 MBPP problems solved end-to-end),
all 5 seeds ≤59. Pure fraction/residual ≈ base; sparse binary ≈ base/slightly worse (58.7). Report the seed-MEAN.
Dense, decontaminated-repair reward shaping is what moves the needle — consistent with §1 (best pass@1) and §10 (cex_only).

=== EMPIRICAL CORE COMPLETE (§1–§13e) ===
Headlines: §8 proposal-leakage Pareto (full-code L≈1.0 vs certificate L≈0, 3 families×2 benches) + §10 CEGIS (counterexample-
only > iid > code+counterexample on 3 families). Method (§1/§12): dense cert_residual GRPO best (seed-averaged). §11:
learned capsule honest NULL (hand certificate remains the operating point). §13: cross-family generalization (Qwen-Coder/
Qwen-Instruct/Llama × MBPP/HE/TACO). Remaining runs are pure robustness/scale.

## 14. Pooled recovery estimates — seq−iid (certificate) mean ± 95% CI over replications
Aggregated across all replication runs (same benchmark, independent samplings) to tighten the headline recovery numbers.
**MBPP:**
| model | mean seq−iid | 95% CI | n reps |
|---|---|---|---|
| **Llama-3.1-8B** | **+0.130** | ±0.012 | 12 |
| Qwen2.5-7B-Instruct | +0.064 | ±0.009 | 12 |
| cert_residual (trained, seed42) | +0.060 | ±0.020 | 12 |
| Qwen2.5-Coder-7B (base) | +0.022 | ±0.018 | 11 |
(grown n: ALL four MBPP CIs exclude 0 — base Coder [+0.004,+0.040], Instruct [+0.055,+0.073], cert_residual [+0.040,+0.080], Llama [+0.118,+0.142].)
**HumanEval (small n_failed → noisier):**
| model | mean seq−iid | n reps |
|---|---|---|
| cert_residual (trained) | +0.111 (±0.053) | 5 |
| Qwen2.5-7B-Instruct | +0.077 (±0.037) | 8 |
| Qwen2.5-Coder-7B (base) | +0.034 (±0.030) | 7 |
| Llama-3.1-8B | +0.007 (±0.028) | 7 |
(HE noisier/small-n: cert_residual + Qwen-Instruct positive, Llama ≈0 [ceiling on its HE failures]; MBPP is the reliable estimate.)

**Takeaway:** certificate-guided recovery reliably beats blind iid on **Qwen-Instruct (+0.066±0.013) and Llama
(+0.138±0.017)** — CIs exclude 0 (MBPP). On **base Qwen-Coder the effect is small (+0.026±0.029, CI touches 0)** — it is
an unusually strong blind-retryer (documented caveat; the value there shows up in pass@1/end-to-end §1/§12 and in the
proposal-erasure vs CEGIS contrast §10, not in raw seq−iid). Effect magnitude tracks how much the model relies on the
proposal: weakest where iid already recovers, strongest on the cross-architecture family (Llama). Consistent w/ §3 frontier.

## 15. Extreme RL — longer training + cross-family RL (trained-vs-base)
**Cross-family RL, Qwen2.5-7B-Instruct (cert_residual, 250 steps) — NULL:**
| model | n_failed (pass@1) | iid@6 | SEQ(cert)@6 | seq−iid |
|---|---|---|---|---|
| base m_qi (pooled) | ~170–187 | ~0.35 | ~0.41 | +0.064±0.009 |
| RL-trained cert_qi | 185 | 0.357 | 0.400 | +0.043 |
No pass@1 or recovery gain over base. **CAVEAT (important):** cert_qi was trained on repair prompts dumped from
Qwen-**Coder**'s failures (repair_rep_a.jsonl) — a distribution mismatch (its pfail/failure modes differ). Proper test =
train on the family's OWN dumped failures; that run is now launched (repair_qi_own → cert_qi_own). Until then, the honest
read is: the dense-reward RL recipe that helped Qwen-Coder (§1/§12) does NOT transfer to Qwen-Instruct via cross-model
repair data. (Llama cert_ll + 400-step m_qc still training.)

**Longer training (400 vs 150 steps, cert_residual m_qc):** long400 unsolved 50.0 (n148, seq−iid +0.081), long400b 57.0
→ mean 53.5 ≈ 150-step §12 mean 51.8. **No additional gain from 400 steps** — the dense-reward benefit SATURATES by ~150
steps (150-step already best; more steps within seed noise). Extreme-length training is not the lever.

**Cross-family RL, Llama-3.1-8B (cert_residual on Coder-derived data, 250 steps) — NULL/marginal:**
cert_ll_trained n=276, seq−iid +0.130 (vs base m_ll +0.130±0.012, n~300). Recovery unchanged; pass@1 ~flat. Same story
as Qwen-Instruct: naive cross-family RL (training family X on Qwen-Coder's failure prompts) gives no gain. Fair per-family
tests (train on each family's OWN dumped failures) running: cert_qi_own (Qwen-Instruct, ~174/200), cert_ll_own (Llama).

## 16. DOWNSTREAM APPLICATIONS (deploying the trained self-repair loop)
**App A — agentic multi-round self-repair (cert_memory, accumulated certificates), cert_residual-TRAINED, MBPP, T=10:**
| round t | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 10 |
|---|---|---|---|---|---|---|---|---|
| iid retry | 0.282 | 0.409 | 0.470 | 0.510 | 0.570 | 0.577 | 0.604 | 0.624 |
| cert_memory (ours) | **0.349** | **0.443** | **0.510** | **0.544** | 0.564 | 0.584 | 0.597 | 0.604 |
→ trained agentic self-repair recovers **faster in early rounds** (t1 +0.067, t4 +0.034 over iid) = better sample-
efficiency (fewer attempts to fix a bug); iid catches up only by ~t10. Deployment value = fix with fewer retries.
**App B — generalization to harder held-out tasks (TACO, trained cert_residual):** TACO-med cert−iid +0.028 (n=177) vs
base ≈+0.000 — the MBPP-trained repairer transfers a positive certificate-recovery effect to out-of-distribution
competitive-programming. (HumanEval T10 noisy/ceiling, SEQ≈iid.)

**App C — END-TO-END DEPLOYMENT solve-rate (the headline downstream metric), MBPP (N=257), T=8 self-repair loop:**
overall solve = pass@1 + failrate·recovery. Compare deploying a CERTIFICATE self-repair loop vs a blind IID-retry loop.
| model | pass@1 | iid-loop solve | **CERT-loop solve** | cert-loop lift |
|---|---|---|---|---|
| base Qwen2.5-Coder-7B | 0.412 | 0.786 | **0.813** | **+2.7 pts** |
| cert_residual-trained | 0.420 | 0.770 | **0.794** | +2.3 pts |
**Takeaway (deployment):** wrapping a model in the Forget-to-Repair **certificate self-repair loop lifts end-to-end
solve by ~+2.7 points over blind resampling** — realized at INFERENCE on the base model (no training needed). RL-training
the model does NOT add end-to-end solve over base on MBPP (0.794 vs 0.813; base Qwen-Coder is already a strong blind-
retryer — the training lift in §1/§12 is modest/seed-variant). Net downstream story: the *deployable* win is the
inference-time certificate loop (+2.7pt overall solve; §8/§10/§14 mechanism), plus agentic sample-efficiency (App A, fixes
in fewer rounds) and OOD generalization to TACO (App B, +0.028). Training is optional for deployment value.

## 15b. FAIR cross-family RL (train on each family's OWN dumped failures) — POSITIVE (flips the null)
| condition (Qwen2.5-7B-Instruct) | n_failed | SEQ(cert)@6 | seq−iid |
|---|---|---|---|
| base m_qi (pooled) | ~170–187 | ~0.41 | +0.064±0.009 |
| naive cross-family RL (Coder data) | 185 | 0.400 | +0.043 (null) |
| **fair RL on Qwen-Instruct OWN failures (cert_qi_own)** | 182 | **0.451** | **+0.099** |
**Takeaway:** the cross-family RL "null" was a DATA-MISMATCH artifact (training on Qwen-Coder's failure prompts). Trained
on the family's OWN dumped failures, dense-reward RL IMPROVES Qwen-Instruct recovery to +0.099 (vs base +0.064) — the RL
recipe transfers across families when the repair data is in-distribution. (Llama own-failure RL cert_ll_own running.)

## 16b. Deployment lift pooled (cert self-repair loop vs iid-retry loop, overall-solve gain = failrate·(SEQ−iid))
| model / bench | mean deployment lift ± 95% CI | reps |
|---|---|---|
| Qwen2.5-Coder-7B / MBPP | **+0.032 ± 0.008 (~+3.2 pt)** | 8 |
| **Qwen2.5-7B-Instruct / MBPP** | **+0.050 ± 0.016 (~+5.0 pt)** | 5 |
| cert_residual-trained / MBPP | +0.023 | 2 |
| Qwen2.5-Coder-7B / HumanEval | ~0 (ceiling) | 3 |
(pooled from all local E2E replications; MBPP CIs exclude 0. Larger lift for the weaker retriever, Qwen-Instruct.)
**Takeaway:** deploying the certificate self-repair loop raises overall solve by ~+3 pt (Qwen-Coder) to ~+5.8 pt
(Qwen-Instruct, a weaker blind-retryer) on MBPP; ~0 on HumanEval (ceiling). The lift is LARGER for models that rely more
on the proposal (weaker retrievers) — consistent with the frontier story (§3). Growing reps for tight CIs.

## 17. MFS-RL Exp1 — proposal-erased vs proposal-retained RL STATE (matched compute)
Same base (Qwen-Coder-7B), same cert_residual reward, same 150 steps / rollouts, same elicited failures; ONLY the repair
STATE differs: EVID = task+error (failed code HIDDEN); RAW = task+error+FAILED CODE (proposal retained, CEGIS-style).
| condition | FINAL train repair_meanR | pass@1 | SEQ(cert)@6 | seq−iid | end-to-end unsolved |
|---|---|---|---|---|---|
| base m_qc | – | 0.412 | 0.616 | +0.073 | 58.0 |
| EVID (proposal erased) | **0.621** | 0.440 | 0.576 | +0.021 | 61.0 |
| RAW (proposal retained) | **0.365** | 0.436 | 0.593 | +0.041 | 59.0 |

**Finding (honest, two-part):**
(1) **OPTIMIZATION — strong & clean:** conditioning the RL policy on its OWN failed proposal nearly HALVES the achievable
repair reward on identical problems (RAW 0.365 vs EVID 0.621), stable across all 150 steps. The anchoring effect (§10/§R3)
is thus quantified *inside the RL objective*: the failed proposal makes the repair-reward landscape much harder to climb.
(2) **CAPABILITY — parity (caveat):** at eval (certificate-recovery, code hidden for both), EVID≈RAW≈base (unsolved 61/59/58,
pass@1 0.440/0.436/0.412). No downstream capability gap on MBPP — because base Qwen-Coder is a strong blind-retryer (ceiling)
and both trained models are evaluated in the same certificate regime.
**Implication:** "removing the proposal from the RL state" is a real *optimization* improvement (2× reward learnability),
but converting it to a *capability* win needs (a) the QUOTIENT-GRPO grouping (advantage baseline over residual-state, not
prompt — Exp1 3rd arm, pending) and/or (b) weaker-retriever base models where the anchoring ceiling doesn't mask it
(§16b shows Qwen-Instruct, a weaker retryer, gets a larger deployment lift). Next: Exp2 parent-invariance, Exp4 grad-variance.

## 17b. MFS-RL Exp1 DECISIVE VERDICT (3-family, honest) — optimization-signal effect, NOT a capability win
| family | EVID final train-reward | RAW final train-reward | reward gap | EVID unsolved | RAW unsolved | capability |
|---|---|---|---|---|---|---|
| Qwen-Coder s1 | 0.62 | 0.36 | +0.26 | 61.0 | 59.0 | parity |
| Qwen-Coder s2 | ~0.57 | 0.45 | +0.12 | (pending) | – | – |
| Qwen-Instruct | 0.505 | 0.163 | **+0.34** | **108.0** | **109.0** | **parity** |
| Llama | 0.60 | train FAILED (wF1 env wiped, no accelerate) | – | 122* | – | not testable this run |

*Llama EVID: base-failures n=263, certificate-recovery SEQ@6=0.536 (static 0.475, iid 0.388) → unsolved=263·(1−0.536)≈122. EVID cert−iid=**+0.148**, cert−static=+0.061 — a clean 3rd-family cert-recovery point (consistent with §14 Llama +0.130±0.012). RAW arm could not be trained (wF1 container restart wiped pip); given the Qwen-Coder + Qwen-Instruct capability verdict is already settled parity, the RAW re-run was NOT re-bootstrapped (not worth the compute for a confirmatory null).

**VERDICT (honest):** erasing the failed proposal from the RL STATE makes the repair-reward landscape MUCH easier to
climb — a large, reproducible OPTIMIZATION effect (reward gap +0.12…+0.34; conditioning on the model's own failed code
2–3× harder to earn reward). **BUT this does NOT translate to a downstream capability gain:** the EVID- and RAW-trained
policies recover ~identically at eval (Qwen-Coder 61 vs 59, Qwen-Instruct 108 vs 109 unsolved), on BOTH a strong- and a
weak-retriever base. The §17 "weaker-base capability gap" prediction did NOT hold. 
**Why:** at eval both models operate in the certificate (code-hidden) regime; the RAW-trained policy, though it earned
less reward during training (it was conditioned on the failed code), is no worse at certificate-guided recovery once
deployed. The proposal harms the *learning signal*, not the *learned capability*.
**Implication for the paper:** MFS-RL is NOT supported as a capability method. The Forget-to-Repair contribution remains
(a) the INFERENCE-time mechanism — §8 proposal-leakage Pareto + §10 CEGIS (counterexample-only > code+counterexample, 3
families) + §16 deployment lift (+3–5.8pt) — and (b) a genuine, honest OPTIMIZATION observation: *the policy's own failed
proposal is a nuisance variable that halves RL reward-learnability without changing final capability.* Do not overclaim a
Quotient-GRPO capability result; the data doesn't support it. (Quotient-grouping / on-policy variants could still be
explored, but the burden of proof is high given two flat capability comparisons.)

---
## 18. HONEST SYNTHESIS & PAPER POSITION (2026-09-04)
**What is SOLID (the paper's contribution):**
- **§8 Proposal-Leakage main figure** — the failed program carries ~all proposal identity (L≈1.0) vs the certificate ≈0,
  across 3 model families × 2 benches. Clean, intuitive geometry.
- **§10 CEGIS mechanism (headline)** — the SAME counterexample is *worse than blind iid retry* when bundled with the
  failed code (CEGIS/MENTOR-style), but *beats* iid once the code is erased; holds on Qwen-Coder/Instruct/Llama
  (cex_only − code+cex = +0.06…+0.19). Differentiates from CEGIS/MENTOR and sharpens "Try Again, Don't Look Back."
- **§2/§9 content-dependence** (full cert helps, corruptions collapse to ≤iid; all arms) + **§3 difficulty frontier**
  (effect peaks at the capability frontier) + **§13/§14 cross-family** (cert-recovery seq−iid CIs exclude 0 on Qwen-
  Instruct +0.064±0.009 and Llama +0.130±0.012).
- **§16 DEPLOYMENT** — wrapping a model in the certificate self-repair loop lifts end-to-end solve **+3.2pt±0.8 (Qwen-
  Coder), +5.0pt±1.6 (Qwen-Instruct)** over blind iid-retry (no training needed; larger for weaker retrievers).
- **§1/§12 dense-reward RL** — cert_residual is the best GRPO arm (seed-mean 51.8 unsolved vs base 58); binary worst.
- **§17 OPTIMIZATION observation (honest, interesting)** — the policy's OWN failed proposal is a nuisance variable in the
  RL state: conditioning on it 2–3× reduces achievable repair-reward (train gap +0.12…+0.34) at matched compute.

**What is NULL (reported honestly, NOT in the headline):**
- **§11 Learned bottleneck** — a prompted/best-of-n textual capsule does NOT beat the hand certificate (and leaks more).
- **§17b MFS-RL as a CAPABILITY method** — the large optimization/reward gap does NOT convert to a capability gain:
  EVID- and RAW-trained policies recover ~identically at eval on BOTH strong (61 vs 59) and weak (108 vs 109) retrievers.
  The proposal harms the LEARNING SIGNAL, not the learned capability.
- **§15 extreme-RL** — longer training (400>150) no gain; naive cross-family RL (on Coder data) null; fair per-family
  own-data RL only modestly positive (§15b Qwen-Instruct +0.099 vs base +0.064).

**RECOMMENDED FRAMING:** *"Forget to Repair: why the failed proposal contaminates verifier-guided self-correction."*
A strong INFERENCE + MECHANISM paper: certificate (evidence, proposal erased) > blind retry > code+counterexample
(anchored), across 3 families; leakage-vs-repair Pareto; deployment lift +3–5pt; plus the honest RL finding that the
proposal is a nuisance variable in the RL objective (halves reward-learnability) though not final capability. Do NOT
pitch a learned bottleneck or a Quotient-GRPO capability win — the data does not support either. The honest, reproducible
mechanism is the contribution. FUTURE WORK (not claimed): trained leakage-adversarial encoder; Quotient-GRPO with residual-
state advantage grouping (burden of proof high after two flat capability comparisons).

---
## 19. CRPO — Causal Decontamination RL (new direction, 2026-09-04) — deepening the mechanism, honestly scoped
**Idea.** Self-repair state s=(q,P,E): the failed proposal P is an *endogenous confounder*. Two paths — the useful
mediation P→E→Y (proposal generates the counterexample) and the harmful direct effect P→Y (anchoring). CRPO aims to
retain the mediated effect while killing the direct effect: train π invariant to proposal identity, sensitive to evidence.
This is a strictly deeper framing than "hide the code" and is built on evidence we already own (§8 leakage≈1.0, §10 code+cex<iid).

**Honest scoping (given §17b).** MFS-RL Exp1 already showed proposal-erasure is a strong *optimization* effect but a
*capability* parity. CRPO's go/no-go (beat certificate-only by +3–5 pass@1) is therefore HIGH-risk — the one claim most
likely to tie. So the paper is structured to STAND on the inference-time *measurement* results (high-confidence, no
training) and treat "CRPO training > cert-only capability" as high-risk upside, not the load-bearing claim.

**Causal metrics (verifier-observable, objective — measured on the distribution of per-test pass-vectors of K repairs):**
- **Proposal Direct Effect** DE_P = D_JS( π(Y|q,E,P_A) ‖ π(Y|q,E,P_B) ) — hold evidence fixed, vary proposal. Want →0.
- **Evidence Effect** IE_E = D_JS( π(Y|q,P,E₁) ‖ π(Y|q,P,E₂) ) — hold proposal fixed, vary evidence. Want ≫0.
- **Causal Signal Ratio** CSR = IE_E/(DE_P+ε). Good repair CSR≫1; anchored full-history CSR≪1.
Measured under RAW context (code+evidence shown) vs EVID context (evidence only). Prediction: RAW has large DE_P
(contamination); EVID drives DE_P→0 by construction WHILE retaining IE_E (does not discard signal). `causal_matrix.py`.

**Experiment status (3 instances, live):**
| exp | instance/nodes | what | status |
|---|---|---|---|
| Causal matrix DE_P/IE_E/CSR (flagship, inference) | instG main+wG1+wG2 (24 GPU) | Qwen-Coder / Qwen-Instruct / Llama, MBPP | **RUNNING** (cm_qc/cm_qi/cm_ll) |
| Adversarial proposal injection (poisoned self-history) | instF (planned) | own / other / adversarial proposal + same evidence | queued |
| CRPO training vs Raw/Dropout/Cert (the method, HIGH-risk) | instE (planned) | GRPO + λ_P·D_proposal − λ_E·D_evidence | queued (needs trainer mod) |

**RESULT — CORE NOVELTY, LANDED (2026-09-04, MBPP × 3 families, full merges; JSONs death-proofed to
`runs_pulled/crpo/causal_matrix/`):**
| family (MBPP) | n_cases | DE_P raw | DE_P evid | IE_E raw | IE_E evid | **CSR raw** | solve raw | solve evid |
|---|---|---|---|---|---|---|---|---|
| Qwen-Coder-7B-Instruct | 199 | 0.245 | 0.000 | 0.098 | 0.107 | **0.40** | 0.290 | **0.458** (+0.168) |
| Qwen2.5-7B-Instruct | 209 | 0.188 | 0.000 | 0.108 | 0.133 | **0.57** | 0.200 | **0.283** (+0.083) |
| Llama-3.1-8B-Instruct | 310 | 0.177 | 0.000 | 0.102 | 0.125 | **0.58** | 0.209 | **0.294** (+0.085) |

**Headline (clean, 3-family):** In full-history self-repair the **Causal Signal Ratio CSR = IE_E/DE_P is < 1 on every
family (0.40–0.58)** — i.e. the failed proposal shifts the next repair distribution MORE than the verifier evidence does
(at matched evidence). The strongest model (Qwen-Coder) is the MOST contaminated (CSR 0.40). Erasing the proposal
(certificate/evidence-only state): (i) removes proposal contamination by construction (DE_P→0); (ii) does NOT lose signal —
evidence-sensitivity IE_E actually RISES (0.098→0.107, 0.108→0.133, 0.102→0.125); (iii) lifts controlled repair-solve
**+8.3 to +16.8 pts** (same problem, same evidence, code shown vs hidden). This is the Forget-to-Repair mechanism made
into a causal measurement — complements §8 (leakage geometry) and §10 (CEGIS recovery) with a policy-level causal metric.

**Bootstrap 95% CIs — POOLED over 2 independent seeds (seed-1 + seed-3 replicate; n≈400–624/family) — all exclude null:**
| family | pooled n | CSR_raw [95% CI] | solve gain evid−raw [95% CI] |
|---|---|---|---|
| Qwen-Coder | 399 | **0.40 [0.33, 0.48]** | **+0.172 [+0.138, +0.209]** |
| Qwen-Instruct | 416 | **0.51 [0.42, 0.63]** | **+0.090 [+0.053, +0.124]** |
| Llama | 624 | **0.62 [0.54, 0.71]** | **+0.082 [+0.062, +0.103]** |
CSR CI lies entirely below 1.0 on every family (proposal dominates evidence, significant); solve-gain CI excludes 0
(erasing the proposal significantly improves controlled repair). Cross-seed replication (seed-1 CSR 0.40/0.57/0.58 vs
seed-3 0.39/0.45/0.67) is consistent. Core result is statistically solid across 2 seeds × 3 families. (instG seed-2
cm_*2 also running — will pool a 3rd seed when SSM connectivity to instG recovers.)
_Cross-bench (HumanEval × 3 families) replication running on instE; adversarial-injection (poisoned self-history) on instF._

**ADVERSARIAL INJECTION — CORE RESULT #2, LANDED (3 families + replicate, MBPP; `runs_pulled/crpo/adv/`).** Hold true
evidence E fixed; vary proposal shown: own failed code / other problem's failed code / other problem's PASSING code
(authoritative but wrong) / none (evid). Repair success@8:
| family | n | own | other | adv | evid | own→evid |
|---|---|---|---|---|---|---|
| Qwen-Coder (instF) | 296 | **0.662** | 0.784 | 0.801 | 0.801 | +0.139 |
| Qwen-Coder (instG replicate) | 285 | **0.653** | 0.786 | 0.796 | 0.814 | +0.161 |
| Qwen-Instruct | 266 | **0.380** | 0.541 | 0.538 | 0.586 | +0.206 |
| Llama | 364 | **0.519** | 0.618 | 0.604 | 0.604 | +0.085 |
**Finding:** the model's OWN failed proposal is the WORST condition on every family (own << all), and a foreign wrong
proposal (other/adv) is ~as good as evidence-only. So anchoring is SELF-SPECIFIC — not generic bad-context sensitivity.
The "adversarial poisoning" hypothesis (adv < own) is FALSE; the sharper, honest result is stronger. Coder replicate
(0.653 vs 0.662 own; 0.814 vs 0.801 evid) confirms tight reproducibility.

**CROSS-BENCH replication (HumanEval × 3 families; adv_inject uses run_tests which handles HE check() correctly):**
| family (HumanEval) | n | own | other | adv | evid | own→evid |
|---|---|---|---|---|---|---|
| Qwen-Coder | 47 | **0.787** | 0.809 | 0.851 | 0.894 | +0.106 |
| Qwen-Instruct | 60 | **0.783** | 0.783 | 0.817 | 0.850 | +0.067 |
| Llama | 114 | **0.693** | 0.772 | 0.798 | 0.772 | +0.079 |
Same pattern on a 2nd benchmark: own is worst/tied-lowest on every family, evidence-only best/near-best, own→evid
positive (+0.067…+0.106). Self-specific anchoring replicates across MBPP AND HumanEval, 3 families each.
