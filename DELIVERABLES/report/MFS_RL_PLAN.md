# Minimal Failure State RL (MFS-RL / Quotient-GRPO) — Experiment Plan (next-step, doubling down)

**Paper reframe:** *Forget to Repair: Proposal-Invariant State Abstractions for Verifier-Guided RL.*
**Thesis:** self-repair is a STATE-ABSTRACTION problem before a policy-optimization problem. The model's own failed
proposal P is a *causal nuisance variable* in the RL state s=(q,P,E): it anchors the next policy toward the failed
hypothesis. Train the policy on a proposal-invariant, verifier-sufficient state z=φ(q,P,E) — or its interpretable
special case (certificate / evidence-only) — and optimize residual-verifier-state transitions on the quotient MDP S/∼.

**Empirical premise (already in FORGET_TO_REPAIR_MASTER.md):** §8 full-code leakage≈1.0 vs certificate≈0; §10 CEGIS
(code+cex < iid < cex-only, 3 families); §1/§12 dense cert_residual best RL arm; §15b fair on-policy per-family RL
positive (Qwen-Instruct own-failures +0.099 > base +0.064). MFS-RL turns these into a method.

**Counterintuitive headline to prove:** *more information can make RL worse when it encodes the policy's own failed
action* — evidence-state (q,E) and quotient-state (q,z) beat raw-state (q,P,E) GRPO at IDENTICAL compute.

---
## Method components (build order)
- **A. Residual verifier state / quotient.** u(s) = per-test pass vector + concrete counterexamples {(input, got, expected)}
  over FAILED tests. Equivalence s_i∼s_j iff u(s_i)=u(s_j). Interpretable z0 = the Failure Certificate (already built,
  certify.py). Learned z = small encoder over structured verifier fields [h_q, h_failinput, h_got, h_expected, h_testmask,
  h_runtime] → soft-prefix/LoRA-conditioned into the repair LLM (NO free-text capsule — §11 falsified that).
- **B. Residual-state reward** (already implemented as cert_residual in rewards.py): r = fixed-fraction over parent-FAILING
  tests − λ·regressions over parent-passing + α·1[all pass]. Verifier VECTOR transition, not scalar.
- **C. On-policy failure refresh** (motivated by §15/§15b): every M steps, regenerate failures from the CURRENT policy,
  rebuild residual states, retrain. Failure distribution is policy-dependent.
- **Quotient-GRPO:** group rollouts by residual state z (possibly DIFFERENT failed parents), advantage baseline over G(z)
  = "what remains broken", not "which broken program produced it".

---
## The 9 experiments (mapped to our harness + 12-node plan)

**EXP 1 (KILLER — run first). Raw-state vs Evidence-state vs Quotient-state GRPO at identical compute.**
Same failures, same residual reward, same rollouts/tokens/steps; ONLY the repair-state representation differs:
- RAW  : repair prompt = task + FAILED CODE + error (proposal retained; CEGIS-style anchored state).
- EVID : repair prompt = certificate/error, code HIDDEN (= our existing cert_residual training data).
- QUOT : structured z via encoder (Component A; phase-2 once A trains).
Build: dump_repair_data variant that KEEPS code in the prompt → repair_raw.jsonl; train GRPO (RAW) vs reuse cert_residual
(EVID). Metrics: pass@1, recovery@1/6, end-to-end solve, training-reward curve, gradient variance. WANT EVID(&QUOT) > RAW.

**EXP 2. Parent intervention during training / at eval.** Pair the SAME certificate with (true parent | other parent |
no parent); measure KL(π(·|E,P_A) ‖ π(·|E,P_B)). Raw-state policy shifts; MFS/evidence policy ≈ invariant (→0). Proves
proposal-invariance. (Reuse causal_parent.py two-parent generation + certify.)

**EXP 3. State-abstraction Pareto (the defining figure).** x = proposal leakage I(z;P) (proposal_leakage.py), y = verifier
sufficiency / retained repair value. Points: raw code, AST, sketch, error-only, hand cert, learned z, iid. Method must
dominate low-leakage/high-sufficiency corner. (Have full-code≈1.0 vs cert≈0 already; add AST/sketch/learned.)

**EXP 4. Gradient variance.** Log tr(Cov(g)) per step for binary / residual / raw-state / quotient GRPO. Hypothesis:
quotient removes nuisance state → lower variance + smoother learning. Optimization-side novelty.

**EXP 5. On-policy vs stale failures.** Train on static base-model failures vs periodically-refreshed own failures
(Component C). Expect on-policy +3–5pt / faster convergence. (§15b already shows own>cross data.)

**EXP 6. Capability-frontier curriculum.** Bucket by base pass@1 {[0,.05],(.05,.2],(.2,.5],(.5,.8]}; train MFS-RL mostly
on the .2–.5 frontier band; compare uniform. Hypothesis: failure-state RL most useful where competence exists but
hypothesis selection failed (ties to §3 U-shape).

**EXP 7. Multi-round quotient RL / MFS memory.** z1→y2→z2→y3, never retain raw proposals; vs blind-iid, full-history
self-repair, cert_memory. Target same solve at 30–50% fewer attempts (App-A agentic efficiency already hints this).

**EXP 8. OOD transfer.** Train MBPP; test TACO-med, CodeContests, MBPP+/EvalPlus. Claim: learned failure-state
utilization transfers farther than solution-policy improvement. (§7/§16 App B: trained +0.028 TACO already.)

**EXP 9. Scaling law.** 1.5B/3B/7B/14B × {50,150,500,1500} updates; Δ = MFS-RL − standard GRPO. Expect: small models
large anchoring penalty→large quotient benefit; very large less. A scaling law for WHEN state decontamination matters.

---
## Node allocation (12 nodes, phase 1)
- 3× EXP1 seeds: RAW-state vs EVID-state (cert_residual) vs (later) QUOT — m_qc, matched compute.
- 2× build+train structured MFS encoder + proposal adversary (Component A).
- 1× gradient-variance logging (EXP4).
- 1× on-policy failure refresh (EXP5).
- 1× TACO-med OOD transfer (EXP8).
- 1× multi-round MFS memory (EXP7).
- 2× frontier curriculum (EXP6) / parent-intervention (EXP2).

**Primary success criterion:** at identical RL compute, EVID/QUOT-state GRPO > RAW-state GRPO on convergence AND final
capability (pass@1 / end-to-end), with lower gradient variance. That converts a strong inference paper into a new RL
methodology paper.

**Do NOT re-run:** longer vanilla GRPO, more hand-cert variants, free-text learned capsules (§11 null), single-option
routers, static diversity — all already eliminated by our own data.
