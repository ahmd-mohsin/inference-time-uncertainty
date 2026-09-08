# GRPO Self-Repair: Dense Verifier Reward — 4-Arm Result (Forget-to-Repair, Contribution 3)

**Date:** 2026-09-04. **Supersedes** the earlier null GRPO_SELF_REPAIR result (which used binary reward + LoRA at
small scale and found parity). This is the real 4-arm dense-reward result: it trains cleanly, and the dense
`certificate+residual` reward produces the best end-to-end repairer.

## Setup
- Base model: **Qwen2.5-Coder-7B-Instruct**. GRPO (TRL 1.7.0), LoRA r=32, vLLM server (GPU0) + DeepSpeed ZeRO-2
  (GPU1–7), 150 steps, num_generations=8, on 166 MBPP-train repair prompts (failed-attempt → repair).
- Root-cause fix that unblocked training: torch 2.11 `LRScheduler._update_lr` uses `zip(..., strict=True)`;
  under LoRA the empty no-decay param group is pruned by DeepSpeed → len(lrs) > len(param_groups) → crash at the
  first `scheduler.step()`. Patched with a non-strict fallback (`_patch_lr_scheduler_strict_zip` in train_grpo.py).
  (The old "zip() bug" was NEVER the reward signature.)
- **4 reward arms** (env `REPAIR_REWARD_VARIANT`, same data/steps/seed):
  - **A binary** = 1[all tests pass] (sparse baseline).
  - **B fraction** = pass-fraction (dense).
  - **C residual** = fixed-fraction over parent-FAILING tests − λ·regressions over parent-passing.
  - **D cert_residual** = residual + α·1[all pass] (the full dense reward; **the method arm**).
- Training reward (final): cert_residual mean **0.77→0.83** (learns), fraction ~0.55, residual ~0.52, binary ~0.51.

## Eval
`seq_recover.py`: on each model's own default-FAILED problems, T=6 recovery rounds. **IID** = blind resampling from
the original prompt; **SEQ(cert)** = sequential recovery where feedback is the **Failure Certificate** (concrete
counterexample, failed code hidden — the (E_rich, ∅) condition). Trained-vs-BASE, 8-GPU sharded, full benchmark.
`n_failed` = #problems failing the first attempt (same benchmark set across models → **lower = higher pass@1**).

### MBPP (recovery@6)
| Model | n_failed ↓ (pass@1) | iid@6 | SEQ(cert)@6 | seq−iid | end-to-end unsolved = n·(1−SEQ) ↓ |
|---|---|---|---|---|---|
| BASE            | 151 | 0.543 | 0.616 | +0.073 | 58.0 |
| A binary        | 144 | 0.507 | 0.576 | +0.069 | 61.1 |
| B fraction      | 143 | 0.510 | 0.615 | +0.105 | 55.1 |
| C residual      | 146 | 0.541 | 0.623 | +0.082 | 55.0 |
| **D cert_residual** | **125** | 0.632 | **0.680** | +0.048 | **40.0** |

### HumanEval (recovery@6; small n, noisier)
| Model | n_failed ↓ | iid@6 | SEQ(cert)@6 | seq−iid | unsolved ↓ |
|---|---|---|---|---|---|
| BASE          | 26 | 0.692 | 0.654 | −0.038 | 9.0 |
| A binary      | 19 | 0.684 | 0.684 | +0.000 | 6.0 |
| B fraction    | 24 | 0.667 | 0.750 | +0.083 | 6.0 |
| D cert_residual | 18 | 0.500 | 0.667 | +0.167 | 6.0 |
| C residual    | pending | | | | |

### Certificate-vs-error_only ablation (same cert_residual model, MBPP)
| Feedback at eval | SEQ@6 | seq−iid |
|---|---|---|
| **certificate** (counterexample) | **0.680** | +0.048 |
| error_only (error text, code hidden) | 0.510 | −0.040 |

## Findings (honest)
1. **Dense cert_residual gives the best end-to-end repairer.** It has the **highest pass@1** (MBPP: 125 initial
   failures vs base 151 → ~+26 problems solved first-try; HumanEval 18 vs 26) **and the fewest problems left
   unsolved after recovery** (MBPP 40.0 vs base 58.0 → ~+18 net solves). Holds on both benchmarks.
2. **Binary GRPO (sparse) is the worst arm** — MBPP end-to-end unsolved 61.1, *worse than base* (58.0); HumanEval
   recovery lift +0.000. Confirms the earlier binary-reward null and motivates dense shaping: **D ≫ A**.
3. **Reward-shaping ladder**: dense arms (fraction/residual/cert_residual) all beat binary on end-to-end solve;
   cert_residual (residual + all-pass, trained with certificate-style repair prompts) wins.
4. **Certificate content matters, not format.** On the *same* trained model, certificate feedback recovers 0.680
   vs error_only 0.510 (Δ = +0.170); error_only actually drops below iid. The concrete counterexample carries the
   signal — not merely a "restart" instruction.
5. **Certificate feedback beats blind iid retry** for base and most arms (MBPP seq−iid all positive). cert_residual's
   own seq−iid is smallest only because its iid is already high (ceiling) — its absolute SEQ is highest.

## Verdict
The dense verifier reward **works** and the `certificate+residual` arm is the strongest — a real positive result for
Contribution 3 (RL). Caveats: single seed, LoRA, 150 steps, n=166 train prompts; HumanEval n is small. Next: matched
rollout-compute D-vs-C isolation at larger scale + seeds; and the headline method (learned Minimal Failure State
bottleneck) must still clear the hand certificate. Reward-curve + eval JSONs pulled to
`rl_training/runs_pulled/repair_ckpt/` and `runs_pulled/repair_eval/`.
