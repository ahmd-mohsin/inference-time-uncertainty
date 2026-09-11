# Method Register v3 (governing) — positive methods derived from surviving results
2026-09-11. Principle: OOD transfer ~ verified COVERAGE (distinct verified prompts reaching the gradient), gated by headroom;
the objective is NOT the lever (§114). Currency = newly-covered prompts per token. Methods: acquire coverage cheaper (M1,M3,M4,M6),
spend it better (M2), allocate where cheapest (M5,M6); M7 = law-predicted RL control.
Sequence: coverage factorial first; M1/M2/M3/M7 parallel now; M6 after factorial; M5 after M1; M4 iff F2 says drift matters.
Every method: matched-cost control + preregistered effect size (predicted from headroom) + kill rule + derived-from-a-result.

M1 CTH (coverage-targeted harvesting): sample-until-first-success + redirect saved budget to zero-success prompts (raise k, temp) then RFT.
   Derived: frozen saturation (§82b), ~70% dead groups (§80b), higher-temp reached 205 vs 159 prompts (§71b). Control: uniform-k @1.5x tokens.
M2 NCW-RFT: up-weight/up-sample verified examples whose prompt is NEWLY covered (N_new). Derived: removal control (§89, N_new carries the value).
   Control: random-mass up-weighting (equal total mass). Kill: NCW ~ random-mass.
M3 Scaffolded harvesting: harvest DAG with placeholder-scaffold PROMPT (no values), verify, strip, distill plain. Derived: §112 (Tmask ties Tvalue).
   Test-time = plain prompts. Control: scaffold placebo (irrelevant annotation). Kill: zero-success-cohort coverage unchanged, or placebo matches.
M4 Decoupled sampling (iff F2): union by MARGINAL coverage over {π_t,π_{t-1},π_0}, never mix solutions on shared prompts. Derived: §81b vs §82/§84b crossover.
M5 Headroom-aware pool allocation: train at hardest pool with non-trivial coverage; split T_gen by pilot marginal-coverage/token. Derived: fixed-hard>escalation (§90b).
M6 Marginal-cost portfolio (RANK 1): bin prompts by p̂; allocate budget to cheapest coverage operator per bin over {iid, repair, decomposition}; distill to direct prompt.
   Derived: iid cost c/p unbounded (§101), repair 3/16/40% (§92b), decomp 8/29/35% @2.1x (§94/§104). Predicts a crossover p* + gain that GROWS with scale.
   Controls: matched total tokens (incl execution); uniform @1.3x; shuffled-cost table. Kill: portfolio <= CTH at both sizes, or no crossover.
M7 Dense intermediate-verification GRPO reward (DAG): reward = frac correct nodes. Prediction: D_fail halves, in-dist reliability up, OOD flat (reweight, not N_dist).
   Either outcome helps the paper. Derived: §80b dead groups all-fail; §111 (RL coarse feedback channel); §112 killed value TARGETS not value REWARDS.
Ranking: M6 (~35%, award-shaped) > M1 (~55% modest) > M2 (~40%) > M3/M5 (~35-40%) > M7 (control) > M4 (~25%). Win = collapse curve holds + one of M1/M6 lands at predicted magnitude.
