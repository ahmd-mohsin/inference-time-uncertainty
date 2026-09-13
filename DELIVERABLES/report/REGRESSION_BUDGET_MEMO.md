# Regression-budget research protocol (external memo, 2026-09-13)
Supersedes composition/value-supervision/linear-headroom hypotheses. Core: constrain per-problem behavioral REGRESSION (F_eps = E[(q-p_theta-eps)_+] <= delta)
during RL from a shared RFT checkpoint; test if it beats RFT/VSF/ReMind/CoKL on the acquisition-retention-compute frontier.
GATE = H0 (hidden acquisition + ORACLE ENVELOPE O=E[max(q,p_RL)]). If O can't beat continued-RFT at matched cost, the retention-fix paper is dead.
H1 per-problem-budget vs aggregate-floor vs uniform-review vs VSF. H2 valid-solution freedom (code). H3 affordable risk estimator. H4 randomized
bank-availability (replaces linear-headroom law). H5 continued learning math<->code. Baselines: iterative RFT, RFT->GRPO, VSF, ExGRPO, ReMind, CoKL, aggregate-floor.
Novelty target: affordable PER-PROBLEM regression control (not replay/Lagrange/identity alone). Full memo in chat 2026-09-13.
