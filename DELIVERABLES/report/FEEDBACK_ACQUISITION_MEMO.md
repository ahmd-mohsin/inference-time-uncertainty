# Verified self-training via informative feedback acquisition (external memo, 2026-09-13)
CLOSE regression-budget branch (done §138/§139); preserve v10 as a characterization+measurement paper.
FUND ONE gated acquisition experiment: active feedback SELECTION (query-by-committee on failed attempts: sample K candidate programs, build M valid
probe inputs, pick high-DISAGREEMENT inputs, query trusted oracle for expected output, feedback-update via SDPO, accept only after full verify, then RFT).
Claim target: more TRANSFERABLE verified experience per unit compute than RFT + rich-feedback baselines (SDPO/RESD/repair/CodeIt). NOT beating GRPO.
GATES (stop at first fail): A0 does extra verified experience help the recipient at all? (RFT vs RFT-bigger-bank). A1 does trusted feedback help
the zero-success cohort? (SDPO real vs placebo). A2 does query SELECTION beat random? A3 does it TRANSFER through RFT to a held-out recipient? A4 scale+blind-spots (7B, 2nd family, committee misspecification).
Honest scope: "RFT generally leads, near-parity at 7B" (not literal ceiling). Report pass@1 vs pass@k separately; sampling-noise churn control; matched recipient tokens + unique-prompt coverage; all oracle/candidate-exec costs charged. Full memo in chat 2026-09-13.
