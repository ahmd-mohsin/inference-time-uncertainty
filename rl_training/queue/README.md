# Memo-v8 experiment queue (award-strengthening; run on fresh nodes once env is bootstrapped)
Each qN is a self-contained sequential "queue" for one node ($HOME/gu paths, writes RESULTS_qN.md). Launch: `setsid nohup bash qN.sh &`.
- q1_method.sh  — NODE1 (memo W0/E1/Pareto): 7B-hard {base,GRPO,RFT,VSF,VSF-minus-PG}, eval OOD+in-dist. Decides "repair vs dominates".
- q2_dissoc.sh  — NODE2 (memo X-series): 1.5B-hard erosion-vs-acquisition {base,RFT,GRPO,X6-zeroneg,X3-KL,X1-disjoint-replay,X9-prior-support}.
- q3_scale.sh   — NODE3 (memo L4): 14B depth-16 high-headroom {base,GRPO,RFT,VSF} — tests §132 prereg (regression returns w/ headroom at 14B).
- node_bootstrap.sh — installs vllm==0.23.0 + trl==1.7.0 (MATCH old-node stack). BLOCKER: torch 2.11.0+cu130 needs the correct pip index-url
  (default PyPI stalls). Supply the original node bootstrap's `--index-url` (or copy ~/.local site-packages from a working old node).
VSF-minus-PG implemented via train_grpo `--vsf-pg-weight 0` (rl_training/vsf_trainer.py).
