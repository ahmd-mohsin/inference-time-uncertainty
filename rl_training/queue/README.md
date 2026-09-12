# Memo-v8 experiment queue (award-strengthening; run on fresh nodes once env is bootstrapped)
Each qN is a self-contained sequential "queue" for one node ($HOME/gu paths, writes RESULTS_qN.md). Launch: `setsid nohup bash qN.sh &`.
- q1_method.sh  — NODE1 (memo W0/E1/Pareto): 7B-hard {base,GRPO,RFT,VSF,VSF-minus-PG}, eval OOD+in-dist. Decides "repair vs dominates".
- q2_dissoc.sh  — NODE2 (memo X-series): 1.5B-hard erosion-vs-acquisition {base,RFT,GRPO,X6-zeroneg,X3-KL,X1-disjoint-replay,X9-prior-support}.
- q3_scale.sh   — NODE3 (memo L4): 14B depth-16 high-headroom {base,GRPO,RFT,VSF} — tests §132 prereg (regression returns w/ headroom at 14B).
- node_bootstrap.sh — installs vllm==0.23.0 + trl==1.7.0 (MATCH old-node stack). BLOCKER: torch 2.11.0+cu130 needs the correct pip index-url
  (default PyPI stalls). Supply the original node bootstrap's `--index-url` (or copy ~/.local site-packages from a working old node).
VSF-minus-PG implemented via train_grpo `--vsf-pg-weight 0` (rl_training/vsf_trainer.py).

## WORKING bootstrap recipe (validated 2026-09-12 on fresh pytorch-base-24.12)
1. `pip install --break-system-packages --ignore-installed vllm==0.23.0` (pulls torch 2.11+cu130 from PyPI; ~3GB, be patient — site-packages stays ~210MB until torch extracts at the end).
2. `pip install --break-system-packages trl==1.7.0 peft deepspeed accelerate datasets sympy`
3. CRITICAL FIX: step1/2 pull transformers 5.17.0 which needs huggingface_hub>=1.5 (not installed) -> import breaks.
   `pip install --break-system-packages --force-reinstall transformers==4.57.6`  (consistent w/ vllm0.23 `>=4.56,!=5.0-5.5` + hf_hub 0.36.2 + tokenizers). fix_and_run.sh does this in a retry loop then launches the queue.
Gotchas: /tmp/instance_storage is root-owned (no sudo) -> use $HOME. Node DNS flaky -> retry pip. SSM tunnels drop ~30-40min -> re-tunnel to read (jobs run on-node).
