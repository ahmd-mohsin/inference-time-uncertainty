#!/usr/bin/env bash
# E3-unbiased: merge held-out E2 bank -> strategy-cluster -> teacher-force base/grpo/floor (parallel
# on GPU 0/1/2) -> per-trace logprobs. Join+certify done on laptop. Death-tolerant: each stage writes
# to nvme. Usage: bash go_e3_unbiased.sh
set -uo pipefail
export HOME=/home/greenland-user
cd $HOME/inference-time-uncertainty
export PATH=$HOME/.local/bin:$PATH
PY=/usr/bin/python3
NV=/tmp/instance_storage/gu; LOGS=$NV/logs; mkdir -p "$LOGS"
MERGED=$NV/bank_e2_olympiad.jsonl
CLUST=$NV/bank_e2_clustered.jsonl

# 1) merge shards
cat $NV/bank_e2_olympiad_shard*.jsonl > "$MERGED"
echo ">> merged $(wc -l < "$MERGED") witnesses -> $MERGED"

# 2) strategy-cluster -> add mode_id
$PY -m rl_training.strategy_bank cluster --bank "$MERGED" --dist-thresh 0.15 \
    --emit-bank "$CLUST" --out "$NV/E2_modes_heldout.json" 2>&1 | tail -3

# 3) score base / grpo / floor in parallel (each single-GPU, different device)
declare -A MODELS=( [base]=$NV/base_qwen25math7b [grpo]=$NV/eval_r2_grpo [floor]=$NV/eval_r2_floor )
gpu=0
for arm in base grpo floor; do
  CUDA_VISIBLE_DEVICES=$gpu HF_HUB_OFFLINE=1 setsid nohup $PY rl_training/score_bank_logprobs.py \
    --model "${MODELS[$arm]}" --bank "$CLUST" --out "$NV/scored_hb_${arm}.jsonl" --field logp_theta \
    > "$LOGS/score_hb_${arm}.log" 2>&1 &
  echo ">> scoring $arm on GPU$gpu (pid $!)"
  gpu=$((gpu+1))
done
wait
echo ">> E3-unbiased scoring DONE:"
for arm in base grpo floor; do echo "   $arm=$(wc -l < $NV/scored_hb_${arm}.jsonl 2>/dev/null) lines"; done
