#!/usr/bin/env bash
#
# rl_all_arms.sh — orchestrate the 4-arm RL-expansion study across 4 nodes.
# RUN THIS ON THE MAIN NODE. It SSH-hops to each worker and launches one arm per node,
# each using that node's 8 GPUs. Storage is per-node (hostPath /mnt/nvme), so every node
# is self-contained; results are rsync'd back to main (then pulled to local dev).
#
# Node map (edit if the job changes):
#   main 10.3.145.117  -> base   (+ difficulty prepass, shared input)
#   w0   10.3.103.188  -> grpo
#   w1   10.3.197.81   -> oursA
#   w2   10.3.188.41   -> oursAB
set -uo pipefail

MODEL="${1:-Qwen/Qwen3-8B}"
DATASET="${2:-aime_all}"
MAIN_IP=10.3.145.117
declare -A NODE=( [grpo]=10.3.103.188 [oursA]=10.3.197.81 [oursAB]=10.3.188.41 )
SSH="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=8 -p 2222"

launch_remote () {  # $1=ip  $2=arm  $3=extra(diff json path or "")
  $SSH greenland-user@$1 "cd ~/inference-time-uncertainty && nohup bash scripts/rl_experiment.sh $2 $MODEL $DATASET $3 > ~/logs/rl_${2}.log 2>&1 & echo launched $2 on \$(hostname)"
}

echo "=== STEP 1: difficulty prepass on main (Component C input) ==="
cd ~/inference-time-uncertainty
mkdir -p ~/logs
python -m rl_training.difficulty_prepass --model-path "$MODEL" --dataset "$DATASET" \
  --k 64 --output rl_training/runs/difficulty.json > ~/logs/difficulty.log 2>&1
echo "difficulty done: $(tail -1 ~/logs/difficulty.log)"

# distribute difficulty.json to workers (per-node storage)
DIFF=rl_training/runs/difficulty.json
for ip in "${NODE[@]}"; do
  rsync -az -e "$SSH" "$DIFF" greenland-user@$ip:inference-time-uncertainty/rl_training/runs/ 2>/dev/null || true
done

echo "=== STEP 2: launch arms (base on main; grpo/oursA/oursAB on workers) ==="
nohup bash scripts/rl_experiment.sh base "$MODEL" "$DATASET" > ~/logs/rl_base.log 2>&1 &
echo "launched base on main"
launch_remote "${NODE[grpo]}"   grpo   "$DIFF"
launch_remote "${NODE[oursA]}"  oursA  "$DIFF"
launch_remote "${NODE[oursAB]}" oursAB "$DIFF"
echo "=== all 4 arms launched. monitor ~/logs/rl_*.log on each node ==="
