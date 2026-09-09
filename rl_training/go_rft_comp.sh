#!/usr/bin/env bash
# Iterative REJECTION-FINE-TUNING on a compositional pool (§64, one of the two training procedures for the 6-cell run).
# Loop R rounds from a common SFT init: merge -> vLLM sample K + verify (comp_gen) -> SFT on accepted -> repeat.
# Usage: go_rft_comp.sh <POOL.jsonl> <INIT_ADAPTER_DIR> <GPU> <ROUNDS> <TAG> [SEED]
set -uo pipefail
G=/tmp/instance_storage/gu
export HOME=/home/greenland-user PATH=$HOME/.local/bin:$PATH HF_HUB_DISABLE_XET=1 PYTHONPATH=$G/shim \
  WANDB_MODE=disabled GEN_GPU_MEM=0.55 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $HOME/inference-time-uncertainty
PY=/usr/bin/python3
POOL="$1"; INIT="$2"; GPU="$3"; ROUNDS="${4:-2}"; TAG="$5"; SEED="${6:-0}"
export CUDA_VISIBLE_DEVICES=$GPU
cur="$INIT"
echo "[rft_comp $TAG] init=$INIT pool=$POOL rounds=$ROUNDS gpu=$GPU $(date -u +%H:%M:%SZ)"
for r in $(seq 1 "$ROUNDS"); do
  merged=$($PY -c "from rl_training.model_utils import merge_adapter_if_needed as m; print(m('$cur'))" 2>>$G/logs/rft_${TAG}.log | tail -1)
  [ -z "$merged" ] && { echo "[rft_comp $TAG] merge failed r$r"; exit 1; }
  $PY -m rl_training.comp_gen --model "$merged" --pool "$POOL" --k 4 --n 400 --temperature 0.8 \
    --out $G/comp_data/accepted_${TAG}_r${r}.jsonl >>$G/logs/rft_${TAG}.log 2>&1
  na=$(wc -l < $G/comp_data/accepted_${TAG}_r${r}.jsonl 2>/dev/null || echo 0)
  echo "[rft_comp $TAG] round $r accepted=$na"
  [ "$na" -lt 5 ] && { echo "[rft_comp $TAG] too few accepted ($na) — stop"; break; }
  $PY -m rl_training.sft_train --model ${COMP_MODEL:-Qwen/Qwen2.5-Coder-1.5B-Instruct} --init-adapter "$cur" \
    --data $G/comp_data/accepted_${TAG}_r${r}.jsonl --out $G/rft_${TAG}_r${r} --max-steps 200 --seed $((SEED+r)) --bsz 4 \
    >>$G/logs/rft_${TAG}.log 2>&1 || { echo "[rft_comp $TAG] sft failed r$r"; exit 1; }
  cur=$G/rft_${TAG}_r${r}
done
ln -sfn "$cur" $G/rft_${TAG}_final
echo "[rft_comp $TAG] DONE -> $cur $(date -u +%H:%M:%SZ)"
