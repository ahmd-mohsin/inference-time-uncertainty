#!/usr/bin/env bash
# Trained-vs-base self-repair eval. Merges an adapter checkpoint into its base ONCE, then runs
# seq_recover (IID vs STATIC vs SEQUENTIAL-w/-feedback) 8-GPU data-parallel + merge.
# Usage: bash eval_seq.sh <ckpt_or_model_dir> <TAG> <BENCH> [DIAG] [T] [MAXPROB]
set -uo pipefail
export HOME=/home/greenland-user; cd ${REPO:-/tmp/instance_storage/gu/repo}
export PATH=$HOME/.local/bin:$PATH HF_HUB_OFFLINE=0 HF_HUB_DISABLE_XET=1 EVAL_ENFORCE_EAGER=1 EVAL_GPU_MEM=0.85
PY=/usr/bin/python3
CKPT="$1"; TAG="$2"; BENCH="$3"; DIAG="${4:-certificate}"; T="${5:-6}"; MAXP="${6:--1}"
NV=/tmp/instance_storage/gu; OUT=$NV/eval_out; LOGS=$NV/logs; mkdir -p "$OUT" "$LOGS"
getent hosts "$(hostname)" >/dev/null 2>&1 || sudo bash -c "echo \"127.0.0.1 $(hostname)\" >> /etc/hosts" || true
# Clear any leftover GPU procs (e.g. an orphaned vLLM from a prior training run holding GPU0) so all 8
# eval shards can init. SAFE: eval_seq only runs on a node with no active training. Kills by GPU compute-app
# PID (never a pkill -f pattern that could match this shell).
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$pid" 2>/dev/null; done
sleep 4
# merge adapter -> full model (idempotent) if CKPT is an adapter dir; else use as-is
MODEL=$(HF_HUB_OFFLINE=1 $PY -c "import sys;sys.path.insert(0,'.');from rl_training.model_utils import merge_adapter_if_needed;print(merge_adapter_if_needed('$CKPT'))" 2> "$LOGS/merge_${TAG}.log" | tail -1)
[ -f "$MODEL/config.json" ] || { echo "FATAL merge failed for $CKPT (got '$MODEL'); see $LOGS/merge_${TAG}.log"; tail -5 "$LOGS/merge_${TAG}.log"; exit 1; }
echo ">> EVAL $TAG model=$MODEL bench=$BENCH diag=$DIAG"
for g in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$g $PY -m rl_training.seq_recover --model-path "$MODEL" --bench "$BENCH" --tag "$TAG" \
    --shard-index $g --num-shards 8 --T "$T" --max-problems "$MAXP" --diag-mode "$DIAG" --output-dir "$OUT" \
    > "$LOGS/seq_${TAG}_s${g}.log" 2>&1 &
done
wait
$PY -m rl_training.seq_recover --merge --tag "$TAG" --num-shards 8 --output-dir "$OUT" 2>&1 | tee "$LOGS/seq_${TAG}_merge.log"
touch "$OUT/SEQ_${TAG}_DONE"; echo ">> EVAL $TAG DONE"
