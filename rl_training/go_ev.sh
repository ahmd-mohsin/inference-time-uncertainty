#!/usr/bin/env bash
# Generic 8-GPU sharded eval launcher for cert_ablate / taco_cert (and any module taking
# --model-path --tag --shard-index --num-shards --output-dir + --merge). Clears GPUs first (run on a
# FREE node only). Usage: bash go_ev.sh <module> <model_or_ckpt> <TAG> "<EXTRA ARGS>"
#   e.g. bash go_ev.sh rl_training.cert_ablate /tmp/instance_storage/gu/m_qc ab_mbpp_qc "--bench mbpp --k 8"
#        bash go_ev.sh rl_training.taco_cert  /tmp/instance_storage/gu/m_qc tc_med_qc  "--difficulty MEDIUM --k 6 --max-problems 400"
set -uo pipefail
export HOME=/home/greenland-user; cd ${REPO:-/tmp/instance_storage/gu/repo}
export PATH=$HOME/.local/bin:$PATH HF_HUB_OFFLINE=0 HF_HUB_DISABLE_XET=1 EVAL_ENFORCE_EAGER=1 EVAL_GPU_MEM=0.85
[ -f $HOME/.hf_token ] && export HF_TOKEN=$(cat $HOME/.hf_token)
PY=/usr/bin/python3
MOD="$1"; MODEL="$2"; TAG="$3"; EXTRA="${4:-}"
NV=/tmp/instance_storage/gu; OUT=$NV/eval_out; LOGS=$NV/logs; mkdir -p "$OUT" "$LOGS"
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$pid" 2>/dev/null; done; sleep 4
getent hosts "$(hostname)" >/dev/null 2>&1 || sudo bash -c "echo \"127.0.0.1 $(hostname)\" >> /etc/hosts" || true
MM=$(HF_HUB_OFFLINE=1 $PY -c "import sys;sys.path.insert(0,'.');from rl_training.model_utils import merge_adapter_if_needed;print(merge_adapter_if_needed('$MODEL'))" 2>"$LOGS/merge_$TAG.log"|tail -1)
[ -f "$MM/config.json" ] || MM="$MODEL"
echo ">> EV $TAG mod=$MOD model=$MM extra=[$EXTRA]"
for g in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$g $PY -m "$MOD" --model-path "$MM" --tag "$TAG" --shard-index $g --num-shards 8 --output-dir "$OUT" $EXTRA > "$LOGS/${TAG}_s${g}.log" 2>&1 &
done
wait
$PY -m "$MOD" --merge --tag "$TAG" --num-shards 8 --output-dir "$OUT" 2>&1 | tee "$LOGS/${TAG}_merge.log"
touch "$OUT/EV_${TAG}_DONE"; echo ">> EV $TAG DONE"
