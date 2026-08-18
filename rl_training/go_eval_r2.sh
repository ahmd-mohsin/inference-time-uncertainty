#!/usr/bin/env bash
# ROUND-2 pass@k ceiling eval on one node (8 GPUs, data-parallel shards). Fetches an r2 arm's
# checkpoint-100 MODEL WEIGHTS from HF (eval needs no optimizer) into nvme, then runs sharded
# pass@k on the fragile/hard Olympiad band (n_samples=1024) matching the round-1 config
# (n_problems=329 via difficulty-json + subset-labels hard). Merges shards -> one result JSON.
# Usage: bash go_eval_r2.sh <grpo|floor>
set -uo pipefail
export HOME=/home/greenland-user
cd $HOME/inference-time-uncertainty
export PATH=$HOME/.local/bin:$PATH
export VLLM_ATTENTION_BACKEND=FLASHINFER
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_XET=1
export HF_TOKEN=$(cat $HOME/.hf_token)
PY=/usr/bin/python3
FORK="$1"
NV=/tmp/instance_storage/gu
REPO="muahmed7338/cov-r2-from-${FORK}-7b"
MODEL="$NV/eval_r2_${FORK}"          # flat model dir (weights only) for eval
DIFF=$HOME/inference-time-uncertainty/rl_training/runs/prepass/difficulty_olympiad_7b.json
OUT="$NV/eval_out"
LOGS="$NV/logs"
mkdir -p "$MODEL" "$OUT" "$LOGS"

# 1) fetch ckpt-100 weights (skip global_step optimizer) -> flat MODEL dir
if [ ! -f "$MODEL/model.safetensors" ]; then
  $PY - <<PY
import os,shutil
os.environ["HF_HUB_ENABLE_HF_TRANSFER"]="0"; os.environ["HF_HUB_DISABLE_XET"]="1"
from huggingface_hub import snapshot_download
tok=open(os.path.expanduser("~/.hf_token")).read().strip()
snapshot_download("$REPO", repo_type="model", local_dir="$MODEL/dl", token=tok,
  allow_patterns=["checkpoint-100/*.safetensors","checkpoint-100/*.json",
                  "checkpoint-100/*.jinja","checkpoint-100/tokenizer*"])
src=os.path.join("$MODEL","dl","checkpoint-100")
for fn in os.listdir(src): shutil.move(os.path.join(src,fn), os.path.join("$MODEL",fn))
shutil.rmtree(os.path.join("$MODEL","dl"), ignore_errors=True)
assert os.path.exists(os.path.join("$MODEL","config.json")) and os.path.exists(os.path.join("$MODEL","model.safetensors")),"FETCH FAILED"
print("eval model ready:", "$MODEL")
PY
fi
[ -f "$MODEL/config.json" ] || { echo "FATAL: $MODEL not ready"; exit 1; }

# 2) sharded pass@k across 8 GPUs (1 GPU/shard), fragile hard band, n_samples=1024
TAG="r2_${FORK}_frag"
for s in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$s HF_HUB_OFFLINE=1 VLLM_ATTENTION_BACKEND=FLASHINFER setsid nohup $PY -m rl_training.evaluate_passk \
    --model-path "$MODEL" --dataset olympiad_bench --n-samples 1024 \
    --difficulty-json "$DIFF" --subset-labels hard \
    --tensor-parallel-size 1 --num-shards 8 --shard-index $s \
    --output-dir "$OUT" --tag "$TAG" \
    > "$LOGS/eval_${FORK}_shard${s}.log" 2>&1 &
done
echo ">> [$FORK] 8 eval shards launched (tag=$TAG). Merge with: $PY -m rl_training.evaluate_passk --merge --num-shards 8 --output-dir $OUT --tag $TAG"
wait
echo ">> [$FORK] shards done; merging"
$PY -m rl_training.evaluate_passk --merge --num-shards 8 --output-dir "$OUT" --tag "$TAG" > "$LOGS/eval_${FORK}_merge.log" 2>&1
echo ">> [$FORK] eval DONE -> $OUT/passk_${TAG}.json"
