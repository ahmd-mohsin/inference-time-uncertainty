#!/usr/bin/env bash
# One OOD arm end-to-end on a node: fetch a fork/base model robustly (small files via snapshot,
# single 15GB safetensors via chunked curl fetch_big — nodes cap single long streams), then run
# the 8-GPU pass@k eval on the OOD dataset. Resumable: fetch_big appends from whatever's on disk.
# Usage: go_ood.sh <hf_repo_id> <tag> <dataset> <total_bytes> [n_samples]
set -uo pipefail
export HOME=/home/greenland-user
cd $HOME/inference-time-uncertainty
export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
echo <HF_TOKEN_REDACTED> > $HOME/.hf_token
REPO="$1"; TAG="$2"; DATASET="$3"; TOTAL="${4:-0}"; NS="${5:-256}"
D=/tmp/instance_storage/gu/eval_$TAG
LOG=/tmp/instance_storage/gu/logs
mkdir -p "$D" "$LOG"
PY=/usr/bin/python3

# 1) small files (configs/tokenizer + any sharded weights)
$PY - "$REPO" "$D" <<'PY'
import os,sys; os.environ["HF_HUB_DISABLE_XET"]="1"; os.environ["HF_HUB_ENABLE_HF_TRANSFER"]="0"
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], local_dir=sys.argv[2],
  token=open(os.path.expanduser("~/.hf_token")).read().strip(),
  allow_patterns=["config.json","generation_config.json","*.jinja","tokenizer*","vocab*","merges*",
                  "*.model","model.safetensors.index.json","model-*.safetensors"])
print("small files ready")
PY

# 2) single-file model.safetensors via chunked fetch (skip if sharded)
if [ ! -f "$D/model.safetensors.index.json" ]; then
  echo ">> chunked fetch model.safetensors (total=$TOTAL)"
  $PY -u -m rl_training.fetch_big "$REPO" model.safetensors "$D/model.safetensors" 32 "$TOTAL"
fi

# 3) eval on OOD dataset (local model path; N_SAMPLES from arg)
export N_SAMPLES="$NS"
echo ">> launching eval $TAG on $DATASET"
bash $HOME/inference-time-uncertainty/rl_training/go_eval2.sh "$D" "$TAG" "$DATASET" ""
