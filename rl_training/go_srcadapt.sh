#!/usr/bin/env bash
# Fetch a source model from HF (base=sharded snapshot | fork=single-file chunked fetch_big) then run the
# adaptation-shock (go_adapt) on a SHIFT dataset. Reward-vs-step = adaptation curve. Uses only durable
# HF models (no dependence on stranded local checkpoints).
# Usage: bash go_srcadapt.sh <base:HFID | fork:REPO> <tag> <shift_dataset> [steps]
set -uo pipefail
export HOME=/home/greenland-user
cd $HOME/inference-time-uncertainty
export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
echo <HF_TOKEN_REDACTED> > $HOME/.hf_token
PY=/usr/bin/python3
SPEC="$1"; TAG="$2"; DATASET="$3"; STEPS="${4:-80}"
NV=/tmp/instance_storage/gu; D=$NV/src_$TAG; mkdir -p $D $NV/logs
$PY -m pip uninstall -y hf-xet >/dev/null 2>&1 || true

case "$SPEC" in
  base:*) HFID="${SPEC#base:}"
    $PY - "$HFID" "$D" <<'PY'
import os,sys; os.environ["HF_HUB_DISABLE_XET"]="1"; os.environ["HF_HUB_ENABLE_HF_TRANSFER"]="0"
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1],local_dir=sys.argv[2],
  allow_patterns=["*.json","*.txt","*.model","tokenizer*","vocab*","merges*","model-*.safetensors","model.safetensors"])
print("base fetched")
PY
    ;;
  fork:*) REPO="${SPEC#fork:}"
    # small files
    $PY - "$REPO" "$D" <<'PY'
import os,sys; os.environ["HF_HUB_DISABLE_XET"]="1"
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1],local_dir=sys.argv[2],token=open(os.path.expanduser("~/.hf_token")).read().strip(),
  allow_patterns=["config.json","generation_config.json","*.jinja","tokenizer*"])
PY
    # single-file weights via chunked fetch (defeats per-connection cap)
    SZ=$($PY - "$REPO" <<'PY'
import os,sys; os.environ["HF_HUB_DISABLE_XET"]="1"
from huggingface_hub import HfApi
info=HfApi(token=open(os.path.expanduser("~/.hf_token")).read().strip()).model_info(sys.argv[1],files_metadata=True)
print(next((s.size for s in info.siblings if s.rfilename=="model.safetensors"),0))
PY
)
    $PY -u -m rl_training.fetch_big "$REPO" model.safetensors "$D/model.safetensors" 32 "$SZ"
    ;;
  *) echo "bad SPEC $SPEC"; exit 1 ;;
esac
[ -f "$D/config.json" ] || { echo "FATAL: src not fetched"; exit 1; }
echo ">> source ready $D ; launching adaptation on $DATASET"
bash $HOME/inference-time-uncertainty/rl_training/go_adapt.sh "$D" "$TAG" "$DATASET" "$STEPS"
