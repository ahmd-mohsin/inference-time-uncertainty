#!/usr/bin/env bash
# Fetch a model (base:/fork:/ckpt:/path) then run FREE-generation routing (behavioral rho via
# LLM-judge). Self-contained per node. Usage: bash go_ff.sh <SPEC> <TAG> <DATASET> [MAXPROB] [NSAMPLES]
set -uo pipefail
export HOME=/home/greenland-user; cd $HOME/inference-time-uncertainty
export PATH=$HOME/.local/bin:$PATH HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_OFFLINE=0
echo <HF_TOKEN_REDACTED> > $HOME/.hf_token; export HF_TOKEN=$(cat $HOME/.hf_token)
grep -q "$(hostname)" /etc/hosts || echo "127.0.0.1 $(hostname)" | sudo tee -a /etc/hosts >/dev/null 2>&1 || true
PY=/usr/bin/python3; $PY -m pip uninstall -y hf-xet >/dev/null 2>&1 || true
SPEC="$1"; TAG="$2"; DATASET="$3"; MAXPROB="${4:-150}"; NS="${5:-64}"
NV=/tmp/instance_storage/gu; mkdir -p $NV/logs $NV/eval_out
DL(){ $PY - "$@" <<'PY'
import os,sys; os.environ["HF_HUB_DISABLE_XET"]="1"; os.environ["HF_HUB_ENABLE_HF_TRANSFER"]="0"
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1],local_dir=sys.argv[2],token=open(os.path.expanduser("~/.hf_token")).read().strip(),allow_patterns=sys.argv[3].split(","))
PY
}
SZ(){ $PY - "$1" "$2" <<'PY'
import os,sys; os.environ["HF_HUB_DISABLE_XET"]="1"
from huggingface_hub import HfApi
i=HfApi(token=open(os.path.expanduser("~/.hf_token")).read().strip()).model_info(sys.argv[1],files_metadata=True)
print(next((x.size for x in i.siblings if x.rfilename==sys.argv[2]),0))
PY
}
case "$SPEC" in
  /*) D="$SPEC" ;;
  base:*) H="${SPEC#base:}"; D=$NV/ff_src_$TAG; mkdir -p "$D"
    DL "$H" "$D" "*.json,*.txt,*.model,tokenizer*,vocab*,merges*,*.jinja,configuration*,*.py,model-*.safetensors,model.safetensors,*.index.json"
    # if sharded .bin (deepseek etc.), fetch each bin via fetch_big
    if [ ! -f "$D/model.safetensors" ] && [ ! -f "$D/model.safetensors.index.json" ]; then
      for B in $($PY - "$H" <<'PY'
import os,sys; os.environ["HF_HUB_DISABLE_XET"]="1"
from huggingface_hub import HfApi
i=HfApi(token=open(os.path.expanduser("~/.hf_token")).read().strip()).model_info(sys.argv[1],files_metadata=True)
print(" ".join(x.rfilename for x in i.siblings if x.rfilename.endswith(".bin")))
PY
); do $PY -u -m rl_training.fetch_big "$H" "$B" "$D/$B" 32 "$(SZ "$H" "$B")"; done
    fi ;;
  fork:*) R="${SPEC#fork:}"; D=$NV/ff_src_$TAG; mkdir -p "$D"
    DL "$R" "$D" "config.json,generation_config.json,*.jinja,tokenizer*,vocab*,merges*"
    $PY -u -m rl_training.fetch_big "$R" model.safetensors "$D/model.safetensors" 32 "$(SZ "$R" model.safetensors)" ;;
  *) echo "bad SPEC $SPEC"; exit 1 ;;
esac
[ -f "$D/config.json" ] || { echo "FATAL: model not at $D"; exit 1; }
echo ">> FF model ready $D ; free-routing on $DATASET"
bash $HOME/inference-time-uncertainty/rl_training/go_free.sh "$D" "$TAG" "$DATASET" "$MAXPROB" "$NS"
