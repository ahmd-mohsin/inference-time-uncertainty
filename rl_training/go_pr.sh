#!/usr/bin/env bash
# Combined probe (competence c, prefix-forced) + routing (rho, Delta log rho) on ONE node.
# Fetches the model to completeness, then runs go_probe (prefix) and go_route back-to-back so a
# single node yields BOTH sides of pi = rho * c for one (family, policy) or one checkpoint.
# Usage: bash go_pr.sh <SPEC> <TAG> <DATASET> [MAXPROB]
#   SPEC: base:HFID (sharded) | fork:REPO (single-file 15G) | ckpt:REPO@checkpoint-N (subdir single-file) | /abs/path
set -uo pipefail
export HOME=/home/greenland-user
cd $HOME/inference-time-uncertainty
export PATH=$HOME/.local/bin:$PATH
export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_OFFLINE=0
echo <HF_TOKEN_REDACTED> > $HOME/.hf_token
export HF_TOKEN=$(cat $HOME/.hf_token)
grep -q "$(hostname)" /etc/hosts || echo "127.0.0.1 $(hostname)" | sudo tee -a /etc/hosts >/dev/null 2>&1 || true
PY=/usr/bin/python3
$PY -m pip uninstall -y hf-xet >/dev/null 2>&1 || true
SPEC="$1"; TAG="$2"; DATASET="$3"; MAXPROB="${4:-150}"
NV=/tmp/instance_storage/gu; mkdir -p $NV/logs $NV/eval_out

case "$SPEC" in
  /*) D="$SPEC" ;;
  base:*) HFID="${SPEC#base:}"; D=$NV/pr_src_$TAG; mkdir -p "$D"
    $PY - "$HFID" "$D" <<'PY'
import os,sys; os.environ["HF_HUB_DISABLE_XET"]="1"; os.environ["HF_HUB_ENABLE_HF_TRANSFER"]="0"
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1],local_dir=sys.argv[2],token=open(os.path.expanduser("~/.hf_token")).read().strip(),
  allow_patterns=["*.json","*.txt","*.model","tokenizer*","vocab*","merges*","*.jinja","model-*.safetensors","model.safetensors"])
print("base fetched")
PY
    ;;
  fork:*) REPO="${SPEC#fork:}"; D=$NV/pr_src_$TAG; mkdir -p "$D"
    $PY - "$REPO" "$D" <<'PY'
import os,sys; os.environ["HF_HUB_DISABLE_XET"]="1"
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1],local_dir=sys.argv[2],token=open(os.path.expanduser("~/.hf_token")).read().strip(),
  allow_patterns=["config.json","generation_config.json","*.jinja","tokenizer*","vocab*","merges*"])
PY
    SZ=$($PY - "$REPO" <<'PY'
import os,sys; os.environ["HF_HUB_DISABLE_XET"]="1"
from huggingface_hub import HfApi
info=HfApi(token=open(os.path.expanduser("~/.hf_token")).read().strip()).model_info(sys.argv[1],files_metadata=True)
print(next((s.size for s in info.siblings if s.rfilename=="model.safetensors"),0))
PY
)
    $PY -u -m rl_training.fetch_big "$REPO" model.safetensors "$D/model.safetensors" 32 "$SZ"
    ;;
  ckpt:*) rest="${SPEC#ckpt:}"; REPO="${rest%@*}"; CK="${rest#*@}"; D=$NV/pr_src_$TAG; mkdir -p "$D"
    $PY - "$REPO" "$D" "$CK" <<'PY'
import os,sys; os.environ["HF_HUB_DISABLE_XET"]="1"
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1],local_dir=sys.argv[2],token=open(os.path.expanduser("~/.hf_token")).read().strip(),
  allow_patterns=[f"{sys.argv[3]}/config.json",f"{sys.argv[3]}/generation_config.json",
                  f"{sys.argv[3]}/*.jinja",f"{sys.argv[3]}/tokenizer*",f"{sys.argv[3]}/vocab*",f"{sys.argv[3]}/merges*"])
PY
    SZ=$($PY - "$REPO" "$CK" <<'PY'
import os,sys; os.environ["HF_HUB_DISABLE_XET"]="1"
from huggingface_hub import HfApi
info=HfApi(token=open(os.path.expanduser("~/.hf_token")).read().strip()).model_info(sys.argv[1],files_metadata=True)
print(next((s.size for s in info.siblings if s.rfilename==f"{sys.argv[2]}/model.safetensors"),0))
PY
)
    $PY -u -m rl_training.fetch_big "$REPO" "$CK/model.safetensors" "$D/$CK/model.safetensors" 32 "$SZ"
    D="$D/$CK"
    ;;
  *) echo "bad SPEC $SPEC"; exit 1 ;;
esac
[ -f "$D/config.json" ] || { echo "FATAL: model not at $D"; exit 1; }
echo ">> PR model ready: $D ; probe(prefix) then route on $DATASET"
bash $HOME/inference-time-uncertainty/rl_training/go_probe.sh "$D" "$TAG" "$DATASET" "$MAXPROB" 8 4 prefix
bash $HOME/inference-time-uncertainty/rl_training/go_route.sh "$D" "$TAG" "$DATASET" "$MAXPROB"
echo ">> PR $TAG COMPLETE (probe_$TAG.json + route_$TAG.json)"
