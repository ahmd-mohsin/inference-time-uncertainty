#!/usr/bin/env bash
# Counterfactual Strategy Disagreement (diagnostic diversity) experiment. Fetch model (base:snapshot |
# fork:fetch_big) then run cf_disagree (iid + forced-per-strategy, save answers, error-AUROC). 8-GPU DP + merge.
# Usage: bash go_cf.sh <SPEC> <TAG> <DATASET> [MAXPROB] [NIID] [NFORCED]
set -uo pipefail
export HOME=/home/greenland-user; cd $HOME/inference-time-uncertainty
export PATH=$HOME/.local/bin:$PATH HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_OFFLINE=0
export EVAL_ENFORCE_EAGER=1 EVAL_GPU_MEM=0.85
echo <HF_TOKEN — set locally in ~/.hf_token> > $HOME/.hf_token; export HF_TOKEN=$(cat $HOME/.hf_token)
PY=/usr/bin/python3; $PY -m pip uninstall -y hf-xet >/dev/null 2>&1 || true
SPEC="$1"; TAG="$2"; DATASET="$3"; MAXPROB="${4:-200}"; NIID="${5:-16}"; NFORCED="${6:-3}"
NV=/tmp/instance_storage/gu; OUT=$NV/eval_out; LOGS=$NV/logs; mkdir -p "$OUT" "$LOGS"
case "$SPEC" in
  /*) D="$SPEC" ;;
  base:*) H="${SPEC#base:}"; D=$NV/cf_src_$TAG; mkdir -p "$D"
    $PY - "$H" "$D" <<'PY'
import os,sys;os.environ["HF_HUB_DISABLE_XET"]="1";os.environ["HF_HUB_ENABLE_HF_TRANSFER"]="0"
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1],local_dir=sys.argv[2],token=open(os.path.expanduser("~/.hf_token")).read().strip(),
 allow_patterns=["*.json","*.txt","*.model","tokenizer*","vocab*","merges*","*.jinja","model-*.safetensors","model.safetensors","*.index.json"])
PY
    ;;
  fork:*) R="${SPEC#fork:}"; D=$NV/cf_src_$TAG; mkdir -p "$D"
    $PY - "$R" "$D" <<'PY'
import os,sys;os.environ["HF_HUB_DISABLE_XET"]="1"
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1],local_dir=sys.argv[2],token=open(os.path.expanduser("~/.hf_token")).read().strip(),
 allow_patterns=["config.json","generation_config.json","*.jinja","tokenizer*","vocab*","merges*"])
PY
    SZ=$($PY - "$R" <<'PY'
import os,sys;os.environ["HF_HUB_DISABLE_XET"]="1"
from huggingface_hub import HfApi
print(next((x.size for x in HfApi(token=open(os.path.expanduser("~/.hf_token")).read().strip()).model_info(sys.argv[1],files_metadata=True).siblings if x.rfilename=="model.safetensors"),0))
PY
)
    $PY -u -m rl_training.fetch_big "$R" model.safetensors "$D/model.safetensors" 32 "$SZ" ;;
  *) echo "bad SPEC"; exit 1;;
esac
[ -f "$D/config.json" ] || { echo "FATAL: model not at $D"; exit 1; }
for g in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$g $PY -m rl_training.cf_disagree --model-path "$D" --dataset "$DATASET" --tag "$TAG" \
    --shard-index $g --num-shards 8 --max-problems "$MAXPROB" --n-iid "$NIID" --n-forced "$NFORCED" \
    --output-dir "$OUT" > "$LOGS/cf_${TAG}_s${g}.log" 2>&1 &
done
wait
$PY -m rl_training.cf_disagree --merge --tag "$TAG" --num-shards 8 --output-dir "$OUT" 2>&1 | tee "$LOGS/cf_${TAG}_merge.log"
touch "$OUT/CF_${TAG}_DONE"; echo ">> CF $TAG COMPLETE"
