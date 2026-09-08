#!/usr/bin/env bash
# Applied code experiment: fetch a coder model (base:HFID sharded snapshot | /localdir) then run
# code_passk (iid vs stratified pass@k with unit-test verification), 8-GPU data-parallel + merge.
# Usage: bash go_code.sh <SPEC> <TAG> <BENCH> [MAXPROB] [NIID] [NFORCED]
set -uo pipefail
export HOME=/home/greenland-user; cd $HOME/inference-time-uncertainty
export PATH=$HOME/.local/bin:$PATH HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_OFFLINE=0
export EVAL_ENFORCE_EAGER=1 EVAL_GPU_MEM=0.85
echo <HF_TOKEN — set locally in ~/.hf_token> > $HOME/.hf_token; export HF_TOKEN=$(cat $HOME/.hf_token)
PY=/usr/bin/python3; $PY -m pip uninstall -y hf-xet >/dev/null 2>&1 || true
SPEC="$1"; TAG="$2"; BENCH="$3"; MAXPROB="${4:--1}"; NIID="${5:-24}"; NFORCED="${6:-4}"
NV=/tmp/instance_storage/gu; OUT=$NV/eval_out; LOGS=$NV/logs; mkdir -p "$OUT" "$LOGS"
case "$SPEC" in
  /*) D="$SPEC" ;;
  base:*) H="${SPEC#base:}"; D=$NV/code_src_$TAG; mkdir -p "$D"
    $PY - "$H" "$D" <<'PY'
import os,sys; os.environ["HF_HUB_DISABLE_XET"]="1"; os.environ["HF_HUB_ENABLE_HF_TRANSFER"]="0"
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1],local_dir=sys.argv[2],token=open(os.path.expanduser("~/.hf_token")).read().strip(),
  allow_patterns=["*.json","*.txt","*.model","tokenizer*","vocab*","merges*","*.jinja","model-*.safetensors","model.safetensors","*.index.json"])
print("fetched")
PY
    ;;
  *) echo "bad SPEC $SPEC"; exit 1;;
esac
[ -f "$D/config.json" ] || { echo "FATAL: model not at $D"; exit 1; }
for g in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$g $PY -m rl_training.code_passk --model-path "$D" --bench "$BENCH" --tag "$TAG" \
    --shard-index $g --num-shards 8 --max-problems "$MAXPROB" --n-iid "$NIID" --n-forced "$NFORCED" \
    --output-dir "$OUT" > "$LOGS/code_${TAG}_s${g}.log" 2>&1 &
done
wait
$PY -m rl_training.code_passk --merge --tag "$TAG" --num-shards 8 --output-dir "$OUT" 2>&1 | tee "$LOGS/code_${TAG}_merge.log"
touch "$OUT/CODE_${TAG}_DONE"; echo ">> CODE $TAG COMPLETE"
