#!/usr/bin/env bash
# Routing-vs-competence probe on ONE node, 8-GPU data-parallel (one shard/GPU) + merge.
# Fetches the policy model to COMPLETENESS (snapshot for base | fetch_big for single-file forks —
# stops the silent vLLM-on-incomplete-model hang), then runs strategy_probe across 8 shards.
# Usage: bash go_probe.sh <SPEC> <TAG> <DATASET> [MAXPROB] [DEFAULT_N] [FORCED_N]
#   SPEC: /abs/path (already-fetched local dir) | base:HFID | fork:REPO
set -uo pipefail
export HOME=/home/greenland-user
cd $HOME/inference-time-uncertainty
export PATH=$HOME/.local/bin:$PATH
export VLLM_ATTENTION_BACKEND=FLASHINFER HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
export EVAL_ENFORCE_EAGER=1 EVAL_GPU_MEM=0.85
export HF_HUB_OFFLINE=0
echo <HF_TOKEN_REDACTED> > $HOME/.hf_token
export HF_TOKEN=$(cat $HOME/.hf_token)
grep -q "$(hostname)" /etc/hosts || echo "127.0.0.1 $(hostname)" | sudo tee -a /etc/hosts >/dev/null 2>&1 || true
PY=/usr/bin/python3
$PY -m pip uninstall -y hf-xet >/dev/null 2>&1 || true

SPEC="$1"; TAG="$2"; DATASET="$3"; MAXPROB="${4:-150}"; DN="${5:-8}"; FN="${6:-4}"; FORCE="${7:-prefix}"
NV=/tmp/instance_storage/gu; OUT=$NV/eval_out; LOGS=$NV/logs; mkdir -p "$OUT" "$LOGS"

# ---- resolve SPEC -> complete local model dir D ----
case "$SPEC" in
  /*) D="$SPEC" ;;
  base:*) HFID="${SPEC#base:}"; D=$NV/probe_src_$TAG; mkdir -p "$D"
    $PY - "$HFID" "$D" <<'PY'
import os,sys; os.environ["HF_HUB_DISABLE_XET"]="1"; os.environ["HF_HUB_ENABLE_HF_TRANSFER"]="0"
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1],local_dir=sys.argv[2],
  allow_patterns=["*.json","*.txt","*.model","tokenizer*","vocab*","merges*","model-*.safetensors","model.safetensors"])
print("base fetched")
PY
    ;;
  fork:*) REPO="${SPEC#fork:}"; D=$NV/probe_src_$TAG; mkdir -p "$D"
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
  *) echo "bad SPEC $SPEC"; exit 1 ;;
esac
[ -f "$D/config.json" ] || { echo "FATAL: model not fetched at $D"; exit 1; }
echo ">> probe model ready: $D"

# ---- 8-GPU data-parallel probe ----
N=8
for g in $(seq 0 $((N-1))); do
  CUDA_VISIBLE_DEVICES=$g $PY -m rl_training.strategy_probe \
    --model-path "$D" --dataset "$DATASET" --tag "$TAG" \
    --shard-index $g --num-shards $N --max-problems "$MAXPROB" \
    --default-n "$DN" --forced-n "$FN" --force-mode "$FORCE" --output-dir "$OUT" \
    > "$LOGS/probe_${TAG}_s${g}.log" 2>&1 &
done
wait
echo ">> all shards done; merging"
$PY -m rl_training.strategy_probe --merge --tag "$TAG" --num-shards $N --output-dir "$OUT" \
  2>&1 | tee "$LOGS/probe_${TAG}_merge.log"
touch "$OUT/PROBE_${TAG}_DONE"
echo ">> PROBE $TAG COMPLETE -> $OUT/probe_${TAG}.json"
