#!/usr/bin/env bash
# Generic 8-GPU data-parallel pass@k eval on one node. Reconstructed (the original lived only on a
# dead node and was never committed — now tracked so it survives node death).
#
# Usage: bash go_eval2.sh <SPEC> <TAG> <DATASET> <DIFF_JSON> [MAXLEN]
#   SPEC:
#     ckpt:<HF_REPO>   fetch checkpoint-100 WEIGHTS only from an HF repo into a flat nvme dir
#     hf:<HF_ID>       download a full HF model id into a flat nvme dir
#     /abs/path        use a local model dir directly (e.g. r2_*/checkpoint-100)
#   TAG      output tag -> $OUT/passk_<TAG>.json
#   DATASET  evaluate_passk dataset name (e.g. omni_math_hard, olympiad_bench)
#   DIFF     difficulty-json path (subset-labels hard); "" to skip
#   MAXLEN   optional --max-new-tokens (default: evaluate_passk default)
#
# Matches the round-2 ceiling config used for qm/lloly cells: n_samples=1024, subset=hard, 8 shards.
set -uo pipefail
export HOME=/home/greenland-user
cd $HOME/inference-time-uncertainty
export PATH=$HOME/.local/bin:$PATH
export VLLM_ATTENTION_BACKEND=FLASHINFER
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_XET=1
[ -f $HOME/.hf_token ] && export HF_TOKEN=$(cat $HOME/.hf_token)
PY=/usr/bin/python3

SPEC="$1"; TAG="$2"; DATASET="$3"; DIFF="${4:-}"; MAXLEN="${5:-}"
NV=/tmp/instance_storage/gu
OUT="$NV/eval_out"; LOGS="$NV/logs"
mkdir -p "$OUT" "$LOGS"

# ---- resolve SPEC -> flat MODEL dir --------------------------------------------------------
case "$SPEC" in
  ckpt:*)
    REPO="${SPEC#ckpt:}"; MODEL="$NV/eval_${TAG}"
    if [ ! -f "$MODEL/model.safetensors" ] && [ ! -f "$MODEL/model.safetensors.index.json" ]; then
      mkdir -p "$MODEL"
      $PY - "$REPO" "$MODEL" <<'PY'
import os,sys,shutil
os.environ["HF_HUB_ENABLE_HF_TRANSFER"]="0"; os.environ["HF_HUB_DISABLE_XET"]="1"
from huggingface_hub import snapshot_download
repo, model = sys.argv[1], sys.argv[2]
tok=open(os.path.expanduser("~/.hf_token")).read().strip()
snapshot_download(repo, repo_type="model", local_dir=os.path.join(model,"dl"), token=tok,
  allow_patterns=["checkpoint-100/*.safetensors","checkpoint-100/*.json",
                  "checkpoint-100/*.jinja","checkpoint-100/tokenizer*"])
src=os.path.join(model,"dl","checkpoint-100")
for fn in os.listdir(src): shutil.move(os.path.join(src,fn), os.path.join(model,fn))
shutil.rmtree(os.path.join(model,"dl"), ignore_errors=True)
assert os.path.exists(os.path.join(model,"config.json")),"FETCH FAILED"
print("eval model ready:", model)
PY
    fi ;;
  hf:*)
    HFID="${SPEC#hf:}"; MODEL="$NV/eval_${TAG}"
    mkdir -p "$MODEL"
    # small files (configs, tokenizer, and any SHARDED weights) via snapshot; these download reliably
    $PY - "$HFID" "$MODEL" <<'PY'
import os,sys
os.environ["HF_HUB_ENABLE_HF_TRANSFER"]="0"; os.environ["HF_HUB_DISABLE_XET"]="1"
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], repo_type="model", local_dir=sys.argv[2],
  token=(open(os.path.expanduser("~/.hf_token")).read().strip() if os.path.exists(os.path.expanduser("~/.hf_token")) else None),
  allow_patterns=["config.json","generation_config.json","*.jinja","tokenizer*","vocab*","merges*",
                  "*.model","model.safetensors.index.json","model-*.safetensors"])
print("hf small files ready:", sys.argv[2])
PY
    # single-file model.safetensors: these nodes have a per-connection byte cap that freezes one long
    # 15GB stream (~1.35GB) — sharded models are fine (fresh conn/shard). Chunked segmented download.
    if [ ! -f "$MODEL/model.safetensors.index.json" ]; then
      echo ">> chunked-fetch model.safetensors via fetch_big.py"
      $PY -m rl_training.fetch_big "$HFID" model.safetensors "$MODEL/model.safetensors" 32
    fi ;;
  *)
    MODEL="$SPEC" ;;   # local path
esac
[ -f "$MODEL/config.json" ] || { echo "FATAL: model dir $MODEL not ready"; exit 1; }

# ---- self-clear this node's GPUs (kill stranded vLLM/EngineCore from prior runs) -----------
echo ">> self-clearing GPUs before launch"
pkill -9 -f "VLLM::EngineCore" 2>/dev/null
pkill -9 -f "vllm" 2>/dev/null
pkill -9 -f "evaluate_passk" 2>/dev/null
fuser -k /dev/nvidia* 2>/dev/null
# kill any lingering compute apps by pid
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
sleep 5

# ---- 8-GPU data-parallel pass@k ------------------------------------------------------------
MNT=""; [ -n "$MAXLEN" ] && MNT="--max-new-tokens $MAXLEN"
DFLAG=""; [ -n "$DIFF" ] && DFLAG="--difficulty-json $DIFF --subset-labels hard"
NSAMP="${N_SAMPLES:-1024}"     # override for OOD/full-set evals (256 is enough for coverage+extinction)
NPROB_FLAG=""; [ -n "${N_PROBLEMS:-}" ] && NPROB_FLAG="--n-problems $N_PROBLEMS"
for s in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$s HF_HUB_DISABLE_XET=1 VLLM_ATTENTION_BACKEND=FLASHINFER setsid nohup $PY -m rl_training.evaluate_passk \
    --model-path "$MODEL" --dataset "$DATASET" --n-samples $NSAMP $DFLAG $MNT $NPROB_FLAG \
    --tensor-parallel-size 1 --num-shards 8 --shard-index $s \
    --output-dir "$OUT" --tag "$TAG" \
    > "$LOGS/eval_${TAG}_shard${s}.log" 2>&1 &
done
echo ">> [$TAG] 8 eval shards launched (dataset=$DATASET model=$MODEL)"
wait
echo ">> [$TAG] shards done; merging"
$PY -m rl_training.evaluate_passk --merge --num-shards 8 --output-dir "$OUT" --tag "$TAG" > "$LOGS/eval_${TAG}_merge.log" 2>&1
echo ">> [$TAG] eval DONE -> $OUT/passk_${TAG}.json"
