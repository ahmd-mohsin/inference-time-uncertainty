#!/usr/bin/env bash
# Generic diversity-cell pipeline (one node, 8 GPUs). For a (model, dataset) it runs the full chain:
#   download base -> difficulty prepass (8-shard+merge) -> base-correct bank (8-shard) + ref_logprob
#   -> train PLAIN GRPO -> train expSR (frozen method) -> score both over the bank.
# Shows the core claim per cell: plain collapses base modes, expSR preserves them.
# Usage: bash go_cell.sh <MODEL_ID> <NAME> <DATASET> [SUBSET=hard] [K=64] [NPROB=-1] [STEPS=150]
# Death-tolerant: stage markers in $CELL; re-run resumes at first incomplete stage.
set -uo pipefail
export HOME=/home/greenland-user
cd $HOME/inference-time-uncertainty
export PATH=$HOME/.local/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DS_SKIP_CUDA_CHECK=1
PY=/usr/bin/python3
MODEL_ID="${1:?model id}"; NAME="${2:?name}"; DATASET="${3:?dataset}"
SUBSET="${4:-hard}"; K="${5:-64}"; NPROB="${6:--1}"; STEPS="${7:-150}"
NV=/tmp/instance_storage/gu; LOGS=$NV/logs; mkdir -p "$LOGS"
BASE=$NV/base_$NAME
CELL=$NV/cell_${NAME}_${DATASET}; mkdir -p "$CELL"
DIFF=$CELL/difficulty.json
BANKRAW=$CELL/bankraw.jsonl; BASESCORED=$CELL/base_scored.jsonl; BANK=$CELL/bank.jsonl
L=$LOGS/cell_${NAME}_${DATASET}.log
say(){ echo "$(date -u +%H:%M:%SZ) $*" >> "$L"; }
say "=== CELL START $NAME x $DATASET (subset=$SUBSET k=$K nprob=$NPROB steps=$STEPS) on $(hostname) ==="

freegpu(){ for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 $pid 2>/dev/null; done; pkill -9 -f trl.scripts.vllm_serve 2>/dev/null; sleep 5; }

# ---- S0: base model ----
if [ ! -f "$BASE/config.json" ]; then
  say "S0 downloading $MODEL_ID -> $BASE"
  $PY - "$MODEL_ID" "$BASE" <<'PY' >> "$L" 2>&1
import sys,os
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], local_dir=sys.argv[2], token=os.environ.get("HF_TOKEN"),
  allow_patterns=["*.json","*.safetensors","*.txt","tokenizer*","vocab*","merges*","*.model","*.bin"])
print("BASE_DL_DONE")
PY
fi
[ -f "$BASE/config.json" ] || { say "FATAL no base"; exit 1; }

# ---- S1: difficulty prepass (8 shards -> merge) ----
if [ ! -f "$CELL/PREPASS_DONE" ]; then
  say "S1 prepass k=$K"
  freegpu
  for i in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$i HF_HUB_OFFLINE=1 setsid nohup $PY rl_training/difficulty_prepass.py \
      --model-path "$BASE" --dataset "$DATASET" --n-problems "$NPROB" --k "$K" \
      --num-shards 8 --shard-index $i --output "$DIFF" > "$LOGS/prep_${NAME}_${DATASET}_$i.log" 2>&1 &
  done
  wait
  $PY rl_training/difficulty_prepass.py --merge --num-shards 8 --output "$DIFF" --dataset "$DATASET" --k "$K" >> "$L" 2>&1
  [ -f "$DIFF" ] && touch "$CELL/PREPASS_DONE" && say "S1 done: $(grep -o '\"counts\":[^}]*}' $DIFF | head -1)"
fi
[ -f "$CELL/PREPASS_DONE" ] || { say "FATAL prepass failed"; exit 1; }

# ---- S2: base-correct bank (8 shards) + ref_logprob ----
if [ ! -f "$CELL/BANK_DONE" ]; then
  say "S2 bank sampling"
  freegpu
  for i in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$i HF_HUB_OFFLINE=1 setsid nohup $PY rl_training/sample_base_solutions.py \
      --model "$BASE" --n 128 --dataset "$DATASET" --difficulty-json "$DIFF" --subset "$SUBSET" \
      --num-shards 8 --shard-index $i --out "$CELL/bank_shard$i.jsonl" --max-keep 8 \
      > "$LOGS/bank_${NAME}_${DATASET}_$i.log" 2>&1 &
  done
  wait
  cat $CELL/bank_shard*.jsonl > "$BANKRAW"
  say "S2 bankraw witnesses=$(wc -l < $BANKRAW)"
  freegpu
  CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 $PY rl_training/score_bank_logprobs.py \
    --model "$BASE" --bank "$BANKRAW" --out "$BASESCORED" --field logp_theta >> "$L" 2>&1
  $PY rl_training/build_route_bank.py setref --prefix-bank "$BANKRAW" --scored "$BASESCORED" --out "$BANK" >> "$L" 2>&1
  [ -s "$BANK" ] && touch "$CELL/BANK_DONE" && say "S2 done bank=$(wc -l < $BANK)"
fi
[ -f "$CELL/BANK_DONE" ] || { say "FATAL bank failed"; exit 1; }

# ---- train+score one arm ----
train_arm(){
  local arm="$1"; shift; local extra="$*"
  local run=$CELL/$arm
  [ -f "$CELL/${arm}_DONE" ] && { say "$arm already done"; return 0; }
  say "TRAIN $arm extra=[$extra]"
  rm -f "$run/TRAIN_DONE"; mkdir -p "$run"; freegpu
  CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 VLLM_ATTENTION_BACKEND=FLASHINFER setsid nohup $PY -m trl.scripts.vllm_serve \
    --model "$BASE" --tensor_parallel_size 1 --max_model_len 4096 --gpu_memory_utilization 0.82 --port 8000 \
    > "$LOGS/${NAME}_${DATASET}_${arm}_vllm.log" 2>&1 &
  for t in $(seq 1 120); do curl -s localhost:8000/health >/dev/null 2>&1 && break; sleep 3; done
  CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 HF_HUB_OFFLINE=1 VLLM_ATTENTION_BACKEND=FLASHINFER $PY -m accelerate.commands.launch \
    --config_file rl_training/accelerate_zero3.yaml --num_processes 7 --main_process_ip 127.0.0.1 \
    --main_process_port 29501 --rdzv_backend c10d -m rl_training.train_grpo --no-lora --no-novelty \
    --model "$BASE" --dataset "$DATASET" --difficulty-json "$DIFF" \
    --num-train-steps "$STEPS" --num-generations 8 --max-completion-length 2560 --output-dir "$run" $extra \
    >> "$LOGS/${NAME}_${DATASET}_${arm}_train.log" 2>&1
  local rc=$?; freegpu
  [ "$rc" = 0 ] || { say "$arm TRAIN FAILED rc=$rc"; return 1; }
  CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY \
    rl_training/score_bank_logprobs.py --model "$run" --bank "$BANK" \
    --out "$CELL/scored_${arm}.jsonl" --field logp_theta >> "$L" 2>&1
  [ $? = 0 ] && touch "$CELL/${arm}_DONE" && say "$arm scored lines=$(wc -l < $CELL/scored_${arm}.jsonl 2>/dev/null)"
}

train_arm plain
train_arm expSR --support-ratchet --ratchet-bank "$BANK" --ratchet-alpha 0.5 --ratchet-mu 0.5
touch "$CELL/CELL_DONE"
say "=== CELL DONE $NAME x $DATASET ==="
