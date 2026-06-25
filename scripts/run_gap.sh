#!/usr/bin/env bash
#
# run_gap.sh — Verification-Generation gap sweep: 2 models x several math datasets,
# 8-way data-parallel per (model,dataset). Each combo -> its own data/vgap_<model>_<ds> dir.
#
# Models: Qwen2.5-7B-Instruct and Qwen3-4B.
# Datasets: math500, aime_all, amc, olympiad_bench (range of difficulty, clean answers;
#           gsm8k skipped = saturated, no gap to study).
# Chains finish: max_new_tokens=32768 (config). n_chains=16.
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate topo
cd ~/inference-time-uncertainty

MODELS=("Qwen/Qwen2.5-7B-Instruct" "Qwen/Qwen3-4B")
DATASETS=("math500" "aime_all" "amc" "olympiad_bench")
# per-dataset problem cap (-1 = all). aime_all is 90; cap big sets so the sweep finishes.
declare -A NCAP=( ["math500"]=200 ["aime_all"]=90 ["amc"]=-1 ["olympiad_bench"]=200 )
NCHAINS=16
NGPU=8
mkdir -p ~/logs

for MODEL in "${MODELS[@]}"; do
  MTAG=$(echo "$MODEL" | awk -F/ '{print tolower($NF)}' | tr -d '.')
  for DS in "${DATASETS[@]}"; do
    DIR="data/vgap_${MTAG}_${DS}"
    NPROB=${NCAP[$DS]}
    echo "===== $MODEL x $DS  (n=$NPROB) -> $DIR ====="
    pids=()
    for s in $(seq 0 $((NGPU-1))); do
      CUDA_VISIBLE_DEVICES=$s python -m verification_gap.run_gap \
        --model "$MODEL" --dataset "$DS" --n-problems "$NPROB" --n-chains "$NCHAINS" \
        --output-dir "$DIR" --shard-index "$s" --num-shards "$NGPU" \
        > ~/logs/gap_${MTAG}_${DS}_shard${s}.log 2>&1 &
      pids+=($!)
    done
    fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
    python -m verification_gap.run_gap --output-dir "$DIR" --num-shards "$NGPU" --merge-only \
      > ~/logs/gap_${MTAG}_${DS}_merge.log 2>&1
    python -m verification_gap.analyze_gap --data-dir "$DIR" \
      > ~/logs/gap_${MTAG}_${DS}_analysis.log 2>&1
    echo "  done $MODEL x $DS (fail=$fail)"
  done
done

echo "===== GAP SWEEP COMPLETE ====="
touch ~/GAP_SWEEP_DONE
