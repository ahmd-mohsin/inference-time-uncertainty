#!/usr/bin/env bash
# Run the 4 GPU Wave-1 probes on ONE model, one after another (each frees GPU0 first; all
# resumable via per-probe DONE flags). Usage: bash scripts/run_wave1.sh <model> <tag> [dataset]
source ~/miniconda3/etc/profile.d/conda.sh; conda activate topo
cd ~/inference-time-uncertainty
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODEL="${1:?model path/id}"; TAG="${2:?tag e.g. base|grpo|oursABC}"; DS="${3:-math500}"
OUT=rl_training/runs/wave1/$TAG; mkdir -p "$OUT" ~/logs
free_gpu(){ pkill -9 -f "trl vllm" 2>/dev/null; pkill -9 -f EngineCore 2>/dev/null; pkill -9 -f vllm 2>/dev/null
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 $pid 2>/dev/null; done
  for i in $(seq 1 20); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1); [ "${u:-99999}" -lt 2000 ] 2>/dev/null && break; sleep 3; done; }
for probe in gen_verify prompt_recover modes brittleness; do
  o="$OUT/${probe}.jsonl"
  if [ -f "$o.DONE" ]; then echo ">> $TAG/$probe already done, skip"; continue; fi
  echo ">> $TAG :: $probe"; free_gpu
  CUDA_VISIBLE_DEVICES=0 python -m rl_training.wave1_probes --probe $probe \
    --model "$MODEL" --dataset "$DS" --n-problems -1 --k 32 --max-new-tokens 3072 \
    --out "$o" 2>&1 | tee -a ~/logs/wave1_${TAG}.log | tail -3
done
free_gpu
echo ">> WAVE-1 DONE for $TAG -> $OUT"; touch ~/WAVE1_${TAG}_DONE
