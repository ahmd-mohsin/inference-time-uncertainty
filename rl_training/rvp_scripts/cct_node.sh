#!/bin/bash
# CCT C0 fan on ONE node: 8 GPUs x cct_gen on disjoint DAG-task shards, using this node's RFT merged ckpt. arg: BASE seed offset
export HOME=/home/greenland-user PATH=$HOME/.local/bin:$PATH HF_HOME=$HOME/.cache HF_HUB_DISABLE_XET=1 VLLM_ATTENTION_BACKEND=FLASHINFER WANDB_MODE=disabled PYTHONPATH=$HOME/gu/shim
L13=$(find $HOME/.local -name "libcudart.so.13*" 2>/dev/null|head -1); export LD_LIBRARY_PATH="$(dirname $L13):${LD_LIBRARY_PATH:-}"
cd $HOME/inference-time-uncertainty && git pull --rebase 2>&1|tail -1
G=$HOME/gu; L=$G/logs; mkdir -p $L $G/cct
BASE=${1:-0}
M=$(ls -d $G/m_*_rft_s1/merged_full 2>/dev/null|head -1); [ -z "$M" ] && M=$(ls -d $G/w_*_rft/merged_full 2>/dev/null|head -1)
TAG=$(basename $(dirname $M)|sed 's/^m_//;s/_rft_s1//;s/^w_//;s/_rft$//')
echo "CCT_NODE tag=$TAG model=$M base=$BASE $(date -u)" > $G/cct/NODE_$TAG.log
[ -z "$M" ] && { echo "NO_RFT_CKPT" >> $G/cct/NODE_$TAG.log; exit 1; }
for gpu in 0 1 2 3 4 5 6 7; do
  seed0=$((BASE + gpu*64))
  CUDA_VISIBLE_DEVICES=$gpu GEN_GPU_MEM=0.45 setsid nohup python3 -m rl_training.cct_gen --model "$M" --seed0 $seed0 --n-tasks 64 --n-nodes 6 --k 6 --out $G/cct/C0_${TAG}_g${gpu}.json >$L/cct_${TAG}_g${gpu}.log 2>&1 &
  sleep 2
done
echo "LAUNCHED 8 gpu shards tag=$TAG $(date -u)" >> $G/cct/NODE_$TAG.log
