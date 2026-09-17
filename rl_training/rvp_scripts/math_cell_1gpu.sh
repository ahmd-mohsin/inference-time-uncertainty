#!/bin/bash
# SINGLE-GPU full RVP cell: bank -> RFT(LoRA) -> pairs -> DPO(LoRA, NOT sharded) -> eval, all pinned
# to ONE GPU. Run 8 per node (GPU 0..7) => 72 concurrent cells across 9 nodes = 100% GPU utilization.
# env: GPU BASE EVAL TAG [HARDNEG SEED NEVAL NBANK GEN_GPU_MEM]
export HOME=/home/greenland-user PATH=$HOME/.local/bin:$PATH HF_HOME=$HOME/.cache HF_HUB_DISABLE_XET=1 VLLM_ATTENTION_BACKEND=FLASHINFER WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${GPU:?}
cd $HOME/inference-time-uncertainty
BASE=${BASE:?}; EVAL=${EVAL:?}; TAG=${TAG:?}; SEED=${SEED:-1}; NEVAL=${NEVAL:-200}; NBANK=${NBANK:-500}; GM=${GEN_GPU_MEM:-0.45}
export MAXLEN=${MAXLEN:-3072} MAXTOK=${MAXTOK:-1024}
G=$HOME/gu; V=$G/$TAG; L=$G/logs; mkdir -p $V $L; R=$V/RES.md; : >$R
echo "# CELL1GPU base=$BASE eval=$EVAL hardneg=${HARDNEG:-0} seed=$SEED gpu=$GPU $(date -u)" >>$R
# 1) bank (verified-correct self-samples)
[ -s $V/bank.jsonl ] || GEN_GPU_MEM=$GM python3 -m rl_training.math_rvp --mode bank --model $BASE --dataset math_full --split train --n $NBANK --k 8 --out $V/bank.jsonl >$L/${TAG}_bank.log 2>&1
echo "bank=$(wc -l <$V/bank.jsonl 2>/dev/null)" >>$R
# 2) RFT (LoRA single-GPU) -> merged
[ -d $V/rft/merged_full ] || { python3 -m rl_training.sft_train --model $BASE --data $V/bank.jsonl --out $V/rft --seed 1 --max-steps 300 --bsz 8 >$L/${TAG}_rft.log 2>&1; python3 -c "from rl_training.model_utils import merge_adapter_if_needed as m;m('$V/rft')" >>$L/${TAG}_rft.log 2>&1; }
RFT=$V/rft/merged_full
[ -f $RFT/config.json ] || { echo "RFT_FAILED" >>$R; cat $R; exit 1; }
# 3) pairs (verified y+/y- from RFT; optional hard-negative mining)
[ -s $V/pairs.jsonl ] || GEN_GPU_MEM=$GM python3 -m rl_training.math_rvp --mode pairs --model $RFT --dataset math_full --split train --n $NBANK --k 12 --max-pairs-per 2 ${HARDNEG:+--hard-neg} --out $V/pairs.jsonl >$L/${TAG}_pairs.log 2>&1
echo "pairs=$(wc -l <$V/pairs.jsonl 2>/dev/null)" >>$R
python3 rl_training/rvp_scripts/s3_sync.py $TAG >/dev/null 2>&1 || true
# 4) RVP = LoRA DPO (single-GPU, precomputes ref logps so only policy in memory) -> merged
[ -d $V/rvp/merged_full ] || { python3 -m rl_training.dpo_train --model $RFT --data $V/pairs.jsonl --out $V/rvp --seed $SEED --beta 0.1 --max-steps 300 --bsz 4 >$L/${TAG}_dpo.log 2>&1; python3 -c "from rl_training.model_utils import merge_adapter_if_needed as m;m('$V/rvp')" >>$L/${TAG}_dpo.log 2>&1; }
RVP=$V/rvp/merged_full
# 5) eval base / rft / rvp (pass@1 + coverage)
for arm in base rft rvp; do
  case $arm in base) M=$BASE;; rft) M=$RFT;; rvp) M=$RVP;; esac
  [ "$arm" = base ] || [ -f "$M/config.json" ] || { echo "$arm MISSING" >>$R; continue; }
  [ -s $V/ev_$arm.json ] || GEN_GPU_MEM=$GM python3 -m rl_training.math_rvp --mode eval --model "$M" --dataset $EVAL --split test --n $NEVAL --k 16 --out $V/ev_$arm.json >$L/${TAG}_ev_$arm.log 2>&1
  python3 -c "import json;d=json.load(open('$V/ev_$arm.json'));print('$arm pass1=%.4f cov=%.4f'%(d['pass1'],d['coverage_passk']))" >>$R 2>/dev/null
done
echo "CELL1GPU_DONE $(date -u)" >>$R; cat $R
python3 rl_training/rvp_scripts/s3_sync.py $TAG >/dev/null 2>&1 || true
