#!/bin/bash
# On-node watcher: as each base-model bx_m15_* cell's RVP checkpoint appears, run the award follow-ons
# (self-consistency frontier + iterative-RFT/ReST + probability-mass accounting), S3-sync, mark done.
# Detached; polls up to ~6h. GPU per dataset matches the cell's launch GPU. Env: BASE (default Qwen2.5-Math-1.5B).
export HOME=/home/greenland-user PATH=$HOME/.local/bin:$PATH HF_HUB_DISABLE_XET=1
cd $HOME/inference-time-uncertainty
BASE=${BASE:-Qwen/Qwen2.5-Math-1.5B}; G=$HOME/gu; L=$G/logs; mkdir -p $L
declare -A EV=([m500]=math500 [ol]=olympiad_bench [om]=omni_math [gsm]=gsm8k [amc]=amc [dm]=deepmath)
declare -A NV=([m500]=200 [ol]=150 [om]=200 [gsm]=200 [amc]=60 [dm]=200)
declare -A GP=([m500]=0 [ol]=0 [amc]=0 [gsm]=1 [om]=1 [dm]=1)
for i in $(seq 1 72); do
  pend=0
  for V in $G/bx_m15_*; do
    [ -d "$V" ] || continue
    tag=$(basename $V); dab=${tag#bx_m15_}; dab=${dab%_s1}
    [ -f $V/rvp/merged_full/config.json ] || { pend=1; continue; }
    [ -f $V/followon_done ] && continue
    ev=${EV[$dab]}; nv=${NV[$dab]}; g=${GP[$dab]:-0}
    env MODE=frontier GPU=$g EVAL=$ev TAG=$tag BASE=$BASE bash rl_training/rvp_scripts/run_5h.sh >$L/${tag}_foll_front.log 2>&1
    env MODE=iterrft  GPU=$g EVAL=$ev TAG=$tag BASE=$BASE bash rl_training/rvp_scripts/run_5h.sh >$L/${tag}_foll_iter.log 2>&1
    models="base=$BASE,rft=$V/rft/merged_full,rvp=$V/rvp/merged_full"
    [ -f $V/rft2/merged_full/config.json ] && models="$models,rft2=$V/rft2/merged_full"
    CUDA_VISIBLE_DEVICES=$g python3 -m rl_training.mass_accounting --models "$models" --data $V/pairs.jsonl --n 80 --out $V/mass_${tag}.json >$L/${tag}_foll_mass.log 2>&1
    python3 rl_training/rvp_scripts/s3_sync.py $tag >/dev/null 2>&1 || true
    touch $V/followon_done; echo "[bx_followon] $tag done $(date -u)"
  done
  [ $pend -eq 0 ] && break
  sleep 300
done
echo "[bx_followon] ALL done $(date -u)"
