#!/bin/bash
# WHY-BETTER mech-interp: logit margin logp(y+)-logp(y-) under base/GRPO/RFT/RVP/shuf on the SAME held-out pairs.
export HOME=/home/greenland-user PATH=$HOME/.local/bin:$PATH HF_HOME=$HOME/.cache VLLM_ATTENTION_BACKEND=FLASHINFER PYTHONPATH=$HOME/gu/shim
L13=$(find $HOME/.local -name "libcudart.so.13*" 2>/dev/null|head -1); export LD_LIBRARY_PATH="$(dirname $L13):$LD_LIBRARY_PATH"
cd $HOME/inference-time-uncertainty; G=$HOME/gu; L=$G/logs; V=$(ls -d $G/rvp_* $G/mrvp 2>/dev/null|head -1); R=$V/MECHFULL.md; : >$R
P=$V/pairs.jsonl; hdr=$(head -1 $V/RES.md); B=$(echo "$hdr"|grep -oE "base=[^ ]+"|cut -d= -f2)
declare -A M=( [base]=$B [grpo]=$V/grpo/merged_full [rft]=$V/rft/merged_full [rvp]=$V/rvp_s1/merged_full [shuf]=$V/shuf/merged_full )
echo "# MECHFULL $(basename $V) base=$B $(date -u) pairs=$(wc -l <$P)" >>$R
g=0
for tag in base grpo rft rvp shuf; do
  m=${M[$tag]}; [ -d "$m" -o -n "$(echo $m|grep /)" ] || m=$B; [ -e "$m" ] || [ "$tag" = base ] || continue
  CUDA_VISIBLE_DEVICES=$((g%8)) setsid nohup python3 -m rl_training.rvp_margin --model "$m" --pairs $P --n 150 --tag $tag --out $V/mf_$tag.json >$L/mf_$tag.log 2>&1 &
  g=$((g+1)); sleep 2
done
wait
for tag in base grpo rft rvp shuf; do [ -f $V/mf_$tag.json ] && python3 -c "import json;d=json.load(open('$V/mf_$tag.json'));print('$tag logp+=%.3f logp-=%.3f margin=%.3f'%(d['logp_pos'],d['logp_neg'],d['margin']))" >>$R 2>/dev/null; done
echo "MECHFULL_DONE $(date -u)" >>$R
