#!/bin/bash
# 72-CELL PACK: 8 single-GPU RVP cells per node x 9 nodes = 72 concurrent = 100% GPU utilization.
# Matrix = 3 models x 4 datasets x 2 variants(std/hard-neg) x 3 seeds = 72 (generalization + hard-neg
# ablation + 3-seed CIs in one saturated sweep). Model-major so each node reuses one model's HF cache.
# Kills any wave-5 sharded flywheel first (repacking the same pods).
set -u
AP=/tmp/askpass_empty.sh
[ -f "$AP" ] || { printf '#!/bin/bash\necho ""\n' > "$AP"; chmod +x "$AP"; }
SSHM=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o PubkeyAuthentication=no -o PreferredAuthentications=password,keyboard-interactive -o NumberOfPasswordPrompts=1 -o ConnectTimeout=25 -o LogLevel=ERROR)
WSSH="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o PubkeyAuthentication=no -o PreferredAuthentications=password -o ConnectTimeout=20 -o LogLevel=ERROR -p 2222"

MODELS=("Qwen/Qwen2.5-Math-1.5B:m15" "Qwen/Qwen2.5-Math-7B:m7" "Qwen/Qwen2.5-Math-7B-Instruct:m7i")
DS=("math500:m500:200" "olympiad_bench:ol:150" "omni_math:om:200" "gsm8k:gsm:200")
VARS=("0:s" "1:h"); SEEDS=(1 2 3)
# build the 72-cell flat list (model-major)
SPECS=()
for mm in "${MODELS[@]}"; do model=${mm%%:*}; mab=${mm##*:}
  for dd in "${DS[@]}"; do ev=${dd%%:*}; dab=$(echo $dd|cut -d: -f2); nv=$(echo $dd|cut -d: -f3)
    for vv in "${VARS[@]}"; do h=${vv%%:*}; vab=${vv##*:}
      for s in "${SEEDS[@]}"; do
        SPECS+=("$model|$ev|$nv|$h|$s|p_${mab}_${dab}_${vab}_s${s}")
      done; done; done; done
echo "total cells: ${#SPECS[@]}"   # expect 72

# node access: idx -> (port, host) ; host=LOCAL means main via tunnel port.
# A(4210): main 10.2.202.77 + w 10.2.88.179,10.2.140.204 ; B(4211): main + w 10.2.110.65,10.2.68.155 ;
# C(4212): main 10.2.236.82 + w 10.2.80.244,10.2.226.121
NODES=("4210:LOCAL" "4210:10.2.88.179" "4210:10.2.140.204" \
       "4211:LOCAL" "4211:10.2.110.65" "4211:10.2.68.155" \
       "4212:LOCAL" "4212:10.2.80.244" "4212:10.2.226.121" )

# emit an 8-cell launch script for node n (cells n*8 .. n*8+7 -> GPU 0..7)
node_script() { local n=$1; local out=""
  out+="export HOME=/home/greenland-user PATH=/home/greenland-user/.local/bin:\$PATH\n"
  out+="cd \$HOME/inference-time-uncertainty && git pull --rebase 2>&1|tail -1\n"
  out+="pkill -9 -f math_hard_shard 2>/dev/null; pkill -9 -f accelerate.commands 2>/dev/null; pkill -9 -f 'math_rvp --mode' 2>/dev/null; pkill -9 -f sft_train 2>/dev/null; pkill -9 -f dpo_train 2>/dev/null; sleep 4\n"
  for g in 0 1 2 3 4 5 6 7; do local idx=$((n*8+g)); IFS='|' read m e nv h s tag <<< "${SPECS[$idx]}"
    out+="setsid nohup env GPU=$g BASE=$m EVAL=$e NEVAL=$nv SEED=$s TAG=$tag ${h:+HARDNEG=$h} bash rl_training/rvp_scripts/math_cell_1gpu.sh >\$HOME/gu/logs/cell_${tag}.log 2>&1 & echo launched-${tag}-g$g\n"
    out+="sleep 8\n"
  done
  printf "%b" "$out" | base64 | tr -d '\n'
}

echo "=== killing wave-5 + launching 72-cell pack ==="
declare -A DONEPORT
for n in 0 1 2 3 4 5 6 7 8; do
  IFS=: read port host <<< "${NODES[$n]}"; B=$(node_script $n)
  if [ "$host" = LOCAL ]; then
    SSH_ASKPASS=$AP SSH_ASKPASS_REQUIRE=force DISPLAY=:0 ssh -p $port "${SSHM[@]}" greenland-user@localhost "echo $B | base64 -d | bash" 2>&1 | grep -vE "Warning|Permanently|Pseudo"
  else
    SSH_ASKPASS=$AP SSH_ASKPASS_REQUIRE=force DISPLAY=:0 ssh -p $port "${SSHM[@]}" greenland-user@localhost "sshpass -p '' ssh $WSSH greenland-user@$host \"echo $B | base64 -d | bash\" </dev/null 2>&1 | grep -vaE 'Warning|Permanently|Pseudo'" 2>&1 | grep -vE "Warning|Permanently|Pseudo"
  fi
done
echo "[launch_pack] 72 cells dispatched $(date -u)"
