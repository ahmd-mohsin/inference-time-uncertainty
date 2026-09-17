#!/bin/bash
# Wave-7: fill idle clusters A(4210)+C(4212) = 48 GPUs with NEW breadth experiments, 1-GPU packed
# (math_cell_1gpu.sh). Matrix = 2 small math bases x 4 NEW datasets x 2 variants x 3 seeds = 48.
#   A = Qwen2.5-Math-1.5B ; C = Qwen2.5-Math-1.5B-Instruct   (both reliable + 1-GPU packable)
#   NEW datasets: competition_math, deepmath (moderate, expect gains) + amc, aime (hard endpoints, honest).
# Extends generalization breadth beyond the {m500,ol,om,gsm} core. Reliable models only (no dud risk).
set -u
AP=/tmp/askpass_empty.sh
[ -f "$AP" ] || { printf '#!/bin/bash\necho ""\n' > "$AP"; chmod +x "$AP"; }
SSHM=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o PubkeyAuthentication=no -o PreferredAuthentications=password,keyboard-interactive -o NumberOfPasswordPrompts=1 -o ConnectTimeout=25 -o LogLevel=ERROR)
WSSH="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o PubkeyAuthentication=no -o PreferredAuthentications=password -o ConnectTimeout=20 -o LogLevel=ERROR -p 2222"
DS=("competition_math:cm:200" "deepmath:dm:200" "amc:amc:60" "aime:aime:60")
VARS=("0:s" "1:h"); SEEDS=(1 2 3)
# build 24 specs per model (ds-major)
build() { local model=$1 mab=$2; SPECS=()
  for dd in "${DS[@]}"; do ev=${dd%%:*}; dab=$(echo $dd|cut -d: -f2); nv=$(echo $dd|cut -d: -f3)
    for vv in "${VARS[@]}"; do h=${vv%%:*}; vab=${vv##*:}
      for s in "${SEEDS[@]}"; do SPECS+=("$model|$ev|$nv|$h|$s|w7_${mab}_${dab}_${vab}_s${s}"); done; done; done
}
node_script() { local base=$1; local out=""   # $1=start idx into global SPECS (8 cells -> GPU 0..7)
  out+="export HOME=/home/greenland-user PATH=/home/greenland-user/.local/bin:\$PATH\n"
  out+="cd \$HOME/inference-time-uncertainty && git pull --rebase 2>&1|tail -1\n"
  for g in 0 1 2 3 4 5 6 7; do local idx=$((base+g)); IFS='|' read m e nv h s tag <<< "${SPECS[$idx]}"
    out+="setsid nohup env GPU=$g BASE=$m EVAL=$e NEVAL=$nv SEED=$s TAG=$tag GEN_GPU_MEM=0.45 ${h:+HARDNEG=$h} bash rl_training/rvp_scripts/math_cell_1gpu.sh >\$HOME/gu/logs/cell_${tag}.log 2>&1 & echo launched-${tag}-g$g\n"
    out+="sleep 8\n"; done
  printf "%b" "$out" | base64 | tr -d '\n'
}
run_cluster() { local port=$1 w1=$2 w2=$3   # 3 nodes = 24 cells (idx 0-7 main, 8-15 w1, 16-23 w2)
  local B0 B1 B2; B0=$(node_script 0); B1=$(node_script 8); B2=$(node_script 16)
  SSH_ASKPASS=$AP SSH_ASKPASS_REQUIRE=force DISPLAY=:0 ssh -p $port "${SSHM[@]}" greenland-user@localhost "
    echo -n 'MAIN '; echo $B0 | base64 -d | bash
    echo -n 'W1 '; sshpass -p '' ssh $WSSH greenland-user@$w1 \"echo $B1 | base64 -d | bash\" </dev/null 2>&1 | grep -vaE 'Warning|Permanently|Pseudo'
    echo -n 'W2 '; sshpass -p '' ssh $WSSH greenland-user@$w2 \"echo $B2 | base64 -d | bash\" </dev/null 2>&1 | grep -vaE 'Warning|Permanently|Pseudo'
  " 2>&1 | grep -vE "Warning|Permanently|Pseudo"
}
echo "=== A(4210): Qwen2.5-Math-1.5B x new datasets ==="
build "Qwen/Qwen2.5-Math-1.5B" m15; run_cluster 4210 10.2.88.179 10.2.140.204
echo "=== C(4212): Qwen2.5-Math-1.5B-Instruct x new datasets ==="
build "Qwen/Qwen2.5-Math-1.5B-Instruct" m15i; run_cluster 4212 10.2.80.244 10.2.226.121
echo "[launch_wave7] 48 cells dispatched $(date -u)"
