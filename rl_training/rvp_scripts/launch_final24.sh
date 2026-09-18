#!/bin/bash
# FINAL 24h plan: dedicate all 3 clusters (72 GPUs) to the important paper experiments, on the
# RESILIENT 1-GPU pack (per-stage S3 sync -> survives pod cycling). One model per cluster, each
# = 6 datasets x 4 seeds = 24 cells (8/node). Delivers rigorous generalization + CIs + breadth
# across 3 reliable model classes. Kills any prior run first.
#   A(4210)=Qwen2.5-Math-1.5B  B(4211)=Qwen2.5-Math-1.5B-Instruct  C(4212)=Qwen2.5-Math-7B-Instruct
set -u
AP=/tmp/askpass_empty.sh
[ -f "$AP" ] || { printf '#!/bin/bash\necho ""\n' > "$AP"; chmod +x "$AP"; }
SSHM=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o PubkeyAuthentication=no -o PreferredAuthentications=password,keyboard-interactive -o NumberOfPasswordPrompts=1 -o ConnectTimeout=25 -o LogLevel=ERROR)
WSSH="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o PubkeyAuthentication=no -o PreferredAuthentications=password -o ConnectTimeout=20 -o LogLevel=ERROR -p 2222"
DS=("math500:m500:200" "olympiad_bench:ol:150" "omni_math:om:200" "gsm8k:gsm:200" "amc:amc:60" "deepmath:dm:200")
SEEDS=(1 2 3 4)
build() { local mab=$1; SPECS=()
  for dd in "${DS[@]}"; do ev=${dd%%:*}; dab=$(echo $dd|cut -d: -f2); nv=$(echo $dd|cut -d: -f3)
    for s in "${SEEDS[@]}"; do SPECS+=("$ev|$nv|$dab|$s|f24_${mab}_${dab}_s${s}"); done; done
}   # 6 ds x 4 seeds = 24 specs (ev|neval|dab|seed|tag)
node_script() { local base=$1 gm=$2 start=$3; local out=""
  out+="export HOME=/home/greenland-user PATH=/home/greenland-user/.local/bin:\$PATH\n"
  out+="cd \$HOME/inference-time-uncertainty && git pull --rebase 2>&1|tail -1\n"
  out+="pkill -9 -f math_hard_shard 2>/dev/null; pkill -9 -f math_cell_1gpu 2>/dev/null; pkill -9 -f accelerate.commands 2>/dev/null; pkill -9 -f 'math_rvp --mode' 2>/dev/null; for p in \$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 \$p 2>/dev/null; done; sleep 5\n"
  for g in 0 1 2 3 4 5 6 7; do local idx=$((start+g)); IFS='|' read ev nv dab s tag <<< "${SPECS[$idx]}"
    out+="setsid nohup env GPU=$g BASE=$base EVAL=$ev NEVAL=$nv SEED=$s TAG=$tag GEN_GPU_MEM=$gm bash rl_training/rvp_scripts/math_cell_1gpu.sh >\$HOME/gu/logs/cell_${tag}.log 2>&1 & echo launched-${tag}-g$g\n"
    out+="sleep 10\n"; done
  printf "%b" "$out" | base64 | tr -d '\n'
}
run_cluster() { local port=$1 w1=$2 w2=$3 base=$4 gm=$5
  local B0 B1 B2; B0=$(node_script "$base" "$gm" 0); B1=$(node_script "$base" "$gm" 8); B2=$(node_script "$base" "$gm" 16)
  SSH_ASKPASS=$AP SSH_ASKPASS_REQUIRE=force DISPLAY=:0 ssh -p $port "${SSHM[@]}" greenland-user@localhost "
    which sshpass >/dev/null 2>&1 || sudo apt-get install -y sshpass >/dev/null 2>&1
    echo -n 'MAIN '; echo $B0 | base64 -d | bash
    echo -n 'W1 '; sshpass -p '' ssh $WSSH greenland-user@$w1 \"echo $B1 | base64 -d | bash\" </dev/null 2>&1 | grep -vaE 'Warning|Permanently|Pseudo'
    echo -n 'W2 '; sshpass -p '' ssh $WSSH greenland-user@$w2 \"echo $B2 | base64 -d | bash\" </dev/null 2>&1 | grep -vaE 'Warning|Permanently|Pseudo'
  " 2>&1 | grep -vE "Warning|Permanently|Pseudo"
}
echo "=== A(4210) Qwen2.5-Math-1.5B ==="; build m15;  run_cluster 4210 10.2.88.179 10.2.140.204 Qwen/Qwen2.5-Math-1.5B 0.45
echo "=== B(4211) Qwen2.5-Math-1.5B-Instruct ==="; build m15i; run_cluster 4211 10.2.110.65 10.2.68.155 Qwen/Qwen2.5-Math-1.5B-Instruct 0.45
echo "=== C(4212) Qwen2.5-Math-7B-Instruct ==="; build m7i; run_cluster 4212 10.2.80.244 10.2.226.121 Qwen/Qwen2.5-Math-7B-Instruct 0.30
echo "[launch_final24] 72 cells dispatched $(date -u)"
