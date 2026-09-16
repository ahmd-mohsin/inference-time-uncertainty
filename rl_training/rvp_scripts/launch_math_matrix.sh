#!/bin/bash
# Laptop-side launcher: 9-cell MATH GENERALIZATION MATRIX on 3 fresh clusters (72 GPUs).
# Matrix = {Qwen2.5-Math-7B, Qwen2.5-7B, Yi-1.5-9B} x {MATH-500, OlympiadBench, Omni-MATH}.
# FULL RFT->RVP pipeline (SKIP_RFT=0), gentle DPO beta=0.1 / 300 steps (the CompDAG winning recipe).
# Each cell = one node (8xA100-40GB): sharded ZeRO-3 full-param DPO + full control suite
# (base, rft, xrft, shuf, rvp_s1) + pass@1/coverage eval + S3 auto-sync.
# Robust: installs sshpass on mains, ships per-node scripts as base64 (no quoting hell).
# Requires SSM tunnels UP on 4210/4211/4212 -> the 3 cluster mains' container sshd (port 2222).
set -u
AP=/tmp/askpass_empty.sh
[ -f "$AP" ] || { printf '#!/bin/bash\necho ""\n' > "$AP"; chmod +x "$AP"; }
SSHM=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o PubkeyAuthentication=no -o PreferredAuthentications=password,keyboard-interactive -o NumberOfPasswordPrompts=1 -o ConnectTimeout=25 -o LogLevel=ERROR)

CELL_ENV="SKIP_RFT=0 DPO_BETA=0.1 DPO_STEPS=300 NSEED=1 DPO_MAXLEN=768 EVAL_CONC=2 CONSOLIDATE=0 ACC_CFG=rl_training/accelerate_zero3_offload.yaml MAXLEN=3072 GEN_GPU_MEM=0.55 EVAL_GPU_MEM=0.5 VLLM_TP=1"

# emit the per-node bootstrap+launch script (fresh pytorch-base -> clone -> FULL_BOOTSTRAP -> flywheel)
node_script() { # $1=model $2=tag $3=eval $4=neval
  cat <<EOS
export HOME=/home/greenland-user PATH=/home/greenland-user/.local/bin:\$PATH
mkdir -p \$HOME/gu/logs
[ -d \$HOME/inference-time-uncertainty ] || git clone -q https://github.com/ahmd-mohsin/inference-time-uncertainty.git \$HOME/inference-time-uncertainty
cd \$HOME/inference-time-uncertainty && git pull --rebase 2>&1 | tail -1
pkill -9 -f math_hard_shard 2>/dev/null; pkill -9 -f accelerate.commands 2>/dev/null; sleep 2
setsid nohup env CELL_BASE=$1 CELL_TAG=$2 CELL_EVAL=$3 NEVAL=$4 $CELL_ENV bash rl_training/rvp_scripts/boot_and_run.sh > \$HOME/gu/logs/cell_$2.log 2>&1 &
echo "launched $2 pid \$!"
EOS
}
b64() { node_script "$@" | base64 | tr -d '\n'; }

launch_cluster() { # $1=port $2=w1 $3=w2  $4,$5,$6 = "model|tag|eval|neval" (main,w1,w2)
  local port=$1 w1=$2 w2=$3; shift 3
  local IFS='|'; read m0 t0 e0 n0 <<< "$1"; read m1 t1 e1 n1 <<< "$2"; read m2 t2 e2 n2 <<< "$3"; unset IFS
  local B0 B1 B2; B0=$(b64 "$m0" "$t0" "$e0" "$n0"); B1=$(b64 "$m1" "$t1" "$e1" "$n1"); B2=$(b64 "$m2" "$t2" "$e2" "$n2")
  SSH_ASKPASS=$AP SSH_ASKPASS_REQUIRE=force DISPLAY=:0 ssh -p $port "${SSHM[@]}" greenland-user@localhost "
    which sshpass >/dev/null 2>&1 || sudo apt-get install -y sshpass >/dev/null 2>&1
    echo -n 'MAIN '; echo $B0 | base64 -d | bash
    echo -n 'W1($w1) '; sshpass -p '' ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o PubkeyAuthentication=no -o PreferredAuthentications=password -o ConnectTimeout=20 -o LogLevel=ERROR -p 2222 greenland-user@$w1 \"echo $B1 | base64 -d | bash\" </dev/null 2>&1 | grep -vaE 'Warning|Permanently|Pseudo'
    echo -n 'W2($w2) '; sshpass -p '' ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o PubkeyAuthentication=no -o PreferredAuthentications=password -o ConnectTimeout=20 -o LogLevel=ERROR -p 2222 greenland-user@$w2 \"echo $B2 | base64 -d | bash\" </dev/null 2>&1 | grep -vaE 'Warning|Permanently|Pseudo'
  " 2>&1 | grep -vE "Warning|Permanently|Pseudo"
}

echo "=== Cluster A (4210): Qwen2.5-Math-7B ==="
launch_cluster 4210 10.2.15.19 10.2.105.252 \
  "Qwen/Qwen2.5-Math-7B|q7m_m500|math500|200" \
  "Qwen/Qwen2.5-Math-7B|q7m_olymp|olympiad_bench|150" \
  "Qwen/Qwen2.5-Math-7B|q7m_omni|omni_math|200"

echo "=== Cluster B (4211): Qwen2.5-7B ==="
launch_cluster 4211 10.2.85.135 10.2.38.129 \
  "Qwen/Qwen2.5-7B|q7_m500|math500|200" \
  "Qwen/Qwen2.5-7B|q7_olymp|olympiad_bench|150" \
  "Qwen/Qwen2.5-7B|q7_omni|omni_math|200"

echo "=== Cluster C (4212): Yi-1.5-9B ==="
launch_cluster 4212 10.2.67.195 10.2.183.215 \
  "01-ai/Yi-1.5-9B-Chat|yi9_m500|math500|200" \
  "01-ai/Yi-1.5-9B-Chat|yi9_olymp|olympiad_bench|150" \
  "01-ai/Yi-1.5-9B-Chat|yi9_omni|omni_math|200"

echo "[launch_math_matrix] all 9 cells dispatched $(date -u)"
