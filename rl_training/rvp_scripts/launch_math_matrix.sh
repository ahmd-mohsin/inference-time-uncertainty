#!/bin/bash
# Laptop-side launcher: 9-cell MATH GENERALIZATION MATRIX on 3 fresh clusters (72 GPUs).
# Matrix = {Qwen2.5-Math-7B, Qwen2.5-7B, Yi-1.5-9B} x {MATH-500, OlympiadBench, Omni-MATH}.
# FULL RFT->RVP pipeline (SKIP_RFT=0), gentle DPO beta=0.1 / 300 steps (the CompDAG winning recipe).
# Each cell = one node (8xA100-40GB) running sharded ZeRO-3 full-param DPO + full control suite
# (base, rft, xrft positive-only, shuf shuffled-pair, rvp_s1) + pass@1/coverage eval + S3 sync.
# Requires SSM tunnels UP: 4210->clusterA-main, 4211->clusterB-main, 4212->clusterC-main.
set -u
AP=/tmp/askpass_empty.sh
[ -f "$AP" ] || { printf '#!/bin/bash\necho ""\n' > "$AP"; chmod +x "$AP"; }

# Per-cell training/eval knobs. 7-9B all fit single-GPU gen (no GEN_TP). DPO sharded on all 8 GPUs.
CELL_ENV="SKIP_RFT=0 DPO_BETA=0.1 DPO_STEPS=300 NSEED=1 DPO_MAXLEN=768 EVAL_CONC=2 CONSOLIDATE=0 \
ACC_CFG=rl_training/accelerate_zero3_offload.yaml MAXLEN=3072 GEN_GPU_MEM=0.55 EVAL_GPU_MEM=0.5 VLLM_TP=1"

# bootstrap+launch one cell on a node (fresh pytorch-base -> clone -> FULL_BOOTSTRAP -> flywheel)
node_cmd() { # $1=model $2=tag $3=eval $4=neval
  echo "export HOME=/home/greenland-user PATH=/home/greenland-user/.local/bin:\$PATH; cd \$HOME; \
mkdir -p \$HOME/gu/logs; \
[ -d inference-time-uncertainty ] || git clone -q https://github.com/ahmd-mohsin/inference-time-uncertainty.git; \
cd inference-time-uncertainty && git pull --rebase 2>&1|tail -1; \
pkill -9 -f math_hard_shard 2>/dev/null; pkill -9 -f accelerate.commands 2>/dev/null; \
setsid nohup env CELL_BASE=$1 CELL_TAG=$2 CELL_EVAL=$3 NEVAL=$4 $CELL_ENV \
  bash rl_training/rvp_scripts/boot_and_run.sh > \$HOME/gu/logs/cell_$2.log 2>&1 & echo launched-$2-pid-\$!"
}

# ssh into a cluster main (via tunnel port) and fan out: main cell + 2 worker cells
launch_cluster() { # $1=port  $2=w1ip $3=w2ip  then 3 specs "model|tag|eval|neval" (main,w1,w2)
  local port=$1 w1=$2 w2=$3; shift 3
  local IFS='|'; read m0 t0 e0 n0 <<< "$1"; read m1 t1 e1 n1 <<< "$2"; read m2 t2 e2 n2 <<< "$3"; unset IFS
  local MAINCMD; MAINCMD=$(node_cmd "$m0" "$t0" "$e0" "$n0")
  local W1CMD;  W1CMD=$(node_cmd "$m1" "$t1" "$e1" "$n1")
  local W2CMD;  W2CMD=$(node_cmd "$m2" "$t2" "$e2" "$n2")
  SSH_ASKPASS=$AP SSH_ASKPASS_REQUIRE=force DISPLAY=:0 ssh -p $port -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null -o PubkeyAuthentication=no -o PreferredAuthentications=password,keyboard-interactive \
    -o NumberOfPasswordPrompts=1 -o ConnectTimeout=25 -o LogLevel=ERROR greenland-user@localhost "
      echo '### MAIN'; bash -lc '$MAINCMD'
      echo '### W1 $w1'; sshpass -p '' ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o PubkeyAuthentication=no -o PreferredAuthentications=password -o ConnectTimeout=20 -o LogLevel=ERROR -p 2222 greenland-user@$w1 '$W1CMD' </dev/null 2>&1 | grep -vaE 'Warning|Permanently|Pseudo'
      echo '### W2 $w2'; sshpass -p '' ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o PubkeyAuthentication=no -o PreferredAuthentications=password -o ConnectTimeout=20 -o LogLevel=ERROR -p 2222 greenland-user@$w2 '$W2CMD' </dev/null 2>&1 | grep -vaE 'Warning|Permanently|Pseudo'
    " 2>&1 | grep -vE "Warning|Permanently|Pseudo"
}

echo "=== Cluster A (4210): Qwen2.5-Math-7B x {MATH-500, Olympiad, Omni-MATH} ==="
launch_cluster 4210 10.2.15.19 10.2.105.252 \
  "Qwen/Qwen2.5-Math-7B|q7m_m500|math500|200" \
  "Qwen/Qwen2.5-Math-7B|q7m_olymp|olympiad_bench|150" \
  "Qwen/Qwen2.5-Math-7B|q7m_omni|omni_math|200"

echo "=== Cluster B (4211): Qwen2.5-7B x {MATH-500, Olympiad, Omni-MATH} ==="
launch_cluster 4211 10.2.85.135 10.2.38.129 \
  "Qwen/Qwen2.5-7B|q7_m500|math500|200" \
  "Qwen/Qwen2.5-7B|q7_olymp|olympiad_bench|150" \
  "Qwen/Qwen2.5-7B|q7_omni|omni_math|200"

echo "=== Cluster C (4212): Yi-1.5-9B x {MATH-500, Olympiad, Omni-MATH} ==="
launch_cluster 4212 10.2.67.195 10.2.183.215 \
  "01-ai/Yi-1.5-9B-Chat|yi9_m500|math500|200" \
  "01-ai/Yi-1.5-9B-Chat|yi9_olymp|olympiad_bench|150" \
  "01-ai/Yi-1.5-9B-Chat|yi9_omni|omni_math|200"

echo "[launch_math_matrix] all 9 cells dispatched $(date -u)"
