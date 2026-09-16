#!/bin/bash
# Phase-2 launcher on the 72-GPU fleet:
#   (A) Cluster A: MECHANISM PANEL (#2) — teacher-force base vs RVP over each winning cell's own
#       pairs, report logp(y+)/logp(y-)/margin (Prop-2 signature). Eval-only, reuses existing
#       q7m_* checkpoints + pairs on cluster A (main=q7m_m500, w1=q7m_olymp, w2=q7m_omni).
#   (B) Clusters B & C: 6 NEW math-specialized-base cells (#4), FULL RFT->RVP (beta=0.1/300).
#       B = DeepSeek-Math-7B x {MATH-500, Olympiad, Omni-MATH}
#       C = Qwen2.5-Math-1.5B x {MATH-500, Olympiad, Omni-MATH}
# Requires tunnels 4210(A)/4211(B)/4212(C) -> container sshd 2222.
set -u
AP=/tmp/askpass_empty.sh
[ -f "$AP" ] || { printf '#!/bin/bash\necho ""\n' > "$AP"; chmod +x "$AP"; }
SSHM=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o PubkeyAuthentication=no -o PreferredAuthentications=password,keyboard-interactive -o NumberOfPasswordPrompts=1 -o ConnectTimeout=25 -o LogLevel=ERROR)
WSSH="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o PubkeyAuthentication=no -o PreferredAuthentications=password -o ConnectTimeout=20 -o LogLevel=ERROR -p 2222"
CELL_ENV="SKIP_RFT=0 DPO_BETA=0.1 DPO_STEPS=300 NSEED=1 DPO_MAXLEN=768 EVAL_CONC=2 CONSOLIDATE=0 ACC_CFG=rl_training/accelerate_zero3_offload.yaml MAXLEN=3072 GEN_GPU_MEM=0.55 EVAL_GPU_MEM=0.5 VLLM_TP=1"

# ---- (A) mechanism-panel node script: run margin(base) + margin(rvp_s1) on this node's cell ----
mech_script() { # $1=base-model $2=tag
  cat <<EOS
export HOME=/home/greenland-user PATH=/home/greenland-user/.local/bin:\$PATH HF_HUB_DISABLE_XET=1
cd \$HOME/inference-time-uncertainty && git pull --rebase 2>&1|tail -1
V=\$HOME/gu/$2
if [ -s \$V/pairs.jsonl ] && [ -f \$V/rvp_s1/config.json ]; then
  for p in \$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 \$p 2>/dev/null; done; sleep 4
  CUDA_VISIBLE_DEVICES=0 DPO_MAXLEN=768 python3 -m rl_training.math_rvp --mode margin --model $1 --data \$V/pairs.jsonl --n 200 --out \$V/margin_base.json > \$HOME/gu/logs/${2}_margin_base.log 2>&1
  CUDA_VISIBLE_DEVICES=0 DPO_MAXLEN=768 python3 -m rl_training.math_rvp --mode margin --model \$V/rvp_s1 --data \$V/pairs.jsonl --n 200 --out \$V/margin_rvp.json > \$HOME/gu/logs/${2}_margin_rvp.log 2>&1
  python3 rl_training/rvp_scripts/s3_sync.py $2 2>&1|tail -1
  echo "margin-$2 done: \$(python3 -c "import json;b=json.load(open('\$V/margin_base.json'));r=json.load(open('\$V/margin_rvp.json'));print('base m=%.3f (pos %.3f neg %.3f) -> rvp m=%.3f (pos %.3f neg %.3f)'%(b['margin'],b['logp_pos'],b['logp_neg'],r['margin'],r['logp_pos'],r['logp_neg']))")"
else echo "margin-$2 SKIP (missing pairs/rvp)"; fi
EOS
}
# ---- (B/C) fresh-cell bootstrap+launch (full RFT->RVP), base64-shipped ----
node_script() { # $1=model $2=tag $3=eval $4=neval
  cat <<EOS
export HOME=/home/greenland-user PATH=/home/greenland-user/.local/bin:\$PATH
mkdir -p \$HOME/gu/logs
[ -d \$HOME/inference-time-uncertainty ] || git clone -q https://github.com/ahmd-mohsin/inference-time-uncertainty.git \$HOME/inference-time-uncertainty
cd \$HOME/inference-time-uncertainty && git pull --rebase 2>&1|tail -1
pkill -9 -f math_hard_shard 2>/dev/null; pkill -9 -f accelerate.commands 2>/dev/null; sleep 2
setsid nohup env CELL_BASE=$1 CELL_TAG=$2 CELL_EVAL=$3 NEVAL=$4 $CELL_ENV bash rl_training/rvp_scripts/boot_and_run.sh > \$HOME/gu/logs/cell_$2.log 2>&1 &
echo "launched $2 pid \$!"
EOS
}
b64() { "$@" | base64 | tr -d '\n'; }

echo "=== (A) MECHANISM PANEL on cluster A (4210) ==="
MB0=$(b64 mech_script Qwen/Qwen2.5-Math-7B q7m_m500)
MB1=$(b64 mech_script Qwen/Qwen2.5-Math-7B q7m_olymp)
MB2=$(b64 mech_script Qwen/Qwen2.5-Math-7B q7m_omni)
SSH_ASKPASS=$AP SSH_ASKPASS_REQUIRE=force DISPLAY=:0 ssh -p 4210 "${SSHM[@]}" greenland-user@localhost "
  which sshpass >/dev/null 2>&1 || sudo apt-get install -y sshpass >/dev/null 2>&1
  echo -n 'A-main '; echo $MB0 | base64 -d | bash
  echo -n 'A-w1 '; sshpass -p '' ssh $WSSH greenland-user@10.2.15.19 \"echo $MB1 | base64 -d | bash\" </dev/null 2>&1 | grep -vaE 'Warning|Permanently|Pseudo'
  echo -n 'A-w2 '; sshpass -p '' ssh $WSSH greenland-user@10.2.105.252 \"echo $MB2 | base64 -d | bash\" </dev/null 2>&1 | grep -vaE 'Warning|Permanently|Pseudo'
" 2>&1 | grep -vE "Warning|Permanently|Pseudo"

launch_cluster() { # $1=port $2=w1 $3=w2  specs...
  local port=$1 w1=$2 w2=$3; shift 3
  local IFS='|'; read m0 t0 e0 n0 <<< "$1"; read m1 t1 e1 n1 <<< "$2"; read m2 t2 e2 n2 <<< "$3"; unset IFS
  local B0 B1 B2; B0=$(b64 node_script "$m0" "$t0" "$e0" "$n0"); B1=$(b64 node_script "$m1" "$t1" "$e1" "$n1"); B2=$(b64 node_script "$m2" "$t2" "$e2" "$n2")
  SSH_ASKPASS=$AP SSH_ASKPASS_REQUIRE=force DISPLAY=:0 ssh -p $port "${SSHM[@]}" greenland-user@localhost "
    which sshpass >/dev/null 2>&1 || sudo apt-get install -y sshpass >/dev/null 2>&1
    echo -n 'MAIN '; echo $B0 | base64 -d | bash
    echo -n 'W1 '; sshpass -p '' ssh $WSSH greenland-user@$w1 \"echo $B1 | base64 -d | bash\" </dev/null 2>&1 | grep -vaE 'Warning|Permanently|Pseudo'
    echo -n 'W2 '; sshpass -p '' ssh $WSSH greenland-user@$w2 \"echo $B2 | base64 -d | bash\" </dev/null 2>&1 | grep -vaE 'Warning|Permanently|Pseudo'
  " 2>&1 | grep -vE "Warning|Permanently|Pseudo"
}

echo "=== (B) DeepSeek-Math-7B on cluster B (4211) ==="
launch_cluster 4211 10.2.85.135 10.2.38.129 \
  "deepseek-ai/deepseek-math-7b-base|dsm_m500|math500|200" \
  "deepseek-ai/deepseek-math-7b-base|dsm_olymp|olympiad_bench|150" \
  "deepseek-ai/deepseek-math-7b-base|dsm_omni|omni_math|200"

echo "=== (C) Qwen2.5-Math-1.5B on cluster C (4212) ==="
launch_cluster 4212 10.2.67.195 10.2.183.215 \
  "Qwen/Qwen2.5-Math-1.5B|q15m_m500|math500|200" \
  "Qwen/Qwen2.5-Math-1.5B|q15m_olymp|olympiad_bench|150" \
  "Qwen/Qwen2.5-Math-1.5B|q15m_omni|omni_math|200"

echo "[launch_phase2] dispatched $(date -u)"
