#!/bin/bash
# Wave-3 launcher on the death-proofed flywheel (EVAL_CONC=1, ckpt-sync-before-eval).
#   A (4210): Qwen2.5-Math-1.5B, NSEED=3  -> CIs on the 1.5B headline (q15c_*)
#   B (4211): Qwen2.5-Math-7B-Instruct    -> instruct-vs-base discriminator (qm7i_*)
#   C (4212): Mathstral-7B-v0.1           -> 3rd math family, generalization breadth (mstl_*)
# Tunnels -> container sshd 2222. mi: 4210=mi-00a4504279949fcfd 4211=mi-0140fa8330632162e 4212=mi-00d411038ca81382a
set -u
AP=/tmp/askpass_empty.sh
[ -f "$AP" ] || { printf '#!/bin/bash\necho ""\n' > "$AP"; chmod +x "$AP"; }
SSHM=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o PubkeyAuthentication=no -o PreferredAuthentications=password,keyboard-interactive -o NumberOfPasswordPrompts=1 -o ConnectTimeout=25 -o LogLevel=ERROR)
WSSH="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o PubkeyAuthentication=no -o PreferredAuthentications=password -o ConnectTimeout=20 -o LogLevel=ERROR -p 2222"

node_b64() { # $1=model $2=tag $3=eval $4=neval $5=nseed
  local env="SKIP_RFT=0 DPO_BETA=0.1 DPO_STEPS=300 NSEED=$5 DPO_MAXLEN=768 EVAL_CONC=1 CONSOLIDATE=0 ACC_CFG=rl_training/accelerate_zero3_offload.yaml MAXLEN=3072 GEN_GPU_MEM=0.55 EVAL_GPU_MEM=0.5 VLLM_TP=1"
  cat <<EOS | base64 | tr -d '\n'
export HOME=/home/greenland-user PATH=/home/greenland-user/.local/bin:\$PATH
mkdir -p \$HOME/gu/logs
[ -d \$HOME/inference-time-uncertainty ] || git clone -q https://github.com/ahmd-mohsin/inference-time-uncertainty.git \$HOME/inference-time-uncertainty
cd \$HOME/inference-time-uncertainty && git pull --rebase 2>&1|tail -1
pkill -9 -f math_hard_shard 2>/dev/null; sleep 2
setsid nohup env CELL_BASE=$1 CELL_TAG=$2 CELL_EVAL=$3 NEVAL=$4 $env bash rl_training/rvp_scripts/boot_and_run.sh > \$HOME/gu/logs/cell_$2.log 2>&1 &
echo "launched $2 (nseed=$5) pid \$!"
EOS
}
launch() { # $1=port $2=w1 $3=w2 $4=nseed  $5,$6,$7 = "model|tag|eval|neval"
  local port=$1 w1=$2 w2=$3 ns=$4; shift 4
  local IFS='|'; read m0 t0 e0 n0 <<<"$1"; read m1 t1 e1 n1 <<<"$2"; read m2 t2 e2 n2 <<<"$3"; unset IFS
  local B0 B1 B2; B0=$(node_b64 "$m0" "$t0" "$e0" "$n0" "$ns"); B1=$(node_b64 "$m1" "$t1" "$e1" "$n1" "$ns"); B2=$(node_b64 "$m2" "$t2" "$e2" "$n2" "$ns")
  SSH_ASKPASS=$AP SSH_ASKPASS_REQUIRE=force DISPLAY=:0 ssh -p $port "${SSHM[@]}" greenland-user@localhost "
    which sshpass >/dev/null 2>&1 || sudo apt-get install -y sshpass >/dev/null 2>&1
    echo -n 'MAIN '; echo $B0 | base64 -d | bash
    echo -n 'W1 '; sshpass -p '' ssh $WSSH greenland-user@$w1 \"echo $B1 | base64 -d | bash\" </dev/null 2>&1 | grep -vaE 'Warning|Permanently|Pseudo'
    echo -n 'W2 '; sshpass -p '' ssh $WSSH greenland-user@$w2 \"echo $B2 | base64 -d | bash\" </dev/null 2>&1 | grep -vaE 'Warning|Permanently|Pseudo'
  " 2>&1 | grep -vE "Warning|Permanently|Pseudo"
}

echo "=== A(4210): Qwen2.5-Math-1.5B NSEED=3 (CIs) ==="
launch 4210 10.2.15.19 10.2.105.252 3 \
  "Qwen/Qwen2.5-Math-1.5B|q15c_m500|math500|200" \
  "Qwen/Qwen2.5-Math-1.5B|q15c_olymp|olympiad_bench|150" \
  "Qwen/Qwen2.5-Math-1.5B|q15c_omni|omni_math|200"
echo "=== B(4211): Qwen2.5-Math-7B-Instruct (discriminator) ==="
launch 4211 10.2.85.135 10.2.38.129 1 \
  "Qwen/Qwen2.5-Math-7B-Instruct|qm7i_m500|math500|200" \
  "Qwen/Qwen2.5-Math-7B-Instruct|qm7i_olymp|olympiad_bench|150" \
  "Qwen/Qwen2.5-Math-7B-Instruct|qm7i_omni|omni_math|200"
echo "=== C(4212): Mathstral-7B-v0.1 (3rd family) ==="
launch 4212 10.2.67.195 10.2.183.215 1 \
  "mistralai/Mathstral-7B-v0.1|mstl_m500|math500|200" \
  "mistralai/Mathstral-7B-v0.1|mstl_olymp|olympiad_bench|150" \
  "mistralai/Mathstral-7B-v0.1|mstl_omni|omni_math|200"
echo "[launch_wave3] dispatched $(date -u)"
