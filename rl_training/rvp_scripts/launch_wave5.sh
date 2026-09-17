#!/bin/bash
# Wave-5 (death-proofed flywheel + early bank/pairs S3 sync). Hard-negative-mining ablation, 48h fleet.
#   A (4210): Qwen2.5-Math-7B  STANDARD RVP x {m500,olymp,omni}  (q7s_*) -- also rebuilds 7B RFT (beta-sweep prep)
#   C (4212): Qwen2.5-Math-7B  HARD-NEG RVP x {m500,olymp,omni}  (q7h_*) -- matched A/C hard-neg ablation
#   B (4211): Qwen2.5-Math-1.5B HARD-NEG RVP x {m500,olymp,omni} (q15h_*) -- vs banked 1.5B standard (+.172/.089/.046)
# spec = "model|tag|eval|neval|hardneg". Tunnels->2222. Fleet 2026-09-17b:
#   4210=mi-0941a0f1ebf549a65 (10.2.202.77; w 10.2.88.179,10.2.140.204)
#   4211=mi-0eb661b604e071309 (10.2.214.85; w 10.2.110.65,10.2.68.155)
#   4212=mi-0096b9b2e25c90588 (10.2.236.82; w 10.2.80.244,10.2.226.121)
set -u
AP=/tmp/askpass_empty.sh
[ -f "$AP" ] || { printf '#!/bin/bash\necho ""\n' > "$AP"; chmod +x "$AP"; }
SSHM=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o PubkeyAuthentication=no -o PreferredAuthentications=password,keyboard-interactive -o NumberOfPasswordPrompts=1 -o ConnectTimeout=25 -o LogLevel=ERROR)
WSSH="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o PubkeyAuthentication=no -o PreferredAuthentications=password -o ConnectTimeout=20 -o LogLevel=ERROR -p 2222"
node_b64() { # model tag eval neval hardneg
  local hn=""; [ "$5" = 1 ] && hn="HARDNEG=1 "
  local env="${hn}SKIP_RFT=0 DPO_BETA=0.1 DPO_STEPS=300 NSEED=1 DPO_MAXLEN=768 EVAL_CONC=1 CONSOLIDATE=0 ACC_CFG=rl_training/accelerate_zero3_offload.yaml MAXLEN=3072 GEN_GPU_MEM=0.55 EVAL_GPU_MEM=0.5 VLLM_TP=1"
  cat <<EOS | base64 | tr -d '\n'
export HOME=/home/greenland-user PATH=/home/greenland-user/.local/bin:\$PATH
mkdir -p \$HOME/gu/logs
[ -d \$HOME/inference-time-uncertainty ] || git clone -q https://github.com/ahmd-mohsin/inference-time-uncertainty.git \$HOME/inference-time-uncertainty
cd \$HOME/inference-time-uncertainty && git pull --rebase 2>&1|tail -1
pkill -9 -f math_hard_shard 2>/dev/null; sleep 2
setsid nohup env CELL_BASE=$1 CELL_TAG=$2 CELL_EVAL=$3 NEVAL=$4 $env bash rl_training/rvp_scripts/boot_and_run.sh > \$HOME/gu/logs/cell_$2.log 2>&1 &
echo "launched $2 (hardneg=$5) pid \$!"
EOS
}
launch() { # port w1 w2  spec0 spec1 spec2
  local port=$1 w1=$2 w2=$3; shift 3
  local IFS='|'; read m0 t0 e0 n0 h0 <<<"$1"; read m1 t1 e1 n1 h1 <<<"$2"; read m2 t2 e2 n2 h2 <<<"$3"; unset IFS
  local B0 B1 B2; B0=$(node_b64 "$m0" "$t0" "$e0" "$n0" "$h0"); B1=$(node_b64 "$m1" "$t1" "$e1" "$n1" "$h1"); B2=$(node_b64 "$m2" "$t2" "$e2" "$n2" "$h2")
  SSH_ASKPASS=$AP SSH_ASKPASS_REQUIRE=force DISPLAY=:0 ssh -p $port "${SSHM[@]}" greenland-user@localhost "
    which sshpass >/dev/null 2>&1 || sudo apt-get install -y sshpass >/dev/null 2>&1
    echo -n 'MAIN '; echo $B0 | base64 -d | bash
    echo -n 'W1 '; sshpass -p '' ssh $WSSH greenland-user@$w1 \"echo $B1 | base64 -d | bash\" </dev/null 2>&1 | grep -vaE 'Warning|Permanently|Pseudo'
    echo -n 'W2 '; sshpass -p '' ssh $WSSH greenland-user@$w2 \"echo $B2 | base64 -d | bash\" </dev/null 2>&1 | grep -vaE 'Warning|Permanently|Pseudo'
  " 2>&1 | grep -vE "Warning|Permanently|Pseudo"
}
echo "=== A(4210): Qwen2.5-Math-7B STANDARD ==="
launch 4210 10.2.88.179 10.2.140.204 \
  "Qwen/Qwen2.5-Math-7B|q7s_m500|math500|200|0" "Qwen/Qwen2.5-Math-7B|q7s_olymp|olympiad_bench|150|0" "Qwen/Qwen2.5-Math-7B|q7s_omni|omni_math|200|0"
echo "=== C(4212): Qwen2.5-Math-7B HARD-NEG ==="
launch 4212 10.2.80.244 10.2.226.121 \
  "Qwen/Qwen2.5-Math-7B|q7h_m500|math500|200|1" "Qwen/Qwen2.5-Math-7B|q7h_olymp|olympiad_bench|150|1" "Qwen/Qwen2.5-Math-7B|q7h_omni|omni_math|200|1"
echo "=== B(4211): Qwen2.5-Math-1.5B HARD-NEG ==="
launch 4211 10.2.110.65 10.2.68.155 \
  "Qwen/Qwen2.5-Math-1.5B|q15h_m500|math500|200|1" "Qwen/Qwen2.5-Math-1.5B|q15h_olymp|olympiad_bench|150|1" "Qwen/Qwen2.5-Math-1.5B|q15h_omni|omni_math|200|1"
echo "[launch_wave5] dispatched $(date -u)"
