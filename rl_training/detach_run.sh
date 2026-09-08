#!/usr/bin/env bash
# Reliable on-node detached launcher (ssh-hop nohup gets SIGHUP'd in shared-PID-namespace pods).
# Usage: detach_run.sh <logname> <script> [args...]   e.g. detach_run.sh A_q3b go_math_m.sh 0 Qwen/Qwen2.5-3B q3b
cd /tmp/instance_storage/gu/repo
LOG=/tmp/instance_storage/gu/logs/${1}.log; shift
SCRIPT="$1"; shift
setsid bash "rl_training/$SCRIPT" "$@" > "$LOG" 2>&1 < /dev/null &
PID=$!
disown 2>/dev/null || true
sleep 2
echo "detached $SCRIPT -> $LOG (pgid $PID); head:"; head -1 "$LOG" 2>/dev/null || echo "(log not yet written)"
