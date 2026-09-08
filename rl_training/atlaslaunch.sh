#!/usr/bin/env bash
pkill -9 -f math_recover 2>/dev/null; pkill -9 -f stratified_passk 2>/dev/null; pkill -9 -f "vllm" 2>/dev/null; sleep 7
cd /home/greenland-user/inference-time-uncertainty
exec bash rl_training/go_recover.sh "$1" "$2" "$3" -1 4 12
