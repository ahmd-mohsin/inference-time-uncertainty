#!/usr/bin/env bash
# safe kill of prior LSP jobs (argv is "bash cflaunch.sh ..." so pkill patterns don't self-match) then launch go_cf
pkill -9 -f stratified_passk 2>/dev/null; pkill -9 -f "vllm" 2>/dev/null; sleep 6
cd /home/greenland-user/inference-time-uncertainty
exec bash rl_training/go_cf.sh "$1" "$2" "$3" 200 16 3
