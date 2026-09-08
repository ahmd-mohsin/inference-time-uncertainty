#!/usr/bin/env bash
pkill -9 -f stratified_passk 2>/dev/null; pkill -9 -f code_recover 2>/dev/null; pkill -9 -f grad_align 2>/dev/null; pkill -9 -f cf_disagree 2>/dev/null; pkill -9 -f "vllm" 2>/dev/null; sleep 7
cd /home/greenland-user/inference-time-uncertainty
exec bash rl_training/go_mrec.sh "$1" "$2" "$3" 8 12
