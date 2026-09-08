#!/usr/bin/env bash
pkill -9 -f grad_align 2>/dev/null; sleep 5
cd /home/greenland-user/inference-time-uncertainty
CUDA_VISIBLE_DEVICES=7 exec /usr/bin/python3 -m rl_training.grad_align --model-path "$1" --dataset "$2" --tag "$3" --k 60 --last-layers 1
