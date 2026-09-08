#!/bin/bash
# Certify a kill run's checkpoints: sample p_hat(q) on the fragile band at each checkpoint -> tiny JSONs
# = the durable Gp-transition trajectory (survives node death, unlike the 15GB checkpoints).
export HOME=/home/greenland-user
NV=/tmp/instance_storage/gu; RT=$HOME/inference-time-uncertainty/rl_training
RUN="$1"; G="$2"
export N_SAMPLES=256
for step in 10 30 60 90 120 150; do
  C=$NV/$RUN/checkpoint-$step
  [ -f "$C/model.safetensors" ] || continue
  [ -f "$NV/eval_out/passk_cert_g${G}_s${step}.json" ] && continue
  bash $RT/go_eval2.sh "$C" cert_g${G}_s${step} olympiad_bench $NV/difficulty_olympiad_7b.json >$NV/logs/cert_g${G}_s${step}.log 2>&1
done
echo "CERT_DONE g$G"
