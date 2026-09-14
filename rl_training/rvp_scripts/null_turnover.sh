#!/bin/bash
# alpha/beta null-turnover control: eval an UNCHANGED RFT ckpt TWICE (independent RNG), per-problem, compute null turnover.
export HOME=/home/greenland-user PATH=$HOME/.local/bin:$PATH HF_HOME=$HOME/.cache HF_HUB_DISABLE_XET=1 VLLM_ATTENTION_BACKEND=FLASHINFER WANDB_MODE=disabled PYTHONPATH=$HOME/gu/shim
L13=$(find $HOME/.local -name "libcudart.so.13*" 2>/dev/null|head -1); export LD_LIBRARY_PATH="$(dirname $L13):${LD_LIBRARY_PATH:-}"
cd $HOME/inference-time-uncertainty
G=$HOME/gu; M=$(ls -d $G/m_c3hard_rft_s1/merged_full 2>/dev/null|head -1); OOD=$(ls $G/comp_data/w_ood_d*.jsonl|head -1)
R=$G/cct/NULL_TURNOVER.md; : > $R; echo "# alpha/beta null turnover  ckpt=$M pool=$OOD k=16  $(date -u)" >> $R
CUDA_VISIBLE_DEVICES=0 EVAL_ENFORCE_EAGER=1 python3 -m rl_training.comp_eval --model $M --pool $OOD --k 16 --temperature 0.8 --out $G/cct/nt_passA >$G/logs/nt_A.log 2>&1
CUDA_VISIBLE_DEVICES=1 EVAL_ENFORCE_EAGER=1 python3 -m rl_training.comp_eval --model $M --pool $OOD --k 16 --temperature 0.8 --out $G/cct/nt_passB >$G/logs/nt_B.log 2>&1
python3 - <<PY >> $R 2>&1
import json
A=json.load(open("$G/cct/nt_passA/comp_comp.json"))["per_problem"]
B=json.load(open("$G/cct/nt_passB/comp_comp.json"))["per_problem"]
a=[x.get("ok",0) for x in A]; b=[x.get("ok",0) for x in B]; n=min(len(a),len(b))
acc_a=sum(a)/n; acc_b=sum(b)/n
# null "acquisition": second solves what first missed; null "regression": second misses what first solved
acq_den=sum(1 for i in range(n) if a[i]==0); reg_den=sum(1 for i in range(n) if a[i]==1)
null_alpha=sum(1 for i in range(n) if a[i]==0 and b[i]==1)/max(1,acq_den)
null_beta =sum(1 for i in range(n) if a[i]==1 and b[i]==0)/max(1,reg_den)
union=sum(1 for i in range(n) if a[i] or b[i])/n
print(f"n={n} accA={acc_a:.3f} accB={acc_b:.3f}  NULL_alpha={null_alpha:.3f} NULL_beta={null_beta:.3f} oracle_union={union:.3f}")
print("=> subtract/compare these from reported cross-checkpoint alpha/beta (§141) to separate training-induced acquire/regress from sampling turnover.")
PY
echo "NULL_DONE $(date -u)" >> $R
