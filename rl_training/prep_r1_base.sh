#!/usr/bin/env bash
# Fetch the round-1 fork's final checkpoint-400 from HF and FLATTEN into runs/r1_${FORK}_ckpt
# (config.json + model.safetensors at the top level) so go_r2_resume.sh can use it as the vLLM
# base model. Idempotent: skips if already flat. Usage: bash prep_r1_base.sh <grpo|floor>
set -uo pipefail
export HOME=/home/greenland-user
cd $HOME/inference-time-uncertainty
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_TOKEN=$(cat $HOME/.hf_token)
PY=/usr/bin/python3
FORK="${1:?fork: grpo|floor}"
R1REPO="muahmed7338/cov-r1-${FORK}-7b"
BASE=$HOME/inference-time-uncertainty/rl_training/runs/r1_${FORK}_ckpt
mkdir -p "$BASE"
if [ -f "$BASE/config.json" ] && [ -f "$BASE/model.safetensors" ]; then
  echo ">> $FORK base already flat at $BASE — skip"; exit 0; fi
$PY - <<PY
import os; os.environ["HF_HUB_ENABLE_HF_TRANSFER"]="0"
from huggingface_hub import HfApi, snapshot_download
import shutil
tok=os.environ["HF_TOKEN"]; api=HfApi(token=tok)
f=list(api.list_repo_files("$R1REPO", repo_type="model"))
cks=sorted({int(x.split('-')[1].split('/')[0]) for x in f if x.startswith('checkpoint-')})
assert cks, "no r1 checkpoint in $R1REPO"
ck=f"checkpoint-{cks[-1]}"; print("latest r1:", ck)
snapshot_download("$R1REPO", repo_type="model", allow_patterns=f"{ck}/*", local_dir="$BASE/dl", token=tok)
src=os.path.join("$BASE","dl",ck)
for fn in os.listdir(src):
    shutil.move(os.path.join(src,fn), os.path.join("$BASE",fn))
shutil.rmtree(os.path.join("$BASE","dl"), ignore_errors=True)
assert os.path.exists(os.path.join("$BASE","config.json")), "FLATTEN FAILED: no config.json"
assert os.path.exists(os.path.join("$BASE","model.safetensors")), "FLATTEN FAILED: no model.safetensors"
print("r1 $FORK base ready at $BASE")
PY
