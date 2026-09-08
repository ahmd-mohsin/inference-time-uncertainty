#!/usr/bin/env bash
# Bootstrap pytorch-base-24.12 (torch 2.6) for trl+vLLM GRPO/SFT. Idempotent-ish. Run per node.
set -o pipefail
export HOME=/home/greenland-user
export PATH=$HOME/.local/bin:$PATH PIP_DISABLE_PIP_VERSION_CHECK=1
GU=/tmp/instance_storage/gu; mkdir -p "$GU/shim" "$GU/logs"
echo "[boot $(hostname)] installing python stack SEQUENTIALLY $(date)"
PI="python3 -m pip install --no-cache-dir --user"
# sequential (avoid one-shot resolver conflict); vllm dictates transformers, trl follows, no forced tf pin
for spec in "vllm==0.23.0" "trl==1.7.0" "peft" "accelerate" "deepspeed" "datasets<4" "numpy<2.1" "safetensors" "sentencepiece"; do
  echo "[boot] pip install $spec"; $PI "$spec" 2>&1 | grep -iE "Successfully installed|ERROR|already satisfied" | tail -2
done
python3 -m pip uninstall -y wandb 2>/dev/null | tail -1
# --- extra fixes discovered on cluster-3 (all required) ---
$PI jsonlines nvtx 2>&1 | grep -iE "Successfully|already" | tail -1     # GRPO needs jsonlines; nvtx.get_domain for DeepSpeed path
sudo rm -rf /usr/local/lib/python3.12/dist-packages/flash_attn* 2>/dev/null  # stale flash_attn (torch2.6 ABI) breaks vLLM on torch2.11
echo "[boot $(hostname)] extra-fixes done (jsonlines/nvtx/flash_attn-rm)"
# --- container symbol shim (transformers float8 + flex_attention import on torch 2.6) ---
cat > "$GU/shim/sitecustomize.py" <<'PY'
try:
    import torch
    if not hasattr(torch, "float8_e8m0fnu") and hasattr(torch, "float8_e4m3fn"):
        torch.float8_e8m0fnu = torch.float8_e4m3fn
except Exception: pass
try:
    import torch._higher_order_ops.flex_attention as _fa
    import transformers.integrations.flex_attention as _tfa
    if not hasattr(_tfa, "TransformGetItemToIndex") and hasattr(_fa, "TransformGetItemToIndex"):
        _tfa.TransformGetItemToIndex = _fa.TransformGetItemToIndex
except Exception: pass
PY
echo "[boot $(hostname)] verify:"
PYTHONPATH="$GU/shim" python3 -c 'import trl,vllm,transformers,peft,accelerate,deepspeed,datasets as d; print("trl",trl.__version__,"vllm",vllm.__version__,"tf",transformers.__version__,"peft ok","ds",d.__version__)' 2>&1 | tail -2
echo "[boot $(hostname)] DONE $(date)"
