# Shared model-path utilities for the RL pipeline.
#
# The recurring failure mode across eval / harvest / the oursAB segment handoff is that
# GRPOTrainer/SFTTrainer with PEFT save a *bare LoRA adapter* (adapter_model.safetensors +
# adapter_config.json, NO config.json / full weights). vLLM and a fresh trainer cannot load a
# bare adapter as a model ("Invalid repository ID ... ensure the presence of a 'config.json'").
# Any stage that consumes a previous stage's output as a *base model* (vLLM generation, the next
# GRPO segment, SFT) must first MERGE the adapter into its base and load the merged full model.
#
# merge_adapter_if_needed() is the single, tested implementation of that merge, reused everywhere.

import json
import os
from pathlib import Path


def is_adapter_dir(path: str) -> bool:
    """True if `path` is a bare LoRA adapter dir (adapter_config.json present, config.json absent)."""
    p = Path(path)
    return (p / "adapter_config.json").exists() and not (p / "config.json").exists()


def merge_adapter_if_needed(model_path: str, base_fallback: str = "Qwen/Qwen3-8B",
                            cleanup_reuse: bool = True) -> str:
    """If `model_path` is a bare LoRA adapter dir, merge it into its base model and return the
    merged full-model dir (`<model_path>/merged_full`). Otherwise return `model_path` unchanged.

    - The adapter's recorded base_model_name_or_path may be a stale absolute snapshot path from a
      dead host; if it doesn't resolve locally we fall back to `base_fallback` (the hub id).
    - If a merged dir with config.json already exists it is reused (idempotent across resumes).
    - Merge is done on CPU-then-save (no CUDA needed, avoids contending with a live vLLM server).
    """
    if not is_adapter_dir(model_path):
        return model_path  # already a full model, or a HF repo id

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    merged_dir = str(Path(model_path) / "merged_full")
    if (Path(merged_dir) / "config.json").exists():
        print(f"[merge] reusing merged model at {merged_dir}")
        return merged_dir

    acfg = json.load(open(Path(model_path) / "adapter_config.json"))
    base = acfg.get("base_model_name_or_path") or base_fallback
    if os.path.isabs(base) and not Path(base).exists():
        base = base_fallback
    print(f"[merge] merging adapter {model_path} into base {base} -> {merged_dir}")

    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16,
                                                 trust_remote_code=True)
    model = PeftModel.from_pretrained(model, model_path)
    model = model.merge_and_unload()
    Path(merged_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(merged_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(base, trust_remote_code=True).save_pretrained(merged_dir)
    del model
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    print(f"[merge] saved merged model -> {merged_dir}")
    return merged_dir
