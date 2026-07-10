# Reconstruct the final oursAB (A+B+C) model by replaying the segment merge-chain.
#
# oursAB trains as a sequence of LoRA adapters, each fit ON TOP of the merged result of all
# previous ones:
#   base ─+seg0─merge→ M0 ─+seg0_sft─merge→ M1 ─+seg1─merge→ M2 ─ ... ─+seg3_sft─merge→ FINAL
# So the final model is NOT any single adapter; it is the whole chain applied in order. This
# script walks the chain deterministically and writes the final full model, which eval/vLLM can
# load directly. Idempotent: if the final dir already has config.json it is reused.
#
# Usage:
#   python -m rl_training.rebuild_chain --run-dir rl_training/runs/oursAB \
#       --base Qwen/Qwen2.5-Math-1.5B --out rl_training/runs/oursAB/final_full
import argparse, os
from pathlib import Path

# canonical oursAB chain order (must match scripts/rl_experiment.sh segment handoff)
CHAIN = ["seg0", "seg0_sft", "seg1", "seg1_sft", "seg2", "seg2_sft", "seg3", "seg3_sft"]


def rebuild(run_dir, base, out, chain=CHAIN):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(out)
    if (out / "config.json").exists():
        print(f"[rebuild] final model already exists at {out} — reuse")
        return str(out)

    # verify the full chain is present before doing any expensive work
    missing = [s for s in chain if not (Path(run_dir) / s / "adapter_model.safetensors").exists()]
    if missing:
        raise FileNotFoundError(f"chain broken — missing adapters: {missing}")

    print(f"[rebuild] base = {base}")
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16,
                                                 trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    for s in chain:
        adapter = str(Path(run_dir) / s)
        print(f"[rebuild] applying + merging {s} ...")
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()          # fold this segment in, back to a plain model
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    print(f"[rebuild] FINAL oursAB model -> {out}")
    return str(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="dir containing seg0..seg3_sft adapters")
    ap.add_argument("--base", default="Qwen/Qwen2.5-Math-1.5B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--chain", default=",".join(CHAIN),
                    help="comma-sep segment order (default: the standard 8-step oursAB chain)")
    a = ap.parse_args()
    rebuild(a.run_dir, a.base, a.out, chain=[c for c in a.chain.split(",") if c])


if __name__ == "__main__":
    main()
