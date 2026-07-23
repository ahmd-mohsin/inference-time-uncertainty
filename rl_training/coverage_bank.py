# Coverage bank for the support ratchet: base-correct traces on the fragile band + their reference
# (base) sequence log-probs. The ratchet constrains the online policy's log-prob on each banked
# trace to stay within log(alpha) of these ref values (rl_training/support_ratchet.py).
#
# Build in TWO steps (both offline, once, before RL):
#   1) SAMPLE   base-correct rollouts on the fragile band  (reuse harvest.harvest; writes jsonl of
#               {prompt, completion, problem_id}). Fragile band = difficulty_json 'hard' label, i.e.
#               base pass@1 in the samplable-but-rare zone — exactly the modes RLVR prunes.
#   2) SCORE    add ref_logprob = teacher-forced summed log pi_base(completion | prompt) under the
#               BASE model, so the floor is anchored to the reference policy. Written back into the
#               bank jsonl as an extra field.
#
# The bank is small (a few hundred traces) and loaded once into the trainer.

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_bank_traces(base_model, dataset, difficulty_json, out_jsonl, k=64, max_keep=4,
                      max_new_tokens=3072, temperature=1.0, tensor_parallel_size=1,
                      n_problems=-1, max_total=0):
    """Step 1: harvest DISTINCT base-correct traces on the fragile ('hard') band -> jsonl.
    Thin wrapper over harvest.harvest with the base model as the sampler (max_pass_rate=1.0: we keep
    all correct traces on hard problems; these problems are rare-correct by construction)."""
    from rl_training.harvest import harvest
    n = harvest(model_path=base_model, dataset=dataset, difficulty_json=difficulty_json, k=k,
                max_keep=max_keep, max_new_tokens=max_new_tokens, temperature=temperature,
                tensor_parallel_size=tensor_parallel_size, out_jsonl=out_jsonl,
                n_problems=n_problems, all_problems=False, max_pass_rate=1.0, max_total=max_total)
    print(f"[bank] harvested {n} base-correct fragile-band traces -> {out_jsonl}")
    return n


def add_ref_logprobs(base_model, in_jsonl, out_jsonl, batch_size=8, device="cuda"):
    """Step 2: compute ref_logprob = summed teacher-forced log pi_base(completion|prompt) for each
    banked trace under the BASE model, write bank with the extra field. Uses HF (not vLLM) because we
    need exact per-token log-probs on given text, which vLLM does not expose cleanly."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from rl_training.model_utils import merge_adapter_if_needed
    from rl_training.support_ratchet import sequence_logprob

    base_model = merge_adapter_if_needed(base_model)
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16,
                                                 trust_remote_code=True).to(device).eval()

    rows = [json.loads(l) for l in open(in_jsonl) if l.strip()]
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), batch_size):
            chunk = rows[i:i + batch_size]
            # build full = prompt+completion; mask marks ONLY completion tokens
            full_ids, masks, maxlen = [], [], 0
            enc = []
            for r in chunk:
                pids = tok(r["prompt"], add_special_tokens=False)["input_ids"]
                cids = tok(r["completion"], add_special_tokens=False)["input_ids"]
                ids = pids + cids
                m = [0] * len(pids) + [1] * len(cids)
                enc.append((ids, m)); maxlen = max(maxlen, len(ids))
            pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
            for ids, m in enc:
                full_ids.append(ids + [pad] * (maxlen - len(ids)))
                masks.append(m + [0] * (maxlen - len(m)))
            ids_t = torch.tensor(full_ids, device=device)
            mask_t = torch.tensor(masks, dtype=torch.float32, device=device)
            logits = model(ids_t).logits                       # (B,T,V)
            # align: logits[:, t-1] predicts token t -> shift
            lp = sequence_logprob(logits[:, :-1, :], ids_t[:, 1:], mask_t[:, 1:])  # (B,)
            for r, val in zip(chunk, lp.tolist()):
                r["ref_logprob"] = float(val)
                out.append(r)
            print(f"[bank] ref_logprob {min(i+batch_size,len(rows))}/{len(rows)}")
    with open(out_jsonl, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"[bank] wrote {len(out)} traces with ref_logprob -> {out_jsonl}")
    return len(out)


def load_bank(bank_jsonl):
    """Load the finished bank -> list of {prompt, completion, problem_id, ref_logprob}."""
    rows = [json.loads(l) for l in open(bank_jsonl) if l.strip()]
    assert rows and "ref_logprob" in rows[0], f"bank {bank_jsonl} missing ref_logprob (run add_ref_logprobs)"
    return rows


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Build coverage bank (base-correct fragile-band traces + ref logprob)")
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--dataset", default="math500")
    ap.add_argument("--difficulty-json", required=True, help="fragile band = 'hard' labels")
    ap.add_argument("--out", required=True, help="final bank jsonl (with ref_logprob)")
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--max-keep", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=3072)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--max-total", type=int, default=0)
    ap.add_argument("--skip-harvest", action="store_true", help="reuse existing raw traces jsonl (--out.raw)")
    a = ap.parse_args()
    raw = a.out + ".raw"
    if not a.skip_harvest:
        build_bank_traces(a.base_model, a.dataset, a.difficulty_json, raw, k=a.k,
                          max_keep=a.max_keep, max_new_tokens=a.max_new_tokens,
                          tensor_parallel_size=a.tensor_parallel_size, max_total=a.max_total)
    add_ref_logprobs(a.base_model, raw, a.out)


if __name__ == "__main__":
    main()
