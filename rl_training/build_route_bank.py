# E5 ROUTE-LEVEL bank builder. Turns a full-trace mode bank into a ROUTE bank: each witness is
# truncated to its strategy PREFIX (first N tokens) so the coverage floor protects *entry into the
# reasoning basin* p_theta(z|q) rather than the full trajectory. "Preserve entry into valuable
# reasoning basins; don't preserve the trajectory" — RL stays free to improve reasoning inside a
# preserved basin.
#
# E5 then trains as plain expSR (--support-ratchet) but with --ratchet-bank <route bank>: no trainer
# change, the route-ness is entirely in the bank (short prefix completions => the teacher-forced floor
# acts only on the prefix logp).
#
# Two steps (ref_logprob for the route bank must be the BASE's logp over the PREFIX, not the full trace):
#   1) build   : truncate each completion to its first N tokens -> prefix bank (ref_logprob placeholder)
#                then score the BASE over it with score_bank_logprobs.py to get base prefix logp.
#   2) setref  : merge that base-prefix-logp in as ref_logprob -> final route bank.
#
#   python build_route_bank.py build  --bank full.jsonl --model <base> --prefix-tokens 64 --out route_noref.jsonl
#   python score_bank_logprobs.py --model <base> --bank route_noref.jsonl --out base_route_scored.jsonl
#   python build_route_bank.py setref --prefix-bank route_noref.jsonl --scored base_route_scored.jsonl --out route_bank.jsonl

import argparse, json, sys


def build(a):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    n_in = n_out = 0
    with open(a.bank) as f, open(a.out, "w") as g:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            r = json.loads(line)
            comp = r.get("completion", "")
            # tokenize WITHOUT special tokens (the completion is raw model output), keep first N ids
            ids = tok(comp, add_special_tokens=False)["input_ids"]
            pref_ids = ids[:a.prefix_tokens]
            prefix = tok.decode(pref_ids, skip_special_tokens=True)
            if not prefix.strip():
                continue  # degenerate (empty prefix) — drop
            out = {"prompt": r["prompt"], "completion": prefix,
                   "problem_id": r.get("problem_id"), "mode_id": r.get("mode_id"),
                   "ref_logprob": 0.0,  # placeholder; filled by `setref`
                   "full_ntok": len(ids), "prefix_ntok": len(pref_ids)}
            g.write(json.dumps(out) + "\n")
            n_out += 1
    print(f"[build] {n_out}/{n_in} route witnesses (prefix_tokens={a.prefix_tokens}) -> {a.out}")


def setref(a):
    scored = [json.loads(l) for l in open(a.scored) if l.strip()]
    pref = [json.loads(l) for l in open(a.prefix_bank) if l.strip()]
    if len(scored) != len(pref):
        sys.exit(f"length mismatch: prefix_bank={len(pref)} scored={len(scored)} (must be line-aligned)")
    n = 0
    with open(a.out, "w") as g:
        for pb, sc in zip(pref, scored):
            # score_bank_logprobs writes this model's logp under 'logp_theta'; here model==base,
            # so logp_theta == base logp over the PREFIX == the route-level ref.
            base_prefix_logp = sc.get("logp_theta")
            if base_prefix_logp is None:
                sys.exit("scored file missing logp_theta")
            pb = dict(pb)
            pb["ref_logprob"] = float(base_prefix_logp)
            g.write(json.dumps(pb) + "\n")
            n += 1
    print(f"[setref] wrote {n} route witnesses with base-prefix ref_logprob -> {a.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--bank", required=True)
    b.add_argument("--model", required=True, help="tokenizer source (the base model dir)")
    b.add_argument("--prefix-tokens", type=int, default=64)
    b.add_argument("--out", required=True)
    b.set_defaults(fn=build)
    s = sub.add_parser("setref")
    s.add_argument("--prefix-bank", required=True)
    s.add_argument("--scored", required=True, help="score_bank_logprobs output on the BASE over the prefix bank")
    s.add_argument("--out", required=True)
    s.set_defaults(fn=setref)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
