#!/usr/bin/env python3
"""Death-proof a cell's outputs to S3 so node wipes don't cost results/checkpoints.
Usage: python3 s3_sync.py <TAG>   (env SYNC_CKPT=1 to also push the model dirs)
Bucket: greenland-intern-artifacts-703671891219-us-east-2-an (same-account, Intern-role writable)."""
import sys, os, boto3
B = "greenland-intern-artifacts-703671891219-us-east-2-an"; PFX = "cmohsinm-rvp"
tag = sys.argv[1]; V = os.path.expanduser(f"~/gu/{tag}")
if not os.path.isdir(V): print("no dir", V); sys.exit(0)
s3 = boto3.client("s3", region_name="us-east-2")
ckpt = os.environ.get("SYNC_CKPT") == "1"
def up(path, key):
    try: s3.upload_file(path, B, key); return 1
    except Exception as e: print("err", key, str(e)[:80]); return 0
n = 0
# results: RES*.md, ev_*.json, *.jsonl (small, always)
for f in sorted(os.listdir(V)):
    fp = os.path.join(V, f)
    if os.path.isfile(fp) and (f.endswith((".md", ".json", ".jsonl"))):
        n += up(fp, f"{PFX}/{tag}/{f}")
# checkpoints (large, opt-in): trained model dirs
if ckpt:
    for d in ["rft/merged_full", "xrft/merged_full", "rvp_s1", "rvp_s2", "rvp_s3", "shuf"]:
        dp = os.path.join(V, d)
        if not os.path.isdir(dp): continue
        for root, _, files in os.walk(dp):
            for fn in files:
                lp = os.path.join(root, fn)
                n += up(lp, f"{PFX}/{tag}/{os.path.relpath(lp, V)}")
print(f"[s3_sync] {tag}: {n} objects -> s3://{B}/{PFX}/{tag}/" + (" (+ckpt)" if ckpt else ""))
