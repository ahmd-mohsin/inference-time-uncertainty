#!/usr/bin/env python3
"""Robust chunked downloader for ONE large HF file. Greenland nodes freeze on a single long stream
(~1.35GB per-connection cap) AND urllib hangs on the CDN redirect — but short curl ranged GETs are
fast (~40MB/s). So: get exact size via HfApi (works), then pull the file in small segments, each a
fresh short-lived `curl` ranged GET to a temp file, verified by length and appended. Resumable +
idempotent: re-run to finish a partial file.

Usage: fetch_big.py <hf_repo_id> <filename> <out_path> [seg_mb=32] [total_bytes]

total_bytes: pass explicitly to skip size discovery (HEAD hangs on these nodes' CDN redirect); if
omitted, size is read from a `bytes=0-0` ranged GET's Content-Range (HEAD is never used).
"""
import os, sys, subprocess, time

def main():
    repo, fname, out = sys.argv[1], sys.argv[2], sys.argv[3]
    seg = (int(sys.argv[4]) if len(sys.argv) > 4 else 32) * 1024 * 1024
    tp = os.path.expanduser("~/.hf_token")
    tok = open(tp).read().strip() if os.path.exists(tp) else ""
    url = f"https://huggingface.co/{repo}/resolve/main/{fname}"
    auth = ["-H", f"Authorization: Bearer {tok}"] if tok else []
    total = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    if total <= 0:                                    # discover via ranged GET (never HEAD)
        cr = subprocess.run(["curl", "-s", "--max-time", "30", "-r", "0-0", "-D", "-",
                             "-o", "/dev/null", *auth, url],
                            capture_output=True, text=True).stdout
        for ln in cr.splitlines():
            if ln.lower().startswith("content-range") and "/" in ln:
                total = int(ln.split("/")[-1].strip())
    assert total and total > 0, f"could not get size of {fname}"
    print(f"[fetch_big] {repo}/{fname} total={total/1e9:.2f}GB seg={seg//(1024*1024)}MB -> {out}", flush=True)

    have = os.path.getsize(out) if os.path.exists(out) else 0
    tmp = out + ".seg"
    with open(out, "ab") as f:
        while have < total:
            end = min(have + seg, total) - 1
            exp = end - have + 1
            ok = False
            for attempt in range(1, 9):
                subprocess.run(["curl", "-sL", "--max-time", "90", "-o", tmp,
                                *auth, "-H", f"Range: bytes={have}-{end}", url],
                               check=False)
                got = os.path.getsize(tmp) if os.path.exists(tmp) else 0
                if got == exp:
                    with open(tmp, "rb") as s:
                        f.write(s.read()); f.flush()
                    have += got; ok = True
                    break
                print(f"[fetch_big] seg {have}-{end} got {got}/{exp} attempt {attempt}", flush=True)
                time.sleep(min(2 * attempt, 12))
            if not ok:
                print("[fetch_big] SEGMENT FAILED — aborting", flush=True); sys.exit(1)
            if have % (seg * 8) < seg or have >= total:
                print(f"[fetch_big] {have/1e9:.2f}/{total/1e9:.2f}GB ({100*have/total:.1f}%)", flush=True)
    if os.path.exists(tmp):
        os.remove(tmp)
    print(f"[fetch_big] DONE {out} = {os.path.getsize(out)} bytes", flush=True)

if __name__ == "__main__":
    main()
