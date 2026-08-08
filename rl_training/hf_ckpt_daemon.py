#!/usr/bin/env python3
"""Death-proof checkpoint sync via HuggingFace Hub.

Runs on the NODE alongside training. Watches a run dir for new checkpoint-N/ subdirs and
uploads each to a private HF repo over the node's FAST internet (HTTP multipart, ~minutes for 15GB,
no git-lfs). Because the node's uplink is GB/s (it pulled the 7B model in ~2min), this beats node
death — unlike the ~2MB/s laptop SSM tunnel. On a fresh node, `resume` downloads the latest
checkpoint back (also fast) so training continues with <=1 checkpoint-interval of lost work.

Usage:
  push-daemon: python hf_ckpt_daemon.py watch  --run-dir DIR --repo USER/NAME [--every 60]
  resume:      python hf_ckpt_daemon.py resume --run-dir DIR --repo USER/NAME   # downloads latest ckpt
Token from env HF_TOKEN. Never printed.
"""
import argparse, os, sys, time, re
from huggingface_hub import HfApi

def latest_ckpt(names):
    cks = [(int(m.group(1)), n) for n in names if (m := re.match(r"checkpoint-(\d+)$", n))]
    return max(cks)[1] if cks else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["watch", "resume"])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--repo", required=True)          # e.g. muahmed7338/cov-r1-grpo-7b
    ap.add_argument("--every", type=int, default=60)  # poll seconds
    a = ap.parse_args()
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        print("ERR: HF_TOKEN not set", file=sys.stderr); sys.exit(1)
    api = HfApi(token=tok)
    api.create_repo(a.repo, private=True, exist_ok=True, repo_type="model")

    if a.mode == "resume":
        # find the highest checkpoint-N present in the repo, download it into run-dir
        try:
            files = api.list_repo_files(a.repo, repo_type="model")
        except Exception as e:
            print(f"resume: repo empty/unreadable ({e}); starting fresh"); return
        cks = latest_ckpt(sorted({f.split("/")[0] for f in files if f.startswith("checkpoint-")}))
        if not cks:
            print("resume: no checkpoint in repo; starting fresh"); return
        from huggingface_hub import snapshot_download
        dst = os.path.join(a.run_dir, cks)
        os.makedirs(dst, exist_ok=True)
        snapshot_download(a.repo, repo_type="model", allow_patterns=f"{cks}/*",
                          local_dir=a.run_dir, token=tok)
        print(f"RESUMED {cks} -> {dst}")
        return

    # watch mode: upload each new checkpoint-N as it appears, then a DONE marker
    pushed = set()
    print(f"[hf-daemon] watching {a.run_dir} -> {a.repo}, every {a.every}s")
    while True:
        try:
            names = os.listdir(a.run_dir) if os.path.isdir(a.run_dir) else []
        except Exception:
            names = []
        for n in sorted(names):
            if re.match(r"checkpoint-(\d+)$", n) and n not in pushed:
                d = os.path.join(a.run_dir, n)
                # only push once the checkpoint is fully written (model.safetensors present)
                if not os.path.exists(os.path.join(d, "model.safetensors")):
                    continue
                t0 = time.time()
                try:
                    api.upload_folder(folder_path=d, path_in_repo=n, repo_id=a.repo,
                                      repo_type="model", commit_message=f"ckpt {n}")
                    pushed.add(n)
                    print(f"[hf-daemon] pushed {n} in {int(time.time()-t0)}s", flush=True)
                except Exception as e:
                    print(f"[hf-daemon] push {n} FAILED: {e}", flush=True)
        # stop when training wrote a DONE sentinel
        if os.path.exists(os.path.join(a.run_dir, "TRAIN_DONE")) and \
           latest_ckpt(names) in pushed:
            print("[hf-daemon] TRAIN_DONE and latest pushed; exiting"); break
        time.sleep(a.every)

if __name__ == "__main__":
    main()
