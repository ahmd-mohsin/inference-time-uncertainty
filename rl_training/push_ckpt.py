"""Manually push one local checkpoint dir to its HF repo. Usage: python push_ckpt.py <fork> <N>"""
import os, sys
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import HfApi

fork, n = sys.argv[1], sys.argv[2]
tok = open(os.path.expanduser("~/.hf_token")).read().strip()
api = HfApi(token=tok)
d = f"/tmp/instance_storage/gu/r2_from_{fork}/checkpoint-{n}"
repo = f"muahmed7338/cov-r2-from-{fork}-7b"
assert os.path.exists(os.path.join(d, "model.safetensors")), f"no model in {d}"
api.upload_folder(folder_path=d, path_in_repo=f"checkpoint-{n}", repo_id=repo, repo_type="model",
                  commit_message=f"manual push ckpt-{n}")
print(f"PUSHED checkpoint-{n} -> {repo}")
