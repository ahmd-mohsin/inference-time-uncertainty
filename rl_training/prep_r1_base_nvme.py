"""Fetch r1 fork's checkpoint-400 (weights only) from HF into the NVME base dir, flattened.
Writes to /tmp/instance_storage/gu/r1_<fork>_ckpt (nvme, 6.9TB, not ephemeral-metered).
Skips global_step*/ optimizer shards (vLLM base needs weights only). Usage: python prep_r1_base_nvme.py <grpo|floor>
"""
import os, sys, shutil
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
os.environ['HF_HUB_DISABLE_XET'] = '1'
NV = '/tmp/instance_storage/gu'
os.environ.setdefault('HF_HOME', NV + '/hf')
os.environ.setdefault('HF_HUB_CACHE', NV + '/hf/hub')
from huggingface_hub import HfApi, snapshot_download

fork = sys.argv[1]
tok = open(os.path.expanduser('~/.hf_token')).read().strip()
api = HfApi(token=tok)
repo = f'muahmed7338/cov-r1-{fork}-7b'
f = list(api.list_repo_files(repo, repo_type='model'))
cks = sorted({int(x.split('-')[1].split('/')[0]) for x in f if x.startswith('checkpoint-')})
ck = f'checkpoint-{cks[-1]}'
print('latest', ck, flush=True)
base = f'{NV}/r1_{fork}_ckpt'
if os.path.exists(base + '/config.json') and os.path.exists(base + '/model.safetensors'):
    print('already flat at', base, flush=True); sys.exit(0)
os.makedirs(base, exist_ok=True)
snapshot_download(repo, repo_type='model', local_dir=base + '/dl', token=tok,
                  allow_patterns=[f'{ck}/*.safetensors', f'{ck}/*.json',
                                  f'{ck}/*.jinja', f'{ck}/tokenizer*'])
src = os.path.join(base, 'dl', ck)
for fn in os.listdir(src):
    shutil.move(os.path.join(src, fn), os.path.join(base, fn))
shutil.rmtree(os.path.join(base, 'dl'), ignore_errors=True)
assert os.path.exists(base + '/config.json') and os.path.exists(base + '/model.safetensors'), 'FLATTEN FAILED'
print(f'r1 {fork} base ready (nvme):', base, flush=True)
