import os
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
os.environ['HF_HUB_DISABLE_XET'] = '1'
import shutil
from huggingface_hub import HfApi, snapshot_download

tok = open(os.path.expanduser('~/.hf_token')).read().strip()
api = HfApi(token=tok)
repo = 'muahmed7338/cov-r1-floor-7b'
f = list(api.list_repo_files(repo, repo_type='model'))
cks = sorted({int(x.split('-')[1].split('/')[0]) for x in f if x.startswith('checkpoint-')})
ck = f'checkpoint-{cks[-1]}'
print('latest', ck, flush=True)
base = os.path.expanduser('~/inference-time-uncertainty/rl_training/runs/r1_floor_ckpt')
os.makedirs(base, exist_ok=True)
# model weights + config + tokenizer only; SKIP global_step*/ optimizer (not needed for vLLM base)
snapshot_download(repo, repo_type='model', local_dir=base + '/dl', token=tok,
                  allow_patterns=[f'{ck}/*.safetensors', f'{ck}/*.json',
                                  f'{ck}/*.jinja', f'{ck}/tokenizer*'])
src = os.path.join(base, 'dl', ck)
for fn in os.listdir(src):
    shutil.move(os.path.join(src, fn), os.path.join(base, fn))
shutil.rmtree(os.path.join(base, 'dl'), ignore_errors=True)
assert os.path.exists(base + '/config.json') and os.path.exists(base + '/model.safetensors'), 'FLATTEN FAILED'
print('floor base ready (lite)', flush=True)
