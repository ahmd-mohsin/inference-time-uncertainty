#!/usr/bin/env python3
# Scan ~/gu/p_*/ pack cells, print tag + status + base/rft/rvp pass@1 (+cov). No shell quoting needed.
import glob, json, os
for d in sorted(glob.glob(os.path.expanduser('~/gu/p_*/'))):
    res = d + 'RES.md'
    if not os.path.exists(res): continue
    done = 'CELL1GPU_DONE' in open(res).read()
    out = {}
    for a in ('base', 'rft', 'rvp'):
        f = d + 'ev_%s.json' % a
        if os.path.exists(f):
            try:
                j = json.load(open(f)); out[a] = (j['pass1'], j.get('coverage_passk', 0))
            except Exception: pass
    if out:
        s = ' '.join('%s=%.4f/%.3f' % (k, v[0], v[1]) for k, v in out.items())
        print('%-22s %-7s %s' % (os.path.basename(d.rstrip('/')), 'DONE' if done else 'part', s))
