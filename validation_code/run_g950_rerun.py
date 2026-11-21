#!/usr/bin/env python3
"""Re-run selected MI350X units with: test/ + tools/ trees, the tree root on PYTHONPATH, and pytest's
'not found' ids dropped and retried (pytest runs nothing if any node id is missing)."""
import grp, json, os, subprocess, sys, glob
import run_g950 as m
GIDS = ' '.join(f'--group-add {grp.getgrnam(g).gr_gid}' for g in ('video', 'render'))
RUN_SH = r'''#!/bin/bash
pip install -q --no-deps /w/WHEEL
cd /t/test
export PYTHONPATH=/t:DIRS PYTORCH_TEST_WITH_ROCM=1 TORCHINDUCTOR_CACHE_DIR=/o/ic TRITON_CACHE_DIR=/o/tc
cp /o/ids.txt /o/ids_run.txt; : > /o/missing.txt
for attempt in 1 2 3 4; do
  echo "== attempt $attempt n=$(wc -l < /o/ids_run.txt)" >> /o/run.log
  timeout 5400 python -m pytest -v -rA -p no:cacheprovider --junitxml=/o/junit.xml $(cat /o/ids_run.txt) >> /o/run.log 2>&1; rc=$?
  echo "exit=$rc" >> /o/run.log
  [ $rc -ne 4 ] && break
  grep -oE "ERROR: (file or directory )?not found: [^ ]+" /o/run.log | sed -E 's/ERROR: (file or directory )?not found: //; s#^/t/test/##' | sort -u > /o/nf.txt
  [ -s /o/nf.txt ] || break
  cat /o/nf.txt >> /o/missing.txt; grep -vxFf /o/nf.txt /o/ids_run.txt > /o/ids2.txt; mv /o/ids2.txt /o/ids_run.txt
  [ -s /o/ids_run.txt ] || break
done
echo $rc > /o/rc
'''
def tree(sha):
    d = f'{m.TREES}/{sha[:10]}_tt'
    if not os.path.isdir(f'{d}/tools'):
        os.makedirs(d, exist_ok=True); m.sh(f'git -C {m.SRC} archive {sha} test tools | tar -C {d} -xf -')
    return d
def run_side(u, side, sha, gpus, rep):
    o = f'{m.RES}/{u["pick"][:10]}/{side}/r{rep}'
    subprocess.run(f'sudo rm -rf {o}', shell=True); os.makedirs(o, exist_ok=True); os.chmod(o, 0o777)
    open(f'{o}/ids.txt', 'w').write('\n'.join(u['tests']) + '\n')
    img = open(f'{m.WHL}/{sha[:10]}/IMAGE').read().strip(); t = tree(u['pick']); w = glob.glob(f'{m.WHL}/{sha[:10]}/torch-*.whl')[0]
    dirs = ':'.join(sorted({'/t/test/' + os.path.dirname(x.split('::')[0]) for x in u['tests']}))
    open(f'{o}/run.sh', 'w').write(RUN_SH.replace('WHEEL', os.path.basename(w)).replace('DIRS', dirs)); os.chmod(f'{o}/run.sh', 0o755)
    r = m.sh(f'docker run --rm --device=/dev/kfd --device=/dev/dri {GIDS} --ipc=host --shm-size=64g --security-opt seccomp=unconfined '
             f'-e HIP_VISIBLE_DEVICES={gpus} -v {os.path.dirname(w)}:/w:ro -v {t}:/t:ro -v {o}:/o {img} bash -l /o/run.sh')
    if r.returncode != 0 and not os.path.exists(f'{o}/run.log'): raise RuntimeError(f'docker failed {side} r{rep}: {r.stderr[-300:]}')
m.run_side = run_side
if __name__ == '__main__':
    from concurrent.futures import ThreadPoolExecutor
    want = set(sys.argv[1:]); units = [u for u in json.load(open(f'{m.A}/g950_units.json')) if u['pick'][:10] in want]
    for u in units: subprocess.run(f'sudo rm -rf {m.RES}/{u["pick"][:10]}', shell=True)
    with ThreadPoolExecutor(max_workers=m.NGPU) as p: list(p.map(m.test_unit, units))
    m.log('RERUN DONE', len(units), 'units')
