#!/usr/bin/env python3
"""Tester for the MI350X run, alongside run_g950.py (which keeps building wheels).

run_g950.py's test step passed `--group-add render`, which fails in CI images that have no 'render' group, so
its units never actually ran. This process reuses its functions with a fixed run_side (numeric render/video
GIDs, docker exit status checked) and tests every unit whose wheels exist and that has no valid result yet.
A unit is valid once all its runs produced run.log; marker: results/<pick10>/.valid
"""
import grp, json, os, subprocess, time, glob
import run_g950 as m

GIDS = ' '.join(f'--group-add {grp.getgrnam(g).gr_gid}' for g in ('video', 'render'))

def run_side(u, side, sha, gpus, rep):
    o = f'{m.RES}/{u["pick"][:10]}/{side}/r{rep}'
    subprocess.run(f'sudo rm -rf {o}', shell=True); os.makedirs(o, exist_ok=True); os.chmod(o, 0o777)
    open(f'{o}/ids.txt', 'w').write('\n'.join(u['tests']) + '\n')
    img = open(f'{m.WHL}/{sha[:10]}/IMAGE').read().strip(); t = m.tree(u['pick']); w = glob.glob(f'{m.WHL}/{sha[:10]}/torch-*.whl')[0]
    dirs = ':'.join(sorted({'/t/test/' + os.path.dirname(x.split('::')[0]) for x in u['tests']}))
    inner = ('pip install -q --no-deps /w/' + os.path.basename(w) + ' && cd /t/test && '
             f'PYTHONPATH={dirs} PYTORCH_TEST_WITH_ROCM=1 TORCHINDUCTOR_CACHE_DIR=/o/ic TRITON_CACHE_DIR=/o/tc '
             'timeout 5400 python -m pytest -v -rA -p no:cacheprovider --junitxml=/o/junit.xml \\$(cat /o/ids.txt) > /o/run.log 2>&1; echo \\$? > /o/rc')
    r = m.sh(f'docker run --rm --device=/dev/kfd --device=/dev/dri {GIDS} --ipc=host --shm-size=64g '
             f'--security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES={gpus} -v {os.path.dirname(w)}:/w:ro -v {t}:/t:ro -v {o}:/o {img} bash -lc "{inner}"')
    if r.returncode != 0 and not os.path.exists(f'{o}/run.log'):
        raise RuntimeError(f'docker failed for {side} r{rep}: {r.stderr[-300:]}')

m.run_side = run_side  # test_unit looks it up at call time

def wheels_ready(u): return all(os.path.exists(f'{m.WHL}/{s[:10]}/DONE') for s in (u['parent'], u['pick']))
def valid(u):
    d = f'{m.RES}/{u["pick"][:10]}'
    return all(os.path.exists(f'{d}/{s}/r{i}/run.log') for s in ('base', 'fix') for i in range(m.REPEATS))

if __name__ == '__main__':
    from concurrent.futures import ThreadPoolExecutor
    units = json.load(open(f'{m.A}/g950_units.json')); pool = ThreadPoolExecutor(max_workers=m.NGPU); started = set()
    def job(u):
        m.test_unit(u)
        if valid(u): open(f'{m.RES}/{u["pick"][:10]}/.valid', 'w').close()
    while True:
        for u in units:
            if u['pick'] in started or os.path.exists(f'{m.RES}/{u["pick"][:10]}/.valid'): continue
            if wheels_ready(u): started.add(u['pick']); pool.submit(job, u); m.log('tester queued', u['pick'][:10])
        if all(os.path.exists(f'{m.RES}/{u["pick"][:10]}/.valid') for u in units): break
        if len(started) == len(units) and pool._work_queue.empty() and not any(t.is_alive() for t in pool._threads if False): pass
        time.sleep(60)
    pool.shutdown(wait=True); m.log('TESTER ALL DONE')
