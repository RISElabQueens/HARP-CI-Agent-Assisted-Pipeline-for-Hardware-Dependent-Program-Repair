#!/usr/bin/env python3
"""MI350X (gfx950) validation of agent-picked fixes, all on one host with Docker.

For every unit (agent pick = fix_addressing_commit) build gfx950 wheels for pick^ and pick inside each
commit's own CI image, then run the unit's CI-log failing tests at both, using pick's test/ tree on both
sides, REPEATS times each, inside the CI image with GPU access. Results: ~/apr/results/<pick10>/summary.json
"""
import json, os, re, subprocess, threading, time, glob, collections, xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed

A = os.path.expanduser('~/apr'); os.chdir(A)
SRC = f'{A}/src/pytorch'; WHL = f'{A}/wheels'; RES = f'{A}/results'; TREES = f'{A}/trees'
for d in (WHL, RES, TREES, f'{A}/logs', f'{A}/nosc'): os.makedirs(d, exist_ok=True)
REPEATS = 3; BUILDS = int(os.environ.get('BUILDS', '8')); MAXJ = os.environ.get('MAXJ', '90')
NGPU = 8; free = list(range(NGPU)); cv = threading.Condition(); gitlock = threading.Lock(); locks = {}; glock = threading.Lock()

def log(*a): print(time.strftime('%H:%M:%S'), *a, flush=True)
def sh(c, **kw): return subprocess.run(c, shell=True, text=True, capture_output=True, **kw)
def git(a): return sh(f'git -C {SRC} {a}').stdout.strip()
def lock(k):
    with glock: return locks.setdefault(k, threading.Lock())

def image(sha):
    wf = ''
    for f in ('periodic-rocm-mi355.yml', 'rocm-mi355.yml', 'periodic-rocm-mi300.yml'):
        wf = git(f'show {sha}:.github/workflows/{f}')
        if wf: break
    m = re.search(r'docker-image-name: ci-image:(\S+)', wf)
    name = m.group(1) if m else 'pytorch-linux-noble-rocm-n-py3'
    return f'ghcr.io/pytorch/ci-image:{name}-' + git(f'rev-parse {sha}:.ci/docker')

def pull(img):
    with lock('pull' + img):
        if sh(f'docker image inspect {img}').returncode != 0:
            log('pulling', img.split(':')[-1][:60]); r = sh(f'docker pull -q {img}'); log('pulled', r.returncode, r.stderr[-200:])

def build(sha):
    out = f'{WHL}/{sha[:10]}'
    if os.path.exists(f'{out}/DONE'): return True
    os.makedirs(out, exist_ok=True); wt = f'{A}/src/wt_{sha[:10]}'; t0 = time.time()
    with gitlock:
        sh(f'sudo rm -rf {wt}; git -C {SRC} worktree prune')
        r = sh(f'git -C {SRC} fetch -q origin {sha} && git -C {SRC} worktree add -f -q {wt} {sha} && git -C {wt} submodule update --init --recursive --jobs 16 -q')
        if r.returncode: log('checkout FAILED', sha[:10], r.stderr[-300:]); return False
        sh(f'chmod -R a+rwX {wt}; chmod 777 {out}')  # CI images build as user jenkins, not as the checkout owner
    img = image(sha); pull(img)
    cmd = (f'docker run --rm -v {wt}:/src -v {out}:/out -v {A}/nosc:/nosc:ro -w /src '
           f'-e PYTORCH_ROCM_ARCH=gfx950 -e USE_ROCM=1 -e BUILD_TEST=0 -e USE_CUDA=0 -e MAX_JOBS={MAXJ} -e SCCACHE_DISABLE=1 '
           f'{img} bash -lc "export PATH=/nosc:\\$PATH; python tools/amd_build/build_amd.py >/dev/null && '
           f'pip wheel --no-build-isolation --no-deps -v . -w /tmp/dist > /out/build.log 2>&1 && cp /tmp/dist/torch-*.whl /out/"')
    r = sh(cmd); ok = r.returncode == 0 and glob.glob(f'{out}/torch-*.whl')
    if ok:
        open(f'{out}/COMMIT', 'w').write(sha + '\n'); open(f'{out}/IMAGE', 'w').write(img + '\n'); open(f'{out}/DONE', 'w').close()
    log('build', 'OK' if ok else 'FAILED', sha[:10], f'{(time.time()-t0)/60:.0f} min', '' if ok else (r.stderr[-200:] + sh(f'tail -5 {out}/build.log').stdout))
    with gitlock: sh(f'sudo rm -rf {wt}; git -C {SRC} worktree prune')
    return bool(ok)

def tree(sha):
    d = f'{TREES}/{sha[:10]}'
    with lock('tree' + sha):
        if not os.path.isdir(f'{d}/test'):
            os.makedirs(d, exist_ok=True); sh(f'git -C {SRC} archive {sha} test | tar -C {d} -xf -')
    return d

def run_side(u, side, sha, gpus, rep):
    o = f'{RES}/{u["pick"][:10]}/{side}/r{rep}'; os.makedirs(o, exist_ok=True)
    os.chmod(o, 0o777)  # test containers run as user jenkins
    open(f'{o}/ids.txt', 'w').write('\n'.join(u['tests']) + '\n')
    img = open(f'{WHL}/{sha[:10]}/IMAGE').read().strip(); t = tree(u['pick']); w = glob.glob(f'{WHL}/{sha[:10]}/torch-*.whl')[0]
    dirs = ':'.join(sorted({'/t/test/' + os.path.dirname(x.split('::')[0]) for x in u['tests']}))
    inner = ('pip install -q --no-deps /w/' + os.path.basename(w) + ' && cd /t/test && '
             f'PYTHONPATH={dirs} PYTORCH_TEST_WITH_ROCM=1 TORCHINDUCTOR_CACHE_DIR=/o/ic TRITON_CACHE_DIR=/o/tc '
             'timeout 5400 python -m pytest -v -rA -p no:cacheprovider --junitxml=/o/junit.xml $(cat /o/ids.txt) > /o/run.log 2>&1; echo $? > /o/rc')
    sh(f'docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --group-add render --ipc=host --shm-size=64g '
       f'--security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES={gpus} -v {os.path.dirname(w)}:/w:ro -v {t}:/t:ro -v {o}:/o {img} bash -lc "{inner}"')

def parse(d):
    out = {}
    for xf in glob.glob(f'{d}/junit*.xml'):
        try: root = ET.parse(xf).getroot()
        except Exception: continue
        for tc in root.iter('testcase'):
            st = 'passed'
            for t, v in (('failure', 'failed'), ('error', 'error'), ('skipped', 'skipped')):
                if tc.find(t) is not None: st = v
            out[(tc.get('classname', '').split('.')[-1], tc.get('name'))] = st
    return out

def summarize(u):
    agg = {}; bad = lambda s: s in ('failed', 'error')
    key = lambda t: ((t.split('::')[-2] if t.count('::') >= 2 else ''), t.split('::')[-1])
    for t in u['tests']:
        b = [parse(f'{RES}/{u["pick"][:10]}/base/r{i}').get(key(t), 'not_run') for i in range(REPEATS)]
        f = [parse(f'{RES}/{u["pick"][:10]}/fix/r{i}').get(key(t), 'not_run') for i in range(REPEATS)]
        k = ('F2P' if all(map(bad, b)) and all(x == 'passed' for x in f) else 'P2P' if all(x == 'passed' for x in b + f) else
             'F2F' if all(map(bad, b + f)) else 'FLAKY' if any(map(bad, b + f)) and any(x == 'passed' for x in b + f) else 'OTHER')
        agg[t] = {'base': b, 'fix': f, 'class': k}
    json.dump({'pick': u['pick'], 'parent': u['parent'], 'cases': u['cases'], 'tests': agg}, open(f'{RES}/{u["pick"][:10]}/summary.json', 'w'), indent=1)
    return collections.Counter(v['class'] for v in agg.values())

def test_unit(u):
    need = 4 if u['distributed'] else 1
    with cv:
        while len(free) < need: cv.wait()
        mine = [free.pop(0) for _ in range(need)]
    try:
        g = ','.join(map(str, mine))
        for rep in range(REPEATS):
            for side, sha in (('base', u['parent']), ('fix', u['pick'])): run_side(u, side, sha, g, rep)
        log('unit', u['pick'][:10], 'cases', len(u['cases']), '->', dict(summarize(u)))
    except Exception as e: log('unit', u['pick'][:10], 'ERROR', repr(e))
    finally:
        with cv: free.extend(mine); cv.notify_all()

if __name__ == '__main__':
    units = json.load(open(f'{A}/g950_units.json'))
    if not os.path.isdir(f'{SRC}/.git'): sh(f'git clone -q --filter=blob:none https://github.com/pytorch/pytorch.git {SRC}')
    open(f'{A}/nosc/sccache', 'w').write('#!/bin/sh\nexec "$@"\n'); os.chmod(f'{A}/nosc/sccache', 0o755)
    sh(f'git -C {SRC} fetch -q origin ' + ' '.join(u['pick'] for u in units))
    for u in units: u['parent'] = git(f'rev-parse {u["pick"]}^')
    json.dump(units, open(f'{A}/g950_units.json', 'w'), indent=1)
    shas = list(dict.fromkeys(s for u in units for s in (u['parent'], u['pick'])))
    log(len(units), 'units,', len(shas), 'wheels to build')
    done = {}; tpool = ThreadPoolExecutor(max_workers=NGPU); started = set()
    def maybe_test():
        for u in units:
            if u['pick'] not in started and done.get(u['parent']) and done.get(u['pick']) and not os.path.exists(f'{RES}/{u["pick"][:10]}/summary.json'):
                started.add(u['pick']); tpool.submit(test_unit, u)
    with ThreadPoolExecutor(max_workers=BUILDS) as bp:
        futs = {bp.submit(build, s): s for s in shas}
        for f in as_completed(futs):
            done[futs[f]] = f.result(); maybe_test()
    tpool.shutdown(wait=True); log('ALL DONE')
