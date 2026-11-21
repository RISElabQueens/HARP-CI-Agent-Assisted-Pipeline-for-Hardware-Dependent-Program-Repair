#!/usr/bin/env python3
"""Build gfx90a (MI210) wheels for the gfx90a units on this host, inside each commit's own CI image.
The image is the one the commit's workflows use for the case's build environment (e.g. linux-jammy-rocm-py3.10-mi200).
Output: ~/apr/wheels_g90a/<sha10>/{torch-*.whl,COMMIT,IMAGE,DONE}. Testing runs elsewhere (MI210 pod)."""
import glob, json, os, re, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import run_g950 as m

WHL = f'{m.A}/wheels_g90a'; os.makedirs(WHL, exist_ok=True)

def image_for(sha, env):
    hits = m.git(f'grep -n "build-environment: {env}$" {sha} -- .github/workflows').splitlines()
    for h in hits:
        _, path, line = h.split(':', 2)[0], h.split(':')[1], int(h.split(':')[2])
        body = m.git(f'show {sha}:{path}').splitlines()
        for l in body[line - 1: line + 8]:
            mm = re.search(r'docker-image-name: ci-image:(\S+)', l)
            if mm: return f'ghcr.io/pytorch/ci-image:{mm.group(1)}-' + m.git(f'rev-parse {sha}:.ci/docker')
    name = 'pytorch-linux-jammy-rocm-n-py3'
    return f'ghcr.io/pytorch/ci-image:{name}-' + m.git(f'rev-parse {sha}:.ci/docker')

def build(sha, env):
    out = f'{WHL}/{sha[:10]}'
    if os.path.exists(f'{out}/DONE'): return True
    os.makedirs(out, exist_ok=True); wt = f'{m.A}/src/wt90_{sha[:10]}'; t0 = time.time()
    with m.gitlock:
        m.sh(f'sudo rm -rf {wt}; git -C {m.SRC} worktree prune')
        r = m.sh(f'git -C {m.SRC} fetch -q origin {sha} && git -C {m.SRC} worktree add -f -q {wt} {sha} && git -C {wt} submodule update --init --recursive --jobs 16 -q')
        if r.returncode: m.log('checkout FAILED', sha[:10], r.stderr[-300:]); return False
        m.sh(f'chmod -R a+rwX {wt}; chmod 777 {out}')
    img = image_for(sha, env); m.pull(img)
    cmd = (f'docker run --rm -v {wt}:/src -v {out}:/out -v {m.A}/nosc:/nosc:ro -w /src '
           f'-e PYTORCH_ROCM_ARCH=gfx90a -e USE_ROCM=1 -e BUILD_TEST=0 -e USE_CUDA=0 -e MAX_JOBS={m.MAXJ} -e SCCACHE_DISABLE=1 '
           f'{img} bash -lc "export PATH=/nosc:\\$PATH; python tools/amd_build/build_amd.py >/dev/null && '
           f'pip wheel --no-build-isolation --no-deps -v . -w /tmp/dist > /out/build.log 2>&1 && cp /tmp/dist/torch-*.whl /out/"')
    r = m.sh(cmd); ok = r.returncode == 0 and glob.glob(f'{out}/torch-*.whl')
    if ok:
        open(f'{out}/COMMIT', 'w').write(sha + '\n'); open(f'{out}/IMAGE', 'w').write(img + '\n'); open(f'{out}/DONE', 'w').close()
    m.log('build90', 'OK' if ok else 'FAILED', sha[:10], img.split(':')[-1][:40], f'{(time.time()-t0)/60:.0f} min', '' if ok else m.sh(f'tail -5 {out}/build.log').stdout)
    with m.gitlock: m.sh(f'sudo rm -rf {wt}; git -C {m.SRC} worktree prune')
    return bool(ok)

if __name__ == '__main__':
    units = json.load(open(f'{m.A}/g90a_units.json')); cases = {c['case']: c for c in json.load(open(f'{m.A}/g90a_cases.json'))}
    m.sh(f'git -C {m.SRC} fetch -q origin ' + ' '.join(u['pick'] for u in units))
    jobs = {}
    for u in units:
        u['parent'] = m.git(f'rev-parse {u["pick"]}^')
        env = re.sub(r'\s*/.*', '', cases[u['cases'][0]]['job']).strip()
        u['build_env'] = env
        for s in (u['parent'], u['pick']): jobs.setdefault(s, env)
    json.dump(units, open(f'{m.A}/g90a_units.json', 'w'), indent=1)
    m.log(len(units), 'gfx90a units,', len(jobs), 'wheels to build')
    with ThreadPoolExecutor(max_workers=m.BUILDS) as p:
        for f in as_completed([p.submit(build, s, e) for s, e in jobs.items()]): f.result()
    m.log('BUILD90 ALL DONE')
