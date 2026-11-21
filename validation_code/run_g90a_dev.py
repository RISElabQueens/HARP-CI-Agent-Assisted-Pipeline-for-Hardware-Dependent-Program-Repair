#!/usr/bin/env python3
"""Test the gfx90a units on the MI210 inside the provided pod (namespace <namespace>, 1 GPU).

Each commit runs in its own CI runtime: the commit's CI image is unpacked inside the pod (oci_extract.py) to
/tmp/apr/img/<tag12>, and env.sh there points Python/ROCm/Triton at it. Wheels are streamed from the MI350X
build host and installed into /tmp/apr/site/<sha10>; tests come from the pick's test/+tools/ tree.
Results: <outdir>/validation/g90a_agentpick/<pick10>/{base,fix}/r*/ and summary.json
"""
import collections, glob, json, os, subprocess, sys, threading, time, xml.etree.ElementTree as ET

KUBE = ['kubectl', '--kubeconfig', os.path.expanduser('~/.kube/kubeconfig.yaml')]
HOST = '<build-host>'; SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'LogLevel=ERROR', HOST]
OUT = '<outdir>/validation/g90a_agentpick'; REPEATS = 3; os.makedirs(OUT, exist_ok=True)
locks = {}; glock = threading.Lock()

def log(*a): print(time.strftime('%H:%M:%S'), *a, flush=True)
def run(cmd, **kw): return subprocess.run(cmd, text=True, capture_output=True, **kw)
def pod(): return run(KUBE + ['get', 'pods', '-o', 'name']).stdout.split()[0].split('/')[1]
def kx(*args, inp=None): return run(KUBE + ['exec'] + (['-i'] if inp is not None else []) + [pod(), '--'] + list(args), input=inp)
def lock(k):
    with glock: return locks.setdefault(k, threading.Lock())

ENV_SH = r'''# runtime env for an unpacked CI image at ROOT
R=ROOT
if ls -d $R/opt/conda/envs/py_3.* >/dev/null 2>&1; then
  E=$(ls -d $R/opt/conda/envs/py_3.* | head -1); export PY=$E/bin/python
  RO=$(ls -d $R/opt/rocm-* 2>/dev/null | head -1); LDX=
  if [ -z "$RO" ]; then  # ROCm >= 10 ships as pip packages inside the conda env
    RO=$(ls -d $E/lib/python3.*/site-packages/_rocm_sdk_devel | head -1); LDX=$RO/lib/host-math/lib:$RO/lib/rocm_sysdeps/lib:
  fi
  export ROCM_PATH=$RO ROCM_HOME=$RO HIP_PATH=$RO PYTHONPATH_BASE=
  export LD_LIBRARY_PATH=$RO/lib:$LDX$RO/lib/llvm/lib:$E/lib TRITON_HIP_LLD_PATH=$RO/lib/llvm/bin/ld.lld
  export PATH=$RO/bin:$RO/lib/llvm/bin:$E/bin:/usr/bin:/bin
else
  V=$(ls -d $R/opt/python-3.*-venv | head -1); PV=$(basename $V | sed -E 's/python-(3\.[0-9]+)-venv/\1/')
  SP=$V/lib/python$PV/site-packages; RO=$SP/_rocm_sdk_devel
  mkdir -p $R/sitecust; echo "import site; site.addsitedir('$SP')" > $R/sitecust/sitecustomize.py
  export PY=$R/usr/bin/python$PV PYTHONHOME=$R/usr PYTHONPATH_BASE=$R/sitecust
  export ROCM_PATH=$RO ROCM_HOME=$RO HIP_PATH=$RO TRITON_HIP_LLD_PATH=$RO/lib/llvm/bin/ld.lld C_INCLUDE_PATH=$R/usr/include
  export LD_LIBRARY_PATH=$RO/lib:$RO/lib/host-math/lib:$RO/lib/rocm_sysdeps/lib:$RO/lib/llvm/lib:$R/usr/lib/x86_64-linux-gnu  # image's newer libgcc_s/libstdc++
  export PATH=$RO/bin:$RO/lib/llvm/bin:/usr/bin:/bin
fi
'''
RUN_SH = r'''#!/bin/bash
# usage: run.sh <env_root> <site_dir> <tree_dir> <out_dir>
. $1/env.sh; S=$2 T=$3 O=$4; rm -rf -- "$O"; mkdir -p "$O"; cd $T/test  # fresh output dir per run
export PYTHONPATH=${PYTHONPATH_BASE:+$PYTHONPATH_BASE:}$S:$T:$(cut -d: -f1 $O/../../ids.txt | xargs -n1 dirname | sort -u | sed "s#^#$T/test/#" | paste -sd:)
export HIP_VISIBLE_DEVICES=0 PYTORCH_TEST_WITH_ROCM=1 TORCHINDUCTOR_CACHE_DIR=$O/ic TRITON_CACHE_DIR=$O/tc
cp $O/../../ids.txt $O/ids_run.txt; : > $O/missing.txt
for attempt in 1 2 3 4; do
  echo "== attempt $attempt n=$(wc -l < $O/ids_run.txt)" >> $O/run.log
  timeout 5400 $PY -m pytest -v -rA -p no:cacheprovider --junitxml=$O/junit.xml $(cat $O/ids_run.txt) >> $O/run.log 2>&1; rc=$?
  echo "exit=$rc" >> $O/run.log
  [ $rc -ne 4 ] && break
  grep -oE "ERROR: (file or directory )?not found: [^ ]+" $O/run.log | sed -E "s/ERROR: (file or directory )?not found: //; s#^$T/test/##" | sort -u > $O/nf.txt
  [ -s $O/nf.txt ] || break
  cat $O/nf.txt >> $O/missing.txt; grep -vxFf $O/nf.txt $O/ids_run.txt > $O/ids2.txt; mv $O/ids2.txt $O/ids_run.txt
  [ -s $O/ids_run.txt ] || break
done
echo $rc > $O/rc
'''

def ensure_image(img):
    tag = img.split(':')[-1]; key = tag[-40:][:12]; root = f'/tmp/apr/img/{key}'
    with lock('img' + key):
        if kx('test', '-f', f'{root}/.ok').returncode != 0:
            log('extracting', tag[:60]); r = kx('python', '/tmp/oci_extract.py', img, root)
            if r.returncode: raise RuntimeError(f'extract {tag}: {r.stderr[-300:]}')
        kx('sh', '-c', f'cat > {root}/env.sh', inp=ENV_SH.replace('ROOT', root))  # always refresh
    return root

def ensure_wheel(sha, root):
    s10 = sha[:10]; site = f'/tmp/apr/site/{s10}'
    with lock('whl' + s10):
        if kx('test', '-f', f'{site}/.ok').returncode == 0: return site
        name = run(SSH + [f'ls apr/wheels_g90a/{s10}/']).stdout.split(); whl = [n for n in name if n.endswith('.whl')][0]
        kx('mkdir', '-p', f'/tmp/apr/whl/{s10}')
        p = subprocess.run(f'{" ".join(SSH)} cat apr/wheels_g90a/{s10}/{whl} | {" ".join(KUBE)} exec -i {pod()} -- sh -c "cat > /tmp/apr/whl/{s10}/{whl}"', shell=True)
        r = kx('bash', '-c', f'. {root}/env.sh; export PYTHONPATH=$PYTHONPATH_BASE; $PY -m pip install -q --no-deps --target {site} /tmp/apr/whl/{s10}/{whl} && touch {site}/.ok')
        if r.returncode: raise RuntimeError(f'install {s10}: {r.stderr[-400:]}')
    return site

def ensure_tree(sha):
    t = f'/tmp/apr/trees/{sha[:10]}'
    with lock('tree' + sha):
        if kx('test', '-d', f'{t}/test').returncode != 0:
            kx('mkdir', '-p', t)
            subprocess.run(f'{" ".join(SSH)} git -C apr/src/pytorch archive {sha} test tools | {" ".join(KUBE)} exec -i {pod()} -- tar -C {t} -xf -', shell=True)
    return t

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

def summarize(u, d):
    bad = lambda s: s in ('failed', 'error'); key = lambda t: ((t.split('::')[-2] if t.count('::') >= 2 else ''), t.split('::')[-1])
    agg = {}
    for t in u['tests']:
        b = [parse(f'{d}/base/r{i}').get(key(t), 'not_run') for i in range(REPEATS)]
        f = [parse(f'{d}/fix/r{i}').get(key(t), 'not_run') for i in range(REPEATS)]
        c = ('F2P' if all(map(bad, b)) and all(x == 'passed' for x in f) else 'P2P' if all(x == 'passed' for x in b + f) else
             'F2F' if all(map(bad, b + f)) else 'FLAKY' if any(map(bad, b + f)) and any(x == 'passed' for x in b + f) else 'OTHER')
        agg[t] = {'base': b, 'fix': f, 'class': c}
    json.dump({'pick': u['pick'], 'parent': u['parent'], 'cases': u['cases'], 'tests': agg}, open(f'{d}/summary.json', 'w'), indent=1)
    return collections.Counter(v['class'] for v in agg.values())

def test_unit(u, images):
    p10 = u['pick'][:10]; d = f'{OUT}/{p10}'; rd = f'/tmp/apr/res/{p10}'
    roots = {s: ensure_image(images[s[:10]]) for s in (u['parent'], u['pick'])}
    sites = {s: ensure_wheel(s, roots[s]) for s in (u['parent'], u['pick'])}
    tree = ensure_tree(u['pick'])
    kx('mkdir', '-p', rd); kx('sh', '-c', f'cat > {rd}/ids.txt', inp='\n'.join(u['tests']) + '\n')
    kx('sh', '-c', 'cat > /tmp/apr/run.sh && chmod +x /tmp/apr/run.sh', inp=RUN_SH)
    for rep in range(REPEATS):
        for side, sha in (('base', u['parent']), ('fix', u['pick'])):
            kx('bash', '/tmp/apr/run.sh', roots[sha], sites[sha], tree, f'{rd}/{side}/r{rep}')
    os.makedirs(d, exist_ok=True)
    subprocess.run(f'{" ".join(KUBE)} exec {pod()} -- tar -C {rd} --exclude=ic --exclude=tc -cf - . | tar -C {d} -xf -', shell=True)
    log('unit', p10, 'cases', len(u['cases']), '->', dict(summarize(u, d)))

if __name__ == '__main__':
    units = json.loads(run(SSH + ['cat apr/g90a_units.json']).stdout); images = json.loads(run(SSH + ['cat apr/g90a_images.json']).stdout)
    kx('mkdir', '-p', '/tmp/apr'); subprocess.run(KUBE + ['cp', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'oci_extract.py'), f'{pod()}:/tmp/oci_extract.py'])
    todo = [u for u in units if not u['distributed'] and not os.path.exists(f'{OUT}/{u["pick"][:10]}/summary.json')]
    log(len(todo), 'single-GPU units to test;', sum(u['distributed'] for u in units), 'distributed units need a 2-GPU quota')
    # pre-extract images in the background (3 at a time) while waiting for wheels
    from concurrent.futures import ThreadPoolExecutor
    def pre_extract(img):
        try: ensure_image(img)
        except Exception as e: log('pre-extract ERROR', img.split(':')[-1][:50], repr(e)[:300])
    pre = ThreadPoolExecutor(max_workers=3); [pre.submit(pre_extract, img) for img in sorted(set(images.values()))]
    while todo:
        for u in list(todo):
            if run(SSH + [f'test -f apr/wheels_g90a/{u["parent"][:10]}/DONE -a -f apr/wheels_g90a/{u["pick"][:10]}/DONE']).returncode == 0:
                try: test_unit(u, images)
                except Exception as e: log('unit', u['pick'][:10], 'ERROR', repr(e)[:400])
                todo.remove(u)
        if todo: time.sleep(300)
    pre.shutdown(wait=True); log('G90A TESTS ALL DONE')
