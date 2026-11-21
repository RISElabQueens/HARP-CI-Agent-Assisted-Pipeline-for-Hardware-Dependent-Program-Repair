#!/usr/bin/env python3
"""Extract the runtime parts of a public ghcr image into a directory, without Docker.

usage: oci_extract.py ghcr.io/pytorch/ci-image:<tag> <dest>
Downloads the linux/amd64 manifest's layers in order and extracts only the paths a test run needs:
opt/ (ROCm, conda/venv), usr/bin/python3*, usr/lib/python3*, usr/include/python3* and the multiarch pyconfig dir.
Whiteout files are honoured for those paths. Writes <dest>/.ok when complete.
"""
import io, json, os, shutil, sys, tarfile, urllib.request

KEEP = ('opt/', 'usr/bin/python3', 'usr/lib/python3', 'usr/include/python3', 'usr/include/x86_64-linux-gnu/python3',
        'usr/lib/x86_64-linux-gnu/libpython3', 'usr/lib/x86_64-linux-gnu/libgcc_s', 'usr/lib/x86_64-linux-gnu/libstdc++')

def req(url, token=None, accept=None):
    h = {}
    if token: h['Authorization'] = f'Bearer {token}'
    if accept: h['Accept'] = accept
    return urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=600)

def fetch_blob(repo, layer, token, dest, tries=8):
    import fcntl
    cache = os.environ.get('BLOB_CACHE', '/tmp/apr/blobs'); os.makedirs(cache, exist_ok=True)
    with open(os.path.join(cache, layer['digest'].split(':')[1] + '.lock'), 'w') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)  # another extraction may be fetching the same shared layer
        return _fetch_blob(repo, layer, token, dest, tries)

def _fetch_blob(repo, layer, token, dest, tries=8):
    """Download a layer to a file, resuming with HTTP Range after drops, and verify its sha256."""
    import hashlib, time
    cache = os.environ.get('BLOB_CACHE', '/tmp/apr/blobs'); os.makedirs(cache, exist_ok=True)  # layers are shared across CI images
    path = os.path.join(cache, layer['digest'].split(':')[1]); size = layer['size']
    if os.path.exists(path + '.verified'): return path
    for attempt in range(tries):
        have = os.path.getsize(path) if os.path.exists(path) else 0
        if have < size:
            try:
                token = json.load(req(f'https://ghcr.io/token?scope=repository:{repo}:pull'))['token']  # ghcr tokens expire in minutes
                h = {'Authorization': f'Bearer {token}'}
                if have: h['Range'] = f'bytes={have}-'
                with urllib.request.urlopen(urllib.request.Request(f'https://ghcr.io/v2/{repo}/blobs/{layer["digest"]}', headers=h), timeout=600) as r, open(path, 'ab') as out:
                    shutil.copyfileobj(r, out, 8 << 20)
            except Exception as e:
                print('retry', attempt, e, flush=True); time.sleep(10); continue
        sha = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8 << 20), b''): sha.update(chunk)
        if 'sha256:' + sha.hexdigest() == layer['digest']:
            open(path + '.verified', 'w').close(); return path
        print('digest mismatch, refetching', flush=True); os.remove(path)
    raise SystemExit('blob download failed: ' + layer['digest'])

def main(ref, dest):
    repo, tag = ref.split('/', 1)[1].rsplit(':', 1)
    token = json.load(req(f'https://ghcr.io/token?scope=repository:{repo}:pull'))['token']
    acc = ('application/vnd.oci.image.index.v1+json,application/vnd.docker.distribution.manifest.list.v2+json,'
           'application/vnd.oci.image.manifest.v1+json,application/vnd.docker.distribution.manifest.v2+json')
    man = json.load(req(f'https://ghcr.io/v2/{repo}/manifests/{tag}', token, acc))
    if 'manifests' in man:
        d = next(x for x in man['manifests'] if x.get('platform', {}).get('architecture') == 'amd64')['digest']
        man = json.load(req(f'https://ghcr.io/v2/{repo}/manifests/{d}', token, acc))
    os.makedirs(dest, exist_ok=True)
    for i, layer in enumerate(man['layers']):
        mt = layer['mediaType']
        if 'zstd' in mt: raise SystemExit(f'zstd layer not supported: {mt}')
        print(f'layer {i+1}/{len(man["layers"])} {layer["size"]/1e9:.1f} GB', flush=True)
        blob = fetch_blob(repo, layer, token, dest)
        with open(blob, 'rb') as fh, tarfile.open(fileobj=fh, mode='r|*') as tf:
            for m in tf:
                name = m.name.lstrip('./')
                base, leaf = os.path.split(name)
                if leaf.startswith('.wh.'):
                    target = os.path.join(dest, base, leaf[4:]) if leaf != '.wh..wh..opq' else None
                    if target and name.startswith(KEEP) or (target and any(os.path.join(base, leaf[4:]).startswith(k) for k in KEEP)):
                        if os.path.isdir(target) and not os.path.islink(target): shutil.rmtree(target, ignore_errors=True)
                        elif os.path.lexists(target): os.remove(target)
                    continue
                if not name.startswith(KEEP) and not any(k.startswith(name + '/') for k in KEEP): continue
                try:
                    p = os.path.join(dest, name)
                    if os.path.lexists(p) and not os.path.isdir(p): os.remove(p)
                    tf.extract(m, dest, set_attrs=True, numeric_owner=True)  # keep mode bits (executables)
                except Exception as e:
                    print('skip', name, e, file=sys.stderr)
    open(os.path.join(dest, '.ok'), 'w').close(); print('done', dest)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
