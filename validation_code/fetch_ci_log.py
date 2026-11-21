#!/usr/bin/env python3
"""Download PyTorch GitHub Actions job logs by job id, including logs GitHub has expired.

Source order:
  1. PyTorch's public S3 mirror  https://ossci-raw-job-status.s3.amazonaws.com/log/<job_id>
     (written by PyTorch CI infra; outlives GitHub's ~90-day log retention)
  2. GitHub API  GET /repos/pytorch/pytorch/actions/jobs/<job_id>/logs
     (only if GH_TOKEN / GITHUB_TOKEN is set; returns 410 once expired)

A mirror entry can hold an error body instead of a log (e.g. a GitHub
"API rate limit exceeded" JSON captured at upload time), so content is checked
before it is accepted.

Usage:
  fetch_ci_log.py 102313187406 85487999320 ...         # job ids
  fetch_ci_log.py --csv cases.csv [--col job_id]       # ids from a CSV column
  options: --out DIR (default ./ci_logs)  --parse  --no-github  --keep-timestamps

Writes DIR/<job_id>.log (plain text) and DIR/index.json with per-job source,
size and status; with --parse also the failing / flaky tests found in the log.
Stdlib only.
"""
import argparse, csv, gzip, json, os, re, sys, urllib.error, urllib.request

S3 = "https://ossci-raw-job-status.s3.amazonaws.com/log/{}"
GH = "https://api.github.com/repos/pytorch/pytorch/actions/jobs/{}/logs"
TS = re.compile(r"^\d{4}-\d\d-\d\dT[\d:.]+Z ", re.M)


def http_get(url, headers=None, timeout=120):
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except urllib.error.URLError as e:
        return None, str(e.reason).encode()


def decode(body):
    try:
        body = gzip.decompress(body)
    except OSError:
        pass
    return body.decode("utf-8", "replace")


def looks_like_log(text):
    """Reject stored error bodies (JSON error objects, tiny responses)."""
    head = text.lstrip()[:400]
    if head.startswith("{") and '"message"' in head:
        return False, "stored error body: " + re.sub(r"\s+", " ", head)[:160]
    if len(text) < 2000:
        return False, f"too small ({len(text)} bytes)"
    return True, ""


def fetch(job_id, use_github=True):
    status, body = http_get(S3.format(job_id))
    tried = [f"s3:{status}"]
    if status == 200:
        text = decode(body)
        ok, why = looks_like_log(text)
        if ok:
            return text, "s3", tried
        tried[-1] += f" ({why})"
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if use_github and token:
        status, body = http_get(GH.format(job_id), {"Authorization": f"Bearer {token}",
                                                    "Accept": "application/vnd.github+json"})
        tried.append(f"github:{status}")
        if status == 200:
            text = decode(body)
            if looks_like_log(text)[0]:
                return text, "github", tried
    return None, None, tried


FAILED = re.compile(r"FAILED (?:\[[\d.]+s\] )?((?:test/)?[\w/]+\.py::[\w:\[\]\-.]+)")
CONSISTENT = re.compile(r"The following tests failed consistently: \[([^\]]*)\]")
FLAKY = re.compile(r"The following tests failed and then succeeded when run in a new process\s*\[([^\]]*)\]")


def parse(text):
    norm = lambda t: re.sub(r"^test/", "", t)
    failed = {norm(t) for t in FAILED.findall(text)}
    for blob in CONSISTENT.findall(text):
        failed |= {norm(t) for t in re.findall(r"'([^']+)'", blob)}
    flaky = {norm(t) for blob in FLAKY.findall(text) for t in re.findall(r"'([^']+)'", blob)}
    return {"failed_tests": sorted(failed - flaky), "flaky_tests": sorted(flaky),
            "timed_out": "has timed out after" in text}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("job_ids", nargs="*")
    ap.add_argument("--csv"); ap.add_argument("--col", default="job_id")
    ap.add_argument("--out", default="ci_logs")
    ap.add_argument("--parse", action="store_true")
    ap.add_argument("--no-github", action="store_true")
    ap.add_argument("--keep-timestamps", action="store_true")
    a = ap.parse_args()

    ids = list(a.job_ids)
    if a.csv:
        csv.field_size_limit(10**9)
        with open(a.csv, encoding="utf-8") as f:
            ids += [r[a.col] for r in csv.DictReader(f) if r.get(a.col)]
    ids = list(dict.fromkeys(i.split(".")[0].strip() for i in ids if i.strip()))  # "123.0" -> "123", dedupe
    if not ids:
        ap.error("no job ids given")

    os.makedirs(a.out, exist_ok=True)
    index = {}
    for n, jid in enumerate(ids, 1):
        text, src, tried = fetch(jid, use_github=not a.no_github)
        entry = {"source": src, "tried": tried}
        if text is not None:
            if not a.keep_timestamps:
                text = TS.sub("", text)
            path = os.path.join(a.out, f"{jid}.log")
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            entry.update(path=path, bytes=len(text))
            if a.parse:
                entry.update(parse(text))
        index[jid] = entry
        extra = f" failed={len(entry.get('failed_tests', []))}" if a.parse and text else ""
        print(f"[{n}/{len(ids)}] {jid}: {src or 'NOT FOUND'} {' '.join(tried)}{extra}", file=sys.stderr)

    with open(os.path.join(a.out, "index.json"), "w") as f:
        json.dump(index, f, indent=1)
    got = sum(1 for e in index.values() if e["source"])
    print(f"{got}/{len(ids)} logs saved to {a.out}/ (index.json has details)", file=sys.stderr)
    return 0 if got == len(ids) else 1


if __name__ == "__main__":
    sys.exit(main())
