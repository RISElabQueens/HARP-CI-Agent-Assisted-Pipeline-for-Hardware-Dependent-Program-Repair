#!/usr/bin/env python3
"""Assemble the execution-validation package for the 78-case CI-mined benchmark."""
import collections, csv, glob, json, os, re, shutil

T = '<workdir>'; V = '<outdir>/validation'
OUT = f'{V}/package_2026-11-17'; shutil.rmtree(OUT, ignore_errors=True); os.makedirs(OUT)
csv.field_size_limit(10**9)

ARCH = {  # ci arch -> (hardware, results dir, cases file)
    'gfx942': ('MI325X (gfx942)', f'{V}/g942_agentpick_2026-11-10', f'{T}/g942_cases.json'),
    'gfx950': ('MI350X (gfx950)', f'{V}/g950_agentpick_2026-11-13', f'{T}/g950_cases.json'),
    'gfx90a': ('MI210 (gfx90a)', f'{V}/g90a_agentpick', f'{T}/g90a_cases.json'),
}
SUBJ = dict(l.rstrip('\n').split('\t', 1) for l in open(f'{T}/pick_subjects.tsv'))
TEST_DELETING = {'70cd101890': 'the pick (an autorevert) deletes the failing test together with the feature',
                 'fa77e69278': 'the pick (a revert) deletes the failing test',
                 'aaa0086980': 'the pick (a revert) deletes 3 of the 5 failing tests; the other 2 go fail->pass'}
ORDER = ['VALIDATED', 'FIX_DOES_NOT_FIX', 'NOT_REPRODUCED', 'FLAKY', 'SKIPPED_AT_BOTH', 'INVALID_TEST_DELETED_BY_FIX',
         'TEST_ABSENT_AT_PICK', 'NEEDS_2_GPUS', 'NOT_TESTABLE_TIMEOUT', 'NOT_TESTABLE_NO_FAILING_TEST', 'NOT_TESTABLE_NO_LOG']

def first_error(log):
    if not os.path.exists(log): return ''
    for l in open(log, errors='replace'):
        m = re.match(r'^E\s+(\w*(Error|Exception)\b.*)', l)
        if m: return m.group(1).strip()[:220]
    return ''

def pick_verdict(arch, c, p):
    d = f'{ARCH[arch][1]}/{p[:10]}'
    if not os.path.exists(f'{d}/summary.json'):
        return ('NEEDS_2_GPUS' if c.get('dist') and arch == 'gfx90a' else 'NOT_RUN'), {}, d
    s = json.load(open(f'{d}/summary.json'))
    tests = {t: s['tests'][t] for t in c['tests'] if t in s['tests']}
    cls = collections.Counter(v['class'] for v in tests.values())
    other = [v for v in tests.values() if v['class'] == 'OTHER']
    absent = any(set(t['base'] + t['fix']) <= {'not_run', 'missing'} for t in tests.values())
    if p[:10] in ('70cd101890', 'fa77e69278') and absent and not cls['F2P']: v = 'INVALID_TEST_DELETED_BY_FIX'
    elif cls['F2P'] and not cls['P2F']: v = 'VALIDATED'
    elif cls['F2F'] and not cls['F2P']: v = 'FIX_DOES_NOT_FIX'
    elif cls['FLAKY'] and not cls['F2P']: v = 'FLAKY'
    elif cls['P2P'] and not (cls['F2P'] or cls['F2F'] or cls['FLAKY']): v = 'NOT_REPRODUCED'
    elif other and all(o['base'] == ['skipped'] * 3 and o['fix'] == ['skipped'] * 3 for o in other): v = 'SKIPPED_AT_BOTH'
    elif other and all(set(o['base']) <= {'not_run', 'missing'} for o in other): v = 'TEST_ABSENT_AT_PICK'
    else: v = 'OTHER'
    return v, tests, d

def explain(v, arch, p, tests, d):
    hw = ARCH[arch][0]; n = len(tests)
    if v == 'VALIDATED':
        f2p = sum(1 for t in tests.values() if t['class'] == 'F2P')
        s = f'{f2p}/{n} failing test(s) fail at the parent and pass at the pick in all 3 repeats on {hw}.'
    elif v == 'FIX_DOES_NOT_FIX':
        e = first_error(f'{d}/fix/r0/run.log')
        s = f'The failure reproduces at both the parent and the pick (3/3 each) on {hw}; the pick does not fix it.' + (f' Error at the pick: "{e}"' if e else '')
    elif v == 'NOT_REPRODUCED': s = f'The failing test(s) pass at both the parent and the pick (3/3 each) on {hw}; the CI failure did not reproduce (likely flaky in CI or environment-dependent).'
    elif v == 'FLAKY': s = f'Outcomes differ between repeats on {hw}; the test is flaky, so a single CI fail->pass is not evidence of a fix.'
    elif v == 'SKIPPED_AT_BOTH': s = f'The failing test(s) are skipped on {hw} at both commits (e.g. skipIfRocm for this architecture).'
    elif v == 'INVALID_TEST_DELETED_BY_FIX': s = TEST_DELETING.get(p[:10], 'The pick deletes the failing test.') + ' CI turns green without a repair.'
    elif v == 'TEST_ABSENT_AT_PICK': s = 'The failing test does not exist at the pick or its parent (removed earlier in the window by another commit), so the pick cannot be evaluated.'
    elif v == 'NEEDS_2_GPUS': s = 'Distributed test needing 2 GPUs; not run yet (1-GPU quota). Wheels are built.'
    else: s = 'See per-test outcomes.'
    if p[:10] in TEST_DELETING and v == 'VALIDATED': s += ' Note: ' + TEST_DELETING[p[:10]] + '.'
    return s

rows_pick, rows_test, case_info = [], [], {}
for arch, (hw, rdir, cfile) in ARCH.items():
    for c in json.load(open(cfile)):
        key = c['case']; picks = c['picks']
        scope = c.get('scope', 'in')
        if scope != 'in':
            v = {'timeout': 'NOT_TESTABLE_TIMEOUT', 'no_stable_failing_test': 'NOT_TESTABLE_TIMEOUT', 'no_failing_test': 'NOT_TESTABLE_NO_FAILING_TEST',
                 'no_log': 'NOT_TESTABLE_NO_LOG'}.get(scope, 'NOT_TESTABLE_' + scope.upper())
            why = {'NOT_TESTABLE_TIMEOUT': 'The CI job timed out; the log has no stable failing test (only flaky failures, if any).',
                   'NOT_TESTABLE_NO_FAILING_TEST': 'The CI log has no failing pytest test to run.',
                   'NOT_TESTABLE_NO_LOG': 'The job log is unavailable on GitHub and on the PyTorch S3 mirror.'}[v]
            case_info[key] = dict(arch=arch, hw=hw, verdict=v, picks=[], explanation=why, tests=c.get('tests', []), job_id=c.get('job_id', ''))
            continue
        per = []
        for p, runs in picks.items():
            v, tests, d = pick_verdict(arch, c, p)
            par = json.load(open(f'{d}/summary.json'))['parent'] if os.path.exists(f'{d}/summary.json') else ''
            kind = 'revert' if SUBJ.get(p, '').startswith('Revert') else 'fix'
            ex = explain(v, arch, p, tests, d)
            cnt = collections.Counter(t['class'] for t in tests.values())
            rows_pick.append([key, arch, hw, p, par, ';'.join(runs), kind, SUBJ.get(p, ''), v, ' '.join(f'{k}={n}' for k, n in sorted(cnt.items())), ex])
            for t, o in tests.items():
                rows_test.append([key, arch, p, t, *o['base'], *o['fix'], o['class']])
            per.append((v, p, kind, ex))
        best = min(per, key=lambda x: ORDER.index(x[0]) if x[0] in ORDER else 99)
        case_info[key] = dict(arch=arch, hw=hw, verdict=best[0], picks=per, explanation=best[3], tests=c['tests'], job_id=c.get('job_id', ''),
                              best_pick=best[1], best_kind=best[2])

# 1. Huizi's 78-row file + validation columns
src = f'<outdir>/from_huizi/2026-11-11/run2_run3_run4_union_cross_run_agent_audit_with_failure_evidence_one_row_per_case_all_columns.csv'
rows = list(csv.DictReader(open(src, encoding='utf-8'))); cols = list(rows[0].keys())
new = ['val_ci_arch', 'val_hardware', 'val_verdict', 'val_best_pick', 'val_best_pick_kind', 'val_picks_tested', 'val_failing_tests_from_ci_log', 'val_explanation']
with open(f'{OUT}/ci_recover_78_with_validation.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=cols + new); w.writeheader()
    for r in rows:
        ci = case_info.get(r['_case_key'], {})
        r.update(val_ci_arch=ci.get('arch', ''), val_hardware=ci.get('hw', ''), val_verdict=ci.get('verdict', 'NOT_IN_VALIDATION'),
                 val_best_pick=ci.get('best_pick', ''), val_best_pick_kind=ci.get('best_kind', ''),
                 val_picks_tested='; '.join(f'{p}:{v}' for v, p, _, _ in ci.get('picks', [])),
                 val_failing_tests_from_ci_log='; '.join(ci.get('tests', [])), val_explanation=ci.get('explanation', ''))
        w.writerow(r)
missing = [r['_case_key'] for r in rows if r['_case_key'] not in case_info]

# 2. per-pick and per-test tables
with open(f'{OUT}/validation_per_pick.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['case_key', 'ci_arch', 'hardware', 'agent_pick', 'parent', 'agent_runs', 'pick_kind', 'pick_subject', 'verdict', 'test_classes', 'explanation']); w.writerows(rows_pick)
with open(f'{OUT}/validation_per_test.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['case_key', 'ci_arch', 'agent_pick', 'test', 'parent_r1', 'parent_r2', 'parent_r3', 'pick_r1', 'pick_r2', 'pick_r3', 'class']); w.writerows(rows_test)

# 3. raw results (summaries + pytest logs)
for arch, (hw, rdir, _) in ARCH.items():
    for sfile in glob.glob(f'{rdir}/*/summary.json'):
        u = os.path.basename(os.path.dirname(sfile)); dst = f'{OUT}/results/{arch}/{u}'; os.makedirs(dst, exist_ok=True)
        shutil.copy(sfile, dst)
        for log in glob.glob(f'{rdir}/{u}/*/r*/run.log'):
            rel = os.path.relpath(log, f'{rdir}/{u}'); os.makedirs(os.path.dirname(f'{dst}/{rel}'), exist_ok=True); shutil.copy(log, f'{dst}/{rel}')

# 4. totals
tot = collections.defaultdict(collections.Counter)
for k, ci in case_info.items(): tot[ci['arch']][ci['verdict']] += 1
json.dump({a: dict(c) for a, c in tot.items()}, open(f'{OUT}/totals.json', 'w'), indent=1)
print('cases covered:', len(case_info), '| not in validation:', missing)
for a, c in tot.items(): print(a, dict(c))
allc = collections.Counter(); [allc.update(c) for c in tot.values()]; print('ALL', dict(allc))
