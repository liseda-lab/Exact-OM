"""Close a fully spent endpoint shard after confirmed descendant cleanup.

The frozen scientific runner is only used to validate source and admission.
No generation, selection, labeling or warm-up is repeated. The failed final row
stays unknown, and its original receipt, payloads and guard remain untouched.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import importlib.util
from pathlib import Path
import sys

CAUSE = 'RuntimeError: Nested worker cleanup incomplete; reconcile descendants before continuation'
COMMAND_ERROR = 'command 0 exited 1: RuntimeError: Worker cleanup incomplete; keep inflight guard'
SCHEMA = 'exact-repair/scaling-endpoint-receipt-recovery/v1'


def check_owner(owner, step):
    if (owner['step_id'] != step or owner['surviving_processes'] != 0
            or not owner['cgroup_absent'] or owner['signals_sent'] != 0):
        raise ValueError('Original owner cleanup is unresolved')


def validate_rows(plan, schedule, identity, science):
    from exact.repair.records import canonical_hash
    from tools.repair.expanded_corpus import binding, bound
    from tools.repair.expanded_profile import checked_checkpoint

    original = Path(plan['original'])
    start, stop = plan['slice']
    if (type(start) is not int or type(stop) is not int or start % 6
            or stop != start + 6 or not 0 <= start < stop <= len(schedule['rows'])
            or plan['failed_index'] != stop - 1 or len(plan['rows']) != stop - start):
        raise ValueError('A fully spent six-row shard with only its final row failed is required')
    rows = []
    for row, ref in zip(schedule['rows'][start:stop], plan['rows'], strict=True):
        path = original / 'rows' / (row['id'] + '.json')
        key = canonical_hash((identity, row))
        saved = checked_checkpoint(path, key)
        if (binding(path) != ref or saved != bound(ref) or saved['row'] != row
                or not saved['cleanup_complete']):
            raise ValueError('Original row identity or outer cleanup differs')
        science.validate_payloads(saved)
        payloads = [binding(p) for p in sorted((original / 'payloads' / row['id']).rglob('*'))
                    if p.is_file() and p.suffix != '.lock']
        if payloads != saved['payloads']:
            raise ValueError('Original payload inventory changed')
        guard = original / 'inflight' / path.name
        if row == schedule['rows'][stop - 1]:
            if (binding(guard) != plan['guard'] or checked_checkpoint(guard, key) != bound(plan['guard'])
                    or bound(plan['guard'])['row'] != row or saved['status'] != 'error'
                    or saved['detail'] != CAUSE or saved['result'] is not None
                    or row['cache_mode'] != 'warm'):
                raise ValueError('Failed final row must retain its exact error and ownership guard')
        else:
            if guard.exists() or saved['status'] != 'complete':
                raise ValueError('Earlier row has unresolved work')
            science.check_software_errors(saved)
        rows.append(saved)
    if ({p.name for p in (original / 'rows').glob('*.json')}
            != {Path(ref['path']).name for ref in plan['rows']}
            or {p.name for p in (original / 'inflight').glob('*.json')}
            != {Path(plan['guard']['path']).name}):
        raise ValueError('Unexpected original checkpoint or guard')
    return rows


def validate(plan):
    from exact.repair.records import canonical_hash
    from exact.repair.study import runtime_manifest
    from tools.repair import scaling as science
    from tools.repair.batch import checked_batch, sha
    from tools.repair.expanded_corpus import binding, bound

    if (plan['schema'] != SCHEMA or plan['adapter_sha256'] != sha(__file__)
            or plan['prior_costs_reset'] is not False or plan['scientific_budgets_changed'] is not False
            or plan['repair_attempt'] != 1 or plan['max_repairs'] != 2):
        raise ValueError('Recovery contract or frozen adapter changed')
    batch = checked_batch(plan['scientific_batch']['path'])
    if (batch != bound(plan['scientific_batch']) or batch['code'] != plan['source_root']
            or str(Path(science.__file__).resolve().parents[2]) != plan['source_root']
            or runtime_manifest() != plan['runtime']):
        raise ValueError('Frozen scientific source or runtime differs')
    spec = importlib.util.spec_from_file_location('frozen_endpoint_study', plan['study']['path'])
    if binding(plan['study']['path']) != plan['study']:
        raise ValueError('Frozen endpoint runner changed')
    study = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(study)
    bound(plan['schedule']); bound(plan['qualification'])
    schedule, allowed = study.admission(plan['schedule']['path'], plan['qualification']['path'])
    if not allowed[plan['slice'][0] // 6]:
        raise ValueError('Original native admission differs')
    completion, owner, command = (bound(plan[k]) for k in ('completion', 'owner', 'command'))
    jobs = [j for j in batch['jobs'] if j['id'] == completion['job_id']]
    start, stop = plan['slice']
    expected = ['/usr/bin/env', 'PYTHONPATH=' + batch['code'], batch['python'], plan['study']['path'],
                'run', '--schedule', plan['schedule']['path'], '--qualification', plan['qualification']['path'],
                '--output', plan['original'], '--start', str(start), '--stop', str(stop)]
    expanded = [[arg.replace('{python}', batch['python']).replace('{code}', batch['code'])
                 .replace('{work}', completion['work']) for arg in cmd]
                for cmd in jobs[0]['commands']] if len(jobs) == 1 else []
    attempt = Path(plan['completion']['path']).parent
    if (completion['status'] != 'failed' or completion['exit_code'] != 1
            or completion['error']['message'] != COMMAND_ERROR or completion['step_id'] != plan['original_step']
            or completion['batch'] != plan['scientific_batch']['path']
            or owner != dict(step_id=completion['step_id'], dispatch_nonce=completion['dispatch_nonce'])
            or Path(plan['owner']['path']) != attempt / 'step.json'
            or Path(plan['command']['path']) != attempt / 'command-0.json'
            or command != dict(argv=expected, cwd=batch['code']) or expanded != [expected]
            or Path(plan['original']).parent != Path(completion['work'])
            or Path(plan['output']).parent != Path(completion['work'])
            or plan['original'] == plan['output']):
        raise ValueError('Original command, owner or output differs')
    charge = bound(plan['ledger_snapshot'])['attempts'].get(str(attempt), {})
    if (charge.get('status') != 'settled' or charge.get('logical_id') != completion['job_id']
            or charge.get('elapsed_seconds', -1) < completion['elapsed_seconds']
            or (charge.get('cpu_seconds') or 0) < completion['cpu_seconds']):
        raise ValueError('Original attempt must remain fully charged')
    observations = [bound(ref) for ref in plan['owner_checks']]
    if len(observations) < 2 or observations[-1]['checked_epoch'] - observations[0]['checked_epoch'] < 30:
        raise ValueError('Separated ownership checks are required')
    for observation in observations:
        check_owner(observation, plan['original_step'])
        if observation['checked_epoch'] < completion['finished_epoch']:
            raise ValueError('Ownership check predates completion')
    identity = canonical_hash((plan['schedule'], sha(science.__file__), plan['runtime']))
    rows = validate_rows(plan, schedule, identity, science)
    guard = bound(plan['guard'])
    if not completion['started_epoch'] <= guard['started_epoch'] <= completion['finished_epoch']:
        raise ValueError('Guard does not belong to original attempt')
    return rows


def publish(plan_path, plan, rows, owner):
    from exact.repair.records import canonical_hash
    from tools.repair.expanded_corpus import binding, immutable
    from tools.repair.expanded_profile import checked_checkpoint, checkpoint

    check_owner(owner, plan['original_step'])
    output = Path(plan['output'])
    identity = canonical_hash(binding(plan_path))
    original = rows[-1]
    expected = {k: v for k, v in original.items() if k not in ('identity', 'content_hash')}
    expected.update(status='unknown_after_cleanup_reconciliation',
        detail='Final row remains unknown after owner cleanup; no scientific replay or result reconstruction.',
        original_status=original['status'], original_detail=original['detail'],
        original_receipt=plan['rows'][-1], original_guard=plan['guard'],
        original_completion=plan['completion'], ownership=owner,
        recovery_plan=binding(plan_path), additional_scientific_seconds=0, prior_costs_reset=False)
    revised = output / 'reconciled-unknown.json'
    saved = checked_checkpoint(revised, identity)
    if saved is not None:
        expected['ownership'] = saved['ownership']
        if {k: v for k, v in saved.items() if k not in ('identity', 'content_hash')} != expected:
            raise ValueError('Reconciled unknown provenance changed')
    else:
        saved = checkpoint(revised, identity, **expected)
    refs = [dict(ref, status=row['status']) for ref, row in zip(plan['rows'][:-1], rows[:-1], strict=True)]
    refs.append(dict(binding(revised), status=saved['status']))
    fields = dict(schema='exact-repair/scaling-shard/v1', status='complete',
        schedule=plan['schedule'], runtime=plan['runtime'], rows=refs,
        scheduled_rows=len(rows), recorded_rows=len(rows), counts=dict(Counter(r['status'] for r in refs)),
        recovery_plan=binding(plan_path), scientific_source=plan['scientific_batch'],
        original_costs_preserved=True, scientific_rows_replayed=0, additional_scientific_seconds=0,
        retained_finished_rows=len(rows)-1, retained_unknown_rows_without_replay=1,
        repair_attempt=1, max_repairs=2, study_complete=False, gates_passed=False)
    report_path = output / 'report.json'
    report = checked_checkpoint(report_path, identity)
    if report is None:
        report = checkpoint(report_path, identity, **fields)
    elif {k: v for k, v in report.items() if k not in ('identity', 'content_hash')} != fields:
        raise ValueError('Recovery report changed')
    immutable(output / 'recovery.json', dict(schema=SCHEMA, plan=binding(plan_path),
        report=binding(report_path), scientific_rows_replayed=0, original_guard_preserved=True))
    return report


def run(plan_path, *, owner_check, validate_only=False):
    from tools.repair.batch import read

    plan = read(plan_path)
    original, output = Path(plan['original']), Path(plan['output'])
    with (original / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows = validate(plan)
        owner = owner_check(plan['original_step'])
        check_owner(owner, plan['original_step'])
        if validate_only:
            return dict(status='compatible', retained_rows=len(rows)-1, unknown_rows=1,
                        scientific_rows_replayed=0, ownership=owner)
        output.mkdir(parents=True, exist_ok=True)
        with (output / 'recovery.lock').open('a') as revision_lock:
            fcntl.flock(revision_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return publish(plan_path, plan, rows, owner)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    parser.add_argument('--code', type=Path, required=True)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('owner_check', Path(__file__).with_name('schema_cleanup_recovery.py'))
    owner_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owner_module)
    sys.path.insert(0, str(args.code.resolve()))
    import json
    print(json.dumps(run(args.plan, owner_check=owner_module.confirm_owner_gone, validate_only=args.validate_only)))
