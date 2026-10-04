"""Retain a stopped scaling pair and continue only untouched cold/warm pairs.

This adapter uses the original scientific and payload-adapter exports. It never
replays the failed warm row, upgrades its unknown result, or removes its guard.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import importlib.util
from pathlib import Path
import subprocess
import sys

CAUSE = 'RuntimeError: Nested worker cleanup incomplete; reconcile descendants before continuation'
COMMAND_ERROR = 'command 0 exited 1: RuntimeError: Worker cleanup incomplete; keep inflight guard'


def load_adapter(reference):
    from tools.repair.batch import sha
    if sha(reference['path']) != reference['sha256']:
        raise ValueError('Frozen payload adapter changed')
    spec = importlib.util.spec_from_file_location('frozen_scaling_payload', reference['path'])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_previous_export(reference):
    from tools.repair.expanded_corpus import bound
    batch = bound(reference)
    subprocess.run([batch['python'], '-c',
        'import sys; from tools.repair.batch import checked_batch; checked_batch(sys.argv[1])',
        reference['path']], cwd=batch['code'], check=True, capture_output=True, text=True, timeout=120)
    return batch


def validate_original(plan):
    from exact.repair.records import canonical_hash
    from tools.repair.expanded_corpus import bound
    from tools.repair.expanded_profile import checked_checkpoint
    from tools.repair import scaling as science

    previous, evidence = bound(plan['previous_plan']), bound(plan['evidence'])
    continuation = bound(plan['continuation_plan'])
    start, stop = previous['slice']; failed = plan['failed_index']
    if (previous.get('recovery') is not None or type(failed) is not int
            or not start <= failed < stop - 1 or failed % 2 != 1
            or start % 2 or stop % 2 or plan['repair_attempt'] != 1
            or plan['max_repairs'] != 2 or plan['prior_costs_reset'] is not False
            or continuation != dict(previous, slice=[failed + 1, stop],
                output=str(Path(plan['output']) / 'continuation'))):
        raise ValueError('Only untouched pairs after the first failed warm row may continue')
    batch = validate_previous_export(plan['previous_batch'])
    completion, owner, command = (bound(evidence[k]) for k in ('completion', 'owner', 'command'))
    expected_argv = [batch['python'], previous['adapter']['path'], plan['previous_plan']['path'],
        previous['output'], '--code', previous['source_root']]
    jobs = [j for j in batch['jobs'] if j['id'] == completion['job_id']]
    original = Path(previous['output'])
    if (completion['status'] != 'failed' or completion['exit_code'] != 1
            or completion['error']['message'] != COMMAND_ERROR
            or completion['step_id'] != plan['original_step']
            or completion['batch'] != plan['previous_batch']['path']
            or owner != dict(step_id=completion['step_id'], dispatch_nonce=completion['dispatch_nonce'])
            or Path(evidence['command']['path']) != Path(evidence['completion']['path']).parent / 'command-0.json'
            or command != dict(argv=expected_argv, cwd=batch['code'])
            or len(jobs) != 1 or [[arg.replace('{python}', batch['python']).replace('{code}', batch['code']).replace('{work}', completion['work']) for arg in cmd] for cmd in jobs[0]['commands']] != [expected_argv]
            or original.parent != Path(completion['work'])
            or Path(plan['output']).parent != original.parent or Path(plan['output']) == original):
        raise ValueError('Original scaling execution, owner or output differs')
    schedule = bound(previous['schedule'])
    identity = canonical_hash((plan['previous_plan'], previous['scientific_identity']))
    adapter = load_adapter(previous['adapter'])
    prior = {}
    for item in evidence['rows']:
        i, ref = item['index'], item['receipt']
        if i not in range(start, failed + 1) or i in prior:
            raise ValueError('Retained prefix differs')
        row = schedule['rows'][i]; key = row['id']
        path = original / 'rows' / (key + '.json')
        expected = canonical_hash((identity, row))
        saved = bound(ref)
        if (ref['path'] != str(path) or checked_checkpoint(path, expected) != saved
                or saved['row'] != row or not saved['cleanup_complete']):
            raise ValueError('Original row identity or outer cleanup differs')
        call = bound(item['call'])
        if (item['call']['path'] != str(original / 'calls' / path.name)
                or call != checked_checkpoint(Path(item['call']['path']), expected)
                or any(saved[k] != v for k, v in call.items() if k != 'content_hash')):
            raise ValueError('Original call telemetry differs')
        if adapter.payload_manifest(original / 'payloads' / key) != (saved['payloads'], saved['unpublished_payloads']):
            raise ValueError('Original payload inventory changed')
        science.validate_payloads(saved)
        guard = original / 'inflight' / path.name
        if i == failed:
            if (evidence['guard']['path'] != str(guard)
                    or checked_checkpoint(guard, expected) != bound(evidence['guard'])
                    or saved['status'] != 'error' or saved['detail'] != CAUSE
                    or saved['result'] is not None or row['cache_mode'] != 'warm'):
                raise ValueError('Failed row must retain exact nested cleanup error and guard')
        elif guard.exists() or saved['status'] == 'error' or 'cleanup incomplete' in saved['detail']:
            raise ValueError('Earlier row has unresolved software failure')
        prior[i] = (ref, saved)
    if set(prior) != set(range(start, failed + 1)):
        raise ValueError('Every spent row must be retained')
    for row in schedule['rows'][failed + 1:stop]:
        if any(p.exists() for p in (original / 'rows' / (row['id'] + '.json'),
                original / 'calls' / (row['id'] + '.json'),
                original / 'inflight' / (row['id'] + '.json'), original / 'payloads' / row['id'])):
            raise ValueError('Continuation row already spent work')
    return previous, evidence, prior, adapter


def run(plan_path, output, *, owner_check):
    from exact.repair.records import canonical_hash
    from exact.repair.study import runtime_manifest
    from tools.repair.batch import read, sha, checked_batch
    from tools.repair.expanded_corpus import binding, bound
    from tools.repair.expanded_profile import checked_checkpoint, checkpoint
    from tools.repair import scaling as science

    plan, output = read(plan_path), Path(output).resolve()
    previous = bound(plan['previous_plan'])
    if (binding(__file__) != plan['runner'] or str(output) != plan['output']
            or str(Path(science.__file__).resolve().parents[2]) != previous['source_root']
            or runtime_manifest() != previous['runtime']
            or canonical_hash((previous['schedule'], sha(science.__file__), previous['runtime']))
                != previous['scientific_identity']):
        raise ValueError('Frozen runner, scientific source, runtime or output changed')
    scientific = checked_batch(previous['scientific_batch']['path'])
    if scientific != bound(previous['scientific_batch']) or scientific['code'] != previous['source_root']:
        raise ValueError('Original scientific batch changed')
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'recovery.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous, evidence, prior, adapter = validate_original(plan)
        owner = owner_check(plan['original_step'])
        if (owner['step_id'] != plan['original_step'] or owner['surviving_processes'] != 0
                or not owner['cgroup_absent'] or owner['signals_sent'] != 0):
            raise ValueError('Original owner cleanup is unresolved')
        start, stop = previous['slice']; failed = plan['failed_index']
        identity = canonical_hash((binding(plan_path), previous['scientific_identity']))
        original_ref, original = prior[failed]
        revised = output / 'reconciled-unknown.json'
        expected = {k: v for k, v in original.items() if k not in ('identity', 'content_hash')}
        expected.update(status='unknown_after_cleanup_reconciliation',
            detail='No final scientific result; original nested cleanup error retained. Owner gone; spent row not replayed.',
            original_status=original['status'], original_detail=original['detail'],
            original_receipt=original_ref, original_guard=evidence['guard'],
            cleanup_reconciliation=plan['evidence'], ownership=owner,
            additional_scientific_seconds=0, prior_costs_reset=False)
        saved = checked_checkpoint(revised, identity)
        if saved:
            expected['ownership'] = saved['ownership']
            if {k: v for k, v in saved.items() if k not in ('identity', 'content_hash')} != expected:
                raise ValueError('Reconciled unknown provenance changed')
        else:
            saved = checkpoint(revised, identity, **expected)
        continuation = adapter.run(plan['continuation_plan']['path'], output / 'continuation')
        schedule = bound(previous['schedule'])
        refs = [dict(prior[i][0], status=prior[i][1]['status']) for i in range(start, failed)]
        refs.append(dict(binding(revised), status=saved['status']))
        refs.extend(continuation['rows'])
        if (continuation['scheduled_rows'] != stop - failed - 1
                or continuation['recorded_rows'] != stop - failed - 1
                or [bound(ref)['row'] for ref in refs] != schedule['rows'][start:stop]):
            raise ValueError('Recovered denominator, row order or identities changed')
        return checkpoint(output / 'report.json', identity,
            schema='exact-repair/scaling-shard/v1', status='complete', schedule=previous['schedule'],
            runtime=previous['runtime'], rows=refs, scheduled_rows=stop-start, recorded_rows=len(refs),
            counts=dict(Counter(ref['status'] for ref in refs)), recovery_plan=binding(plan_path),
            scientific_source=previous['scientific_batch'], study_complete=False, gates_passed=False,
            original_costs_preserved=True, scientific_rows_replayed=0,
            cleanup_recovery=dict(retained_finished_rows=failed-start,
                retained_unknown_rows_without_replay=1, continued_rows=stop-failed-1,
                repair_attempt=1, max_repairs=2, original_error_retained=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--code', type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('owner_check', Path(__file__).with_name('schema_cleanup_recovery.py'))
    owner_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(owner_module)
    sys.path.insert(0, str(args.code.resolve()))
    run(args.plan, args.output, owner_check=owner_module.confirm_owner_gone)
