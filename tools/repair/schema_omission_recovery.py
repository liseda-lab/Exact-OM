"""Recover an omitted-vocabulary failure without replaying spent scientific rows.

Freeze this runner separately and pass --code for a source export differing
only in the bound ontology-omission correction. Original receipts stay immutable.
The failed row remains unknown; only untouched rows use the corrected generator.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import re
import subprocess
import sys
import time

CAUSE = 'ValueError: Omitted vocabulary reintroduced'
COMMAND_ERROR = 'command 0 exited 1: RuntimeError: Scientific worker software failure: ' + CAUSE


def confirm_owner_gone(step):
    if not re.fullmatch(r'14408\.\d+', step) or step == '14408.0':
        raise ValueError('Recovery requires an experiment step in allocation 14408')
    component = 'step_' + step.split('.')[1]
    result = subprocess.run(['scontrol', 'show', 'step', step], capture_output=True, text=True, timeout=15)
    if f'Job step {step} not found' not in result.stdout + result.stderr:
        raise RuntimeError('Original Slurm ownership has not ended')
    root = Path('/sys/fs/cgroup/system.slice/slurmstepd.scope')
    if not root.is_dir() or list(root.glob('*/' + component)):
        raise RuntimeError('Original step cgroup is unresolved')
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if proc.stat().st_uid == os.getuid() and component in (proc / 'cgroup').read_text().strip().split('/'):
                raise RuntimeError('Original step has a surviving process')
        except (FileNotFoundError, ProcessLookupError):
            continue
    return dict(step_id=step, checked_epoch=time.time(), surviving_processes=0,
                cgroup_absent=True, signals_sent=0)


def slice_bounds(plan):
    start, stop = plan['original_slice']
    failed = plan['failed_index']
    if (any(type(v) is not int for v in (start, stop, failed))
            or not 0 <= start <= failed < stop - 1
            or plan['only_unstarted_slice'] != [failed + 1, stop]
            or plan['original_row_count'] != stop - start
            or plan['prior_costs_reset'] is not False):
        raise ValueError('Recovery must retain the whole slice and continue only untouched rows')
    return start, stop, failed


def reconciled_unknown(previous, original_ref, evidence_ref, ownership, step):
    if (previous['status'] != 'error' or previous['detail'] != CAUSE
            or not previous['cleanup_complete'] or previous['result'] is not None
            or previous['recovery_action'] != 'compatible_remaining_budget_execution'):
        raise ValueError('Only the exact retained omission error admits reconciliation')
    if (ownership['step_id'] != step or ownership['surviving_processes'] != 0
            or not ownership['cgroup_absent'] or ownership['signals_sent'] != 0):
        raise ValueError('Original owner cleanup has not been independently confirmed')
    result = {k: v for k, v in previous.items() if k not in ('identity', 'content_hash')}
    result.update(status='unknown_after_omission_failure',
        detail='No final scientific result; original omitted-vocabulary error and costs retained. Row is not replayed.',
        original_status=previous['status'], original_detail=previous['detail'],
        original_receipt=original_ref, omission_reconciliation=evidence_ref,
        recovery_action='retain_unknown_without_replay', ownership=ownership,
        additional_elapsed_seconds=0.0, prior_costs_reset=False)
    return result


def validate_source(plan):
    from tools.repair.batch import sha, read
    from tools.repair.expanded_corpus import bound
    from exact.repair.study import runtime_manifest
    from tools.repair import schema_recovery as science

    original, revised = Path(plan['source_root']), Path(plan['corrected_source_root'])
    if Path(science.__file__).resolve().parents[2] != revised.resolve():
        raise ValueError('Recovery must import the declared corrected scientific export')
    manifest = bound(plan['source_revision'])
    batch = bound(plan['frozen_batch'])
    if batch['code'] != str(original):
        raise ValueError('Original frozen export differs')
    expected = {str(Path(p).relative_to(original)): digest
                for p, digest in batch['frozen_files'].items() if Path(p).is_relative_to(original)}
    original_runtime = original.parent / 'runtime.json'
    if (sha(original_runtime) != batch['frozen_files'][str(original_runtime)]
            or read(original_runtime) != plan['original_runtime']):
        raise ValueError('Original runtime binding changed')
    allowed = {'exact/repair/grammar.py', 'exact/repair/pipeline.py'}
    if set(manifest['changed_files']) != allowed or set(manifest['files']) != set(expected):
        raise ValueError('Correction changed files outside the omission implementation')
    for relative, old_digest in expected.items():
        if sha(original / relative) != old_digest or sha(revised / relative) != manifest['files'][relative]:
            raise ValueError('Original or corrected source bytes changed')
        if relative not in allowed and manifest['files'][relative] != old_digest:
            raise ValueError('Unrelated scientific source changed')
    actual = runtime_manifest()
    if actual != plan['runtime']:
        raise ValueError('Corrected runtime identity differs')
    before = dict(plan['original_runtime']); after = dict(actual)
    before_code, after_code = before.pop('code_hashes'), after.pop('code_hashes')
    if before != after or set(before_code) != set(after_code) or {
            name for name in before_code if before_code[name] != after_code[name]
            } != {'grammar.py', 'pipeline.py'}:
        raise ValueError('Unrelated runtime or source dependencies changed')
    return manifest


def validate_original(plan, evidence, identity):
    from exact.repair.records import canonical_hash
    from tools.repair.expanded_corpus import bound
    from tools.repair.expanded_profile import checked_checkpoint
    from tools.repair import schema_recovery as science

    start, stop, failed = slice_bounds(plan)
    completion, owner, command = (bound(evidence[k]) for k in ('completion', 'ownership', 'command'))
    work = Path(completion['work']) / 'evaluation'
    if (completion.get('error', {}).get('message') != COMMAND_ERROR
            or Path(evidence['command']['path']) != Path(evidence['completion']['path']).parent / 'command-0.json'):
        raise ValueError('Omission error must bind its exact original command receipt')
    if (completion['status'] != 'failed' or completion['exit_code'] != 1
            or completion['step_id'] != plan['original_step']
            or owner != dict(step_id=completion['step_id'], dispatch_nonce=completion['dispatch_nonce'])
            or completion['batch'] != plan['frozen_batch']['path']
            or command['cwd'] != plan['source_root']
            or command['argv'][1:] != ['-m', 'tools.repair.schema_recovery',
                plan['schema_plan']['path'], str(work), '--start', str(start), '--stop', str(stop)]):
        raise ValueError('Original execution, ownership, error or slice differs')
    schema_plan = bound(plan['schema_plan'])
    schedule, preflight = bound(schema_plan['schedule']), bound(schema_plan['preflight'])
    admission = {r['row_id']: r for r in preflight['rows']}
    prior = {}
    for item in evidence['rows']:
        index, ref = item['index'], item['receipt']
        if index not in range(start, failed + 1) or index in prior:
            raise ValueError('Retained prefix differs')
        row, entry = schedule['rows'][index], schema_plan['rows'][index]
        expected = canonical_hash((identity, row, entry, admission[row['id']]))
        if Path(ref['path']) != work / 'rows' / (row['id'] + '.json'):
            raise ValueError('Original row path differs')
        saved = bound(ref)
        if (saved != checked_checkpoint(Path(ref['path']), expected) or saved['row'] != row
                or not saved['cleanup_complete']):
            raise ValueError('Original row identity or outer cleanup differs')
        science.fresh.validate_payloads(saved)
        guard = work / 'inflight' / (row['id'] + '.json')
        if guard.exists():
            raise ValueError('Original row has an unresolved guard')
        if index != failed:
            science.fresh.raise_on_software_failure(saved)
        elif saved['status'] != 'error' or saved['detail'] != CAUSE or saved['result'] is not None:
            raise ValueError('Failed row does not match the omission error')
        prior[index] = (ref, saved)
    if set(prior) != set(range(start, failed + 1)):
        raise ValueError('Every retained prefix row is required')
    for index in range(failed + 1, stop):
        row_id = schedule['rows'][index]['id']
        if any(p.exists() for p in (work / 'rows' / (row_id + '.json'),
            work / 'inflight' / (row_id + '.json'), work / 'payloads' / row_id)):
            raise ValueError('Continuation row already spent work in the failed attempt')
    return prior


def run(plan_path, output):
    from exact.repair.records import canonical_hash
    from tools.repair.batch import read, sha
    from tools.repair.expanded_corpus import binding, bound
    from tools.repair.expanded_profile import checked_checkpoint, checkpoint
    from tools.repair import schema_recovery as science

    plan, output = read(plan_path), Path(output).resolve()
    start, stop, failed = slice_bounds(plan)
    if binding(__file__) != plan['runner'] or str(output) != plan['output']:
        raise ValueError('Frozen runner or output changed')
    validate_source(plan)
    identity = canonical_hash((plan['schema_plan']['sha256'], sha(science.__file__),
                               sha(science.fresh.__file__), plan['original_runtime']))
    if identity != plan['schema_identity']:
        raise ValueError('Original scientific identity changed')
    evidence = bound(plan['evidence'])
    # Keep all referenced original rows inside the same registered work root.
    if output.parent != Path(bound(evidence['completion'])['work']).resolve():
        raise ValueError('Recovery report must retain the original registered work root')
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'recovery.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prior = validate_original(plan, evidence, identity)
        ownership = confirm_owner_gone(plan['original_step'])
        recovery_identity = canonical_hash((binding(plan_path), identity))
        revised = output / 'reconciled-unknown.json'
        expected = reconciled_unknown(prior[failed][1], prior[failed][0],
            plan['evidence'], ownership, plan['original_step'])
        saved = checked_checkpoint(revised, recovery_identity)
        if saved:
            # Ownership observation time may advance; the original proof is immutable.
            expected['ownership'] = saved['ownership']
            if {k: v for k, v in saved.items() if k not in ('identity', 'content_hash')} != expected:
                raise ValueError('Reconciled unknown provenance changed')
        else:
            saved = checkpoint(revised, recovery_identity, **expected)
        continuation = science.run(Path(plan['schema_plan']['path']), output / 'continuation', failed + 1, stop)
        if continuation['scheduled'] != stop - failed - 1 or continuation['recorded'] != stop - failed - 1:
            raise ValueError('Continuation denominator changed')
        refs = [dict(**prior[i][0], row_id=prior[i][1]['row']['id'], status=prior[i][1]['status'])
                for i in range(start, failed)]
        refs += [dict(**binding(revised), row_id=saved['row']['id'], status=saved['status'])]
        refs += continuation['rows']
        schema_plan = bound(plan['schema_plan'])
        schedule = bound(schema_plan['schedule'])
        if [ref['row_id'] for ref in refs] != [r['id'] for r in schedule['rows'][start:stop]]:
            raise ValueError('Recovered row order or identities changed')
        return checkpoint(output / 'report.json', recovery_identity,
            schema='exact-repair/schema-recovery/v1', status='complete',
            plan=plan['schema_plan'], schedule=schema_plan['schedule'], rows=refs,
            scheduled=stop-start, recorded=len(refs), outcomes=dict(Counter(r['status'] for r in refs)),
            gates_passed=False, study_complete=False, prior_costs_reset=False,
            checkpoint_parameters_changed=False, followup=schedule.get('followup', 'xr21-expanded-evaluation-001'),
            omission_recovery=dict(plan=binding(plan_path), evidence=plan['evidence'],
                retained_finished_rows=failed-start, retained_unknown_rows_without_replay=1,
                continued_rows=stop-failed-1, original_error_retained=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--code', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.code.resolve()))
    run(args.plan, args.output)
