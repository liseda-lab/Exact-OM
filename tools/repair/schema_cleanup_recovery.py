"""Reconcile one stopped schema slice without replaying a spent evaluation row.

Freeze this runner separately and pass --code for the original scientific export.
The original receipts and in-flight guard stay immutable. No semantic result is
reconstructed from partial search artifacts: the stopped row remains unknown.
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

CAUSE = 'RuntimeError: Nested worker cleanup incomplete; reconcile descendants before continuation'
COMMAND_ERROR = 'command 1 exited 1: RuntimeError: Schema recovery cleanup incomplete; retain in-flight guard'


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
            or not 0 <= start <= failed < stop
            or plan['only_unstarted_slice'] != [failed + 1, stop]
            or plan['original_row_count'] != stop - start
            or plan['prior_costs_reset'] is not False):
        raise ValueError('Recovery must retain the whole slice and continue only untouched rows')
    return start, stop, failed


def reconciled_unknown(previous, guard, original_ref, evidence_ref, ownership, step):
    if (previous['status'] != 'error' or previous['detail'] != CAUSE
            or not previous['cleanup_complete'] or previous['result'] is not None
            or previous['recovery_action'] != 'compatible_remaining_budget_execution'
            or previous['identity'] != guard['identity'] or previous['row'] != guard['row']
            or previous['remaining_budget'] != guard['remaining_budget']):
        raise ValueError('Only the exact retained nested cleanup error admits reconciliation')
    if (ownership['step_id'] != step or ownership['surviving_processes'] != 0
            or not ownership['cgroup_absent'] or ownership['signals_sent'] != 0):
        raise ValueError('Original owner cleanup has not been independently confirmed')
    result = {k: v for k, v in previous.items() if k not in ('identity', 'content_hash')}
    result.update(status='unknown_after_cleanup_reconciliation',
        detail='No final scientific result; original nested cleanup failure and partial search retained. Slurm descendants now absent; row is not replayed.',
        original_status=previous['status'], original_detail=previous['detail'],
        original_receipt=original_ref, cleanup_reconciliation=evidence_ref,
        recovery_action='retain_unknown_without_replay', ownership=ownership,
        additional_elapsed_seconds=0.0, prior_costs_reset=False)
    return result


def validate_original(plan, evidence, identity):
    from exact.repair.records import canonical_hash
    from tools.repair.expanded_corpus import bound
    from tools.repair.expanded_profile import checked_checkpoint
    from tools.repair import schema_recovery as science

    start, stop, failed = slice_bounds(plan)
    completion, owner, command = (bound(evidence[k]) for k in ('completion', 'ownership', 'command'))
    work = Path(completion['work']) / 'evaluation'
    command_error = re.fullmatch(
        r'command (0|[1-9][0-9]*) exited 1: RuntimeError: Schema recovery cleanup incomplete; retain in-flight guard',
        completion.get('error', {}).get('message', ''))
    if (command_error is None or Path(evidence['command']['path']) !=
            Path(evidence['completion']['path']).parent / f'command-{command_error[1]}.json'):
        raise ValueError('Cleanup error must bind its exact original command receipt')
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
        if index != failed:
            if guard.exists():
                raise ValueError('Finished row has an unresolved guard')
            science.fresh.raise_on_software_failure(saved)
        elif (str(guard) != evidence['guard']['path']
              or checked_checkpoint(guard, expected) != bound(evidence['guard'])):
            raise ValueError('Failed row guard identity differs')
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
    from exact.repair.study import runtime_manifest
    from tools.repair.batch import read, sha, checked_batch
    from tools.repair.expanded_corpus import binding, bound
    from tools.repair.expanded_profile import checked_checkpoint, checkpoint
    from tools.repair import schema_recovery as science

    plan, output = read(plan_path), Path(output).resolve()
    start, stop, failed = slice_bounds(plan)
    if (binding(__file__) != plan['runner'] or str(output) != plan['output']
            or Path(science.__file__).resolve().parents[2] != Path(plan['source_root']).resolve()
            or runtime_manifest() != plan['runtime']):
        raise ValueError('Frozen runner, scientific source, runtime or output changed')
    identity = canonical_hash((plan['schema_plan']['sha256'], sha(science.__file__),
                               sha(science.fresh.__file__), plan['runtime']))
    if identity != plan['schema_identity']:
        raise ValueError('Original scientific dependency identity changed')
    frozen = bound(plan['frozen_batch'])
    if Path(frozen['code']).resolve() != Path(plan['source_root']).resolve():
        raise ValueError('Original frozen export differs')
    checked_batch(plan['frozen_batch']['path'])
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
        expected = reconciled_unknown(prior[failed][1], bound(evidence['guard']),
            prior[failed][0], plan['evidence'], ownership, plan['original_step'])
        saved = checked_checkpoint(revised, recovery_identity)
        if saved:
            # Ownership observation time may advance; the original proof is immutable.
            expected['ownership'] = saved['ownership']
            if {k: v for k, v in saved.items() if k not in ('identity', 'content_hash')} != expected:
                raise ValueError('Reconciled unknown provenance changed')
        else:
            saved = checkpoint(revised, recovery_identity, **expected)
        # A last-row failure has no untouched suffix. Publish the retained
        # denominator without entering the scientific runner or replaying work.
        continuation = (science.run(Path(plan['schema_plan']['path']),
                                    output / 'continuation', failed + 1, stop)
                        if failed + 1 < stop else
                        dict(scheduled=0, recorded=0, rows=[]))
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
            cleanup_recovery=dict(plan=binding(plan_path), evidence=plan['evidence'],
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
