"""Recover an acquisition slice without replaying its spent native-call budget.

Run this file with --code pointing to the original frozen scientific export.
The new runner is frozen separately; original acquisition receipts remain immutable.
"""
from __future__ import annotations
import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

CAUSE = 'RuntimeError: Acquisition child cleanup incomplete'


def confirm_owner_gone(step):
    import re

    if not re.fullmatch(r"14408\.\d+", step) or step == "14408.0":
        raise ValueError("Recovery requires an experiment step in allocation 14408")
    component = "step_" + step.split(".")[1]
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
            if proc.stat().st_uid != os.getuid():
                continue
            if component in (proc / 'cgroup').read_text().strip().split('/'):
                raise RuntimeError('Original step has a surviving process')
        except (FileNotFoundError, ProcessLookupError):
            continue
    return dict(step_id=step, checked_epoch=time.time(), surviving_processes=0,
                cgroup_absent=True, signals_sent=0)


def reconciled_timeout(previous, call, original_ref, evidence_ref, ownership, step):
    if (previous['status'] != 'error' or previous['outer_status'] != 'error'
            or previous['detail'] != CAUSE or not previous['cleanup_complete']
            or previous['result'] is not None or previous['supervision_eligible']
            or call['status'] != 'timeout' or call['detail'] != 'stage deadline exhausted'
            or call['cleanup_complete'] or call['assignment'] is not None
            or call['case_id'] != previous['case_id'] or call['split'] != previous['split']
            or call['resources']['wall_seconds'] < call['deadline_seconds']):
        raise ValueError('Only the exact retained native timeout admits cleanup reconciliation')
    if (ownership['step_id'] != step or ownership['surviving_processes'] != 0
            or not ownership['cgroup_absent'] or ownership['signals_sent'] != 0):
        raise ValueError('Original owner cleanup has not been independently confirmed')
    result = {k: v for k, v in previous.items() if k not in ('identity', 'content_hash')}
    result.update(status='unknown_intended_parent',
        detail='Native intended-parent deadline exhausted; original inner cleanup limit and error retained in linked evidence; outer/Slurm cleanup now confirmed',
        original_receipt=original_ref, cleanup_reconciliation=evidence_ref,
        recovery_action='retain_native_timeout_without_rerun', original_detail=previous['detail'],
        ownership=ownership, additional_elapsed_seconds=0.0, prior_costs_reset=False)
    return result


def slice_bounds(plan):
    start, stop = plan['original_slice']
    failed = plan['failed_index']
    if (any(type(v) is not int for v in (start, stop, failed))
            or not 0 <= start <= failed == stop - 2
            or plan['only_unstarted_slice'] != [failed + 1, stop]
            or plan['original_case_count'] != stop - start
            or plan['prior_costs_reset'] is not False):
        raise ValueError('Recovery must preserve the slice and run only its untouched final case')
    return start, stop, failed


def validate_original(plan, evidence, identity):
    from exact.repair.records import canonical_hash
    from tools.repair.expanded_corpus import bound
    from tools.repair.expanded_profile import checked_checkpoint
    from tools.repair.prepare import read_label_cache
    from tools.repair import acquisition

    start, stop, failed = slice_bounds(plan)
    step = plan['original_step']
    original = bound(evidence['completion']); owner = bound(evidence['ownership'])
    if (original['status'] != 'failed' or original['step_id'] != step
            or original['step_id'] != owner['step_id']
            or original['dispatch_nonce'] != owner['dispatch_nonce']
            or original['error']['message'] != 'command 0 exited 1: RuntimeError: Acquisition software failure: ' + CAUSE
            or not evidence['cleanup_now_complete']):
        raise ValueError('Original failed completion ownership or cause differs')
    command = bound(evidence['command'])
    argv = command['argv']
    if (original['batch'] != plan['frozen_batch']['path']
            or command['cwd'] != plan['source_root']
            or argv[1:3] != ['-m', 'tools.repair.acquisition']
            or argv[3:5] != [plan['acquisition_plan']['path'], str(Path(original['work']) / 'acquisition')]
            or argv[5:] != ['--start', str(start), '--stop', str(stop)]):
        raise ValueError('Original executed acquisition source, plan or slice differs')
    cases = acquisition.validate_schedule(bound(plan['acquisition_plan']))
    prior = {}
    for item in evidence['rows']:
        index, ref = item['index'], item['receipt']
        if index not in range(start, failed + 1) or index in prior:
            raise ValueError('Original scheduled rows differ')
        row, record, case = cases[index]
        expected = canonical_hash((identity, row))
        saved = bound(ref)
        if checked_checkpoint(Path(ref['path']), expected) != saved or saved['case_id'] != case.case_id:
            raise ValueError('Original acquisition row identity differs')
        if not saved['cleanup_complete']:
            raise ValueError('Original outer cleanup incomplete')
        for native in saved['native_evidence']:
            bound(native)
        if saved.get('result'):
            read_label_cache(saved['result']['artifact'], Path(ref['path']).parent / 'cache', case)
        if (Path(ref['path']).parent / 'inflight.json').exists():
            raise ValueError('Original case has unresolved ownership')
        if index != failed:
            acquisition.raise_on_software_failure(saved)
        prior[index] = (ref, saved)
    if set(prior) != set(range(start, failed + 1)):
        raise ValueError('Every finished case and the recorded timeout must be retained')
    last = cases[failed + 1][0]
    key = canonical_hash((last['case_id'], last['case_hash']))
    if (Path(original['work']) / 'acquisition' / 'cases' / key).exists():
        raise ValueError('The last case already spent work; reconcile before continuation')
    call = bound(evidence['failed_native_call'])
    if evidence['failed_native_call'] not in prior[failed][1]['native_evidence']:
        raise ValueError('Timeout call does not belong to the failed case')
    failed_case = cases[failed][2]
    if (call['input_hash'] != failed_case.problem.content_hash
            or call['policy_hash'] != failed_case.problem.policy.content_hash
            or call['query_hash'] != canonical_hash(failed_case.probes)
            or call['parent'] != failed_case.structural_parent):
        raise ValueError('Native timeout input or semantic scope differs')
    return prior, call


def run(plan_path, output):
    from exact.repair.records import canonical_hash
    from exact.repair.study import runtime_manifest
    from tools.repair.batch import read, sha
    from tools.repair.expanded_corpus import binding, bound
    from tools.repair.expanded_profile import checked_checkpoint, checkpoint
    from tools.repair import acquisition

    plan, output = read(plan_path), Path(output).resolve()
    start, stop, failed = slice_bounds(plan)
    if binding(__file__) != plan['runner'] or str(output) != plan['output']:
        raise ValueError('Frozen runner or output differs')
    if (Path(acquisition.__file__).resolve().parents[2] != Path(plan['source_root']).resolve()
            or runtime_manifest() != plan['runtime']):
        raise ValueError('Original acquisition source/runtime changed')
    identity = canonical_hash((plan['acquisition_plan']['sha256'], sha(acquisition.__file__),
                               sha(Path(acquisition.__file__).with_name('train.py')), plan['runtime']))
    if identity != plan['acquisition_identity']:
        raise ValueError('Original acquisition dependency identity changed')
    from tools.repair.batch import checked_batch

    frozen = bound(plan['frozen_batch'])
    if Path(frozen['code']).resolve() != Path(plan['source_root']).resolve():
        raise ValueError('Original frozen export differs')
    checked_batch(plan['frozen_batch']['path'])
    evidence = bound(plan['evidence'])
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'recovery.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prior, call = validate_original(plan, evidence, identity)
        ownership = confirm_owner_gone(plan['original_step'])
        recovery_identity = canonical_hash((binding(plan_path), identity))
        revised = output / 'reconciled-timeout.json'
        saved = checked_checkpoint(revised, recovery_identity)
        if saved:
            if (saved['original_receipt'] != prior[failed][0] or saved['cleanup_reconciliation'] != plan['evidence']
                    or saved['recovery_action'] != 'retain_native_timeout_without_rerun'):
                raise ValueError('Reconciled timeout provenance changed')
        else:
            checkpoint(revised, recovery_identity, **reconciled_timeout(
                prior[failed][1], call, prior[failed][0], plan['evidence'], ownership, plan['original_step']))
        continuation = acquisition.run(Path(plan['acquisition_plan']['path']), output / 'continuation', failed + 1, stop)
        if continuation['scheduled_rows'] != 1 or continuation['recorded_rows'] != 1:
            raise ValueError('Continuation changed the last-case denominator')
        refs = [prior[i][0] for i in range(start, failed)] + [binding(revised)] + continuation['rows']
        return checkpoint(output / 'report.json', recovery_identity,
            schema='exact-repair/' + ('training' if bound(plan['acquisition_plan']).get('split', 'development') == 'train' else 'development') + '-acquisition/v1', status='complete',
            plan=plan['acquisition_plan'], scheduled_rows=stop-start, recorded_rows=len(refs), rows=refs,
            counts=dict(Counter(bound(ref)['status'] for ref in refs)),
            supervision_eligible=False, heldout_cases_opened=False,
            next_stage='xr21-expanded-training-001',
            cleanup_recovery=dict(plan=binding(plan_path), evidence=plan['evidence'],
                retained_finished_rows=failed-start, retained_native_timeouts_without_rerun=1, newly_executed_rows=1,
                original_error_retained=True, prior_costs_reset=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--code', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.code.resolve()))
    run(args.plan, args.output)
