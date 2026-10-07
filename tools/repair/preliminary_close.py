"""Authenticate preliminary closure without replay, launches or notifications."""
from __future__ import annotations

import argparse
from copy import deepcopy
import math
from pathlib import Path
import time

from exact.experiments.completion import idle_completion
from exact.experiments.supervision import inspect_runs
from exact.repair.api import write_artifact
from tools.repair.batch import locked, read
from tools.repair.overlap_audit import binding, require
from tools.repair.preliminary_report import (
    StreamEvidence, authenticate, authenticate_history, reconcile_costs,
)
from tools.supervise_experiments import slurm_steps

RUN = 'xr21-expanded-preliminary-final-001'
SLOT = 'xr21-expanded-report-001'
OUTPUTS = {'final/report.json', 'final/scope-readiness.json', 'final/REPORT.md',
           'final/attempt-history.json', 'final/costs.json', 'final/verified-files.json',
           'validation.xml'}


def completion_candidate(registry, observation, dispatch, steps, supervisor_step):
    """Refuse queue closure if any registered or scheduler-owned work remains."""
    require(steps is not None, 'Scheduler evidence unavailable')
    allowed = {'14408.0', '14408.extern', supervisor_step}
    require('14408.0' in steps and supervisor_step in steps and not set(steps) - allowed,
            'Live allocation ownership differs or another worker remains')
    require([p['id'] for p in registry['pending_batches']] == [SLOT],
            'Additional or missing preparation slots')
    slot = registry['pending_batches'][0]
    require(slot.get('preparation_only') and not slot.get('launch'), 'Slot is executable')
    active = {r['id'] for r in registry['runs']
              if r.get('enabled', True) and not r.get('superseded_by')}
    require({r['run_id'] for r in observation['findings']} == active,
            'Incomplete live observation')
    require(RUN in active and set(slot['depends_on']) <= active, 'Missing final dependencies')
    candidate = deepcopy(registry)
    candidate.update(pending_batches=[], remaining_work_status='terminal')
    observation = deepcopy(observation)
    observation['incidents'] = [r for r in observation['incidents'] if r['kind'] != 'next_batch']
    require(idle_completion(candidate, observation, dispatch) is not None,
            'Unaccounted failure, active worker or dispatcher reservation')
    return candidate, observation


def validate_readiness(report, scope, amendment):
    require(report['status'] == 'complete' and report['local_report_complete'] and
            scope == report['scope_readiness'] and scope['local_scientific_dependencies_accounted'],
            'Incomplete preliminary report')
    require(not any(report[k] for k in ('campaign_complete', 'full_program_complete',
            'gates_passed', 'scientific_rows_replayed', 'new_native_calls', 'new_fitting_runs_executed')),
            'Scientific claim or scope promoted')
    require(not any(scope[k] for k in ('local_scope_complete', 'full_expanded_program_complete',
            'fitting_eligible', 'learning_efficiency_qualified', 'strongest_symbolic_comparison_qualified')),
            'Premature scientific completion')
    require(scope['gates'] == dict.fromkeys(('G0', 'G1', 'G2'), 'not_established'), 'Gates promoted')
    expected = [dict(status='deferred_by_user', obligation=x) for x in amendment['deferred_work']]
    require(len(expected) == 4 and scope['deferred_work'] == report['deferred_work'] == expected,
            'User deferrals lost')
    require(report['proposed_cluster_design']['status'] == 'proposal_not_frozen_or_approved',
            'Cluster proposal promoted')
    require(report['api_spend_usd'] == scope['api_spend_usd'] == 0 and
            scope['llm_labels'] == 'disabled', 'External label/API scope changed')


def preserve_accounting(previous, current, *, settling=()):
    """Settled entries are immutable; authorized settlements retain reservations."""
    require(current['limit_worker_seconds'] is None, 'Campaign time amendment lost')
    for key, old in previous['attempts'].items():
        new = current['attempts'].get(key)
        if key not in settling:
            require(new == old, 'Historical charge changed or disappeared: ' + key)
        else:
            require(old['status'] == 'reserved' and new is not None and new['status'] == 'settled',
                    'Unexpected maintenance settlement')
            require(all(new.get(k) == old[k] for k in
                        ('logical_id', 'resources', 'reserved_seconds', 'started_epoch')),
                    'Reservation identity lost')
    require(all(row['status'] == 'settled' for row in current['attempts'].values()),
            'Unsettled costs remain')
    rows = list(current['attempts'].values())
    expected = dict(worker_seconds=sum(r['elapsed_seconds'] for r in rows),
        allocated_cpu_seconds=sum(r['elapsed_seconds']*r['resources']['cpus'] for r in rows),
        allocated_gpu_seconds=sum(r['elapsed_seconds']*r['resources']['gpus'] for r in rows),
        allocated_memory_mb_seconds=sum(r['elapsed_seconds']*r['resources']['memory_mb'] for r in rows),
        measured_cpu_seconds=sum(r.get('cpu_seconds') or 0 for r in rows), external_api_cost_usd=0)
    require(all(math.isclose(current['cumulative'][k], v, rel_tol=1e-12, abs_tol=1e-6)
                for k, v in expected.items()), 'Cumulative ledger does not reconcile')


def audit(campaign, output, supervisor_step):
    """Hash the final worker and its existing evidence indexes, with no new science."""
    campaign, output = Path(campaign).resolve(), Path(output).resolve()
    with locked(campaign/'supervisor/registry.json.lock'):
        require(not (output/'authentication.json').exists(), 'Use a new audit revision')
        registry = read(campaign/'supervisor/registry.json')
        steps = slurm_steps('14408')
        observation = inspect_runs(registry['runs'], step_states=steps)
        dispatch = read(campaign/'supervisor/dispatch-state.json')
        completion_candidate(registry, observation, dispatch, steps, supervisor_step)
        with locked(campaign/'resource-ledger.lock'):
            ledger = read(campaign/'resource-ledger.json')
        output.mkdir(parents=True, exist_ok=True)
        write_artifact(output/'registry-audited.json', registry)
        write_artifact(output/'ledger-audited.json', ledger)
        write_artifact(output/'scheduler-audited.json', dict(epoch=time.time(), steps=steps,
            observation=observation, dispatch=dispatch))
    evidence = StreamEvidence()
    run = next(r for r in registry['runs'] if r['id'] == RUN)
    attempt = Path(run['completion_path']).parent
    complete = read(run['completion_path'])
    batch_path = Path(complete['batch'])
    batch = read(batch_path)
    work = Path(complete['work'])
    item = dict(run=run, completion=binding(run['completion_path']), step=binding(attempt/'step.json'),
        exit=binding(run['exit_path']), batch=binding(batch_path),
        batch_hash=binding(batch_path.with_name('batch.sha256')),
        runtime=binding(batch_path.with_name('runtime.json')), outputs=binding(attempt/'outputs.json'),
        report=binding(work/'final/report.json'), validation_minimum=64)
    registered = registry['expanded_preliminary_final_report']
    require(item['batch'] == registered['batch'] and batch['commit'] == run['source_commit'],
            'Final batch differs from registration')
    require(set(read(attempt/'outputs.json')) == OUTPUTS, 'Missing or unexpected final output')
    report, owner = authenticate(evidence, item, ledger, registry)
    require(report['manifest'] == run['audit_manifest'] == registered['manifest'], 'Manifest differs')
    scope = read(work/'final/scope-readiness.json')
    amendment = evidence.read(registry['preliminary_scope_amendment'])
    validate_readiness(report, scope, amendment)
    require(scope['scope_amendment'] == registry['preliminary_scope_amendment'], 'Amendment differs')
    prior = read(work/'final/attempt-history.json')
    current = {r['id']: r for r in registry['runs']}
    require(len(prior) == report['all_attempts'] == len(current)-1 and
            {x['run']['id'] for x in prior} == set(current)-{RUN}, 'Prior history denominator differs')
    for row in prior:
        require(current[row['run']['id']] == row['run'], 'Prior run or recovery lineage changed')
        require(ledger['attempts'][str(Path(row['run']['completion_path']).parent)] == row['charge'],
                'Prior worker charge changed')
    receipts = [dict(id=r['id'], completion=binding(r['completion_path'])) for r in registry['runs']]
    history = authenticate_history(evidence, registry, ledger, receipts)
    job = next(j for j in batch['jobs'] if j['id'] == complete['job_id'])
    for index, command in enumerate(job['commands']):
        actual = evidence.read(binding(attempt/f'command-{index}.json'))
        expected = [a.format(python=batch['python'], work=str(work), protocol=batch['protocol'],
                             code=batch['code']) for a in command]
        require(actual == dict(argv=expected, cwd=batch['code']), 'Executed command differs')
    cost = read(work/'final/costs.json')
    summary = reconcile_costs(ledger, cost['historical']['pilot-ledger.json'],
                             cost['historical']['smoke-ledger.json'])
    result = dict(schema='exact-repair/preliminary-closure-authentication/v1', status='complete',
        final_worker=item, worker=owner, report=item['report'], scope_readiness=binding(work/'final/scope-readiness.json'),
        final_outputs={n:dict(path=str(work/n),sha256=h) for n,h in read(attempt/'outputs.json').items()},
        registry_snapshot=binding(output/'registry-audited.json'),
        ledger_snapshot=binding(output/'ledger-audited.json'), scheduler=binding(output/'scheduler-audited.json'),
        source_commit=batch['commit'], all_registered_attempts=len(history),
        active_completed_attempts=len(observation['findings']), verified_file_count=len(evidence.files),
        cost_summary_at_audit=summary, scientific_rows_replayed=0, new_native_calls=0,
        scope_amendment=registry['preliminary_scope_amendment'])
    with locked(campaign/'supervisor/registry.json.lock'):
        require(read(campaign/'supervisor/registry.json')['runs'] == registry['runs'],
                'Registry changed during authentication')
        write_artifact(output/'attempt-history.json', history)
        write_artifact(output/'verified-files.json', evidence.files)
        result.update(attempt_history=binding(output/'attempt-history.json'),
                      verified_files=binding(output/'verified-files.json'))
        write_artifact(output/'authentication.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('campaign'); p.add_argument('output'); p.add_argument('--supervisor-step', required=True)
    args = p.parse_args()
    result = audit(args.campaign, args.output, args.supervisor_step)
    print({k:result[k] for k in ('status','all_registered_attempts','active_completed_attempts','verified_file_count')})


if __name__ == '__main__':
    main()
