"""Closure must reject unfinished work and accounting/scientific promotion."""
from copy import deepcopy
import pytest

from tools.repair.preliminary_close import (
    RUN, SLOT, completion_candidate, preserve_accounting, validate_readiness,
)


def queue():
    registry = dict(campaign='test', runs=[dict(id=RUN)], remaining_work_status='pending',
        pending_batches=[dict(id=SLOT, preparation_only=True, depends_on=[RUN])])
    observation = dict(findings=[dict(run_id=RUN, step_id='14408.555', status='complete')],
                       incidents=[dict(kind='next_batch')])
    return registry, observation, {}, {'14408.0':'RUNNING', '14408.338':'RUNNING'}, '14408.338'


def test_queue_candidate_does_not_mutate_registry_or_observation():
    args = queue(); saved = deepcopy(args)
    candidate, observation = completion_candidate(*args)
    assert candidate['pending_batches'] == [] and candidate['remaining_work_status'] == 'terminal'
    assert observation['incidents'] == [] and args == saved


@pytest.mark.parametrize('mutation', ['live', 'scheduler', 'missing_access', 'slot', 'launch',
                                    'findings', 'failure', 'incident', 'dispatch'])
def test_unfinished_or_unobserved_work_prevents_closure(mutation):
    args = list(queue()); registry, observation, dispatch, steps, _ = args
    if mutation == 'live': steps['14408.555'] = 'RUNNING'
    if mutation == 'scheduler': args[3] = None
    if mutation == 'missing_access': del steps['14408.0']
    if mutation == 'slot': registry['pending_batches'].append(dict(id='unaccounted'))
    if mutation == 'launch': registry['pending_batches'][0]['launch'] = {'argv':['srun']}
    if mutation == 'findings': observation['findings'] = []
    if mutation == 'failure': observation['findings'][0]['status'] = 'failed'
    if mutation == 'incident': observation['incidents'].append(dict(kind='run_failed'))
    if mutation == 'dispatch': dispatch['work'] = dict(status='reserved')
    with pytest.raises(ValueError): completion_candidate(*args)


def accounting():
    row = dict(status='settled', elapsed_seconds=5, reserved_seconds=20,
               resources=dict(cpus=1,gpus=0,memory_mb=8192), logical_id='old', started_epoch=1)
    return dict(limit_worker_seconds=None, attempts={'failed':row}, cumulative=dict(
        worker_seconds=5,allocated_cpu_seconds=5,allocated_gpu_seconds=0,
        allocated_memory_mb_seconds=40960, measured_cpu_seconds=0,external_api_cost_usd=0))


def test_preserves_failed_cost_and_original_reservation_on_settlement():
    previous=accounting(); previous['attempts']['maintenance']=dict(
        status='reserved',reserved_seconds=2400,logical_id='close',started_epoch=10,
        resources=dict(cpus=1,gpus=0,memory_mb=8192))
    current=deepcopy(previous)
    current['attempts']['maintenance'].update(status='settled',elapsed_seconds=3)
    current['cumulative'].update(worker_seconds=8,allocated_cpu_seconds=8,
                                 allocated_memory_mb_seconds=65536)
    preserve_accounting(previous,current,settling={'maintenance'})


@pytest.mark.parametrize('mutation', ['drop', 'cost', 'reservation', 'memory', 'cpu', 'api', 'limit'])
def test_any_lost_cost_or_unsettled_reservation_rejects_closure(mutation):
    previous=accounting(); current=deepcopy(previous)
    if mutation=='drop': del current['attempts']['failed']
    if mutation=='cost': current['attempts']['failed']['elapsed_seconds']=4
    if mutation=='reservation': current['attempts']['new']=dict(status='reserved')
    if mutation=='memory': current['cumulative']['allocated_memory_mb_seconds']=1
    if mutation=='cpu': current['cumulative']['allocated_cpu_seconds']=1
    if mutation=='api': current['cumulative']['external_api_cost_usd']=1
    if mutation=='limit': current['limit_worker_seconds']=172800
    with pytest.raises(ValueError): preserve_accounting(previous,current)


def readiness():
    deferred=[dict(status='deferred_by_user',obligation=str(n)) for n in range(4)]
    scope=dict(local_scientific_dependencies_accounted=True,local_scope_complete=False,
        full_expanded_program_complete=False,fitting_eligible=False,learning_efficiency_qualified=False,
        strongest_symbolic_comparison_qualified=False,gates=dict.fromkeys(('G0','G1','G2'),'not_established'),
        deferred_work=deferred,api_spend_usd=0,llm_labels='disabled')
    report=dict(status='complete',local_report_complete=True,scope_readiness=scope,
        campaign_complete=False,full_program_complete=False,gates_passed=False,scientific_rows_replayed=0,
        new_native_calls=0,new_fitting_runs_executed=0,deferred_work=deferred,api_spend_usd=0,
        proposed_cluster_design=dict(status='proposal_not_frozen_or_approved'))
    return report,scope,dict(deferred_work=[str(n) for n in range(4)])


def test_local_closure_keeps_training_and_gates_unqualified():
    validate_readiness(*readiness())


@pytest.mark.parametrize('mutation',['fitting','gate','deferral','cluster','completion','labels'])
def test_reporting_cannot_promote_scientific_readiness(mutation):
    report,scope,amendment=readiness()
    if mutation=='fitting': report['new_fitting_runs_executed']=18
    if mutation=='gate': scope['gates']['G0']='passed'
    if mutation=='deferral': scope['deferred_work'].pop()
    if mutation=='cluster': report['proposed_cluster_design']['status']='approved'
    if mutation=='completion': scope['local_scope_complete']=True
    if mutation=='labels': scope['llm_labels']='enabled'
    with pytest.raises(ValueError):validate_readiness(report,scope,amendment)
