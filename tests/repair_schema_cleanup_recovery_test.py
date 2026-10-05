import copy
from pathlib import Path

import pytest

from tools.repair import schema_cleanup_recovery as recovery

STEP = '14408.223'


def fixture():
    row = dict(id='row8', seconds=300, cpu_seconds=600)
    budget = dict(wall_seconds=282.8, prior_wall_seconds=17.2,
                  cpu_seconds=590.9, prior_cpu_seconds=9.1,
                  generation_seconds=45.9, prior_generation_seconds=14.1)
    saved = dict(row=row, status='error', detail=recovery.CAUSE, cleanup_complete=True,
        result=None, payloads=[], elapsed_seconds=219.8, resources={'cpu_seconds':37.02},
        remaining_budget=budget, recovery_action='compatible_remaining_budget_execution',
        identity='original', content_hash='old')
    guard = dict(row=copy.deepcopy(row), remaining_budget=copy.deepcopy(budget), identity='original')
    owner = dict(step_id=STEP, surviving_processes=0, cgroup_absent=True, signals_sent=0)
    return saved, guard, owner


def test_reconciliation_retains_unknown_error_costs_and_budget_without_replay():
    saved, guard, owner = fixture()
    original = copy.deepcopy(saved)
    result = recovery.reconciled_unknown(saved, guard, {'path':'old'}, {'path':'proof'}, owner, STEP)
    assert saved == original
    assert result['status'] == 'unknown_after_cleanup_reconciliation' and result['result'] is None
    assert result['original_status'] == 'error' and result['original_detail'] == recovery.CAUSE
    for key in ('resources', 'elapsed_seconds', 'remaining_budget', 'payloads', 'row'):
        assert result[key] == original[key]
    assert result['additional_elapsed_seconds'] == 0 and not result['prior_costs_reset']


@pytest.mark.parametrize('change', ['other_error', 'positive_result', 'outer_cleanup', 'different_row',
    'budget_reset', 'identity', 'live_owner', 'wrong_step', 'signals', 'cgroup'])
def test_reconciliation_rejects_unqualified_evidence(change):
    saved, guard, owner = fixture()
    if change == 'other_error': saved['detail'] = 'ValueError: unrelated crash'
    elif change == 'positive_result': saved['result'] = {'assignment':[0]}
    elif change == 'outer_cleanup': saved['cleanup_complete'] = False
    elif change == 'different_row': guard['row']['id'] = 'other'
    elif change == 'budget_reset': guard['remaining_budget']['wall_seconds'] = 300
    elif change == 'identity': guard['identity'] = 'different'
    elif change == 'live_owner': owner['surviving_processes'] = 1
    elif change == 'wrong_step': owner['step_id'] = '14408.0'
    elif change == 'signals': owner['signals_sent'] = 1
    else: owner['cgroup_absent'] = False
    with pytest.raises(ValueError):
        recovery.reconciled_unknown(saved, guard, {}, {}, owner, STEP)


@pytest.mark.parametrize('change', ['replay_failed', 'skip_row', 'cost_reset', 'wrong_count'])
def test_recovery_rejects_changed_slice(change):
    plan = dict(original_slice=[352,368], failed_index=360, only_unstarted_slice=[361,368],
                original_row_count=16, prior_costs_reset=False)
    if change == 'replay_failed': plan['only_unstarted_slice'] = [360,368]
    elif change == 'skip_row': plan['failed_index'] = 361
    elif change == 'cost_reset': plan['prior_costs_reset'] = True
    else: plan['original_row_count'] = 15
    with pytest.raises(ValueError): recovery.slice_bounds(plan)


@pytest.mark.parametrize('change', [None, 'nonce', 'command', 'missing_prefix', 'row_identity',
    'guard_identity', 'spent_next_row', 'spent_next_payload', 'finished_guard',
    'wrong_command_index', 'other_error', 'different_attempt'])
@pytest.mark.parametrize('command_index', [0, 1])
def test_original_validation_binds_prefix_guard_and_unstarted_rows(tmp_path, change, command_index):
    from exact.repair.api import write_artifact
    from exact.repair.records import canonical_hash
    from tools.repair.expanded_corpus import binding
    from tools.repair.expanded_profile import checkpoint

    work=tmp_path/'work'/'evaluation'
    rows=[dict(id=f'row{i}') for i in range(3)]
    entries=[dict(row_id=r['id']) for r in rows]
    admission=[dict(row_id=r['id'],compatible=True) for r in rows]
    schedule=tmp_path/'schedule.json';write_artifact(schedule,dict(rows=rows))
    preflight=tmp_path/'preflight.json';write_artifact(preflight,dict(rows=admission))
    schema_plan=tmp_path/'schema-plan.json'
    write_artifact(schema_plan,dict(rows=entries,schedule=binding(schedule),preflight=binding(preflight)))
    plan=dict(original_slice=[0,3],failed_index=1,only_unstarted_slice=[2,3],
        original_row_count=3,prior_costs_reset=False,original_step=STEP,
        frozen_batch={'path':'/original/batch.json'},source_root='/original/code',schema_plan=binding(schema_plan))
    receipt=tmp_path/'completion.json'
    write_artifact(receipt,dict(status='failed',exit_code=1,step_id=STEP,dispatch_nonce='nonce',
        error={'message': ('command 0 exited 1: ValueError: other failure' if change=='other_error' else
            recovery.COMMAND_ERROR.replace('command 1', f'command {command_index}'))},
        batch='/original/batch.json',work=str(work.parent)))
    owner=tmp_path/'step.json';write_artifact(owner,dict(step_id=STEP,dispatch_nonce='wrong' if change=='nonce' else 'nonce'))
    receipt_index=command_index+1 if change=='wrong_command_index' else command_index
    command=(tmp_path/'other-attempt' if change=='different_attempt' else tmp_path)/f'command-{receipt_index}.json'
    write_artifact(command,dict(cwd='/wrong' if change=='command' else '/original/code',
        argv=['python','-m','tools.repair.schema_recovery',str(schema_plan),str(work),'--start','0','--stop','3']))
    refs=[]
    for i in range(2):
        path=work/'rows'/f'row{i}.json'
        identity=canonical_hash(('science',rows[i],entries[i],admission[i]))
        checkpoint(path,'changed' if change=='row_identity' else identity,
            row=rows[i],status='complete' if i==0 else 'error',detail='' if i==0 else recovery.CAUSE,
            cleanup_complete=True,payloads=[],result=None)
        refs.append(dict(index=i,receipt=binding(path)))
    guard=work/'inflight'/'row1.json'
    identity=canonical_hash(('science',rows[1],entries[1],admission[1]))
    checkpoint(guard,'changed' if change=='guard_identity' else identity,row=rows[1])
    if change=='spent_next_row': write_artifact(work/'rows'/'row2.json',{})
    if change=='spent_next_payload': (work/'payloads'/'row2').mkdir(parents=True)
    if change=='finished_guard': write_artifact(work/'inflight'/'row0.json',{})
    evidence=dict(completion=binding(receipt),ownership=binding(owner),command=binding(command),
                  rows=refs[1:] if change=='missing_prefix' else refs,guard=binding(guard))
    if change:
        with pytest.raises(ValueError): recovery.validate_original(plan,evidence,'science')
    else:
        assert set(recovery.validate_original(plan,evidence,'science'))=={0,1}


@pytest.mark.parametrize('failed_index', [8, 15])
def test_runner_reuses_prefix_preserves_guard_and_science_denominator(tmp_path, monkeypatch, failed_index):
    from exact.experiments.science_health import inspect_science
    from exact.repair.api import write_artifact
    from exact.repair.records import canonical_hash
    from exact.repair import study
    from tools.repair import schema_recovery as science, batch
    from tools.repair.batch import sha
    from tools.repair.expanded_corpus import binding
    from tools.repair.expanded_profile import checkpoint

    source = tmp_path/'source'; tool = source/'tools/repair'; tool.mkdir(parents=True)
    for name in ('schema_recovery.py','fresh_evaluation.py'): (tool/name).write_text('original source')
    monkeypatch.setattr(science, '__file__', str(tool/'schema_recovery.py'))
    monkeypatch.setattr(science.fresh, '__file__', str(tool/'fresh_evaluation.py'))
    monkeypatch.setattr(study, 'runtime_manifest', lambda:{'fixture':'same'})
    work = tmp_path/'work'; work.mkdir()
    schedule = tmp_path/'schedule.json'
    write_artifact(schedule, {'rows':[{'id':f'row{i}'} for i in range(16)]})
    schema_plan = tmp_path/'schema-plan.json'; write_artifact(schema_plan, {'schedule':binding(schedule)})
    completion = tmp_path/'failed-completion.json'; write_artifact(completion, {'work':str(work)})
    previous, guard, owner = fixture(); prior = {}
    previous['row']['id'] = guard['row']['id'] = f'row{failed_index}'
    for index in range(failed_index):
        payload = work/f'result-{index}.json'; write_artifact(payload, {'row_id':f'row{index}','status':'timeout'})
        path = work/f'original-{index}.json'
        saved = checkpoint(path, 'old', row={'id':f'row{index}'}, status='complete', result=binding(payload))
        prior[index] = (binding(path), saved)
    failed = work/'failed.json'; write_artifact(failed, previous); prior[failed_index]=(binding(failed), previous)
    guard_path = work/'guard.json'; write_artifact(guard_path, guard)
    evidence = tmp_path/'evidence.json'; write_artifact(evidence, {'guard':binding(guard_path),'completion':binding(completion)})
    originals = {p:Path(p).read_bytes() for p in [guard_path,*[ref['path'] for ref,_ in prior.values()]]}
    monkeypatch.setattr(recovery, 'validate_original', lambda p,e,i:prior)
    monkeypatch.setattr(recovery, 'confirm_owner_gone', lambda step:owner)
    requested, executed = [], []
    def continue_rows(plan, output, start, stop):
        assert failed_index < 15, 'Final-row reconciliation must make no scientific call'
        requested.append((start,stop)); refs=[]
        for index in range(start,stop):
            path=output/f'row{index}.json'
            if not path.exists():
                executed.append(index);checkpoint(path,'original',row={'id':f'row{index}'},status='timeout',result=None)
            refs.append(dict(**binding(path),row_id=f'row{index}',status='timeout'))
        return dict(scheduled=stop-start,recorded=stop-start,rows=refs)
    monkeypatch.setattr(science, 'run', continue_rows)
    frozen=tmp_path/'batch.json';write_artifact(frozen, {'code':str(source)})
    monkeypatch.setattr(batch, 'checked_batch', lambda p: {'code':str(source)})
    identity=canonical_hash((sha(schema_plan),sha(science.__file__),sha(science.fresh.__file__),{'fixture':'same'}))
    output=work/'evaluation-revision-002';plan=tmp_path/'plan.json'
    write_artifact(plan,dict(runner=binding(recovery.__file__),output=str(output),source_root=str(source),
        runtime={'fixture':'same'},schema_plan=binding(schema_plan),schema_identity=identity,
        evidence=binding(evidence),frozen_batch=binding(frozen),original_step=STEP,
        original_slice=[0,16],failed_index=failed_index,only_unstarted_slice=[failed_index+1,16],original_row_count=16,prior_costs_reset=False))
    first=recovery.run(plan,output);second=recovery.run(plan,output)
    assert first==second
    assert requested==([(failed_index+1,16)]*2 if failed_index < 15 else [])
    assert executed==list(range(failed_index+1,16))
    assert first['cleanup_recovery']['continued_rows']==15-failed_index
    if failed_index == 15:
        assert not (output/'continuation').exists()
    assert first['scheduled']==first['recorded']==16
    expected={'complete':failed_index,'unknown_after_cleanup_reconciliation':1}
    if failed_index < 15: expected['timeout']=15-failed_index
    assert first['outcomes']==expected
    assert all(Path(path).read_bytes()==raw for path,raw in originals.items())
    attempt=tmp_path/'attempt';attempt.mkdir()
    write_artifact(attempt/'outputs.json',{'evaluation-revision-002/report.json':sha(output/'report.json')})
    run=dict(step_id='14408.999',dispatch_nonce='nonce',science_report_relative='evaluation-revision-002/report.json',
             completion_path=str(attempt/'completion.json'))
    complete=dict(status='complete',step_id=run['step_id'],dispatch_nonce='nonce',work=str(work))
    assert inspect_science(run,complete)==dict(failures=[],errors=[])


@pytest.mark.parametrize('failed_index', [-1, 16, 17, True])
def test_recovery_rejects_failure_outside_slice(failed_index):
    plan = dict(original_slice=[0,16], failed_index=failed_index,
                only_unstarted_slice=[failed_index+1,16], original_row_count=16,
                prior_costs_reset=False)
    with pytest.raises(ValueError):
        recovery.slice_bounds(plan)
