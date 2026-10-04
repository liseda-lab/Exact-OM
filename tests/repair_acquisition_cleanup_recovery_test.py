import copy
from pathlib import Path

import pytest

from tools.repair import acquisition_recovery as recovery

STEP = '14408.204'


def fixture():
    previous=dict(case_id='case22',split='development',status='error',outer_status='error',detail=recovery.CAUSE,
                  cleanup_complete=True,result=None,supervision_eligible=False,elapsed_seconds=35.8,
                  resources={'cpu_seconds':17.76},native_evidence=[],identity='original',content_hash='old')
    call=dict(case_id='case22',split='development',status='timeout',detail='stage deadline exhausted',
              cleanup_complete=False,assignment=None,deadline_seconds=30.0,resources={'wall_seconds':30.206})
    owner=dict(step_id=STEP,surviving_processes=0,cgroup_absent=True,signals_sent=0)
    return previous,call,owner


def test_reconciliation_preserves_timeout_cost_and_original_evidence():
    previous,call,owner=fixture();original=copy.deepcopy(previous)
    result=recovery.reconciled_timeout(previous,call,{'path':'old','sha256':'oldhash'},{'path':'evidence','sha256':'proof'},owner,STEP)
    assert previous==original
    assert result['status']=='unknown_intended_parent' and result['result'] is None
    assert result['outer_status']=='error' and result['original_detail']==recovery.CAUSE
    assert result['resources']==previous['resources'] and result['elapsed_seconds']==previous['elapsed_seconds']
    assert result['additional_elapsed_seconds']==0 and not result['supervision_eligible']
    assert not result['prior_costs_reset']


@pytest.mark.parametrize('change', ['software_error','complete','clean_native','different_case','budget_not_spent','live_owner','positive_label'])
def test_reconciliation_rejects_unqualified_or_positive_evidence(change):
    previous,call,owner=fixture()
    if change=='software_error':call.update(status='error',detail='RuntimeError: native crash')
    elif change=='complete':call['status']='complete'
    elif change=='clean_native':call['cleanup_complete']=True
    elif change=='different_case':call['case_id']='other'
    elif change=='budget_not_spent':call['resources']['wall_seconds']=29
    elif change=='live_owner':owner['surviving_processes']=1
    elif change=='positive_label':previous['result']={'label':'must not accept'}
    with pytest.raises(ValueError):
        recovery.reconciled_timeout(previous,call,{}, {},owner,STEP)


@pytest.mark.parametrize('start,split', [(16,'development'), (48,'train')])
def test_runner_reuses_six_rows_and_never_reexecutes_reconciled_timeout(tmp_path,monkeypatch,start,split):
    from exact.repair.api import write_artifact
    from exact.repair.records import canonical_hash
    from exact.repair import study
    from tools.repair import acquisition
    from tools.repair.batch import sha
    from tools.repair.expanded_corpus import binding
    from tools.repair.expanded_profile import checkpoint

    source=tmp_path/'source';tool=source/'tools/repair';tool.mkdir(parents=True)
    for name in ('acquisition.py','train.py'):(tool/name).write_text('fixture source')
    monkeypatch.setattr(acquisition,'__file__',str(tool/'acquisition.py'))
    monkeypatch.setattr(study,'runtime_manifest',lambda:{'fixture':'same'})
    acquisition_plan=tmp_path/'acquisition-plan.json';write_artifact(acquisition_plan,{'fixed':'plan','split':split})
    evidence=tmp_path/'evidence.json';write_artifact(evidence,{'fixed':'ownership'})
    previous,call,owner=fixture();prior={}
    for index in range(start,start+6):
        path=tmp_path/f'original-{index}.json';saved=checkpoint(path,'old',case_id=f'case{index}',status='partial',result=None)
        prior[index]=(binding(path),saved)
    failed=tmp_path/'failed.json';write_artifact(failed,previous);prior[start+6]=(binding(failed),previous)
    originals={ref['path']:Path(ref['path']).read_bytes() for ref,saved in prior.values()}
    monkeypatch.setattr(recovery,'validate_original',lambda p,e,i:(prior,call))
    monkeypatch.setattr(recovery,'confirm_owner_gone',lambda step:owner)
    requested=[];executed=[]
    def last_case(plan,output,start,stop):
        requested.append((start,stop));path=output/'last.json'
        if not path.exists():
            executed.append(stop-1);checkpoint(path,'last',case_id='case23',status='timeout',result=None)
        return dict(scheduled_rows=1,recorded_rows=1,rows=[binding(path)])
    monkeypatch.setattr(acquisition,'run',last_case)
    output=tmp_path/'recovery';plan_path=tmp_path/'plan.json'
    from tools.repair import batch
    frozen=tmp_path/'batch.json';write_artifact(frozen,{'code':str(source)})
    monkeypatch.setattr(batch,'checked_batch',lambda p: {'code':str(source)})
    identity=canonical_hash((sha(acquisition_plan),sha(tool/'acquisition.py'),sha(tool/'train.py'),{'fixture':'same'}))
    write_artifact(plan_path,dict(runner=binding(recovery.__file__),output=str(output),source_root=str(source),
        runtime={'fixture':'same'},acquisition_plan=binding(acquisition_plan),acquisition_identity=identity,evidence=binding(evidence), frozen_batch=binding(frozen),
        original_slice=[start,start+8],failed_index=start+6,only_unstarted_slice=[start+7,start+8],
        original_case_count=8,prior_costs_reset=False,original_step=STEP))
    first=recovery.run(plan_path,output);second=recovery.run(plan_path,output)
    assert first==second and requested==[(start+7,start+8)]*2 and executed==[start+7]
    assert first['scheduled_rows']==first['recorded_rows']==8
    assert first['counts']=={'partial':6,'unknown_intended_parent':1,'timeout':1}
    assert first['rows'][:6]==[prior[i][0] for i in range(start,start+6)]
    assert all(Path(path).read_bytes()==raw for path,raw in originals.items())


@pytest.mark.parametrize('change', ['spent_last', 'skip_row', 'cost_reset', 'wrong_count'])
def test_recovery_rejects_changed_budget_or_denominator(change):
    plan=dict(original_slice=[48,56],failed_index=54,only_unstarted_slice=[55,56],
              original_case_count=8,prior_costs_reset=False)
    if change=='spent_last':plan['only_unstarted_slice']=[54,56]
    elif change=='skip_row':plan['failed_index']=53
    elif change=='cost_reset':plan['prior_costs_reset']=True
    else:plan['original_case_count']=7
    with pytest.raises(ValueError):recovery.slice_bounds(plan)
