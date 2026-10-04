"""Owner-checked scaling recovery conserves spent rows and cold/warm budgets."""
from pathlib import Path

import pytest

from exact.repair.api import write_artifact
from exact.repair.records import canonical_hash
from exact.repair.workers import CallResult
from tools.repair import scaling, scaling_payload_recovery as payload, scaling_cleanup_recovery as recovery
from tools.repair.batch import read, sha
from tools.repair.expanded_corpus import binding, bound
from tools.repair.expanded_profile import checkpoint


def fixture(tmp_path, monkeypatch):
    import exact.repair.study
    import tools.repair.batch
    monkeypatch.setattr(exact.repair.study, 'runtime_manifest', lambda: {})
    source = str(Path(scaling.__file__).resolve().parents[2])
    batch_path = tmp_path/'science.json'; write_artifact(batch_path, dict(code=source))
    monkeypatch.setattr(tools.repair.batch, 'checked_batch', lambda p: read(p))
    monkeypatch.setattr(recovery, 'validate_previous_export', bound)
    rows = [dict(id=str(i), cache_mode='warm' if i%2 else 'cold', pair_id=str(i//2),
        seconds=300, cpu_seconds=600, memory_mb=8192) for i in range(4)]
    schedule = tmp_path/'schedule.json';write_artifact(schedule,dict(rows=rows))
    science_identity = canonical_hash((binding(schedule),sha(scaling.__file__),{}))
    work=tmp_path/'work';original=work/'revision-003';output=work/'revision-004'
    previous=tmp_path/'previous.json'
    write_artifact(previous,dict(schedule=binding(schedule),slice=[0,4],output=str(original),
        adapter=binding(payload.__file__),source_root=source,runtime={},
        scientific_batch=binding(batch_path),scientific_identity=science_identity,recovery=None))
    identity=canonical_hash((binding(previous),science_identity));refs=[]
    for i in range(2):
        row=rows[i];rid=canonical_hash((identity,row));directory=original/'payloads'/str(i)
        directory.mkdir(parents=True);(directory/'partial.json').write_text('{}')
        published,debris=payload.payload_manifest(directory)
        raw=dict(row=row,status='error' if i==1 else 'timeout',detail=recovery.CAUSE if i==1 else '',
            result=None,cleanup_complete=True,resources={'wall_seconds':238},elapsed_seconds=238)
        receipt=original/'rows'/f'{i}.json';call=original/'calls'/f'{i}.json'
        checkpoint(call,rid,**raw);checkpoint(receipt,rid,**raw,payloads=published,unpublished_payloads=debris)
        refs.append(dict(index=i,receipt=binding(receipt),call=binding(call)))
    guard=original/'inflight/1.json';checkpoint(guard,canonical_hash((identity,rows[1])),row=rows[1],started_epoch=1)
    argv=['python',payload.__file__,str(previous),str(original),'--code',source]
    wrapper=tmp_path/'wrapper.json';write_artifact(wrapper,dict(python='python',code=source,
        jobs=[dict(id='job',commands=[['{python}',*argv[1:]]])]))
    attempt=tmp_path/'attempt';completion=attempt/'completion.json';owner=attempt/'step.json';command=attempt/'command-0.json'
    write_artifact(completion,dict(status='failed',exit_code=1,step_id='14408.267',dispatch_nonce='nonce',
        batch=str(wrapper),work=str(work),job_id='job',error={'message':recovery.COMMAND_ERROR}))
    write_artifact(owner,dict(step_id='14408.267',dispatch_nonce='nonce'))
    write_artifact(command,dict(argv=argv,cwd=source))
    evidence=tmp_path/'evidence.json';write_artifact(evidence,dict(completion=binding(completion),
        owner=binding(owner),command=binding(command),guard=binding(guard),rows=refs))
    continuation=tmp_path/'continuation.json';write_artifact(continuation,dict(read(previous),slice=[2,4],output=str(output/'continuation')))
    plan=tmp_path/'plan.json';write_artifact(plan,dict(previous_plan=binding(previous),previous_batch=binding(wrapper),
        evidence=binding(evidence),continuation_plan=binding(continuation),failed_index=1,
        repair_attempt=1,max_repairs=2,prior_costs_reset=False,original_step='14408.267',
        runner=binding(recovery.__file__),output=str(output)))
    return plan,output,original


def owner(step):
    return dict(step_id=step,cgroup_absent=True,surviving_processes=0,signals_sent=0)


def test_recovery_never_replays_spent_rows_and_preserves_warm_cache(tmp_path,monkeypatch):
    import exact.repair.workers
    plan,output,original=fixture(tmp_path,monkeypatch);calls=[]
    before={str(p):sha(p) for p in original.rglob('*.json')}
    def call(fn,schedule,row,directory,cache_source,**limits):
        calls.append((fn,row['id'],cache_source,limits))
        return CallResult('timeout',cleanup_complete=True)
    monkeypatch.setattr(exact.repair.workers,'bounded_call',call)
    report=recovery.run(plan,output,owner_check=owner)
    assert report['recorded_rows']==report['scheduled_rows']==4
    assert report['scientific_rows_replayed']==0
    assert [x[1] for x in calls]==['2','3']
    assert all(x[0] is scaling.evaluate and x[3]==dict(timeout=300,cpu_seconds=600,memory_mb=8192) for x in calls)
    assert calls[0][2] is None and calls[1][2]==str(output/'continuation/payloads/2/compiler-cache')
    unknown=bound(report['rows'][1]);assert unknown['result'] is None
    assert unknown['status']=='unknown_after_cleanup_reconciliation' and unknown['original_detail']==recovery.CAUSE
    assert unknown['additional_scientific_seconds']==0
    assert {str(p):sha(p) for p in original.rglob('*.json')}==before
    assert recovery.run(plan,output,owner_check=owner)==report and len(calls)==2


def test_rejects_surviving_owner_before_any_continuation(tmp_path,monkeypatch):
    plan,output,_=fixture(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='owner cleanup'):
        recovery.run(plan,output,owner_check=lambda s:dict(owner(s),surviving_processes=1))
    assert not (output/'continuation').exists()


@pytest.mark.parametrize('kind',['rows','calls','inflight','payloads'])
def test_rejects_previously_attempted_continuation(tmp_path,monkeypatch,kind):
    plan,output,original=fixture(tmp_path,monkeypatch)
    path=original/kind/('2' if kind=='payloads' else '2.json');path.parent.mkdir(exist_ok=True)
    path.mkdir() if kind=='payloads' else path.write_text('{}')
    with pytest.raises(ValueError,match='already spent work'):
        recovery.run(plan,output,owner_check=owner)


@pytest.mark.parametrize('field',['rows','guard','owner','command','call','payload'])
def test_rejects_missing_or_changed_evidence(tmp_path,monkeypatch,field):
    plan,output,original=fixture(tmp_path,monkeypatch);p=read(plan);e=bound(p['evidence'])
    if field=='rows':
        e['rows']=e['rows'][1:];write_artifact(p['evidence']['path'],e)
        p['evidence']=binding(p['evidence']['path']);write_artifact(plan,p)
    else:
        path=(original/'payloads/0/partial.json' if field=='payload' else
            Path(e['rows'][0]['call']['path']) if field=='call' else Path(e[field]['path']))
        path.write_text('{"changed":true}')
    with pytest.raises((ValueError,KeyError)):
        recovery.run(plan,output,owner_check=owner)


def test_rejects_budget_reset_in_continuation(tmp_path,monkeypatch):
    plan,output,_=fixture(tmp_path,monkeypatch);p=read(plan);c=bound(p['continuation_plan'])
    c['slice']=[0,4];write_artifact(p['continuation_plan']['path'],c)
    p['continuation_plan']=binding(p['continuation_plan']['path']);write_artifact(plan,p)
    with pytest.raises(ValueError,match='untouched pairs'):
        recovery.run(plan,output,owner_check=owner)
