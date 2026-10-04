"""No-replay admission and denominator preservation for the ontology omission repair."""
import copy
from pathlib import Path

import pytest

from tools.repair import schema_omission_recovery as recovery


def test_unknown_preserves_original_failure_cost_and_payloads():
    previous = dict(status='error', detail=recovery.CAUSE, cleanup_complete=True, result=None,
        recovery_action='compatible_remaining_budget_execution', identity='old', content_hash='old',
        elapsed_seconds=25, resources={'cpu_seconds':12}, payloads=[{'path':'partial'}],
        remaining_budget={'wall_seconds':300}, row={'id':'failed'})
    original = copy.deepcopy(previous)
    owner = dict(step_id='14408.392', surviving_processes=0, cgroup_absent=True, signals_sent=0)
    saved = recovery.reconciled_unknown(previous, {'path':'old'}, {'path':'evidence'}, owner, '14408.392')
    assert previous == original and saved['status'] == 'unknown_after_omission_failure'
    assert saved['result'] is None and saved['additional_elapsed_seconds'] == 0
    assert saved['original_detail'] == recovery.CAUSE and not saved['prior_costs_reset']
    for key in ('elapsed_seconds', 'resources', 'payloads', 'row', 'remaining_budget'):
        assert saved[key] == original[key]
    for change in ('detail','cleanup_complete','result'):
        altered = copy.deepcopy(previous)
        altered[change] = {'detail':'Other error','cleanup_complete':False,'result':{'score':1}}[change]
        with pytest.raises(ValueError): recovery.reconciled_unknown(altered, {}, {}, owner, '14408.392')
    owner['surviving_processes'] = 1
    with pytest.raises(ValueError): recovery.reconciled_unknown(previous, {}, {}, owner, '14408.392')


@pytest.mark.parametrize('change', [None,'nonce','command','identity','missing_prefix','guard','spent_suffix','other_error'])
def test_original_validation_and_no_replay_report(tmp_path, monkeypatch, change):
    from exact.repair.api import write_artifact
    from exact.repair.records import canonical_hash
    from tools.repair.batch import sha
    from tools.repair.expanded_corpus import binding
    from tools.repair.expanded_profile import checkpoint
    from tools.repair import schema_recovery as science

    work=tmp_path/'work'/'evaluation'; source=tmp_path/'source'; (source/'tools/repair').mkdir(parents=True)
    for name in ('schema_recovery.py','fresh_evaluation.py'): (source/'tools/repair'/name).write_text('original')
    monkeypatch.setattr(science,'__file__',str(source/'tools/repair/schema_recovery.py'))
    monkeypatch.setattr(science.fresh,'__file__',str(source/'tools/repair/fresh_evaluation.py'))
    rows=[dict(id=f'row{i}') for i in range(3)]; entries=[dict(row_id=r['id']) for r in rows]
    admission=[dict(row_id=r['id'],compatible=True) for r in rows]
    schedule=tmp_path/'schedule.json';write_artifact(schedule,dict(rows=rows))
    preflight=tmp_path/'preflight.json';write_artifact(preflight,dict(rows=admission))
    schema=tmp_path/'schema.json';write_artifact(schema,dict(rows=entries,schedule=binding(schedule),preflight=binding(preflight)))
    identity=canonical_hash((sha(schema),sha(science.__file__),sha(science.fresh.__file__),{}))
    completion=tmp_path/'completion.json';write_artifact(completion,dict(status='failed',exit_code=1,
        step_id='14408.392',dispatch_nonce='nonce',work=str(work.parent),batch=str(tmp_path/'batch.json'),
        error={'message':recovery.COMMAND_ERROR}))
    owner=tmp_path/'step.json';write_artifact(owner,dict(step_id='14408.392',dispatch_nonce='bad' if change=='nonce' else 'nonce'))
    command=tmp_path/'command-0.json';write_artifact(command,dict(cwd=str(source),argv=['python','-m',
        'wrong' if change=='command' else 'tools.repair.schema_recovery',str(schema),str(work),'--start','0','--stop','3']))
    refs=[]
    for i in range(2):
        p=work/'rows'/(rows[i]['id']+'.json')
        checkpoint(p,'bad' if change=='identity' else canonical_hash((identity,rows[i],entries[i],admission[i])),
            row=rows[i],status='complete' if i==0 else 'error',
            detail='' if i==0 else ('Other error' if change=='other_error' else recovery.CAUSE),
            cleanup_complete=True,result=None,payloads=[],recovery_action='compatible_remaining_budget_execution',
            elapsed_seconds=25,resources={'cpu_seconds':12},remaining_budget={'wall_seconds':300})
        refs.append(dict(index=i,receipt=binding(p)))
    if change=='guard':write_artifact(work/'inflight/row1.json',{})
    if change=='spent_suffix':(work/'payloads/row2').mkdir(parents=True)
    evidence=tmp_path/'evidence.json';write_artifact(evidence,dict(completion=binding(completion),ownership=binding(owner),
        command=binding(command),rows=refs[1:] if change=='missing_prefix' else refs))
    output=work.parent/'evaluation-revision-002'
    plan=tmp_path/'plan.json';write_artifact(plan,dict(original_slice=[0,3],failed_index=1,
        only_unstarted_slice=[2,3],original_row_count=3,prior_costs_reset=False,original_step='14408.392',
        frozen_batch={'path':str(tmp_path/'batch.json')},source_root=str(source),schema_plan=binding(schema),
        evidence=binding(evidence),runner=binding(recovery.__file__),output=str(output),runtime={},original_runtime={},schema_identity=identity))
    from tools.repair.batch import read
    if change:
        with pytest.raises(ValueError):recovery.validate_original(read(plan),read(evidence),identity)
        return
    monkeypatch.setattr(recovery,'validate_source',lambda p:None)
    monkeypatch.setattr(recovery,'confirm_owner_gone',lambda step:dict(step_id=step,surviving_processes=0,cgroup_absent=True,signals_sent=0))
    executed=[]
    def untouched(plan_path,destination,start,stop):
        assert (start,stop)==(2,3)
        p=destination/'rows/row2.json'
        if not p.exists():
            checkpoint(p,'new',row=rows[2],status='timeout',result=None);executed.append(2)
        return dict(scheduled=1,recorded=1,rows=[dict(**binding(p),row_id='row2',status='timeout')])
    monkeypatch.setattr(science,'run',untouched)
    originals={ref['receipt']['path']:Path(ref['receipt']['path']).read_bytes() for ref in refs}
    result=recovery.run(plan,output);assert recovery.run(plan,output)==result
    assert executed==[2] and result['scheduled']==result['recorded']==3
    assert result['outcomes']==dict(complete=1,unknown_after_omission_failure=1,timeout=1)
    assert all(Path(p).read_bytes()==b for p,b in originals.items())


@pytest.mark.parametrize('change', [None,'unrelated_source','original_source','runtime','original_runtime','missing_file'])
def test_source_revision_admits_only_two_bound_implementation_changes(tmp_path, monkeypatch, change):
    from exact.repair import study
    from exact.repair.api import write_artifact
    from tools.repair.batch import sha
    from tools.repair.expanded_corpus import binding
    from tools.repair import schema_recovery as science

    original=tmp_path/'old/code';revised=tmp_path/'new/code'
    names=['exact/repair/grammar.py','exact/repair/pipeline.py','tools/repair/schema_recovery.py']
    files={};old_files={}
    for name in names:
        a,b=original/name,revised/name;a.parent.mkdir(parents=True,exist_ok=True);b.parent.mkdir(parents=True,exist_ok=True)
        a.write_text('old');b.write_text('new' if name.startswith('exact/') else 'old')
        old_files[str(a)]=sha(a);files[name]=sha(b)
    old_runtime=dict(code_hashes={'grammar.py':'old','pipeline.py':'old'},dependency='same')
    new_runtime=dict(code_hashes={'grammar.py':'new','pipeline.py':'new'},dependency='same')
    runtime=original.parent/'runtime.json';write_artifact(runtime,old_runtime);old_files[str(runtime)]=sha(runtime)
    batch=tmp_path/'batch.json';write_artifact(batch,dict(code=str(original),frozen_files=old_files))
    if change=='unrelated_source':
        (revised/names[2]).write_text('different');files[names[2]]=sha(revised/names[2])
    if change=='original_source':(original/names[0]).write_text('tampered')
    if change=='runtime':new_runtime['dependency']='different'
    if change=='original_runtime':old_runtime['dependency']='different'
    if change=='missing_file':files.pop(names[2])
    manifest=tmp_path/'source.json';write_artifact(manifest,dict(files=files,changed_files=names[:2]))
    plan=dict(source_root=str(original),corrected_source_root=str(revised),frozen_batch=binding(batch),
        source_revision=binding(manifest),original_runtime=old_runtime,runtime=new_runtime)
    monkeypatch.setattr(science,'__file__',str(revised/names[2]))
    monkeypatch.setattr(study,'runtime_manifest',lambda:new_runtime)
    if change:
        with pytest.raises(ValueError):recovery.validate_source(plan)
    else:assert recovery.validate_source(plan)['files']==files
