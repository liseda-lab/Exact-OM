import copy
from pathlib import Path

import pytest

from tools.repair import schema_cleanup_descendant as recovery
from tools.repair import schema_cleanup_recovery as first


def fixture(tmp_path, monkeypatch):
    from exact.repair.api import write_artifact
    from exact.repair.records import canonical_hash
    from exact.repair import study
    from tools.repair import batch,schema_recovery as science
    from tools.repair.batch import sha,read
    from tools.repair.expanded_corpus import binding
    from tools.repair.expanded_profile import checkpoint
    runtime={'test':'frozen'}
    monkeypatch.setattr(study,'runtime_manifest',lambda:runtime)
    monkeypatch.setattr(batch,'checked_batch',lambda path:read(path))
    monkeypatch.setattr(first,'confirm_owner_gone',lambda step:dict(step_id=step,cgroup_absent=True,surviving_processes=0,signals_sent=0))
    monkeypatch.setattr(recovery,'prior_adapter',lambda plan:(read(plan['previous_cleanup_plan']['path']),first))
    source=str(Path(science.__file__).resolve().parents[2])
    rows=[dict(id=f'row{i}',seconds=300,cpu_seconds=600) for i in range(7)]
    entries=[dict(row_id=r['id']) for r in rows]
    admission=[dict(row_id=r['id'],compatible=True) for r in rows]
    schedule=tmp_path/'schedule.json';write_artifact(schedule,dict(rows=rows))
    preflight=tmp_path/'preflight.json';write_artifact(preflight,dict(rows=admission))
    schema_plan=tmp_path/'schema-plan.json';write_artifact(schema_plan,dict(rows=entries,schedule=binding(schedule),preflight=binding(preflight)))
    identity=canonical_hash((sha(schema_plan),sha(science.__file__),sha(science.fresh.__file__),runtime))
    frozen=tmp_path/'original-batch.json';write_artifact(frozen,dict(code=source))
    work=tmp_path/'work';original=work/'evaluation';old_output=work/'revision2';output=work/'revision3'
    budget=dict(wall_seconds=300,generation_seconds=60,cpu_seconds=600,prior_wall_seconds=0,prior_cpu_seconds=0,prior_generation_seconds=0)
    def make_rows(directory,start,failed):
        refs=[]
        for i in range(start,failed+1):
            row_identity=canonical_hash((identity,rows[i],entries[i],admission[i]))
            path=directory/'rows'/f'row{i}.json'
            checkpoint(path,row_identity,row=rows[i],status='error' if i==failed else 'timeout',
                detail=first.CAUSE if i==failed else '',cleanup_complete=True,result=None,payloads=[],
                elapsed_seconds=201+i,resources=dict(cpu_seconds=20+i),remaining_budget=budget,
                recovery_action='compatible_remaining_budget_execution')
            refs.append(dict(index=i,receipt=binding(path)))
        guard=directory/'inflight'/f'row{failed}.json'
        checkpoint(guard,row_identity,row=rows[failed],remaining_budget=budget)
        return refs,binding(guard)
    refs,guard=make_rows(original,0,1)
    attempt=tmp_path/'attempt1';completion=attempt/'completion.json';command=attempt/'command-0.json';owner=attempt/'step.json'
    write_artifact(completion,dict(status='failed',exit_code=1,step_id='14408.235',dispatch_nonce='n1',batch=str(frozen),work=str(work),error={'message':first.COMMAND_ERROR.replace('command 1','command 0')}))
    write_artifact(command,dict(cwd=source,argv=['python','-m','tools.repair.schema_recovery',str(schema_plan),str(original),'--start','0','--stop','7']))
    write_artifact(owner,dict(step_id='14408.235',dispatch_nonce='n1'))
    prior_evidence=tmp_path/'evidence1.json';write_artifact(prior_evidence,dict(rows=refs,guard=guard,completion=binding(completion),command=binding(command),ownership=binding(owner)))
    prior_plan=tmp_path/'plan1.json'
    common=dict(original_slice=[0,7],original_row_count=7,prior_costs_reset=False,
        source_root=source,frozen_batch=binding(frozen),schema_plan=binding(schema_plan),runtime=runtime,schema_identity=identity)
    previous=dict(common,failed_index=1,only_unstarted_slice=[2,7],output=str(old_output),runner=binding(first.__file__),
        evidence=binding(prior_evidence),original_step='14408.235')
    write_artifact(prior_plan,previous)
    unknown=old_output/'reconciled-unknown.json'
    expected=first.reconciled_unknown(read(refs[1]['receipt']['path']),read(guard['path']),refs[1]['receipt'],binding(prior_evidence),first.confirm_owner_gone('14408.235'),'14408.235')
    checkpoint(unknown,canonical_hash((binding(prior_plan),identity)),**expected)
    second_refs,second_guard=make_rows(old_output/'continuation',2,4)
    second_batch=tmp_path/'second-batch.json'
    argv=['python',first.__file__,str(prior_plan),str(old_output),'--code',source]
    write_artifact(second_batch,dict(code='/separate/adapter/code',python='python',jobs=[dict(id='job',commands=[['link'],['{python}',*argv[1:]]])]))
    attempt2=tmp_path/'attempt2';completion2=attempt2/'completion.json';command2=attempt2/'command-1.json';owner2=attempt2/'step.json'
    write_artifact(completion2,dict(status='failed',exit_code=1,step_id='14408.238',dispatch_nonce='n2',batch=str(second_batch),work=str(work),job_id='job',error={'message':first.COMMAND_ERROR}))
    write_artifact(command2,dict(cwd='/separate/adapter/code',argv=argv));write_artifact(owner2,dict(step_id='14408.238',dispatch_nonce='n2'))
    evidence=tmp_path/'evidence2.json';write_artifact(evidence,dict(rows=second_refs,guard=second_guard,previous_unknown=binding(unknown),completion=binding(completion2),command=binding(command2),ownership=binding(owner2)))
    plan=dict(common,runner=binding(recovery.__file__),output=str(output),failed_index=4,only_unstarted_slice=[5,7],original_step='14408.238',
        previous_cleanup_plan=binding(prior_plan),previous_batch=binding(second_batch),repair_attempt=2,max_repairs=2,unsuccessful_prior_repairs=1,evidence=binding(evidence))
    return plan,read(evidence),identity


@pytest.mark.parametrize('change',[None,'attempt_reset','attempt3','missing_prefix','spent_row','guard','nonce','command','changed_unknown','changed_source','wrong_batch_command'])
def test_chain_requires_exact_ancestry_and_retains_all_spent_rows(tmp_path,monkeypatch,change):
    from exact.repair.api import write_artifact
    from tools.repair.batch import read
    from tools.repair.expanded_corpus import binding
    plan,evidence,identity=fixture(tmp_path,monkeypatch)
    if change=='attempt_reset': plan['repair_attempt']=1
    elif change=='attempt3': plan['repair_attempt']=3;plan['unsuccessful_prior_repairs']=2
    elif change=='missing_prefix': evidence['rows']=evidence['rows'][1:]
    elif change=='spent_row': write_artifact(tmp_path/'work/revision2/continuation/rows/row5.json',{})
    elif change=='changed_source': plan['source_root']='/other'
    elif change in ('nonce','command','changed_unknown','guard'):
        key={'nonce':'ownership','command':'command','changed_unknown':'previous_unknown','guard':'guard'}[change]
        path=Path(evidence[key]['path']);data=read(path)
        if change=='nonce':data['dispatch_nonce']='other'
        elif change=='command':data['argv'][2]='/other-plan'
        elif change=='changed_unknown':data['elapsed_seconds']=0
        else:data['identity']='changed'
        write_artifact(path,data);evidence[key]=binding(path)
    elif change=='wrong_batch_command':
        path=Path(plan['previous_batch']['path']);data=read(path);data['jobs'][0]['commands'][1][-1]='/other'
        write_artifact(path,data);plan['previous_batch']=binding(path)
    if change:
        with pytest.raises(ValueError):recovery.validate_chain(plan,evidence,identity)
    else:
        prefix,owner,adapter=recovery.validate_chain(plan,evidence,identity)
        assert set(prefix)==set(range(5)) and prefix[1][1]['status']=='unknown_after_cleanup_reconciliation'
        assert owner['step_id']=='14408.238' and adapter is first


def test_runner_continues_only_untouched_rows_and_keeps_both_unknowns(tmp_path,monkeypatch):
    from exact.repair.api import write_artifact
    from tools.repair import schema_recovery as science
    from tools.repair.expanded_corpus import binding
    from tools.repair.expanded_profile import checkpoint
    plan,evidence,identity=fixture(tmp_path,monkeypatch)
    plan_path=tmp_path/'plan2.json';write_artifact(plan_path,plan)
    original={p:p.read_bytes() for p in (tmp_path/'work').rglob('*.json')}
    calls=[];executed=[]
    def continue_rows(path,output,start,stop):
        calls.append((start,stop));refs=[]
        for i in range(start,stop):
            path=output/'rows'/f'row{i}.json'
            if not path.exists():executed.append(i);checkpoint(path,identity,row={'id':f'row{i}'},status='timeout',result=None)
            refs.append(dict(**binding(path),row_id=f'row{i}',status='timeout'))
        return dict(scheduled=stop-start,recorded=stop-start,rows=refs)
    monkeypatch.setattr(science,'run',continue_rows)
    first_report=recovery.run(plan_path,plan['output']);second_report=recovery.run(plan_path,plan['output'])
    assert first_report==second_report and calls==[(5,7)]*2 and executed==[5,6]
    assert first_report['outcomes']=={'timeout':5,'unknown_after_cleanup_reconciliation':2}
    assert first_report['recorded']==first_report['scheduled']==7
    assert all(p.read_bytes()==data for p,data in original.items())
    assert first_report['cleanup_recovery']['repair_attempt']==2
