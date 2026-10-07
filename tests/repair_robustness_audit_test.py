"""Failure-denominator, ancestry, recovery and omission evidence boundaries."""
import copy
import json
from pathlib import Path

import pytest

from exact.repair.records import canonical_hash
from tools.repair import robustness_audit as audit
from tools.repair import schema_cleanup_recovery as cleanup
from tools.repair import schema_omission_recovery as omission
from tools.repair import schema_removal_recovery as removal


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return audit.binding(path)


def checkpoint(path, value):
    return save(path, dict(value, content_hash=canonical_hash(value)))


def fixture(tmp_path):
    cases = [dict(case_id=str(i),group_id='parent',family='family',family_exposure='seen_family',
        control='corrupted' if i==0 else 'coherent',condition='baseline',base_case_id='base'+str(i),
        intervention=dict(applicable=False)) for i in range(2)]
    rows = [dict(id=str(i),case_index=i,arm_id='model') for i in range(2)]
    schedule = dict(schema='exact-repair/fresh-evaluation/v1',rows=rows,cases=cases,arms=[dict(id='model')])
    schedule_ref = save(tmp_path/'schedule.json',schedule)
    admissions = [dict(row_id=r['id']) for r in rows]
    plan = dict(schedule=schedule_ref,preflight=save(tmp_path/'admission.json',dict(rows=admissions)),rows=admissions)
    plan_ref = save(tmp_path/'plan.json',plan)
    batch = dict(code=str(tmp_path),commit='original',frozen_files={
        str(tmp_path/'tools/repair/schema_recovery.py'):'schema-source',
        str(tmp_path/'tools/repair/fresh_evaluation.py'):'fresh-source'})
    identity = audit.scientific_identity(audit.Evidence(),plan_ref,batch,{})
    refs = []
    for row,entry in zip(rows,admissions):
        value = dict(identity=canonical_hash((identity,row,entry,entry)),row=row,status='timeout',
            cleanup_complete=True,elapsed_seconds=300,resources={},payloads=[],result=None)
        ref = checkpoint(tmp_path/'rows'/(row['id']+'.json'),value)
        refs.append(dict(ref,row_id=row['id'],status=value['status']))
    report = dict(identity=identity,schedule=schedule_ref,plan=plan_ref,status='complete',
                  scheduled=2,recorded=2,rows=refs,outcomes={'timeout':2})
    item = dict(report=checkpoint(tmp_path/'report.json',report),row_slice=[0,2],
                batch=save(tmp_path/'batch.json',batch),run={'id':'schema-001'})
    return schedule,item,report


def test_unknowns_and_noops_keep_full_denominator(tmp_path):
    schedule,item,_ = fixture(tmp_path)
    rows = audit.audit_rows(audit.AuditEvidence(),schedule,[(item,{'runtime':{}})])
    assert len(rows)==2
    assert all(r['semantic_benefit'] is None and r['logical_status']=='UNKNOWN' and
        r['source_regime']=='original' and r['intervention_scope']['final_exclusion_verified'] is None
        and r['intervention_scope']['applicable'] is False for r in rows)


@pytest.mark.parametrize('change',['missing','duplicate','runtime','guard','status','identity','plan_order'])
def test_rejects_row_ownership_denominator_and_source_changes(tmp_path,change):
    schedule,item,report=fixture(tmp_path); source={'runtime':{}}
    if change=='missing':report['rows'].pop()
    elif change=='duplicate':report['rows'][1]=report['rows'][0]
    elif change=='runtime':source['runtime']={'changed':True}
    elif change=='guard':save(tmp_path/'inflight/0.json',{})
    elif change=='status':report['rows'][0]['status']='complete'
    elif change=='identity':report['identity']='other'
    else:
        schedule['rows'][0]['case_index']=1
    item['report']=checkpoint(tmp_path/'report.json',report)
    with pytest.raises(ValueError):audit.audit_rows(audit.AuditEvidence(),schedule,[(item,source)])


@pytest.mark.parametrize('kind',[cleanup,omission,removal])
def test_reconciled_unknown_exact_provenance_and_no_promotion(tmp_path,kind):
    previous=dict(identity='original',row={'id':'stopped'},status='error',detail=kind.CAUSE,
        cleanup_complete=True,result=None,payloads=[],elapsed_seconds=219,resources={},
        remaining_budget={'wall_seconds':280},recovery_action='compatible_remaining_budget_execution')
    if kind == removal:
        result = save(tmp_path/'generation-error.json',dict(status='generation_error',detail=kind.CAUSE,
            row_id='stopped',logical_status='UNKNOWN',semantic_benefit=None))
        previous.update(status='complete',result=result,payloads=[result])
    original=checkpoint(tmp_path/'original.json',previous)
    guard=dict(identity='original',row=previous['row'],remaining_budget=previous['remaining_budget'])
    proof=save(tmp_path/'proof.json',dict(guard=checkpoint(tmp_path/'guard.json',guard)))
    plan=dict(original_step='14408.223',schema_identity='schema',evidence=proof)
    plan_ref=save(tmp_path/'recovery-plan.json',plan)
    ownership=dict(step_id='14408.223',surviving_processes=0,cgroup_absent=True,signals_sent=0)
    args=[previous]+([guard] if kind==cleanup else [])+[original,proof,ownership,'14408.223']
    saved=kind.reconciled_unknown(*args);saved['identity']=canonical_hash((plan_ref,'schema'))
    ref=checkpoint(tmp_path/'unknown.json',saved)
    audit.validate_unknown(audit.Evidence(),saved,ref,[(plan_ref,plan)])
    changed=copy.deepcopy(saved);changed['result']={'assignment':[0]}
    with pytest.raises(ValueError,match='changed or promoted'):
        audit.validate_unknown(audit.Evidence(),changed,ref,[(plan_ref,plan)])
    with pytest.raises(ValueError,match='exact recovery plan'):
        audit.validate_unknown(audit.Evidence(),saved,ref,[])


def test_distinct_cleanup_descendants_cannot_share_evidence(tmp_path):
    _,item,report=fixture(tmp_path)
    proof=save(tmp_path/'proof.json',{})
    previous=dict(schema_plan=report['plan'],evidence=proof,prior_costs_reset=False)
    previous_ref=save(tmp_path/'previous.json',previous)
    current=dict(previous,previous_cleanup_plan=previous_ref)
    current_ref=save(tmp_path/'current.json',current)
    report['cleanup_recovery']=dict(plan=current_ref,evidence=proof)
    plans=audit.recovery_plans(audit.Evidence(),report)
    with pytest.raises(ValueError,match='exact recovery plan'):
        audit.validate_unknown(audit.Evidence(),dict(status='unknown_after_cleanup_reconciliation',
            cleanup_reconciliation=proof),{},plans)


def test_cached_evidence_rejects_conflicting_digest(tmp_path):
    ref=save(tmp_path/'record.json',{'x':1});evidence=audit.AuditEvidence()
    evidence.read(ref)
    with pytest.raises(ValueError,match='Conflicting'):
        evidence.read(dict(ref,sha256='changed'))


def test_published_pool_must_match_saved_exclusion_audit(tmp_path,monkeypatch):
    from tools.repair import robustness
    monkeypatch.setattr(audit,'read_record',lambda x:x)
    monkeypatch.setattr(robustness,'audit_final_pool',lambda problem,item:dict(final_exclusion_verified=True))
    pool=save(tmp_path/'pool.json',dict(input={}))
    saved=dict(result=save(tmp_path/'result.json',dict(intervention_audit={'final_exclusion_verified':False})))
    with pytest.raises(ValueError,match='final exclusion audit differs'):
        audit.intervention_scope(audit.Evidence(),saved,dict(condition='final_pool_050'),dict(pool=pool))


def schedule_fixture(tmp_path):
    protocol=save(tmp_path/'protocol.json',dict(resources=dict(case_wall_seconds=300,
        case_cpu_seconds=600,case_rss_mb=8192,generation_seconds=60)))
    arms=[dict(id=str(i),kind='learned' if i<6 else 'control',protocol=protocol) for i in range(9)]
    bases=[dict(case_id=str(i),group_id=str(i//2),family='family',family_exposure='seen_family',
        control='corrupted' if i%2==0 else 'coherent',split='fresh_evaluation',
        structural_parent='parent'+str(i//2),source_parent=str(i//2),fingerprints=[str(i//2)],
        evaluator={},status='unavailable') for i in range(64)]
    cases=[]
    for base in bases:
        cases.extend(dict(base,base_case_id=base['case_id'],condition=c,
                          case_id=base['case_id']+':robustness:'+c) for c in audit.CONDITIONS)
    rows=[]
    for arm in arms:
        for i,case in enumerate(cases):
            row=dict(arm_id=arm['id'],case_index=i,case_id=case['case_id'],seconds=300,cpu_seconds=600,memory_mb=8192)
            row['id']=canonical_hash(row);rows.append(row)
    value=dict(schema='exact-repair/fresh-evaluation/v1',cases=cases,arms=arms,rows=rows,
        base_schedule=save(tmp_path/'base.json',dict(cases=bases,arms=arms)),
        conditions=[[c] for c in audit.CONDITIONS],profile_timeouts_retained=4,
        test_feedback_for_selection=False,warm_start=False)
    return value


def test_unavailable_variants_preserve_parent_pairing_and_denominators(tmp_path):
    value=schedule_fixture(tmp_path)
    checked=audit.validate_schedule(audit.Evidence(),dict(schedule=save(tmp_path/'schedule.json',value)))
    assert len(checked['rows'])==2880
    assert len({c['group_id'] for c in checked['cases']})==32


@pytest.mark.parametrize('change',['renamed_parent','split','fingerprint','missing_row','missing_variant','budget'])
def test_rejects_changed_split_ancestry_and_scientific_budgets(tmp_path,change):
    value=schedule_fixture(tmp_path)
    if change=='renamed_parent':value['cases'][1]['group_id']='renamed_independent'
    elif change=='split':value['cases'][1]['split']='train'
    elif change=='fingerprint':value['cases'][1]['fingerprints']=['different']
    elif change=='missing_row':value['rows'].pop()
    elif change=='missing_variant':value['cases'].pop()
    else:value['rows'][0]['seconds']=600
    with pytest.raises(ValueError):
        audit.validate_schedule(audit.Evidence(),dict(schedule=save(tmp_path/'schedule.json',value)))


def test_cleanup_unknown_keeps_pool_evidence_without_promoting_semantics(tmp_path,monkeypatch):
    from tools.repair import robustness
    monkeypatch.setattr(audit,'read_record',lambda x:x)
    monkeypatch.setattr(robustness,'audit_final_pool',lambda problem,item:dict(final_exclusion_verified=True))
    pool=save(tmp_path/'pool.json',dict(input={}))
    scope=audit.intervention_scope(audit.Evidence(),dict(result=None,status='unknown_after_cleanup_reconciliation'),
        dict(condition='retrieval_symbols_025'),dict(pool=pool))
    assert scope['final_exclusion_verified'] is True
    assert scope['outer_exclusion_receipt']=='unavailable_retained_unknown'


def reuse_fixture(tmp_path):
    original=dict(identity='prior',row={'id':'row'},status='timeout',resources={'cpu_seconds':4},
                  elapsed_seconds=5,cleanup_complete=True,result=None,payloads=[])
    prior=checkpoint(tmp_path/'original/rows/row.json',original)
    batch=save(tmp_path/'original-batch.json',dict(commit='original-science'))
    completion=save(tmp_path/'completion.json',dict(step_id='14408.12',dispatch_nonce='original-nonce',
        work=str(tmp_path/'original'),batch=batch['path']))
    outputs=save(tmp_path/'outputs.json',{'rows/row.json':prior['sha256']})
    saved=dict(original,identity='schema-view',reuse_of=prior,additional_elapsed_seconds=0,
        reuse_source=dict(completion=completion,outputs=outputs,step_id='14408.12',nonce='original-nonce'),
        recovery_action='reused_original_unchanged')
    return saved


def test_reused_result_reports_actual_scientific_source(tmp_path):
    saved=reuse_fixture(tmp_path)
    result=audit.execution_provenance(audit.Evidence(),saved,'new-schema-wrapper')
    assert result['source_commit']=='original-science'
    assert result['additional_elapsed_seconds']==0


@pytest.mark.parametrize('change',['nonce','cost','outcome','payload'])
def test_reuse_cannot_change_owner_effort_or_science(tmp_path,change):
    saved=reuse_fixture(tmp_path)
    if change=='nonce':saved['reuse_source']['nonce']='other'
    elif change=='cost':saved['elapsed_seconds']=0
    elif change=='outcome':saved['status']='complete'
    else:saved['result']={'assignment':[0]}
    with pytest.raises(ValueError):audit.execution_provenance(audit.Evidence(),saved,'wrapper')
