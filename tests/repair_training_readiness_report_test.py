"""Failure-inclusive reporting rejects label reassignment and denominator loss."""
from copy import deepcopy

import pytest

from exact.repair.api import write_artifact
from tools.repair import training_readiness_report as audit
from tools.repair.development_collection import sample_assignments
from tools.repair.expanded_profile import parent_case
from tools.repair.overlap_audit import Evidence, binding


def slots_fixture(tmp_path):
    case=parent_case('overlap',2,13)
    sample=sample_assignments(case.problem,case.problem,13)
    refs={};slots=[]
    for row in sample['rows']:
        key=tuple(row['assignment'])
        duplicate=key in refs
        if not duplicate:
            p=tmp_path/f'{len(refs)}.json'
            write_artifact(p,dict(assignment=row['assignment'],feasible=None,benefit=None))
            refs[key]=binding(p)
        slots.append(dict(**row,label=refs[key],reused_exact_assignment=duplicate))
    return sample,slots


def test_exact_duplicates_keep_all_eight_slots(tmp_path):
    sample,slots=slots_fixture(tmp_path)
    unique,labels=audit.slot_labels(Evidence(),sample,slots)
    assert len(labels)==8 and len(unique)<8
    assert all(l['feasible'] is None for l in labels)


@pytest.mark.parametrize('mutation,match',[
    ('omit','denominator'),('order','Sampler'),('assignment','Sampler'),
    ('reuse','Duplicate flag'),('probability','Sampler'),('duplicate_label','Duplicate slot')])
def test_slot_corruption_is_rejected(tmp_path,mutation,match):
    sample,slots=slots_fixture(tmp_path);slots=deepcopy(slots)
    if mutation=='omit':slots.pop()
    elif mutation=='order':slots.reverse()
    elif mutation=='assignment':slots[0]['assignment']=[999]
    elif mutation=='reuse':slots[0]['reused_exact_assignment']=True
    elif mutation=='probability':slots[2]['draw_probability']=.9
    elif mutation=='duplicate_label':
        duplicate=next(s for s in slots if s['reused_exact_assignment'])
        old=Evidence().read(duplicate['label']);p=tmp_path/'duplicate.json'
        write_artifact(p,{**old,'benefit':100});duplicate['label']=binding(p)
    with pytest.raises(ValueError,match=match):audit.slot_labels(Evidence(),sample,slots)


def test_label_cannot_be_attached_to_another_assignment(tmp_path):
    sample,slots=slots_fixture(tmp_path)
    p=tmp_path/'wrong.json';write_artifact(p,dict(assignment=[999]))
    slots[0]['label']=binding(p)
    with pytest.raises(ValueError,match='Label assignment'):audit.slot_labels(Evidence(),sample,slots)


def test_changed_label_payload_rejected_even_when_assignment_matches(tmp_path):
    sample,slots=slots_fixture(tmp_path)
    write_artifact(slots[0]['label']['path'],dict(assignment=slots[0]['assignment'],benefit=1))
    with pytest.raises(ValueError,match='Changed evidence'):audit.slot_labels(Evidence(),sample,slots)


def test_generation_failures_and_duplicates_preserve_denominators():
    common=dict(parent='one',elapsed_seconds=10,label_calls=[])
    rows=[dict(common,case_id='clean',status='generation_timeout',recorded_slots=0,
               unknown_slots=8,unique_labels=0,usable_labels=0,duplicate_slots=0),
          dict(common,case_id='corrupt',status='sampled',recorded_slots=8,
               unknown_slots=3,unique_labels=2,usable_labels=1,duplicate_slots=6)]
    result=audit.summarize(rows)
    assert result['scheduled_slots']==16 and result['unknown_slots']==11
    assert result['unique_labels']==2 and result['usable_unique_labels']==1
    assert result['parent_groups']==result['parents_with_usable_labels']==1
    assert result['duplicate_slots']==6


def test_unknowns_never_qualify_fitting_or_budget_increase():
    gaps=audit.corrections(dict(statuses={'generation_timeout':32},usable_unique_labels=0,scheduled_slots=256))
    coverage=next(g for g in gaps if g['id']=='label_and_query_coverage')
    assert coverage['status']=='not_qualified_for_fitting'
    assert next(g for g in gaps if g['id']=='semantic_controls_and_query_design')['status']=='deferred_by_user'


def unavailable_case(tmp_path,monkeypatch):
    from tools.repair import development_collection as collection
    from tools.repair.prepare import case_to_dict
    from exact.repair.workers import CallResult
    case=parent_case('overlap',2,13);record=case_to_dict(case)
    for name,value in [('observable',case.problem.to_dict()),('evaluator',record),
                       ('protocol',dict(objective=dict(edit_weights={})))]:
        write_artifact(tmp_path/f'{name}.json',value)
    row=dict(case_id=case.case_id,case_hash=record['hash'],input_hash=case.problem.content_hash,
        structural_parent=case.structural_parent,family=case.family,
        observable=binding(tmp_path/'observable.json'),evaluator=binding(tmp_path/'evaluator.json'))
    plan=dict(budget=collection.BUDGET,seed=13,protocol=binding(tmp_path/'protocol.json'))
    monkeypatch.setattr(collection,'bounded_call',lambda *a,**k:CallResult('timeout',detail='test deadline'))
    ref=collection.one_case(row,tmp_path/'work','source',plan)
    def forbidden(*args,**kwargs):raise AssertionError('Audit repeated a native call')
    monkeypatch.setattr(collection,'bounded_call',forbidden)
    return ref,row,plan


def test_collection_audit_preserves_timeout_without_repeating_science(tmp_path,monkeypatch):
    ref,row,plan=unavailable_case(tmp_path,monkeypatch)
    actual=audit.collection_case(Evidence(),ref,row,'source',plan)
    assert actual['unknown_slots']==8 and actual['usable_labels']==0
    assert actual['label_calls']==[] and actual['generation_call']['status']=='timeout'


@pytest.mark.parametrize('mutation,match', [('denominator','denominator'),('promote','generation'),('split','split')])
def test_checkpoint_with_consistent_hash_cannot_hide_scientific_corruption(tmp_path,monkeypatch,mutation,match):
    from exact.repair.records import canonical_hash
    ref,row,plan=unavailable_case(tmp_path,monkeypatch)
    value=Evidence().read(ref)
    if mutation=='denominator':value['result']['unknown_slots']=7
    elif mutation=='promote':value['status']='sampled'
    else:value['split']='train'
    value['content_hash']=canonical_hash({k:v for k,v in value.items() if k!='content_hash'})
    write_artifact(ref['path'],value)
    with pytest.raises(ValueError,match=match):audit.collection_case(Evidence(),binding(ref['path']),row,'source',plan)


def test_persisted_native_obligations_use_verdict_and_expected(tmp_path,monkeypatch):
    from pathlib import Path
    from exact.repair.records import canonical_hash
    ref,row,plan=unavailable_case(tmp_path,monkeypatch)
    receipt=Evidence().read(ref);native=Path(ref['path']).parent/'native'/'check-0.json'
    write_artifact(native,dict(logical_status='UNKNOWN',
        support=dict(reasoner='hermit',backend='python'),
        obligations=[dict(kind='required_entailment',verdict=False,expected=True,complete=True),
                     dict(kind='consistency',verdict=None,expected=True,complete=False)]))
    receipt['payloads'].append(binding(native))
    receipt['content_hash']=canonical_hash({k:v for k,v in receipt.items() if k!='content_hash'})
    write_artifact(ref['path'],receipt)
    result=audit.collection_case(Evidence(),binding(ref['path']),row,'source',plan)
    assert result['native_trace_summary']['obligations']=={
        "('required_entailment', False, True)":1,"('consistency', None, False)":1}
    assert result['unknown_slots']==8 and result['usable_labels']==0
