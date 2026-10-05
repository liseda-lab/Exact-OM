"""Lossless transport and a strict no-replay boundary for saved development labels."""
from dataclasses import asdict, replace
import pickle
from types import SimpleNamespace

import pytest

from exact.repair.api import write_artifact
from exact.repair.workers import bounded_call, CallResult
from exact.repair.records import canonical_hash
from tools.repair import development_collection as collection
from tools.repair import development_collection_recovery as recovery
from tools.repair.historical_regression import binding, verify_binding
from tools.repair.prepare import case_to_dict
from tools.repair.expanded_profile import parent_case


def large_label_worker(directory):
    # Exercise the production label publisher through a real bounded worker,
    # with synthetic support data exceeding the real 16 MiB frame ceiling.
    from tools.repair import train
    case = parent_case('overlap', 2, 13)
    case = replace(case, split='development')
    original = train._assignment_label
    from exact.repair.learning import RepairLabel, SupportTarget
    support = SupportTarget((0,)*len(case.problem.objects), 'fixture', 'class_satisfiability', None,
        'theory', 'policy', 'backend', tuple((o.object_id,'candidate') for o in case.problem.objects), ('ab'*(8*1024*1024+1),), ())
    label = RepairLabel(support.assignment, None, None, 0.0, support_targets=(support,))
    try:
        train._assignment_label = lambda *a, **k: label
        return collection.label_assignment(case_to_dict(case), label.assignment, (), directory)
    finally:
        train._assignment_label = original


def test_production_label_publisher_transports_only_hash_receipt(tmp_path):
    result = bounded_call(large_label_worker, str(tmp_path), timeout=30, memory_mb=2048)
    assert result.status == 'complete', result.detail
    assert result.cleanup_complete and len(pickle.dumps(result)) < 2048
    label = verify_binding(result.value)
    assert (tmp_path/'label.json').stat().st_size > 16*1024*1024
    assert label['feasible'] is None and label['benefit'] is None
    assert len(label['support_targets'][0]['asserted_axioms'][0]) > 16*1024*1024
    with (tmp_path/'label.json').open('a') as f:
        f.write(' ')
    with pytest.raises(ValueError, match='changed'):
        verify_binding(result.value)


def fixture(tmp_path):
    pool = tmp_path/'pool.json'; label = tmp_path/'label.json'; call = tmp_path/'call.json'
    write_artifact(pool, dict(input='original'))
    write_artifact(label, dict(assignment=[0], feasible=True, benefit=1.0))
    write_artifact(call, dict(resource_usage={'wall_seconds': 274.9}, status='error', detail=recovery.CAUSE))
    evidence = dict(pool=binding(pool), label=binding(label), label_call=binding(call), generation_call=binding(call))
    science = SimpleNamespace(generate=lambda: None, label_assignment=lambda: None)
    executed = []
    def execute(fn, *args, **limits):
        executed.append((fn,args,limits))
        return CallResult('timeout')
    return recovery.ReuseCalls(science, evidence, {'case':'same'}, [('cost',1)], [[1]], execute), science, executed


def test_reuses_exact_generation_and_label_but_only_calls_untouched_assignment(tmp_path):
    router, science, executed = fixture(tmp_path)
    output = tmp_path/'output'; output.mkdir()
    result = router(science.generate, {}, {}, str(output))
    assert verify_binding(result.value)=={'input':'original'} and not executed
    label = router(science.label_assignment, {'case':'same'}, (0,), [('cost',1)], str(output/'labels'))
    assert verify_binding(label.value)['benefit']==1.0 and not executed
    limits=dict(timeout=300,cpu_seconds=600,memory_mb=8192)
    router(science.label_assignment, {'case':'same'}, (1,), [('cost',1)], str(output/'new'), **limits)
    assert len(executed)==1 and executed[0][0] is recovery.label_reference
    assert executed[0][2]==limits
    with pytest.raises(ValueError, match='repeated'):
        router(science.label_assignment, {'case':'same'}, (1,), [('cost',1)], str(output/'new'), **limits)


@pytest.mark.parametrize('change', ['record','assignment','budget','profile'])
def test_recovery_rejects_changed_identity_or_spent_budget(tmp_path, change):
    router, science, executed = fixture(tmp_path)
    record, assignment, profile = {'case':'same'}, (1,), [('cost',1)]
    limits=dict(timeout=300,cpu_seconds=600,memory_mb=8192)
    if change=='record': record={'case':'other'}
    if change=='assignment': assignment=(2,)
    if change=='profile': profile=[('cost',2)]
    if change=='budget': limits['timeout']=301
    with pytest.raises(ValueError):
        router(science.label_assignment, record, assignment, profile, str(tmp_path/'new'), **limits)
    assert not executed


def native_value(snapshot):
    from exact.repair.owl import _query_id
    return dict(theory_hash=snapshot.logical_fingerprint.hex, logical_status='VERIFIED_FEASIBLE',
        verification_scope='complete_supported_fragment',
        obligations=[dict(kind='consistency', query_id=_query_id(None), expected=True,
            verdict=True, complete=True, reason=None, class_iri=None)],
        support=dict(reasoner='hermit', backend='python', input_supported=True,
            query_support=[['consistency',_query_id(None),True]]))


@pytest.mark.parametrize('change', ['theory','query','backend','support','aggregate'])
def test_saved_native_evidence_rejects_mismatched_scope(change):
    from exact.repair.owl import snapshot_from_axioms
    snapshot=snapshot_from_axioms(())
    value=native_value(snapshot)
    if change=='theory':value['theory_hash']='wrong'
    if change=='query':value['obligations'][0]['expected']=False
    if change=='backend':value['support']['backend']='other'
    if change=='support':value['support']['query_support']=[]
    if change=='aggregate':value['logical_status']='UNKNOWN'
    with pytest.raises(ValueError):
        recovery.checked_native(value,snapshot)


def large_legacy_worker(directory):
    original = collection.label_assignment
    def legacy(*args):
        payload = {'payload': 'z'*(17*1024*1024)}
        write_artifact(collection.Path(args[-1])/'label.json', payload)
        return payload
    try:
        collection.label_assignment = legacy
        return recovery.label_reference({}, (), (), directory)
    finally:
        collection.label_assignment = original


def test_adapter_transports_oversized_legacy_return_without_truncation(tmp_path):
    result=bounded_call(large_legacy_worker,str(tmp_path),timeout=30,memory_mb=2048)
    assert result.status=='complete',result.detail
    assert len(pickle.dumps(result))<2048
    assert len(verify_binding(result.value)['payload'])==17*1024*1024


def test_full_case_recovery_keeps_duplicate_unknown_slots_and_resumes_without_calls(tmp_path, monkeypatch):
    import json
    from exact.repair.learning import RepairLabel
    from tools.repair.prepare import read_label_cache, case_from_dict
    case=replace(parent_case('overlap',2,13),split='development')
    write_artifact(tmp_path/'observable.json',case.problem.to_dict())
    record=case_to_dict(case);write_artifact(tmp_path/'evaluator.json',record)
    write_artifact(tmp_path/'protocol.json',dict(objective=dict(edit_weights={})))
    row=dict(case_id=case.case_id,case_hash=record['hash'],input_hash=case.problem.content_hash,
        structural_parent=case.structural_parent,family=case.family,
        observable=binding(tmp_path/'observable.json'),evaluator=binding(tmp_path/'evaluator.json'))
    plan=dict(budget=collection.BUDGET,seed=13,protocol=binding(tmp_path/'protocol.json'))
    write_artifact(tmp_path/'pool.json',dict(input=case.problem.to_dict(),proposal_reports=[]))
    sample=collection.sample_assignments(case.problem,case.problem,13)
    first=sample['rows'][0]['assignment']
    write_artifact(tmp_path/'label.json',asdict(RepairLabel(tuple(first),None,None,0)))
    write_artifact(tmp_path/'call.json',dict(status='error',detail=recovery.CAUSE,resource_usage={}))
    evidence=dict(pool=binding(tmp_path/'pool.json'),generation_call=binding(tmp_path/'call.json'),
        label=binding(tmp_path/'label.json'),label_call=binding(tmp_path/'call.json'))
    untouched={tuple(s['assignment']) for s in sample['rows']}-{tuple(first)}
    calls=[]
    def execute(fn,*args,**limits):
        calls.append(tuple(args[1]));return CallResult('timeout',detail='deadline')
    router=recovery.ReuseCalls(collection,evidence,json.loads(json.dumps(record)),[],untouched,execute)
    monkeypatch.setattr(collection,'bounded_call',router)
    ref=collection.one_case(row,tmp_path/'revised','identity',plan)
    saved=verify_binding(ref);result=saved['result']
    assert result['recorded_slots']==result['unknown_slots']==8
    assert set(calls)==untouched and len(calls)==len(untouched)
    assert result['slots'][0]['label']==result['slots'][4]['label']
    cache=read_label_cache(result['cache'],collection.Path(result['cache']['path']).parent,case)
    assert not cache.complete and not any(l.feasible is not None for l in cache.labels)
    assert collection.one_case(row,tmp_path/'revised','identity',plan)==ref
    assert len(calls)==len(untouched)
    assert verify_binding(evidence['label'])['feasible'] is None
