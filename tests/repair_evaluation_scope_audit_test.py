"""Receipt integrity tests; fabricated native outcomes are reader fixtures only."""
from dataclasses import asdict
from pathlib import Path
import json

import pytest

from exact.repair.api import write_artifact
from exact.repair.learning import OwlTeacherOracle, evaluate_teacher
from exact.repair.owl import CheckReport, ObligationResult, SupportReport, snapshot_from_axioms
from exact.repair.records import canonical_hash
from tools.repair import evaluation_scope_audit as audit
from tools.repair.expanded_profile import checkpoint
from tools.repair.historical_regression import binding
from tests.repair_evaluation_intended_test import case_and_row


def native_fixture(path, case, sequence):
    snapshot = snapshot_from_axioms(case.intended_theory)
    if sequence == 0:
        expected = audit.expected_queries(case.problem.policy.monitored_classes,
            case.problem.policy.required, case.problem.policy.prohibited, case.intended_active)
    else:
        expected = audit.expected_queries(required=[p.axiom for p in case.probes],
            activated={c for p in case.probes if p.desired for c in p.conditions()})
    obligations = tuple(ObligationResult(kind, key, truth, truth, True) for kind, key, truth in expected)
    support = SupportReport('hermit', 'fixture-only', 'python', 'fixture-only', True, True, (), (),
        query_support=tuple((o.kind, o.query_id, o.complete) for o in obligations))
    report = CheckReport('VERIFIED_FEASIBLE', 'complete_supported_fragment', snapshot.logical_fingerprint.hex,
                         obligations, support)
    write_artifact(path/f'native/check-{sequence}.json', asdict(report))
    write_artifact(path/f'native/check-{sequence}.started.json', dict(reasoner='hermit', backend='python'))
    (path/f'native/check-{sequence}.obligations.jsonl').write_text(''.join(
        json.dumps(dict(obligation=asdict(o), support=asdict(support)))+'\n' for o in obligations))
    return report


def fixture(tmp_path, outer='complete'):
    row, _, case = case_and_row(tmp_path/'input')
    row['family_exposure'] = 'fixture'
    path = tmp_path/'row';path.mkdir()
    result = None
    if outer == 'complete':
        policy = native_fixture(path, case, 0)
        queries = native_fixture(path, case, 1)
        semantic = evaluate_teacher(OwlTeacherOracle(audit.RecordedOracle(queries),
            snapshot_from_axioms(case.intended_theory), case.probes), case.probes)
        result = dict(logical_status=policy.logical_status, query_complete=semantic.complete,
            original_guard_passed=semantic.complete, intended_target_satisfied=True,
            semantic=asdict(semantic), query_denominator=len(case.probes), query_scope=audit.query_scope(case))
        write_artifact(path/'intended-policy.json', asdict(policy))
        write_artifact(path/'intended-queries.json', result)
    call = dict(status=outer, cleanup_complete=True, budget=audit.BUDGET, detail='', resources={})
    write_artifact(path/'call.json', call)
    saved = dict(case_id=case.case_id, parent=case.structural_parent, split='test', family=case.family,
        control=case.control, query_scope=audit.query_scope(case), query_denominator=len(case.probes),
        budget=audit.BUDGET, cleanup_complete=True, supervision_admitted=False, original_results_replaced=False,
        outer_status=outer, status='qualified' if result else outer, detail='', resources={}, elapsed_seconds=1,
        result=result)
    return path, row, case, saved


def save(path, row, saved):
    payloads = [binding(p) for p in sorted(path.rglob('*')) if p.is_file() and p.name != 'completion.json']
    checkpoint(path/'completion.json', canonical_hash(('identity', row)), **saved, payloads=payloads)
    return binding(path/'completion.json')


def test_reconstructs_policy_and_typed_semantics_without_reasoner(tmp_path, monkeypatch):
    def forbidden(*a, **kw):
        raise AssertionError('Reader must not open reasoner')
    monkeypatch.setattr('exact.repair.owl.OwlVerifier._open', forbidden)
    path,row,case,saved = fixture(tmp_path)
    result = audit.audit_row(audit.Evidence(), save(path,row,saved), row,case,'identity')
    assert result['status'] == 'qualified' and len(result['native_reports']) == 2
    assert result['query_denominator'] == len(case.probes)
    assert not result['original_arm_results_replaced']


@pytest.mark.parametrize('outer', ['timeout', 'error', 'memory_limit'])
def test_unknown_native_outcomes_keep_case_query_denominator(tmp_path, outer):
    path,row,case,saved = fixture(tmp_path, outer)
    result = audit.audit_row(audit.Evidence(), save(path,row,saved), row,case,'identity')
    assert result['status'] == outer and result['query_denominator'] == len(case.probes)
    assert result['intended_target_satisfied'] is None and not result['native_reports']


@pytest.mark.parametrize('mutation', ['theory','obligation','backend','progress','support','scalar','status',
    'budget','scope','identity','call','cleanup','admission'])
def test_rehashed_but_inconsistent_receipts_are_rejected(tmp_path, mutation):
    path,row,case,saved = fixture(tmp_path)
    native = path/'native/check-1.json';value=json.loads(native.read_text())
    if mutation == 'theory':value['theory_hash']='wrong'
    elif mutation == 'obligation':value['obligations'][1]['query_id']='foreign'
    elif mutation == 'backend':value['support']['backend']='different'
    elif mutation == 'support':value['support']['query_support']=[]
    elif mutation == 'progress':(path/'native/check-1.obligations.jsonl').write_text('')
    elif mutation == 'scalar':saved['result']['semantic']['benefit']=100
    elif mutation == 'status':saved['status']='native_unknown_or_infeasible'
    elif mutation == 'budget':saved['budget']={**audit.BUDGET,'wall_seconds':601}
    elif mutation == 'scope':saved['query_denominator']+=1
    elif mutation == 'identity':saved['parent']='renamed'
    elif mutation == 'call':write_artifact(path/'call.json',dict(status='timeout',cleanup_complete=True,budget=audit.BUDGET,detail='',resources={}))
    elif mutation == 'cleanup':saved['cleanup_complete']=False
    elif mutation == 'admission':saved['supervision_admitted']=True
    write_artifact(native,value)
    with pytest.raises(ValueError):
        audit.audit_row(audit.Evidence(),save(path,row,saved),row,case,'identity')


def test_mutated_payload_hash_rejected(tmp_path):
    path,row,case,saved=fixture(tmp_path)
    ref=save(path,row,saved)
    (path/'native/check-0.obligations.jsonl').write_text('')
    with pytest.raises(ValueError,match='Changed evidence'):
        audit.audit_row(audit.Evidence(),ref,row,case,'identity')


def test_control_review_cannot_declare_qualification(tmp_path):
    ref=tmp_path/'review.json'
    write_artifact(ref,dict(schema='exact-repair/fresh-control-review/v1',
        test_outcomes_used_for_design=False,evaluator_inputs_allowed_at_inference=False,
        implementation_qualified=True))
    with pytest.raises(ValueError,match='boundary'):
        audit.control_review(audit.Evidence(),dict(control_review=binding(ref)),{})


@pytest.mark.parametrize('verdict,complete,status,guard,target', [
    (None,False,'native_unknown_or_infeasible',False,None),
    (False,True,'intended_target_mismatch',True,False)])
def test_saved_partial_or_mismatched_query_is_not_upgraded(tmp_path,verdict,complete,status,guard,target):
    path,row,case,saved=fixture(tmp_path)
    p=path/'native/check-1.json';value=json.loads(p.read_text())
    o=next(o for o in value['obligations'] if o['kind']=='required_entailment')
    o.update(verdict=verdict,complete=complete,reason='fixture unknown' if not complete else None)
    value['support']['query_support']=[[o['kind'],o['query_id'],o['complete']] for o in value['obligations']]
    value['logical_status']='UNKNOWN' if not complete else 'VERIFIED_INFEASIBLE'
    value['verification_scope']='partial_detection' if not complete else 'complete_supported_fragment'
    write_artifact(p,value)
    (path/'native/check-1.obligations.jsonl').write_text(''.join(json.dumps(dict(obligation=o,support=value['support']))+'\n' for o in value['obligations'] if o['complete']))
    report=CheckReport(value['logical_status'],value['verification_scope'],value['theory_hash'],
        tuple(ObligationResult(**o) for o in value['obligations']),SupportReport(**value['support']))
    semantic=evaluate_teacher(OwlTeacherOracle(audit.RecordedOracle(report),snapshot_from_axioms(case.intended_theory),case.probes),case.probes)
    saved['result'].update(semantic=asdict(semantic),query_complete=semantic.complete,
        original_guard_passed=guard,intended_target_satisfied=target)
    saved['status']=status
    write_artifact(path/'intended-queries.json',saved['result'])
    result=audit.audit_row(audit.Evidence(),save(path,row,saved),row,case,'identity')
    assert (result['status'],result['original_guard_passed'],result['intended_target_satisfied'])==(status,guard,target)
    assert result['query_denominator']==len(case.probes)
