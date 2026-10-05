"""Failure-inclusive development evidence must resist scope and ownership drift."""
import copy
import json
from pathlib import Path

import pytest

from exact.repair.records import canonical_hash
from tools.repair import common_inventory_audit as audit
from tools.repair.common_inventory_recovery import CAUSE, reconciled_unknown


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return audit.binding(path)


def checkpoint(path, value):
    return save(path, dict(value, content_hash=canonical_hash(value)))


def fixture(tmp_path):
    rows = [dict(id=str(i), case_index=0, case_id='case', arm_id='model') for i in range(2)]
    case = dict(case_id='case', group_id='parent', structural_parent='parent',
                family='family', family_exposure='seen_family', control='corrupted')
    schedule = dict(rows=rows, cases=[case], arms=[dict(id='model')])
    schedule_ref = save(tmp_path/'schedule.json', schedule)
    batch = dict(code=str(tmp_path), frozen_files={
        str(tmp_path/'tools/repair/common_inventory.py'): 'decode',
        str(tmp_path/'tools/repair/fresh_evaluation.py'): 'fresh'})
    identity = audit.scientific_identity(schedule_ref, batch, {})
    refs = []
    for row in rows:
        result = save(tmp_path/'payloads'/row['id']/'result.json', dict(row_id=row['id'],
            case_id='case', schedule_hash=canonical_hash(schedule), status='generation_timeout', study_kind='common_inventory_diagnostic',
            common_inventory_regret=None, regret_status='unavailable_no_complete_external_teacher',
            generation_statuses=[]))
        value = dict(identity=canonical_hash((identity, row)), row=row,
            status='complete', cleanup_complete=True, elapsed_seconds=63,
            resources={}, payloads=[result], result=result)
        ref = checkpoint(tmp_path/'rows'/(row['id']+'.json'), value)
        refs.append(dict(ref, row_id=row['id'], status='generation_timeout', study_kind='common_inventory_diagnostic',
            common_inventory_regret=None, regret_status='unavailable_no_complete_external_teacher',
            generation_statuses=[]))
    report = dict(identity=identity, status='complete', schedule=schedule_ref,
        runtime={}, slice=[0, 2], scheduled=2, recorded=2, rows=refs,
        outcomes={'generation_timeout': 2})
    item = dict(report=checkpoint(tmp_path/'report.json', report), row_slice=[0, 2],
                batch=save(tmp_path/'batch.json', batch), run=dict(id='job-001'))
    return schedule, item, report


def test_process_completion_preserves_timeout_denominator(tmp_path):
    schedule, item, _ = fixture(tmp_path)
    rows = audit.audit_rows(audit.Evidence(), schedule, [(item, {'runtime': {}})])
    assert len(rows) == 2
    assert all(r['process_status'] == 'complete' and r['scientific_status'] == 'generation_timeout'
               and r['logical_status'] == 'UNKNOWN' and not r['semantic_claim_qualified']
               and r['value_diagnostic']['status'] == 'unavailable' for r in rows)


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'runtime', 'identity', 'status', 'guard', 'payload'])
def test_reject_changed_denominator_provenance_or_results(tmp_path, change):
    schedule, item, report = fixture(tmp_path)
    source = {'runtime': {}}
    if change == 'missing': report['rows'].pop()
    elif change == 'duplicate': report['rows'][1] = report['rows'][0]
    elif change == 'runtime': source['runtime'] = {'other': True}
    elif change == 'identity': report['identity'] = 'other-source'
    elif change == 'status': report['rows'][0]['status'] = 'evaluated'
    elif change == 'guard': save(tmp_path/'inflight/0.json', {})
    elif change == 'payload': save(tmp_path/'payloads/0/result.json', {'status': 'evaluated'})
    item['report'] = checkpoint(tmp_path/'report.json', report)
    with pytest.raises(ValueError):
        audit.audit_rows(audit.Evidence(), schedule, [(item, source)])


def unknown_fixture(tmp_path):
    row = dict(id='failed')
    identity = canonical_hash(('science', row))
    original = checkpoint(tmp_path/'original.json', dict(identity=identity, row=row,
        status='error', detail=CAUSE, cleanup_complete=True, result=None, payloads=[],
        elapsed_seconds=200, resources={'cpu_seconds': 30}))
    guard = checkpoint(tmp_path/'guard.json', dict(identity=identity, row=row))
    proof = save(tmp_path/'proof.json', dict(guard=guard))
    plan = save(tmp_path/'plan.json', dict(evidence=proof, original_step='14408.369'))
    owner = dict(step_id='14408.369', surviving_processes=0, cgroup_absent=True, signals_sent=0)
    saved = reconciled_unknown(json.loads(Path(original['path']).read_text()),
        json.loads(Path(guard['path']).read_text()), original, proof, owner, '14408.369')
    saved['identity'] = canonical_hash((plan, 'science'))
    ref = checkpoint(tmp_path/'unknown.json', saved)
    return ref, saved, dict(plan=plan, evidence=proof)


def test_unknown_keeps_error_costs_and_zero_replay(tmp_path):
    ref, saved, recovery = unknown_fixture(tmp_path)
    audit.unknown_row(audit.Evidence(), ref, saved, recovery, 'science')
    assert saved['result'] is None and saved['elapsed_seconds'] == 200
    assert saved['additional_scientific_seconds'] == 0 and saved['original_detail'] == CAUSE


@pytest.mark.parametrize('change', ['promote', 'cost', 'replay', 'owner', 'identity'])
def test_unknown_rejects_promotion_replay_or_ancestry_change(tmp_path, change):
    ref, saved, recovery = unknown_fixture(tmp_path)
    if change == 'promote': saved['result'] = {'assignment': [0]}
    elif change == 'cost': saved['elapsed_seconds'] = 0
    elif change == 'replay': saved['additional_scientific_seconds'] = 20
    elif change == 'owner': saved['ownership']['surviving_processes'] = 1
    elif change == 'identity': saved['identity'] = 'other'
    ref = checkpoint(Path(ref['path']), saved)
    with pytest.raises(ValueError):
        audit.unknown_row(audit.Evidence(), ref, saved, recovery, 'science')


def manifest_fixture(tmp_path):
    row = dict(id='failed')
    guard = checkpoint(tmp_path/'guard.json', dict(identity=canonical_hash(('science', row)), row=row))
    result = save(tmp_path/'result.json', dict(row_id='failed', status='verification_timeout',
        logical_status='UNKNOWN', semantic_benefit=None, semantic_status='not_evaluated', elapsed_seconds=244))
    proof_value = dict(result=result, guard=guard, payloads=[result], temporary=dict(path='untouched.tmp',mode=0,size=0))
    proof = save(tmp_path/'proof.json', proof_value)
    plan = save(tmp_path/'plan.json', dict(evidence=proof, original_step='14408.397'))
    saved = dict(row=row, status='recovered_result_after_manifest_failure',
        detail='Original published UNKNOWN timeout retained without replay; outer call telemetry was not persisted.',
        result=result, cleanup_complete=True, resources={}, elapsed_seconds=244,
        payloads=[result], unpublished_payloads=[proof_value['temporary']], original_guard=guard,
        recovery_evidence=proof, ownership=dict(step_id='14408.397', surviving_processes=0,
            cgroup_absent=True, signals_sent=0), additional_scientific_seconds=0,
        outer_telemetry_available=False, prior_costs_reset=False, identity=canonical_hash((plan,'science')))
    return checkpoint(tmp_path/'retained.json', saved), saved, dict(plan=plan,evidence=proof)


def test_manifest_timeout_preserves_missing_telemetry_and_cost(tmp_path):
    ref, saved, recovery = manifest_fixture(tmp_path)
    audit.manifest_unknown(audit.Evidence(), ref, saved, recovery, 'science')
    assert not saved['outer_telemetry_available'] and saved['elapsed_seconds'] == 244


@pytest.mark.parametrize('change', ['cost','telemetry','replay','owner','identity','result'])
def test_manifest_timeout_cannot_be_promoted_or_replayed(tmp_path, change):
    ref, saved, recovery = manifest_fixture(tmp_path)
    if change == 'cost': saved['elapsed_seconds'] = 0
    elif change == 'telemetry': saved['outer_telemetry_available'] = True
    elif change == 'replay': saved['additional_scientific_seconds'] = 10
    elif change == 'owner': saved['ownership']['surviving_processes'] = 1
    elif change == 'identity': saved['identity'] = 'other'
    elif change == 'result': saved['result'] = save(tmp_path/'promoted.json', {'semantic_status':'known'})
    ref = checkpoint(Path(ref['path']), saved)
    with pytest.raises(ValueError):
        audit.manifest_unknown(audit.Evidence(), ref, saved, recovery, 'science')


def value_fixture(tmp_path):
    from exact.repair.records import make_objective, VerificationReportV3, ObligationV2, RepairResultV3
    from tools.repair.corpus import generate_corpus

    case = generate_corpus(split_counts={'train':0,'development':1,'test':0}, siblings_per_parent=1,
                           families=('overlap',), revision='v3')[0]
    problem = case.problem
    objective = make_objective(problem.objects, tuple(tuple(0.5 for _ in o.candidates) for o in problem.objects))
    assignment = tuple(0 for _ in problem.objects)
    verification = VerificationReportV3(assignment_hash=canonical_hash(assignment), theory_hash='fixture',
        policy_hash=problem.policy.content_hash, verdict='VERIFIED_FEASIBLE', scope='complete_supported_fragment',
        obligations=(ObligationV2('fixture','pass',True),), expected_obligations=('fixture',))
    repair = RepairResultV3(input_hash=problem.content_hash, objective_hash=objective.content_hash,
        logical_status='VERIFIED_FEASIBLE', search_status='OPTIMAL_IN_POOL',
        verification_scope='complete_supported_fragment', candidate_coverage='fixture', assignment=assignment,
        selected=(), alignment=None, ontology_patch=None, lower_bound=None, upper_bound=None, verification=verification)
    pool = dict(input=problem.to_dict(), objective=objective.to_dict(), model_hash='frozen', proposal_reports=[])
    scoring = dict(original_inventory_hash=canonical_hash(problem.objects),scored_inventory_hash=canonical_hash(problem.objects),
        generation=False,fitting=False,evaluator_opened=False,coverage=dict(model_hash='frozen'))
    predicted = objective.score(assignment)/objective.scale
    result = dict(study_kind='common_inventory_diagnostic', common_inventory_regret=None,
        regret_status='unavailable_no_complete_external_teacher', generation_statuses=[],
        inventory_hash=canonical_hash(problem.objects), selected_assignment=list(assignment),
        selected_objective_utility=predicted, selected_utility=0.3,
        selected_utility_error=predicted-0.3, semantic_status='known')
    refs=[save(tmp_path/'pool.json',pool),save(tmp_path/'inventory-scoring.json',scoring),save(tmp_path/'repair.json',repair.to_dict())]
    saved=dict(row=dict(case_index=0,arm_id='model'),result=save(tmp_path/'result.json',result),payloads=refs)
    schedule=dict(cases=[dict(observable=save(tmp_path/'observable.json',problem.to_dict()))],
                  arms=[dict(id='model',kind='learned',model_hash='frozen')])
    return saved,schedule,result,pool,scoring


def test_value_error_uses_saved_objective_and_retains_conditional_scope(tmp_path):
    saved,schedule,result,_,_=value_fixture(tmp_path)
    value=audit.value_diagnostic(audit.Evidence(),saved,schedule)
    assert value['selected_utility_error']==result['selected_utility_error']
    assert value['common_inventory_regret'] is None and not value['semantic_claim_qualified']


@pytest.mark.parametrize('change', ['prediction','error','regret','model','generation','hidden','assignment','pool','unknown'])
def test_value_diagnostic_rejects_changed_objective_or_scope(tmp_path,change):
    saved,schedule,result,pool,scoring=value_fixture(tmp_path)
    if change=='prediction':result['selected_objective_utility']+=1
    elif change=='error':result['selected_utility_error']+=1
    elif change=='regret':result['common_inventory_regret']=0
    elif change=='model':pool['model_hash']='other'
    elif change=='generation':pool['proposal_reports']=[{'status':'generated'}]
    elif change=='hidden':scoring['evaluator_opened']=True
    elif change=='assignment':result['selected_assignment']=[100]
    elif change=='pool':scoring['scored_inventory_hash']='other'
    elif change=='unknown':result['semantic_status']='unknown'
    saved['result']=save(tmp_path/'result.json',result)
    saved['payloads'][:2]=[save(tmp_path/'pool.json',pool),save(tmp_path/'inventory-scoring.json',scoring)]
    with pytest.raises(ValueError):audit.value_diagnostic(audit.Evidence(),saved,schedule)
