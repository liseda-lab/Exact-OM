"""Failure-inclusive development evidence must resist scope and ownership drift."""
import copy
import json
from pathlib import Path

import pytest

from exact.repair.records import canonical_hash
from tools.repair import development_decode_audit as audit
from tools.repair.development_decode_recovery import CAUSE, reconciled_unknown


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
        str(tmp_path/'tools/repair/development_decode.py'): 'decode',
        str(tmp_path/'tools/repair/fresh_evaluation.py'): 'fresh'})
    identity = audit.scientific_identity(schedule_ref, batch, {})
    refs = []
    for row in rows:
        result = save(tmp_path/'payloads'/row['id']/'result.json', dict(row_id=row['id'],
            case_id='case', schedule_hash=canonical_hash(schedule), status='generation_timeout'))
        value = dict(identity=canonical_hash((identity, row)), row=row,
            status='complete', cleanup_complete=True, elapsed_seconds=63,
            resources={}, payloads=[result], result=result)
        ref = checkpoint(tmp_path/'rows'/(row['id']+'.json'), value)
        refs.append(dict(ref, row_id=row['id'], status='generation_timeout'))
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
               and r['native_label']['status'] == 'unavailable_no_label_call' for r in rows)


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


def test_known_score_without_native_call_cannot_qualify(tmp_path):
    result = save(tmp_path/'result.json', {'semantic_status': 'known'})
    with pytest.raises(ValueError, match='completed native label call'):
        audit.native_evidence(audit.Evidence(), dict(result=result, payloads=[]), {})


def test_foreign_native_call_cannot_qualify(tmp_path):
    call = save(tmp_path/'native-label/call.json', dict(case_id='foreign', split='test', parent='parent'))
    with pytest.raises(ValueError, match='Native label scope'):
        audit.native_evidence(audit.Evidence(), dict(payloads=[call]),
                              dict(case_id='case', structural_parent='parent'))


@pytest.mark.parametrize('change', [None, 'query', 'policy', 'input', 'split', 'assignment'])
def test_native_call_binds_generated_input_query_policy_and_assignment(tmp_path, change):
    from tools.repair.corpus import generate_corpus
    from tools.repair.prepare import case_to_dict

    case = generate_corpus(split_counts={'train': 0, 'development': 1, 'test': 0},
        siblings_per_parent=1, families=('overlap',), revision='v3')[0]
    item = dict(case_id=case.case_id, structural_parent=case.structural_parent,
                evaluator=save(tmp_path/'evaluator.json', case_to_dict(case)))
    assignment = [0 for _ in case.problem.objects]
    call = dict(case_id=case.case_id, parent=case.structural_parent, split='development',
        query_hash=canonical_hash(case.probes), policy_hash=case.problem.policy.content_hash,
        input_hash=case.problem.content_hash, assignment=assignment, status='complete',
        cleanup_complete=True)
    if change == 'query': call['query_hash'] = 'different-query'
    elif change == 'policy': call['policy_hash'] = 'different-policy'
    elif change == 'input': call['input_hash'] = 'different-pool'
    elif change == 'split': call['split'] = 'test'
    elif change == 'assignment': call['assignment'] = [100]
    result = save(tmp_path/'result.json', dict(semantic_status='known'))
    payloads = [save(tmp_path/'native-label/call.json', call),
        save(tmp_path/'native-label/check-0.json', {'logical_status':'VERIFIED_FEASIBLE'}),
        save(tmp_path/'pool.json', dict(input=case.problem.to_dict(), proposal_reports=[])),
        save(tmp_path/'selected-label.json', dict(assignment=assignment))]
    saved = dict(result=result, payloads=payloads)
    if change:
        with pytest.raises(ValueError): audit.native_evidence(audit.Evidence(), saved, item)
    else:
        native = audit.native_evidence(audit.Evidence(), saved, item)
        assert native['status'] == 'complete' and not native['supervision_admitted']
