"""Integrity boundaries for original and migrated fresh-evaluation receipts."""
import copy
import json

import pytest

from exact.repair.records import canonical_hash
from tools.repair import fresh_evaluation_audit as audit
from tools.repair.schema_cleanup_recovery import CAUSE, reconciled_unknown


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return audit.binding(path)


def checkpoint(path, value):
    return save(path, dict(value, content_hash=canonical_hash(value)))


def receipt_fixture(tmp_path):
    complete = dict(job_id='schema', step_id='14408.9', dispatch_nonce='nonce',
                    status='complete', exit_code=0, elapsed_seconds=2, cpu_seconds=1)
    batch = save(tmp_path / 'batch.json', dict(jobs=[dict(id='schema')],
                                             frozen_files={}, commit='frozen'))
    complete['batch'] = batch['path']
    cp = save(tmp_path / 'attempt/completion.json', complete)
    run = dict(id='schema-001', logical_id='primary', original_run_id='primary-001',
               step_id='14408.9', dispatch_nonce='nonce', completion_path=cp['path'])
    (tmp_path / 'exit').write_text('0')
    (tmp_path / 'batch.sha256').write_text(batch['sha256'])
    item = dict(run=run, completion=cp, step=save(tmp_path / 'step.json', run),
        exit=audit.binding(tmp_path / 'exit'), batch=batch,
        batch_hash=audit.binding(tmp_path / 'batch.sha256'), runtime=save(tmp_path / 'runtime.json', {}))
    ledger = dict(attempts={str(tmp_path / 'attempt'): dict(status='settled', logical_id='schema',
                                                         elapsed_seconds=2, cpu_seconds=1)})
    return item, ledger


def test_schema_alias_preserves_both_lineage_and_charged_worker(tmp_path):
    item, ledger = receipt_fixture(tmp_path)
    result = audit.validate_attempt(audit.Evidence(), item, ledger, True)
    assert result['run']['logical_id'] == 'primary'
    assert result['charge']['logical_id'] == result['completion']['job_id'] == 'schema'


@pytest.mark.parametrize('change', ['nonce', 'alias', 'cost', 'worker', 'unsettled'])
def test_attempt_rejects_forged_ownership_alias_or_lost_cost(tmp_path, change):
    item, ledger = receipt_fixture(tmp_path)
    if change == 'nonce':
        item['run']['dispatch_nonce'] = 'other'
    elif change == 'alias':
        item['run']['original_run_id'] = 'unrelated-001'
    elif change == 'worker':
        item['run']['id'] = 'other-001'
    elif change == 'cost':
        ledger['attempts'][str(tmp_path / 'attempt')]['elapsed_seconds'] = 1
    else:
        ledger['attempts'][str(tmp_path / 'attempt')]['status'] = 'reserved'
    with pytest.raises(ValueError):
        audit.validate_attempt(audit.Evidence(), item, ledger, True)


def rows_fixture(tmp_path):
    rows = [dict(id=str(i), case_index=0, arm_id='model') for i in range(2)]
    case = dict(case_id='case', group_id='parent', family='family', family_exposure='unseen_family',
                control='corrupted')
    schedule = dict(schema='exact-repair/fresh-evaluation/v1', rows=rows, cases=[case],
                    arms=[dict(id='model')])
    schedule_ref = save(tmp_path / 'schedule.json', schedule)
    admission = save(tmp_path / 'admission.json', dict(rows=[dict(row_id=r['id']) for r in rows]))
    plan = dict(schedule=schedule_ref, preflight=admission,
                rows=[dict(row_id=r['id']) for r in rows])
    plan_ref = save(tmp_path / 'plan.json', plan)
    source = dict(code=str(tmp_path), frozen_files={
        str(tmp_path / 'tools/repair/schema_recovery.py'): 'schema-source',
        str(tmp_path / 'tools/repair/fresh_evaluation.py'): 'fresh-source'})
    identity = audit.schema_identity(audit.Evidence(), plan_ref, source, {})
    refs = []
    for row, entry in zip(rows, plan['rows']):
        value = dict(identity=canonical_hash((identity, row, entry, entry)),
            row=row, status='timeout', cleanup_complete=True, elapsed_seconds=300,
            resources={}, payloads=[], result=None)
        ref = checkpoint(tmp_path / 'rows' / (row['id'] + '.json'), value)
        refs.append(dict(ref, row_id=row['id'], status=value['status']))
    report = dict(identity=identity, schedule=schedule_ref, plan=plan_ref,
                  status='complete', scheduled=2, recorded=2, rows=refs, outcomes={'timeout': 2})
    item = dict(report=checkpoint(tmp_path / 'report.json', report), row_slice=[0, 2],
                batch=save(tmp_path / 'batch.json', source), run={'id': 'schema-001'})
    return schedule, item, report


def test_timeout_rows_keep_denominator_and_unknown_semantics(tmp_path):
    schedule, item, _ = rows_fixture(tmp_path)
    rows = audit.audit_rows(audit.Evidence(), schedule, [(item, {'runtime': {}})])
    assert len(rows) == 2
    assert all(r['logical_status'] == 'UNKNOWN' and r['semantic_benefit'] is None and
               not r['semantic_claim_qualified'] for r in rows)


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'runtime', 'guard', 'status', 'identity'])
def test_rows_reject_denominator_source_and_ownership_changes(tmp_path, change):
    schedule, item, report = rows_fixture(tmp_path)
    source = {'runtime': {}}
    if change == 'missing':
        report['rows'].pop()
    elif change == 'duplicate':
        report['rows'][1] = report['rows'][0]
    elif change == 'runtime':
        source['runtime'] = {'changed': True}
    elif change == 'guard':
        save(tmp_path / 'inflight/0.json', {})
    elif change == 'status':
        report['rows'][0]['status'] = 'complete'
    elif change == 'identity':
        report['identity'] = 'another-source'
    report.pop('content_hash', None)
    item['report'] = checkpoint(tmp_path / 'report.json', report)
    with pytest.raises(ValueError):
        audit.audit_rows(audit.Evidence(), schedule, [(item, source)])


def test_cleanup_unknown_retains_original_cost_and_rejects_promotion(tmp_path):
    row = {'id': 'stopped'}
    previous = dict(identity='original', row=row, status='error', detail=CAUSE,
        cleanup_complete=True, result=None, payloads=[], elapsed_seconds=219, resources={},
        remaining_budget={'wall_seconds': 280}, recovery_action='compatible_remaining_budget_execution')
    original = checkpoint(tmp_path / 'original.json', previous)
    guard = checkpoint(tmp_path / 'guard.json', dict(identity='original', row=row,
                                                   remaining_budget=previous['remaining_budget']))
    proof = save(tmp_path / 'proof.json', {'guard': guard})
    plan = save(tmp_path / 'plan.json', dict(original_step='14408.223', schema_identity='schema'))
    ownership = dict(step_id='14408.223', surviving_processes=0, cgroup_absent=True, signals_sent=0)
    saved = reconciled_unknown(previous, dict(identity='original', row=row,
        remaining_budget=previous['remaining_budget']), original, proof, ownership, '14408.223')
    saved['identity'] = canonical_hash((plan, 'schema'))
    ref = checkpoint(tmp_path / 'unknown.json', saved)
    audit.validate_unknown(audit.Evidence(), saved, dict(plan=plan, evidence=proof), ref)
    changed = copy.deepcopy(saved)
    changed['result'] = {'assignment': [0]}
    with pytest.raises(ValueError, match='changed or promoted'):
        audit.validate_unknown(audit.Evidence(), changed, dict(plan=plan, evidence=proof), ref)


def test_recovery_cannot_change_an_original_payload(tmp_path):
    row = {'id': 'row'}
    payload = save(tmp_path / 'payload.json', {'status': 'error'})
    ref = checkpoint(tmp_path / 'original.json', dict(row=row, payloads=[payload]))
    save(tmp_path / 'payload.json', {'status': 'success'})
    with pytest.raises(ValueError, match='Changed evidence'):
        audit.lineage_row(audit.Evidence(), ref, row)


def test_known_semantic_score_requires_native_and_label_receipts(tmp_path):
    schedule, item, report = rows_fixture(tmp_path)
    ref = report['rows'][0]
    saved = json.loads(open(ref['path']).read())
    saved['result'] = save(tmp_path / 'result.json', dict(row_id='0',
        schedule_hash=canonical_hash(schedule), case_id='case', semantic_status='known'))
    with pytest.raises(ValueError, match='lacks label or native verification'):
        audit.row_summary(audit.Evidence(), saved, schedule['cases'][0], schedule)


def test_budget_cannot_restore_previously_spent_generation(tmp_path):
    from tools.repair.schema_recovery import ERROR

    row = dict(id='row', seconds=300, cpu_seconds=600)
    result = save(tmp_path / 'result.json', dict(status='generation_error', detail=ERROR,
        row_id='row', generation_resources={'wall_seconds': 15}))
    previous = checkpoint(tmp_path / 'previous.json', dict(row=row, status='complete',
        cleanup_complete=True, elapsed_seconds=20, resources={'cpu_seconds': 10}, result=result))
    protocol = save(tmp_path / 'protocol.json', dict(resources={'generation_seconds': 60}))
    budget = dict(wall_seconds=280, prior_wall_seconds=20, cpu_seconds=590, prior_cpu_seconds=10,
                  generation_seconds=45, prior_generation_seconds=15)
    saved = dict(row=row, remaining_budget=budget, recovery_of=previous)
    audit.validate_budget(audit.Evidence(), saved, {'protocol': protocol})
    budget.update(generation_seconds=60, prior_generation_seconds=0)
    with pytest.raises(ValueError, match='restored spent'):
        audit.validate_budget(audit.Evidence(), saved, {'protocol': protocol})
