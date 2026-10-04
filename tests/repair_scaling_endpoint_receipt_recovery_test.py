"""Receipt recovery cannot replay work or promote a cleanup error to science."""
from pathlib import Path

import pytest

from exact.repair.records import canonical_hash
from tools.repair import scaling, scaling_endpoint_receipt_recovery as recovery
from tools.repair.batch import read, sha
from tools.repair.expanded_corpus import binding, immutable
from tools.repair.expanded_profile import checkpoint


def fixture(tmp_path):
    original = tmp_path / 'scaling'
    refs, rows, scheduled = [], [], []
    for i in range(6):
        row = dict(id=str(i), case_index=0, method=str(i//2), cache_mode='warm' if i % 2 else 'cold')
        scheduled.append(row)
        payload = original / 'payloads' / str(i) / 'result.json'
        immutable(payload, dict(row=row, logical_status='UNKNOWN'))
        saved = checkpoint(original / 'rows' / (str(i)+'.json'), canonical_hash(('science', row)),
            row=row, status='error' if i == 5 else 'complete', detail=recovery.CAUSE if i == 5 else '',
            result=None if i == 5 else binding(payload), payloads=[binding(payload)],
            elapsed_seconds=237, resources=dict(cpu_seconds=40), cleanup_complete=True)
        refs.append(binding(original / 'rows' / (str(i)+'.json')))
        rows.append(saved)
    guard = original / 'inflight/5.json'
    checkpoint(guard, canonical_hash(('science', scheduled[-1])), row=scheduled[-1], started_epoch=10)
    immutable(tmp_path/'schedule.json', dict(rows=scheduled))
    immutable(tmp_path/'completion.json', dict(status='failed', step_id='14408.307'))
    plan = dict(original=str(original), output=str(tmp_path/'recovery'), slice=[0,6], failed_index=5,
        rows=refs, guard=binding(guard), original_step='14408.307',
        schedule=binding(tmp_path/'schedule.json'), runtime={}, scientific_batch={},
        completion=binding(tmp_path/'completion.json'))
    immutable(tmp_path/'plan.json', plan)
    owner = dict(step_id='14408.307', surviving_processes=0, cgroup_absent=True, signals_sent=0)
    return tmp_path/'plan.json', plan, dict(rows=scheduled), rows, owner


def test_full_denominator_is_preserved_without_scientific_calls(tmp_path, monkeypatch):
    path, plan, schedule, rows, owner = fixture(tmp_path)
    from exact.repair import workers
    monkeypatch.setattr(workers, 'bounded_call', lambda *a, **kw: pytest.fail('Scientific replay'))
    before = {str(p): sha(p) for p in Path(plan['original']).rglob('*.json')}
    assert recovery.validate_rows(plan, schedule, 'science', scaling) == rows
    report = recovery.publish(path, plan, rows, owner)
    assert report['scheduled_rows'] == report['recorded_rows'] == 6
    assert report['counts'] == {'complete': 5, 'unknown_after_cleanup_reconciliation': 1}
    assert report['scientific_rows_replayed'] == report['additional_scientific_seconds'] == 0
    for old, new in zip(plan['rows'][:-1], report['rows'][:-1], strict=True):
        assert all(new[k] == v for k, v in old.items())
    unknown = read(report['rows'][-1]['path'])
    assert unknown['result'] is None
    assert unknown['original_detail'] == recovery.CAUSE
    assert unknown['original_guard'] == plan['guard']
    assert unknown['resources'] == rows[-1]['resources']
    assert {str(p): sha(p) for p in Path(plan['original']).rglob('*.json')} == before
    assert recovery.publish(path, plan, rows, dict(owner, checked_epoch=100)) == report


@pytest.mark.parametrize('field,value', [('surviving_processes',1), ('cgroup_absent',False),
                                       ('signals_sent',1), ('step_id','14408.0')])
def test_unresolved_owner_prevents_publication(tmp_path, field, value):
    path, plan, _, rows, owner = fixture(tmp_path)
    with pytest.raises(ValueError, match='owner cleanup'):
        recovery.publish(path, plan, rows, dict(owner, **{field:value}))
    assert not Path(plan['output']).exists()


@pytest.mark.parametrize('mutation', ['payload', 'extra_payload', 'row', 'guard', 'prefix_guard',
                                     'incomplete_slice', 'earlier_failure'])
def test_changed_or_incomplete_evidence_is_rejected(tmp_path, mutation):
    _, plan, schedule, rows, _ = fixture(tmp_path)
    original = Path(plan['original'])
    if mutation == 'payload':
        Path(rows[0]['payloads'][0]['path']).write_text('{}')
    elif mutation == 'extra_payload':
        (original/'payloads/0/extra.json').write_text('{}')
    elif mutation == 'row':
        Path(plan['rows'][0]['path']).write_text('{}')
    elif mutation == 'guard':
        Path(plan['guard']['path']).write_text('{}')
    elif mutation == 'prefix_guard':
        (original/'inflight/0.json').write_text('{}')
    elif mutation == 'incomplete_slice':
        plan['rows'].pop()
    else:
        plan['failed_index'] = 3
    with pytest.raises(ValueError):
        recovery.validate_rows(plan, schedule, 'science', scaling)
    assert not Path(plan['output']).exists()


@pytest.mark.parametrize('target', ['reconciled-unknown.json', 'report.json'])
def test_existing_revision_cannot_be_replaced(tmp_path, target):
    path, plan, _, rows, owner = fixture(tmp_path)
    recovery.publish(path, plan, rows, owner)
    changed = Path(plan['output'])/target
    changed.write_text('{}')
    with pytest.raises(ValueError):
        recovery.publish(path, plan, rows, owner)
    assert changed.read_text() == '{}'
