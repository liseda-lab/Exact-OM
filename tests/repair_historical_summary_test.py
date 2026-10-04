import pytest

from tools.repair.historical_summary import event_backends, revised_view


def rows():
    original = [dict(case_id='a', stage='migrated_inventory', status='error',
                     receipt=dict(path='old.json', sha256='abc')),
                dict(case_id='b', stage='generated_cold', status='unavailable_generated_pool',
                     receipt=dict(path='missing.json', sha256='def'))]
    replacement = dict(case_id='a', stage='migrated_inventory', status='timeout',
        repair_of=original[0]['receipt'], cleanup_complete=True, same_cause_repair_attempt=1,
        prior_elapsed_seconds=2, remaining_total_seconds=58, original_total_seconds=60)
    return original, replacement


def test_revised_historical_view_preserves_failure_denominator_and_original():
    original, replacement = rows()
    revised = revised_view(original, [replacement])
    assert [r['status'] for r in revised] == ['timeout', 'unavailable_generated_pool']
    assert original[0]['status'] == 'error'


@pytest.mark.parametrize('change', [
    dict(repair_of=dict(path='unrelated.json', sha256='abc')),
    dict(remaining_total_seconds=60), dict(cleanup_complete=False),
    dict(same_cause_repair_attempt=3), dict(case_id='unscheduled')])
def test_historical_replacement_rejects_unqualified_ancestry_or_reset_budget(change):
    original, replacement = rows()
    with pytest.raises(ValueError):
        revised_view(original, [dict(replacement, **change)])


def test_historical_replacement_cannot_replace_success_or_duplicate_row():
    original, replacement = rows()
    with pytest.raises(ValueError):
        revised_view(original, [replacement, replacement])
    original[0]['status'] = 'complete'
    with pytest.raises(ValueError):
        revised_view(original, [replacement])


def test_historical_backend_scope_counts_events_not_admitted_routes():
    event = {'$record': 'VerificationEventV3', 'backend': 'elk/0.2.1:rust'}
    assert event_backends({'routes': ['elk', 'hermit'], 'backend': 'hermit',
                           'baseline': [event, event]}) == {'elk/0.2.1:rust'}
    assert event_backends({'routes': ['elk', 'hermit']}) == set()
