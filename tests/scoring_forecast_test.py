import json
import sqlite3

from tools.forecast_scoring_successor import decision_sensitivities, read_hosted_history


def test_incomplete_receipts_do_not_lower_observed_token_or_cost_averages(tmp_path):
    path = tmp_path / 'requests.sqlite3'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE requests(request_id TEXT, identity TEXT)')
        db.execute('CREATE TABLE attempts(request_id TEXT, state TEXT, elapsed_seconds REAL, usage TEXT)')
        db.execute('INSERT INTO requests VALUES (?, ?)', ('a', json.dumps({'role': 'decision'})))
        values = [(2, {'prompt_tokens': 90, 'completion_tokens': 10, 'cost': .2}),
                  (None, {'prompt_tokens': 1000}), (None, None)]
        db.executemany('INSERT INTO attempts VALUES (?, ?, ?, ?)',
                       [('a', 'completed', seconds, json.dumps(usage) if usage else None)
                        for seconds, usage in values])
    row, = read_hosted_history(path)
    assert row['attempts'] == 3
    assert row['token_attempts'] == row['priced_attempts'] == 1
    assert row['measured_tokens'] == 100
    estimate = decision_sensitivities(1000, row)[0]
    assert estimate['primary_decision_requests'] == 1
    assert estimate['primary_decision_tokens'] == 100
    assert estimate['primary_decision_usd'] == .2
    assert estimate['primary_decision_service_seconds'] == 2
    assert estimate['unknown_token_attempts'] == estimate['unpriced_attempts'] == 2


def test_unobserved_roles_remain_unknown_instead_of_zero():
    estimate = decision_sensitivities(1000, {})[0]
    assert estimate['primary_decision_tokens'] is None
    assert estimate['primary_decision_usd'] is None
    assert estimate['primary_decision_service_seconds'] is None
