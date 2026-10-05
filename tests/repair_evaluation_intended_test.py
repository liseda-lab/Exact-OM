"""Fresh release identity, immutable targets, native evidence and no-replay checks."""
from dataclasses import replace

import pytest

from exact.repair.api import write_artifact
from exact.repair.workers import CallResult
from tools.repair import evaluation_intended as intended
from tools.repair.corpus import generate_corpus
from tools.repair.expanded_profile import parent_fingerprints
from tools.repair.historical_regression import binding, verify_binding
from tools.repair.prepare import case_to_dict


def case_and_row(tmp_path):
    case = generate_corpus(split_counts={'train': 0, 'development': 0, 'test': 1},
        siblings_per_parent=1, families=('overlap',), revision='v3')[0]
    record = case_to_dict(case)
    write_artifact(tmp_path / 'case.json', record)
    write_artifact(tmp_path / 'input.json', case.problem.to_dict())
    row = dict(case_id=case.case_id, case_hash=record['hash'], input_hash=case.problem.content_hash,
        structural_parent=case.structural_parent, group_id=case.structural_parent,
        control=case.control, family=case.family, split='fresh_evaluation', status='materialized',
        fingerprints=list(parent_fingerprints(case)), evaluator=binding(tmp_path / 'case.json'),
        observable=binding(tmp_path / 'input.json'))
    return row, record, case


@pytest.mark.parametrize('change', [dict(split='test'), dict(split='development'),
    dict(expected_cases=63), dict(expected_parents=31), dict(model_fitting=True),
    dict(supervision_admitted=True), dict(test_feedback_for_selection=True),
    dict(original_results_replaced=True), dict(scientific_comparison_budgets_changed=True),
    dict(budget={})])
def test_boundary_fails_before_file_access(change):
    plan = dict(schema=intended.SCHEMA, split='fresh_evaluation', expected_cases=64,
        expected_parents=32, model_fitting=False, supervision_admitted=False,
        test_feedback_for_selection=False, original_results_replaced=False,
        scientific_comparison_budgets_changed=False, budget=intended.BUDGET)
    with pytest.raises(ValueError, match='boundary'):
        intended.validate_schedule({**plan, **change})


@pytest.mark.parametrize('change', [dict(split='test'), dict(structural_parent='renamed'),
    dict(case_hash='bad'), dict(input_hash='bad'), dict(fingerprints=[]), dict(control='wrong'),
    dict(status='unavailable')])
def test_fresh_identity_cannot_be_substituted(tmp_path, change):
    row, record, case = case_and_row(tmp_path)
    checked_row, checked_record, checked_case = intended.qualified_case(row)
    assert checked_row == row and checked_record['hash'] == record['hash']
    assert intended.canonical_hash(checked_case) == intended.canonical_hash(case)
    with pytest.raises(ValueError, match='Frozen fresh'):
        intended.qualified_case({**row, **change})


@pytest.mark.parametrize('status,cleanup', [('timeout', True), ('error', True), ('timeout', False)])
def test_fresh_unknown_or_error_keeps_evidence_without_replay(tmp_path, monkeypatch, status, cleanup):
    row, record, case = case_and_row(tmp_path)
    calls = []
    def execute(*args, **kwargs):
        calls.append((args[0], kwargs))
        return CallResult(status, detail='ValueError: fixture failure' if status == 'error'
            else 'deadline exhausted', cleanup_complete=cleanup)
    monkeypatch.setattr(intended.native, 'bounded_call', execute)
    for _ in range(2):
        if status == 'error' or not cleanup:
            with pytest.raises(RuntimeError):
                intended.one_case(row, record, case, tmp_path, 'frozen', intended.BUDGET)
        else:
            saved = verify_binding(intended.one_case(row, record, case, tmp_path, 'frozen', intended.BUDGET))
            assert saved['schema'] == intended.SCHEMA and saved['split'] == 'test'
            assert saved['status'] == 'timeout' and saved['query_denominator'] == len(case.probes)
            assert not saved['supervision_admitted'] and not saved['original_results_replaced']
    assert calls == [(intended.case_worker, dict(timeout=600, cpu_seconds=1200, memory_mb=8192))]


def test_worker_rejects_training_before_native(tmp_path):
    _, _, case = case_and_row(tmp_path)
    with pytest.raises(ValueError, match='test cases'):
        intended.case_worker(case_to_dict(replace(case, split='train')), tmp_path)


@pytest.mark.slow
def test_native_fresh_fixture_preserves_policy_and_typed_queries(tmp_path):
    _, record, case = case_and_row(tmp_path)
    call = intended.native.bounded_call(intended.case_worker, record, str(tmp_path / 'native-case'),
        timeout=120, cpu_seconds=240, memory_mb=8192)
    assert call.status == 'complete', call.detail
    assert call.cleanup_complete and call.value['original_guard_passed']
    assert call.value['intended_target_satisfied']
    assert len(call.value['semantic']['outcomes']) == len(case.probes)
    assert list((tmp_path / 'native-case/native').glob('*.obligations.jsonl'))
