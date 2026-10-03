"""Active unchanged expressions and debit-before-repair regression coverage."""
import dataclasses

import pytest

from tools.repair import historical_audit as audit
from tools.repair.historical_audit_recovery import ERROR, repaired_input
from tests.repair_historical_audit_test import plan_and_input
from exact.repair.records import read_record


def complex_original():
    _, record = plan_and_input()
    problem = read_record(record)
    obj = problem.objects[0]
    cls = problem.policy.monitored_classes[0]
    unchanged = next(c for c in obj.candidates if set(c.axioms) == set(obj.original_axioms))
    candidate = dataclasses.replace(unchanged, active_expressions=(cls,))
    return dataclasses.replace(problem, objects=(dataclasses.replace(obj,
        candidates=tuple(candidate if c == unchanged else c for c in obj.candidates)), *problem.objects[1:]))


def prior(**kwargs):
    return dict(status='error', detail=ERROR, stage='migrated_inventory', result=None,
                cleanup_complete=True, elapsed_seconds=2.0, **kwargs)


def test_active_original_reaches_repair_and_routes_preserve_expressions(tmp_path, monkeypatch):
    import exact.repair.kernel
    import exact.repair.owl
    problem = complex_original()
    captured = {}
    def routes(snapshot, queries, active):
        captured['active'] = active
        return ('hermit',)
    class ReachedRepair(Exception):
        pass
    def repair(value, objective, **kwargs):
        assert value == dataclasses.replace(problem, budgets=dataclasses.replace(problem.budgets, memory_mb=8192))
        assert kwargs['preserve_verified_input'] is False
        raise ReachedRepair
    monkeypatch.setattr(exact.repair.owl, 'qualified_routes', routes)
    monkeypatch.setattr(exact.repair.kernel, 'repair', repair)
    with pytest.raises(ReachedRepair):
        audit.native_probe(problem.to_dict(), {}, 1000, 8192, tmp_path)
    assert captured['active'] == (problem.policy.monitored_classes[0],)


def test_repair_debits_prior_wall_preserves_every_other_input_field():
    problem = complex_original()
    result = repaired_input(problem.to_dict(), prior())
    assert result.budgets.total_seconds == problem.budgets.total_seconds - 2
    assert dataclasses.replace(result, budgets=problem.budgets) == problem
    assert dataclasses.replace(result.budgets, total_seconds=problem.budgets.total_seconds) == problem.budgets


@pytest.mark.parametrize('change', [dict(status='timeout'), dict(detail='different software cause'),
    dict(stage='generated_cold'), dict(result={'partial':True}), dict(cleanup_complete=False),
    dict(elapsed_seconds=60), dict(elapsed_seconds=-1), dict(elapsed_seconds=float('nan'))])
def test_completed_or_exhausted_rows_are_not_admitted(change):
    row = prior(); row.update(change)
    with pytest.raises(ValueError):
        repaired_input(complex_original().to_dict(), row)


def test_other_inputs_cannot_be_silently_repaired():
    _, record = plan_and_input()
    with pytest.raises(ValueError, match='witness'):
        repaired_input(record, prior())
