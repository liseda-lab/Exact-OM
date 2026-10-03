"""Fresh evaluation boundaries: no hidden outcomes, immutable retries, actual native controls."""

import dataclasses
from pathlib import Path

import pytest

from exact.repair.api import write_artifact
from exact.repair.records import canonical_hash
from exact.repair.workers import CallResult, bounded_call
from tools.repair import fresh_evaluation as fresh
from tools.repair.prepare import load_preparation, case_to_dict

pytest_plugins = ["tests.repair_prepare_controls_test"]


def small_schedule(tmp_path, *, missing=False):
    cases = [dict(case_id="a", status="unavailable" if missing else "materialized")]
    arms = [dict(id="model", status="available")]
    rows = fresh.make_rows(cases, arms)
    return dict(cases=cases, arms=arms, rows=rows), rows[0]


def test_denominator_keeps_missing_cases_and_all_arms():
    cases = [dict(status="unavailable_structural_parent") for _ in range(64)]
    arms = [dict(id=str(n)) for n in range(9)]
    rows = fresh.make_rows(cases, arms)
    assert len(rows) == len({r["id"] for r in rows}) == 576
    assert all((r["seconds"], r["cpu_seconds"], r["memory_mb"]) == (300, 600, 8192) for r in rows)


def test_unavailable_and_timeout_are_not_silently_repeated(tmp_path, monkeypatch):
    import exact.repair.workers

    calls = []

    def bounded(*args, **kw):
        calls.append(kw)
        return CallResult("timeout", detail="native budget", cleanup_complete=True)

    monkeypatch.setattr(exact.repair.workers, "bounded_call", bounded)
    schedule, row = small_schedule(tmp_path)
    first = fresh.one_row(schedule, row, tmp_path, "frozen")
    assert first == fresh.one_row(schedule, row, tmp_path, "frozen")
    assert first["status"] == "timeout" and len(calls) == 1
    assert calls[0]["timeout"] == 300 and calls[0]["cpu_seconds"] == 600
    missing, row = small_schedule(tmp_path, missing=True)
    result = fresh.one_row(missing, row, tmp_path / "missing", "missing")
    assert result["status"] == "unavailable" and len(calls) == 1
    with pytest.raises(ValueError, match="dependencies"):
        fresh.one_row(schedule, row, tmp_path, "changed")


def test_inflight_and_unclean_nested_calls_block_reexecution(tmp_path, monkeypatch):
    import exact.repair.workers

    schedule, row = small_schedule(tmp_path)
    write_artifact(tmp_path / "inflight" / (row["id"] + ".json"), {"previous_owner": "14408.8"})
    with pytest.raises(RuntimeError, match="reconciliation"):
        fresh.one_row(schedule, row, tmp_path, "frozen")
    monkeypatch.setattr(
        exact.repair.workers,
        "bounded_call",
        lambda *a, **k: CallResult(
            "error", detail="Nested worker cleanup incomplete", cleanup_complete=True
        ),
    )
    with pytest.raises(RuntimeError, match="cleanup incomplete"):
        fresh.one_row(schedule, row, tmp_path / "nested", "frozen")
    with pytest.raises(RuntimeError, match="inflight reconciliation"):
        fresh.one_row(schedule, row, tmp_path / "nested", "frozen")


def test_all_partial_payloads_are_bound_even_after_timeout(tmp_path, monkeypatch):
    import exact.repair.workers

    schedule, row = small_schedule(tmp_path)

    def bounded(*args, **kwargs):
        payload = Path(args[3])
        write_artifact(payload / "search-ledger.json", {"queries": 3})
        return CallResult("timeout")

    monkeypatch.setattr(exact.repair.workers, "bounded_call", bounded)
    result = fresh.one_row(schedule, row, tmp_path, "frozen")
    assert len(result["payloads"]) == 1
    Path(result["payloads"][0]["path"]).write_text("{}")
    with pytest.raises(ValueError, match="payload changed"):
        fresh.one_row(schedule, row, tmp_path, "frozen")


def test_controls_keep_costs_native_pool_identity_and_filter_language(prepared):
    source, protocol = prepared
    cases, _, _ = load_preparation(source)
    problem = cases[0].problem
    settings = fresh.read(protocol)
    symbolic, objective = fresh.control_objective(problem, "symbolic_rich_action", settings)
    uniform, zero = fresh.control_objective(problem, "uniform", settings)
    deleted, reduced = fresh.control_objective(problem, "deletion", settings)
    assert symbolic is uniform is problem
    assert objective.costs == zero.costs and not objective.pairs
    assert objective.pool_hash == canonical_hash(problem.objects)
    assert reduced.pool_hash == canonical_hash(deleted.objects)
    assert any(any(x > 0 for x in values) for values in objective.benefit)
    assert all(x == 0 for values in zero.benefit for x in values)
    assert all(
        "keep" in c.action_tags or ("delete" in c.action_tags and not c.axioms)
        for o in deleted.objects
        for c in o.candidates
    )
    assert problem.objects != deleted.objects


@pytest.mark.parametrize("arm", fresh.CONTROLS)
def test_native_control_generation_and_evaluator_pipeline(prepared, tmp_path, arm):
    source, protocol = prepared
    cases, _, _ = load_preparation(source)
    case = next(c for c in cases if c.split == "development")
    record = case_to_dict(case)
    item = dict(
        case_id=case.case_id,
        structural_parent=case.structural_parent,
        family=case.family,
        family_exposure="seen_family",
        status="materialized",
        observable=fresh.immutable(tmp_path / "input.json", case.problem.to_dict()),
        evaluator=fresh.immutable(tmp_path / "evaluator.json", record),
    )
    arms = [dict(id=arm, kind="control", status="available", protocol=fresh.binding(protocol))]
    schedule = dict(cases=[item], arms=arms)
    row = fresh.make_rows([item], arms)[0]
    result = bounded_call(
        fresh.evaluate_row,
        schedule,
        row,
        str(tmp_path / "run"),
        timeout=300,
        cpu_seconds=600,
        memory_mb=8192,
    )
    assert result.status == "complete", result.detail
    payload = fresh.bound(result.value)
    assert payload["status"] == "evaluated", payload
    pool = fresh.read(tmp_path / "run/pool.json")
    from exact.repair.records import read_record

    problem = read_record(pool["input"])
    objective = read_record(pool["objective"])
    assert objective.pool_hash == canonical_hash(problem.objects)
    assert all(r["payload"]["arm"] == "grammar_uniform" for r in pool["proposal_reports"])
    if payload["logical_status"] == "VERIFIED_FEASIBLE":
        assert payload["semantic_status"] == "known", payload
        assert (tmp_path / "run/selected-label.json").exists()


def test_corpus_requires_nonce_bound_completed_attempt(tmp_path):
    completion = tmp_path / "completion.json"
    outputs = tmp_path / "outputs.json"
    write_artifact(
        completion,
        dict(status="complete", exit_code=0, step_id="14408.48", dispatch_nonce="foreign"),
    )
    write_artifact(outputs, {})
    with pytest.raises(ValueError, match="ownership"):
        fresh.validate_corpus(
            dict(
                completion=fresh.binding(completion),
                outputs=fresh.binding(outputs),
                step_id="14408.48",
                nonce="expected",
            )
        )
