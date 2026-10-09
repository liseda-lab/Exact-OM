"""Evidence identity, failure denominators and no-repeat native audit contracts."""

import dataclasses
import json
from pathlib import Path

import pytest

from exact.repair.api import write_artifact
from exact.repair.records import canonical_hash
from exact.repair.workers import CallResult
from tools.repair import historical_audit as audit
from tools.repair.expanded_profile import checkpoint
from tools.repair.historical_regression import migrate
from tools.repair.prepare import case_from_dict
from tests.repair_historical_regression_test import archived_case


def plan_and_input():
    record, _ = migrate(archived_case())
    case = case_from_dict(record)
    return dict(profile={}, scale=1000, memory_mb=8192), case.problem.to_dict()


def test_historical_resource_errors_and_unsupported_bundles_are_distinct():
    for text in ("CircuitBudgetExceeded: limit", "TimeoutError: deadline exhausted"):
        assert (
            audit.generation_classification(dict(call_status="error", detail=text))
            == "resource_bounded_original_limits"
        )
    assert (
        audit.generation_classification(
            dict(
                call_status="error",
                detail="cannot infer complete original relation for endpoint retrieval",
            )
        )
        == "unsupported_original_mapping_bundle"
    )
    assert (
        audit.generation_classification(
            dict(call_status="error", detail="AttributeError: wrong API")
        )
        == "requires_software_or_resource_review"
    )


def test_missing_pool_retains_row_and_cannot_be_replaced_silently(tmp_path, monkeypatch):
    import exact.repair.workers

    def forbidden(*a, **k):
        pytest.fail("Unavailable generated pool must not execute fallback inventory")

    monkeypatch.setattr(exact.repair.workers, "bounded_call", forbidden)
    plan, problem = plan_and_input()
    row = audit._one(tmp_path, "source-v1", "row", None, plan, dict(case_id="case"))
    assert row["status"] == "unavailable_generated_pool" and row["result"] is None
    assert audit._one(tmp_path, "source-v1", "row", None, plan, dict(case_id="case")) == row
    with pytest.raises(ValueError, match="dependencies"):
        audit._one(tmp_path, "source-v1", "row", problem, plan, dict(case_id="case"))


def test_completed_timeout_not_retried_and_original_budget_retained(tmp_path, monkeypatch):
    import exact.repair.workers

    plan, problem = plan_and_input()
    calls = []

    def bounded(*a, **kw):
        calls.append(kw)
        return CallResult("timeout", detail="bounded", cleanup_complete=True)

    monkeypatch.setattr(exact.repair.workers, "bounded_call", bounded)
    first = audit._one(tmp_path, "source-v1", "row", problem, plan, dict(case_id="case"))
    assert first["status"] == "timeout"
    assert audit._one(tmp_path, "source-v1", "row", problem, plan, dict(case_id="case")) == first
    assert len(calls) == 1 and calls[0]["timeout"] == 60
    assert not (tmp_path / "inflight/row.json").exists()


def test_interrupted_or_unclean_probe_blocks_reexecution(tmp_path, monkeypatch):
    import exact.repair.workers

    plan, problem = plan_and_input()
    write_artifact(tmp_path / "inflight/row.json", {"owner": "previous"})
    with pytest.raises(RuntimeError, match="reconciliation"):
        audit._one(tmp_path, "source", "row", problem, plan, {})
    monkeypatch.setattr(
        exact.repair.workers,
        "bounded_call",
        lambda *a, **k: CallResult("timeout", cleanup_complete=False),
    )
    with pytest.raises(RuntimeError, match="cleanup incomplete"):
        audit._one(tmp_path, "source", "second", problem, plan, {})
    assert (tmp_path / "inflight/second.json").exists()


def test_completed_payload_tampering_rejected(tmp_path, monkeypatch):
    import exact.repair.workers

    plan, problem = plan_and_input()
    payload = tmp_path / "payload.json"
    write_artifact(payload, {"receipt": 1})
    monkeypatch.setattr(
        exact.repair.workers,
        "bounded_call",
        lambda *a, **k: CallResult(
            "complete", value={"payload": audit.binding(payload)}, cleanup_complete=True
        ),
    )
    audit._one(tmp_path, "source", "row", problem, plan, {})
    write_artifact(payload, {"receipt": 2})
    with pytest.raises(ValueError, match="evidence changed"):
        audit._one(tmp_path, "source", "row", problem, plan, {})


def test_completion_nonce_rejected_before_output_use(tmp_path):
    completion = tmp_path / "completion.json"
    outputs = tmp_path / "outputs.json"
    write_artifact(
        completion,
        dict(status="complete", exit_code=0, step_id="14408.44", dispatch_nonce="foreign"),
    )
    write_artifact(outputs, {})
    with pytest.raises(ValueError, match="ownership"):
        audit.validate_attempt(
            dict(
                completion=audit.binding(completion),
                outputs=audit.binding(outputs),
                step_id="14408.44",
                nonce="authorized",
            )
        )


def test_native_solver_records_policy_receipts_without_teacher_access(tmp_path):
    from exact.repair.workers import bounded_call

    plan, problem = plan_and_input()
    outcome = bounded_call(
        audit.native_probe, problem, {}, 1000, 8192, str(tmp_path), timeout=60, memory_mb=8192
    )
    assert outcome.status == "complete", outcome.detail
    value = outcome.value
    assert value["status"] == "recorded" and value["routes"]
    assert value["semantic_benefit"] is None
    payload = audit.verify_binding(value["payload"])
    assert (
        payload["original_budgets"]["total_seconds"]
        == payload["effective_budgets"]["total_seconds"]
        == 60
    )
    assert payload["result"]["schema"].endswith("/v3")
    assert (tmp_path / "search-ledger.json").exists()
