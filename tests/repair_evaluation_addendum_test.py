"""Addenda keep original identities, outcomes, denominators and cumulative costs."""

import copy
import json
from pathlib import Path

import pytest

from exact.repair.api import write_artifact
from exact.repair.records import canonical_hash
from exact.repair.workers import CallResult
from tools.repair import evaluate_campaign as evaluation
from tools.repair import evaluation_addendum as addendum


@pytest.fixture
def prior(tmp_path, monkeypatch):
    source = tmp_path / "source.json"
    write_artifact(source, {"fixture": True})
    arms = [
        {
            "id": name,
            "protocol": evaluation.binding(source),
            "status": "unavailable",
            "reason": "repair_limit_exhausted",
        }
        for name in addendum.EPOCHS
    ] + [
        {"id": f"model-{i}", "protocol": evaluation.binding(source), "status": "available"}
        for i in range(4)
    ]
    rows = [
        {
            "kind": "generated_pool",
            "arm_id": a["id"],
            "case_id": str(c),
            "seconds": 300,
            "cpu_seconds": 600,
            "memory_mb": 8192,
        }
        for a in arms
        for c in range(4)
    ] + [
        {
            "kind": "circuit",
            "method": method,
            "case_id": str(c),
            "object_id": str(obj),
            "seconds": 60,
            "cpu_seconds": 120,
            "memory_mb": 8192,
        }
        for c in range(4)
        for obj in range(4)
        for method in evaluation.METHODS
    ]
    for row in rows:
        row["id"] = canonical_hash(row)
    schedule = {
        "schema": evaluation.SCHEMA,
        "source_preparation": evaluation.binding(source),
        "plan": evaluation.binding(source),
        "arms": arms,
        "rows": rows,
        "cases": [],
        "inner_seconds": 24900,
        "planned_model_arm_count": 6,
        "planned_primary_rows": 24,
        "scheduled_rows": 72,
        "seed": 13,
        "worker_seconds": 25200,
    }
    path = tmp_path / "original.json"
    write_artifact(path, schedule)
    calls = []

    def worker(_function, frozen, row, directory, **_options):
        calls.append(row["id"])
        artifact = Path(directory) / "result.json"
        write_artifact(
            artifact,
            {
                "schedule_hash": canonical_hash(frozen),
                "row_id": row["id"],
                "status": "verification_timeout",
                "logical_status": "UNKNOWN",
            },
        )
        return CallResult(
            "complete", evaluation.binding(artifact), resource_usage=(("cpu_seconds", 0.1),)
        )

    monkeypatch.setattr(evaluation, "bounded_call", worker)
    work = tmp_path / "work"
    report = evaluation.run(path, work)
    calls.clear()
    completion = tmp_path / "completion.json"
    write_artifact(completion, {"status": "complete", "exit_code": 0, "work": str(work)})
    outputs = tmp_path / "outputs.json"
    write_artifact(
        outputs,
        {
            str(p.relative_to(work)): evaluation.file_hash(p)
            for p in work.rglob("*")
            if p.is_file() and p.suffix != ".lock"
        },
    )
    auth = tmp_path / "authorization.json"
    write_artifact(
        auth, {"original_evaluation": evaluation.binding(work / "evaluation-report.json")}
    )
    current = copy.deepcopy(schedule)
    for arm in current["arms"][:2]:
        arm.update(status="available", selected_epoch=addendum.EPOCHS[arm["id"]])
    budget = json.loads((work / "evaluation-budget.json").read_text())
    current["inner_seconds"] -= budget["spent_seconds"]
    current["reuse"] = {
        "schedule": evaluation.binding(path),
        "report": evaluation.binding(work / "evaluation-report.json"),
        "completion": evaluation.binding(completion),
        "outputs": evaluation.binding(outputs),
        "authorization": evaluation.binding(auth),
        "stage_budget": evaluation.binding(work / "evaluation-budget.json"),
        "row_ids": sorted(r["id"] for r in rows if r.get("arm_id") not in addendum.EPOCHS),
    }
    return current, report, work, calls


def test_only_missing_rows_execute_original_receipts_and_unknowns_survive(prior, tmp_path):
    schedule, original, work, calls = prior
    original_files = {p: p.read_bytes() for p in work.rglob("*") if p.is_file()}
    path = tmp_path / "addendum.json"
    write_artifact(path, schedule)
    result = evaluation.run(path, work / "addendum-003")
    assert len(calls) == 8
    assert result["reused_rows"] == 64 and result["new_rows"] == 8
    assert result["scheduled"] == 72 and result["planned_primary_rows"] == 24
    assert result["statuses"] == {"verification_timeout": 72}
    reused = set(schedule["reuse"]["row_ids"])
    assert [r for r in result["rows"] if r["id"] in reused] == [
        r for r in original["rows"] if r["id"] in reused
    ]
    assert all(p.read_bytes() == data for p, data in original_files.items())
    evaluation.run(path, work / "addendum-003")
    assert len(calls) == 8


@pytest.mark.parametrize(
    "change",
    ["cases", "seed", "budget", "epoch", "protocol", "non_hgt", "rows", "reuse_denominator"],
)
def test_changed_dependencies_rejected_before_execution(prior, change):
    schedule, _, work, calls = prior
    if change == "cases":
        schedule["cases"] = [{"different": True}]
    elif change == "seed":
        schedule["seed"] = 37
    elif change == "budget":
        schedule["inner_seconds"] += 10
    elif change == "epoch":
        schedule["arms"][0]["selected_epoch"] = 10
    elif change == "protocol":
        schedule["arms"][0]["protocol"] = schedule["reuse"]["schedule"]
    elif change == "non_hgt":
        schedule["arms"][2]["selected_epoch"] = 7
    elif change == "rows":
        schedule["rows"].pop()
    else:
        schedule["reuse"]["row_ids"].pop()
    path = work.parent / "changed.json"
    write_artifact(path, schedule)
    with pytest.raises(ValueError):
        evaluation.run(path, work / "addendum-003")
    assert calls == []


def test_tampered_old_evidence_fails_before_new_work(prior):
    schedule, report, _, calls = prior
    artifact = next(r["artifact"] for r in report["rows"] if r["artifact"])
    Path(artifact["path"]).write_text("{}")
    with pytest.raises(ValueError, match="dependency changed"):
        addendum.checked_reuse(schedule)
    assert calls == []


def test_prior_completion_required(prior):
    schedule, _, _, _ = prior
    path = Path(schedule["reuse"]["completion"]["path"])
    value = json.loads(path.read_text())
    value.update(status="failed", exit_code=1)
    write_artifact(path, value)
    schedule["reuse"]["completion"] = evaluation.binding(path)
    with pytest.raises(ValueError, match="incomplete"):
        addendum.checked_reuse(schedule)


def test_preparation_compares_serialized_cases_without_querying_outcomes(
    prior, monkeypatch, tmp_path
):
    current, _, work, calls = prior
    original_path = Path(current["reuse"]["schedule"]["path"])
    # The real preparation returns tuple-bearing records, but writes JSON lists.
    serialized = copy.deepcopy(current)
    serialized.pop("reuse")
    serialized["inner_seconds"] = 24900
    original = json.loads(original_path.read_text())
    original["cases"] = [{"example_tuple": [1, 2]}]
    serialized["cases"] = [{"example_tuple": (1, 2)}]

    def prepare_models(_campaign, output):
        write_artifact(output, serialized)
        return serialized

    # Isolate metadata normalization; full provenance rejection is covered above.
    def check_reuse(schedule):
        assert schedule["cases"] == original["cases"]
        return {}

    monkeypatch.setattr(evaluation, "prepare", prepare_models)
    monkeypatch.setattr(addendum, "checked_reuse", check_reuse)
    monkeypatch.setattr(addendum, "check_receipt", lambda _: {"work": str(work)})
    monkeypatch.setattr(addendum, "output_binding", lambda _, name: evaluation.binding(work / name))
    campaign = tmp_path / "campaign"
    write_artifact(
        campaign / "supervisor/registry.json",
        {
            "runs": [
                {
                    "id": "xr21-t2-evaluation-002",
                    "completion_path": current["reuse"]["completion"]["path"],
                }
            ],
            "user_recovery_authorization": current["reuse"]["authorization"],
        },
    )
    path = tmp_path / "addendum.json"
    result = addendum.prepare(campaign, original_path, path)
    assert result["cases"] == original["cases"]
    assert calls == []
    with pytest.raises(FileExistsError):
        addendum.prepare(campaign, original_path, path)
