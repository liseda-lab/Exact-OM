"""Held-out evaluation freezes choices, keeps missing arms and resumes bound rows."""

import json
from pathlib import Path

import pytest

from exact.repair.api import write_artifact
from exact.repair.records import canonical_hash
from exact.repair.workers import CallResult, bounded_call
from tools.repair import evaluate_campaign as evaluation
from tools.repair.prepare import load_preparation

pytest_plugins = ["tests.repair_prepare_controls_test"]


def campaign_fixture(prepared, tmp_path, monkeypatch):
    source, protocol = prepared
    campaign = tmp_path / "campaign"
    arms, runs, terminal = [], [], []
    for index in range(6):
        name = f"model-{index}"
        completion = campaign / f"{name}-completion.json"
        write_artifact(completion, {"status": "complete", "exit_code": 0})
        arms.append(
            {"id": name, "model": str(campaign / name / "model.pt"), "protocol": str(protocol)}
        )
        run = {"id": name + "-001", "completion_path": str(completion)}
        runs.append(run)
        if index < 2:
            accounting = campaign / f"{name}-terminal.json"
            write_artifact(accounting, {"status": "unavailable"})
            terminal.append(
                {
                    "logical_id": name,
                    "resolved_run_id": run["id"],
                    "accounting_path": str(accounting),
                    "completion_path": str(completion),
                    "accounting_sha256": evaluation.file_hash(accounting),
                    "completion_sha256": evaluation.file_hash(completion),
                }
            )
    plan = {
        "labels": {"source_preparation": str(source)},
        "arms": arms,
        "case_counts": {"test": 1},
        "budgets": {"evaluation": 25200},
    }
    write_artifact(campaign / "plan.json", plan)
    write_artifact(campaign / "supervisor/registry.json", {"runs": runs, "terminal_arms": terminal})
    monkeypatch.setattr(evaluation, "selected_model", lambda *_: {"status": "available"})
    schedule = campaign / "schedule.json"
    return schedule, evaluation.prepare(campaign, schedule)


def test_preparation_never_queries_tests_preserves_missing_arms(prepared, tmp_path, monkeypatch):
    import tools.repair.train as train

    monkeypatch.setattr(train, "_assignment_label", lambda *_a, **_k: pytest.fail("sealed tests"))
    path, schedule = campaign_fixture(prepared, tmp_path, monkeypatch)
    assert schedule["planned_model_arm_count"] == 6
    assert schedule["planned_primary_rows"] == 6
    assert sum(arm["status"] == "unavailable" for arm in schedule["arms"]) == 2
    assert schedule["test_outcomes_opened"] is False
    assert schedule["original_label_seconds"] == 19
    assert all(case["case"]["split"] == "test" for case in schedule["cases"])
    assert {row["method"] for row in schedule["rows"] if row["kind"] == "circuit"} == set(
        evaluation.METHODS
    )
    with pytest.raises(FileExistsError):
        evaluation.prepare(path.parent, path)


def test_bound_results_resume_without_reevaluation_and_keep_unknowns(
    prepared, tmp_path, monkeypatch
):
    path, schedule = campaign_fixture(prepared, tmp_path, monkeypatch)
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
    output = tmp_path / "results"
    report = evaluation.run(path, output)
    assert len(report["rows"]) == schedule["scheduled_rows"]
    assert report["statuses"]["unavailable"] == 2
    assert report["statuses"]["verification_timeout"] == len(calls)
    count = len(calls)
    assert evaluation.run(path, output)["rows"] == report["rows"]
    assert len(calls) == count
    artifact = next(row["artifact"] for row in report["rows"] if row["artifact"])
    Path(artifact["path"]).write_text("{}")
    with pytest.raises(ValueError, match="dependency changed"):
        evaluation.run(path, output)


def test_changed_input_rejected_before_test_execution(prepared, tmp_path, monkeypatch):
    path, schedule = campaign_fixture(prepared, tmp_path, monkeypatch)
    Path(schedule["source_preparation"]["path"]).write_text("{}")
    monkeypatch.setattr(evaluation, "bounded_call", lambda *_a, **_k: pytest.fail("changed inputs"))
    with pytest.raises(ValueError, match="dependency changed"):
        evaluation.run(path, tmp_path / "results")


@pytest.mark.parametrize("method", evaluation.METHODS)
def test_native_circuit_decoder_keeps_language_and_records_failures(prepared, tmp_path, method):
    source, protocol_path = prepared
    cases, _, _ = load_preparation(source)
    case = next(case for case in cases if case.split == "development")
    protocol = json.loads(protocol_path.read_text())
    row = {"method": method, "object_id": case.problem.objects[0].object_id, "seconds": 60}
    result = bounded_call(
        evaluation.circuit_row,
        case,
        row,
        protocol,
        tmp_path / method,
        timeout=65,
        cpu_seconds=120,
        memory_mb=8192,
    )
    assert result.status == "complete", result.detail
    payload = result.value
    assert payload["status"] in {"complete", "partial"}
    assert payload["language_hash"] and payload["retrieval_hash"]
    assert len(payload["sampled_ids"]) == 32
    assert payload["duplicate_draws"] + payload["unique_sampled"] == 32
    assert payload["logical_status"] == "NOT_EVALUATED"
    if method == "semantic_enumeration_decoder":
        assert set(payload["sampled_ids"]) <= set(payload["support_ids"])


def test_native_generated_row_uses_v3_preparation_and_independent_labels(prepared, tmp_path):
    import torch

    from exact.repair.graph import EffectivePreparation
    from exact.repair.model import RepairModel
    from exact.repair.pipeline import model_digest
    from exact.repair.records import read_record
    from exact.repair.retrieval import retrieve_vocabulary

    source, protocol_path = prepared
    cases, _, _ = load_preparation(source)
    case = next(case for case in cases if case.split == "development")
    protocol = json.loads(protocol_path.read_text())
    options = evaluation.generation_options(protocol, tmp_path / "cache")
    preparation = EffectivePreparation(
        options["max_graph_nodes"],
        options["max_graph_edges"],
        options["max_explanations"],
        options["max_text_tokens"],
        options["pair_factor_limit_per_object"],
        options["pair_max_pairs"],
        options["pair_max_factors"],
        options["retrieval_config"],
        "v3",
    )
    graph = preparation.graph(
        case.problem, retrieve_vocabulary(case.problem, config=options["retrieval_config"])
    )
    torch.manual_seed(13)
    model = RepairModel(
        graph.metadata,
        hidden_dim=8,
        heads=2,
        layers=0,
        dropout=0,
        encoder="none",
        revision="v3",
        pairwise=False,
    )
    path = tmp_path / "model.pt"
    torch.save(
        {
            "model_schema": "exact-repair/model/v3",
            "metadata": model.metadata,
            "config": model.config,
            "state_dict": model.state_dict(),
        },
        path,
    )
    arm = {"model": evaluation.binding(path), "model_hash": model_digest(model)}
    directory = tmp_path / "native"
    directory.mkdir()
    outcome = bounded_call(
        evaluation.generated_row,
        case,
        arm,
        protocol,
        directory,
        timeout=300,
        cpu_seconds=600,
        memory_mb=8192,
    )
    assert outcome.status == "complete", outcome.detail
    assert outcome.value["status"] == "evaluated", outcome.value
    pool = json.loads((directory / "pool.json").read_text())
    problem = read_record(pool["input"])
    assert type(problem).__name__ == "RepairInputV3"
    assert problem.graph_identity
    assert pool["model_hash"] == arm["model_hash"]
    assert (directory / "repair.json").exists()
    if outcome.value["logical_status"] == "VERIFIED_FEASIBLE":
        assert outcome.value["semantic_status"] == "known", outcome.value
        assert (directory / "selected-label.json").exists()


def test_worker_protocol_roundtrip_preserves_schema_alias(prepared, tmp_path, monkeypatch):
    from tools.repair.prepare import case_to_dict

    source, protocol = prepared
    cases, _, _ = load_preparation(source)
    case = next(c for c in cases if c.split == "development")
    schedule = {
        "cases": [case_to_dict(case)],
        "arms": [{"id": "fixture", "protocol": evaluation.binding(protocol)}],
    }
    row = {"kind": "circuit", "id": "fixture", "case_id": case.case_id}

    def check(_case, _row, value, _directory):
        assert value["schema"] == "exact-repair/protocol/v3"
        assert "schema_id" not in value
        evaluation.generation_options(value, tmp_path / "cache")
        return {"status": "complete"}

    monkeypatch.setattr(evaluation, "circuit_row", check)
    artifact = evaluation.evaluate_row(schedule, row, tmp_path / "worker")
    assert evaluation.check_binding(artifact).exists()


def test_protocol_alias_error_retry_keeps_error_evidence_and_budget(
    prepared, tmp_path, monkeypatch
):
    path, schedule = campaign_fixture(prepared, tmp_path, monkeypatch)
    calls = []

    def fail(_function, _schedule, row, _directory, **_options):
        calls.append(row["id"])
        return CallResult(
            "error",
            detail="2 validation errors for RepairProtocolV3 schema_id Extra inputs are not permitted",
            resource_usage=(("cpu_seconds", 1.0),),
        )

    monkeypatch.setattr(evaluation, "bounded_call", fail)
    output = tmp_path / "results"
    first = evaluation.run(path, output)
    before = len(calls)
    evaluation.run(path, output)
    assert len(calls) == 2 * before
    available = next(row for row in first["rows"] if row["status"] == "error")
    directory = output / "rows" / available["id"]
    assert json.loads((directory / "protocol-alias-error-001.json").read_text()) == available
    budget = json.loads((directory / "budget.json").read_text())
    assert len(budget["attempts"]) == 2
    assert budget["spent_cpu_seconds"] == 2.0
    assert budget["limit_seconds"] == available["seconds"]

    with pytest.raises(ValueError, match="exceeds its one retry"):
        evaluation.run(path, output)
