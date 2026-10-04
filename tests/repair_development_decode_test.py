"""Split admission, disposable weights and resumable development denominators."""
import copy
from pathlib import Path

import pytest
import torch

from exact.repair.api import write_artifact
from exact.repair.graph_schema import generic_graph_schema
from tools.repair import development_decode as decode
from tools.repair.historical_regression import binding

pytest_plugins = ["tests.repair_prepare_controls_test"]


def schedule_for(tmp_path):
    from tools.repair.corpus import generate_corpus

    case = generate_corpus(split_counts={"train": 0, "development": 1, "test": 0},
        siblings_per_parent=1, families=("overlap",), revision="v3")[0]
    observable = tmp_path / "observable.json"
    write_artifact(observable, case.problem.to_dict())
    cases = [dict(case_id=str(i), structural_parent=str(i // 2), split="development",
        control="corrupted" if i % 2 else "coherent", status="materialized",
        observable=binding(observable), input_hash=case.problem.content_hash,
        evaluator={"path": "must-never-open"}) for i in range(32)]
    release = tmp_path / "release.json"
    write_artifact(release, dict(rows=cases))
    declaration = tmp_path / "declaration.json"
    write_artifact(declaration, generic_graph_schema())
    protocol = tmp_path / "protocol.json"
    write_artifact(protocol, dict(resources=dict(case_wall_seconds=300, case_cpu_seconds=600,
        case_rss_mb=8192, generation_seconds=60), model=dict(graph_schema=generic_graph_schema())))
    weight = tmp_path / "bound-weights"
    weight.write_text("Only binding checked by schedule admission; real weights tested separately")
    arms = [dict(id=arm, status="available", kind="learned" if arm in decode.ARMS else "control",
        protocol=binding(protocol), model=binding(weight), diagnostic_only=True,
        fitting_steps=0, warm_start_allowed=False) for arm in decode.ALL_ARMS]
    return dict(schema=decode.SCHEMA, split="development", expected_cases=32, scheduled_rows=288,
        heldout_use=False, model_fitting=False, warm_start=False, supervision_admitted=False,
        release=binding(release), cases=cases, arms=arms, rows=decode.make_rows(cases),
        declaration=binding(declaration), program=binding(release), authorization=binding(release),
        corpus_completion=binding(release))


def test_admission_never_opens_evaluator_and_keeps_all_arms(tmp_path):
    schedule = schedule_for(tmp_path)
    decode.validate_schedule(schedule)
    assert len({r["id"] for r in schedule["rows"]}) == 288
    assert all({k: r[k] for k in decode.BUDGET} == decode.BUDGET for r in schedule["rows"])


@pytest.mark.parametrize("field,value", [("split", "test"), ("heldout_use", True),
    ("model_fitting", True), ("warm_start", True), ("supervision_admitted", True),
    ("expected_cases", 31), ("scheduled_rows", 287)])
def test_reject_boundary_before_opening_files(field, value):
    schedule = dict(schema=decode.SCHEMA, split="development", expected_cases=32, scheduled_rows=288,
        heldout_use=False, model_fitting=False, warm_start=False, supervision_admitted=False)
    schedule[field] = value
    with pytest.raises(ValueError, match="boundary"):
        decode.validate_schedule(schedule)


@pytest.mark.parametrize("mutation", ["split", "duplicate", "budget", "fitted", "warm_start", "pair"])
def test_reject_changed_schedule(tmp_path, mutation):
    schedule = schedule_for(tmp_path)
    if mutation == "split":
        schedule["cases"][0]["split"] = "test"
    elif mutation == "duplicate":
        schedule["cases"][1] = copy.deepcopy(schedule["cases"][0])
    elif mutation == "pair":
        schedule["cases"][1]["control"] = "coherent"
    elif mutation == "budget":
        schedule["rows"][0]["seconds"] = 301
    elif mutation == "fitted":
        schedule["arms"][0]["fitting_steps"] = 1
    else:
        schedule["arms"][0]["warm_start_allowed"] = True
    with pytest.raises(ValueError):
        decode.validate_schedule(schedule)


@pytest.mark.parametrize("arm", decode.ARMS)
def test_initialized_model_roundtrip_is_deterministic_full_schema_and_disposable(tmp_path, arm, monkeypatch):
    from exact.repair.model import RepairModel
    from exact.repair.pipeline import model_digest

    monkeypatch.setattr(torch.optim.AdamW, "step", lambda *a, **k: pytest.fail("No fitting"))
    settings = dict(seed=13, model=dict(hidden_dim=8, layers=1, heads=2, dropout=0.1,
        plan_risk=True, support_enabled=False, revision="v3"))
    first = decode.initialize_model(generic_graph_schema(), settings, arm, tmp_path / "first.pt")
    second = decode.initialize_model(generic_graph_schema(), settings, arm, tmp_path / "second.pt")
    assert first["model_hash"] == second["model_hash"]
    state = torch.load(first["model"]["path"], weights_only=True, map_location="cpu")
    assert state["diagnostic_only"] and state["fitting_steps"] == 0 and not state["warm_start_allowed"]
    model = RepairModel(state["metadata"], **state["config"])
    model.load_state_dict(state["state_dict"], strict=True)
    assert model_digest(model) == first["model_hash"]
    assert len(model.metadata[0]) == 13 and len(model.metadata[1]) == 553
    with pytest.raises(ValueError, match="overwrite"):
        decode.initialize_model(generic_graph_schema(), settings, arm, tmp_path / "first.pt")
    declaration = tmp_path / "schema.json"
    write_artifact(declaration, generic_graph_schema())
    schedule = dict(declaration=binding(declaration), model_initialization_seed=13,
                    arms=[dict(id=arm, **first)])
    assert len(decode.validate_weights(schedule)) == 1
    state["fitting_steps"] = 1
    torch.save(state, tmp_path / "forbidden.pt")
    schedule["arms"][0]["model"] = binding(tmp_path / "forbidden.pt")
    with pytest.raises(ValueError, match="provenance"):
        decode.validate_weights(schedule)


def test_diagnostic_requests_native_call_evidence(monkeypatch):
    calls = []
    monkeypatch.setattr(decode.fresh, "evaluate_row", lambda *a, **k: calls.append(k) or "bound")
    assert decode.evaluate_row({}, {}, "unused") == "bound"
    assert calls == [dict(record_native_labels=True)]


def test_timeout_retained_without_replay_and_guard_blocks_restart(tmp_path, monkeypatch):
    from exact.repair.workers import CallResult
    import exact.repair.workers as workers

    schedule = schedule_for(tmp_path)
    row = schedule["rows"][0]
    calls = []
    monkeypatch.setattr(workers, "bounded_call", lambda *a, **k:
        calls.append(k) or CallResult("timeout", detail="deadline", cleanup_complete=True))
    first = decode.fresh.one_row(schedule, row, tmp_path, "frozen", evaluator=decode.evaluate_row)
    assert first["status"] == "timeout"
    assert decode.fresh.one_row(schedule, row, tmp_path, "frozen", evaluator=decode.evaluate_row) == first
    assert len(calls) == 1
    other = schedule["rows"][1]
    write_artifact(tmp_path / "inflight" / (other["id"] + ".json"), {"owner": "14408.999"})
    with pytest.raises(RuntimeError, match="reconciliation"):
        decode.fresh.one_row(schedule, other, tmp_path, "frozen", evaluator=decode.evaluate_row)


def test_native_label_calls_are_persisted(prepared, tmp_path):
    from exact.repair.workers import bounded_call
    from tools.repair.prepare import load_preparation, case_to_dict
    from tools.repair.batch import read

    source, protocol = prepared
    cases, _, _ = load_preparation(source)
    case = next(c for c in cases if c.split == "development")
    write_artifact(tmp_path / "input.json", case.problem.to_dict())
    write_artifact(tmp_path / "evaluator.json", case_to_dict(case))
    item = dict(case_id=case.case_id, structural_parent=case.structural_parent,
        family=case.family, family_exposure="seen_family", status="materialized",
        observable=binding(tmp_path / "input.json"), evaluator=binding(tmp_path / "evaluator.json"))
    arms = [dict(id="uniform", kind="control", status="available", protocol=binding(protocol))]
    row = decode.fresh.make_rows([item], arms)[0]
    outcome = bounded_call(decode.evaluate_row, dict(cases=[item], arms=arms), row,
                           str(tmp_path / "run"), timeout=300, cpu_seconds=600, memory_mb=8192)
    write_artifact(tmp_path / "native-call-outcome.json", dict(status=outcome.status,
        detail=outcome.detail, cleanup_complete=outcome.cleanup_complete,
        resources=dict(outcome.resource_usage), event_journal=outcome.event_journal))
    assert outcome.status == "complete", outcome.detail
    result = read(tmp_path / "run/result.json")
    assert result["logical_status"] == "VERIFIED_FEASIBLE"
    call = read(tmp_path / "run/native-label/call.json")
    assert call["status"] == "complete" and call["cleanup_complete"]
    assert call["assignment"] is not None and call["deadline_seconds"] > 0
    assert (tmp_path / "run/native-label/check-0.json").exists()
