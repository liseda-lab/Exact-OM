"""Closed TEST metadata, exact shared identities, real native fixtures and no replay."""

import copy
import json
import operator
import time

import pyowl_core as owl
import pytest

from exact.repair.api import write_artifact
from exact.repair.records import canonical_hash
from tests.repair_primary_runtime_test import evaluation_inputs
from tools.repair.historical_regression import binding
from tools.repair.prepare_primary_runtime import evaluation_schedule
from tools.repair.primary_evaluation import (
    SCHEMA,
    compile_rows,
    evaluate_input,
    execute_rows,
    remaining,
)


def schedule():
    generated, conference = evaluation_inputs()
    return evaluation_schedule(
        generated, conference, dict(path="/not-opened/protocol", sha256="sealed"), {}
    )


def public_fixture(tmp_path):
    from exact.repair.candidates import mapping_candidates
    from exact.repair.records import PolicyV3, RepairInputV3, RevisionObjectV3
    from tools.repair.corrective_study import PUBLIC_INPUT_SCHEMA

    s, t = (owl.Class(owl.IRI("urn:primary-fixture:" + x)) for x in ("s", "t"))
    pool = mapping_candidates("m", s, t, "<")
    obj = RevisionObjectV3("m", "mapping", pool[0].axioms, pool, source_entity=s, target_entity=t)
    problem = RepairInputV3(
        (owl.DisjointClasses(owl.CanonicalSet((s, t))),), (obj,), PolicyV3((s, t))
    )
    case = dict(
        case_id="fixture",
        structural_parent="fixture-parent",
        family="overlap",
        control="corrupted",
        split="development",
        input_hash=problem.content_hash,
    )
    public = dict(schema=PUBLIC_INPUT_SCHEMA, metadata=case, input=problem.to_dict())
    protocol = dict(
        resources=dict(case_wall_seconds=30, verification_seconds=5, case_rss_mb=4096),
        objective=dict(edit_weights={"mapping_deletion": 0.1}, integer_scale=1000),
    )
    write_artifact(tmp_path / "public.json", public)
    write_artifact(tmp_path / "protocol.json", protocol)
    return case, binding(tmp_path / "public.json"), binding(tmp_path / "protocol.json")


def fixture_manifest(tmp_path):
    case, public, protocol = public_fixture(tmp_path)
    row = dict(
        id="native",
        case_id="fixture",
        protocol=protocol,
        case_seconds=40,
        settings={},
        adapter=dict(operation="native_deletion", model=None, pool_dependency=None),
    )
    write_artifact(
        tmp_path / "ledger.json", dict(stages=dict(evaluation=dict(started_epoch=time.time())))
    )
    return dict(
        schema=SCHEMA,
        scope="fixture",
        execution_authorized=True,
        rows=[row],
        cases={"fixture": case},
        public_inputs={"fixture": public},
        models={},
        memory_mb=4096,
        device="cpu",
        ledger_snapshot=binding(tmp_path / "ledger.json"),
        deadline_epoch=time.time() + 120,
    )


def test_compile_retains_all_references_and_opens_no_payloads(monkeypatch):
    value = schedule()

    def forbidden(*args, **kwargs):
        raise AssertionError("compilation opened a TEST payload")

    monkeypatch.setattr("pathlib.Path.open", forbidden)
    rows = compile_rows(value)
    assert len(rows) == 1535
    ids = {r["id"] for r in rows}
    assert all(c["left"] in ids and c["right"] in ids for c in value["comparisons"])
    assert len(value["comparisons"]) == 518
    assert not any(r["experiment"].startswith("E6") for r in rows)
    assert all(r["adapter"]["pool_dependency"] in ids for r in rows if r["pool"])
    assert sum(r["adapter"]["operation"] == "pool" for r in rows) == 71
    assert sum(r["adapter"]["operation"] == "native_deletion" for r in rows) == 71


def test_e4_refuses_a_different_objective_even_with_rehashed_row():
    value = schedule()
    row = next(r for r in value["expected_runs"] if r["experiment"] == "E4")
    old = row["id"]
    row["settings"]["objective_and_cuts"] = "changed"
    row["id"] = canonical_hash({k: v for k, v in row.items() if k not in {"id", "status"}})
    for item in [*value["comparisons"], *value["semantic_slots"]]:
        for key in ("left", "right"):
            if item[key] == old:
                item[key] = row["id"]
    with pytest.raises(ValueError, match="E4 changed"):
        compile_rows(value)


def test_disabled_and_pre_freeze_entrypoints_open_no_inputs(tmp_path):
    with pytest.raises(ValueError, match="not admitted"):
        execute_rows(dict(schema=SCHEMA, execution_authorized=False), tmp_path / "disabled")
    manifest = dict(
        schema=SCHEMA,
        execution_authorized=True,
        scope="primary_test",
        not_before_epoch=time.time() + 100,
    )
    with pytest.raises(ValueError, match="TEST remains closed"):
        execute_rows(manifest, tmp_path / "closed")
    assert not (tmp_path / "closed").exists()


def test_native_row_executes_and_resumes_exact_receipt_without_replay(tmp_path, monkeypatch):
    manifest = fixture_manifest(tmp_path)
    # The missing row stays in the denominator without any callback.
    manifest["rows"].append(dict(manifest["rows"][0], id="missing", case_id="absent"))
    monkeypatch.setenv("EXACT_REPAIR_DEADLINE_EPOCH", str(time.time() + 100))
    first = execute_rows(manifest, tmp_path / "run")
    assert first["rows"][0]["status"] == "complete", first
    assert first["rows"][0]["result"]["logical_status"] == "VERIFIED_FEASIBLE", first
    assert first["rows"][1]["status"] == "unavailable_public_input"
    receipt = (tmp_path / "run/rows/native/completion.json").read_bytes()
    second = execute_rows(manifest, tmp_path / "run", evaluator=operator.add)
    assert second == first
    assert receipt == (tmp_path / "run/rows/native/completion.json").read_bytes()
    output = first["rows"][0]["artifacts"][0]
    from pathlib import Path

    Path(output["path"]).write_text("tampered")
    with pytest.raises(ValueError, match="checksum changed"):
        execute_rows(manifest, tmp_path / "run")


def test_interrupted_case_cannot_renew_its_budget(tmp_path):
    manifest = fixture_manifest(tmp_path)
    directory = tmp_path / "run/rows/native"
    write_artifact(
        directory / "started.json",
        dict(
            identity=canonical_hash((canonical_hash(manifest), manifest["rows"][0])),
            deadline_epoch=time.time() - 10,
            reserved_seconds=40,
        ),
    )
    with pytest.raises(RuntimeError, match="no renewed case budget"):
        execute_rows(manifest, tmp_path / "run")
    report = json.loads((tmp_path / "run/report.json").read_text())
    assert report["expected_rows"] == 1
    assert report["rows"][0]["status"] == "interrupted_unknown"
    assert report["charged_seconds"] == 40
    assert (directory / "started.json").exists()
    assert not (directory / "completion.json").exists()


def test_terminal_timeout_and_shared_pool_failure_are_not_replayed(tmp_path, monkeypatch):
    from exact.repair.workers import CallResult

    manifest = fixture_manifest(tmp_path)
    row = manifest["rows"][0]
    row["adapter"]["operation"] = "pool"
    consumer = copy.deepcopy(row)
    consumer.update(id="consumer", pool="native")
    consumer["adapter"]["pool_dependency"] = "native"
    manifest["rows"].append(consumer)
    calls = []

    def timed_out(*args, **kwargs):
        calls.append(kwargs["timeout"])
        return CallResult("timeout", detail="fixture timeout")

    monkeypatch.setattr("exact.repair.workers.bounded_call", timed_out)
    first = execute_rows(manifest, tmp_path / "run")
    second = execute_rows(manifest, tmp_path / "run")
    assert first == second and len(calls) == 1
    assert first["rows"][0]["status"] == "timeout"
    assert first["rows"][1]["status"] == "unavailable_shared_pool"
    assert first["rows"][1]["shared_cost_reference"] == "native"


def test_deadline_preserves_unattempted_rows_and_reserve(tmp_path, monkeypatch):
    manifest = fixture_manifest(tmp_path)
    monkeypatch.setenv("EXACT_REPAIR_DEADLINE_EPOCH", str(time.time() + 0.1))
    report = execute_rows(manifest, tmp_path / "run")
    assert report["rows"][0]["status"] == "not_attempted"
    assert not (tmp_path / "run/rows/native/started.json").exists()
    with pytest.raises(TimeoutError):
        remaining(time.time() - 1)


def test_nested_worker_inherits_expired_campaign_deadline(monkeypatch):
    from exact.repair.workers import bounded_call

    monkeypatch.setenv("EXACT_REPAIR_DEADLINE_EPOCH", str(time.time() - 1))
    result = bounded_call(operator.add, 1, 2, timeout=90)
    assert result.status == "timeout" and result.cleanup_complete


def test_observable_queries_are_not_replaced_by_reference_truth(tmp_path, monkeypatch):
    case, public, protocol = public_fixture(tmp_path)
    monkeypatch.setenv("EXACT_REPAIR_DEADLINE_EPOCH", str(time.time() + 30))
    row = dict(adapter=dict(operation="complete_plan_observable_query"), settings={})
    result = evaluate_input(
        row, case, public, protocol, None, None, tmp_path / "output", time.time() + 30, "cpu"
    )
    assert result["status"] == "unavailable_observable_queries"


def test_endpoint_projection_preserves_missing_rows_and_semantic_reserves():
    from tools.repair.common_training import development_schedule
    from tools.repair.prepare_primary_evaluation import endpoint_projection

    declarations = [
        dict(
            case_id=f"d{i}",
            family=f"f{i//4}",
            control="coherent" if i % 2 else "corrupted",
            structural_parent=f"p{i//2}",
            split="development",
        )
        for i in range(32)
    ]
    dev = development_schedule(declarations)
    report = dict(
        cases=[dict(cache_available=i < 126, optimizer_updates=int(i < 99)) for i in range(128)],
        update_seconds=[3.0] * 98 + [18.0],
    )
    limits = {
        f"model-{c}-{s}": dict(gpu_seconds=10800)
        for c in ("symbolic", "symbolic_plus_llm")
        for s in (13, 37, 73)
    }
    limits.update(
        primary_training=dict(gpu_seconds=64800),
        learning=dict(deadline_epoch=999999, elapsed_seconds=237600),
    )
    ledger = dict(
        stage_limits=limits,
        stages=dict(
            primary_training=dict(cumulative=dict(gpu_seconds=650)),
            learning=dict(started_epoch=100),
        ),
    )
    prior = copy.deepcopy(ledger)
    result = endpoint_projection(report, 650, ledger, dev, now=1000)
    assert ledger == prior
    assert result["unmeasured_cached_cases"] == 27 and result["missing_cache_cases"] == 2
    assert result["unique_semantic_slots"] == 64 and result["independent_swapped_calls"] == 14
    assert len(result["development_rows"]) == 384
    assert result["candidate_epochs"] is None and not result["endpoint_admitted"]
    assert all(
        a["annotation_reserve_seconds"] == b["annotation_reserve_seconds"]
        for a, b in zip(dev["rows"], result["development_rows"])
    )
    assert result["primary_remaining_gpu_seconds"] == 64150


def inherited_deadline_worker():
    import os
    import time

    return float(os.environ["EXACT_REPAIR_DEADLINE_EPOCH"]) - time.time()


def test_nested_worker_receives_enclosing_stage_deadline(monkeypatch):
    import os

    from exact.repair.workers import bounded_call

    deadline = time.time() + 90
    monkeypatch.setenv("EXACT_REPAIR_DEADLINE_EPOCH", str(deadline))
    result = bounded_call(inherited_deadline_worker, timeout=8)
    assert result.status == "complete", result
    assert 0 < result.value <= 8
    assert float(os.environ["EXACT_REPAIR_DEADLINE_EPOCH"]) == deadline


def test_failure_stops_schedule_and_retains_all_remaining_rows(tmp_path, monkeypatch):
    from exact.repair.workers import CallResult

    manifest = fixture_manifest(tmp_path)
    manifest["rows"].append(dict(manifest["rows"][0], id="later"))
    calls = []

    def failed(*args, **kwargs):
        calls.append(1)
        return CallResult("error", detail="confirmed fixture defect")

    monkeypatch.setattr("exact.repair.workers.bounded_call", failed)
    with pytest.raises(RuntimeError, match="Implementation failure"):
        execute_rows(manifest, tmp_path / "run")
    report = json.loads((tmp_path / "run/report.json").read_text())
    assert [r["status"] for r in report["rows"]] == ["error", "not_attempted"]
    with pytest.raises(RuntimeError, match="specific recovery"):
        execute_rows(manifest, tmp_path / "run")
    assert len(calls) == 1


@pytest.mark.parametrize("risk", ["on", "off"])
def test_e4_runtime_keeps_pool_objective_and_solver_settings(tmp_path, monkeypatch, risk):
    from types import SimpleNamespace

    from exact.repair.pipeline import FrozenNeuralRound
    from exact.repair.records import make_objective, read_record
    from tools.repair.shared_release import bound

    case, public, protocol_ref = public_fixture(tmp_path)
    monkeypatch.setenv("EXACT_REPAIR_DEADLINE_EPOCH", str(time.time() + 30))
    problem = read_record(bound(public)["input"])
    pool_path = tmp_path / "pool.json"
    write_artifact(pool_path, problem.to_dict())
    protocol = bound(protocol_ref)
    protocol["selection"] = dict(shortlist_size=4, utility_window=123, construction_seconds=1)
    write_artifact(tmp_path / "matched-protocol.json", protocol)
    objective = make_objective(problem.objects, profile=(("mapping_deletion", 0.1),))
    risk_scorer = object()
    frozen = FrozenNeuralRound(problem, objective, "graph", "model", (), risk_scorer)
    captured = []

    def score(*args, **kwargs):
        assert kwargs["device"] == "cpu"
        return frozen

    def solve(actual, **kwargs):
        captured.append((actual, kwargs))
        assert actual.objective is objective
        assert canonical_hash(actual.problem.objects) == canonical_hash(problem.objects)
        assert actual.risk_scorer is (risk_scorer if risk == "on" else None)
        assert kwargs["shortlist_size"] == 4 and kwargs["utility_window"] == 123
        assert kwargs["preserve_verified_input"] is False
        return SimpleNamespace(
            logical_status="UNKNOWN",
            search_status="RESOURCE_LIMIT",
            verification=None,
            first_verified_seconds=None,
            checks=0,
            selected=(),
            to_dict=lambda: dict(status="fixture"),
        )

    monkeypatch.setattr("tools.repair.common_inventory.freeze_inventory", score)
    monkeypatch.setattr("exact.repair.pipeline.repair_neural_round", solve)
    row = dict(adapter=dict(operation="fixed_learned"), arm="risk_off", settings=dict(risk=risk))
    result = evaluate_input(
        row,
        case,
        public,
        binding(tmp_path / "matched-protocol.json"),
        dict(path="never-read", sha256="frozen", model_hash="model"),
        binding(pool_path),
        tmp_path / "result",
        time.time() + 30,
        "cpu",
    )
    assert len(captured) == 1 and result["logical_status"] == "UNKNOWN"


def test_frozen_checkpoint_sets_eval_and_declared_device(tmp_path, monkeypatch):
    import io

    import torch

    from exact.repair.graph_schema import declared_metadata, generic_graph_schema
    from exact.repair.model import RepairModel
    from exact.repair.pipeline import _freeze_worker

    model = RepairModel(
        declared_metadata(generic_graph_schema()),
        encoder="none",
        hidden_dim=8,
        heads=2,
        layers=0,
        dropout=0.5,
        revision="v3",
    )
    buffer = io.BytesIO()
    torch.save(
        dict(
            metadata=model.metadata,
            config=model.config,
            model_schema="exact-repair/model/v3",
            state_dict=model.state_dict(),
        ),
        buffer,
    )

    def frozen(problem, loaded, **options):
        assert not loaded.training
        assert loaded.empty_bundle.device.type == "cpu"
        assert "device" not in options
        return "checked"

    monkeypatch.setattr("exact.repair.pipeline.freeze_neural_round", frozen)
    assert _freeze_worker(None, buffer.getvalue(), dict(device="cpu")) == "checked"
