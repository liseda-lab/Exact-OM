"""Exercise actual DEV entry points and require operational repair evidence."""

from types import SimpleNamespace

import pytest

from exact.repair.workers import CallResult
from tools.repair import train


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_dev_inherits_model_device_retains_failure_stages_and_explicit_budget(monkeypatch, device):
    calls = []
    stages = ({"phase": "selection_verification", "status": "UNKNOWN", "wall_seconds": 42},)

    def staged(*args, **options):
        calls.append(options)
        return CallResult(
            "complete",
            SimpleNamespace(
                frozen=None, stages=stages, generation_status="UNRESOLVED_NO_INCUMBENT"
            ),
        )

    monkeypatch.setattr(train, "_staged_training_model", staged)
    model = SimpleNamespace(parameters=lambda: iter([SimpleNamespace(device=device)]))
    case = SimpleNamespace(
        problem=None, probes=(), structural_parent="fixed-parent", final_candidate_removals=()
    )
    result = train.generated_development(
        model,
        case,
        None,
        seconds=240,
        elementary_seconds=90,
        execution_schedule="staged_verified_repair",
    )
    assert calls[0]["device"] == result["device"] == device
    assert calls[0]["seconds"] == 210  # explicit 30s semantic-label reserve
    assert calls[0]["elementary_seconds"] == 90
    assert result["stages"] == stages
    assert result["status"] == "staged_UNRESOLVED_NO_INCUMBENT"
    assert result["cleanup_complete"]


def test_staged_reuses_bound_snapshot_without_serializing_again(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(train, "save_training_state", lambda *a: pytest.fail("reserialized"))
    monkeypatch.setattr(train, "bounded_call", lambda *a, **k: calls.append(k))
    train._staged_training_model(
        None,
        None,
        seconds=240,
        elementary_seconds=90,
        selection_options={},
        device="cuda",
        model_snapshot={"path": str(tmp_path / "weights.pt"), "sha256": "a" * 64},
    )
    assert calls[0]["checkpoint_sha256"] == "a" * 64
    assert calls[0]["generation_options"]["device"] == "cuda"
    assert calls[0]["elementary_seconds"] == 90


def test_staged_diagnoses_once_and_persists_progress(monkeypatch, tmp_path):
    from tests.repair_october_core_corrections_test import cls
    import pyowl_core as owl
    from exact.repair import kernel, pipeline
    from exact.repair.candidates import mapping_candidates
    from exact.repair.records import BudgetsV2, PolicyV2, RepairInputV2, RevisionObjectV2

    s, t = map(cls, ("S", "T"))
    pool = mapping_candidates("m", s, t, "<")
    obj = RevisionObjectV2("m", "mapping", pool[0].axioms, pool, source_entity=s, target_entity=t)
    problem = RepairInputV2((), (obj,), PolicyV2((s, t)), budgets=BudgetsV2(total_seconds=30))
    original = kernel.repair
    calls = []

    def checked(problem, objective, **options):
        calls.append(options)
        assert options["diagnose"] is False
        assert options["baseline_evidence"].identity == kernel.baseline_identity(problem)
        return original(problem, objective, **options)

    monkeypatch.setattr(kernel, "repair", checked)
    report = tmp_path / "stages.json"
    result = pipeline.staged_verified_repair(
        problem,
        checkpoint_path="unused",
        total_seconds=30,
        elementary_seconds=12,
        stage_report_path=str(report),
    )
    assert result.result.verification.authorizes
    assert len([s for s in result.stages if s["phase"] == "baseline"]) == 1
    import json

    assert json.loads(report.read_text())["stages"][-1]["phase"] == "initial_coherence"
    assert len(calls) == 1


def test_technical_gate_requires_corrupted_repairs_and_clean_lifecycle():
    from tools.repair.validation_recovery import qualification

    coherent = dict(
        case_id="ok",
        family="domain",
        control="coherent",
        result=dict(status="generated", decoded=dict(logical_status="VERIFIED_FEASIBLE")),
    )
    corrupted = {**coherent, "case_id": "repair", "control": "corrupted"}
    assert not qualification([coherent], ["domain"])["passed"]
    assert qualification([coherent, corrupted], ["domain"])["passed"]
    broken = {**corrupted, "result": dict(status="staged_cleanup_incomplete")}
    assert not qualification([coherent, corrupted, broken], ["domain"])["passed"]


def test_supervisor_detects_no_selected_models_and_waits_for_registered_recovery(tmp_path):
    import hashlib
    import json
    from exact.experiments.completion import readiness_incidents

    path = tmp_path / "review.json"
    path.write_text(json.dumps(dict(training_integrity_passed=True, selected_models_available=0)))
    gate = dict(
        id="validation",
        report=dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
        required_values=dict(training_integrity_passed=True, selected_models_available=6),
        recovery_run_ids=["repair"],
    )
    registry = dict(readiness_gates=[gate], pending_batches=[])
    observation = dict(findings=[])
    incident = readiness_incidents(registry, observation, now=100)[0]
    assert incident["id"] == "readiness-validation"
    registry["pending_batches"] = [dict(id="repair")]
    assert readiness_incidents(registry, observation, now=100) == []
    registry["pending_batches"] = [dict(id="repair", not_before_epoch=200)]
    assert readiness_incidents(registry, observation, now=100)[0]["id"] == incident["id"]
    registry["pending_batches"] = []
    path.write_text("{}")
    assert "integrity changed" in readiness_incidents(registry, observation, now=100)[0]["reason"]
