"""Exercise actual DEV entry points and require operational repair evidence."""

from types import SimpleNamespace

import pytest

from exact.repair.workers import CallResult
from tools.repair import train


def test_recovery_uses_actual_training_entrypoint_defaults(monkeypatch):
    from tools.repair.validation_recovery import runtime_configuration

    monkeypatch.setattr(
        train, "_protocol_arguments", lambda _: {"encoder": "hgt", "hidden_dim": 128}
    )
    monkeypatch.setattr(train, "_protocol_profile", lambda _: ())
    config = runtime_configuration({})
    assert config["proposal_arm"] == "grammar_mixture"
    assert config["plan_rating_aggregation"] is None
    assert config["hidden_dim"] == 128


def test_all_evaluated_snapshots_survive_even_without_a_selected_model(tmp_path):
    import torch
    from exact.repair.model import RepairModel
    from exact.repair.graph_schema import declared_metadata, generic_graph_schema

    model = RepairModel(
        declared_metadata(generic_graph_schema()), encoder="none", hidden_dim=8, heads=2, layers=0
    )
    first = train.development_snapshot(model, tmp_path, epoch=5, optimizer_updates=630)
    assert train.development_snapshot(model, tmp_path, epoch=5, optimizer_updates=630) == first
    with torch.no_grad():
        next(model.parameters()).add_(0.01)
    with pytest.raises(ValueError, match="weights changed"):
        train.development_snapshot(model, tmp_path, epoch=5, optimizer_updates=630)
    second = train.development_snapshot(model, tmp_path, epoch=6, optimizer_updates=756)
    assert first["sha256"] != second["sha256"]
    assert (tmp_path / "epoch-5.json").exists() and (tmp_path / "epoch-6.json").exists()


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


@pytest.mark.parametrize("weak_benefit,selected", [(0.8, True), (None, False)])
def test_deferred_annotations_keep_denominators_and_never_select_missing_labels(
    tmp_path, monkeypatch, weak_benefit, selected
):
    import time
    from tools.repair import validation_recovery as recovery, corrective_semantics, prepare
    from tools.repair.shared_release import immutable
    from tools.repair.historical_regression import binding
    from exact.repair import protocol, semantic_fidelity
    from exact.repair.records import canonical_hash

    def save(name, value):
        path = tmp_path / (name + ".json")
        immutable(path, value)
        return binding(path)

    config = dict(
        mixture_symbolic_weight=0.5,
        plan_rating_aggregation=None,
        development_use_policy="development_selection/v1",
        minimum_generated_coverage=1.0,
        quality_uncertainty_z=1.96,
        missing_label_fallback="stop",
    )
    monkeypatch.setattr(recovery, "runtime_configuration", lambda p: config)
    monkeypatch.setattr(protocol, "load_protocol_v3", lambda *a, **k: {})
    monkeypatch.setattr(protocol, "training_projection_v3", lambda p: p)
    monkeypatch.setattr(
        prepare, "case_from_dict", lambda p: SimpleNamespace(case_id="c0", split="development")
    )
    monkeypatch.setattr(semantic_fidelity, "read_fidelity_training_artifact", lambda *a: {})
    monkeypatch.setattr(
        semantic_fidelity,
        "offline_plan_label",
        lambda *a, **k: SimpleNamespace(benefit=weak_benefit),
    )
    calls = []
    labels = save("labels", {})
    monkeypatch.setattr(
        corrective_semantics, "annotate_decoded", lambda *a, **k: calls.append(a) or labels["path"]
    )
    monkeypatch.setenv("EXACT_REPAIR_DEADLINE_EPOCH", str(time.time() + 3600))
    slot = dict(seed=13, supervision_condition="symbolic", epoch=6, case_id="c0")
    annotation = save(
        "annotation",
        dict(frozen_annotation_slots={canonical_hash(slot): dict(selection_slot=slot)}),
    )
    primary = save("primary", dict(protocol=save("protocol", {})))
    weights = save(
        "weights",
        dict(
            primary_manifest=primary,
            seed=13,
            condition="symbolic",
            epoch=6,
            artifact=dict(model_hash="m", path="saved.pt", sha256="a" * 64),
        ),
    )
    historical = save(
        "historical",
        dict(
            models=[
                dict(
                    seed=13,
                    condition="symbolic",
                    development=[dict(epoch=5, selection_criterion=None)],
                )
            ]
        ),
    )
    manifest = save(
        "manifest",
        dict(
            weights=weights,
            test_allowed=False,
            annotation_manifest=annotation,
            annotation_state_directory=str(tmp_path / "annotation-state"),
            case_ids=[f"c{i}" for i in range(32)],
            historical_review=historical,
        ),
    )
    request = save(
        "request",
        dict(
            request_identity="saved-request",
            selection_slot=slot,
            model_hash="m",
            case={},
            assignment=[0],
            profile=[],
            symbolic_label=dict(benefit=0.6, cost=0.1),
        ),
    )
    rows = []
    for i in range(32):
        decoded = dict(
            status="verified", logical_status="VERIFIED_FEASIBLE", selected_utility=0.5, checks=1
        )
        if i == 0:
            decoded.update(annotation_pending=True, annotation_request=request["path"])
        rows.append(
            save(
                f"row-{i}",
                dict(
                    manifest=manifest,
                    case_id=f"c{i}",
                    selection_slot={**slot, "case_id": f"c{i}"},
                    result=dict(status="generated", decoded=decoded),
                ),
            )
        )
    report = save("report", dict(manifest=manifest, expected_rows=32, rows=rows))
    monkeypatch.setenv("EXACT_REPAIR_DEADLINE_EPOCH", str(time.time() + 60))
    deferred = recovery.annotate_report(report["path"], tmp_path / "out")
    assert deferred["status"] == "transport_deferred"
    assert deferred["reason"] == "worker_slice_exhausted" and not calls
    monkeypatch.setenv("EXACT_REPAIR_DEADLINE_EPOCH", str(time.time() + 3600))
    result = recovery.annotate_report(report["path"], tmp_path / "out")
    assert (result["selected_checkpoint"] is not None) is selected
    assert result["rows"] == 32 and result["selection_rule_unchanged"]
    assert recovery.annotate_report(report["path"], tmp_path / "out") == result
    assert len(calls) == 1  # terminal missing/abstained supervision is not resampled
    assert (tmp_path / "annotation-state/saved-request/request.json").exists()


def test_validation_shards_require_disjoint_complete_coverage_and_matched_hardware(tmp_path):
    from tools.repair.validation_recovery import aggregate
    from tools.repair.shared_release import immutable
    from tools.repair.historical_regression import binding
    from tools.repair.batch import read

    def save(name, value):
        path = tmp_path / (name + ".json")
        immutable(path, value)
        return binding(path)

    manifest = save(
        "manifest",
        dict(
            case_ids=["a", "b"],
            case_gpu_uuids={"a": "GPU-1", "b": "GPU-2"},
            families=["domain"],
        ),
    )
    reports = []
    for case, gpu in (("a", "GPU-1"), ("b", "GPU-2")):
        row = dict(
            manifest=manifest,
            case_id=case,
            gpu_uuid=gpu,
            family="domain",
            control="corrupted",
            selection_slot=dict(case_id=case),
            result=dict(status="generated", decoded=dict(logical_status="VERIFIED_FEASIBLE")),
        )
        ref = save(case, row)
        report = dict(
            manifest=manifest,
            status="complete",
            expected_rows=1,
            rows=[ref],
            optimizer_updates=0,
            hosted_calls=0,
            test_outcomes_opened=False,
            gpu_uuid=gpu,
        )
        reports.append(save("shard-" + case, report)["path"])
    aggregate(manifest["path"], reports, tmp_path / "joined")
    assert read(tmp_path / "joined/report.json")["technical_gate"]["passed"]
    for invalid in (reports[:1], [reports[0], reports[0]]):
        with pytest.raises(ValueError, match="overlapping or incomplete"):
            aggregate(manifest["path"], invalid, tmp_path / "invalid")
    row["gpu_uuid"] = "GPU-1"
    report["rows"] = [save("wrong-gpu", row)]
    wrong = save("wrong-shard", report)
    with pytest.raises(ValueError, match="hardware assignment changed"):
        aggregate(manifest["path"], [reports[0], wrong["path"]], tmp_path / "invalid")
