"""Control freezing keeps teachers and reserved tests outside selector inputs."""

import json
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path

import pytest

from exact.repair.api import write_artifact
from exact.repair.learning import RepairLabel, SemanticTargetSpec, enumerate_teacher
from exact.repair.protocol import RepairProtocolV3, training_projection_v3
from exact.repair.records import (
    ObjectiveV3,
    RepairResultV3,
    candidate_cost,
    canonical_hash,
)
from exact.repair.study import StudyOutcomeV3, filter_inventory, load_schedule
from tools.repair.corpus import generate_corpus
from tools.repair.prepare import load_preparation, save_preparation
from tools.repair.prepare_controls import DEFERRED, prepare_controls, report_controls


@pytest.fixture
def prepared(tmp_path):
    path = Path("specs/exact-repair/protocol/xr21-review2-conformance.json")
    protocol = json.loads(path.read_text().replace("UNFROZEN", "control-fixture"))
    protocol["identity"]["execution_authorized"] = True
    projection = training_projection_v3(RepairProtocolV3.model_validate(protocol))
    profile = tuple(sorted(projection["preferences"]["cost_weights"].items()))
    cases = generate_corpus(
        split_counts={"train": 1, "development": 1, "test": 1},
        siblings_per_parent=1,
        families=("range",),
        revision="v3",
    )
    weights = protocol["teacher"]["family_weights"]
    caches = {}
    for case in cases:
        if case.split == "test":
            continue

        def label(assignment):
            cost = sum(
                candidate_cost(o, o.candidates[a], profile)
                for o, a in zip(case.problem.objects, assignment)
            )
            return RepairLabel(assignment, True, 1.0, cost)

        hashes = {
            "input": case.problem.content_hash,
            "patch": canonical_hash(case.problem.objects),
            "policy": case.problem.policy.content_hash,
            "query": canonical_hash(case.probes),
            "inventory": canonical_hash(tuple(o.candidates for o in case.problem.objects)),
            "profile": canonical_hash(profile),
            "backend": canonical_hash(("pyhermit", version("pyhermit"), "python")),
            "teacher_weights": canonical_hash((weights["desired"], weights["unwanted"])),
            "semantic_target": SemanticTargetSpec(
                canonical_hash(case.probes), weights["desired"], weights["unwanted"]
            ).content_hash,
        }
        cache = enumerate_teacher(
            tuple(len(o.candidates) for o in case.problem.objects), label, hashes=hashes
        )
        caches[case.case_id] = replace(cache, schema="exact-repair/teacher-cache/v3")
    source, protocol_path = tmp_path / "source.json", tmp_path / "protocol.json"
    save_preparation(
        source,
        cases,
        dict(
            protocol=projection,
            protocol_hash=canonical_hash(projection),
            label_seconds=19.0,
            label_cpu_seconds=7.0,
        ),
        caches,
    )
    write_artifact(protocol_path, protocol)
    return source, protocol_path


def test_freezing_preserves_inventory_budget_costs_and_reserved_denominator(prepared, tmp_path):
    source, protocol = prepared
    schedule = tmp_path / "schedule.json"
    before = source.read_bytes()
    preparation = prepare_controls(source, protocol, schedule)
    cases, arms, _ = load_schedule(schedule)
    originals, _, _ = load_preparation(source)
    assert preparation["scheduled"] == 15
    assert preparation["eligible"] == 10 and preparation["deferred"] == 5
    assert preparation["original_label_seconds"] == 19
    assert preparation["original_label_cpu_seconds"] == 7
    assert source.read_bytes() == before
    for case, original in zip(cases, originals):
        if case.declared_split == "test":
            assert case.problem is case.objective is None
            assert case.availability == "unavailable" and case.detail == DEFERRED
        else:
            assert replace(case.problem, budgets=original.problem.budgets) == original.problem
            assert case.problem.budgets.total_seconds <= original.problem.budgets.total_seconds
            assert case.problem.budgets.memory_mb is not None
            assert case.external_objective is None and case.complete_teacher_optimum is None
            assert not case.teacher_cache_hash
            for arm in arms:
                _, objective, _ = filter_inventory(case, arm)
                assert isinstance(objective, ObjectiveV3)
                assert objective.target_basis.startswith("diagnostic-")
            uniform, symbolic = (dict(case.objective_variants)[k] for k in ("uniform", "symbolic"))
            assert uniform.costs == symbolic.costs
    with pytest.raises(FileExistsError):
        prepare_controls(source, protocol, schedule)


@pytest.mark.parametrize("change", ["input", "semantic_target", "test_labels", "missing", "legacy"])
def test_unqualified_cache_rejected_before_any_schedule(prepared, tmp_path, change):
    source, protocol = prepared
    cases, caches, report = load_preparation(source)
    key = next(iter(caches))
    if change == "test_labels":
        caches[next(c.case_id for c in cases if c.split == "test")] = caches[key]
    elif change == "missing":
        del caches[key]
    elif change == "legacy":
        caches[key] = replace(caches[key], schema="exact-repair/teacher-cache/v2")
    else:
        hashes = {**dict(caches[key].hashes), change: "changed"}
        caches[key] = replace(caches[key], hashes=tuple(sorted(hashes.items())))
    save_preparation(source, cases, report, caches)
    schedule = tmp_path / "schedule.json"
    with pytest.raises(ValueError):
        prepare_controls(source, protocol, schedule)
    assert not schedule.exists() and not (tmp_path / "schedule_cases").exists()


def test_teacher_values_cannot_change_control_objectives(prepared, tmp_path):
    source, protocol = prepared
    first = tmp_path / "first.json"
    prepare_controls(source, protocol, first)
    cases, caches, report = load_preparation(source)
    caches = {
        key: replace(cache, labels=tuple(replace(label, benefit=17.0) for label in cache.labels))
        for key, cache in caches.items()
    }
    save_preparation(source, cases, report, caches)
    second = tmp_path / "second.json"
    prepare_controls(source, protocol, second)
    assert load_schedule(first) == load_schedule(second)


def test_reporting_preserves_missing_rows_and_deferred_test(prepared, tmp_path):
    source, protocol = prepared
    schedule = tmp_path / "schedule.json"
    prepare_controls(source, protocol, schedule)
    cases, arms, _ = load_schedule(schedule)
    case = next(c for c in cases if c.declared_split == "train")
    arm = arms[0]
    from tests.repair_v3_study_test import report as verification

    result_record = RepairResultV3(
        case.problem.content_hash,
        case.objective.content_hash,
        "VERIFIED_FEASIBLE",
        "OPTIMAL_IN_POOL",
        "complete_supported_fragment",
        "fixture",
        (0,),
        (),
        (),
        (),
        0,
        0,
        verification(case.problem, (0,), safe=True),
    )
    outcome = StudyOutcomeV3(
        case.case_id,
        arm.arm_id,
        case.content_hash,
        arm.content_hash,
        "completed",
        "train",
        result=result_record,
        common_assignment=(0,),
    )
    write_artifact(tmp_path / "outcome.json", outcome.to_dict())
    result = {
        "rows": [
            {
                "case_id": case.case_id,
                "arm_id": arm.arm_id,
                "artifact": "outcome.json",
                "artifact_hash": outcome.content_hash,
            }
        ]
    }
    write_artifact(tmp_path / "results.json", result)
    report = report_controls(schedule, tmp_path / "results.json", tmp_path / "report.json")
    assert report["scheduled"] == 15 and report["recorded"] == 1
    assert report["statuses"] == {"completed": 1, "incomplete": 9, "deferred": 5}
    assert (
        next(row for row in report["rows"] if row["status"] == "completed")["semantic_benefit"] == 1
    )
    altered = replace(outcome, case_hash="another-inventory")
    write_artifact(tmp_path / "outcome.json", altered.to_dict())
    with pytest.raises(ValueError, match="frozen control schedule"):
        report_controls(schedule, tmp_path / "results.json", tmp_path / "report.json")
