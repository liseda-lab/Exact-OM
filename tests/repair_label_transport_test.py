"""Lossless bounded label transport and cumulative selective preparation recovery."""

import json
import pickle
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from exact.repair.learning import SupportTarget
from exact.repair.protocol import load_protocol_v3, training_projection_v3
from exact.repair.records import canonical_hash
from exact.repair.workers import CallResult, bounded_call
from tests.repair_review2_architecture_test import protocol_fixture
from tests.repair_training_completion_test import cache_for
from tools.repair import train
from tools.repair.corpus import generate_corpus
from tools.repair.prepare import load_preparation, publish_label_cache, read_label_cache


def oversized_cache(case):
    cache = replace(
        cache_for(case, unknown=True), schema="exact-repair/teacher-cache/v3", elapsed_seconds=1.25
    )
    label = cache.labels[0]
    target = SupportTarget(
        label.assignment,
        "fixture-witness",
        "class_satisfiability",
        None,
        "fixture-theory",
        case.problem.policy.content_hash,
        "fixture-backend",
        tuple((obj.object_id, "candidate") for obj in case.problem.objects),
        ("ab" * (8 * 1024 * 1024 + 1),),
        (),
    )
    return replace(cache, labels=(replace(label, support_targets=(target,)), *cache.labels[1:]))


def publish_oversized(case, directory):
    return publish_label_cache(oversized_cache(case), directory)


def test_large_cache_crosses_real_bounded_worker_without_losing_unknowns_or_provenance(tmp_path):
    case = generate_corpus(
        revision="v3", parents_per_family=1, siblings_per_parent=1, families=("range",)
    )[0]
    cache = oversized_cache(case)
    assert len(pickle.dumps(cache, protocol=5)) > 16 * 1024 * 1024
    failed = bounded_call(oversized_cache, case, timeout=15, memory_mb=1024)
    assert failed.status == "error" and "transport frame limit" in failed.detail
    result = bounded_call(publish_oversized, case, tmp_path, timeout=15, memory_mb=1024)
    assert result.status == "complete", result.detail
    assert len(pickle.dumps(result)) < 2048
    assert result.value["size_bytes"] > 16 * 1024 * 1024
    restored = read_label_cache(result.value, tmp_path, case)
    assert restored == cache
    assert restored.coverage["unknown_policy"] > 0
    assert not restored.complete

    with pytest.raises(ValueError, match="different input"):
        read_label_cache(
            result.value, tmp_path, replace(case, problem=replace(case.problem, evidence=()))
        )
    with pytest.raises(ValueError, match="outside its declared output"):
        read_label_cache(result.value, tmp_path / "other", case)
    path = Path(result.value["path"])
    with path.open("r+b") as stream:
        stream.seek(-1, 2)
        stream.write(b"x")
    with pytest.raises(ValueError, match="integrity mismatch"):
        read_label_cache(result.value, tmp_path, case)
    with path.open("ab") as stream:
        stream.write(b"x")
    with pytest.raises(ValueError, match="size mismatch"):
        read_label_cache(result.value, tmp_path, case)
    path.unlink()
    with pytest.raises(ValueError, match="missing"):
        read_label_cache(result.value, tmp_path, case)


def test_transport_recovery_preserves_valid_partial_unknown_test_rows_and_costs(
    tmp_path, monkeypatch
):
    path, _ = protocol_fixture(tmp_path, True)
    protocol = training_projection_v3(load_protocol_v3(path, for_execution=True))
    cases = generate_corpus(
        revision="v3",
        split_counts={"train": 3, "development": 1, "test": 1},
        siblings_per_parent=1,
        families=("range",),
    )
    selected = [case for case in cases if case.split != "test"]
    calls, recovering = [], False

    def worker(function, *args, **options):
        if function.__name__ == "generated_from_protocol":
            return CallResult("complete", cases, resource_usage=(("cpu_seconds", 0.25),))
        assert function is train._label_payload
        case, directory = args
        index = selected.index(case)
        calls.append(case.case_id)
        assert case.split != "test"
        if index == 2 and not recovering:
            return CallResult(
                "error",
                detail="ValueError: worker result exceeds the transport frame limit",
                resource_usage=(("cpu_seconds", 0.5),),
            )
        if index == 3:
            return CallResult(
                "error", detail="Unverified intended parent", resource_usage=(("cpu_seconds", 0.5),)
            )
        cache = replace(cache_for(case, unknown=index == 1), schema="exact-repair/teacher-cache/v3")
        return CallResult(
            "complete",
            publish_label_cache(cache, directory),
            resource_usage=(("cpu_seconds", 0.5),),
        )

    monkeypatch.setattr(train, "bounded_call", worker)
    monkeypatch.delenv("SLURM_STEP_GPUS", raising=False)
    monkeypatch.delenv("SLURM_JOB_GPUS", raising=False)
    output = tmp_path / "run"
    args = ["repair-train", "--protocol", str(path), "--output", str(output), "--prepare-only"]
    monkeypatch.setattr(sys, "argv", args)
    assert train.main() == 0
    _, before, before_report = load_preparation(output / "preparation.json")
    first_budget = json.loads((output / "label-budget.json").read_text())
    assert len(first_budget["attempts"]) == 4
    assert first_budget["spent_cpu_seconds"] == 2.0
    assert before_report["protocol_hash"] == canonical_hash(protocol)

    calls.clear()
    recovering = True
    monkeypatch.setattr(sys, "argv", [*args, "--retry-label-transport-errors"])
    assert train.main() == 0
    assert calls == [selected[2].case_id]
    restored_cases, after, report = load_preparation(output / "preparation.json")
    assert restored_cases == cases
    assert all(after[key] == cache for key, cache in before.items())
    assert selected[3].case_id not in after
    assert all(case.case_id not in after for case in cases if case.split == "test")
    assert report["label_retry_history"] == [before_report["label_rows"][2]]
    assert report["label_rows"][3] == before_report["label_rows"][3]
    second_budget = json.loads((output / "label-budget.json").read_text())
    assert second_budget["attempts"][:4] == first_budget["attempts"]
    assert len(second_budget["attempts"]) == 5
    assert second_budget["spent_cpu_seconds"] == 2.5
    assert second_budget["spent_seconds"] > first_budget["spent_seconds"]
    calls.clear()
    assert train.main() == 0
    assert not calls
    assert json.loads((output / "label-budget.json").read_text()) == second_budget
