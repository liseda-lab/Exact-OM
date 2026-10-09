"""Boundaries, typed queries and no-replay recovery for development resources."""

from dataclasses import replace

import pytest

from exact.repair.api import write_artifact
from exact.repair.workers import CallResult
from tools.repair import development_intended as intended
from tools.repair.expanded_profile import parent_case
from tools.repair.historical_regression import binding
from tools.repair.prepare import case_to_dict


def case_and_row():
    case = parent_case("overlap", 2, 13)
    record = case_to_dict(case)
    return dict(case_hash=record["hash"], case_id=case.case_id), record, case


@pytest.mark.parametrize(
    "change",
    [
        dict(split="test"),
        dict(split="train"),
        dict(expected_cases=31),
        dict(expected_parents=15),
        dict(heldout_use=True),
        dict(model_fitting=True),
        dict(supervision_admitted=True),
        dict(budget={}),
    ],
)
def test_boundary_fails_before_opening_files(change):
    plan = dict(
        schema=intended.SCHEMA,
        split="development",
        expected_cases=32,
        expected_parents=16,
        heldout_use=False,
        model_fitting=False,
        supervision_admitted=False,
        budget=intended.BUDGET,
    )
    with pytest.raises(ValueError, match="boundary"):
        intended.validate_schedule({**plan, **change})


def test_nonvacuity_is_derived_not_absent():
    case = parent_case("disjointness", 2, 13)
    probes = intended.query_scope(case)
    assert all(p["nonvacuity_mode"] == "typed_derived" for p in probes)
    assert len(probes[0]["conditions"]) == 2
    assert all(p["conditions"] for p in probes)


@pytest.mark.parametrize("status,cleanup", [("timeout", True), ("error", True), ("timeout", False)])
def test_timeout_or_error_is_saved_and_never_replayed(tmp_path, monkeypatch, status, cleanup):
    row, record, case = case_and_row()
    calls = []

    def execute(*args, **kwargs):
        calls.append(kwargs)
        return CallResult(
            status,
            detail=(
                "ValueError: diagnostic fixture"
                if status == "error"
                else "stage deadline exhausted"
            ),
            cleanup_complete=cleanup,
        )

    monkeypatch.setattr(intended, "bounded_call", execute)
    if status == "error" or not cleanup:
        with pytest.raises(RuntimeError):
            intended.one_case(row, record, case, tmp_path, "source", intended.BUDGET)
        with pytest.raises(RuntimeError):
            intended.one_case(row, record, case, tmp_path, "source", intended.BUDGET)
    else:
        ref = intended.one_case(row, record, case, tmp_path, "source", intended.BUDGET)
        assert intended.one_case(row, record, case, tmp_path, "source", intended.BUDGET) == ref
        saved = intended.read(ref["path"])
        assert saved["status"] == "timeout" and saved["query_denominator"] == len(case.probes)
        assert saved["result"] is None and not saved["supervision_admitted"]
        payload = next(p for p in saved["payloads"] if p["path"].endswith("call.json"))
        write_artifact(payload["path"], dict(changed=True))
        with pytest.raises(ValueError, match="payload changed"):
            intended.one_case(row, record, case, tmp_path, "source", intended.BUDGET)
    assert len(calls) == 1
    assert calls[0] == dict(timeout=600, cpu_seconds=1200, memory_mb=8192)


def test_guard_blocks_unreceipted_work(tmp_path, monkeypatch):
    row, record, case = case_and_row()
    directory = tmp_path / "cases" / intended.canonical_hash((case.case_id, row["case_hash"]))
    write_artifact(directory / "inflight.json", dict(identity="interrupted"))
    monkeypatch.setattr(intended, "bounded_call", lambda *a, **k: pytest.fail("No replay"))
    with pytest.raises(RuntimeError, match="owner/budget"):
        intended.one_case(row, record, case, tmp_path, "source", intended.BUDGET)


@pytest.mark.parametrize("split", ["test", "train"])
def test_worker_rejects_other_splits_before_native(tmp_path, split):
    _, _, case = case_and_row()
    with pytest.raises(ValueError, match="development"):
        intended.case_worker(case_to_dict(replace(case, split=split)), tmp_path)


@pytest.mark.slow
def test_native_worker_preserves_policy_queries_and_progress(tmp_path):
    from tools.repair.corpus import generate_corpus

    case = generate_corpus(
        split_counts={"train": 0, "development": 1, "test": 0},
        siblings_per_parent=1,
        families=("overlap",),
        revision="v3",
    )[0]
    call = intended.bounded_call(
        intended.case_worker,
        case_to_dict(case),
        str(tmp_path),
        timeout=120,
        cpu_seconds=240,
        memory_mb=8192,
    )
    assert call.status == "complete", call.detail
    assert call.cleanup_complete and call.value["original_guard_passed"]
    assert call.value["intended_target_satisfied"]
    assert len(call.value["semantic"]["outcomes"]) == len(case.probes)
    assert list((tmp_path / "native").glob("*.obligations.jsonl"))
    assert (tmp_path / "intended-policy.json").exists()
    assert (tmp_path / "intended-queries.json").exists()
