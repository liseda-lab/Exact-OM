import pytest

from exact.repair.api import write_artifact
from tools.repair import schema_recovery as recovery
from tools.repair.batch import read
from tools.repair.expanded_corpus import binding


def records():
    row = dict(id="row", seconds=300, cpu_seconds=600, memory_mb=8192, case_index=0, arm_id="m")
    payload = dict(
        row_id="row",
        status="generation_error",
        detail=recovery.ERROR,
        generation_resources=dict(wall_seconds=12),
    )
    old = dict(
        status="complete",
        cleanup_complete=True,
        row=row,
        elapsed_seconds=15,
        resources=dict(cpu_seconds=8),
        payloads=[],
    )
    return row, payload, old


def test_schema_replay_debits_each_prior_resource_and_rejects_reset():
    row, payload, old = records()
    limits = recovery.remaining_budget(old, payload, row, dict(generation_seconds=60))
    assert (limits["wall_seconds"], limits["generation_seconds"], limits["cpu_seconds"]) == (
        285,
        48,
        592,
    )
    recovery.validate_remaining_budget(limits, row, dict(generation_seconds=60))
    with pytest.raises(ValueError, match="exceeded"):
        recovery.validate_remaining_budget(
            dict(limits, generation_seconds=60), row, dict(generation_seconds=60)
        )
    with pytest.raises(ValueError, match="confirmed"):
        recovery.remaining_budget(
            old, dict(payload, detail="unrelated exception"), row, dict(generation_seconds=60)
        )


def test_schema_recovery_reuses_complete_original_payload_without_worker(tmp_path, monkeypatch):
    import exact.repair.workers

    row, payload, old = records()
    write_artifact(tmp_path / "result.json", dict(payload, status="evaluated", detail=""))
    old["result"] = binding(tmp_path / "result.json")
    old["payloads"] = [old["result"]]
    write_artifact(tmp_path / "old.json", old)
    monkeypatch.setattr(
        exact.repair.workers, "bounded_call", lambda *a, **kw: pytest.fail("must reuse")
    )
    result = recovery.one(
        {},
        row,
        dict(previous=binding(tmp_path / "old.json")),
        dict(compatible=False, status="unavailable_model_schema"),
        tmp_path / "output",
        "frozen",
    )
    assert result["result"] == old["result"] and result["payloads"] == old["payloads"]
    assert result["recovery_action"] == "reused_original_unchanged"
    assert read(tmp_path / "old.json") == old


def test_schema_recovery_marks_incompatible_encoder_without_worker(tmp_path, monkeypatch):
    import exact.repair.workers

    row, payload, old = records()
    write_artifact(tmp_path / "result.json", payload)
    old["result"] = binding(tmp_path / "result.json")
    write_artifact(tmp_path / "old.json", old)
    monkeypatch.setattr(
        exact.repair.workers, "bounded_call", lambda *a, **kw: pytest.fail("no new weights")
    )
    schedule = dict(cases=[dict(status="materialized")], arms=[dict(id="m", status="available")])
    result = recovery.one(
        schedule,
        row,
        dict(previous=binding(tmp_path / "old.json")),
        dict(compatible=False, status="unavailable_model_schema"),
        tmp_path / "output",
        "frozen",
    )
    assert result["status"] == "unavailable_model_schema" and result["result"] is None
    assert result["recovery_of"] == binding(tmp_path / "old.json")


def test_schema_recovery_executes_only_remaining_budget_and_preserves_original_row_id(
    tmp_path, monkeypatch
):
    import exact.repair.workers
    from exact.repair.workers import CallResult

    row, payload, old = records()
    write_artifact(tmp_path / "result.json", payload)
    old["result"] = binding(tmp_path / "result.json")
    write_artifact(tmp_path / "old.json", old)
    write_artifact(tmp_path / "protocol.json", dict(resources=dict(generation_seconds=60)))
    schedule = dict(
        cases=[dict(status="materialized")],
        arms=[dict(id="m", status="available", protocol=binding(tmp_path / "protocol.json"))],
    )
    calls = []

    def call(function, *args, **kwargs):
        calls.append(kwargs)
        assert args[1] == row
        return CallResult("timeout", detail="retained budget")

    monkeypatch.setattr(exact.repair.workers, "bounded_call", call)
    entry = dict(previous=binding(tmp_path / "old.json"))
    compatibility = dict(compatible=True, status="compatible")
    result = recovery.one(schedule, row, entry, compatibility, tmp_path / "output", "frozen")
    assert calls[0]["timeout"] == 285 and calls[0]["cpu_seconds"] == 592
    assert calls[0]["remaining_budget"]["generation_seconds"] == 48
    assert result == recovery.one(
        schedule, row, entry, compatibility, tmp_path / "output", "frozen"
    )
    assert len(calls) == 1 and result["row"] == row


@pytest.mark.parametrize(
    "status,detail,raises",
    [
        ("generation_error", "TypeError: broken adapter", True),
        ("error", "ValueError: broken semantic evaluator", True),
        ("generation_timeout", "stage deadline exhausted", False),
        ("unavailable_model_schema", "Unregistered edge type", False),
        (
            "generation_error",
            "ValueError: cannot infer complete original relation for endpoint retrieval",
            False,
        ),
    ],
)
def test_scientific_fail_fast_distinguishes_software_from_scoped_outcomes(
    tmp_path, status, detail, raises
):
    from tools.repair.fresh_evaluation import raise_on_software_failure

    write_artifact(tmp_path / "result.json", dict(status=status, detail=detail))
    saved = dict(status="complete", result=binding(tmp_path / "result.json"))
    if raises:
        with pytest.raises(RuntimeError, match="software failure"):
            raise_on_software_failure(saved)
    else:
        raise_on_software_failure(saved)


def test_legacy_unsupported_mapping_gets_typed_revision_without_rerun(tmp_path, monkeypatch):
    import exact.repair.workers

    row, payload, old = records()
    payload["detail"] = "ValueError: cannot infer complete original relation for endpoint retrieval"
    write_artifact(tmp_path / "result.json", payload)
    old["result"] = binding(tmp_path / "result.json")
    write_artifact(tmp_path / "old.json", old)
    monkeypatch.setattr(
        exact.repair.workers, "bounded_call", lambda *a, **kw: pytest.fail("must not rerun")
    )
    result = recovery.one(
        {},
        row,
        dict(previous=binding(tmp_path / "old.json")),
        dict(compatible=False, status="unsupported_original_mapping_bundle"),
        tmp_path / "output",
        "frozen",
    )
    revised = read(result["result"]["path"])
    assert revised["status"] == "unsupported_original_mapping_bundle"
    assert revised["original_result"] == old["result"] and read(tmp_path / "result.json") == payload
