"""Successful launcher receipts cannot hide bounded-worker software failures."""

import hashlib
import json
from pathlib import Path

import pytest

from exact.experiments.science_health import inspect_science, software_failure
from exact.experiments.supervision import inspect_runs, pending_batches


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def example(tmp_path, status="generation_error", detail="ValueError: Unregistered edge type"):
    work = tmp_path / "work"
    result = put(work / "payloads/result.json", dict(row_id="row", status=status, detail=detail))
    row = put(work / "rows/row.json", dict(row=dict(id="row"), status="complete", result=result))
    report = put(
        work / "evaluation/report.json",
        dict(
            schema="exact-repair/fresh-evaluation/v1",
            status="complete",
            scheduled=1,
            rows=[dict(row, row_id="row")],
        ),
    )
    run = dict(
        id="run",
        step_id="1.2",
        dispatch_nonce="nonce",
        completion_path=str(tmp_path / "attempt/completion.json"),
        science_report_relative="evaluation/report.json",
    )
    complete = dict(status="complete", step_id="1.2", dispatch_nonce="nonce", work=str(work))
    put(tmp_path / "attempt/outputs.json", {"evaluation/report.json": report["sha256"]})
    return run, complete


def test_successful_outer_receipt_exposes_inner_software_failure(tmp_path):
    run, complete = example(tmp_path)
    inspected = inspect_science(run, complete)
    assert not inspected["errors"]
    assert inspected["failures"][0]["row_ids"] == ["row"]
    assert inspected["failures"][0]["signature"] == dict(
        status="generation_error", detail="ValueError: Unregistered edge type"
    )


def test_corrective_inline_receipts_preserve_unknowns_but_detect_software(tmp_path):
    run, complete = example(tmp_path)
    rows = [
        dict(id="bug", status="error", detail="ValueError: invalid edge"),
        dict(
            id="unknown",
            status="complete",
            result=dict(status="unresolved", logical_status="UNKNOWN"),
        ),
        dict(id="late", status="not_attempted", result=None),
    ]
    report = put(
        tmp_path / "work/evaluation/report.json",
        dict(
            schema="exact-repair/corrective-study/v1", expected_rows=3, completed_rows=2, rows=rows
        ),
    )
    put(tmp_path / "attempt/outputs.json", {"evaluation/report.json": report["sha256"]})
    inspected = inspect_science(run, complete)
    assert not inspected["errors"]
    assert len(inspected["failures"]) == 1
    assert inspected["failures"][0]["row_ids"] == ["bug"]
    assert inspected["failures"][0]["signature"]["detail"] == "ValueError: invalid edge"


def test_corrective_incomplete_denominator_is_not_qualified_failure(tmp_path):
    run, complete = example(tmp_path)
    report = put(
        tmp_path / "work/evaluation/report.json",
        dict(
            schema="exact-repair/corrective-study/v1",
            expected_rows=2,
            completed_rows=1,
            rows=[dict(id="bug", status="error", detail="ValueError: invalid edge")],
        ),
    )
    put(tmp_path / "attempt/outputs.json", {"evaluation/report.json": report["sha256"]})
    inspected = inspect_science(run, complete)
    assert not inspected["failures"] and "denominator" in inspected["errors"][0]


def annotation_example(tmp_path, *, legacy=False, status_code=401):
    run, complete = example(tmp_path)
    complete["status"] = "complete" if legacy else "failed"
    rows = [dict(id="slot", status="unavailable" if legacy else "provider_error", artifact=None)]
    report = dict(
        schema="exact-repair/corrective-annotation-report/v1",
        status="complete" if legacy else "blocked_external",
        scheduled=1,
        recorded=1,
        rows=rows,
    )
    outputs = {}
    if legacy:
        receipt = put(
            tmp_path / "work/slot/receipt.json",
            dict(
                status="annotation_unavailable",
                retry_permitted=False,
                costs_reset=False,
                error=f"Annotation worker error: RuntimeError: OpenRouter HTTP {status_code}; request "
                + "a" * 64
                + " retained; reservation retained",
            ),
        )
        outputs["slot/receipt.json"] = receipt["sha256"]
    else:
        report.update(
            blocker=dict(
                kind="provider_authentication",
                status_code=status_code,
                requires_user=True,
                retry_permitted=False,
                detail="Credential rejected",
            ),
            confirmed_failure=dict(
                kind="authentication",
                http_status=status_code,
                retry_permitted=False,
                request_id="a" * 64,
                response_sha256="b" * 64,
            ),
        )
    ref = put(tmp_path / "work/evaluation/report.json", report)
    outputs["evaluation/report.json"] = ref["sha256"]
    put(tmp_path / "attempt/outputs.json", outputs)
    put(tmp_path / "attempt/completion.json", complete)
    return run, complete


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("status_code", [401, 403])
def test_annotation_auth_is_one_external_blocker_even_after_nonzero_exit(
    tmp_path, legacy, status_code
):
    run, complete = annotation_example(tmp_path, legacy=legacy, status_code=status_code)
    inspected = inspect_science(run, complete)
    assert not inspected["errors"] and not inspected["failures"]
    assert inspected["blockers"][0]["status_code"] == status_code
    health = inspect_runs([run], step_states={})
    assert len(health["incidents"]) == 1
    assert health["incidents"][0]["kind"] == "provider_authentication"
    registry = dict(
        runs=[run],
        capacity=dict(cpus=2),
        pending_batches=[
            dict(id="independent", resources=dict(cpus=1)),
            dict(id="dependent", resources=dict(cpus=1), depends_on=["run"]),
        ],
    )
    assert [row["batch_id"] for row in pending_batches(registry, health)] == ["independent"]


@pytest.mark.parametrize("invalid", [None, "digest", "response", "index", "method", "status"])
def test_free_authentication_preflight_requires_bound_http_evidence(tmp_path, invalid):
    run, complete = annotation_example(tmp_path)
    path = tmp_path / "work/evaluation/report.json"
    report = json.loads(path.read_text())
    raw = json.dumps({"error": {"code": 401, "message": "Credential rejected"}})
    proof = dict(
        source="read_only_authentication_preflight",
        kind="authentication",
        http_status=401,
        profile="annotation",
        method="GET",
        endpoint="https://openrouter.ai/api/v1/key",
        generation_requests=0,
        response_sha256=hashlib.sha256(raw.encode()).hexdigest(),
        retry_permitted=False,
        costs_reset=False,
    )
    receipt = dict(proof, raw_response=raw)
    if invalid == "response":
        receipt["raw_response"] = "{}"
    elif invalid == "method":
        receipt["method"] = proof["method"] = "POST"
    elif invalid == "status":
        receipt["raw_response"] = json.dumps({"error": {"code": 403}})
        receipt["response_sha256"] = proof["response_sha256"] = hashlib.sha256(
            receipt["raw_response"].encode()
        ).hexdigest()
    evidence_path = tmp_path / "work/authentication-evidence/failed.json"
    proof["evidence"] = put(evidence_path, receipt)
    report["confirmed_failure"] = proof
    outputs = {"evaluation/report.json": put(path, report)["sha256"]}
    if invalid != "index":
        outputs["authentication-evidence/failed.json"] = proof["evidence"]["sha256"]
    if invalid == "digest":
        evidence_path.write_text("{}")
    put(tmp_path / "attempt/outputs.json", outputs)
    inspected = inspect_science(run, complete)
    if invalid:
        assert inspected["errors"] and not inspected.get("blockers")
    else:
        assert not inspected["errors"] and not inspected["failures"]
        assert inspected["blockers"][0]["status_code"] == 401


def test_annotation_auth_requires_hash_bound_transport_receipt(tmp_path):
    run, complete = annotation_example(tmp_path, legacy=True)
    (tmp_path / "work/slot/receipt.json").write_text("{}")
    result = inspect_science(run, complete)
    assert not result.get("blockers") and "digest mismatch" in result["errors"][0]


def test_unqualified_provider_error_does_not_become_authentication_blocker(tmp_path):
    run, complete = annotation_example(tmp_path, legacy=True, status_code=500)
    assert inspect_science(run, complete) == dict(failures=[], errors=[])


@pytest.mark.parametrize(
    "status", ["generation_timeout", "verification_timeout", "unknown", "unavailable", "evaluated"]
)
def test_bounded_unknown_or_timeout_is_not_a_software_incident(tmp_path, status):
    run, complete = example(tmp_path, status)
    assert inspect_science(run, complete) == dict(failures=[], errors=[])


def test_modified_nested_result_is_retryable_evidence_not_qualified_failure(tmp_path):
    run, complete = example(tmp_path)
    (tmp_path / "work/payloads/result.json").write_text("{}")
    inspected = inspect_science(run, complete)
    assert not inspected["failures"] and "digest mismatch" in inspected["errors"][0]


def test_opt_in_and_completion_ownership_are_required(tmp_path):
    run, complete = example(tmp_path)
    assert not inspect_science(
        {k: v for k, v in run.items() if k != "science_report_relative"}, complete
    )["failures"]
    assert not inspect_science(run, dict(complete, status="running"))["failures"]
    assert (
        "ownership mismatch"
        in inspect_science(run, dict(complete, dispatch_nonce="wrong"))["errors"][0]
    )


def test_report_must_be_bound_to_completed_outputs(tmp_path):
    run, complete = example(tmp_path)
    put(tmp_path / "attempt/outputs.json", {})
    assert "missing from completed output index" in inspect_science(run, complete)["errors"][0]


def test_traceback_location_does_not_change_software_signature(tmp_path):
    first, a = example(
        tmp_path / "one", detail='Traceback:\n  File "one.py", line 3\nValueError: unsupported edge'
    )
    second, b = example(
        tmp_path / "two", detail='Traceback:\n  File "two.py", line 7\nValueError: unsupported edge'
    )
    assert (
        inspect_science(first, a)["failures"][0]["signature"]
        == inspect_science(second, b)["failures"][0]["signature"]
    )


def test_supervisor_and_dispatch_block_descendants_of_nested_software_failure(tmp_path):
    run, complete = example(tmp_path)
    put(tmp_path / "attempt/completion.json", complete)
    health = inspect_runs([run], step_states={})
    assert health["incidents"][0]["kind"] == "scientific_software_error"
    assert health["findings"][0]["status"] == "needs_attention"
    registry = dict(
        runs=[run],
        capacity=dict(cpus=2),
        pending_batches=[dict(id="next", depends_on=["run"], resources=dict(cpus=1))],
    )
    assert pending_batches(registry, health) == []


def test_replacement_reuses_failure_identity_without_erasing_retry_history(tmp_path):
    original, a = example(tmp_path / "first")
    put(tmp_path / "first/attempt/completion.json", a)
    prior = inspect_runs([original], step_states={})["incidents"][0]
    replacement, b = example(tmp_path / "replacement")
    replacement.update(id="replacement", step_id="1.3")
    b["step_id"] = "1.3"
    put(tmp_path / "replacement/attempt/completion.json", b)
    ancestor = dict(original, enabled=False, superseded_by="replacement")
    current = inspect_runs([ancestor, replacement], step_states={})["incidents"][0]
    assert prior["id"] == current["id"]
    assert current["scope_run_id"] == "run" and current["focus_run_id"] == "replacement"


def test_cached_evidence_is_rechecked_when_payload_changes(tmp_path):
    run, complete = example(tmp_path)
    assert inspect_science(run, complete)["failures"]
    (tmp_path / "work/payloads/result.json").write_text('{"row_id": "row"}')
    changed = inspect_science(run, complete)
    assert not changed["failures"] and changed["errors"]


def test_qualified_work_paths_normalize_but_external_dataset_paths_do_not(tmp_path):
    first, a = example(
        tmp_path / "first", detail="ValueError: " + str(tmp_path / "first/work/cache/file")
    )
    second, b = example(
        tmp_path / "second", detail="ValueError: " + str(tmp_path / "second/work/cache/file")
    )
    one = inspect_science(first, a)["failures"][0]["signature"]
    two = inspect_science(second, b)["failures"][0]["signature"]
    assert (
        one
        == two
        == dict(status="generation_error", detail="ValueError: <science-work>/cache/file")
    )
    first, a = example(tmp_path / "first", detail="ValueError: /dataset/one.owl")
    second, b = example(tmp_path / "second", detail="ValueError: /dataset/two.owl")
    assert (
        inspect_science(first, a)["failures"][0]["signature"]
        != inspect_science(second, b)["failures"][0]["signature"]
    )


def test_report_path_escape_is_retryable_metadata_error(tmp_path):
    run, complete = example(tmp_path)
    result = inspect_science(dict(run, science_report_relative="../outside.json"), complete)
    assert not result["failures"] and "escapes registered work directory" in result["errors"][0]


def test_unhashable_status_is_retryable_metadata_error(tmp_path):
    run, complete = example(tmp_path, status={"bad": "status"})
    result = inspect_science(run, complete)
    assert not result["failures"] and result["errors"]


def test_oversized_or_deep_metadata_is_bounded(tmp_path, monkeypatch):
    import exact.experiments.science_health as health

    run, complete = example(tmp_path)
    monkeypatch.setattr(health, "_MAX_BYTES", 32)
    health._read_cached.cache_clear()
    assert "exceeds 4 MiB" in inspect_science(run, complete)["errors"][0]
    monkeypatch.setattr(health, "_MAX_BYTES", 4 * 1024 * 1024)
    (tmp_path / "attempt/outputs.json").write_text(
        '{"deep":' + "[" * 20000 + "0" + "]" * 20000 + "}"
    )
    result = inspect_science(run, complete)
    assert not result["failures"] and "RecursionError" in result["errors"][0]


def test_large_documents_are_never_retained_in_metadata_cache(tmp_path):
    import exact.experiments.science_health as health

    health._read_cached.cache_clear()
    path = tmp_path / "large.json"
    put(path, {"text": "x" * (health._MAX_CACHED_BYTES + 1)})
    assert health._read(path)["text"]
    assert health._read_cached.cache_info().currsize == 0


def _bound_file(path):
    return dict(path=str(path), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())


def _update_recovery(run, complete, saved):
    work = Path(complete["work"])
    row = put(work / "rows/row.json", saved)
    report = put(
        work / "evaluation/report.json",
        dict(
            schema="exact-repair/schema-recovery/v1",
            status="complete",
            scheduled=1,
            recorded=1,
            rows=[dict(row, row_id="row")],
        ),
    )
    put(
        Path(run["completion_path"]).with_name("outputs.json"),
        {"evaluation/report.json": report["sha256"], "rows/row.json": row["sha256"]},
    )
    put(Path(run["completion_path"]), complete)


def _recover(old_run, old_complete, destination):
    old_work = Path(old_complete["work"])
    old_report = _bound_file(old_work / "evaluation/report.json")
    old_ref = _bound_file(old_work / "rows/row.json")
    source = dict(
        completion=put(Path(old_run["completion_path"]), old_complete),
        outputs=put(
            Path(old_run["completion_path"]).with_name("outputs.json"),
            {"evaluation/report.json": old_report["sha256"], "rows/row.json": old_ref["sha256"]},
        ),
        report=old_report,
        step_id=old_run["step_id"],
        nonce=old_run["dispatch_nonce"],
    )
    saved = json.loads(Path(old_ref["path"]).read_text())
    saved.update(reuse_of=old_ref, reuse_source=source, recovery_action="reused_original_unchanged")
    run, complete = example(destination, "generation_timeout")
    _update_recovery(run, complete, saved)
    return run, complete, saved


def test_schema_recovery_validates_complete_original_chain_before_external_reuse(tmp_path):
    old, original = example(tmp_path / "original", "generation_timeout")
    run, complete, _ = _recover(old, original, tmp_path / "recovery")
    assert inspect_science(run, complete) == dict(failures=[], errors=[])
    next_run, next_complete, _ = _recover(run, complete, tmp_path / "replacement")
    assert inspect_science(next_run, next_complete) == dict(failures=[], errors=[])


@pytest.mark.parametrize("change", ["ownership", "row_binding", "result_binding", "missing_source"])
def test_arbitrary_external_reuse_is_not_authorized_by_a_digest_alone(tmp_path, change):
    old, original = example(tmp_path / "original", "generation_timeout")
    run, complete, saved = _recover(old, original, tmp_path / "recovery")
    if change == "ownership":
        saved["reuse_source"]["nonce"] = "wrong"
    elif change == "row_binding":
        saved["reuse_of"]["sha256"] = "0" * 64
    elif change == "result_binding":
        saved["result"] = put(
            tmp_path / "foreign/result.json", dict(row_id="row", status="generation_error")
        )
    else:
        saved.pop("reuse_source")
    _update_recovery(run, complete, saved)
    result = inspect_science(run, complete)
    assert not result["failures"] and result["errors"]


def test_only_exact_legacy_endpoint_scope_exception_is_not_software():
    detail = "ValueError: cannot infer complete original relation for endpoint retrieval"
    assert not software_failure("generation_error", detail)
    assert software_failure("generation_error", detail + " with unrelated extra failure")
    assert software_failure("verification_error", detail)
    assert software_failure("generation_error", "ValueError: Unregistered edge type")
    assert not software_failure("unavailable_model_schema", "missing relation weights")
