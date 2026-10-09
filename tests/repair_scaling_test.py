"""Scaling denominator, inherited identities, matched languages and safe recovery."""

import dataclasses as dc
from pathlib import Path

import pytest

from exact.repair.api import write_artifact
from exact.repair.records import canonical_hash, read_record
from exact.repair.workers import CallResult, bounded_call
from tools.repair import scaling
from tools.repair.expanded_profile import parent_case, parent_fingerprints
from tools.repair.expanded_corpus import binding, bound


def settings():
    return dict(
        generation_seconds=60,
        seed=13,
        draws_per_object=4,
        compiler_nodes=100000,
        compiler_call_seconds=5,
        compiler_object_seconds=10,
        compiler_rss_mb=2048,
        max_expressions=1000,
        context_checks=16,
    )


@pytest.mark.parametrize("count", [4, 8, 16])
def test_mapping_load_keeps_parent_and_native_roundtrip(count):
    original = parent_case("papers", 13, 13)
    result = scaling.extend_case(original, count)
    assert len(result.problem.objects) == count
    assert result.structural_parent == original.structural_parent and result.split == "development"
    assert parent_fingerprints(result) == parent_fingerprints(original)
    assert read_record(result.problem.to_dict()) == result.problem
    assert len(result.intended_assignment) == count
    assert result.probes == original.probes
    assert len({o.original_axioms for o in result.problem.objects}) == count


def test_no_heldout_extension_or_bound_factorial():
    with pytest.raises(ValueError, match="development"):
        scaling.extend_case(parent_case("papers", 1, 13, split="test"), 4)
    configs = scaling.configurations()
    assert len(configs) == 10
    assert {c["mapping_count"] for c in configs} == {4, 8, 16}
    assert {c["depth"] for c in configs} == {1, 2}
    assert len({canonical_hash(c) for c in configs}) == 10
    assert all(c["depth"] < 3 for c in configs)


@pytest.mark.parametrize("method", scaling.METHODS)
def test_native_generators_publish_same_input_and_complete_elementary_controls(tmp_path, method):
    case = scaling.extend_case(parent_case("papers", 1, 13), 4)
    config = dict(scaling.configurations()[0], depth=0, constructors=0)
    result = bounded_call(
        scaling.generate,
        case.problem.to_dict(),
        config,
        method,
        settings(),
        str(tmp_path / method),
        str(tmp_path / method / "cache"),
        timeout=80,
        memory_mb=8192,
        cpu_seconds=120,
    )
    assert result.status == "complete", result.detail
    payload = bound(result.value)
    generated = read_record(payload["input"])
    assert payload["completed_objects"] == payload["scheduled_objects"] == 4
    # Finite compiler admission can retain elementary families while others time
    # out; this is a measured outcome, not a failed adapter. Python errors and
    # missing objects still fail this contract test.
    assert all(
        r["status"] in {"complete", "partial"} and "error" not in r for r in payload["reports"]
    )
    if method != "semantic_enumeration_decoder":
        assert all(
            any(f["status"] == "resolved" for f in r["families"]) for r in payload["reports"]
        )
    for old, new in zip(case.problem.objects, generated.objects):
        assert {c.candidate_id for c in old.candidates} <= {c.candidate_id for c in new.candidates}


def test_semantic_methods_share_language(tmp_path):
    case = scaling.extend_case(parent_case("papers", 1, 13), 4)
    config = dict(scaling.configurations()[0], depth=0, constructors=0)
    results = []
    for method in scaling.METHODS[1:]:
        receipt = scaling.generate(
            case.problem.to_dict(),
            config,
            method,
            settings(),
            tmp_path / method,
            tmp_path / method / "cache",
        )
        results.append(bound(receipt))
    assert [r["language_hash"] for r in results[0]["reports"]] == [
        r["language_hash"] for r in results[1]["reports"]
    ]
    assert [r["context_proofs"] for r in results[0]["reports"]] == [
        r["context_proofs"] for r in results[1]["reports"]
    ]


def schedule_fixture(tmp_path):
    protocol = tmp_path / "protocol.json"
    write_artifact(protocol, dict(test=True))
    rows = [
        dict(
            id="cold",
            pair_id="pair",
            cache_mode="cold",
            seconds=300,
            cpu_seconds=600,
            memory_mb=8192,
        ),
        dict(
            id="warm",
            pair_id="pair",
            cache_mode="warm",
            seconds=300,
            cpu_seconds=600,
            memory_mb=8192,
        ),
    ]
    schedule = tmp_path / "schedule.json"
    write_artifact(
        schedule, dict(schema="exact-repair/scaling/v1", protocol=binding(protocol), rows=rows)
    )
    return schedule


def test_failed_cold_and_warm_rows_keep_denominator_and_never_repeat(tmp_path, monkeypatch):
    import exact.repair.workers

    calls = []

    def call(*args, **kwargs):
        calls.append(kwargs)
        return CallResult("timeout", cleanup_complete=True)

    monkeypatch.setattr(exact.repair.workers, "bounded_call", call)
    schedule = schedule_fixture(tmp_path)
    output = tmp_path / "work"
    report = scaling.run(schedule, output, 0, 2)
    assert report["recorded_rows"] == report["scheduled_rows"] == 2
    assert report["counts"] == {"timeout": 2}
    assert scaling.run(schedule, output, 0, 2) == report and len(calls) == 2
    assert all(c["timeout"] == 300 and c["cpu_seconds"] == 600 for c in calls)
    with pytest.raises(ValueError, match="pairs"):
        scaling.run(schedule, tmp_path / "bad", 1, 2)


def test_inflight_and_nested_cleanup_prevent_silent_retry(tmp_path, monkeypatch):
    import exact.repair.workers

    schedule = schedule_fixture(tmp_path)
    output = tmp_path / "work"
    write_artifact(output / "inflight/cold.json", dict(old_owner="14408.80"))
    with pytest.raises(RuntimeError, match="reconciliation"):
        scaling.run(schedule, output, 0, 2)
    monkeypatch.setattr(
        exact.repair.workers,
        "bounded_call",
        lambda *a, **k: CallResult(
            "error", detail="Nested worker cleanup incomplete", cleanup_complete=True
        ),
    )
    with pytest.raises(RuntimeError, match="cleanup incomplete"):
        scaling.run(schedule, tmp_path / "nested", 0, 2)
    assert (tmp_path / "nested/inflight/cold.json").exists()


def test_completed_evaluator_payload_resumes_with_charged_cold_cache(tmp_path, monkeypatch):
    """Exercise the real result producer and consumer, including nested timeouts."""
    import exact.repair.workers
    from tools.repair.prepare import case_to_dict
    from tools.repair.batch import read

    case_path = tmp_path / "case.json"
    write_artifact(case_path, case_to_dict(parent_case("papers", 1, 13)))
    protocol_path = tmp_path / "protocol.json"
    write_artifact(
        protocol_path,
        dict(
            objective=dict(edit_weights={}, integer_scale=100),
            selection=dict(max_candidate_checks=1, max_master_solves=1, retry_budget=0),
            resources=dict(verification_seconds=1),
        ),
    )
    schedule_path = schedule_fixture(tmp_path / "schedule")
    schedule = read(schedule_path)
    schedule.update(
        protocol=binding(protocol_path),
        generation_seconds=1,
        cases=[dict(case=binding(case_path), parent="papers:path-1", config={})],
    )
    for row in schedule["rows"]:
        row.update(case_index=0, method="grammar_circuit")
    write_artifact(schedule_path, schedule)
    outer_calls = []

    def call(fn, *args, **kwargs):
        if fn is scaling.evaluate:
            outer_calls.append(args[1]["id"])
            return CallResult("complete", value=fn(*args), cleanup_complete=True)
        # Native generation/search can time out while the enclosing row finishes
        # correctly and retains UNKNOWN in the scientific denominator.
        return CallResult("timeout", cleanup_complete=True)

    monkeypatch.setattr(exact.repair.workers, "bounded_call", call)
    output = tmp_path / "work"
    report = scaling.run(schedule_path, output, 0, 2)
    assert report["counts"] == {"complete": 2}
    assert outer_calls == ["cold", "warm"]
    cold = bound(read(output / "rows/cold.json")["result"])
    warm = bound(read(output / "rows/warm.json")["result"])
    assert cold["logical_status"] == warm["logical_status"] == "UNKNOWN"
    assert cold["row"] == schedule["rows"][0]
    assert warm["cache_source"] == str(output / "payloads/cold/compiler-cache")
    assert scaling.run(schedule_path, output, 0, 2) == report
    assert outer_calls == ["cold", "warm"]
    (output / "payloads/cold/result.json").write_text("{}")
    with pytest.raises(ValueError, match="payload changed"):
        scaling.run(schedule_path, output, 0, 2)


def test_scaling_payload_rejects_wrong_full_row_even_with_valid_file_hash(tmp_path):
    row = dict(id="same-id", seconds=300, method="grammar_circuit")
    result = tmp_path / "result.json"
    write_artifact(result, dict(row=dict(row, seconds=600)))
    saved = dict(row=row, result=binding(result), payloads=[binding(result)])
    with pytest.raises(ValueError, match="row identity differs"):
        scaling.validate_payloads(saved)
