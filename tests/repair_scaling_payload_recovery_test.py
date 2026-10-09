"""Regression: stopped atomic writes must not replay charged scaling rows."""

from pathlib import Path

import pytest

from exact.repair.api import write_artifact
from exact.repair.records import canonical_hash
from exact.repair.workers import CallResult
from tools.repair import scaling, scaling_payload_recovery as recovery
from tools.repair.batch import read, sha
from tools.repair.expanded_corpus import binding, bound
from tools.repair.expanded_profile import checkpoint


def test_manifest_keeps_unreadable_atomic_debris_without_opening(tmp_path):
    target = tmp_path / "pool-progress.json"
    target.write_text("{}")
    debris = tmp_path / ".pool-progress.json.7af4pfkk"
    debris.touch(mode=0)
    published, excluded = recovery.payload_manifest(tmp_path)
    assert published == [binding(target)]
    assert excluded == [
        dict(
            path=str(debris),
            size=0,
            mode=0,
            uid=debris.stat().st_uid,
            reason="unpublished_atomic_write_temporary",
        )
    ]
    target.chmod(0)
    with pytest.raises(PermissionError):
        recovery.payload_manifest(tmp_path)


def test_batch_outputs_preserve_old_unreadable_debris_without_failure(tmp_path):
    from tools.repair.batch import record_outputs

    work, attempt = tmp_path / "work", tmp_path / "attempt"
    result = work / "scaling-revision-002/payloads/row/result.json"
    write_artifact(result, dict(logical_status="UNKNOWN"))
    debris = result.with_name(".result.json.7af4pfkk")
    debris.touch(mode=0)
    expected = {"scaling-revision-002/payloads/row/result.json": sha(result)}
    assert record_outputs(work, attempt) == expected
    assert read(attempt / "outputs.json") == expected
    inventory = read(attempt / "outputs-unpublished.json")
    assert inventory["files"][0]["path"] == str(debris)
    assert inventory["files"][0]["mode"] == 0 and debris.exists()
    assert not inventory["original_files_modified"]
    result.chmod(0)
    with pytest.raises(PermissionError):
        record_outputs(work, attempt)


def fixture(tmp_path, monkeypatch):
    import exact.repair.study
    import tools.repair.batch

    monkeypatch.setattr(exact.repair.study, "runtime_manifest", lambda: {})
    source = str(Path(scaling.__file__).resolve().parents[2])
    batch_path = tmp_path / "batch.json"
    write_artifact(batch_path, dict(code=source))
    monkeypatch.setattr(tools.repair.batch, "checked_batch", lambda p: read(p))
    rows = [
        dict(
            id=str(i),
            cache_mode="warm" if i % 2 else "cold",
            pair_id=str(i // 2),
            seconds=300,
            cpu_seconds=600,
            memory_mb=8192,
        )
        for i in range(4)
    ]
    schedule = tmp_path / "schedule.json"
    write_artifact(schedule, dict(rows=rows))
    identity = canonical_hash((binding(schedule), sha(scaling.__file__), {}))
    work = tmp_path / "work"
    old = work / "scaling-revision-002"
    refs = []
    for i in range(2):
        path = old / "rows" / f"{i}.json"
        checkpoint(
            path,
            canonical_hash((identity, rows[i])),
            row=rows[i],
            status="timeout",
            result=None,
            cleanup_complete=True,
            payloads=[],
        )
        refs.append(dict(index=i, receipt=binding(path)))
    guard = old / "inflight/2.json"
    checkpoint(guard, canonical_hash((identity, rows[2])), row=rows[2], started_epoch=1)
    payload = old / "payloads/2"
    payload.mkdir(parents=True)
    result = payload / "result.json"
    write_artifact(
        result,
        dict(
            row=rows[2],
            logical_status="UNKNOWN",
            semantic_benefit=None,
            generation_status="timeout",
            selection_status="timeout",
            elapsed_seconds=235,
            pool=None,
        ),
    )
    debris = payload / ".pool-progress.json.7af4pfkk"
    debris.touch(mode=0)
    (payload / "compiler-cache").mkdir()
    (payload / "compiler-cache/circuit.json").write_text("{}")
    published, excluded = recovery.payload_manifest(payload)
    completion = tmp_path / "completion.json"
    write_artifact(
        completion,
        dict(
            status="failed",
            exit_code=1,
            batch=str(batch_path),
            step_id="14408.236",
            dispatch_nonce="nonce",
            work=str(work),
            error=dict(message="command 0 exited 1: PermissionError: " + str(debris)),
        ),
    )
    owner = tmp_path / "step.json"
    write_artifact(owner, dict(step_id="14408.236", dispatch_nonce="nonce"))
    command = tmp_path / "command-0.json"
    write_artifact(
        command,
        dict(
            cwd=source,
            argv=[
                "python",
                "-m",
                "tools.repair.scaling",
                str(schedule),
                str(old),
                "--start",
                "0",
                "--stop",
                "4",
            ],
        ),
    )
    evidence = tmp_path / "evidence.json"
    write_artifact(
        evidence,
        dict(
            completion=binding(completion),
            owner=binding(owner),
            command=binding(command),
            step_id="14408.236",
            rows=refs,
            failed_index=2,
            result=binding(result),
            guard=binding(guard),
            payloads=published,
            temporary=excluded[0],
        ),
    )
    output = work / "scaling-revision-003"
    plan = tmp_path / "plan.json"
    write_artifact(
        plan,
        dict(
            schedule=binding(schedule),
            slice=[0, 4],
            output=str(output),
            adapter=binding(recovery.__file__),
            source_root=source,
            runtime={},
            scientific_batch=binding(batch_path),
            scientific_identity=identity,
            recovery=binding(evidence),
        ),
    )
    return plan, output, old


def test_resume_retains_unknown_and_charged_cache_and_never_replays(tmp_path, monkeypatch):
    import exact.repair.workers

    plan, output, old = fixture(tmp_path, monkeypatch)
    calls = []

    def call(fn, schedule, row, directory, cache_source, **limits):
        calls.append((fn, row["id"], cache_source, limits))
        assert fn is scaling.evaluate
        return CallResult("timeout", cleanup_complete=True)

    monkeypatch.setattr(exact.repair.workers, "bounded_call", call)
    owner = lambda step: dict(
        step_id=step, surviving_processes=0, cgroup_absent=True, signals_sent=0
    )
    report = recovery.run(plan, output, owner_check=owner)
    assert report["scheduled_rows"] == report["recorded_rows"] == 4
    assert report["scientific_rows_replayed"] == 0
    assert len(calls) == 1 and calls[0][1] == "3"
    assert calls[0][2] == str(old / "payloads/2/compiler-cache")
    assert calls[0][3] == dict(timeout=300, cpu_seconds=600, memory_mb=8192)
    retained = bound(report["rows"][2])
    assert bound(retained["result"])["logical_status"] == "UNKNOWN"
    assert (
        retained["additional_scientific_seconds"] == 0 and not retained["outer_telemetry_available"]
    )
    assert (old / "inflight/2.json").exists()
    assert (output / "calls/3.json").exists()
    assert recovery.run(plan, output, owner_check=owner) == report and len(calls) == 1


def test_resume_rejects_previously_started_continuation(tmp_path, monkeypatch):
    plan, output, old = fixture(tmp_path, monkeypatch)
    (old / "payloads/3").mkdir()
    with pytest.raises(ValueError, match="already spent work"):
        recovery.run(plan, output, owner_check=lambda step: {})


def test_resume_rejects_changed_published_payload(tmp_path, monkeypatch):
    plan, output, old = fixture(tmp_path, monkeypatch)
    (old / "payloads/2/compiler-cache/circuit.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="inventory changed"):
        recovery.run(plan, output, owner_check=lambda step: {})
