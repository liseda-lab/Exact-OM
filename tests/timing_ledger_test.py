import json
import multiprocessing
from pathlib import Path

import pytest

from exact.utils.logs import summarize_progress_estimates
from exact.utils.timing import (
    CacheStatus,
    TimingLedger,
    config_fingerprint,
    load_recorded_timings,
    write_recorded_timings,
)


def _append_concurrent_session(run_dir: str, worker: int) -> None:
    ledger = TimingLedger.open(Path(run_dir))
    with ledger.session(command="align", config_fingerprint="shared") as session:
        session.record(
            "Alignment.Inference",
            seconds=float(worker + 1),
            cache_status=CacheStatus.FRESH,
            work_done=1,
            work_total=2,
            unit="examples",
        )


def test_fully_resumed_session_does_not_erase_fresh_compute(tmp_path: Path):
    ledger = TimingLedger.open(tmp_path)
    with ledger.session(command="align", config_fingerprint="config-a") as session:
        session.record(
            "Alignment.Inference",
            seconds=600.0,
            cache_status=CacheStatus.FRESH,
            work_done=100,
            work_total=100,
            unit="examples",
        )
    with ledger.session(command="align", config_fingerprint="config-a") as session:
        session.record(
            "Alignment.Inference",
            seconds=0.0,
            cache_status=CacheStatus.SKIPPED,
            work_done=0,
            work_total=100,
            unit="examples",
        )

    total = ledger.stage_totals(config_fingerprint="config-a")["Alignment.Inference"]
    assert total.compute_seconds == 600.0
    assert total.overhead_seconds == 0.0
    assert total.sessions == 2
    assert ledger.sessions()[-1].stages[0].cache_status is CacheStatus.SKIPPED
    assert load_recorded_timings(tmp_path / "times.txt")["Alignment.Inference"] == 10.0


def test_crashed_partial_session_and_resume_both_count(tmp_path: Path):
    ledger = TimingLedger.open(tmp_path)
    crashed = ledger.start_session(command="align", config_fingerprint="config-a")
    crashed.record(
        "Alignment.Inference",
        seconds=400.0,
        cache_status=CacheStatus.FRESH,
        work_done=400,
        work_total=1000,
        unit="examples",
    )

    with ledger.session(command="align", config_fingerprint="config-a") as resumed:
        resumed.record(
            "Alignment.Inference",
            seconds=500.0,
            cache_status=CacheStatus.RESUMED,
            work_done=600,
            work_total=1000,
            unit="examples",
        )

    sessions = ledger.sessions()
    assert sessions[0].ended_at is None
    assert sessions[0].crashed is True
    assert sessions[1].crashed is False
    assert (
        ledger.stage_totals(config_fingerprint="config-a")["Alignment.Inference"].compute_seconds
        == 900.0
    )
    assert ledger.estimates(config_fingerprint="config-a")["Alignment.Inference"] == 1000.0


def test_cache_hit_is_overhead_and_fresh_dataset_drives_estimate(tmp_path: Path):
    ledger = TimingLedger.open(tmp_path)
    with ledger.session(command="align", config_fingerprint="config-a") as session:
        session.record("Dataset", seconds=300.0, cache_status=CacheStatus.FRESH)
    with ledger.session(command="align", config_fingerprint="config-a") as session:
        session.record(
            "Dataset.CacheLoad",
            seconds=2.0,
            cache_status=CacheStatus.CACHE_HIT,
        )
        session.record("Dataset.Process", seconds=0.0, cache_status=CacheStatus.SKIPPED)

    totals = ledger.stage_totals(config_fingerprint="config-a")
    assert totals["Dataset"].compute_seconds == 300.0
    assert totals["Dataset.CacheLoad"].compute_seconds == 0.0
    assert totals["Dataset.CacheLoad"].overhead_seconds == 2.0
    assert ledger.estimates(config_fingerprint="config-a")["Dataset"] == 300.0
    assert "Dataset.CacheLoad" not in ledger.estimates(config_fingerprint="config-a")
    progress_estimates = summarize_progress_estimates(
        ledger=ledger,
        config_fingerprint="config-a",
    )
    assert progress_estimates["Dataset"] == 5.0


def test_configs_are_separate_and_estimate_falls_back_only_when_needed(tmp_path: Path):
    ledger = TimingLedger.open(tmp_path)
    with ledger.session(command="align", config_fingerprint="config-a") as session:
        session.record("Alignment.Inference", seconds=10.0)
    with ledger.session(command="align", config_fingerprint="config-b") as session:
        session.record("Alignment.Inference", seconds=40.0)

    assert (
        ledger.stage_totals(config_fingerprint="config-a")["Alignment.Inference"].compute_seconds
        == 10.0
    )
    assert (
        ledger.stage_totals(config_fingerprint="config-b")["Alignment.Inference"].compute_seconds
        == 40.0
    )
    assert ledger.estimates(config_fingerprint="config-a")["Alignment.Inference"] == 10.0
    assert ledger.estimates(config_fingerprint="unknown")["Alignment.Inference"] == 25.0


def test_nested_spans_retain_parent_identity(tmp_path: Path):
    ledger = TimingLedger.open(tmp_path)
    with ledger.session(command="align", config_fingerprint="config-a") as session:
        with session.stage("Dataset") as outer:
            with session.stage("Dataset.Process", work_total=3, unit="examples") as inner:
                inner.set_work_done(3)

    stages = {record.stage: record for record in ledger.sessions()[0].stages}
    assert stages["Dataset"].span_id == outer.span_id
    assert stages["Dataset"].parent_span_id is None
    assert stages["Dataset.Process"].parent_span_id == outer.span_id
    assert stages["Dataset.Process"].work_done == 3


def test_concurrent_processes_append_without_lost_sessions(tmp_path: Path):
    context = multiprocessing.get_context("spawn")
    workers = [
        context.Process(target=_append_concurrent_session, args=(str(tmp_path), worker))
        for worker in range(2)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=30)
        assert worker.exitcode == 0

    ledger = TimingLedger.open(tmp_path)
    sessions = [session for session in ledger.sessions() if session.command == "align"]
    assert len(sessions) == 2
    assert (
        ledger.stage_totals(config_fingerprint="shared")["Alignment.Inference"].compute_seconds
        == 3.0
    )
    lines = (tmp_path / "timings.json").read_text(encoding="utf-8").splitlines()
    assert lines
    assert all(json.loads(line)["schema_version"] == 1 for line in lines)


def test_corrupt_ledger_is_backed_up_and_reinitialized(tmp_path: Path):
    (tmp_path / "timings.json").write_text('{"broken":', encoding="utf-8")

    with pytest.warns(RuntimeWarning, match="Corrupt timing ledger moved"):
        ledger = TimingLedger.open(tmp_path)

    assert ledger.sessions() == []
    assert len(list(tmp_path.glob("timings.json.corrupt-*"))) == 1
    first_event = json.loads(
        (tmp_path / "timings.json").read_text(encoding="utf-8").splitlines()[0]
    )
    assert first_event["schema_version"] == 1
    assert first_event["event"] == "ledger_started"


def test_legacy_times_file_is_imported_once(tmp_path: Path):
    write_recorded_timings(
        tmp_path / "times.txt",
        {"Dataset": 5.0, "Alignment.Inference": 10.0},
    )

    first = TimingLedger.open(tmp_path)
    second = TimingLedger.open(tmp_path)

    assert len(first.sessions()) == 1
    assert len(second.sessions()) == 1
    legacy = first.sessions()[0]
    assert legacy.command == "legacy"
    assert legacy.config_fingerprint == "legacy"
    assert {record.stage: record.seconds for record in legacy.stages} == {
        "Dataset": 300.0,
        "Alignment.Inference": 600.0,
    }


def test_config_fingerprint_is_stable_and_excludes_runtime_fields(tmp_path: Path):
    first = config_fingerprint(
        {
            "seed": 7,
            "device": "cuda:0",
            "logging_level": "DEBUG",
            "nested": {"values": [2, 1]},
        },
        run_dir=tmp_path,
    )
    second = config_fingerprint(
        {
            "nested": {"values": [2, 1]},
            "logging_level": "INFO",
            "device": "cpu",
            "seed": 7,
        },
        run_dir=tmp_path,
    )

    assert first == second
    assert first != config_fingerprint({"seed": 8, "nested": {"values": [2, 1]}})
