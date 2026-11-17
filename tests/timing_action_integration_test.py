from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from exact.core.actions.alignment import AlignmentAction
from exact.core.entities.configs.config import ConfigModel
from exact.runs import RunLayout
from exact.utils.timing import CacheStatus, TimingLedger

FIXTURES = Path(__file__).parent / "fixtures" / "ontologies"


def test_alignment_action_twice_preserves_fresh_inference_timing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the action boundary twice without requiring a model download or accelerator."""

    calls: list[dict[str, Any]] = []

    def run_cpu_fixture_session(**kwargs: Any) -> tuple[None, Path]:
        timing_session = kwargs["timing_session"]
        fresh = not calls
        timing_session.record(
            "Alignment.Inference",
            seconds=12.0 if fresh else 0.0,
            cache_status=CacheStatus.FRESH if fresh else CacheStatus.SKIPPED,
            work_done=5 if fresh else 0,
            work_total=5,
            unit="examples",
        )
        calls.append(kwargs)
        stats_path = RunLayout.create(kwargs["output_dir_path"]).run_stats_path
        if not stats_path.exists():
            stats_path.write_text("{}\n", encoding="utf-8")
        return None, stats_path

    monkeypatch.setattr(ConfigModel, "resolve_dependencies", lambda self: None)
    monkeypatch.setattr(
        AlignmentAction,
        "_run_session",
        staticmethod(run_cpu_fixture_session),
    )
    output_dir = tmp_path / "run"
    config = ConfigModel()
    arguments = {
        "source_file_path": FIXTURES / "mini_src.owl",
        "target_file_path": FIXTURES / "mini_tgt.owl",
        "output_dir_path": output_dir,
        "configs_file_path": config,
    }

    with pytest.warns(DeprecationWarning, match="AlignmentAction.run"):
        AlignmentAction.run(**arguments)
    first_projection = (output_dir / "times.txt").read_text(encoding="utf-8")
    with pytest.warns(DeprecationWarning, match="AlignmentAction.run"):
        AlignmentAction.run(**arguments)

    ledger = TimingLedger.open(output_dir)
    sessions = [session for session in ledger.sessions() if session.command == "align"]
    inference_records = [
        next(record for record in session.stages if record.stage == "Alignment.Inference")
        for session in sessions
    ]
    inference_total = ledger.stage_totals(config_fingerprint=sessions[0].config_fingerprint)[
        "Alignment.Inference"
    ]

    assert len(calls) == 2
    assert len(sessions) == 2
    assert sessions[0].config_fingerprint == sessions[1].config_fingerprint
    assert [record.cache_status for record in inference_records] == [
        CacheStatus.FRESH,
        CacheStatus.SKIPPED,
    ]
    assert inference_total.compute_seconds == 12.0
    assert inference_total.overhead_seconds == 0.0
    assert first_projection == (output_dir / "times.txt").read_text(encoding="utf-8")

    stats = json.loads(RunLayout.open(output_dir).run_stats_path.read_text(encoding="utf-8"))
    assert stats["timing"]["run_id"] == sessions[1].run_id
    assert stats["timing"]["cumulative_compute_seconds"]["Alignment.Inference"] == 12.0
