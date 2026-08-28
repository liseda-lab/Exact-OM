from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from exact.core.entities.configs.config import ConfigModel
from exact.core.entities.kinds import EntityKind
from exact.experiments import reporting
from exact.experiments.reporting import (
    cell_metric_rows,
    inspect_dataset_task,
    metric_reports,
    paired_bootstrap_reports,
)
from exact.utils.provenance import file_provenance


class _KnowledgeSource:
    def __init__(self, origin: Path, counts: dict[EntityKind, int]):
        self.origin = origin
        self._counts = counts

    def entities(self, kind: EntityKind = EntityKind.CLASS):
        return [f"{kind.value}-{index}" for index in range(self._counts.get(kind, 0))]


def _write_table(path: Path, header: str, rows: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([header, *rows, ""]), encoding="utf-8")
    return path


def test_inventory_uses_materialized_counts_and_declared_split(tmp_path: Path, monkeypatch) -> None:
    source_path = tmp_path / "source.owl"
    target_path = tmp_path / "target.owl"
    source_path.write_text("source", encoding="utf-8")
    target_path.write_text("target", encoding="utf-8")
    reference = _write_table(
        tmp_path / "valid.tsv",
        "SrcEntity\tTgtEntity\tRelation\tSrcKind\tTgtKind",
        ["s1\tt1\t=\tclass\tclass", "s2\tt2\t=\tclass\tclass"],
    )
    candidates = _write_table(
        tmp_path / "valid.cands.tsv",
        "SrcEntity\tTgtEntity\tTgtCandidates",
        ["s1\tt1\t['t1', 'x']", "s2\tt2\t['t2']"],
    )
    resolved = SimpleNamespace(
        source=source_path,
        target=target_path,
        full_reference=reference,
        candidates=candidates,
    )
    monkeypatch.setattr(reporting, "resolve_alignment_inputs", lambda **_kwargs: resolved)
    monkeypatch.setattr(reporting, "infer_format", lambda *_args, **_kwargs: "owl")
    sources = {
        source_path: _KnowledgeSource(source_path, {EntityKind.CLASS: 4}),
        target_path: _KnowledgeSource(target_path, {EntityKind.CLASS: 5}),
    }
    monkeypatch.setattr(
        reporting,
        "resolve_source",
        lambda path, **_kwargs: sources[Path(path)],
    )

    row = inspect_dataset_task(
        ConfigModel(),
        experiment_id="E00",
        task_id="development",
        stage="screen",
        split_role="development",
        reference_role="valid",
        reference_completeness="complete",
        capabilities=["train_reference"],
        split_availability=["train", "valid", "test"],
    )

    assert row["entity_counts"]["source"]["class"] == 4
    assert row["entity_counts"]["target"]["class"] == 5
    assert row["reference_counts"]["by_kind_relation"] == {"class|equivalence": 2}
    assert row["candidate_pool"]["candidate_pairs"] == 3
    assert row["candidate_pool"]["coverage"] == 0.5
    assert row["split_availability"] == ["test", "train", "valid"]


def test_metric_macros_do_not_hide_missing_task_seed_cells() -> None:
    records = [
        {
            "experiment_id": "E01",
            "arm_id": "candidate",
            "task_id": "one",
            "seed": 1,
            "status": "complete",
            "metrics": {"class.F1": 0.8},
        },
        {
            "experiment_id": "E01",
            "arm_id": "candidate",
            "task_id": "two",
            "seed": 1,
            "status": "complete",
            "metrics": {"class.MRR": 0.7},
        },
    ]

    long_rows, macros = metric_reports(records)

    assert len([row for row in long_rows if row["availability"] == "available"]) == 2
    assert any(
        row["metric_endpoint"] == "candidate_recall" and row["availability"] == "unavailable"
        for row in long_rows
    )
    f1 = next(row for row in macros if row["metric"] == "class.F1")
    assert f1["entity_kind"] == "all"
    assert f1["relation"] == "all"
    assert f1["complete"] is False
    assert f1["macro_over_tasks"] is None
    assert f1["missing_cells"] == ["two/seed-1"]


def test_mandatory_metric_families_preserve_raw_values_and_availability() -> None:
    record = {
        "experiment_id": "E17",
        "arm_id": "stack_all",
        "task_id": "fixture",
        "seed": 3,
        "status": "complete",
        "metrics": {
            "P": 0.8,
            "R": 0.7,
            "F1": 0.746,
            "local.MRR": 0.9,
            "local.Hits@1": 0.85,
        },
        "candidate_recall": 0.95,
        "coverage": 0.75,
        "abstention_rate": 0.25,
        "wall_seconds": 1.5,
        "peak_memory_kb": 1024,
        "llm_usage": {
            "calls": 2,
            "prompt_tokens": 10,
            "completion_tokens": 4,
            "counter_availability": {
                "total_tokens": {
                    "status": "unavailable",
                    "reported_observations": 0,
                }
            },
        },
    }

    rows = cell_metric_rows(record)
    available = {
        (row["metric_family"], row["metric_endpoint"]): row
        for row in rows
        if row["availability"] == "available"
    }

    for endpoint in (
        ("candidate_recall", "candidate_recall"),
        ("coverage", "coverage"),
        ("abstention", "abstention_rate"),
        ("ranking", "MRR"),
        ("ranking", "Hits@1"),
        ("wall_time", "wall_seconds"),
        ("peak_memory", "peak_memory_kb"),
        ("llm_usage", "llm.calls"),
        ("llm_usage", "llm.input_tokens"),
        ("llm_usage", "llm.output_tokens"),
    ):
        assert endpoint in available
    assert available[("candidate_recall", "candidate_recall")]["source"] == "run_manifest"
    assert {"P", "R", "F1", "local.MRR", "local.Hits@1"}.issubset({row["metric"] for row in rows})
    total = next(row for row in rows if row["metric_endpoint"] == "llm.total_tokens")
    assert total["availability"] == "unavailable"
    assert total["availability_reason"] == "runtime_counter_unavailable"
    assert total["availability_detail"]["reported_observations"] == 0


def test_metric_applicability_distinguishes_not_applicable_from_missing() -> None:
    rows = cell_metric_rows(
        {
            "experiment_id": "E01",
            "arm_id": "baseline",
            "task_id": "fixture",
            "seed": 1,
            "status": "complete",
            "metrics": {"F1": 0.5},
            "llm_required": False,
            "metric_applicability": {"ranking": False, "peak_memory": False},
        }
    )
    availability = {
        row["metric_endpoint"]: row["availability"]
        for row in rows
        if row["availability"] != "available"
    }
    assert availability["MRR"] == "not_applicable"
    assert availability["Hits@1"] == "not_applicable"
    assert availability["peak_memory_kb"] == "not_applicable"
    assert availability["llm.calls"] == "not_applicable"
    assert availability["candidate_recall"] == "unavailable"


def test_llm_counter_availability_contradictions_fail_closed() -> None:
    with pytest.raises(ValueError, match="marked available but has no reported counter"):
        cell_metric_rows(
            {
                "experiment_id": "E07",
                "arm_id": "listwise",
                "task_id": "fixture",
                "seed": 1,
                "status": "complete",
                "metrics": {"F1": 0.5},
                "llm_usage": {
                    "counter_availability": {"calls": {"status": "available", "value": 2}}
                },
            }
        )


def test_per_source_index_includes_paper_global_alignment(tmp_path: Path) -> None:
    from exact.experiments.harness import _write_per_source_index

    stage_root = tmp_path / "suite" / "confirm"
    run = stage_root / "runs" / "E17" / "stack_all" / "task" / "seed-3"
    paper_alignment = _write_table(
        run / "alignment" / "paper.maps_global.tsv",
        "SrcEntity\tTgtEntity\tScore\tRelation",
        ["s\tt\t0.9\t="],
    )
    _write_per_source_index(
        stage_root,
        [
            {
                "experiment_id": "E17",
                "arm_id": "stack_all",
                "task_id": "task",
                "seed": 3,
                "status": "complete",
            }
        ],
    )

    payload = json.loads((stage_root / "per_source_outputs.json").read_text(encoding="utf-8"))
    assert payload["runs"][0]["artifacts"] == [str(paper_alignment.relative_to(stage_root))]


def _run_artifacts(root: Path, reference: Path, rows: list[str]) -> Path:
    _write_table(
        root / "alignment" / "maps_global.tsv",
        "SrcEntity\tTgtEntity\tScore",
        rows,
    )
    report = {
        "builtin": {"F1": 1.0},
        "meta": {"refs": {"full_reference": file_provenance(reference)}},
    }
    report_path = root / "evaluation" / "evaluation_results.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return root


def test_production_bootstrap_caller_pairs_sources_and_seeds(tmp_path: Path) -> None:
    reference = _write_table(
        tmp_path / "reference.tsv",
        "SrcEntity\tTgtEntity",
        ["s1\tt1", "s2\tt2"],
    )
    baseline = _run_artifacts(
        tmp_path / "baseline",
        reference,
        ["s1\tt1\t0.9", "s2\twrong\t0.8"],
    )
    candidate = _run_artifacts(
        tmp_path / "candidate",
        reference,
        ["s1\tt1\t0.9", "s2\tt2\t0.8"],
    )
    records = [
        {
            "experiment_id": "E01",
            "arm_id": "baseline",
            "task_id": "task",
            "seed": 17,
            "status": "complete",
            "output_dir": str(baseline),
        },
        {
            "experiment_id": "E01",
            "arm_id": "candidate",
            "task_id": "task",
            "seed": 17,
            "status": "complete",
            "output_dir": str(candidate),
        },
    ]

    reports = paired_bootstrap_reports(
        records,
        {"E01": [("primary", "baseline", "candidate")]},
        resamples=500,
        seed=9,
    )

    assert len(reports) == 1
    assert reports[0]["status"] == "complete"
    assert reports[0]["n_units"] == 2
    assert reports[0]["delta"] == 0.5
    assert reports[0]["ci_low"] <= reports[0]["delta"] <= reports[0]["ci_high"]
