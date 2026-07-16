import importlib.util
import json
import sys
from pathlib import Path

from exact.runs import RunLayout, refresh_manifest


def _load_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "aggregate_results.py"
    spec = importlib.util.spec_from_file_location("aggregate_results", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_collect_runs_resolves_layout_v1_and_v2_evaluation_artifacts(
    tmp_path: Path,
) -> None:
    module = _load_module()
    runs = tmp_path / "runs"

    legacy = runs / "legacy"
    legacy.mkdir(parents=True)
    (legacy / "trial.json").write_text(
        json.dumps({"params": {"threshold": 0.7}}), encoding="utf-8"
    )
    (legacy / "evaluation_results.csv").write_text(
        "metric,value\nf1,0.8\n", encoding="utf-8"
    )

    modern = runs / "modern"
    layout = RunLayout.create(modern)
    (modern / "trial.json").write_text(
        json.dumps({"params": {"threshold": 0.8}}), encoding="utf-8"
    )
    layout.evaluation_path("csv").write_text("metric,value\nf1,0.9\n", encoding="utf-8")
    refresh_manifest(layout, run_id="fixture")

    records = module.collect_runs(runs, "evaluation_results.csv")

    assert [record.name for record in records] == ["legacy", "modern"]
    assert [record.metrics["metric.f1"] for record in records] == ["0.8", "0.9"]
