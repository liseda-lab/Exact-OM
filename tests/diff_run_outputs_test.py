import importlib.util
import json
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "tools" / "diff_run_outputs.py"
SPEC = importlib.util.spec_from_file_location("diff_run_outputs", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _write_run(root: Path, *, score: str = "0.9", seconds: float = 1.0) -> None:
    output = root / "model" / "alignment" / "default"
    output.mkdir(parents=True)
    (root / "model" / "alignment" / "src2tgt.maps_global.tsv").write_text(
        f"Src\tTgt\tScore\ns\tt\t{score}\n", encoding="utf-8"
    )
    (root / "model" / "alignment" / "src2tgt.maps_local.tsv").write_text(
        f"Src\tTgt\tScore\ns\tt\t{score}\n", encoding="utf-8"
    )
    (output / "full_explanations.json").write_text("[]\n", encoding="utf-8")
    (output / "run_stats.json").write_text(
        json.dumps({"pairs": 1, "timing": {"seconds": seconds}}),
        encoding="utf-8",
    )


def test_comparison_ignores_only_timing_blocks(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_run(baseline, seconds=1.0)
    _write_run(candidate, seconds=99.0)

    assert MODULE.compare_run_outputs(baseline, candidate) == []

    stats = candidate / "model" / "alignment" / "default" / "run_stats.json"
    stats.write_text(json.dumps({"pairs": 2, "timing": {"seconds": 99.0}}), encoding="utf-8")
    assert MODULE.compare_run_outputs(baseline, candidate) == [
        "run_stats.json: non-timing content differs"
    ]


def test_comparison_reports_byte_level_alignment_change(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_run(baseline)
    _write_run(candidate, score="0.8")

    differences = MODULE.compare_run_outputs(baseline, candidate)

    assert "src2tgt.maps_global.tsv: byte content differs" in differences
    assert "src2tgt.maps_local.tsv: byte content differs" in differences
