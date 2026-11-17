from __future__ import annotations

import json
from pathlib import Path

from exact.delivery.cli.run import main
from exact.runs import ExplanationStore, RunLayout, RunManifest, refresh_manifest


def _make_run(tmp_path: Path) -> tuple[RunLayout, Path, Path]:
    layout = RunLayout.create(tmp_path / "run")
    store = ExplanationStore(layout.explanations_dir, run_id="session-1")
    store.append(
        [
            {"src_iri": "source-a", "tgt_iri": "target-1", "score": 0.9},
            {"src_iri": "source-b", "tgt_iri": "target-2", "score": 0.8},
        ]
    )
    checkpoint = layout.checkpoints_dir / "inference_old.json"
    checkpoint.write_text("{}", encoding="utf-8")
    foreign = layout.checkpoints_dir / "readme.txt"
    foreign.write_text("foreign", encoding="utf-8")
    manifest = RunManifest.create(layout, run_id="session-1")
    manifest.register(checkpoint, kind="checkpoint", checksum=False)
    manifest.write()
    refresh_manifest(layout, run_id="session-1", manifest=manifest)
    return layout, checkpoint, foreign


def test_run_info_is_human_readable(tmp_path: Path, capsys) -> None:
    layout, _, _ = _make_run(tmp_path)

    assert main(["info", str(layout.root)]) == 0

    output = capsys.readouterr().out
    assert f"Run: {layout.root}" in output
    assert "Layout: v2 (manifest schema v1)" in output
    assert "Sessions: 1 (session-1)" in output
    assert "explanations:" in output


def test_run_clean_preview_and_foreign_file_safety(tmp_path: Path, capsys) -> None:
    layout, checkpoint, foreign = _make_run(tmp_path)

    assert main(["clean", str(layout.root), "--dry-run"]) == 0
    assert checkpoint.exists()
    assert "Would remove 1 file(s)" in capsys.readouterr().out

    assert main(["clean", str(layout.root)]) == 0
    assert not checkpoint.exists()
    assert foreign.read_text(encoding="utf-8") == "foreign"
    assert "Removed 1 file(s)" in capsys.readouterr().out


def test_run_export_filters_one_source(tmp_path: Path, capsys) -> None:
    layout, _, _ = _make_run(tmp_path)
    output = tmp_path / "selected.json"

    assert (
        main(
            [
                "export",
                str(layout.root),
                "--src",
                "source-b",
                "--format",
                "json",
                "--output",
                str(output),
            ]
        )
        == 0
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert [record["src_iri"] for record in payload] == ["source-b"]
    assert f"Exported explanations to {output}" in capsys.readouterr().out
