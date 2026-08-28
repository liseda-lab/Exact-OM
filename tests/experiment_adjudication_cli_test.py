from __future__ import annotations

import json
from pathlib import Path

from exact.experiments.adjudication import verify_sample_manifest
from exact.experiments.adjudication_cli import main


def _disagreements() -> list[dict[str, object]]:
    return [
        {
            "source_id": f"source-{index}",
            "target_id": f"target-{index}",
            "relation": "equivalence",
            "task_id": "fixture",
            "entity_kind": "class",
            "disagreement_group": "arm_only" if index < 2 else "both",
            "score_band": "low" if index % 2 == 0 else "high",
            "evidence": {"labels": [f"concept {index}"]},
        }
        for index in range(4)
    ]


def _completed_annotations(sample: dict[str, object], path: Path) -> Path:
    sample_hash = str(sample["sample_hash"])
    cases = sample["cases"]
    assert isinstance(cases, list)
    rows = ["sample_hash\tcase_id\tannotator_id\trole\tlabel\tnotes"]
    for index, case in enumerate(cases):
        assert isinstance(case, dict)
        label = "match" if index % 2 == 0 else "nonmatch"
        for annotator in ("domain-a", "domain-b"):
            rows.append(f"{sample_hash}\t{case['case_id']}\t{annotator}\tannotator\t{label}\t")
    path.write_text("\n".join([*rows, ""]), encoding="utf-8")
    return path


def test_cli_export_import_analyze_is_hash_bound_and_deterministic(
    tmp_path: Path,
    capsys,
) -> None:
    disagreements = tmp_path / "disagreements.json"
    disagreements.write_text(json.dumps(_disagreements()), encoding="utf-8")
    annotation_template = tmp_path / "annotations.tsv"

    assert (
        main(
            [
                "export",
                "--disagreements",
                str(disagreements),
                "--output",
                str(annotation_template),
                "--target-ci-width",
                "0.4",
                "--seed",
                "23",
            ]
        )
        == 0
    )
    export_record = json.loads(capsys.readouterr().out)
    manifest = Path(export_record["sample_manifest"])
    sample = json.loads(manifest.read_text(encoding="utf-8"))
    assert verify_sample_manifest(sample) == export_record["sample_hash"]
    assert sample["sample_design"]["target_ci_width"] == 0.4
    assert sample["sample_design"]["sample_size"] == 4

    completed = _completed_annotations(sample, tmp_path / "completed.tsv")
    merged_path = tmp_path / "merged.json"
    assert (
        main(
            [
                "import",
                "--sample-manifest",
                str(manifest),
                "--annotations",
                str(completed),
                "--output",
                str(merged_path),
            ]
        )
        == 0
    )
    imported = json.loads(capsys.readouterr().out)
    assert imported["status"] == "complete"
    assert imported["inter_annotator_agreement"]["kappa"] == 1.0

    analysis_path = tmp_path / "analysis.json"
    assert (
        main(
            [
                "analyze",
                "--adjudication",
                str(merged_path),
                "--output",
                str(analysis_path),
                "--resamples",
                "100",
                "--seed",
                "9",
            ]
        )
        == 0
    )
    analyzed = json.loads(capsys.readouterr().out)
    persisted = json.loads(analysis_path.read_text(encoding="utf-8"))
    assert analyzed["precision"] == 0.5
    assert persisted["precision"] == 0.5
    assert persisted["confidence_interval"]["resamples"] == 100
    assert persisted["sample_hash"] == sample["sample_hash"]


def test_cli_import_rejects_annotation_from_another_sample_without_output(
    tmp_path: Path,
    capsys,
) -> None:
    disagreements = tmp_path / "disagreements.json"
    disagreements.write_text(json.dumps(_disagreements()), encoding="utf-8")
    template = tmp_path / "annotations.tsv"
    assert (
        main(
            [
                "export",
                "--disagreements",
                str(disagreements),
                "--output",
                str(template),
                "--target-ci-width",
                "0.4",
            ]
        )
        == 0
    )
    capsys.readouterr()
    manifest = template.with_suffix(".tsv.manifest.json")
    sample = json.loads(manifest.read_text(encoding="utf-8"))
    case = sample["cases"][0]
    annotations = tmp_path / "wrong-sample.tsv"
    annotations.write_text(
        "sample_hash\tcase_id\tannotator_id\trole\tlabel\n"
        f"{'f' * 64}\t{case['case_id']}\tdomain-a\tannotator\tmatch\n",
        encoding="utf-8",
    )
    output = tmp_path / "must-not-exist.json"

    assert (
        main(
            [
                "import",
                "--sample-manifest",
                str(manifest),
                "--annotations",
                str(annotations),
                "--output",
                str(output),
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert "sample_hash mismatch" in captured.err
    assert not output.exists()
