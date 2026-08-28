from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from exact.experiments.adjudication import (
    adjudication_sample_size,
    build_blinded_sample,
    import_annotations,
    inverse_probability_weighted_precision,
    merge_adjudications,
    verify_sample_manifest,
    write_blinded_sample,
)


def _population() -> list[dict[str, object]]:
    return [
        {
            "source_id": f"s{index}",
            "target_id": f"t{index}",
            "relation": "equivalence",
            "task_id": "task-a",
            "entity_kind": "class",
            "disagreement_group": "arm_only" if index < 2 else "both",
            "score_band": "low" if index < 2 else "high",
            "evidence": {"labels": [f"concept {index}"]},
        }
        for index in range(4)
    ]


def test_blinded_sample_is_deterministic_and_export_contains_no_arm(tmp_path: Path) -> None:
    sample = build_blinded_sample(_population(), sample_size=2, seed=17)
    assert sample == build_blinded_sample(_population(), sample_size=2, seed=17)
    assert sample["status"] == "awaiting_annotation"
    assert all(case["inclusion_probability"] == 0.5 for case in sample["cases"])

    table, manifest = write_blinded_sample(sample, tmp_path / "annotations.tsv")

    assert table.is_file() and manifest.is_file()
    assert "arm" not in table.read_text(encoding="utf-8").lower()
    with pytest.raises(ValueError, match="exposes arm"):
        build_blinded_sample([{**_population()[0], "arm_id": "candidate"}], sample_size=1, seed=17)
    with pytest.raises(ValueError, match="nested.score"):
        build_blinded_sample(
            [{**_population()[0], "evidence": {"nested": {"score": 0.9}}}],
            sample_size=1,
            seed=17,
        )
    with pytest.raises(ValueError, match="nested.score_band"):
        build_blinded_sample(
            [{**_population()[0], "evidence": {"nested": {"score_band": "high"}}}],
            sample_size=1,
            seed=17,
        )


def test_annotation_import_merge_and_ipw_never_invent_labels(tmp_path: Path) -> None:
    sample = build_blinded_sample(_population(), sample_size=2, seed=17)
    sample_hash = sample["sample_hash"]
    first, second = [case["case_id"] for case in sample["cases"]]
    annotations = tmp_path / "completed.tsv"
    annotations.write_text(
        "sample_hash\tcase_id\tannotator_id\trole\tlabel\tnotes\n"
        f"{sample_hash}\t{first}\ta1\tannotator\tmatch\t\n"
        f"{sample_hash}\t{first}\ta2\tannotator\tmatch\t\n"
        f"{sample_hash}\t{second}\ta1\tannotator\tnonmatch\t\n"
        f"{sample_hash}\t{second}\ta2\tannotator\tmatch\t\n",
        encoding="utf-8",
    )
    imported = import_annotations(annotations, sample)
    pending = merge_adjudications(sample, imported)
    assert pending["status"] == "pending"
    with pytest.raises(ValueError, match="completed adjudication"):
        inverse_probability_weighted_precision(pending)

    adjudicated = tmp_path / "adjudicated.tsv"
    adjudicated.write_text(
        annotations.read_text(encoding="utf-8")
        + f"{sample_hash}\t{second}\texpert\tadjudicator\tnonmatch"
        "\tresolved disagreement\n",
        encoding="utf-8",
    )
    merged = merge_adjudications(sample, import_annotations(adjudicated, sample))
    result = inverse_probability_weighted_precision(merged, resamples=200, seed=9)
    replay = inverse_probability_weighted_precision(merged, resamples=200, seed=9)

    assert merged["status"] == "complete"
    assert result["precision"] == pytest.approx(0.5)
    assert result["weighted_predictions"] == pytest.approx(4.0)
    assert result["effective_sample_size"] == pytest.approx(2.0)
    assert result == replay
    assert result["raw_precision"] == pytest.approx(0.5)
    assert result["confidence_interval"]["low"] <= result["precision"]
    assert result["confidence_interval"]["high"] >= result["precision"]
    assert merged["inter_annotator_agreement"]["exact_agreement"] == pytest.approx(0.5)


def test_annotation_import_rejects_unknown_case_and_duplicate_annotator(tmp_path: Path) -> None:
    sample = build_blinded_sample(_population(), sample_size=1, seed=1)
    case_id = sample["cases"][0]["case_id"]
    sample_hash = sample["sample_hash"]
    duplicate = tmp_path / "duplicate.tsv"
    duplicate.write_text(
        "sample_hash\tcase_id\tannotator_id\tlabel\n"
        f"{sample_hash}\t{case_id}\ta1\tmatch\n"
        f"{sample_hash}\t{case_id}\ta1\tmatch\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate annotation"):
        import_annotations(duplicate, sample)

    unknown = tmp_path / "unknown.tsv"
    unknown.write_text(
        "sample_hash\tcase_id\tannotator_id\tlabel\n" f"{sample_hash}\tunknown\ta1\tmatch\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown adjudication"):
        import_annotations(unknown, sample)


def test_required_stratification_is_order_independent_and_hides_strata() -> None:
    population = _population()
    forward = build_blinded_sample(population, sample_size=2, seed=23)
    reverse = build_blinded_sample(list(reversed(population)), sample_size=2, seed=23)

    assert forward == reverse
    assert {case["stratum_id"] for case in forward["cases"]} == {
        row["stratum_id"] for row in forward["sampling"]["strata"]
    }
    serialized = str(forward["cases"]).lower()
    assert "score_band" not in serialized
    assert "disagreement_group" not in serialized


def test_sample_design_enforces_minimum_fifty_and_records_width() -> None:
    design = adjudication_sample_size(100, target_ci_width=0.2)
    assert design["sample_size"] >= 50
    assert design["target_ci_width"] == 0.2
    assert design["ci_width_definition"] == "two_sided_total_width"

    population = [
        {
            **_population()[0],
            "source_id": f"s-{index}",
            "target_id": f"t-{index}",
        }
        for index in range(60)
    ]
    with pytest.raises(ValueError, match="at least 50"):
        build_blinded_sample(population, sample_size=49, seed=1)


def test_manifest_hash_and_annotation_hash_are_verified(tmp_path: Path) -> None:
    sample = build_blinded_sample(_population(), sample_size=2, seed=17)
    assert verify_sample_manifest(sample) == sample["sample_hash"]
    tampered = {**sample, "seed": 18}
    with pytest.raises(ValueError, match="sample hash mismatch"):
        verify_sample_manifest(tampered)
    with pytest.raises(ValueError, match="sample hash mismatch"):
        write_blinded_sample(tampered, tmp_path / "tampered.tsv")

    case_id = sample["cases"][0]["case_id"]
    wrong_hash = tmp_path / "wrong-hash.tsv"
    wrong_hash.write_text(
        "sample_hash\tcase_id\tannotator_id\tlabel\n" f"{'f' * 64}\t{case_id}\ta1\tmatch\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="annotation sample_hash mismatch"):
        import_annotations(wrong_hash, sample)

    semantic_tamper = json.loads(json.dumps(sample))
    semantic_tamper["cases"][0]["inclusion_probability"] = 0.125
    unhashed = dict(semantic_tamper)
    unhashed.pop("sample_hash")
    semantic_tamper["sample_hash"] = hashlib.sha256(
        json.dumps(
            unhashed,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    with pytest.raises(ValueError, match="invalid inclusion_probability"):
        verify_sample_manifest(semantic_tamper)


def test_adjudicator_requires_two_primary_annotations_first(tmp_path: Path) -> None:
    sample = build_blinded_sample(_population(), sample_size=1, seed=5)
    sample_hash = sample["sample_hash"]
    case_id = sample["cases"][0]["case_id"]
    path = tmp_path / "premature-adjudicator.tsv"
    path.write_text(
        "sample_hash\tcase_id\tannotator_id\trole\tlabel\n"
        f"{sample_hash}\t{case_id}\ta1\tannotator\tuncertain\n"
        f"{sample_hash}\t{case_id}\texpert\tadjudicator\tmatch\n",
        encoding="utf-8",
    )
    imported = import_annotations(path, sample)
    with pytest.raises(ValueError, match="requires two primary annotations first"):
        merge_adjudications(sample, imported)
