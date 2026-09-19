"""Protocol intervals/multiplicity and local benchmark capture/holdout validation."""

import json

import pytest

from exact.repair.api import write_artifact
from exact.repair.evaluation import holm_adjust, paired_group_effects
from exact.repair.records import ObjectiveV2, PolicyV2, RepairInputV2
from exact.repair.study import StudyCaseV2, grouped_splits, load_schedule
from tools.repair.prepare_study import prepare_study
from tools.repair.report_study import main as report_main


def paired_rows(values):
    return [
        {
            "case_id": f"c{i}",
            "arm_id": arm,
            "group_id": f"g{i}",
            "logical_status": "VERIFIED_FEASIBLE",
            "utility": value if arm == "right" else 0.0,
        }
        for i, value in enumerate(values)
        for arm in ("left", "right")
    ]


def test_group_sign_flip_is_exact_and_confidence_level_controls_interval():
    report = paired_group_effects(
        paired_rows([1.0, 2.0]),
        "left",
        "right",
        "utility",
        bootstrap_replicates=100,
        confidence_level=0.8,
    )
    assert report["paired_pvalue"] == 0.5
    assert report["sign_flip_draws"] == 4
    assert report["pvalue_method"] == "exact_group_sign_flip"
    assert report["confidence_level"] == 0.8
    assert "exploratory_95_percent_interval" not in report
    zero = paired_group_effects(paired_rows([0.0]), "left", "right", "utility")
    assert zero["paired_pvalue"] == 1.0
    assert zero["exploratory_95_percent_interval"] is None


def test_monte_carlo_is_bounded_reproducible_and_missing_rows_keep_denominator():
    rows = paired_rows([1.0] * 17)
    first = paired_group_effects(rows, "left", "right", "utility", sign_flip_replicates=37)
    again = paired_group_effects(rows, "left", "right", "utility", sign_flip_replicates=37)
    assert first == again and first["sign_flip_draws"] == 37
    assert first["pvalue_method"] == "monte_carlo_group_sign_flip_plus_one"
    assert 0 < first["paired_pvalue"] <= 1
    missing = paired_group_effects(rows[:-1], "left", "right", "utility")
    assert missing["scheduled_cases"] == 17
    assert missing["all_scheduled_status"]["right"]["NOT_RECORDED"] == 1
    for config in (
        {"confidence_level": float("nan")},
        {"confidence_level": 1},
        {"sign_flip_replicates": 0},
    ):
        with pytest.raises(ValueError):
            paired_group_effects(rows, "left", "right", "utility", **config)


def test_holm_step_down_and_unavailable_contrasts():
    assert holm_adjust({"a": 0.01, "b": 0.03, "c": 0.04, "unknown": None}) == {
        "a": 0.03,
        "b": 0.06,
        "c": 0.06,
        "unknown": None,
    }
    for value in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError):
            holm_adjust({"invalid": value})


def protocol(tmp_path):
    data = {
        "data": {
            "split_seed": 7,
            "conference": {
                "revision": "2025",
                "reference": "ra1",
                "matcher_strata": ["captured_exact_om"],
                "whole_ontology_holdout": "ekaw",
                "expected_pairs": 21,
            },
            "bio_ml": {
                "releases": ["2024_selected", "2026_whole"],
                "data_revision_2026": "pinned-data",
                "expected_whole_pairs": 3,
            },
        },
        "analysis": {
            "multiplicity": "holm",
            "bootstrap_replicates": 50,
            "bootstrap_seed": 8,
            "interval": 0.8,
            "alpha": 0.05,
        },
    }
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(data))
    return path


def capture(
    tmp_path,
    name,
    *,
    source="cmt",
    target="conference",
    source_identity="source-hash",
    target_identity="target-hash",
    split="train",
    cohort="conference_2025",
    version="2025",
    group=None,
):
    problem = RepairInputV2(
        (),
        (),
        PolicyV2(),
        source_identity=source_identity,
        target_identity=target_identity,
        source_documents=(("source-doc", "", "a" * 64),),
        target_documents=(("target-doc", "", "b" * 64),),
    )
    case = StudyCaseV2(
        name, cohort, version, group or name, problem, ObjectiveV2(()), capture_hash="captured"
    )
    path = tmp_path / f"{name}.json"
    write_artifact(path, case.to_dict())
    return {
        "case_id": name,
        "cohort": cohort,
        "source_version": version,
        "group_id": group or name,
        "artifact": path.name,
        "split": split,
        "reference": "ra1",
        "matcher_stratum": "captured_exact_om",
        "release_revision": "pinned-data",
        "source": {
            "ontology_id": source,
            "snapshot_identity": source_identity,
            "document_sha256": ["a" * 64],
        },
        "target": {
            "ontology_id": target,
            "snapshot_identity": target_identity,
            "document_sha256": ["b" * 64],
        },
    }


def prepare(tmp_path, entries):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"schema": "exact-repair/local-study-inputs/v2", "cases": entries})
    )
    schedule = tmp_path / "study.json"
    report = prepare_study(protocol(tmp_path), manifest, schedule)
    return report, schedule


def test_local_e06_preparation_keeps_explicit_splits_and_expected_counts(tmp_path):
    entry = capture(tmp_path, "case", split="development")
    report, schedule = prepare(tmp_path, [entry])
    assert report["statuses"] == {"available": 1}
    assert report["cohorts"]["conference_2025"]["protocol_expected_pairs"] == 21
    assert report["cohorts"]["conference_2025"]["available_pairs"] == 1
    cases, _, _ = load_schedule(schedule)
    assert grouped_splits(cases, seed=999)["case"] == "development"


def test_missing_captures_and_wrong_release_are_recorded_without_claimed_runs(tmp_path):
    missing = capture(tmp_path, "missing")
    (tmp_path / missing["artifact"]).unlink()
    bad = capture(
        tmp_path,
        "wrong",
        source_identity="different-source",
        target_identity="different-target",
        source="s2",
        target="t2",
        cohort="bioml_2026_whole",
        version="2024_selected",
    )
    report, schedule = prepare(tmp_path, [missing, bad])
    assert report["statuses"] == {"unavailable": 1, "invalid": 1}
    cases, _, _ = load_schedule(schedule)
    assert len(cases) == 2 and cases[0].declared_split == "train"
    assert "release" in cases[1].detail


def test_holdout_identity_aliases_and_pair_sibling_splits_cannot_leak(tmp_path):
    heldout = capture(tmp_path, "heldout", target="ekaw", split="test")
    alias = capture(
        tmp_path,
        "alias",
        source="other-source",
        target="renamed",
        source_identity="another-source",
        split="train",
    )
    report, _ = prepare(tmp_path, [heldout, alias])
    assert report["rows"][0]["status"] == "available"
    assert report["rows"][1]["status"] == "invalid"
    assert "holdout" in report["rows"][1]["detail"]
    sibling = capture(tmp_path, "sibling", split="development")
    original = capture(tmp_path, "original", split="train")
    report, _ = prepare(tmp_path, [original, sibling])
    assert report["statuses"] == {"invalid": 2}
    assert all("siblings" in row["detail"] for row in report["rows"])


def test_snapshot_and_resolved_import_hashes_are_checked(tmp_path):
    entry = capture(tmp_path, "tampered")
    entry["source"]["document_sha256"] = ["c" * 64]
    report, _ = prepare(tmp_path, [entry])
    assert report["statuses"] == {"invalid": 1}
    assert "document hashes" in report["rows"][0]["detail"]


def test_report_cli_adjusts_multiple_pairs_using_protocol(tmp_path):
    rows = paired_rows([1.0, 2.0])
    rows += [
        {**row, "arm_id": "other", "utility": row["utility"] + 1}
        for row in rows
        if row["arm_id"] == "right"
    ]
    results = tmp_path / "results.json"
    results.write_text(
        json.dumps(
            {
                "schema": "exact-repair/study/v2",
                "plan_hash": "plan",
                "scheduled": len(rows),
                "recorded": len(rows),
                "rows": rows,
            }
        )
    )
    output = tmp_path / "report.json"
    assert (
        report_main(
            [
                str(results),
                "--contrast",
                "left",
                "right",
                "--contrast",
                "left",
                "other",
                "--metric",
                "utility",
                "--protocol",
                str(protocol(tmp_path)),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    result = json.loads(output.read_text())
    assert len(result["contrasts"]) == 2 and result["multiplicity"]["method"] == "holm"
    assert all(
        row["confidence_level"] == 0.8 and row["holm_adjusted_pvalue"] is not None
        for row in result["contrasts"].values()
    )
