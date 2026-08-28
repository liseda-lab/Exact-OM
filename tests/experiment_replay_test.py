from __future__ import annotations

import json
from pathlib import Path

import pytest

from exact.experiments.replay import (
    ReplayValidationError,
    canonical_alignment,
    compare_replay_outputs,
    validate_e00_replay,
)
from exact.experiments.schema import ResourceConfig


def _output(
    root: Path,
    rows: list[str],
    *,
    f1: float = 0.8,
    header: str = "SrcEntity\tTgtEntity\tScore\tRelation",
) -> Path:
    alignment = root / "alignment" / "maps_global.tsv"
    alignment.parent.mkdir(parents=True)
    alignment.write_text("\n".join([header, *rows, ""]), encoding="utf-8")
    report = root / "evaluation" / "evaluation_results.json"
    report.parent.mkdir(parents=True)
    report.write_text(
        json.dumps(
            {
                "builtin": {"P": f1, "R": f1, "F1": f1},
                "meta": {"refs": {"full_reference": {"path": "fixture"}}},
            }
        ),
        encoding="utf-8",
    )
    return root


def test_replay_canonicalizes_order_and_applies_device_score_tolerance(
    tmp_path: Path,
) -> None:
    baseline = _output(
        tmp_path / "baseline",
        ["s2\tt2\t0.7\t=", "s1\tt1\t0.9\t<"],
    )
    replay = _output(
        tmp_path / "replay",
        ["s1\tt1\t0.900009\t<", "s2\tt2\t0.7\t="],
        f1=0.80009,
    )

    record = compare_replay_outputs(baseline, replay, execution_kind="gpu")

    assert record["status"] == "passed"
    assert record["canonical_order"] == [
        {"source": "s1", "target": "t1", "relation": "<"},
        {"source": "s2", "target": "t2", "relation": "="},
    ]
    with pytest.raises(ValueError, match="score drift"):
        compare_replay_outputs(baseline, replay, execution_kind="cpu")


@pytest.mark.parametrize(
    "rows, message",
    [
        (["changed\tt1\t0.9\t<", "s2\tt2\t0.7\t="], "canonical IDs"),
        (["s1\tt1\t0.9\t>", "s2\tt2\t0.7\t="], "relation labels"),
    ],
)
def test_replay_rejects_mapping_or_relation_drift(
    tmp_path: Path,
    rows: list[str],
    message: str,
) -> None:
    baseline = _output(
        tmp_path / "baseline",
        ["s1\tt1\t0.9\t<", "s2\tt2\t0.7\t="],
    )
    replay = _output(tmp_path / "replay", rows)

    with pytest.raises(ReplayValidationError, match=message) as caught:
        compare_replay_outputs(baseline, replay, execution_kind="cpu")
    persisted = json.loads(json.dumps(caught.value.as_record(), allow_nan=False))
    assert persisted["status"] == "failed"
    assert persisted["failure"]["code"] == "canonical_mapping_mismatch"


def test_replay_rejects_metric_drift_and_requires_paired_cells(tmp_path: Path) -> None:
    baseline = _output(tmp_path / "baseline", ["s\tt\t0.9\t="], f1=0.8)
    replay = _output(tmp_path / "replay", ["s\tt\t0.9\t="], f1=0.8002)
    with pytest.raises(ValueError, match="metric drift"):
        compare_replay_outputs(baseline, replay, execution_kind="cpu")

    manifest = {
        "experiment_id": "E00",
        "status": "complete",
        "arm_id": "R_0",
        "task_id": "fixture",
        "seed": 17,
        "resource": ResourceConfig(kind="cpu"),
        "fingerprint_payload": {"output_dir": str(baseline)},
    }
    with pytest.raises(ValueError, match="complete paired cells"):
        validate_e00_replay([manifest])


def test_e00_validation_accepts_prepare_cell_manifest_shape(tmp_path: Path) -> None:
    baseline = _output(tmp_path / "baseline", ["s\tt\t0.9\t="])
    replay = _output(tmp_path / "replay", ["s\tt\t0.900009\t="])

    def manifest(arm: str, output: Path) -> dict[str, object]:
        return {
            "experiment_id": "E00",
            "status": "complete",
            "arm_id": arm,
            "task_id": "fixture",
            "seed": 17,
            "resource": ResourceConfig(kind="gpu", device="0"),
            "observed_execution": {
                "device_type": "cuda",
                "device": "cuda:0",
                "device_name": "fixture-gpu",
            },
            "fingerprint_payload": {"output_dir": str(output)},
        }

    record = validate_e00_replay([manifest("R_0", baseline), manifest("baseline_replay", replay)])

    assert record["status"] == "passed"
    assert record["comparisons"][0]["execution_kind"] == "gpu"
    assert record["comparisons"][0]["observed_execution"]["device"] == "cuda:0"


def test_e00_validation_rejects_declared_observed_device_mismatch(tmp_path: Path) -> None:
    baseline = _output(tmp_path / "baseline", ["s\tt\t0.9\t="])
    replay = _output(tmp_path / "replay", ["s\tt\t0.9\t="])

    def manifest(arm: str, output: Path, observed: str) -> dict[str, object]:
        return {
            "experiment_id": "E00",
            "status": "complete",
            "arm_id": arm,
            "task_id": "fixture",
            "seed": 17,
            "resource": ResourceConfig(kind="gpu", device="0"),
            "observed_execution": {"device_type": observed, "device": observed},
            "fingerprint_payload": {"output_dir": str(output)},
        }

    with pytest.raises(ReplayValidationError) as caught:
        validate_e00_replay(
            [
                manifest("R_0", baseline, "cpu"),
                manifest("baseline_replay", replay, "cpu"),
            ]
        )
    assert caught.value.record["failure"]["code"] == "declared_observed_device_mismatch"


def test_replay_rejects_internally_conflicting_execution_metadata(tmp_path: Path) -> None:
    baseline = _output(tmp_path / "baseline", ["s\tt\t0.9\t="])
    replay = _output(tmp_path / "replay", ["s\tt\t0.9\t="])

    with pytest.raises(ReplayValidationError) as caught:
        compare_replay_outputs(
            baseline,
            replay,
            observed_execution={"kind": "gpu", "device": "cpu"},
        )

    assert caught.value.record["failure"]["code"] == ("conflicting_execution_device_metadata")


def test_relation_column_is_mandatory_and_pair_has_one_exact_relation(tmp_path: Path) -> None:
    missing_relation = _output(
        tmp_path / "missing-relation",
        ["s\tt\t0.9"],
        header="SrcEntity\tTgtEntity\tScore",
    )
    with pytest.raises(ValueError, match="relation label"):
        canonical_alignment(missing_relation / "alignment" / "maps_global.tsv")

    ambiguous = _output(
        tmp_path / "ambiguous-relation",
        ["s\tt\t0.9\t=", "s\tt\t0.8\t<"],
    )
    with pytest.raises(ValueError, match="multiple relation labels"):
        canonical_alignment(ambiguous / "alignment" / "maps_global.tsv")


def test_validate_requires_observed_execution_metadata(tmp_path: Path) -> None:
    baseline = _output(tmp_path / "baseline", ["s\tt\t0.9\t="])
    replay = _output(tmp_path / "replay", ["s\tt\t0.9\t="])
    manifests = [
        {
            "experiment_id": "E00",
            "status": "complete",
            "arm_id": arm,
            "task_id": "fixture",
            "seed": 17,
            "resource": ResourceConfig(kind="cpu"),
            "fingerprint_payload": {"output_dir": str(output)},
        }
        for arm, output in (("R_0", baseline), ("baseline_replay", replay))
    ]
    with pytest.raises(ReplayValidationError) as caught:
        validate_e00_replay(manifests)
    assert caught.value.record["failure"]["code"] == "missing_observed_execution_device"
