"""Predeclared calibration population, dependency and whole-family cost guards."""

import copy
import json
from types import SimpleNamespace

import pytest

from tools import queue_e03 as queue


def cells():
    from tests.e08_queue_test import cell

    result = []
    for arm in queue.ARMS:
        item = cell(arm)
        item.task_id = "D0_E03-global_alignment"
        c = item.resolved_config
        c["data"]["train_candidates"] = "/train.tsv"
        c["supervision"]["negative_label_policy"] = "confirmed_negatives"
        c["supervision"]["components"]["calibration"] = (
            "supervised" if arm in {"platt", "isotonic"} else "label_free"
        )
        c["matching"]["calibration"] = {
            "mode": arm if arm in {"platt", "isotonic"} else "none",
            "threshold_mode": "otsu" if arm == "distribution_threshold" else "fixed",
        }
        result.append(item)
    return result


def test_requires_complete_primary_comparison_and_correct_roles():
    values = cells()
    queue.validate_cells(values)
    for change in ("missing", "private", "labels", "threshold", "calibrator", "training"):
        invalid = copy.deepcopy(values)
        c = invalid[1].resolved_config
        if change == "missing":
            invalid.pop()
        elif change == "private":
            c["data"]["refs"]["test"] = "private"
        elif change == "labels":
            c["supervision"]["negative_label_policy"] = "unknown"
        elif change == "threshold":
            c["matching"]["threshold"] = 0.5
        elif change == "calibrator":
            c["matching"]["calibration"]["mode"] = "none"
        else:
            c["data"]["train_candidates"] = None
        with pytest.raises(ValueError):
            queue.validate_cells(invalid)


def test_optional_diagnostic_follows_unchanged_primary_selection():
    states = {"screen": {"status": "planned"}, "confirm": {"status": "planned"}}
    primary = dict(
        id="E03",
        case="D0",
        additional_cases=[],
        inherits=["E05_initial"],
        requires=["E00"],
        selection={"decisions": [{"candidates": list(queue.ARMS[1:])}]},
        arms=[
            {"id": arm, "overlay": {"matching": {"calibration": {"mode": "none"}}}}
            for arm in [*queue.ARMS, "fp_only"]
        ],
        readiness={arm: copy.deepcopy(states) for arm in [*queue.ARMS, "fp_only"]},
    )
    original = {"steps": [primary, {"id": "E24", "readiness": {"a": copy.deepcopy(states)}}]}
    original["cases"] = {"D0": {"candidates": {}, "references": {}}}
    before = copy.deepcopy(original)
    saved = queue.prepare_steps(
        original,
        {"pool": "/pool", "reference": "/train", "pool_sha256": "p", "reference_sha256": "r"},
    )
    assert original == before
    main, diagnostic = saved["steps"][:2]
    assert main["selection"] == primary["selection"]
    assert [a["id"] for a in main["arms"]] == list(queue.ARMS)
    assert diagnostic["id"] == "E03-fp-only" and "E03" in diagnostic["requires"]
    assert "E03" in diagnostic["inherits"]
    assert "matching" not in diagnostic["arms"][0]["overlay"]
    assert diagnostic["readiness"]["fp_only"]["screen"]["status"] == "blocked_input_resolution"


def test_whole_family_uses_each_full_measured_arm_once(tmp_path):
    from tools.resume_cached_batch import load_helpers

    rows = []
    for index, key in enumerate(queue.KEYS):
        out = tmp_path / key
        (out / "dataset").mkdir(parents=True)
        (out / "dataset/candidate_pool_sample_manifest.json").write_text(
            json.dumps({"gold_free_summary": {"pairs": 5842}, "per_kind": {"class": "same"}})
        )
        rows.append(
            dict(
                name=key,
                case_id="D0_E03",
                status="passed",
                execution_status="complete",
                prefix=False,
                new_usage={"attempts": 0},
                processed_pairs=5842,
                dataset_rows=5842,
                output_dir=str(out),
                wall_seconds=100 * (index + 1),
                worker_measurement={"peak_rss_bytes": 1, "peak_cuda_reserved_bytes": 1},
            )
        )
    h = load_helpers(queue.DATA / "e03-qualification-01")
    measurement = tmp_path / "measurement.json"
    measurement.write_text("{}")
    proof = queue.family_forecast(rows, measurement, h)
    assert proof["comparison_seconds"] == 1500
    assert (
        proof["inventory_seconds"] == 1.5 * queue.read(queue.CONTROL)["measurement"]["wall_seconds"]
    )
    assert proof["estimate"]["seconds_per_unit"] == 0
    rows.pop()
    with pytest.raises(ValueError):
        queue.family_forecast(rows, measurement, h)


def test_bounded_training_preserves_labels_and_disjoint_groups(tmp_path):
    import pandas as pd

    from tools.run_experiment_validation import bounded_training

    class Binding:
        def __init__(self, path):
            self.path = path

        def verify(self, root):
            return self.path

    pool = tmp_path / "pool.tsv"
    reference = tmp_path / "train.tsv"
    universe = tmp_path / "valid.sources.txt"
    pd.DataFrame(
        [(str(i), str(i) + "p", 1) for i in range(8)]
        + [(str(i), str(i) + "n", 0) for i in range(8)],
        columns=["Src", "Tgt", "confirmed_label"],
    ).to_csv(pool, sep="\t", index=False)
    pd.DataFrame([(str(i), str(i) + "p") for i in range(8)], columns=["Src", "Tgt"]).to_csv(
        reference, sep="\t", index=False
    )
    universe.write_text("independent\n")
    case = SimpleNamespace(
        candidates={"train": Binding(pool)},
        references={"train": Binding(reference)},
        source_universe=Binding(universe),
    )
    a = bounded_training(case, tmp_path, tmp_path / "one", limit=3)
    b = bounded_training(case, tmp_path, tmp_path / "two", limit=3)
    assert a["sources"] == b["sources"] and a["pairs"] == 6
    assert a["pool_sha256"] == b["pool_sha256"]
    assert set(pd.read_csv(a["pool"], sep="\t").confirmed_label) == {0, 1}


def test_missing_or_reporting_contaminated_fit_cannot_admit(tmp_path):
    rows = []
    sources = [f"s{i}" for i in range(2000)]
    for arm in ("platt", "isotonic"):
        output = tmp_path / arm
        (output / "fitting").mkdir(parents=True)
        payload = {
            "kind": "score_calibrator",
            "calibrator": {"mode": arm},
            "fit_provenance": {
                "training_sources": sources,
                "negative_label_policy": "confirmed_negatives",
                "application": {"source_ids": ["valid"]},
            },
            "oof_predictions": [{"source": s, "fold": i % 5} for i, s in enumerate(sources)],
        }
        (output / "fitting/score_calibrator.json").write_text(json.dumps(payload))
        rows.append({"name": "D0_E03--" + arm, "output_dir": str(output)})
    assert len(queue.fitted_evidence(rows, {"sources": sources})) == 2
    payload["fit_provenance"]["application"]["source_ids"] = [sources[0]]
    (output / "fitting/score_calibrator.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="OOF"):
        queue.fitted_evidence(rows, {"sources": sources})
    with pytest.raises(ValueError, match="Both"):
        queue.fitted_evidence([], {"sources": sources})
