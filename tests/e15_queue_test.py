"""Complete E15 comparisons preserve label roles, matched costs and actual fits."""

import copy
import json

import pytest

from tools import queue_e15 as queue


def cells():
    from tests.e03_queue_test import cells as calibration_cells

    values = calibration_cells()
    for item, arm in zip(values, queue.ARMS):
        item.arm_id = arm
        c = item.resolved_config
        c["supervision"]["components"] = {
            name: (
                "supervised"
                if arm == "current_supervised" and name in {"rerank", "accept"}
                else "label_free"
            )
            for name in (
                "retrieval",
                "fusion",
                "rerank",
                "llm",
                "accept",
                "calibration",
                "structure",
                "relation",
            )
        }
        c["matching"]["calibration"] = {"mode": "none", "threshold_mode": "fixed"}
        c["selector"].update(
            enabled=True,
            runtime_enabled=True,
            runtime_global_only=False,
            label_free_mode=queue.MODES[arm],
        )
    return values


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "private",
        "labels",
        "threshold",
        "calibrator",
        "training",
        "label_free",
        "supervised",
        "mode",
        "hosted",
    ],
)
def test_rejects_changed_science_or_supervision(change):
    values = cells()
    queue.validate_cells(values)
    c = values[1].resolved_config
    if change == "missing":
        values.pop()
    elif change == "private":
        c["data"]["refs"]["test"] = "private"
    elif change == "labels":
        c["supervision"]["negative_label_policy"] = "unknown"
    elif change == "threshold":
        c["matching"]["threshold"] = 0.5
    elif change == "calibrator":
        c["matching"]["calibration"]["mode"] = "platt"
    elif change == "training":
        c["data"]["train_candidates"] = None
    elif change == "label_free":
        c["supervision"]["components"]["accept"] = "supervised"
    elif change == "supervised":
        values[-1].resolved_config["supervision"]["components"]["accept"] = "label_free"
    elif change == "mode":
        c["selector"]["label_free_mode"] = "current_fallback"
    else:
        c["llm"]["experiment"]["gate"]["mode"] = "always"
    with pytest.raises(ValueError):
        queue.validate_cells(values)


def test_operational_alias_preserves_every_treatment_and_selection():
    training = {"pool": "/pool", "reference": "/ref", "pool_sha256": "p", "reference_sha256": "r"}
    original = {
        "cases": {
            "D0_E03": {
                "candidates": {"train": {"path": "/pool", "sha256": "p"}},
                "references": {"train": {"path": "/ref", "sha256": "r"}},
            }
        },
        "steps": [
            {
                "id": "E15",
                "case": "D0",
                "additional_cases": [],
                "inherits": ["E05_initial"],
                "requires": ["E03"],
                "arms": [{"id": arm} for arm in queue.ARMS],
                "selection": {"immutable": True},
            }
        ],
    }
    before = copy.deepcopy(original)
    prepared = queue.prepare_steps(original, training)
    assert original == before
    prepared["steps"][0]["case"] = "D0"
    assert prepared == original
    training["pool_sha256"] = "changed"
    with pytest.raises(ValueError, match="binding"):
        queue.prepare_steps(original, training)


def test_complete_measurements_count_setup_once_and_reject_partial(tmp_path):
    from tools.resume_cached_batch import load_helpers

    rows = []
    for i, name in enumerate(queue.KEYS):
        out = tmp_path / name
        (out / "dataset").mkdir(parents=True)
        (out / "dataset/candidate_pool_sample_manifest.json").write_text(
            json.dumps({"gold_free_summary": {"pairs": 5842}, "per_kind": {"class": "same"}})
        )
        rows.append(
            dict(
                name=name,
                case_id="D0_E03",
                status="passed",
                execution_status="complete",
                prefix=False,
                new_usage={"attempts": 0},
                processed_pairs=5842,
                dataset_rows=5842,
                output_dir=str(out),
                wall_seconds=100 * (i + 1),
                worker_measurement={"peak_rss_bytes": 1, "peak_cuda_reserved_bytes": 1},
            )
        )
    measurement = tmp_path / "measurement.json"
    measurement.write_text("{}")
    proof = queue.family_forecast(
        rows, measurement, load_helpers(queue.DATA / "e15-qualification-01")
    )
    assert proof["comparison_seconds"] == 1500
    assert proof["estimate"]["seconds_per_unit"] == 0
    assert proof["whole_family_seconds"] == 1500 + proof["inventory_seconds"]
    rows[-1]["prefix"] = True
    with pytest.raises(ValueError):
        queue.matched_measurements(rows)


def fit_fixture(tmp_path):
    sources = [f"s{i}" for i in range(2000)]
    directory = tmp_path / "control/fitting/identity"
    directory.mkdir(parents=True)
    path = directory / "selector.json"
    payload = dict(
        kind="fitted_selector",
        rank_model={"weights": [1]},
        accept_model={"weights": [1]},
        fit_provenance=dict(
            training_sources=sources,
            negative_label_policy="confirmed_negatives",
            score_calibration="none",
            application={"source_ids": ["valid"]},
        ),
        folds=[
            dict(
                fold=i,
                heldout_sources=sources[i::5],
                train_sources=sorted(set(sources) - set(sources[i::5])),
            )
            for i in range(5)
        ],
    )
    rows = [{"name": "D0_E03--current_supervised", "output_dir": str(tmp_path / "control")}]
    path.write_text(json.dumps(payload))
    return sources, path, payload, rows


@pytest.mark.parametrize(
    "change", ["overlap", "missing_fold", "leaked_fold", "fake_model", "label_free_fit"]
)
def test_fit_admission_rejects_incomplete_or_contaminated_artifacts(tmp_path, change):
    sources, path, payload, rows = fit_fixture(tmp_path)
    assert len(queue.fitted_evidence(rows, {"sources": sources})) == 1
    if change == "overlap":
        payload["fit_provenance"]["application"]["source_ids"] = [sources[0]]
    elif change == "missing_fold":
        payload["folds"].pop()
    elif change == "leaked_fold":
        payload["folds"][0]["train_sources"].append(sources[0])
    elif change == "fake_model":
        payload["rank_model"] = None
    else:
        rows[0]["name"] = "D0_E03--analytic_fixed"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        queue.fitted_evidence(rows, {"sources": sources})
