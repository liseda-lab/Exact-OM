"""Admission preserves the complete E10 acceptance comparison and safe training."""

import copy
import json

import pytest

from tools import queue_e10_acceptance as queue


def cells():
    from tests.e15_queue_test import cells as selector_cells

    control = selector_cells()[-1]
    values = []
    for arm in queue.ARMS:
        item = copy.deepcopy(control)
        item.arm_id = arm
        item.resolved_config["selector"]["accept_training"] = queue.MODES[arm]
        item.resolved_config["matching"]["fusion"] = dict(
            enabled=True, mode="analytic_shipped", gamma=2, tau=0.5, beta=0.8
        )
        values.append(item)
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
        "supervision",
        "recipe",
        "constants",
        "hosted",
        "unpaired",
    ],
)
def test_rejects_changed_science_or_unpaired_treatments(change):
    values = cells()
    queue.validate_cells(values)
    c = values[-1].resolved_config
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
    elif change == "supervision":
        c["supervision"]["components"]["accept"] = "label_free"
    elif change == "recipe":
        c["selector"]["accept_training"] = "winner_only"
    elif change == "constants":
        c["matching"]["fusion"]["gamma"] = 3
    elif change == "unpaired":
        c["selector"]["unexpected_setting"] = True
    else:
        c["llm"]["experiment"]["gate"]["mode"] = "always"
    with pytest.raises(ValueError):
        queue.validate_cells(values)


def test_alias_preserves_treatments_dependencies_selection_and_original():
    training = dict(pool="/pool", reference="/ref", pool_sha256="p", reference_sha256="r")
    original = {
        "cases": {
            "D0_E03": {
                "candidates": {"train": {"path": "/pool", "sha256": "p"}},
                "references": {"train": {"path": "/ref", "sha256": "r"}},
            }
        },
        "steps": [
            {
                "id": "E10",
                "case": "D0",
                "additional_cases": [],
                "inherits": ["E26", "selected_E10_analytic_setting", "E05_initial"],
                "requires": ["selected_E10_analytic_setting"],
                "arms": [{"id": a} for a in queue.ARMS],
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


def measurements(tmp_path):
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
    return rows


def test_complete_family_counts_each_measured_setup_once(tmp_path):
    from tools.resume_cached_batch import load_helpers

    rows = measurements(tmp_path)
    measurement = tmp_path / "measurement.json"
    measurement.write_text("{}")
    proof = queue.family_forecast(rows, measurement, load_helpers(queue.DATA / "e10-acceptance-01"))
    assert proof["comparison_seconds"] == 450
    assert proof["estimate"]["seconds_per_unit"] == 0
    assert proof["whole_family_seconds"] == 450 + proof["inventory_seconds"]
    rows[-1]["prefix"] = True
    with pytest.raises(ValueError):
        queue.matched_measurements(rows)


@pytest.mark.parametrize("change", ["partial", "failed", "hosted", "wrong_case"])
def test_qualification_basis_rejects_incompatible_measurement(tmp_path, change):
    fit = measurements(tmp_path)[0]
    fit["name"] = "D0_E03--current_supervised"
    assert queue.qualification_allowances(fit) == {k: 150 for k in queue.KEYS}
    if change == "partial":
        fit["prefix"] = True
    elif change == "failed":
        fit["execution_status"] = "failed"
    elif change == "hosted":
        fit["new_usage"]["attempts"] = 1
    else:
        fit["case_id"] = "D1"
    with pytest.raises(ValueError):
        queue.qualification_allowances(fit)


def fits(tmp_path):
    from tests.e15_queue_test import fit_fixture

    sources, _, payload, _ = fit_fixture(tmp_path)
    rows, paths, payloads = [], [], []
    for arm in queue.ARMS:
        p = tmp_path / arm / "fitting/identity/selector.json"
        p.parent.mkdir(parents=True)
        value = copy.deepcopy(payload)
        value["fit_provenance"]["accept_training"] = queue.MODES[arm]
        p.write_text(json.dumps(value))
        rows.append({"name": "D0_E03--" + arm, "output_dir": str(tmp_path / arm)})
        paths.append(p)
        payloads.append(value)
    return sources, rows, paths, payloads


@pytest.mark.parametrize(
    "change", ["overlap", "missing_fold", "leaked_fold", "fake_model", "recipe", "missing_arm"]
)
def test_fit_evidence_rejects_contamination_missing_controls_or_wrong_recipe(tmp_path, change):
    sources, rows, paths, values = fits(tmp_path)
    assert len(queue.fitted_evidence(rows, {"sources": sources})) == 2
    value = values[-1]
    if change == "overlap":
        value["fit_provenance"]["application"]["source_ids"] = [sources[0]]
    elif change == "missing_fold":
        value["folds"].pop()
    elif change == "leaked_fold":
        value["folds"][0]["train_sources"].append(sources[0])
    elif change == "fake_model":
        value["rank_model"] = None
    elif change == "recipe":
        value["fit_provenance"]["accept_training"] = "winner_only"
    else:
        rows.pop()
    paths[-1].write_text(json.dumps(value))
    with pytest.raises(ValueError):
        queue.fitted_evidence(rows, {"sources": sources})


@pytest.mark.parametrize("change", ["training", "model", "pool", "threshold"])
def test_sizing_basis_preserves_actual_workload(tmp_path, change):
    values = cells()
    config = copy.deepcopy(values[0].resolved_config)
    config["matching"]["fusion"]["enabled"] = False
    queue.validate_qualification_basis(values, config)
    if change == "training":
        config["data"]["train_candidates"] = "different"
    elif change == "pool":
        config["data"]["pool"] = "different"
    elif change == "threshold":
        config["matching"]["threshold"] = 0.5
    else:
        config["model"] = {"different": True}
    with pytest.raises(ValueError, match="workload"):
        queue.validate_qualification_basis(values, config)
