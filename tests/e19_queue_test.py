"""E19 admission preserves paired design, fitting roles and cumulative costs."""

import copy
import json
from pathlib import Path

import pytest

from tools import queue_e19 as queue


def cells():
    from tests.e10_acceptance_queue_test import cells as prior

    result = []
    for arm in queue.ARMS:
        cell = copy.deepcopy(prior()[0])
        cell.arm_id = arm
        c = cell.resolved_config
        c["selector"]["enabled"] = False
        c["selector"]["runtime_enabled"] = None
        c["selector"]["runtime_global_only"] = None
        for key in ("rerank", "accept"):
            c["supervision"]["components"][key] = "label_free"
        c["supervision"]["components"]["fusion"] = (
            "label_free" if arm == "analytic_shipped" else "supervised"
        )
        c["matching"]["fusion"]["mode"] = arm
        result.append(cell)
    return result


@pytest.mark.parametrize(
    "change", ["missing", "private", "threshold", "head", "negative", "hosted", "unpaired", "role"]
)
def test_rejects_changed_science(change):
    rows = cells()
    queue.validate_cells(rows)
    c = rows[-1].resolved_config
    if change == "missing":
        rows.pop()
    elif change == "private":
        c["data"]["refs"]["test"] = "private"
    elif change == "threshold":
        c["matching"]["threshold"] = 0.6
    elif change == "head":
        c["matching"]["fusion"]["artifact"] = "old-head"
    elif change == "negative":
        c["supervision"]["negative_label_policy"] = "unknown"
    elif change == "hosted":
        c["llm"]["experiment"]["gate"]["mode"] = "always"
    elif change == "role":
        c["supervision"]["components"]["fusion"] = "label_free"
    else:
        c["model"] = "different"
    with pytest.raises(ValueError):
        queue.validate_cells(rows)


def measurements(tmp_path):
    rows = []
    for i, name in enumerate(queue.KEYS):
        root = tmp_path / name
        (root / "dataset").mkdir(parents=True)
        (root / "dataset/candidate_pool_sample_manifest.json").write_text(
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
                output_dir=str(root),
                wall_seconds=100 * (i + 1),
                worker_measurement={"peak_rss_bytes": 1, "peak_cuda_reserved_bytes": 1},
            )
        )
    return rows


def test_counts_setup_once(tmp_path):
    rows = measurements(tmp_path)
    m = tmp_path / "measurement.json"
    m.write_text("{}")
    proof = queue.family_forecast(rows, m, queue.load_helpers(queue.DATA / "e19-qualification-01"))
    assert proof["comparison_seconds"] == 900
    assert proof["estimate"]["seconds_per_unit"] == 0
    rows[-1]["prefix"] = True
    with pytest.raises(ValueError):
        queue.matched_measurements(rows)


@pytest.mark.parametrize(
    "change", ["missing", "overlap", "fold", "mode", "weight", "constants", "pairs", "control"]
)
def test_rejects_invalid_fit(tmp_path, change):
    rows = measurements(tmp_path)
    sources = [f"s{i}" for i in range(2000)]
    training = dict(sources=sources, pairs=2000)
    folds = [
        dict(
            fold=i,
            heldout_sources=sources[i::5],
            train_sources=sorted(set(sources) - set(sources[i::5])),
        )
        for i in range(5)
    ]
    for row in rows[1:]:
        path = Path(row["output_dir"]) / "fitting/identity/fusion.json"
        path.parent.mkdir(parents=True)
        value = dict(
            mode=row["name"].split("--")[1],
            seed=17,
            negative_label_policy="confirmed_negatives",
            feature_schema=["label", "struct"],
            fit_provenance=dict(
                folds=folds, application={"source_ids": ["dev"]}, regularization=0.001
            ),
            oof_predictions=[{"Src": s} for s in sources],
            parameters=dict(tau=0.5, gamma=2, multipliers={"label": 1, "struct": 1}),
            weights={"label": 1, "struct": 1},
        )
        path.write_text(json.dumps(value))
    assert len(queue.fitted_evidence(rows, training)) == 2
    path = Path(rows[1]["output_dir"]) / "fitting/identity/fusion.json"
    value = json.loads(path.read_text())
    if change == "missing":
        rows.pop()
    elif change == "overlap":
        value["fit_provenance"]["application"]["source_ids"] = [sources[0]]
    elif change == "fold":
        value["fit_provenance"]["folds"].pop()
    elif change == "mode":
        value["mode"] = "learned_global"
    elif change == "weight":
        value["parameters"]["multipliers"]["label"] = -1
    elif change == "constants":
        value["parameters"]["tau"] = 0.8
    elif change == "pairs":
        value["oof_predictions"].pop()
    else:
        control = Path(rows[0]["output_dir"]) / "fitting/fusion.json"
        control.parent.mkdir()
        control.write_text(json.dumps(value))
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        queue.fitted_evidence(rows, training)


def test_latest_budget_and_alias():
    from exact.core.entities.configs.yaml_io import load_yaml_mapping

    original = load_yaml_mapping(queue.PARENT / "campaign.lock.yaml")
    training = queue.read(queue.DATA / "e03-qualification-01/training.json")
    result = queue.prepare_steps(original, training)
    next(s for s in result["steps"] if s["id"] == "E19")["case"] = "D0"
    assert result == original
    state = queue.read(queue.BUDGET)
    queue.validate_budget(state, original)
    state["limits"]["tokens_cap"] += 1
    with pytest.raises(ValueError):
        queue.validate_budget(state, original)
    state = queue.read(queue.BUDGET)
    state["work"]["uncertain/request"] = dict(status="reserved")
    with pytest.raises(ValueError):
        queue.validate_budget(state, original)


@pytest.mark.parametrize("change", ["training", "pool", "threshold", "model"])
def test_qualification_basis_rejects_changed_workload(change):
    from tests.e10_acceptance_queue_test import cells as acceptance_cells

    basis = acceptance_cells()[0].resolved_config
    queue.validate_qualification_basis(cells(), basis)
    if change == "training":
        basis["data"]["train_candidates"] = "changed"
    elif change == "pool":
        basis["data"]["pool"] = "changed"
    elif change == "threshold":
        basis["matching"]["threshold"] = 0.8
    else:
        basis["model"] = "changed"
    with pytest.raises(ValueError, match="workload"):
        queue.validate_qualification_basis(cells(), basis)
