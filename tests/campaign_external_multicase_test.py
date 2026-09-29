"""Historical decisions require every case, mode, seed and arm exactly once."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from exact.experiments import harness
from exact.experiments.campaign import (
    CampaignStep,
    campaign_identity,
    digest,
    external_selection_result,
    load_campaign,
    materialize_campaign,
)
from tests.campaign_v2_test import _binding, _external_selection_fixture


def _multicase_fixture(tmp_path, phase="expansion"):
    path, raw, record = _external_selection_fixture(tmp_path)
    historical_path = Path(record["campaign"]["path"])
    old_raw = yaml.safe_load(historical_path.read_text())
    old_raw["cases"]["D1"] = copy.deepcopy(old_raw["cases"]["D0"])
    old_raw["cases"]["D1"]["task"] = "second-development-pair"
    step = old_raw["steps"][0]
    step.update(
        phase=phase,
        additional_cases=["D1"],
        execution_modes=["global_alignment", "local_ranking"],
        seeds=[17] if phase == "initial" else [17, 29],
    )
    record["campaign"] = _binding(historical_path, yaml.safe_dump(old_raw))
    old, _ = load_campaign(historical_path)
    suite = materialize_campaign(historical_path, tmp_path / "multi-declarations", stage="screen")
    source = suite.sources[0]
    result = harness._runtime_deferred_selection(
        source, suite, reason_code="fixture", reason="Completed historical controls"
    )
    result.update(
        status="screened_out",
        selected_overlay={},
        resolved_arms_hash=harness.hash_payload(
            {arm.id: arm.overlay for arm in harness._component_arms(source.config)}
        ),
    )
    selected = {
        "stage": "screen",
        "suite_hash": campaign_identity(old, tmp_path),
        "experiments": {"E05": result},
    }
    selected["selection_hash"] = digest(selected)
    record["selection"] = _binding(Path(record["selection"]["path"]), json.dumps(selected))
    cells, rows = [], []
    for arm in step["arms"]:
        for case in [step["case"], *step["additional_cases"]]:
            for mode in step["execution_modes"]:
                for seed in step["seeds"]:
                    task_id = f"{case}-{mode}"
                    cell_id = f"E05/screen/{arm['id']}/{task_id}/seed-{seed}"
                    artifacts = {"extraction": digest(cell_id)}
                    cells.append(
                        _binding(
                            tmp_path / f"{arm['id']}-{task_id}-{seed}.json",
                            json.dumps(
                                {
                                    "arm_id": arm["id"],
                                    "experiment_id": "E05",
                                    "stage": "screen",
                                    "status": "complete",
                                    "return_code": 0,
                                    "extraction_complete": True,
                                    "task_id": task_id,
                                    "execution_mode": mode,
                                    "seed": seed,
                                    "source_cap": 300,
                                    "split_role": "development",
                                    "generate_rationales": False,
                                    "recovery": {"artifacts": artifacts},
                                }
                            ),
                        )
                    )
                    rows.append({"cell_id": cell_id, "artifacts": artifacts})
    record["cells"] = cells
    record["result_set"] = _binding(
        Path(record["result_set"]["path"]), json.dumps({"cells": rows})
    )
    raw["cases"] = old_raw["cases"]
    raw["steps"][0] = {**step, "estimate": None}
    return path, raw, record


def _save(path, raw, record):
    raw["steps"][0]["external_selection"] = _binding(
        path.parent / "prior-selection.json", json.dumps(record)
    )
    path.write_text(yaml.safe_dump(raw))
    return load_campaign(path)[0]


@pytest.mark.parametrize("phase", ["initial", "expansion", "sentinel", "late"])
def test_external_selection_accepts_complete_multicase_comparison(tmp_path, phase):
    path, raw, record = _multicase_fixture(tmp_path, phase)
    lock = _save(path, raw, record)
    producers = []
    result = external_selection_result(lock, lock.steps[0], tmp_path, producer_manifests=producers)
    assert result["status"] == "screened_out"
    assert result["new_cells"] == 0
    assert result["current_code_prediction_compatibility"] is False
    assert len(producers) == (8 if phase == "initial" else 16)
    assert {m["task_id"] for m in producers} == {
        f"{case}-{mode}"
        for case in ("D0", "D1")
        for mode in ("global_alignment", "local_ranking")
    }


@pytest.mark.parametrize(
    "change",
    ["missing", "duplicate", "missing_task", "foreign_case", "mode", "seed", "rows", "inputs"],
)
def test_external_selection_rejects_incomplete_or_aliased_multicase_receipts(tmp_path, change):
    path, raw, record = _multicase_fixture(tmp_path)
    if change == "missing":
        record["cells"].pop()
    elif change == "duplicate":
        record["cells"][-1] = record["cells"][0]
    elif change == "rows":
        target = record["result_set"]
        value = json.loads(Path(target["path"]).read_text())
        value["cells"].append(value["cells"][0])
        record["result_set"] = _binding(Path(target["path"]), json.dumps(value))
    elif change == "inputs":
        raw["cases"]["D1"]["source"] = _binding(tmp_path / "changed.owl", "changed ontology")
    else:
        target = record["cells"][0]
        manifest = json.loads(Path(target["path"]).read_text())
        if change == "missing_task":
            manifest.pop("task_id")
        elif change == "foreign_case":
            manifest["task_id"] = "reporting-global_alignment"
        elif change == "mode":
            manifest["execution_mode"] = "local_ranking"
        else:
            manifest["seed"] = 43
        record["cells"][0] = _binding(Path(target["path"]), json.dumps(manifest))
    lock = _save(path, raw, record)
    producers = []
    with pytest.raises(ValueError, match="external selection"):
        external_selection_result(lock, lock.steps[0], tmp_path, producer_manifests=producers)
    assert producers == []


def test_external_single_case_explicit_task_cannot_impersonate_another_case(tmp_path):
    path, raw, record = _external_selection_fixture(tmp_path)
    target = record["cells"][0]
    manifest = json.loads(Path(target["path"]).read_text())
    manifest["task_id"] = "D1-global_alignment"
    record["cells"][0] = _binding(Path(target["path"]), json.dumps(manifest))
    lock = _save(path, raw, record)
    with pytest.raises(ValueError, match="cell is incomplete or mismatched"):
        external_selection_result(lock, lock.steps[0], tmp_path)


@pytest.mark.parametrize("phase", ["freeze", "final"])
def test_external_selection_cannot_import_freeze_or_final_as_development(tmp_path, phase):
    _, raw, _ = _external_selection_fixture(tmp_path)
    step = {**raw["steps"][0], "phase": phase}
    if phase == "final":
        step["source_cap"] = None
    with pytest.raises(ValueError, match="only a completed development screen"):
        CampaignStep.model_validate(step)
