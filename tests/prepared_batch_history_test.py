"""Preparing successors preserves historical cases and skips unfinished placeholders."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from exact.experiments.campaign import CampaignLock
from tests.campaign_v2_test import _binding, _external_selection_fixture
from tools import prepared_batch


def _fixture(tmp_path):
    campaign, retained, _ = _external_selection_fixture(tmp_path)
    retained["generate_rationales"] = False
    case = copy.deepcopy(retained["cases"]["D0"])
    case["task"] = "second-development-pair"
    case["transformation"] = (
        "Preservation-only normalization; original receipt and native proof retained"
    )
    case["references"]["internal_check"] = _binding(tmp_path / "internal-check.tsv")
    retained["cases"]["D1"] = case
    target = copy.deepcopy(retained["steps"][0])
    target.update(
        id="E18",
        case="D1",
        phase="expansion",
        external_selection=None,
        requires=["E05"],
        inherits=["E05"],
    )
    unfinished = copy.deepcopy(target)
    unfinished.update(id="E20", requires=[], inherits=[])
    retained["steps"].extend([target, unfinished])
    campaign.write_text(yaml.safe_dump(retained))
    prepared_case = copy.deepcopy(case)
    prepared_case["transformation"] = "Preservation-only normalization"
    del prepared_case["references"]["internal_check"]
    group = {
        "cases": {"D1": prepared_case},
        "rows": [{"step": "E18", "declaration": copy.deepcopy(target)}],
    }
    group_path = tmp_path / "group.json"
    group_path.write_text(json.dumps(group))
    recipe = {
        "base_campaign": prepared_batch.binding(campaign),
        "group": prepared_batch.binding(group_path),
        "blueprint": retained["blueprint"],
        "scientific_step": "E18",
        "group_step": "E18",
        "depends_on": [],
        "commit": "1" * 40,
        "checks": ["fixture historical integrity"],
    }
    return recipe, retained, group


@pytest.mark.parametrize(
    "unfinished_status",
    ["blocked", "blocked_input_resolution", "deferred_budget", "selected", "screened_out"],
)
def test_prepared_history_skips_known_selection_and_unfinished_placeholders(
    tmp_path, monkeypatch, unfinished_status
):
    recipe, retained, _ = _fixture(tmp_path)
    historical_selection = tmp_path / "old-runtime/screen/selection.json"
    historical_selection.parent.mkdir(parents=True)
    historical_selection.write_text(
        json.dumps(
            {
                "experiments": {
                    "E05": {"status": "screened_out"},
                    "E18": {"status": "blocked"},
                    "E20": {"status": unfinished_status},
                },
            }
        )
    )
    completion = tmp_path / "old-completion.json"
    completion.write_text(
        json.dumps(
            {
                "status": "complete",
                "selection": prepared_batch.binding(historical_selection),
            }
        )
    )
    registry = {"runs": [{"id": "historical-screen", "completion_path": str(completion)}]}
    recipe["depends_on"] = ["historical-screen"]

    def unexpected_source(_):
        pytest.fail(
            "An already imported comparison and unfinished placeholders need no source campaign"
        )

    monkeypatch.setattr(prepared_batch, "source_campaign", unexpected_source)
    lock = prepared_batch.prepare_lock(recipe, tmp_path / "new", registry)
    CampaignLock.model_validate(lock)
    assert lock["steps"][0]["external_selection"] == retained["steps"][0]["external_selection"]
    assert lock["steps"][1]["readiness"]["baseline"]["screen"]["status"] == "screen_ready"
    assert lock["steps"][2]["external_selection"] is None
    assert (
        lock["steps"][2]["readiness"]["baseline"]["screen"]["status"] == "blocked_input_resolution"
    )
    assert not (tmp_path / "new/history").exists()


def test_prepared_case_retains_full_historical_provenance_and_unused_reference_roles(tmp_path):
    recipe, retained, _ = _fixture(tmp_path)
    lock = prepared_batch.prepare_lock(recipe, tmp_path / "new", {"runs": []})
    CampaignLock.model_validate(lock)
    assert lock["cases"]["D1"] == retained["cases"]["D1"]
    assert lock["cases"]["D1"]["transformation"].endswith("native proof retained")
    assert "internal_check" in lock["cases"]["D1"]["references"]


@pytest.mark.parametrize("change", ["valid", "source", "negative_policy", "overlay"])
def test_prepared_case_rejects_changes_to_scientific_bindings(tmp_path, change):
    recipe, _, group = _fixture(tmp_path)
    case = group["cases"]["D1"]
    if change == "valid":
        case["references"]["valid"] = _binding(tmp_path / "different-valid.tsv", "new labels")
    elif change == "source":
        case["source"] = _binding(tmp_path / "different.owl", "new ontology")
    elif change == "negative_policy":
        case["negative_policy"] = "confirmed_only"
    else:
        case["overlay"] = {"io": {"source_options": {"different_native_semantics": True}}}
    group_path = Path(recipe["group"]["path"])
    group_path.write_text(json.dumps(group))
    recipe["group"] = prepared_batch.binding(group_path)
    with pytest.raises(ValueError, match="Prepared (case|development binding) differs"):
        prepared_batch.prepare_lock(recipe, tmp_path / "new", {"runs": []})


@pytest.mark.parametrize("required", [False, True])
def test_failed_independent_history_is_not_a_dependency(tmp_path, required):
    recipe, _, _ = _fixture(tmp_path)
    completion = tmp_path / "failed-completion.json"
    completion.write_text(json.dumps({"status": "failed", "exit_code": 1}))
    registry = {"runs": [{"id": "failed", "completion_path": str(completion)}]}
    recipe["depends_on"] = ["failed"] if required else []
    if required:
        with pytest.raises(ValueError, match="Required comparison is incomplete"):
            prepared_batch.prepare_lock(recipe, tmp_path / "new", registry)
    else:
        CampaignLock.model_validate(prepared_batch.prepare_lock(recipe, tmp_path / "new", registry))


@pytest.mark.parametrize("transitive", [False, True])
def test_required_history_is_checked_through_named_output_ports(tmp_path, monkeypatch, transitive):
    recipe, retained, _ = _fixture(tmp_path)
    retained["steps"][0]["external_selection"] = None
    retained["steps"][0]["produces"] = ["required_policy"]
    target = retained["steps"][1]
    target.update(requires=["required_policy"], inherits=[])
    if transitive:
        retained["steps"][2].update(requires=["required_policy"], produces=["intermediate_policy"])
        target["requires"] = ["intermediate_policy"]
    campaign = Path(recipe["base_campaign"]["path"])
    campaign.write_text(yaml.safe_dump(retained))
    recipe["base_campaign"] = prepared_batch.binding(campaign)
    group_path = Path(recipe["group"]["path"])
    group = json.loads(group_path.read_text())
    group["rows"][0]["declaration"] = target
    group_path.write_text(json.dumps(group))
    recipe["group"] = prepared_batch.binding(group_path)
    selection = tmp_path / "required-selection.json"
    selection.write_text(json.dumps({"experiments": {"E05": {"status": "selected"}}}))
    completion = tmp_path / "required-completion.json"
    completion.write_text(
        json.dumps({"status": "complete", "selection": prepared_batch.binding(selection)})
    )
    registry = {"runs": [{"id": "required", "completion_path": str(completion)}]}

    def required_source(_):
        raise ValueError("Required source must still be verified")

    monkeypatch.setattr(prepared_batch, "source_campaign", required_source)
    with pytest.raises(ValueError, match="Required source must still be verified"):
        prepared_batch.prepare_lock(recipe, tmp_path / "new", registry)


def test_approved_optional_disposition_survives_preparation_and_rejects_drift(tmp_path):
    recipe, retained, _ = _fixture(tmp_path)
    disposition = {"status": "deferred_approved", "reason": "Approved unavailable input"}
    path = tmp_path / "dispositions.json"
    path.write_text(json.dumps({"E20": disposition}))
    recipe["approved_dispositions"] = prepared_batch.binding(path)
    registry = {"runs": [], "approved_dispositions": {"E20": disposition}}
    lock = prepared_batch.prepare_lock(recipe, tmp_path / "new", registry)
    step = next(s for s in lock["steps"] if s["id"] == "E20")
    assert all(s["screen"]["status"] == "inapplicable" for s in step["readiness"].values())
    assert step["requires"] == retained["steps"][2]["requires"]
    registry["approved_dispositions"]["E20"] = {**disposition, "status": "pending"}
    with pytest.raises(ValueError, match="disposition changed"):
        prepared_batch.prepare_lock(recipe, tmp_path / "changed", registry)
