"""Approved E25 metadata reaches execution without overriding generated gates."""

import json
from pathlib import Path

import pytest

from exact.core.entities.configs.yaml_io import dump_yaml_document, load_yaml_mapping
from exact.experiments.campaign import (
    CampaignLock,
    CampaignStep,
    campaign_identity,
    digest,
    external_selection_result,
    load_campaign,
    materialize_campaign,
)
from exact.experiments.preparation import prepare_campaign
from exact.utils.provenance import sha256_file
from tests.campaign_preparation_test import _cases
from tests.campaign_v2_test import _binding, _external_selection_fixture


@pytest.fixture
def campaign(tmp_path):
    root = Path(__file__).resolve().parents[1]
    path = prepare_campaign(
        root / "specs/experiments/campaign-v2.yaml",
        root / "exact/default_config.yaml",
        {"cases": _cases()},
        tmp_path / "prepared",
    )
    return path, load_yaml_mapping(path)


def test_e25_amendment_preserves_generated_scope_and_exposes_explicit_disposition(
    campaign, tmp_path
):
    path, payload = campaign
    steps = {step["id"]: step for step in payload["steps"]}
    steps["E25-oracles"]["frozen_constants"] = {"outcome_semantics": "benchmark_reference"}
    trust = steps["E25-trust"]
    trust["arms"] = [arm for arm in trust["arms"] if arm["id"] != "trust_source"]
    trust["readiness"].pop("trust_source")
    trust["frozen_constants"] = {
        "inapplicable_arms": {"trust_source": "Binary responses contain no comparative choice"}
    }
    CampaignLock.model_validate(payload)
    path.write_text(dump_yaml_document(payload))
    suite = materialize_campaign(path, tmp_path / "materialized", stage="screen")
    sources = {source.config.experiment_id: source.config for source in suite.sources}
    oracle = sources["E25-oracles"]
    assert oracle.frozen_constants["outcome_semantics"] == "benchmark_reference"
    assert oracle.frozen_constants["campaign_v2"]["family"] == "E25"
    assert oracle.frozen_constants["campaign_v2"]["requires"] == ["E25-forced"]
    assert oracle.screen.source_cap == 300 and oracle.screen.seeds == [17]
    assert len(oracle.arms) == 3
    assert {arm.id for arm in sources["E25-trust"].arms} == {"trust_shipped", "trust_constant"}
    assert (
        sources["E25-trust"].frozen_constants["inapplicable_arms"]
        == trust["frozen_constants"]["inapplicable_arms"]
    )
    assert "outcome_semantics" not in sources["E25-trust"].frozen_constants


@pytest.mark.parametrize(
    "step,constants",
    [
        ("E25-oracles", {"campaign_v2": {"family": "E99"}}),
        ("E25-oracles", {"selected_judge": {"producer": "E01"}}),
        ("E25-oracles", {"outcome_semantics": "pretend_complete_reference"}),
        ("E25-trust", {"outcome_semantics": "benchmark_reference"}),
        ("E25-trust", {"inapplicable_arms": {"trust_source": "Still present"}}),
        ("E25-trust", {"inapplicable_arms": {"trust_constant": "Drop arbitrary arm"}}),
    ],
)
def test_campaign_rejects_unsupported_e25_constants(campaign, step, constants):
    _, payload = campaign
    declaration = next(item for item in payload["steps"] if item["id"] == step)
    declaration["frozen_constants"] = constants
    with pytest.raises(ValueError):
        CampaignStep.model_validate(declaration)


def test_imports_legacy_selection_signed_without_optional_constants(tmp_path):
    path, raw, record = _external_selection_fixture(tmp_path)
    historical_path = Path(record["campaign"]["path"])
    historical_raw = load_yaml_mapping(historical_path)
    assert all("frozen_constants" not in step for step in historical_raw["steps"])
    old, _ = load_campaign(historical_path)
    # Reconstruct the pre-amendment schema's canonical payload independently of
    # campaign_identity; signing both sides with today's function hides regressions.
    legacy = old.model_dump(mode="json", exclude={"final_selection"})
    for step in legacy["steps"]:
        step.pop("readiness")
        step.pop("frozen_constants")
        if step.get("external_selection") is None:
            step.pop("external_selection", None)
    base = old.base_config
    legacy["base_config"] = {
        "sha256": sha256_file(base if base.is_absolute() else historical_path.parent / base)
    }

    def content(value):
        if isinstance(value, dict):
            if set(value) == {"path", "sha256"}:
                return {"sha256": value["sha256"]}
            return {key: content(item) for key, item in value.items()}
        if isinstance(value, list):
            return [content(item) for item in value]
        return value

    selected_path = Path(record["selection"]["path"])
    selected = json.loads(selected_path.read_text())
    selected.pop("selection_hash")
    selected["suite_hash"] = digest(content(legacy))
    selected["selection_hash"] = digest(selected)
    record["selection"] = _binding(selected_path, json.dumps(selected))
    raw["steps"][0]["external_selection"] = _binding(
        tmp_path / "legacy-selection.json", json.dumps(record)
    )
    path.write_text(dump_yaml_document(raw))
    lock, _ = load_campaign(path)
    result = external_selection_result(lock, lock.steps[0], tmp_path)
    assert result["status"] == "screened_out"
    assert result["new_cells"] == 0


def test_nonempty_amendment_remains_part_of_campaign_identity(campaign):
    path, raw = campaign
    before = campaign_identity(CampaignLock.model_validate(raw), path.parent)
    step = next(step for step in raw["steps"] if step["id"] == "E25-oracles")
    step["frozen_constants"] = {"outcome_semantics": "benchmark_reference"}
    after = campaign_identity(CampaignLock.model_validate(raw), path.parent)
    assert after != before
