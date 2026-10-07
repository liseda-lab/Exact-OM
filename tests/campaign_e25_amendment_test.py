"""Approved E25 metadata reaches execution without overriding generated gates."""

from pathlib import Path

import pytest

from exact.core.entities.configs.yaml_io import dump_yaml_document, load_yaml_mapping
from exact.experiments.campaign import CampaignLock, CampaignStep, materialize_campaign
from exact.experiments.preparation import prepare_campaign
from tests.campaign_preparation_test import _cases


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
