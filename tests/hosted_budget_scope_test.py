"""Preparations and variant/recovery cells share their frozen scientific family."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import yaml

from exact.experiments import harness, runtime
from exact.experiments.campaign import materialize_campaign
from tests.campaign_v2_test import _lock


def test_frozen_family_overrides_variant_and_parent_scope_for_each_cell(tmp_path, monkeypatch):
    path = _lock(tmp_path)
    value = yaml.safe_load(path.read_text())
    value["steps"][0]["id"] = "E00-pilot"
    path.write_text(yaml.safe_dump(value))
    suite = materialize_campaign(path, tmp_path / "declarations", stage="screen")
    suite.sources[0].config.implementation.status = "ready"
    suite = replace(suite, campaign={"root": str(tmp_path / "runtime")})
    cells = harness.build_cells(
        suite, suite.sources[0], stage="screen", output_root=tmp_path / "output"
    )
    assert cells
    assert all(cell.experiment_id == "E00-pilot" for cell in cells)
    assert all(
        cell.recovery["hosted_scope"]
        == {
            "campaign_id": "fixture-campaign",
            "experiment_id": "E00",
        }
        for cell in cells
    )
    assert "hosted_scope" not in suite.campaign
    selected = {
        "path": "/fixture/policy",
        "sha256": "fixture",
        "policy": {"mode": "hard_pause", "campaign_id": "fixture-campaign"},
    }
    monkeypatch.setattr(runtime, "load_spending_policy", lambda *args: selected)
    monkeypatch.setenv("EXACT_HOSTED_EXPERIMENT_ID", "E21")
    for family in ("E00", "E23"):
        recovery = runtime.CellRecovery.__new__(runtime.CellRecovery)
        recovery.metadata = {
            **cells[0].recovery,
            "hosted_scope": {
                "campaign_id": "fixture-campaign",
                "experiment_id": family,
            },
        }
        recovery.cell = cells[0]
        recovery.store = SimpleNamespace(root=tmp_path / "runtime")
        recovery.dataset_cache_scope = "fixture"
        environment = recovery.environment()
        assert environment["EXACT_HOSTED_CAMPAIGN_ID"] == "fixture-campaign"
        assert environment["EXACT_HOSTED_EXPERIMENT_ID"] == family
    assert runtime.os.environ["EXACT_HOSTED_EXPERIMENT_ID"] == "E21"


@pytest.mark.parametrize(
    "campaign,family", [("other", "E23"), ("campaign", "E23-rich"), ("campaign", None)]
)
def test_scope_rejects_unbound_campaign_or_recovery_name(campaign, family):
    with pytest.raises(ValueError, match="frozen"):
        runtime.hosted_scope_environment(
            campaign,
            family,
            policy={
                "policy": {"mode": "hard_pause", "campaign_id": "campaign"},
            },
        )


def test_legacy_notification_policy_does_not_require_new_scope():
    assert (
        runtime.hosted_scope_environment(
            None,
            None,
            policy={
                "policy": {"mode": "notification_only"},
            },
        )
        == {}
    )


def test_prepared_launch_derives_scope_from_bound_family_before_attaching_guards(
    tmp_path, monkeypatch
):
    from exact.utils import hosted_spending
    from tools import hosted_prompt_guard, prepared_batch, storage_guard

    root, code, supervisor = (tmp_path / name for name in ("prepared", "code", "supervisor"))
    for directory in (root, code / "tools", supervisor):
        directory.mkdir(parents=True)
    (code / "tools/prepared_batch.py").write_text("# fixture")
    base = tmp_path / "campaign.json"
    prepared_batch.write(
        base, {"campaign_id": "campaign", "steps": [{"id": "E23-rich", "family": "E23"}]}
    )
    environment = tmp_path / "environment.json"
    prepared_batch.write(environment, {"EXACT_HOSTED_CAMPAIGN_ID": "campaign"})
    prepared_batch.write(
        supervisor / "policy.json", {"hosted_spending_policy": {"path": "fixture"}}
    )
    monkeypatch.setattr(
        hosted_spending,
        "load_spending_policy",
        lambda *args: {
            "policy": {"mode": "hard_pause", "campaign_id": "campaign"},
        },
    )

    def guard(launch, *args, **kwargs):
        assert launch["run"]["hosted_scope"] == {"campaign_id": "campaign", "experiment_id": "E23"}
        assert prepared_batch.binding(base) in launch["bindings"]
        return launch

    monkeypatch.setattr(storage_guard, "guard_launch", guard)
    monkeypatch.setattr(hosted_prompt_guard, "guard_launch", guard)
    recipe = {
        "repository": str(tmp_path),
        "allocation": "14372",
        "dispatch_nonce": "fixture",
        "campaign_id": "campaign",
        "scientific_step": "E23-rich",
        "base_campaign": prepared_batch.binding(base),
        "group": prepared_batch.binding(base),
        "environment": prepared_batch.binding(environment),
    }
    launch = prepared_batch.prepare_launch(
        recipe, root, code, supervisor, {"id": "recovery-unrelated-name", "depends_on": []}
    )
    assert launch["run"]["hosted_scope"]["experiment_id"] == "E23"
