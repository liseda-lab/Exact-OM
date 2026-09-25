"""Regression checks for cumulative accounting in the cached batch continuation."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from exact.experiments.budget import BudgetLedger
from tools.resume_cached_batch import reconcile_budget


def state():
    return {
        "schema_version": 2,
        "limits": {
            "envelopes_hours": {"reserve": 60, "foundation": 12},
            "node_hours_cap": 336,
            "requests_cap": 50000,
            "tokens_cap": 32000000,
            "final_requests_reserved": 5000,
            "final_tokens_reserved": 8000000,
        },
        "work": {
            "historical/G0": {
                "status": "reserved",
                "group": "foundation",
                "seconds": 43200,
                "requests": 4000,
                "tokens": 2000000,
                "actual_usd": None,
                "projected_usd": 1,
            },
            "failed-unknown-request": {
                "status": "failed",
                "group": "reserve",
                "start": 1,
                "end": 11,
                "seconds": 10,
                "requests": 1,
                "tokens": 700,
                "actual_usd": None,
                "projected_usd": 0.1,
            },
        },
        "intervals": [[1, 11]],
        "node_seconds": 10,
    }


def test_missing_setup_is_added_without_changing_history_or_caps():
    original = state()
    saved = copy.deepcopy(original)
    result = reconcile_budget(original, 11, 21)
    assert original == saved
    assert result["limits"] == saved["limits"]
    assert all(result["work"][key] == value for key, value in saved["work"].items())
    added = [value for key, value in result["work"].items() if key not in saved["work"]]
    assert len(added) == 1
    assert added[0]["status"] == "failed"
    assert added[0]["seconds"] == 10
    assert added[0]["requests"] == added[0]["tokens"] == 0
    assert result["node_seconds"] == 20
    assert sum(row["requests"] for row in result["work"].values()) == 4001
    assert sum(row["tokens"] for row in result["work"].values()) == 2000700


def test_setup_reconciliation_refuses_duplicate_charge():
    reconciled = reconcile_budget(state(), 11, 21)
    with pytest.raises(ValueError):
        reconcile_budget(reconciled, 11, 21)


@pytest.mark.parametrize("start,end", [(21, 11), (float("nan"), 21), (11, float("inf"))])
def test_setup_reconciliation_rejects_invalid_intervals(start, end):
    with pytest.raises(ValueError):
        reconcile_budget(state(), start, end)


def test_budget_limit_mismatch_still_fails_closed(tmp_path):
    original = state()
    path = tmp_path / "budget.json"
    path.write_text(json.dumps(original))
    stale = {**original["limits"], "requests_cap": 20000}
    with pytest.raises(ValueError, match="limits changed"):
        BudgetLedger(path, stale).admit("new", group="reserve", seconds=1)
    assert json.loads(path.read_text()) == original


def test_reconciled_budget_retains_protected_final_allowance(tmp_path):
    original = reconcile_budget(state(), 11, 21)
    path = tmp_path / "budget.json"
    path.write_text(json.dumps(original))
    with pytest.raises(ValueError, match="protected final"):
        BudgetLedger(path, original["limits"]).admit(
            "overspend", group="reserve", seconds=1, requests=41000
        )
    assert json.loads(path.read_text()) == original


def test_failed_cache_copy_retains_known_previous_setup(tmp_path):
    from types import SimpleNamespace

    from tools.resume_cached_batch import copy_reconciled_state

    source, destination = tmp_path / "source", tmp_path / "destination"
    source.mkdir()
    helpers = SimpleNamespace()

    def freeze(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def failed_copy(src, dst):
        assert src == source
        helpers.freeze(dst / "budget.json", state())
        raise OSError("cache copy failed")

    helpers.freeze, helpers.copy_state = freeze, failed_copy
    with pytest.raises(OSError, match="cache copy failed"):
        copy_reconciled_state(helpers, source, destination, (11, 21))
    assert helpers.freeze is freeze
    saved = json.loads((destination / "budget.json").read_text())
    assert saved["node_seconds"] == 20
    assert len(saved["work"]) == 3
    assert saved["limits"] == state()["limits"]


@pytest.mark.parametrize("fail", [False, True])
def test_inventory_accounting_closes_on_success_and_failure(tmp_path, monkeypatch, fail):
    import sys
    from types import SimpleNamespace

    from exact.experiments import harness
    from tools.resume_cached_batch import account_inventory

    original = state()
    (tmp_path / "budget.json").write_text(json.dumps(original))
    usage = {"attempts": 1, "billable_tokens": 700, "unpriced_attempts": 1, "reported_cost_usd": 0}
    monkeypatch.setitem(sys.modules, "qualification", SimpleNamespace(usage=lambda _: dict(usage)))

    def inventory(*args, **kwargs):
        if fail:
            raise ValueError("native inventory failed")
        return "native inventory complete"

    monkeypatch.setattr(harness, "build_dataset_inventory", inventory)
    helpers = SimpleNamespace(HERE=tmp_path, check_pause=lambda: None)
    with account_inventory(helpers, tmp_path, "E01", "reserve", 100):
        if fail:
            with pytest.raises(ValueError, match="native inventory failed"):
                harness.build_dataset_inventory()
        else:
            assert harness.build_dataset_inventory() == "native inventory complete"
        with pytest.raises(ValueError, match="once per family"):
            harness.build_dataset_inventory()
    assert harness.build_dataset_inventory is inventory
    saved = json.loads((tmp_path / "budget.json").read_text())
    row = next(v for k, v in saved["work"].items() if k not in original["work"])
    assert row["status"] == ("failed" if fail else "complete")
    assert row["requests"] == row["tokens"] == 0
    assert saved["limits"] == original["limits"]


@pytest.mark.parametrize("change_science", [False, True])
def test_blueprint_repair_preserves_science_and_only_inherits_recorded_cap(
    tmp_path, change_science
):
    import hashlib
    from types import SimpleNamespace

    import yaml

    from tools.resume_cached_batch import corrected_lock

    old = tmp_path / "old"
    (old / "E01").mkdir(parents=True)
    stale = {
        "profiles": {"core_14d": {"llm_request_planning_cap": 20000}},
        "selection": {"threshold": 0.95},
    }
    amended = copy.deepcopy(stale)
    amended["profiles"]["core_14d"]["llm_request_planning_cap"] = 50000
    if change_science:
        amended["selection"]["threshold"] = 0.90

    def binding(name, data):
        path = tmp_path / name
        path.write_text(yaml.safe_dump(data))
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    lock = {
        "blueprint": binding("stale.yaml", stale),
        "steps": [{"id": "E01", "arms": [1, 2, 3, 4, 5]}],
    }
    (old / "E01/campaign.lock.yaml").write_text(yaml.safe_dump(lock))
    prior = {"blueprint": binding("amended.yaml", amended)}
    parent = tmp_path / "parent.yaml"
    parent.write_text(yaml.safe_dump(prior))

    def verify(item):
        path = Path(item["path"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
        return path

    helpers = SimpleNamespace(PARENT=old, verify=verify)
    if change_science:
        with pytest.raises(ValueError, match="beyond the recorded"):
            corrected_lock(parent, helpers)
    else:
        result = corrected_lock(parent, helpers)
        assert result == {**lock, "blueprint": prior["blueprint"]}
    assert yaml.safe_load((old / "E01/campaign.lock.yaml").read_text()) == lock
