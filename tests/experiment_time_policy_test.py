"""Time estimates never stop real work; API spending and accounting remain guarded."""

import json

import pytest
import yaml

from exact.experiments.budget import BudgetLedger
from exact.experiments.campaign import campaign_plan
from tests.campaign_v2_test import _binding, _lock


def ledger(tmp_path):
    return BudgetLedger(
        tmp_path / "budget.json",
        {
            "envelopes_hours": {"channels": 0.01, "final": 1},
            "node_hours_cap": 0.01,
            "requests_cap": 10,
            "tokens_cap": 100,
            "final_requests_reserved": 2,
            "final_tokens_reserved": 20,
        },
    )


def test_overrun_finishes_and_does_not_block_next_attempt(tmp_path):
    account = ledger(tmp_path)
    account.admit("first", group="channels", seconds=100, requests=1, tokens=10)
    account.finish(
        "first", start=0, end=200, status="complete", requests=1, tokens=10, actual_usd=0.01
    )
    account.admit("next", group="channels", seconds=100)
    state = account.snapshot()
    assert state["work"]["first"]["seconds"] == 200
    assert state["work"]["first"]["status"] == "complete"
    assert len(state["work"]["first"]["time_warnings"]) == 3
    assert state["work"]["next"]["status"] == "reserved"
    assert state["limits"]["node_hours_cap"] == 0.01
    assert state["time_policy"] == "advisory_2026-09-29"
    with pytest.raises(ValueError, match="protected final"):
        account.admit("paid", group="channels", seconds=0, requests=8)
    assert "paid" not in account.snapshot()["work"]


def test_allocation_time_counts_idle_but_not_twice(tmp_path):
    account = ledger(tmp_path)
    account.record_allocation("job", start=10, end=100)
    account.record_allocation("job", start=10, end=120)
    account.record_allocation("overlap", start=20, end=80)
    assert account.snapshot()["allocation_seconds"] == 110
    assert account.snapshot()["intervals"] == []
    with pytest.raises(ValueError, match="rewrite"):
        account.record_allocation("job", start=10, end=90)


def test_campaign_forecast_warns_without_time_admission_block(tmp_path):
    path = _lock(tmp_path)
    data = yaml.safe_load(path.read_text())
    data["steps"][0]["estimate"] = {
        "cold_seconds": 500000,
        "units": 1,
        "seconds_per_unit": 1,
        "peak_ram_gb": 1,
        "measurement_artifact": _binding(tmp_path / "measurement.json", "{}"),
    }
    path.write_text(yaml.safe_dump(data))
    plan = campaign_plan(path, stage="screen")
    assert not plan["budget_errors"]
    assert any("foundation forecast exceeds" in warning for warning in plan["time_warnings"])
    data["steps"][0]["estimate"]["requests"] = 999999
    path.write_text(yaml.safe_dump(data))
    assert campaign_plan(path, stage="screen")["budget_errors"]


def test_missing_forecast_is_recorded_as_unknown(tmp_path):
    account = ledger(tmp_path)
    account.admit("new", group="channels", seconds=0, forecast_known=False)
    account.finish("new", start=0, end=60, status="complete", requests=0, tokens=0, actual_usd=0)
    state = json.loads(account.path.read_text())
    assert not state["work"]["new"]["forecast_known"]
    assert not state["work"]["new"]["time_warnings"]
