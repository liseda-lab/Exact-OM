"""Explicit amendments lift aggregate limits while retaining paid-request evidence."""

import copy
import hashlib
import json
import os
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from exact.experiments.budget import BudgetLedger
from exact.experiments.campaign import campaign_plan
from exact.experiments.runtime import CellRecovery
from exact.llm.ledger import RequestLedger
from exact.utils.hosted_spending import (
    POLICY_PATH_ENV,
    POLICY_SHA256_ENV,
    load_spending_policy,
    spending_policy_environment,
)
from tests.campaign_v2_test import _lock
from tests.prepared_batch_worker_test import _outputs, _worker
from tools import experiment_resources, prepared_batch


@pytest.fixture
def policy(tmp_path, monkeypatch):
    path = tmp_path / "spending-policy.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "exact_om_hosted_spending_policy",
                "mode": "notification_only",
                "notification_tokens": 100000000,
                "authorization": "User approved continued work and a notification after 100M tokens",
            }
        )
    )
    monkeypatch.setenv(POLICY_PATH_ENV, str(path))
    monkeypatch.setenv(POLICY_SHA256_ENV, hashlib.sha256(path.read_bytes()).hexdigest())
    return load_spending_policy()


def limits():
    return {
        "envelopes_hours": {"llm": 0},
        "node_hours_cap": 0,
        "requests_cap": 1,
        "tokens_cap": 10,
        "final_requests_reserved": 1,
        "final_tokens_reserved": 10,
    }


def request(ledger, name, *, max_tokens=8):
    return ledger.plan({"role": "decision", "payload": {"max_tokens": max_tokens, "text": name}})


@pytest.mark.parametrize("missing", [POLICY_PATH_ENV, POLICY_SHA256_ENV])
def test_partial_policy_binding_cannot_disable_limits(policy, monkeypatch, missing):
    monkeypatch.delenv(missing)
    with pytest.raises(ValueError, match="path and SHA256"):
        load_spending_policy()


def test_policy_bytes_are_verified_before_new_paid_attempt(policy, tmp_path):
    ledger = RequestLedger(tmp_path / "wire")
    key = request(ledger, "unpaid")
    Path(policy["path"]).write_text("{}")
    with pytest.raises(ValueError, match="SHA256"):
        ledger.sent(key)
    with ledger._transaction() as db:
        assert db.execute("SELECT count(*) FROM attempts").fetchone()[0] == 0


def test_admission_retains_historical_limits_charges_and_records_amendment(
    policy, tmp_path, monkeypatch
):
    historical = {
        "group": "llm",
        "status": "complete",
        "start": 1,
        "end": 2,
        "requests": 5,
        "tokens": 200000000,
        "actual_usd": 4,
    }
    path = tmp_path / "budget.json"
    state = {"limits": limits(), "work": {"historical": historical}, "intervals": [[1, 2]]}
    path.write_text(json.dumps(state))
    budget = BudgetLedger(path, limits())
    budget.admit(
        "continue",
        group="llm",
        seconds=10,
        requests=100000,
        tokens=300000000,
        hosted_usage_baseline={"attempts": 5, "billable_tokens": 200000000},
    )
    saved = budget.snapshot()
    assert saved["limits"] == state["limits"]
    assert saved["work"]["historical"] == historical
    assert saved["spending_policies"][policy["sha256"]] == policy
    assert saved["work"]["continue"]["spending_policy_sha256"] == policy["sha256"]
    assert saved["work"]["continue"]["hosted_usage_baseline"]["billable_tokens"] == 200000000
    assert saved["work"]["continue"]["time_warnings"]
    with pytest.raises(ValueError, match="limits changed"):
        BudgetLedger(path, {**limits(), "tokens_cap": 999}).snapshot()
    monkeypatch.delenv(POLICY_PATH_ENV)
    monkeypatch.delenv(POLICY_SHA256_ENV)
    with pytest.raises(ValueError, match="deferred_budget"):
        BudgetLedger(path, limits()).admit("unamended", group="llm", seconds=0)


def test_unlimited_aggregate_still_reserves_and_guards_unknown_attempts(
    policy, tmp_path, monkeypatch
):
    for key, value in spending_policy_environment(policy).items():
        monkeypatch.setenv(key, value)
    ledger = RequestLedger(tmp_path / "wire")
    first = request(ledger, "unknown")
    ledger.unknown(first, ledger.sent(first), "transport timeout")
    with pytest.raises(RuntimeError, match="explicit retry authorization"):
        ledger.sent(first)
    second = request(ledger, "new independent request", max_tokens=100000001)
    ledger.sent(second)
    with ledger._transaction() as db:
        rows = db.execute("SELECT spending_policy_sha256 FROM attempts").fetchall()
        assert [row[0] for row in rows] == [policy["sha256"], policy["sha256"]]
        assert db.execute("SELECT count(*) FROM reservations").fetchone()[0] == 2
        assert db.execute("SELECT sum(tokens) FROM reservations").fetchone()[0] > 100000000
        assert (
            json.loads(db.execute("SELECT record FROM spending_policies").fetchone()[0]) == policy
        )
    # Existing successor imports use SQLite backup, which must retain the full
    # policy evidence along with historical attempts and their reservations.
    copied = tmp_path / "copied.sqlite3"
    with sqlite3.connect(ledger.path) as source, sqlite3.connect(copied) as target:
        source.backup(target)
        assert (
            json.loads(target.execute("SELECT record FROM spending_policies").fetchone()[0])
            == policy
        )
        assert target.execute("SELECT count(*) FROM attempts").fetchone()[0] == 2


@pytest.mark.parametrize("bound", [None, 0, -1, True, 1.5, "8"])
def test_amended_wire_requests_still_require_positive_integer_output_bound(policy, tmp_path, bound):
    ledger = RequestLedger(tmp_path / "wire")
    with pytest.raises(ValueError, match="finite max_tokens"):
        ledger.sent(request(ledger, "invalid", max_tokens=bound))


def test_explicit_cached_only_process_caps_remain_hard_under_policy(policy, tmp_path, monkeypatch):
    # Finalizers deliberately freeze process caps at cached usage. The amendment
    # only clears campaign-derived aggregate allowances, never these wire guards.
    monkeypatch.setenv("EXACT_OPENROUTER_REQUEST_CAP", "0")
    ledger = RequestLedger(tmp_path / "wire")
    with pytest.raises(ValueError, match="request budget exhausted"):
        ledger.sent(request(ledger, "must remain unpaid"))


def test_policy_plan_lifts_only_hosted_forecast_blockers_without_rewriting_lock(
    policy, tmp_path, monkeypatch
):
    path = _lock(tmp_path)
    lock = yaml.safe_load(path.read_text())
    lock.update(final_requests_reserved=100000, final_tokens_reserved=300000000)
    path.write_text(yaml.safe_dump(lock))
    before = path.read_bytes()
    amended = campaign_plan(path, stage="screen")
    assert amended["spending_policy"] == policy
    assert amended["budget_errors"] == []
    assert any(row["issues"] for row in amended["rows"])
    assert path.read_bytes() == before
    monkeypatch.delenv(POLICY_PATH_ENV)
    monkeypatch.delenv(POLICY_SHA256_ENV)
    assert len(campaign_plan(path, stage="screen")["budget_errors"]) == 2


def test_cell_environment_propagates_policy_and_clears_only_aggregate_caps(policy, tmp_path):
    recovery = CellRecovery.__new__(CellRecovery)
    recovery.metadata = {"budget_limits": limits(), "stage": "screen", "spending_policy": policy}
    recovery.store = SimpleNamespace(root=tmp_path)
    recovery.cell = SimpleNamespace(output_dir=tmp_path, split_role="development")
    recovery.dataset_cache_scope = "fixture"
    environment = recovery.environment()
    assert all(
        environment[key] == value for key, value in spending_policy_environment(policy).items()
    )
    assert environment["EXACT_OPENROUTER_LEDGER_DIR"] == str(tmp_path / "openrouter")


def test_prepared_successor_ignores_exhausted_aggregate_caps_and_preserves_state(policy, tmp_path):
    ledger = RequestLedger(tmp_path / "openrouter")
    state = {"limits": limits(), "work": {"old": {"requests": 20, "tokens": 300000000}}}
    before = copy.deepcopy(state)
    assert prepared_batch.hosted_caps(state, ledger.path.parent) == spending_policy_environment(
        policy
    )
    assert state == before


def test_prepared_worker_binds_policy_before_dispatch_and_preserves_parent(
    policy, tmp_path, monkeypatch
):
    worker = _worker(tmp_path, monkeypatch)
    recipe = prepared_batch.read(worker.path)
    environment = Path(recipe["environment"]["path"])
    prepared_batch.write(environment, spending_policy_environment(policy))
    recipe["environment"] = prepared_batch.binding(environment)
    prepared_batch.write(worker.path, recipe)
    parent_bytes = worker.parent.read_bytes()

    def execute(*args, **kwargs):
        assert os.environ["EXACT_OPENROUTER_TOKEN_CAP"] == ""
        assert os.environ["EXACT_OPENROUTER_REQUEST_CAP"] == ""
        ledger = RequestLedger(worker.runtime / "openrouter")
        ledger.sent(request(ledger, "above historical allowance", max_tokens=100000001))
        _outputs(worker)

    monkeypatch.setattr(experiment_resources, "guarded_execute", execute)
    prepared_batch.main()
    assert worker.parent.read_bytes() == parent_bytes
    assert prepared_batch.read(worker.root / "launch.json")["spending_policy"] == policy
    state = prepared_batch.read(worker.runtime / "budget.json")
    assert state["limits"] == worker.limits
    assert state["spending_policies"][policy["sha256"]] == policy
