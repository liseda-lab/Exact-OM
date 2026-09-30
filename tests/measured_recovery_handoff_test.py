"""A partial cache transfer cannot authorize stale hosted request history."""

import copy
import sqlite3

import pytest

from tools import finalize_measured_recovery as recovery
from tools import prepared_batch as batch


def test_missing_wire_history_does_not_publish_authoritative_account(tmp_path):
    source = tmp_path / "parent/budget.json"
    state = {"work": {}, "limits": {}}
    batch.write(source, state)
    target = tmp_path / "new"
    with pytest.raises(sqlite3.OperationalError):
        batch.copy_account(source, state, target)
    assert not (target / "budget.json").exists()


def test_explicit_verified_request_history_keeps_newest_budget(tmp_path):
    state = {"work": {"failed-later": {"status": "failed"}}, "limits": {}}
    source = tmp_path / "partial/budget.json"
    batch.write(source, state)
    wire = tmp_path / "verified/requests.sqlite3"
    wire.parent.mkdir()
    with sqlite3.connect(wire) as db:
        db.execute("CREATE TABLE receipt (value TEXT)")
        db.execute("INSERT INTO receipt VALUES ('uncertain-charge')")
    target = tmp_path / "new"
    batch.copy_account(source, state, target, request_ledger=batch.binding(wire))
    assert batch.read(target / "budget.json") == state
    with sqlite3.connect(target / "openrouter/requests.sqlite3") as db:
        assert db.execute("SELECT value FROM receipt").fetchall() == [("uncertain-charge",)]


@pytest.mark.parametrize("field", ["requests", "tokens", "actual_usd"])
def test_older_wire_history_rejects_newer_charges(tmp_path, field):
    original = {"work": {}, "limits": {}}
    budget = tmp_path / "original.json"
    batch.write(budget, original)
    state = copy.deepcopy(original)
    state["work"]["later"] = {"status": "failed", field: 1}
    with pytest.raises(ValueError, match="Newer hosted charges"):
        recovery.retained_request_history({"request_budget": batch.binding(budget)}, state)


def test_older_wire_history_rejects_rewritten_parent_charge(tmp_path):
    original = {"work": {"paid": {"status": "complete", "requests": 1}}, "limits": {}}
    budget = tmp_path / "original.json"
    batch.write(budget, original)
    state = copy.deepcopy(original)
    state["work"]["paid"]["requests"] = 0
    with pytest.raises(ValueError, match="lineage"):
        recovery.retained_request_history({"request_budget": batch.binding(budget)}, state)
