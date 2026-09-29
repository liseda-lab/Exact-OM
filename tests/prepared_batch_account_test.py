"""Prepared successors preserve accounting lineage and hosted spending bounds."""

import copy
import hashlib
import os
import sqlite3
from pathlib import Path

import pytest

from exact.llm.ledger import RequestLedger
from tools import prepared_batch as batch


@pytest.fixture
def account():
    # Historical G0 is intentionally still reserved, but its hosted charges are real.
    return {
        "schema_version": 2,
        "limits": {
            "envelopes_hours": {"foundation": 12, "channels": 48, "final": 54},
            "node_hours_cap": 336,
            "requests_cap": 50000,
            "tokens_cap": 32000000,
            "final_requests_reserved": 5000,
            "final_tokens_reserved": 8000000,
        },
        "work": {
            "historical/G0": {
                "group": "foundation",
                "status": "reserved",
                "seconds": 43200,
                "requests": 4405,
                "tokens": 2726711,
                "projected_usd": 0.4454526,
                "actual_usd": 0.4454526,
                "accounting": "Retained historical reservation, not newly scheduled work",
            },
            "screen/E03/first": charge("complete", start=100, requests=2, tokens=40),
            "screen/E03/retry": charge("failed", start=120, requests=1, tokens=20),
            "screen/E19/prior": charge("interrupted", start=140),
        },
        "intervals": [[100, 110], [120, 130], [140, 150]],
        "allocations": {"14372": {"start": 90, "end": 150}},
        "allocation_seconds": 60,
    }


def charge(status, *, start=200, requests=0, tokens=0):
    return {
        "group": "channels",
        "status": status,
        "start": start,
        "end": start + 10,
        "seconds": 10,
        "requests": requests,
        "tokens": tokens,
        "projected_usd": 0,
        "actual_usd": 0,
    }


def registered(tmp_path, name, state, *, enabled=True, mtime=100):
    root = tmp_path / name
    budget = root / "runtime/budget.json"
    batch.write(budget, state)
    os.utime(budget, (mtime, mtime))
    report = root / "status.json"
    batch.write(report, {"status": "complete", "cumulative_budget": str(budget)})
    return {"id": name, "enabled": enabled, "status_path": str(report)}, budget


def test_latest_account_keeps_real_closed_statuses_and_g0_reservation(tmp_path, account):
    earlier, _ = registered(tmp_path, "earlier", account, mtime=300)
    successor = copy.deepcopy(account)
    successor["work"]["screen/E19/final"] = charge("complete")
    successor["intervals"].append([200, 210])
    successor["allocations"]["14372"]["end"] = 210
    successor["allocation_seconds"] = 120
    newer, path = registered(tmp_path, "newer", successor, mtime=100)
    selected, result = batch.latest_account({"runs": [earlier, newer]})
    assert selected == path
    assert result == successor
    assert result["work"]["historical/G0"] == account["work"]["historical/G0"]


def test_latest_account_uses_observation_time_for_equal_lineage(tmp_path, account):
    earlier, _ = registered(tmp_path, "earlier", account, mtime=100)
    later_state = copy.deepcopy(account)
    later_state["allocations"]["14372"]["end"] = 180
    later_state["allocation_seconds"] = 90
    later, path = registered(tmp_path, "later", later_state, mtime=200)
    assert batch.latest_account({"runs": [later, earlier]}) == (path, later_state)


@pytest.mark.parametrize("status", ["complete", "failed", "interrupted", "unknown"])
def test_latest_account_rejects_lost_or_rewritten_closed_charges(tmp_path, account, status):
    account["work"]["screen/E03/first"]["status"] = status
    earlier, _ = registered(tmp_path, "earlier", account)
    fork = copy.deepcopy(account)
    fork["work"]["screen/E03/first"]["tokens"] -= 1
    fork["work"]["screen/extra"] = charge("complete")
    newer, _ = registered(tmp_path, "newer", fork, mtime=200)
    with pytest.raises(ValueError, match="account|cumulative|charge"):
        batch.latest_account({"runs": [earlier, newer]})


@pytest.mark.parametrize("change", ["omit", "rewrite"])
def test_latest_account_cannot_drop_reserved_historical_g0_spending(tmp_path, account, change):
    earlier, _ = registered(tmp_path, "earlier", account)
    fork = copy.deepcopy(account)
    if change == "omit":
        del fork["work"]["historical/G0"]
    else:
        fork["work"]["historical/G0"]["tokens"] = 0
    for number in range(2):
        fork["work"][f"screen/extra{number}"] = charge("complete")
    newer, _ = registered(tmp_path, "newer", fork, mtime=200)
    with pytest.raises(ValueError, match="account|cumulative|historical"):
        batch.latest_account({"runs": [earlier, newer]})


def test_latest_account_cannot_increase_limits_without_amendment(tmp_path, account):
    earlier, _ = registered(tmp_path, "earlier", account)
    changed = copy.deepcopy(account)
    changed["limits"]["tokens_cap"] *= 2
    changed["work"]["screen/extra"] = charge("complete")
    newer, _ = registered(tmp_path, "newer", changed, mtime=200)
    with pytest.raises(ValueError, match="account|cumulative|limit"):
        batch.latest_account({"runs": [earlier, newer]})


@pytest.mark.parametrize("change", ["omit", "shorten"])
def test_latest_account_retains_scheduler_observed_allocation(tmp_path, account, change):
    earlier, _ = registered(tmp_path, "earlier", account)
    changed = copy.deepcopy(account)
    if change == "omit":
        changed.pop("allocations")
        changed.pop("allocation_seconds")
    else:
        changed["allocations"]["14372"]["end"] = 140
        changed["allocation_seconds"] = 50
    changed["work"]["screen/extra"] = charge("complete")
    newer, _ = registered(tmp_path, "newer", changed, mtime=200)
    with pytest.raises(ValueError, match="account|cumulative|allocation"):
        batch.latest_account({"runs": [earlier, newer]})


def test_latest_account_refuses_live_or_unclosed_scientific_reservation(tmp_path, account):
    account["work"]["screen/E19/pending"] = {
        "group": "channels",
        "status": "reserved",
        "seconds": 100,
        "requests": 0,
        "tokens": 0,
    }
    run, _ = registered(tmp_path, "live", account)
    with pytest.raises(ValueError, match="unclosed"):
        batch.latest_account({"runs": [run]})


def test_latest_account_ignores_disabled_and_unmaterialized_runs(tmp_path, account):
    enabled, path = registered(tmp_path, "active", account)
    ignored = copy.deepcopy(account)
    ignored["work"]["screen/extra"] = charge("complete")
    disabled, _ = registered(tmp_path, "disabled", ignored, enabled=False, mtime=300)
    missing = {"id": "unstarted", "status_path": str(tmp_path / "absent/status.json")}
    assert batch.latest_account({"runs": [disabled, missing, enabled]}) == (path, account)
    with pytest.raises(ValueError, match="authoritative"):
        batch.latest_account({"runs": [disabled, missing]})


@pytest.fixture
def wire(tmp_path, monkeypatch):
    directory = tmp_path / "source/runtime/openrouter"
    monkeypatch.setenv("EXACT_OPENROUTER_REQUEST_CAP", "50000")
    monkeypatch.setenv("EXACT_OPENROUTER_TOKEN_CAP", "32000000")
    ledger = RequestLedger(directory)
    for name, response in (("completed", 200), ("rejected", 429), ("unknown", None)):
        key = ledger.plan({"role": "development", "payload": {"max_tokens": 25, "name": name}})
        attempt = ledger.sent(key)
        if response is None:
            ledger.unknown(key, attempt, "connection dropped")
        else:
            ledger.received(key, attempt, b'{"fixture":true}', response)
            ledger.usage(key, attempt, {"prompt_tokens": 5, "completion_tokens": 7, "cost": 0.001})
    with sqlite3.connect(ledger.path) as db:
        reserved = db.execute(
            "SELECT tokens FROM reservations r JOIN attempts a USING(request_id,number) WHERE state='unknown'"
        ).fetchone()[0]
    return directory, ledger, 24 + reserved


def test_hosted_caps_keep_final_reserve_and_charge_unknown_wire_tokens(account, wire):
    directory, _, billable = wire
    original = copy.deepcopy(account)
    caps = batch.hosted_caps(account, directory)
    assert caps == {
        "EXACT_OPENROUTER_REQUEST_CAP": str(3 + 50000 - 5000 - 4408),
        "EXACT_OPENROUTER_TOKEN_CAP": str(billable + 32000000 - 8000000 - 2726771),
        "EXACT_OPENROUTER_RETRY_UNKNOWN": "0",
    }
    assert account == original


@pytest.mark.parametrize("plural", ["requests", "tokens"])
def test_hosted_caps_refuse_exhausted_protected_spending(account, wire, plural):
    directory, _, _ = wire
    account["work"]["screen/E03/first"][plural] = account["limits"][plural + "_cap"]
    with pytest.raises(ValueError, match="spending exhausted"):
        batch.hosted_caps(account, directory)


def test_zero_remaining_budget_freezes_caps_at_actual_wire_usage(account, wire):
    directory, _, billable = wire
    for plural in ("requests", "tokens"):
        account["limits"][plural + "_cap"] = account["limits"][
            "final_" + plural + "_reserved"
        ] + sum(row[plural] for row in account["work"].values())
    caps = batch.hosted_caps(account, directory)
    assert caps["EXACT_OPENROUTER_REQUEST_CAP"] == "3"
    assert caps["EXACT_OPENROUTER_TOKEN_CAP"] == str(billable)


def test_copy_account_takes_sqlite_backup_including_committed_wal(tmp_path, account, wire):
    directory, ledger, _ = wire
    source = directory.parent / "budget.json"
    batch.write(source, account)
    source_bytes = source.read_bytes()
    destination = tmp_path / "successor/runtime"
    # Keep a WAL connection open so a byte-copy of the database alone would lose this row.
    with sqlite3.connect(ledger.path) as source_db:
        assert source_db.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
        source_db.execute("PRAGMA wal_autocheckpoint=0")
        source_db.execute("CREATE TABLE snapshot_marker (value TEXT)")
        source_db.execute("INSERT INTO snapshot_marker VALUES ('committed-in-wal')")
        source_db.commit()
        assert Path(str(ledger.path) + "-wal").stat().st_size > 0
        batch.copy_account(source, account, destination)
        with sqlite3.connect(destination / "openrouter/requests.sqlite3") as copy_db:
            assert copy_db.execute("PRAGMA quick_check").fetchone() == ("ok",)
            assert copy_db.execute("SELECT value FROM snapshot_marker").fetchone() == (
                "committed-in-wal",
            )
            for table in ("requests", "reservations", "attempts"):
                assert (
                    copy_db.execute(f"SELECT * FROM {table}").fetchall()
                    == source_db.execute(f"SELECT * FROM {table}").fetchall()
                )
    assert source.read_bytes() == source_bytes
    assert batch.read(destination / "budget.json") == account
    receipt = batch.read(destination / "account-import.json")
    assert batch.verified(receipt["source"]) == source
    assert receipt["request_ledger"] == str(ledger.path)
    batch.copy_account(source, account, destination)
    assert batch.read(destination / "account-import.json") == receipt


def test_copy_account_never_overwrites_different_destination_budget(tmp_path, account, wire):
    directory, _, _ = wire
    source = directory.parent / "budget.json"
    batch.write(source, account)
    destination = tmp_path / "successor/runtime"
    conflicting = copy.deepcopy(account)
    conflicting["work"]["screen/own"] = charge("failed")
    batch.write(destination / "budget.json", conflicting)
    with pytest.raises(ValueError, match="Immutable"):
        batch.copy_account(source, account, destination)
    assert batch.read(destination / "budget.json") == conflicting
    assert not (destination / "account-import.json").exists()


@pytest.mark.parametrize("broken", ["missing", "corrupt"])
def test_copy_account_rejects_unreadable_request_history(tmp_path, account, broken):
    source = tmp_path / "source/budget.json"
    batch.write(source, account)
    old = source.parent / "openrouter/requests.sqlite3"
    if broken == "corrupt":
        old.parent.mkdir()
        old.write_bytes(b"not a sqlite database")
    destination = tmp_path / "successor/runtime"
    with pytest.raises((sqlite3.DatabaseError, ValueError)):
        batch.copy_account(source, account, destination)
    assert not (destination / "openrouter/requests.sqlite3").exists()
    assert not (destination / "account-import.json").exists()


def test_resolve_run_follows_replacement_chain_without_using_old_completion():
    rows = [
        {"id": "old", "enabled": False, "superseded_by": "repair"},
        {"id": "repair", "enabled": False, "superseded_by": "current"},
        {"id": "current", "enabled": True},
    ]
    assert batch.resolve_run({"runs": rows}, "old") is rows[-1]
    assert batch.resolve_run({"runs": rows}, "current") is rows[-1]


def test_resolve_run_rejects_cycles_and_missing_replacement():
    rows = [{"id": "first", "superseded_by": "second"}, {"id": "second", "superseded_by": "first"}]
    with pytest.raises(ValueError, match="Cyclic"):
        batch.resolve_run({"runs": rows}, "first")
    with pytest.raises(ValueError, match="no completed submission"):
        batch.resolve_run({"runs": rows[:1]}, "first")


def test_binding_resolves_path_and_detects_modified_bytes(tmp_path):
    artifact = tmp_path / "receipt.json"
    artifact.write_bytes(b'{"status":"complete"}\n')
    alias = tmp_path / "receipt-link.json"
    alias.symlink_to(artifact.name)
    value = batch.binding(alias)
    assert value == {
        "path": str(artifact),
        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
    }
    assert batch.verified(value) == artifact
    artifact.write_bytes(b'{"status":"failed"}\n')
    with pytest.raises(ValueError, match="binding changed"):
        batch.verified(value)


def test_completed_run_verifies_upstream_bytes_and_scientific_status(tmp_path):
    selection, manifest, completion = (
        tmp_path / name for name in ("selection.json", "manifest.json", "completion.json")
    )
    batch.write(selection, {"experiments": {}})
    batch.write(manifest, {"status": "complete"})
    receipt = {
        "status": "complete",
        "exit_code": 0,
        "selection": batch.binding(selection),
        "manifests": [batch.binding(manifest)],
    }
    batch.write(completion, receipt)
    run = {"id": "upstream", "completion_path": str(completion)}
    assert batch.completed_run(run) == receipt
    batch.write(manifest, {"status": "failed"})
    with pytest.raises(ValueError, match="binding changed"):
        batch.completed_run(run)
    receipt["manifests"] = [batch.binding(manifest)]
    batch.write(completion, receipt)
    with pytest.raises(ValueError, match="Incomplete upstream cell"):
        batch.completed_run(run)
