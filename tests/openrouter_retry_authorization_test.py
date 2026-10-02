"""One-shot approvals survive copying without weakening unknown-delivery guards."""

import json
import os
import shutil
import socket
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from exact.llm.ledger import RequestLedger
from tests.openrouter_ledger_test import call, response, setup_client
from tools.authorize_hosted_retries import apply_authorizations
from tools.prepared_batch import binding


def pending(ledger, monkeypatch):
    monkeypatch.setenv("EXACT_OPENROUTER_REQUEST_CAP", "10")
    monkeypatch.setenv("EXACT_OPENROUTER_TOKEN_CAP", "10000")
    key = ledger.plan({"payload": {"max_tokens": 4, "prompt": "fixture"}})
    ledger.sent(key)
    return key


def rows(ledger, table):
    with sqlite3.connect(ledger.path) as db:
        return db.execute("SELECT * FROM " + table).fetchall()


def test_dead_sender_only_and_idempotent_approval(tmp_path, monkeypatch):
    ledger = RequestLedger(tmp_path)
    key = pending(ledger, monkeypatch)
    before = rows(ledger, "reservations")
    with pytest.raises(ValueError, match="still alive"):
        ledger.authorize_unknown_once(key, attempt=1, authorization_id="approval")
    assert rows(ledger, "retry_authorizations") == []

    def exited(pid, signal):
        assert pid == os.getpid()
        raise ProcessLookupError()

    monkeypatch.setattr("exact.llm.ledger.os.kill", exited)
    ledger.authorize_unknown_once(key, attempt=1, authorization_id="approval")
    ledger.authorize_unknown_once(key, attempt=1, authorization_id="approval")
    assert rows(ledger, "reservations") == before
    assert rows(ledger, "attempts")[0][2] == "unknown"
    assert ledger.sent(key) == 2
    ledger.authorize_unknown_once(key, attempt=1, authorization_id="approval")
    with pytest.raises(RuntimeError, match="already used"):
        ledger.sent(key, retry_unknown=True)
    assert rows(ledger, "reservations") == before + [(key, 2, before[0][2])]
    assert rows(ledger, "retry_authorizations") == [(key, 1, "approval", 2)]
    with pytest.raises(ValueError, match="identity changed"):
        ledger.authorize_unknown_once(key, attempt=1, authorization_id="different")


def test_remote_sender_wrong_attempt_and_completed_not_approved(tmp_path, monkeypatch):
    ledger = RequestLedger(tmp_path)
    key = pending(ledger, monkeypatch)
    with ledger._transaction() as db:
        db.execute("UPDATE attempts SET host='remote'")
    with pytest.raises(ValueError, match="remote sender"):
        ledger.authorize_unknown_once(key, attempt=1, authorization_id="approval")
    with pytest.raises(ValueError, match="latest unresolved"):
        ledger.authorize_unknown_once(key, attempt=2, authorization_id="approval")
    ledger.received(key, 1, b"{}", 200)
    with pytest.raises(ValueError, match="latest unresolved"):
        ledger.authorize_unknown_once(key, attempt=1, authorization_id="approval")


@pytest.mark.parametrize("outcome", ["unknown", "rejected"])
def test_only_one_claim_even_concurrently_and_after_copy(tmp_path, monkeypatch, outcome):
    ledger = RequestLedger(tmp_path / "original")
    key = pending(ledger, monkeypatch)
    ledger.unknown(key, 1, "fixture")
    ledger.authorize_unknown_once(key, attempt=1, authorization_id="approval")
    other = ledger.plan({"payload": {"max_tokens": 4, "prompt": "other"}})
    ledger.sent(other)
    ledger.unknown(other, 1, "fixture")
    with pytest.raises(RuntimeError, match="explicit retry authorization"):
        ledger.sent(other)

    def claim(_):
        try:
            return ledger.sent(key)
        except RuntimeError:
            return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        claimed = list(pool.map(claim, range(8)))
    assert claimed.count(2) == 1
    if outcome == "unknown":
        ledger.unknown(key, 2, "second ambiguity")
    else:
        ledger.received(key, 2, b'{"error":"busy"}', 429)
    shutil.copytree(tmp_path / "original", tmp_path / "copied")
    copied = RequestLedger(tmp_path / "copied")
    copied.authorize_unknown_once(key, attempt=1, authorization_id="approval")
    with pytest.raises(RuntimeError, match="already used"):
        copied.sent(key)
    assert len(rows(copied, "attempts")) == 3
    assert rows(copied, "reservations") == rows(ledger, "reservations")


def test_failed_budget_admission_does_not_consume_approval(tmp_path, monkeypatch):
    ledger = RequestLedger(tmp_path)
    key = pending(ledger, monkeypatch)
    ledger.unknown(key, 1, "fixture")
    ledger.authorize_unknown_once(key, attempt=1, authorization_id="approval")
    monkeypatch.setenv("EXACT_OPENROUTER_REQUEST_CAP", "1")
    with pytest.raises(ValueError, match="budget exhausted"):
        ledger.sent(key)
    assert rows(ledger, "retry_authorizations")[0][-1] is None
    monkeypatch.setenv("EXACT_OPENROUTER_REQUEST_CAP", "10")
    assert ledger.sent(key) == 2


@pytest.mark.parametrize("retry_status", [200, 429])
def test_client_obeys_scoped_approval_without_global_override(tmp_path, monkeypatch, retry_status):
    import httpx

    client, profile = setup_client(tmp_path, monkeypatch)
    monkeypatch.setenv("EXACT_OPENROUTER_RETRY_UNKNOWN", "0")
    monkeypatch.setattr(
        client._client, "request", lambda **kw: (_ for _ in ()).throw(httpx.ReadTimeout("fixture"))
    )
    with pytest.raises(RuntimeError, match="unknown delivery"):
        call(client, profile)
    ledger = RequestLedger(client.ledger_dir)
    key = rows(ledger, "attempts")[0][0]
    ledger.authorize_unknown_once(key, attempt=1, authorization_id="approval")
    calls = []
    monkeypatch.setattr(
        client._client,
        "request",
        lambda **kw: calls.append(kw) or response({"choices": []}, retry_status),
    )
    if retry_status == 200:
        assert call(client, profile) == {"choices": []}
        assert call(client, profile) == {"choices": []}
    else:
        with pytest.raises(RuntimeError, match="already used"):
            call(client, profile)
    assert len(calls) == 1
    assert len(rows(ledger, "attempts")) == 2


def test_approval_file_is_bound_to_copied_account_and_original_reservation(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime/campaign"
    ledger = RequestLedger(runtime / "openrouter")
    key = pending(ledger, monkeypatch)
    ledger.unknown(key, 1, "fixture")
    original = rows(ledger, "attempts")[0]
    approval = tmp_path / "approval.json"
    document = dict(
        schema_version=1,
        approved_by="user",
        user_message="Yes, approved",
        scientific_step="E07",
        scope="exactly_one_additional_wire_attempt",
        requests=[
            dict(
                request_id=key,
                attempt=1,
                reserved_tokens=rows(ledger, "reservations")[0][2],
                started=original[10],
                host=socket.gethostname(),
                pid=os.getpid(),
            )
        ],
    )
    approval.write_text(json.dumps(document))
    recipe = dict(
        root=str(tmp_path),
        campaign_id="campaign",
        scientific_step="E07",
        hosted_retry_authorizations=binding(approval),
    )
    with pytest.raises(ValueError, match="copied account"):
        apply_authorizations(recipe, runtime)
    (runtime / "account-import.json").write_text("{}")
    apply_authorizations(recipe, runtime)
    apply_authorizations(recipe, runtime)
    assert (runtime / "retry-authorization.json").is_file()
    assert len(rows(ledger, "retry_authorizations")) == 1
    document["requests"][0]["reserved_tokens"] += 1
    approval.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="binding changed"):
        apply_authorizations(recipe, runtime)
    recipe["hosted_retry_authorizations"] = binding(approval)
    with pytest.raises(ValueError, match="reservation differs"):
        apply_authorizations(recipe, runtime)


def test_legacy_writer_cannot_bypass_consumed_grant_after_account_copy(tmp_path, monkeypatch):
    from tools.prepared_batch import copy_account

    original = tmp_path / "original"
    ledger = RequestLedger(original / "openrouter")
    key = pending(ledger, monkeypatch)
    ledger.unknown(key, 1, "fixture")
    ledger.authorize_unknown_once(key, attempt=1, authorization_id="approval")
    ledger.received(key, ledger.sent(key), b'{"error":"busy"}', 429)
    (original / "budget.json").write_text('{"work":{}}')
    target = tmp_path / "copied"
    copy_account(original / "budget.json", {"work": {}}, target)
    copied = RequestLedger(target / "openrouter")
    # Old clients know only attempts/reservations, not the new Python guard.
    before = rows(copied, "reservations")
    with pytest.raises(sqlite3.IntegrityError, match="Approved paid retry already used"):
        with copied._transaction() as db:
            db.execute("INSERT INTO reservations VALUES (?,?,?)", (key, 3, before[0][2]))
            db.execute(
                "INSERT INTO attempts(request_id,number,state,pid,host) VALUES (?,?,?,?,?)",
                (key, 3, "sent", os.getpid(), socket.gethostname()),
            )
    assert rows(copied, "reservations") == before
    assert len(rows(copied, "attempts")) == 2
    other = copied.plan({"payload": {"max_tokens": 4, "prompt": "unrelated"}})
    assert copied.sent(other) == 1


def test_a_later_explicit_approval_can_authorize_a_new_ambiguous_attempt(tmp_path, monkeypatch):
    ledger = RequestLedger(tmp_path)
    key = pending(ledger, monkeypatch)
    ledger.unknown(key, 1, "fixture")
    ledger.authorize_unknown_once(key, attempt=1, authorization_id="approval-1")
    ledger.unknown(key, ledger.sent(key), "retry ambiguous")
    ledger.authorize_unknown_once(key, attempt=2, authorization_id="separate-approval-2")
    assert ledger.sent(key) == 3
    with pytest.raises(RuntimeError, match="already used"):
        ledger.sent(key)
