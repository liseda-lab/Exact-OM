"""Shared spending cannot be reset by copied request ledgers or recovery names."""

import hashlib
import json
import shutil
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest

from exact.llm import spending_admission as admission
from exact.llm.ledger import RequestLedger
from exact.llm.prompt_budget import PromptBudgetError
from exact.llm.routing import LLMProfile, OpenRouterClient
from exact.utils.hosted_spending import (
    POLICY_PATH_ENV,
    POLICY_SHA256_ENV,
    load_spending_policy,
)


def bound(path, value):
    raw = json.dumps(value, sort_keys=True).encode()
    path.write_bytes(raw)
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for name in (
        POLICY_PATH_ENV,
        POLICY_SHA256_ENV,
        admission.CAMPAIGN_ENV,
        admission.EXPERIMENT_ENV,
        "EXACT_EXPERIMENT_MODE",
        "EXACT_OPENROUTER_LEDGER_DIR",
        "EXACT_OPENROUTER_REQUEST_CAP",
        "EXACT_OPENROUTER_TOKEN_CAP",
        "EXACT_LLM_PROMPT_POLICY_PATH",
        "EXACT_LLM_PROMPT_POLICY_SHA256",
    ):
        monkeypatch.delenv(name, raising=False)


def install(
    tmp_path,
    monkeypatch,
    *,
    total=100,
    experiment=40,
    campaign_cap=10000,
    experiment_cap=8000,
    initialize=True,
):
    source = bound(tmp_path / "historical.json", {"tokens": total})
    bootstrap = bound(
        tmp_path / "bootstrap.json",
        {
            "schema_version": 1,
            "campaign_id": "campaign",
            "accounted_tokens": total,
            "experiment_tokens": {"E02": experiment},
            "reported_usd": 1.25,
            "unpriced_attempts": 2,
            "sources": [source],
        },
    )
    policy = {
        "schema_version": 2,
        "kind": "exact_om_hosted_spending_policy",
        "mode": "hard_pause",
        "authorization": "User approved fixture limits",
        "notification_tokens": 100,
        "campaign_id": "campaign",
        "campaign_tokens_cap": campaign_cap,
        "experiment_tokens_cap": experiment_cap,
        "experiment_warning_tokens": min(100, experiment_cap),
        "admission_store": str(tmp_path / "central.sqlite3"),
        "bootstrap": bootstrap,
    }
    select(tmp_path, monkeypatch, policy)
    monkeypatch.setenv(admission.CAMPAIGN_ENV, "campaign")
    monkeypatch.setenv(admission.EXPERIMENT_ENV, "E02")
    if initialize:
        admission.initialize_admission_store()
    return policy


def select(tmp_path, monkeypatch, policy):
    digest = hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()
    binding = bound(tmp_path / f"policy-{digest}.json", policy)
    monkeypatch.setenv(POLICY_PATH_ENV, binding["path"])
    monkeypatch.setenv(POLICY_SHA256_ENV, binding["sha256"])
    return load_spending_policy()


def reserve(tokens=20, key="request"):
    return admission.reserve_attempt(
        admission.selected_policy(),
        tokens=tokens,
        request_id=key,
        local_ledger="/fixture/requests.sqlite3",
        attempt=1,
    )


def planned(ledger, text="prompt"):
    return ledger.plan({"payload": {"model": "fixture", "prompt": text, "max_tokens": 8}})


def rows(ledger, table):
    with sqlite3.connect(ledger.path) as db:
        return db.execute("SELECT * FROM " + table).fetchall()


def test_explicit_bootstrap_once_preserves_history_and_no_false_pause(tmp_path, monkeypatch):
    install(
        tmp_path,
        monkeypatch,
        total=96190067,
        experiment=72193688,
        campaign_cap=200000000,
        experiment_cap=25000000,
    )
    first = admission.admission_snapshot()
    assert first["accounted_tokens"] == 96190067
    assert first["experiments"]["E02"]["historical_tokens"] == 72193688
    assert first["experiments"]["E02"]["status"] == "over_cap"
    assert first["pauses"] == [] and first["paused_experiments"] == []
    assert first["reported_cost_usd"] == 1.25 and first["unpriced_attempts"] == 2
    assert admission.initialize_admission_store() == first
    with pytest.raises(admission.HostedSpendPause) as caught:
        reserve()
    assert caught.value.details["scope"] == "experiment"
    after = admission.admission_snapshot()
    assert after["accounted_tokens"] == first["accounted_tokens"]
    assert after["paused_experiments"] == ["E02"]
    assert after["pauses"][0]["ledger_path"] == "/fixture/requests.sqlite3"
    assert after["pauses"][0]["request_id"] == "request"


def test_worker_never_creates_missing_store(tmp_path, monkeypatch):
    policy = install(tmp_path, monkeypatch, initialize=False)
    with pytest.raises(admission.HostedSpendingError):
        reserve()
    assert not Path(policy["admission_store"]).exists()


@pytest.mark.parametrize("missing", [admission.CAMPAIGN_ENV, admission.EXPERIMENT_ENV])
def test_missing_scope_rejects_without_claim(tmp_path, monkeypatch, missing):
    install(tmp_path, monkeypatch)
    monkeypatch.delenv(missing)
    with pytest.raises(PromptBudgetError, match="scope"):
        reserve()
    assert admission.admission_snapshot()["accounted_tokens"] == 100


def test_store_binding_mismatch_cannot_reset_history(tmp_path, monkeypatch):
    policy = install(tmp_path, monkeypatch)
    copy = tmp_path / "fork.sqlite3"
    shutil.copyfile(policy["admission_store"], copy)
    copy.with_suffix(".sqlite3.lock").touch()
    select(tmp_path, monkeypatch, {**policy, "admission_store": str(copy)})
    with pytest.raises(admission.HostedSpendingError, match="identity"):
        reserve()


def test_concurrent_claims_share_one_atomic_campaign_limit(tmp_path, monkeypatch):
    install(tmp_path, monkeypatch, total=0, experiment=0, campaign_cap=500, experiment_cap=1000)

    def claim(number):
        try:
            return reserve(100, str(number))
        except admission.HostedSpendPause:
            return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(claim, range(20)))
    assert sum(value is not None for value in results) == 5
    snapshot = admission.admission_snapshot()
    assert snapshot["accounted_tokens"] == snapshot["reserved_tokens"] == 500
    assert snapshot["paused"] is True
    assert len(snapshot["pauses"]) == 1


def test_family_cap_persists_across_recovery_and_does_not_block_other_family(tmp_path, monkeypatch):
    policy = install(tmp_path, monkeypatch, total=0, experiment=0, experiment_cap=100)
    first = reserve(60)
    with pytest.raises(admission.HostedSpendPause):
        reserve(60, "recovery")
    admission.settle_attempt(
        admission.selected_policy(), first, {"prompt_tokens": 1, "completion_tokens": 1}
    )
    # Settling an earlier reservation is not an authorization to resume a paused scope.
    with pytest.raises(admission.HostedSpendPause):
        reserve(1)
    monkeypatch.setenv(admission.EXPERIMENT_ENV, "E03")
    reserve(60)
    monkeypatch.setenv(admission.EXPERIMENT_ENV, "E02")
    select(tmp_path, monkeypatch, {**policy, "authorization": "Reworded only"})
    with pytest.raises(admission.HostedSpendPause):
        reserve(1)
    select(tmp_path, monkeypatch, {**policy, "experiment_tokens_caps": {"E02": 200}})
    reserve(10)
    snapshot = admission.admission_snapshot()
    assert snapshot["experiments"]["E02"]["tokens_cap"] == 200
    assert snapshot["experiments"]["E03"]["tokens_cap"] == 100
    assert snapshot["paused_experiments"] == []
    assert snapshot["pauses"][0]["active"] is False


def test_partial_missing_invalid_usage_and_unknown_cost_remain_conservative(tmp_path, monkeypatch):
    install(tmp_path, monkeypatch)
    grant = reserve(100)
    selected = admission.selected_policy()
    for usage in (
        {},
        {"prompt_tokens": None, "completion_tokens": 1},
        {"prompt_tokens": -1, "completion_tokens": False},
    ):
        admission.settle_attempt(selected, grant, usage)
        assert admission.admission_snapshot()["accounted_tokens"] == 200
    admission.settle_attempt(selected, grant, {"prompt_tokens": 150, "cost": 0.25})
    assert admission.admission_snapshot()["accounted_tokens"] == 250
    admission.settle_attempt(selected, grant, {"prompt_tokens": 150, "cost": 0.25})
    assert admission.admission_snapshot()["accounted_tokens"] == 250
    admission.settle_attempt(
        selected, grant, {"prompt_tokens": 150, "completion_tokens": 5, "cost": 0.25}
    )
    snapshot = admission.admission_snapshot()
    admission.settle_attempt(
        selected, grant, {"prompt_tokens": 150, "completion_tokens": 5, "cost": 0.25}
    )
    assert admission.admission_snapshot() == snapshot
    assert snapshot["accounted_tokens"] == 255 and snapshot["reserved_tokens"] == 0
    assert snapshot["reported_cost_usd"] == 1.5 and snapshot["unpriced_attempts"] == 2
    with pytest.raises(admission.HostedSpendingError, match="rewritten"):
        admission.settle_attempt(selected, grant, {"prompt_tokens": 1, "completion_tokens": 1})


def test_local_rejection_has_no_sent_unknown_reservation_or_consumed_retry(tmp_path, monkeypatch):
    policy = install(tmp_path, monkeypatch, total=0, experiment=0)
    ledger = RequestLedger(tmp_path / "wire")
    key = planned(ledger)
    number = ledger.sent(key)
    ledger.unknown(key, number, "unknown delivery")
    ledger.authorize_unknown_once(key, attempt=number, authorization_id="fixture")
    snapshot = admission.admission_snapshot()
    select(
        tmp_path,
        monkeypatch,
        {
            **policy,
            "experiment_tokens_cap": snapshot["accounted_tokens"],
            "experiment_warning_tokens": 1,
        },
    )
    before = {
        name: rows(ledger, name) for name in ("attempts", "reservations", "retry_authorizations")
    }
    with pytest.raises(admission.HostedSpendPause):
        ledger.sent(key)
    assert {name: rows(ledger, name) for name in before} == before
    assert admission.admission_snapshot()["accounted_tokens"] == snapshot["accounted_tokens"]


def test_forked_response_ledgers_cannot_deduplicate_new_paid_attempts(tmp_path, monkeypatch):
    install(tmp_path, monkeypatch, total=0, experiment=0)
    original = RequestLedger(tmp_path / "original")
    key = planned(original)
    shutil.copytree(original.path.parent, tmp_path / "fork")
    fork = RequestLedger(tmp_path / "fork")
    assert original.sent(key) == fork.sent(key) == 1
    with sqlite3.connect(original.path) as db:
        first = db.execute("SELECT spending_grant_id FROM attempts").fetchone()[0]
    with sqlite3.connect(fork.path) as db:
        second = db.execute("SELECT spending_grant_id FROM attempts").fetchone()[0]
    assert first != second
    reserved = rows(original, "reservations")[0][-1]
    assert admission.admission_snapshot()["accounted_tokens"] == 2 * reserved


def test_central_commit_survives_local_failure_as_conservative_orphan(tmp_path, monkeypatch):
    install(tmp_path, monkeypatch, total=0, experiment=0)
    ledger = RequestLedger(tmp_path / "wire")
    key = planned(ledger)
    with ledger._transaction() as db:
        db.execute(
            "CREATE TRIGGER fixture_fail BEFORE INSERT ON attempts BEGIN SELECT RAISE(ABORT,'fixture'); END"
        )
    with pytest.raises(sqlite3.IntegrityError):
        ledger.sent(key)
    assert rows(ledger, "attempts") == rows(ledger, "reservations") == []
    assert admission.admission_snapshot()["reserved_tokens"] > 0


def test_usage_settles_exact_grant_and_cache_replays_under_pause(tmp_path, monkeypatch):
    policy = install(tmp_path, monkeypatch, total=0, experiment=0)
    client = OpenRouterClient()
    client.ledger_dir = tmp_path / "wire"
    profile = LLMProfile(name="fixture", backend="openrouter", model="fixture")
    calls = []
    monkeypatch.setattr(client, "resolve_api_key", lambda profile: "fake")

    def request(**kwargs):
        calls.append(kwargs)
        return httpx.Response(
            200,
            json={
                "choices": [],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "cost": 0.1},
            },
            request=httpx.Request("POST", "https://example.invalid"),
        )

    monkeypatch.setattr(client._client, "request", request)
    try:
        first = client.completion(profile, "text", max_tokens=8)
        assert admission.admission_snapshot()["accounted_tokens"] == 12
        select(
            tmp_path,
            monkeypatch,
            {**policy, "experiment_tokens_cap": 12, "experiment_warning_tokens": 1},
        )
        assert client.completion(profile, "text", max_tokens=8) == first
        with pytest.raises(admission.HostedSpendPause):
            client.completion(profile, "new", max_tokens=8)
        assert client.completion(profile, "text", max_tokens=8) == first
        assert len(calls) == 1
        assert admission.admission_snapshot()["reported_cost_usd"] == 1.35
    finally:
        client.close()


def test_hard_policy_rejects_ledgerless_requests_before_network(tmp_path, monkeypatch):
    install(tmp_path, monkeypatch)
    client = OpenRouterClient()
    monkeypatch.setattr(client._client, "request", lambda **kwargs: pytest.fail("No hosted call"))
    try:
        with pytest.raises(admission.HostedSpendingError, match="durable request ledger"):
            client.completion(LLMProfile(name="fixture"), "text", max_tokens=1)
    finally:
        client.close()


@pytest.mark.parametrize(
    "overrides",
    [
        {"campaign_tokens_cap": True},
        {"experiment_tokens_cap": 0},
        {"experiment_tokens_caps": {"E21": -1}},
        {"experiment_tokens_caps": []},
        {"admission_store": "relative.sqlite3"},
        {"bootstrap": {}},
    ],
)
def test_invalid_hard_policy_cannot_disable_admission(tmp_path, monkeypatch, overrides):
    policy = install(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        select(tmp_path, monkeypatch, {**policy, **overrides})


def test_batched_outputs_and_each_input_framing_are_reserved(tmp_path, monkeypatch):
    install(tmp_path, monkeypatch, total=0, experiment=0)
    ledger = RequestLedger(tmp_path / "wire")
    payload = {"prompt": ["a", "b"], "max_tokens": 7}
    key = ledger.plan({"payload": payload})
    ledger.sent(key)
    expected = len(json.dumps(payload, ensure_ascii=False).encode()) + 256 * 2 + 7 * 2
    assert admission.admission_snapshot()["accounted_tokens"] == expected


def test_stored_bootstrap_tampering_fails_closed(tmp_path, monkeypatch):
    policy = install(tmp_path, monkeypatch)
    with sqlite3.connect(policy["admission_store"]) as db:
        record = json.loads(db.execute("SELECT record FROM metadata").fetchone()[0])
        record["bootstrap"]["accounted_tokens"] = 0
        db.execute("UPDATE metadata SET record=?", (json.dumps(record),))
    with pytest.raises(admission.HostedSpendingError, match="bootstrap bytes"):
        reserve()


@pytest.mark.parametrize(
    "scope", ["E21-recovery-03", "E23-rich-100", "unattributed", "E2", "other"]
)
def test_recovery_and_variant_names_cannot_reset_family_allowance(tmp_path, monkeypatch, scope):
    install(tmp_path, monkeypatch)
    monkeypatch.setenv(admission.EXPERIMENT_ENV, scope)
    with pytest.raises(admission.HostedSpendingError, match="canonical experiment scope"):
        reserve()
    assert admission.admission_snapshot()["accounted_tokens"] == 100


def test_deleted_authority_cannot_be_reseeded_by_rerunning_initializer(tmp_path, monkeypatch):
    policy = install(tmp_path, monkeypatch)
    reserve()
    Path(policy["admission_store"]).unlink()
    with pytest.raises(admission.HostedSpendingError, match="reconciliation required"):
        admission.initialize_admission_store()
    assert not Path(policy["admission_store"]).exists()


def test_missing_local_grant_policy_is_typed_and_keeps_exposure(tmp_path, monkeypatch):
    install(tmp_path, monkeypatch, total=0, experiment=0)
    ledger = RequestLedger(tmp_path / "wire")
    key = planned(ledger)
    number = ledger.sent(key)
    before = admission.admission_snapshot()["accounted_tokens"]
    with ledger._transaction() as db:
        db.execute("DELETE FROM spending_policies")
    with pytest.raises(PromptBudgetError, match="retained policy"):
        ledger.usage(key, number, {"prompt_tokens": 1, "completion_tokens": 1})
    assert admission.admission_snapshot()["accounted_tokens"] == before
