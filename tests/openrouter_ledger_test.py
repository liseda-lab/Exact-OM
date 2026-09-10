"""Hosted request recovery through actual client routing, without network access."""

import json
import shutil
import sqlite3

import httpx
import pytest

from exact.llm.ledger import RequestLedger
from exact.llm.routing import LLMProfile, LLMRouter, OpenRouterClient


def setup_client(tmp_path, monkeypatch):
    client = OpenRouterClient()
    client.ledger_dir = tmp_path / "ledger"
    profile = LLMProfile(
        name="shared",
        backend="openrouter",
        model="vendor/model-v1",
        revision="weights-v1",
        tokenizer="tokenizer-v1",
    )
    monkeypatch.setattr(client, "resolve_api_key", lambda profile: "secret-not-in-ledger")
    monkeypatch.setattr(client, "_sleep_before_retry", lambda attempt: None)
    return client, profile


def response(payload, status=200):
    return httpx.Response(
        status,
        json=payload,
        request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
    )


def call(client, profile, **kwargs):
    return client.chat_completion(
        profile, [{"role": "user", "content": "A or Z?"}], max_tokens=4, role="decision", **kwargs
    )


def test_completed_requests_replay_after_relocation_without_credentials(tmp_path, monkeypatch):
    client, profile = setup_client(tmp_path, monkeypatch)
    calls = []
    raw = {
        "model": profile.model,
        "provider": "pinned",
        "choices": [{"message": {"content": "A"}}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 1, "cost": 0.003},
    }
    monkeypatch.setattr(client._client, "request", lambda **kw: calls.append(kw) or response(raw))
    first = call(client, profile)
    old = client.ledger_dir
    copied = tmp_path / "copied"
    shutil.copytree(old, copied)
    client.ledger_dir = copied
    shutil.rmtree(old)
    monkeypatch.setattr(client, "resolve_api_key", lambda profile: None)
    assert call(client, profile) == first
    assert len(calls) == 1
    ledger = RequestLedger(copied)
    usage = ledger.summary()["roles"]["decision"]
    assert usage["attempts"] == usage["completed"] == 1
    assert usage["reported_cost_usd"] == 0.003
    assert b"secret-not-in-ledger" not in ledger.path.read_bytes()


def test_unknown_delivery_is_retained_and_explicit_retry_has_separate_attempt(
    tmp_path, monkeypatch
):
    client, profile = setup_client(tmp_path, monkeypatch)
    calls = []

    def timeout(**kwargs):
        calls.append(kwargs)
        raise httpx.ReadTimeout("delivery unknown")

    monkeypatch.setattr(client._client, "request", timeout)
    with pytest.raises(RuntimeError, match="unknown delivery"):
        call(client, profile)
    assert len(calls) == 1
    with pytest.raises(RuntimeError, match="explicit retry authorization"):
        call(client, profile)
    client.retry_unknown_requests = True
    monkeypatch.setattr(client._client, "request", lambda **kw: response({"choices": []}))
    assert call(client, profile) == {"choices": []}
    summary = RequestLedger(client.ledger_dir).summary()["roles"]["decision"]
    assert summary["attempts"] == 2
    assert summary["unknown"] == 1
    assert summary["completed"] == 1
    assert summary["unpriced_attempts"] == 2


def test_raw_response_commits_before_parser_failure(tmp_path, monkeypatch):
    client, profile = setup_client(tmp_path, monkeypatch)
    calls = []
    invalid = httpx.Response(
        200, content=b"not json", request=httpx.Request("POST", "https://example.org")
    )
    monkeypatch.setattr(client._client, "request", lambda **kw: calls.append(kw) or invalid)
    for _ in range(2):
        with pytest.raises(ValueError):
            call(client, profile)
    assert len(calls) == 1
    assert RequestLedger(client.ledger_dir).summary()["roles"]["decision"]["completed"] == 1


def test_http_retries_record_each_raw_response_and_usage(tmp_path, monkeypatch):
    client, profile = setup_client(tmp_path, monkeypatch)
    responses = iter([response({"error": "busy"}, 429), response({"choices": []})])
    monkeypatch.setattr(client._client, "request", lambda **kw: next(responses))
    call(client, profile)
    db = sqlite3.connect(client.ledger_dir / "requests.sqlite3")
    assert db.execute("SELECT state FROM attempts ORDER BY number").fetchall() == [
        ("rejected",),
        ("completed",),
    ]
    db.close()


def test_model_role_prompt_and_provider_changes_create_new_request(tmp_path, monkeypatch):
    client, profile = setup_client(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(
        client._client, "request", lambda **kw: calls.append(kw) or response({"choices": []})
    )
    call(client, profile)
    call(client, profile, temperature=0.2)
    call(client, profile, provider={"only": ["new-provider"]})
    profile.revision = "new-revision"
    call(client, profile)
    client.chat_completion(
        profile, [{"role": "user", "content": "new prompt"}], max_tokens=4, role="summary"
    )
    assert len(calls) == 5
    assert RequestLedger(client.ledger_dir).summary()["roles"]["summary"]["completed"] == 1


def test_duplicate_sender_and_completed_checksum_validation(tmp_path):
    ledger = RequestLedger(tmp_path)
    key = ledger.plan({"role": "decision", "payload": {"model": "locked"}})
    number = ledger.sent(key)
    with pytest.raises(RuntimeError, match="sender"):
        ledger.sent(key)
    with pytest.raises(ValueError, match="alive"):
        ledger.recover_unknown(key)
    ledger.received(key, number, b'{"choices":[]}', 200)
    with sqlite3.connect(ledger.path) as db:
        db.execute("UPDATE attempts SET raw=?", (b"corrupt",))
    with pytest.raises(ValueError, match="Corrupted"):
        ledger.cached(key)


def test_experiment_mode_disables_profile_fallback_and_checks_response_identity(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("EXACT_EXPERIMENT_MODE", "1")
    local = LLMRouter(
        llm_profiles={"local": {"backend": "local_hf", "model": "model"}},
        llm_routing={"default_profile": "local"},
    )
    with pytest.raises(RuntimeError, match="requires an explicit OpenRouter"):
        local.resolve_task("decision")
    client, profile = setup_client(tmp_path, monkeypatch)
    captured = []
    monkeypatch.setattr(
        client._client,
        "request",
        lambda **kw: captured.append(kw) or response({"model": "changed", "choices": []}),
    )
    with pytest.raises(RuntimeError, match="model identity"):
        call(client, profile)
    assert json.loads(captured[0]["content"])["provider"]["allow_fallbacks"] is False


def test_request_budget_stops_before_wire_but_allows_cached_replay(tmp_path, monkeypatch):
    client, profile = setup_client(tmp_path, monkeypatch)
    monkeypatch.setenv("EXACT_OPENROUTER_REQUEST_CAP", "1")
    monkeypatch.setenv("EXACT_OPENROUTER_TOKEN_CAP", "10000")
    calls = []
    raw = {
        "model": profile.model,
        "choices": [],
        "usage": {"prompt_tokens": 7, "completion_tokens": 1},
    }
    monkeypatch.setattr(client._client, "request", lambda **kw: calls.append(kw) or response(raw))
    assert call(client, profile) == raw
    assert call(client, profile) == raw
    with pytest.raises(ValueError, match="request budget exhausted before transmission"):
        call(client, profile, temperature=0.7)
    assert len(calls) == 1


def test_unknown_delivery_keeps_reserved_tokens_and_concurrent_claims_share_cap(
    tmp_path, monkeypatch
):
    from concurrent.futures import ThreadPoolExecutor

    monkeypatch.setenv("EXACT_OPENROUTER_REQUEST_CAP", "1")
    monkeypatch.setenv("EXACT_OPENROUTER_TOKEN_CAP", "10000")
    ledger = RequestLedger(tmp_path)
    keys = [
        ledger.plan({"payload": {"max_tokens": 5, "messages": [{"content": str(i)}]}})
        for i in range(2)
    ]

    def claim(key):
        try:
            return key, ledger.sent(key)
        except ValueError:
            return key, None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(claim, keys))
    assert sum(number is not None for _, number in results) == 1
    key, number = next(row for row in results if row[1] is not None)
    ledger.unknown(key, number, "test ambiguous transmission")
    with sqlite3.connect(ledger.path) as db:
        reserved = db.execute("SELECT tokens FROM reservations").fetchone()[0]
    monkeypatch.setenv("EXACT_OPENROUTER_REQUEST_CAP", "2")
    monkeypatch.setenv("EXACT_OPENROUTER_TOKEN_CAP", str(reserved + 1))
    with pytest.raises(ValueError, match="token budget exhausted before transmission"):
        ledger.sent(key, retry_unknown=True)
    with sqlite3.connect(ledger.path) as db:
        assert db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 1
