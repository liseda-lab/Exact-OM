"""Hosted request recovery through actual client routing, without network access."""

import hashlib
import json
import shutil
import sqlite3

import httpx
import pytest

from exact.llm.ledger import RequestLedger, request_identity
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
    ticks = iter((100.0, 101.25))
    monkeypatch.setattr("exact.llm.routing.monotonic", lambda: next(ticks))
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
    assert usage["measured_attempts"] == 1
    assert usage["unmeasured_attempts"] == 0
    assert usage["elapsed_seconds_total"] == 1.25
    with sqlite3.connect(ledger.path) as db:
        identity = db.execute("SELECT identity FROM requests").fetchone()[0]
    assert "elapsed_seconds" not in identity  # Timing never changes request/cache identity.
    assert b"secret-not-in-ledger" not in ledger.path.read_bytes()


def test_unknown_delivery_is_retained_and_explicit_retry_has_separate_attempt(
    tmp_path, monkeypatch
):
    client, profile = setup_client(tmp_path, monkeypatch)
    ticks = iter((100.0, 102.75, 200.0, 200.125))
    monkeypatch.setattr("exact.llm.routing.monotonic", lambda: next(ticks))
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
    assert summary["measured_attempts"] == 2
    assert summary["unmeasured_attempts"] == 0
    assert summary["elapsed_seconds_total"] == 2.875
    with sqlite3.connect(client.ledger_dir / "requests.sqlite3") as db:
        assert db.execute("SELECT elapsed_seconds FROM attempts ORDER BY number").fetchall() == [
            (2.75,),
            (0.125,),
        ]


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
    ticks = iter((10.0, 10.25, 50.0, 51.5))
    monkeypatch.setattr("exact.llm.routing.monotonic", lambda: next(ticks))
    responses = iter([response({"error": "busy"}, 429), response({"choices": []})])
    monkeypatch.setattr(client._client, "request", lambda **kw: next(responses))
    call(client, profile)
    db = sqlite3.connect(client.ledger_dir / "requests.sqlite3")
    assert db.execute("SELECT state FROM attempts ORDER BY number").fetchall() == [
        ("rejected",),
        ("completed",),
    ]
    assert db.execute("SELECT elapsed_seconds FROM attempts ORDER BY number").fetchall() == [
        (0.25,),
        (1.5,),
    ]  # Retry backoff is not charged to either wire duration.
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


def test_legacy_ledger_migration_preserves_raw_cache_and_unknown_timing(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    key, identity = request_identity({"role": "old", "payload": {"model": "locked"}})
    raw = b'{"choices":[]}'
    with sqlite3.connect(tmp_path / "requests.sqlite3") as db:
        db.executescript(
            """CREATE TABLE requests (request_id TEXT PRIMARY KEY, identity TEXT NOT NULL);
            CREATE TABLE attempts (
                request_id TEXT NOT NULL, number INTEGER NOT NULL, state TEXT NOT NULL,
                pid INTEGER NOT NULL, host TEXT NOT NULL, status INTEGER,
                raw BLOB, sha256 TEXT, usage TEXT, error TEXT,
                started TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(request_id, number)
            );"""
        )
        db.execute("INSERT INTO requests VALUES (?,?)", (key, identity))
        db.execute(
            "INSERT INTO attempts VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                key,
                1,
                "completed",
                1,
                "legacy",
                200,
                raw,
                hashlib.sha256(raw).hexdigest(),
                None,
                None,
                "2000-01-01 00:00:00",
            ),
        )
    # Concurrent first readers must not race ALTER TABLE; reopening is idempotent.
    with ThreadPoolExecutor(max_workers=4) as pool:
        ledgers = list(pool.map(lambda _: RequestLedger(tmp_path), range(4)))
    for ledger in ledgers:
        assert ledger.plan(json.loads(identity)) == key
        assert ledger.cached(key) == raw
        summary = ledger.summary()["roles"]["old"]
        assert summary["completed"] == 1
        assert summary["measured_attempts"] == 0
        assert summary["unmeasured_attempts"] == 1
        assert summary["elapsed_seconds_total"] is None
    with sqlite3.connect(ledgers[0].path) as db:
        assert db.execute("SELECT started,elapsed_seconds FROM attempts").fetchone() == (
            "2000-01-01 00:00:00",
            None,
        )


def test_interrupted_wire_keeps_measured_unknown_attempt(tmp_path, monkeypatch):
    client, profile = setup_client(tmp_path, monkeypatch)
    ticks = iter((4.0, 6.0))
    monkeypatch.setattr("exact.llm.routing.monotonic", lambda: next(ticks))

    def interrupted(**kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(client._client, "request", interrupted)
    with pytest.raises(KeyboardInterrupt):
        call(client, profile)
    summary = RequestLedger(client.ledger_dir).summary()["roles"]["decision"]
    assert summary["unknown"] == summary["measured_attempts"] == 1
    assert summary["elapsed_seconds_total"] == 2.0


@pytest.mark.parametrize("elapsed", [-1.0, float("inf"), float("nan")])
@pytest.mark.parametrize("method", ["received", "unknown"])
def test_invalid_elapsed_time_cannot_finalize_pending_attempt(tmp_path, elapsed, method):
    ledger = RequestLedger(tmp_path)
    key = ledger.plan({"role": "decision"})
    number = ledger.sent(key)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        if method == "received":
            ledger.received(key, number, b"{}", 200, elapsed_seconds=elapsed)
        else:
            ledger.unknown(key, number, "timeout", elapsed_seconds=elapsed)
    summary = ledger.summary()["roles"]["decision"]
    assert summary["completed"] == 0
    assert summary["unmeasured_attempts"] == summary["unknown"] == 1
