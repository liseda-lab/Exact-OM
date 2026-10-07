"""Cached-only routing rejects new generation before credentials or paid accounting."""

import json
import sqlite3

import httpx
import pytest

from exact.llm import routing
from exact.llm.ledger import RequestLedger
from exact.llm.prompt_budget import PromptBudgetError
from exact.llm.routing import HostedCacheOnlyError, LLMProfile, OpenRouterClient


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in (
        "EXACT_HOSTED_CACHE_ONLY",
        "EXACT_EXPERIMENT_MODE",
        "EXACT_OPENROUTER_LEDGER_DIR",
        "EXACT_OPENROUTER_RETRY_UNKNOWN",
        "EXACT_OPENROUTER_REQUEST_CAP",
        "EXACT_OPENROUTER_TOKEN_CAP",
        "EXACT_HOSTED_SPENDING_POLICY_PATH",
        "EXACT_HOSTED_SPENDING_POLICY_SHA256",
        "EXACT_LLM_PROMPT_POLICY_PATH",
        "EXACT_LLM_PROMPT_POLICY_SHA256",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def client():
    value = OpenRouterClient()
    yield value
    value.close()


@pytest.fixture
def profile():
    return LLMProfile(name="fixture", backend="openrouter", model="fixture/model", revision="v1")


def invoke(client, profile, endpoint, text="saved"):
    if endpoint == "chat/completions":
        return client.chat_completion(
            profile, [{"role": "user", "content": text}], max_tokens=4, role="decision"
        )
    return client.completion(profile, text, max_tokens=4, role="decision")


def seed(ledger, profile, endpoint, *, state="completed", strict=False):
    payload = {"model": profile.model, "max_tokens": 4}
    payload.update(
        {"messages": [{"role": "user", "content": "saved"}]}
        if endpoint == "chat/completions"
        else {"prompt": "saved"}
    )
    if strict:
        payload["provider"] = {"allow_fallbacks": False}
    key = ledger.plan(
        {
            "schema_version": 1,
            "role": "decision",
            "endpoint": f"{profile.api_base}/{endpoint}",
            "payload": payload,
            "revision": profile.revision,
            "tokenizer": profile.tokenizer,
            "tokenizer_revision": profile.tokenizer_revision,
            "requested_seed": None,
        }
    )
    number = ledger.sent(key)
    reply = {"model": profile.model, "choices": [{"message": {"content": "saved answer"}}]}
    if state == "completed":
        ledger.received(key, number, json.dumps(reply).encode(), 200, elapsed_seconds=2.5)
        ledger.usage(key, number, {"prompt_tokens": 7, "completion_tokens": 1, "cost": 0.003})
    elif state == "unknown":
        ledger.unknown(key, number, "historical unknown", elapsed_seconds=2.5)
        ledger.authorize_unknown_once(key, attempt=number, authorization_id="historical-approval")
    elif state == "rejected":
        ledger.received(key, number, b'{"error":"historical rejection"}', 429)
    return key, reply


def snapshot(ledger):
    with sqlite3.connect(ledger.path) as db:
        return {
            name: db.execute(f"SELECT * FROM {name} ORDER BY rowid").fetchall()
            for name in (
                "requests",
                "attempts",
                "reservations",
                "retry_authorizations",
                "spending_policies",
            )
        }


def prohibit_paid_boundary(client, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Cached-only generation crossed a credential/accounting/wire boundary")

    monkeypatch.setattr(client, "resolve_api_key", unexpected)
    monkeypatch.setattr(client._client, "request", unexpected)
    monkeypatch.setattr(client, "_http_json", unexpected)
    monkeypatch.setattr(RequestLedger, "sent", unexpected)
    monkeypatch.setattr(RequestLedger, "plan", unexpected)
    monkeypatch.setattr(routing, "validate_prompt_budget", unexpected)
    monkeypatch.setattr(routing, "selected_policy", unexpected)


@pytest.mark.parametrize("endpoint", ["chat/completions", "completions"])
@pytest.mark.parametrize("strict", [False, True])
def test_cached_success_replays_without_new_charges_or_credentials(
    tmp_path, monkeypatch, client, profile, endpoint, strict
):
    client.ledger_dir = tmp_path / "ledger"
    ledger = RequestLedger(client.ledger_dir)
    monkeypatch.setenv("EXACT_OPENROUTER_TOKEN_CAP", "10000")
    _, reply = seed(ledger, profile, endpoint, strict=strict)
    before = snapshot(ledger)
    usage_before = ledger.summary()
    monkeypatch.setenv("EXACT_HOSTED_CACHE_ONLY", "1")
    if strict:
        monkeypatch.setenv("EXACT_EXPERIMENT_MODE", "1")
    prohibit_paid_boundary(client, monkeypatch)
    assert invoke(client, profile, endpoint) == reply
    assert snapshot(ledger) == before
    assert ledger.summary() == usage_before


@pytest.mark.parametrize("endpoint", ["chat/completions", "completions"])
@pytest.mark.parametrize("with_ledger", [False, True])
def test_cache_miss_and_no_ledger_fail_before_any_paid_boundary(
    tmp_path, monkeypatch, client, profile, endpoint, with_ledger
):
    ledger = None
    if with_ledger:
        client.ledger_dir = tmp_path / "ledger"
        ledger = RequestLedger(client.ledger_dir)
    monkeypatch.setenv("EXACT_HOSTED_CACHE_ONLY", "1")
    prohibit_paid_boundary(client, monkeypatch)
    with pytest.raises(HostedCacheOnlyError, match="[Cc]ached-only") as error:
        invoke(client, profile, endpoint, text="unseen")
    assert isinstance(error.value, PromptBudgetError)  # Strict model handlers propagate it.
    if ledger:
        assert all(not rows for rows in snapshot(ledger).values())


@pytest.mark.parametrize("state", ["sent", "unknown", "rejected"])
def test_non_successful_history_never_retries_and_keeps_original_charges(
    tmp_path, monkeypatch, client, profile, state
):
    client.ledger_dir = tmp_path / "ledger"
    ledger = RequestLedger(client.ledger_dir)
    monkeypatch.setenv("EXACT_OPENROUTER_TOKEN_CAP", "10000")
    seed(ledger, profile, "chat/completions", state=state)
    before = snapshot(ledger)
    monkeypatch.setenv("EXACT_HOSTED_CACHE_ONLY", "1")
    monkeypatch.setenv("EXACT_OPENROUTER_RETRY_UNKNOWN", "1")
    client.retry_unknown_requests = True
    prohibit_paid_boundary(client, monkeypatch)
    with pytest.raises(HostedCacheOnlyError, match="no successful response"):
        invoke(client, profile, "chat/completions")
    assert snapshot(ledger) == before


@pytest.mark.parametrize("value", ["", "true", "yes", "2", " 1", "False"])
def test_malformed_cache_only_setting_fails_closed(monkeypatch, client, profile, value):
    monkeypatch.setenv("EXACT_HOSTED_CACHE_ONLY", value)
    prohibit_paid_boundary(client, monkeypatch)
    with pytest.raises(HostedCacheOnlyError, match="must be unset"):
        invoke(client, profile, "completions")


@pytest.mark.parametrize("value", [None, "0"])
def test_disabled_cache_only_preserves_ordinary_generation(monkeypatch, client, profile, value):
    if value is not None:
        monkeypatch.setenv("EXACT_HOSTED_CACHE_ONLY", value)
    calls = []
    monkeypatch.setattr(client, "resolve_api_key", lambda _: "fixture-only")
    monkeypatch.setattr(
        client._client,
        "request",
        lambda **kwargs: calls.append(kwargs)
        or httpx.Response(
            200,
            json={"choices": []},
            request=httpx.Request("POST", "https://example.invalid/completions"),
        ),
    )
    assert invoke(client, profile, "completions") == {"choices": []}
    assert len(calls) == 1


def test_corrupted_completed_response_fails_without_generation(
    tmp_path, monkeypatch, client, profile
):
    client.ledger_dir = tmp_path / "ledger"
    ledger = RequestLedger(client.ledger_dir)
    seed(ledger, profile, "completions")
    with sqlite3.connect(ledger.path) as db:
        db.execute("UPDATE attempts SET raw=?", (b"corrupted",))
    before = snapshot(ledger)
    monkeypatch.setenv("EXACT_HOSTED_CACHE_ONLY", "1")
    prohibit_paid_boundary(client, monkeypatch)
    with pytest.raises(ValueError, match="Corrupted completed"):
        invoke(client, profile, "completions")
    assert snapshot(ledger) == before
