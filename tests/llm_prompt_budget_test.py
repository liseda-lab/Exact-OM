"""Prompt admission is local, final-payload based, and independent of paid retries."""

import hashlib
import json
import sqlite3
import sys
from types import SimpleNamespace

import httpx
import pytest

from exact.llm import prompt_budget as budget
from exact.llm.ledger import RequestLedger
from exact.llm.routing import LLMProfile, OpenRouterClient


class Tokenizer:
    chat_template = "fixture"

    def __init__(self):
        self.texts = []

    def encode(self, text, *, add_special_tokens):
        self.texts.append(text)
        return list(range(len(text)))

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False and add_generation_prompt is True
        return "".join(message["content"] for message in messages)


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in (
        budget.POLICY_PATH_ENV,
        budget.POLICY_SHA256_ENV,
        "EXACT_EXPERIMENT_MODE",
        "EXACT_OPENROUTER_LEDGER_DIR",
        "EXACT_OPENROUTER_RETRY_UNKNOWN",
        "EXACT_OPENROUTER_REQUEST_CAP",
        "EXACT_OPENROUTER_TOKEN_CAP",
        "EXACT_HOSTED_SPENDING_POLICY_PATH",
        "EXACT_HOSTED_SPENDING_POLICY_SHA256",
    ):
        monkeypatch.delenv(name, raising=False)


def policy(tmp_path, monkeypatch, **overrides):
    value = {
        "schema_version": 1,
        "max_input_tokens": 8192,
        "max_input_bytes": 32768,
        "max_output_tokens": 1024,
        "chat_overhead_tokens": 128,
        **overrides,
    }
    path = tmp_path / "prompt-policy.json"
    raw = json.dumps(value).encode()
    path.write_bytes(raw)
    binding = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}
    monkeypatch.setenv(budget.POLICY_PATH_ENV, binding["path"])
    monkeypatch.setenv(budget.POLICY_SHA256_ENV, binding["sha256"])
    return binding


@pytest.fixture
def profile():
    return LLMProfile(
        name="fixture",
        backend="openrouter",
        model="hosted/model",
        tokenizer="different/pinned-tokenizer",
        tokenizer_revision="a" * 40,
    )


@pytest.fixture
def tokenizer(monkeypatch):
    value = Tokenizer()
    monkeypatch.setattr(budget, "_load_tokenizer", lambda name, revision: value)
    return value


def payload(profile, endpoint, text="abc", max_tokens=1):
    body = {"model": profile.model, "max_tokens": max_tokens}
    body.update(
        {"messages": [{"role": "user", "content": text}]}
        if endpoint == "chat/completions"
        else {"prompt": text}
    )
    return body


def invoke(client, profile, endpoint, text="abc", max_tokens=1, role="decision"):
    if endpoint == "chat/completions":
        return client.chat_completion(
            profile, [{"role": "user", "content": text}], max_tokens=max_tokens, role=role
        )
    return client.completion(profile, text, max_tokens=max_tokens, role=role)


def reply(profile, status=200):
    return httpx.Response(
        status,
        json={"model": profile.model, "choices": []},
        request=httpx.Request("POST", "https://example.invalid/completions"),
    )


def setup_client(tmp_path, monkeypatch, *, ledger):
    client = OpenRouterClient()
    if ledger:
        client.ledger_dir = tmp_path / "ledger"
    monkeypatch.setattr(client, "resolve_api_key", lambda profile: "fake-key")
    monkeypatch.setattr(client, "_sleep_before_retry", lambda attempt: None)
    return client


def test_policy_is_optional_only_outside_experiments(monkeypatch):
    assert budget.load_prompt_budget_policy() is None
    monkeypatch.setenv("EXACT_EXPERIMENT_MODE", "1")
    with pytest.raises(budget.PromptBudgetError, match="path and SHA256"):
        budget.load_prompt_budget_policy()


@pytest.mark.parametrize(
    "overrides",
    [
        {"schema_version": True},
        {"schema_version": 2},
        {"max_input_tokens": 12289},
        {"max_input_tokens": 0},
        {"max_output_tokens": True},
        {"max_output_tokens": 1025},
        {"max_input_bytes": 32769},
        {"max_input_bytes": None},
        {"chat_overhead_tokens": 127},
        {"counting_method": "optimistic"},
    ],
)
def test_policy_limits_fail_closed(tmp_path, monkeypatch, overrides):
    policy(tmp_path, monkeypatch, **overrides)
    with pytest.raises(budget.PromptBudgetError):
        budget.load_prompt_budget_policy()


def test_policy_hash_and_missing_bytes_fail_closed(tmp_path, monkeypatch):
    binding = policy(tmp_path, monkeypatch)
    from pathlib import Path

    path = Path(binding["path"])
    path.write_text("{}")
    with pytest.raises(budget.PromptBudgetError, match="SHA256"):
        budget.load_prompt_budget_policy()
    path.unlink()
    with pytest.raises(budget.PromptBudgetError, match="Cannot read"):
        budget.load_prompt_budget_policy()


@pytest.mark.parametrize("endpoint", ["chat/completions", "completions"])
def test_exact_token_boundary_and_appended_evidence(
    tmp_path, monkeypatch, profile, tokenizer, endpoint
):
    policy(tmp_path, monkeypatch, max_input_tokens=131)
    result = budget.validate_prompt_budget(profile, payload(profile, endpoint), endpoint)
    assert result["input_admission_tokens"] == 131
    assert result["local_tokenizer_tokens"] == 3
    assert result["provider_exact_token_count"] is False
    assert result["hosted_model"] != result["tokenizer"]
    with pytest.raises(budget.PromptBudgetError, match="132 exceeds limit 131"):
        budget.validate_prompt_budget(profile, payload(profile, endpoint, "abcX"), endpoint)


def test_utf8_byte_cap_is_independent_and_precedes_tokenizer(tmp_path, monkeypatch, profile):
    policy(tmp_path, monkeypatch, max_input_bytes=6)

    def forbidden(*args):
        pytest.fail("Rejected raw bytes must not load a tokenizer")

    monkeypatch.setattr(budget, "_load_tokenizer", forbidden)
    with pytest.raises(budget.PromptBudgetError, match="bytes 8 exceed limit 6"):
        budget.validate_prompt_budget(
            profile, payload(profile, "completions", "é" * 4), "completions"
        )


def test_completion_batches_sum_input_reserve_and_output(tmp_path, monkeypatch, profile, tokenizer):
    policy(tmp_path, monkeypatch, max_input_tokens=260, max_output_tokens=4)
    body = payload(profile, "completions", ["ab", "cd"], max_tokens=2)
    result = budget.validate_prompt_budget(profile, body, "completions")
    assert result["input_admission_tokens"] == 260
    assert result["requested_output_tokens"] == 4
    body["prompt"].append("suffix")
    with pytest.raises(budget.PromptBudgetError, match="output tokens 6"):
        budget.validate_prompt_budget(profile, body, "completions")
    body["max_tokens"] = 1
    with pytest.raises(budget.PromptBudgetError, match="Input admission"):
        budget.validate_prompt_budget(profile, body, "completions")


def test_text_without_chat_template_counts_all_roles_and_content(
    tmp_path, monkeypatch, profile, tokenizer
):
    policy(tmp_path, monkeypatch)
    tokenizer.chat_template = None
    body = payload(profile, "chat/completions", "final evidence")
    body["messages"].insert(0, {"role": "system", "content": "system instruction"})
    result = budget.validate_prompt_budget(profile, body, "chat/completions")
    assert result["rendering_method"] == "text_roles_json"
    assert all(
        text in tokenizer.texts[-1]
        for text in ("system", "user", "final evidence", "system instruction")
    )
    assert result["utf8_bytes"] == len("systemusersystem instructionfinal evidence".encode())


@pytest.mark.parametrize(
    "extra",
    [
        {"tools": [{"type": "function"}]},
        {"response_format": {"type": "json_schema"}},
        {"n": 2},
        {"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]},
        {"messages": [{"role": "tool", "content": "tool output"}]},
        {"messages": [{"role": "user", "content": "hi", "name": "unaccounted"}]},
    ],
)
def test_unsupported_prompt_fields_and_parts_fail_closed(
    tmp_path, monkeypatch, profile, tokenizer, extra
):
    policy(tmp_path, monkeypatch)
    body = {**payload(profile, "chat/completions"), **extra}
    with pytest.raises(budget.PromptBudgetError, match="Unsupported"):
        budget.validate_prompt_budget(profile, body, "chat/completions")


@pytest.mark.parametrize("revision", [None, "main", "short"])
def test_tokenizer_must_have_immutable_revision(
    tmp_path, monkeypatch, profile, tokenizer, revision
):
    policy(tmp_path, monkeypatch)
    profile.tokenizer_revision = revision
    with pytest.raises(budget.PromptBudgetError, match="pinned tokenizer"):
        budget.validate_prompt_budget(profile, payload(profile, "completions"), "completions")


def test_tokenizer_failure_is_typed_and_offline_flags_are_mandatory(tmp_path, monkeypatch, profile):
    policy(tmp_path, monkeypatch)
    calls = []

    def unavailable(name, **kwargs):
        calls.append((name, kwargs))
        raise OSError("not in offline cache")

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=unavailable)),
    )
    budget._load_tokenizer.cache_clear()
    with pytest.raises(budget.PromptBudgetError, match="offline"):
        budget.validate_prompt_budget(profile, payload(profile, "completions"), "completions")
    assert calls == [
        (
            profile.tokenizer,
            {
                "revision": profile.tokenizer_revision,
                "local_files_only": True,
                "trust_remote_code": False,
            },
        )
    ]


@pytest.mark.parametrize("ledger", [False, True])
@pytest.mark.parametrize("endpoint", ["chat/completions", "completions"])
@pytest.mark.parametrize("role", ["verbaliser", "summary", "decision", "rationale", "caller"])
def test_all_roles_reject_before_transmission_and_attempts(
    tmp_path, monkeypatch, profile, tokenizer, ledger, endpoint, role
):
    policy(tmp_path, monkeypatch, max_input_tokens=130)
    client = setup_client(tmp_path, monkeypatch, ledger=ledger)
    monkeypatch.setattr(
        client,
        "resolve_api_key",
        lambda profile: pytest.fail("No credentials needed for rejection"),
    )
    monkeypatch.setattr(
        client._client, "request", lambda **kwargs: pytest.fail("No hosted transmission allowed")
    )
    try:
        with pytest.raises(budget.PromptBudgetError, match="131 exceeds limit 130"):
            invoke(client, profile, endpoint, role=role)
        if ledger:
            with sqlite3.connect(client.ledger_dir / "requests.sqlite3") as db:
                assert db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 0
                assert db.execute("SELECT COUNT(*) FROM reservations").fetchone()[0] == 0
    finally:
        client.close()


@pytest.mark.parametrize("ledger", [False, True])
def test_strict_missing_policy_rejects_even_without_ledger(tmp_path, monkeypatch, profile, ledger):
    monkeypatch.setenv("EXACT_EXPERIMENT_MODE", "1")
    client = setup_client(tmp_path, monkeypatch, ledger=ledger)
    monkeypatch.setattr(client._client, "request", lambda **kwargs: pytest.fail("No transmission"))
    try:
        with pytest.raises(budget.PromptBudgetError, match="policy path"):
            invoke(client, profile, "chat/completions")
    finally:
        client.close()


@pytest.mark.parametrize("ledger", [False, True])
def test_final_sanitized_content_is_counted_and_transmitted(
    tmp_path, monkeypatch, profile, tokenizer, ledger
):
    policy(tmp_path, monkeypatch)
    client = setup_client(tmp_path, monkeypatch, ledger=ledger)
    sent = []
    monkeypatch.setattr(
        client._client, "request", lambda **kwargs: sent.append(kwargs) or reply(profile)
    )
    try:
        invoke(client, profile, "chat/completions", "A\x00B\ud800")
        body = json.loads(sent[0]["content"])
        assert body["messages"][0]["content"] == "A B?"
        assert tokenizer.texts and all(text == "A B?" for text in tokenizer.texts)
    finally:
        client.close()


@pytest.mark.parametrize("endpoint", ["chat/completions", "completions"])
def test_exact_oversized_cache_replay_needs_no_policy_or_tokenizer(
    tmp_path, monkeypatch, profile, endpoint
):
    client = setup_client(tmp_path, monkeypatch, ledger=True)
    calls = []
    monkeypatch.setattr(
        client._client, "request", lambda **kwargs: calls.append(kwargs) or reply(profile)
    )
    try:
        expected = invoke(client, profile, endpoint, "old oversized" * 3000)
        policy(tmp_path, monkeypatch, max_input_tokens=128)
        monkeypatch.setattr(
            client, "resolve_api_key", lambda profile: pytest.fail("Cache requires no key")
        )
        monkeypatch.setattr(
            budget, "_load_tokenizer", lambda *args: pytest.fail("Cache requires no tokenizer")
        )
        assert invoke(client, profile, endpoint, "old oversized" * 3000) == expected
        assert len(calls) == 1
        with sqlite3.connect(client.ledger_dir / "requests.sqlite3") as db:
            assert (
                budget.POLICY_SHA256_ENV
                not in db.execute("SELECT identity FROM requests").fetchone()[0]
            )
    finally:
        client.close()


def test_oversized_unknown_retry_keeps_unused_approval(tmp_path, monkeypatch, profile, tokenizer):
    client = setup_client(tmp_path, monkeypatch, ledger=True)

    def timeout(**kwargs):
        raise httpx.ReadTimeout("fixture unknown")

    monkeypatch.setattr(client._client, "request", timeout)
    try:
        with pytest.raises(RuntimeError, match="unknown delivery"):
            invoke(client, profile, "completions")
        ledger = RequestLedger(client.ledger_dir)
        with sqlite3.connect(ledger.path) as db:
            key = db.execute("SELECT request_id FROM attempts").fetchone()[0]
        ledger.authorize_unknown_once(key, attempt=1, authorization_id="fixture-approval")
        policy(tmp_path, monkeypatch, max_input_tokens=130)
        with pytest.raises(budget.PromptBudgetError):
            invoke(client, profile, "completions")
        with sqlite3.connect(ledger.path) as db:
            assert db.execute("SELECT number,state FROM attempts").fetchall() == [(1, "unknown")]
            assert db.execute("SELECT consumed_by FROM retry_authorizations").fetchall() == [
                (None,)
            ]
    finally:
        client.close()


@pytest.mark.parametrize("ledger", [False, True])
def test_policy_is_rechecked_before_retry_without_false_unknown(
    tmp_path, monkeypatch, profile, tokenizer, ledger
):
    policy(tmp_path, monkeypatch)
    client = setup_client(tmp_path, monkeypatch, ledger=ledger)
    calls = []

    def request(**kwargs):
        calls.append(kwargs)
        policy(tmp_path, monkeypatch, max_input_tokens=130)
        return reply(profile, status=429)

    monkeypatch.setattr(client._client, "request", request)
    try:
        with pytest.raises(budget.PromptBudgetError):
            invoke(client, profile, "completions")
        assert len(calls) == 1
        if ledger:
            with sqlite3.connect(client.ledger_dir / "requests.sqlite3") as db:
                assert db.execute("SELECT state FROM attempts").fetchall() == [("rejected",)]
    finally:
        client.close()


@pytest.mark.parametrize("ledger", [False, True])
def test_admitted_retry_reuses_exact_validated_bytes(
    tmp_path, monkeypatch, profile, tokenizer, ledger
):
    policy(tmp_path, monkeypatch)
    client = setup_client(tmp_path, monkeypatch, ledger=ledger)
    calls = []

    def request(**kwargs):
        calls.append(kwargs)
        return reply(profile, status=429 if len(calls) == 1 else 200)

    monkeypatch.setattr(client._client, "request", request)
    try:
        assert invoke(client, profile, "completions")["model"] == profile.model
        assert len(calls) == 2
        assert calls[0]["content"] == calls[1]["content"]
        assert len(tokenizer.texts) == 3  # initial admission plus each paid attempt
    finally:
        client.close()


def test_strict_cache_replay_survives_missing_policy(tmp_path, monkeypatch, profile, tokenizer):
    policy(tmp_path, monkeypatch)
    monkeypatch.setenv("EXACT_EXPERIMENT_MODE", "1")
    client = setup_client(tmp_path, monkeypatch, ledger=True)
    calls = []
    monkeypatch.setattr(
        client._client, "request", lambda **kwargs: calls.append(kwargs) or reply(profile)
    )
    try:
        expected = invoke(client, profile, "chat/completions")
        monkeypatch.delenv(budget.POLICY_PATH_ENV)
        monkeypatch.delenv(budget.POLICY_SHA256_ENV)
        monkeypatch.setattr(
            budget, "_load_tokenizer", lambda *args: pytest.fail("Cached reply needs no tokenizer")
        )
        assert invoke(client, profile, "chat/completions") == expected
        assert len(calls) == 1
        assert json.loads(calls[0]["content"])["provider"]["allow_fallbacks"] is False
    finally:
        client.close()
