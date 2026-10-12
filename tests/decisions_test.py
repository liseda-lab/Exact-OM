"""Offline Decisions transport recovery with the real shared spending authority."""

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from exact.llm import prompt_budget as budget
from exact.llm import spending_admission as admission
from exact.llm.decisions import DecisionsClient, predicate_identity
from exact.llm.ledger import request_identity
from tests.hosted_spending_admission_test import install
from tests.llm_prompt_budget_test import Tokenizer, policy


@pytest.fixture
def guarded(tmp_path, monkeypatch):
    for name in (
        "EXACT_EXPERIMENT_MODE",
        "EXACT_OPENROUTER_REQUEST_CAP",
        "EXACT_OPENROUTER_TOKEN_CAP",
        "EXACT_HOSTED_CACHE_ONLY",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    install(tmp_path, monkeypatch)
    policy(tmp_path, monkeypatch)
    tokenizer = Tokenizer()
    monkeypatch.setattr(budget, "_load_tokenizer", lambda *_: tokenizer)
    state = {
        "calls": [],
        "reply": {
            "model": "gpt-6-luna",
            "answers": [{"type": "predicate", "name": "equivalent", "probability": 0.75}],
            "usage": {"input_tokens": 7, "output_tokens": 0, "total_tokens": 7},
        },
    }

    def send(request):
        state["calls"].append(request)
        if state.get("failure"):
            raise state["failure"]
        if "raw" in state:
            return httpx.Response(state.get("status", 200), content=state["raw"])
        return httpx.Response(state.get("status", 200), json=state["reply"])

    client = DecisionsClient(
        tmp_path / "ledger",
        tokenizer="pinned/surrogate",
        tokenizer_revision="a" * 40,
        api_key="fake-never-persisted",
        transport=httpx.MockTransport(send),
    )
    yield client, state, tokenizer
    client.close()


def predict(client, text="Source and target facts"):
    return client.predict(text, "Are these concepts equivalent?")


def attempts(client):
    with sqlite3.connect(client.ledger.path) as db:
        return db.execute("SELECT state,raw,usage FROM attempts ORDER BY number").fetchall()


def test_predicate_replay_counts_reported_usage_once_and_preserves_raw(guarded, monkeypatch):
    client, state, tokenizer = guarded
    first = predict(client)
    client._api_key = None
    monkeypatch.setenv("EXACT_HOSTED_CACHE_ONLY", "1")
    second = predict(client)
    assert first["probability"] == second["probability"] == 0.75
    assert first["request_id"] == second["request_id"]
    assert first["cached"] is False and second["cached"] is True
    assert len(state["calls"]) == 1
    wire = json.loads(state["calls"][0].content)
    assert str(state["calls"][0].url) == budget.DECISIONS_ENDPOINT
    assert set(wire) == {"input", "model", "questions"}
    assert "Are these concepts equivalent?" in tokenizer.texts[0]
    row = attempts(client)[0]
    assert json.loads(row[1]) == state["reply"]
    usage = json.loads(row[2])
    assert usage == {
        "prompt_tokens": 7,
        "completion_tokens": 0,
        "provider_usage": state["reply"]["usage"],
    }
    snapshot = admission.admission_snapshot()
    assert snapshot["accounted_tokens"] == 107
    assert snapshot["reserved_tokens"] == 0
    assert snapshot["reported_cost_usd"] == 1.25  # Historical only; never estimate billed cost.
    assert b"fake-never-persisted" not in client.ledger.path.read_bytes()
    with pytest.raises(RuntimeError, match="cached-only"):
        predict(client, "Uncached facts")
    assert len(attempts(client)) == 1


def test_refusal_remains_unresolved_and_is_charged(guarded):
    client, state, _ = guarded
    state["reply"]["answers"] = [{"type": "refusal", "name": "equivalent"}]
    result = predict(client)
    assert result["refused"] is True and result["probability"] is None
    assert result["refusal"] == "model_refusal"
    assert admission.admission_snapshot()["accounted_tokens"] == 107


@pytest.mark.parametrize(
    "change",
    [
        {"probability": -0.1},
        {"probability": 1.1},
        {"probability": None},
        {"probability": True},
        {"probability": "0.9"},
        {"type": "refusal"},
        {"name": "another_question"},
        {"type": "choice"},
    ],
)
def test_invalid_answer_is_retained_and_never_silently_rebilled(guarded, change):
    client, state, _ = guarded
    state["reply"]["answers"][0].update(change)
    for _ in range(2):
        with pytest.raises(ValueError, match="Decisions"):
            predict(client)
    assert len(state["calls"]) == 1
    assert admission.admission_snapshot()["accounted_tokens"] == 107
    assert attempts(client)[0][0] == "completed"


def test_wrong_model_usage_is_accounted_before_rejection(guarded):
    client, state, _ = guarded
    state["reply"]["model"] = "some-other-model"
    with pytest.raises(ValueError, match="model identity"):
        predict(client)
    assert admission.admission_snapshot()["accounted_tokens"] == 107


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_probabilities_fail_after_retaining_raw_response_and_usage(guarded, value):
    client, state, _ = guarded
    state["reply"]["answers"][0]["probability"] = value
    state["raw"] = json.dumps(state["reply"]).encode()
    with pytest.raises(ValueError, match="finite"):
        predict(client)
    assert admission.admission_snapshot()["accounted_tokens"] == 107
    assert attempts(client)[0][1] == state["raw"]


def test_malformed_json_is_retained_and_never_resent(guarded):
    client, state, _ = guarded
    state["raw"] = b"invalid response"
    for _ in range(2):
        with pytest.raises(ValueError):
            predict(client)
    assert len(state["calls"]) == 1
    assert attempts(client)[0][1] == state["raw"]
    assert admission.admission_snapshot()["reserved_tokens"] > 0


def test_http_failure_is_retained_without_automatic_retry(guarded):
    client, state, _ = guarded
    state["status"] = 503
    with pytest.raises(RuntimeError, match="HTTP 503"):
        predict(client)
    assert len(state["calls"]) == 1 and attempts(client)[0][0] == "rejected"


@pytest.mark.parametrize(
    "usage",
    [
        None,
        {"input_tokens": 7},
        {"input_tokens": True, "output_tokens": 0},
        {"input_tokens": 7, "output_tokens": 0, "total_tokens": 100},
    ],
)
def test_missing_or_inconsistent_usage_keeps_conservative_reservation(guarded, usage):
    client, state, _ = guarded
    state["reply"]["usage"] = usage
    result = predict(client)
    assert result["usage"]["provider_usage"] == usage
    snapshot = admission.admission_snapshot()
    assert snapshot["reserved_tokens"] > 7
    assert snapshot["accounted_tokens"] == 100 + snapshot["reserved_tokens"]


def test_admission_counts_question_and_rejects_before_any_paid_attempt(
    guarded, tmp_path, monkeypatch
):
    client, state, _ = guarded
    policy(tmp_path, monkeypatch, max_input_tokens=300)
    with pytest.raises(budget.PromptBudgetError, match="exceed"):
        client.predict("short input", "x" * 400)
    assert not attempts(client) and not state["calls"]
    policy(tmp_path, monkeypatch, max_input_bytes=100)
    with pytest.raises(budget.PromptBudgetError, match="bytes"):
        predict(client)
    assert not attempts(client) and admission.admission_snapshot()["accounted_tokens"] == 100


def test_unknown_delivery_requires_explicit_one_shot_retry(guarded):
    client, state, _ = guarded
    state["failure"] = httpx.ReadTimeout("fixture")
    with pytest.raises(RuntimeError, match="unknown delivery"):
        predict(client)
    with pytest.raises(RuntimeError, match="explicit retry authorization"):
        predict(client)
    assert len(state["calls"]) == 1
    identity = predicate_identity(
        "Source and target facts",
        "Are these concepts equivalent?",
        tokenizer="pinned/surrogate",
        tokenizer_revision="a" * 40,
    )
    key = request_identity(identity)[0]
    client.ledger.authorize_unknown_once(key, attempt=1, authorization_id="fixture-approval")
    state.pop("failure")
    assert predict(client)["probability"] == 0.75
    assert len(state["calls"]) == 2
    assert admission.admission_snapshot()["reserved_tokens"] > 0


def test_replay_repairs_interrupted_usage_settlement_without_sending_again(guarded, monkeypatch):
    client, state, _ = guarded
    settle = client.ledger.usage
    monkeypatch.setattr(client.ledger, "usage", lambda *args: (_ for _ in ()).throw(OSError()))
    with pytest.raises(OSError):
        predict(client)
    assert admission.admission_snapshot()["reserved_tokens"] > 0
    monkeypatch.setattr(client.ledger, "usage", settle)
    assert predict(client)["cached"] is True
    assert len(state["calls"]) == 1
    assert admission.admission_snapshot()["accounted_tokens"] == 107


def test_duplicate_workers_share_the_same_request_and_charge(guarded):
    client, state, _ = guarded
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: predict(client), range(2)))
    assert len(state["calls"]) == 1
    assert sorted(row["cached"] for row in results) == [False, True]
    assert admission.admission_snapshot()["accounted_tokens"] == 107


def test_zero_output_is_only_admitted_for_the_supported_decisions_contract(guarded):
    client, _, _ = guarded
    for endpoint, payload in [
        ("https://example.invalid/decisions", {"model": "gpt-6-luna", "max_tokens": 0}),
        (budget.DECISIONS_ENDPOINT, {"model": "gpt-6-luna", "input": "a", "questions": []}),
    ]:
        key = client.ledger.plan({"endpoint": endpoint, "payload": payload})
        with pytest.raises((ValueError, budget.PromptBudgetError)):
            client.ledger.sent(key)
    with pytest.raises(budget.PromptBudgetError, match="positive integer max_tokens"):
        budget.validate_prompt_budget(
            client.profile,
            {
                "model": "gpt-6-luna",
                "messages": [{"role": "user", "content": "a"}],
                "max_tokens": 0,
            },
            "chat/completions",
        )
    assert not attempts(client)


def test_hard_spending_cap_pauses_new_calls_but_keeps_cache_usable(guarded, tmp_path, monkeypatch):
    client, state, _ = guarded
    from tests.hosted_spending_admission_test import select

    selected = admission.selected_policy()["policy"]
    select(tmp_path, monkeypatch, {**selected, "experiment_tokens_cap": 600})
    predict(client)
    with pytest.raises(admission.HostedSpendPause):
        predict(client, "x" * 1000)
    assert predict(client)["cached"] is True
    assert len(state["calls"]) == len(attempts(client)) == 1


def test_missing_required_binding_cannot_send(guarded, monkeypatch):
    client, state, _ = guarded
    monkeypatch.delenv("EXACT_HOSTED_SPENDING_POLICY_PATH")
    monkeypatch.delenv("EXACT_HOSTED_SPENDING_POLICY_SHA256")
    with pytest.raises(admission.HostedSpendingError, match="bound v2"):
        predict(client)
    assert not state["calls"] and not attempts(client)


@pytest.mark.parametrize("key", [None, "", "  \n", " sk-or-v1-fixture "])
def test_missing_or_wrong_provider_credential_is_never_sent(guarded, key):
    client, state, _ = guarded
    client._api_key = key
    with pytest.raises(RuntimeError, match="Direct Decisions API"):
        predict(client)
    assert not state["calls"] and not attempts(client)
