"""Concurrent exact requests share one paid attempt without weakening recovery guards."""

import multiprocessing
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import httpx
import pytest

from exact.llm.ledger import RequestLedger
from exact.llm.routing import LLMProfile, OpenRouterClient


def client_at(directory):
    client = OpenRouterClient()
    client.ledger_dir = directory
    client.resolve_api_key = lambda profile: "fixture-only"
    return client


def profile():
    return LLMProfile(name="shared", backend="openrouter", model="fixture/model")


def call(client, text="same"):
    return client.chat_completion(
        profile(), [{"role": "user", "content": text}], max_tokens=4, role="decision"
    )


def response():
    return httpx.Response(
        200,
        json={"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 1}},
        request=httpx.Request("POST", "https://example.invalid"),
    )


def test_duplicate_threads_replay_one_attempt_across_clients(tmp_path, monkeypatch):
    rendezvous = Barrier(4)
    original = RequestLedger.plan

    def plan(self, identity):
        key = original(self, identity)
        rendezvous.wait(timeout=5)
        return key

    monkeypatch.setattr(RequestLedger, "plan", plan)
    sends = []

    def send(**kwargs):
        sends.append(1)
        time.sleep(0.15)
        return response()

    clients = [client_at(tmp_path) for _ in range(4)]
    for client in clients:
        monkeypatch.setattr(client._client, "request", send)
    with ThreadPoolExecutor(max_workers=4) as executor:
        outputs = list(executor.map(call, clients))
    assert outputs == [outputs[0]] * 4
    assert len(sends) == 1
    totals = RequestLedger(tmp_path).summary()["roles"]["decision"]
    assert totals["attempts"] == totals["completed"] == 1
    assert totals["prompt_tokens"] == 3


def test_distinct_requests_keep_transport_concurrency(tmp_path, monkeypatch):
    rendezvous = Barrier(2)
    client = client_at(tmp_path)

    def send(**kwargs):
        rendezvous.wait(timeout=5)
        return response()

    monkeypatch.setattr(client._client, "request", send)
    with ThreadPoolExecutor(max_workers=2) as executor:
        assert len(list(executor.map(lambda text: call(client, text), ["a", "b"]))) == 2
    assert RequestLedger(tmp_path).summary()["roles"]["decision"]["attempts"] == 2


def test_concurrent_unknown_delivery_never_resends(tmp_path, monkeypatch):
    rendezvous = Barrier(2)
    original = RequestLedger.plan

    def plan(self, identity):
        key = original(self, identity)
        rendezvous.wait(timeout=5)
        return key

    monkeypatch.setattr(RequestLedger, "plan", plan)
    sends = []

    def send(**kwargs):
        sends.append(1)
        time.sleep(0.1)
        raise httpx.ReadTimeout("ambiguous fixture delivery")

    client = client_at(tmp_path)
    monkeypatch.setattr(client._client, "request", send)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(call, client) for _ in range(2)]
        errors = []
        for future in futures:
            with pytest.raises(RuntimeError) as error:
                future.result()
            errors.append(str(error.value))
    assert len(sends) == 1
    assert any("explicit retry authorization" in error for error in errors)
    totals = RequestLedger(tmp_path).summary()["roles"]["decision"]
    assert totals["attempts"] == totals["unknown"] == 1


def _process_call(directory, ready, sends, outcomes):
    client = client_at(directory)
    original = RequestLedger.plan

    def plan(self, identity):
        key = original(self, identity)
        ready.wait(timeout=10)
        return key

    RequestLedger.plan = plan

    def send(**kwargs):
        with sends.get_lock():
            sends.value += 1
        time.sleep(0.15)
        return response()

    client._client.request = send
    try:
        outcomes.put(("ok", call(client)))
    except Exception as error:
        outcomes.put(("error", str(error)))


def test_duplicate_processes_share_ledger_request_lock(tmp_path):
    context = multiprocessing.get_context("fork")
    ready, sends, outcomes = context.Barrier(2), context.Value("i", 0), context.Queue()
    children = [
        context.Process(target=_process_call, args=(tmp_path, ready, sends, outcomes))
        for _ in range(2)
    ]
    for child in children:
        child.start()
    for child in children:
        child.join(timeout=15)
        assert child.exitcode == 0
    results = [outcomes.get(timeout=2) for _ in children]
    assert all(status == "ok" for status, _ in results), results
    assert results[0][1] == results[1][1]
    assert sends.value == 1
