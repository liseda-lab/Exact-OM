"""Qualification cannot turn an accidental hosted path into scientific ledger state."""

import pytest

from exact.llm import routing
from tools.benchmark_scoring_throughput import _forbid_hosted_generation


@pytest.mark.parametrize("operation", ["chat_completion", "completion"])
@pytest.mark.parametrize("ledger_source", ["environment", "client"])
def test_qualification_denies_hosted_generation_before_key_or_ledger_access(
    tmp_path, monkeypatch, operation, ledger_source
):
    # Record the original method so the worker-only guard is undone after this test.
    monkeypatch.setattr(routing.OpenRouterClient, "_generation", routing.OpenRouterClient._generation)
    _forbid_hosted_generation()
    ledger = tmp_path / "scientific-ledger"
    key_path = tmp_path / "api_key"
    key_path.write_text("fixture-key-that-must-not-be-read")
    monkeypatch.setenv("EXACT_OPENROUTER_LEDGER_DIR", str(ledger))
    monkeypatch.delenv("EXACT_HOSTED_CACHE_ONLY", raising=False)
    client = routing.OpenRouterClient()
    if ledger_source == "client":
        client.ledger_dir = ledger
    profile = routing.LLMProfile(
        name="fixture", backend="openrouter", model="fixture/model", api_key_path=str(key_path)
    )

    def forbidden(*_args, **_kwargs):
        pytest.fail("Qualification touched a key, ledger or transport before refusing generation")

    monkeypatch.setattr(routing, "RequestLedger", forbidden)
    monkeypatch.setattr(client, "resolve_api_key", forbidden)
    monkeypatch.setattr(client._client, "request", forbidden)
    payload = ({"messages": [{"role": "user", "content": "fixture"}]}
               if operation == "chat_completion" else {"prompt": "fixture"})
    try:
        with pytest.raises(RuntimeError, match="before ledger access"):
            getattr(client, operation)(profile, max_tokens=1, **payload)
    finally:
        client.close()
    assert not ledger.exists()
