"""Campaign accounting retains request-level exposure even without runtime forecasts."""

import json
from types import SimpleNamespace

import pytest

from exact.experiments import campaign, harness
from exact.llm.ledger import RequestLedger


@pytest.mark.parametrize("response", ["unknown", "http_error", "missing_usage", "partial_usage", "priced"])
def test_comparison_preserves_wire_reservations(tmp_path, monkeypatch, response):
    monkeypatch.setenv("EXACT_OPENROUTER_REQUEST_CAP", "20")
    monkeypatch.setenv("EXACT_OPENROUTER_TOKEN_CAP", "10000")
    limits = {"envelopes_hours": {"llm": 0}, "node_hours_cap": 0,
              "requests_cap": 20, "tokens_cap": 10000}
    suite = SimpleNamespace(campaign={"root": str(tmp_path), "budget_limits": limits, "stage": "screen"})
    source = SimpleNamespace(config=SimpleNamespace(
        experiment_id="E25", frozen_constants={"campaign_v2": {"budget_group": "llm", "estimate": None}}
    ))
    monkeypatch.setattr(campaign, "validate_comparison_cells", lambda *a: None)
    reservation = []

    def execute(*args, **kwargs):
        ledger = RequestLedger(tmp_path / "openrouter")
        key = ledger.plan({"role": "decision", "payload": {"max_tokens": 8, "text": "A"}})
        number = ledger.sent(key)
        with ledger._transaction() as db:
            reservation.append(db.execute("SELECT tokens FROM reservations").fetchone()[0])
        if response == "unknown":
            ledger.unknown(key, number, "timeout")
            raise RuntimeError("worker request failed")
        ledger.received(key, number, b"{}", 503 if response == "http_error" else 200)
        if response == "http_error":
            raise RuntimeError("worker request failed")
        if response == "partial_usage":
            ledger.usage(key, number, {"prompt_tokens": 10})
        elif response == "priced":
            ledger.usage(key, number, {"prompt_tokens": 10, "completion_tokens": 2, "cost": 0.004})
        return [{"status": "complete"}]

    monkeypatch.setattr(harness, "run_cells", execute)
    if response in {"unknown", "http_error"}:
        with pytest.raises(RuntimeError, match="worker request failed"):
            campaign.run_comparison([], suite, source)
    else:
        campaign.run_comparison([], suite, source)
    work, = json.loads((tmp_path / "budget.json").read_text())["work"].values()
    assert work["requests"] == 1
    assert work["tokens"] == (12 if response == "priced" else reservation[0])
    assert work["actual_usd"] == (0.004 if response == "priced" else None)
    assert work["forecast_known"] is False
    assert work["status"] == ("failed" if response in {"unknown", "http_error"} else "complete")
