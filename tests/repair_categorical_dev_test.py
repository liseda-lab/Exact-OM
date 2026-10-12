import copy
from dataclasses import replace
import json
from types import SimpleNamespace

import httpx
import pytest

from exact.llm.routing import OpenRouterClient
from exact.repair.api import write_artifact
from exact.repair.records import canonical_hash
from tools.repair import categorical_dev as dev
from tools.repair.batch import read
from tools.repair.historical_regression import binding
from tools.repair.shared_release import immutable
from tests.repair_semantic_fidelity_test import packet
from tests.repair_judge_pilot_test import answer, plan as pilot_plan


@pytest.fixture
def plan(pilot_plan, tmp_path):
    p = packet("development")
    path = tmp_path / "dev-packet.json"
    immutable(path, p.to_dict())
    state = read(tmp_path / "ledger/phase-reservations.json")
    state["request_limits"] = dict(calibration=91, train=251, development=205, test=157)
    write_artifact(tmp_path / "ledger/phase-reservations.json", state)
    immutable(tmp_path / "phase-before.json", state)
    immutable(tmp_path / "quota.json", dict(new_limits=state["request_limits"]))
    return dict(
        pilot_plan,
        schema=dev.VERSION,
        semantic_role=dev.ROLE,
        model=pilot_plan["models"]["luna"],
        previous_phase_state=binding(tmp_path / "phase-before.json"),
        quota_amendment=binding(tmp_path / "quota.json"),
        slots=[
            dict(id="dev-0", selection_key="pair", swapped=False, packet=binding(path)),
            dict(id="dev-1", selection_key="pair", swapped=True, packet=binding(path)),
        ],
    )


def test_categories_abstention_and_grounding():
    p = packet("development")
    good = dev.assess(answer(), p)
    assert good["eligible"] and good["scores"] == [1, 0.5]
    assert not dev.assess(answer("B"), p)["eligible"]
    a = answer("abstain")
    for v in a["criteria"].values():
        v.update(a="unknown", b="unknown")
    result = dev.assess(a, p)
    assert result["unknown"] and not result["eligible"] and result["scores"] == [None, None]
    a = answer()
    a["criteria"]["meaning_retention"]["reason"] = " "
    assert not dev.assess(a, p)["eligible"]
    a = answer()
    a["criteria"]["meaning_retention"]["evidence_ids"] = ["invented"]
    with pytest.raises(ValueError):
        dev.assess(a, p)
    a = answer()
    del a["criteria"]["assertion_fidelity"]
    with pytest.raises(ValueError):
        dev.assess(a, p)


def test_identical_theories_and_native_retention_guard():
    p = packet("development")
    same = replace(p, plan_b=replace(p.plan_b, content=p.plan_a.content))
    assert not dev.assess(answer(), same)["eligible"]
    evidence = dict(
        p.evidence,
        intended=dict(
            kind="native_intended_entailed",
            text=json.dumps(dict(plan_id=p.plan_a.plan_id)),
            symbolic_value="false",
            source_id="proof",
            release="native-v1",
        ),
    )
    result = dev.assess(answer(), replace(p, evidence=evidence))
    assert not result["eligible"] and result["errors"] == [
        "perfect_retention_with_missing_commitment:a"
    ]


def record(slot, p, a):
    return dict(
        slot=slot["id"],
        selection_key=slot["selection_key"],
        swapped=slot["swapped"],
        request_id=slot["id"],
        attempt=1,
        response=dict(sha256=slot["id"]),
        assessment=dev.assess(a, p, slot["swapped"]),
    )


def test_swap_quorum_and_median_use_original_selection_rule(plan):
    p = dev.packet_for(plan["slots"][0])
    a = answer()
    b = answer("B")
    for row in b["criteria"].values():
        row.update(a="mixed", b="full")
    rows = [record(plan["slots"][0], p, a), record(plan["slots"][1], p, b)]
    result = dev.aggregate(plan["slots"], rows, p, None)
    assert result["score"] == 1 and result["quorum"] == 2 and result["votes"] == ["A", "A"]
    rows[1]["assessment"]["eligible"] = False
    assert dev.aggregate(plan["slots"], rows, p, None)["score"] is None
    rows[1] = record(plan["slots"][1], p, a)
    assert dev.aggregate(plan["slots"], rows, p, None)["score"] is None
    with pytest.raises(ValueError):
        dev.aggregate(plan["slots"], [rows[0], rows[0]], p, None)


def test_whole_cohort_budget_is_idempotent_and_preserves_history(plan):
    costs = {s["id"]: dict(reserved_cost_usd=0.01) for s in plan["slots"]}
    path = plan["ledger_directory"] + "/phase-reservations.json"
    before = read(path)
    dev.reserve(plan, costs)
    after = read(path)
    dev.reserve(plan, costs)
    assert read(path) == after and all(
        after["reservations"][k] == v for k, v in before["reservations"].items()
    )
    changed = copy.deepcopy(plan)
    changed["model"]["reasoning"] = dict(effort="high")
    with pytest.raises(ValueError):
        dev.reserve(changed, costs)


@pytest.mark.parametrize("money,count", [(34.99, 0), (0, 205)])
def test_cumulative_budget_denies_before_any_partial_reservation(plan, money, count):
    path = plan["ledger_directory"] + "/phase-reservations.json"
    state = read(path)
    for i in range(count):
        state["reservations"][str(i)] = dict(phase="development", reserved_cost_usd=0)
    state["reservations"]["extra"] = dict(phase="train", reserved_cost_usd=money)
    write_artifact(path, state)
    with pytest.raises(ValueError, match="budget"):
        dev.reserve(plan, {s["id"]: dict(reserved_cost_usd=0.01) for s in plan["slots"]})
    assert read(path) == state


@pytest.mark.parametrize("failure", [None, "timeout", "http400", "invalid", "wrong_model"])
def test_transport_replay_unknown_delivery_and_provenance(plan, tmp_path, monkeypatch, failure):
    calls = []

    def handle(request):
        calls.append(json.loads(request.content))
        if failure == "timeout":
            raise httpx.ReadTimeout("timeout", request=request)
        if failure == "http400":
            return httpx.Response(400, json=dict(error="bad request"))
        text = "{}" if failure == "invalid" else json.dumps(answer())
        return httpx.Response(
            200,
            json=dict(
                model="wrong" if failure == "wrong_model" else "model",
                provider="Provider",
                choices=[dict(finish_reason="stop", message=dict(content=text))],
                usage=dict(cost=0.01),
            ),
        )

    def client():
        c = OpenRouterClient()
        c._client.close()
        c._client = httpx.Client(transport=httpx.MockTransport(handle))
        return c

    def bounded(fn, *args, **kwargs):
        assert kwargs["timeout"] <= 180 and kwargs["cleanup_seconds"] == 2
        try:
            return SimpleNamespace(status="complete", cleanup_complete=True, value=fn(*args))
        except Exception as e:
            return SimpleNamespace(status="error", cleanup_complete=True, value=None)

    monkeypatch.setattr(dev, "OpenRouterClient", client)
    monkeypatch.setattr("exact.repair.workers.bounded_call", bounded)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-not-a-key")
    dev.reserve(plan, {s["id"]: dict(reserved_cost_usd=0.01) for s in plan["slots"]})
    slot = plan["slots"][0]
    output = tmp_path / "outputs"
    first = dev.one(plan, slot, output)
    second = dev.one(plan, slot, output)
    assert first == second and len(calls) == 1
    assert first.get("assessment", {}).get("eligible", False) == (failure is None)
    (output / slot["id"] / "receipt.json").unlink()
    third = dev.one(plan, slot, output)
    assert len(calls) == 1
    if failure in ("timeout", "http400"):
        assert "retransmission prohibited" in third["error"]
    if failure is None:
        assert calls[0]["messages"][0]["content"] == dev.pilot.PROMPT
        assert calls[0]["provider"]["only"] == ["route"]
        tampered = copy.deepcopy(third)
        tampered["assessment"]["scores"][0] = 0.1
        with pytest.raises(ValueError):
            dev.verify_receipt(plan, slot, tampered)
        wrong = copy.deepcopy(slot)
        wrong["swapped"] = True
        with pytest.raises(ValueError):
            dev.verify_receipt(plan, wrong, third)


def test_cache_write_price_is_included(plan, tmp_path):
    model = plan["model"]
    endpoint = read(model["endpoint"]["path"])
    endpoint["data"]["endpoints"][0]["pricing"]["input_cache_write"] = "0.000004"
    path = tmp_path / "cache-prices.json"
    immutable(path, endpoint)
    model["endpoint"] = binding(path)
    cost = dev.request_cost(plan, plan["slots"][0])
    assert cost["reserved_cost_usd"] == pytest.approx(
        cost["input_token_upper_bound"] * (cost["input_usd_per_token"] + 0.000004)
        + cost["reserved_output_tokens"] * cost["output_usd_per_token"]
    )
