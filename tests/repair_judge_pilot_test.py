import copy
import json

import httpx
import pytest

from exact.llm.routing import OpenRouterClient
from tools.repair import judge_pilot as pilot
from tools.repair.batch import read
from tools.repair.historical_regression import binding
from tools.repair.shared_release import immutable


def answer(preference="A"):
    return dict(
        preference=preference,
        criteria={
            k: dict(a="full", b="mixed", reason="D1 supports A more fully.", evidence_ids=["D1"])
            for k in pilot.CRITERIA
        },
    )


def gold():
    return dict(
        preference="A", allowed_anchors={k: dict(a=["full"], b=["mixed"]) for k in pilot.CRITERIA}
    )


def test_categorical_scores_and_preference_are_separate():
    result = pilot.assess(answer(), pilot.response_schema(["D1"]), gold())
    assert result["scores"] == [1, 0.5] and result["automated_pass"]
    wrong = pilot.assess(answer("B"), pilot.response_schema(["D1"]), gold())
    assert not wrong["preference_consistent"] and not wrong["automated_pass"]
    assert result["training_eligible"] is False and result["test_selection_eligible"] is False


def test_swap_tie_abstain_and_invalid_citation():
    a = answer("B")
    for row in a["criteria"].values():
        row.update(a="mixed", b="full")
    assert pilot.assess(a, pilot.response_schema(["D1"]), gold(), swapped=True)["automated_pass"]
    for category, preference in [("unknown", "abstain"), ("full", "tie")]:
        a = answer(preference)
        for row in a["criteria"].values():
            row.update(a=category, b=category)
        g = dict(
            preference=preference,
            allowed_anchors={k: dict(a=[category], b=[category]) for k in pilot.CRITERIA},
        )
        assert pilot.assess(a, pilot.response_schema(["D1"]), g)["automated_pass"]
    a["criteria"]["meaning_retention"]["evidence_ids"] = ["invented"]
    with pytest.raises(Exception):
        pilot.assess(a, pilot.response_schema(["D1"]), g)


@pytest.fixture
def plan(tmp_path):
    packet = tmp_path / "packet.json"
    immutable(
        packet,
        dict(
            namespace="isolated_judge_qualification_20261012",
            study_split=None,
            native_verified=True,
            payload=dict(
                evidence=dict(D1="frozen definition"),
                plan_a=dict(plan_id="a"),
                plan_b=dict(plan_id="b"),
            ),
            gold=gold(),
        ),
    )
    ledger = tmp_path / "ledger"
    ledger.mkdir()
    state = dict(
        schema="exact-repair/annotation-phase-budget/v1",
        lineage="campaign",
        cost_ceiling_usd=35,
        request_limits=dict(calibration=91, train=251, development=170, test=192),
        reservations={"old": dict(phase="train", state="unresolved", reserved_cost_usd=14.0)},
    )
    immutable(ledger / "phase-reservations.json", state)
    immutable(tmp_path / "prior.json", state)
    models = {}
    for name in ("luna", "haiku", "minimax", "gemini"):
        endpoint = tmp_path / (name + ".json")
        immutable(
            endpoint,
            dict(
                data=dict(
                    endpoints=[
                        dict(
                            tag="route",
                            model_id="model",
                            provider_name="Provider",
                            context_length=1000000,
                            max_completion_tokens=128000,
                            supported_parameters=[
                                "reasoning",
                                "max_tokens",
                                "response_format",
                                "structured_outputs",
                            ],
                            pricing=dict(
                                prompt="0.000001",
                                completion="0.000002",
                                overrides=[
                                    dict(
                                        min_prompt_tokens=100,
                                        prompt="0.000002",
                                        completion="0.000004",
                                    )
                                ],
                            ),
                        )
                    ]
                )
            ),
        )
        models[name] = dict(
            route="route",
            provider_name="Provider",
            endpoint=binding(endpoint),
            reasoning=dict(effort="low"),
            accepted_model_ids=["model"],
            profile=dict(
                model="model",
                backend="openrouter",
                timeout_secs=180,
                provider=dict(only=["route"], allow_fallbacks=False, require_parameters=True),
            ),
        )
    return dict(
        schema=pilot.VERSION,
        prompt=pilot.PROMPT,
        budget_phase="development",
        semantic_role="isolated_qualification",
        max_requests=24,
        max_output_tokens=6000,
        max_input_bytes=400000,
        timeout_seconds=180,
        deadline_epoch=9999999999,
        authorization=dict(user_text="test these four models"),
        models=models,
        teacher_candidates=["luna", "haiku", "minimax"],
        independent_assessor="gemini",
        ledger_directory=str(ledger),
        previous_phase_state=binding(tmp_path / "prior.json"),
        slots=[
            dict(
                id=m + "-" + str(i),
                case="c" + str(i % 5),
                swapped=i == 5,
                packet=binding(packet),
                model=m,
            )
            for m in models
            for i in range(6)
        ],
    )


def test_full_panel_reservation_resume_and_changed_dependencies(plan):
    costs = pilot.validate(plan)
    assert all(c["input_usd_per_token"] == 0.000002 for c in costs.values())
    pilot.reserve(plan, costs)
    path = plan["ledger_directory"] + "/phase-reservations.json"
    original = read(path)
    assert len(original["reservations"]) == 25
    pilot.reserve(plan, costs)
    assert read(path) == original
    changed = copy.deepcopy(plan)
    changed["models"]["luna"]["reasoning"] = dict(effort="medium")
    with pytest.raises(ValueError, match="dependencies"):
        pilot.reserve(changed, costs)
    assert read(path) == original


@pytest.mark.parametrize("money,count", [(34.99, 0), (0, 169)])
def test_whole_panel_admission_cannot_overspend_or_take_test_slots(plan, money, count):
    path = plan["ledger_directory"] + "/phase-reservations.json"
    state = read(path)
    for i in range(count):
        state["reservations"][str(i)] = dict(phase="development", reserved_cost_usd=0)
    state["reservations"]["extra"] = dict(phase="train", reserved_cost_usd=money)
    from exact.repair.api import write_artifact

    write_artifact(path, state)
    with pytest.raises(ValueError, match="budget"):
        pilot.reserve(plan, pilot.validate(plan))
    assert read(path) == state


def test_paid_transport_completed_replay_and_crash_recovery(plan, tmp_path, monkeypatch):
    calls = []

    def handle(request):
        calls.append(json.loads(request.content))
        return httpx.Response(
            200,
            json=dict(
                model="model",
                provider="Provider",
                choices=[dict(finish_reason="stop", message=dict(content=json.dumps(answer())))],
                usage=dict(cost=0.01),
            ),
        )

    def client():
        c = OpenRouterClient()
        c._client.close()
        c._client = httpx.Client(transport=httpx.MockTransport(handle))
        return c

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-not-a-key")
    monkeypatch.setattr(pilot, "OpenRouterClient", client)
    pilot.reserve(plan, pilot.validate(plan))
    slot = plan["slots"][0]
    result = pilot.one(plan, slot, tmp_path / "outputs")
    assert result["assessment"]["automated_pass"] and len(calls) == 1
    assert "temperature" not in calls[0]
    assert calls[0]["provider"]["only"] == ["route"]
    (tmp_path / "outputs" / slot["id"] / "receipt.json").unlink()
    replay = pilot.one(plan, slot, tmp_path / "outputs")
    assert replay["answer"] == result["answer"] and len(calls) == 1


@pytest.mark.parametrize("failure", ["timeout", "http400"])
def test_crashed_failed_transmission_is_not_repeated(plan, tmp_path, monkeypatch, failure):
    calls = []

    def handle(request):
        calls.append(request)
        if failure == "timeout":
            raise httpx.ReadTimeout("timeout", request=request)
        return httpx.Response(400, json=dict(error="bad request"))

    def client():
        c = OpenRouterClient()
        c._client.close()
        c._client = httpx.Client(transport=httpx.MockTransport(handle))
        return c

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-not-a-key")
    monkeypatch.setattr(pilot, "OpenRouterClient", client)
    pilot.reserve(plan, pilot.validate(plan))
    slot = plan["slots"][0]
    first = pilot.one(plan, slot, tmp_path / "outputs")
    assert first["status"] == "unavailable"
    (tmp_path / "outputs" / slot["id"] / "receipt.json").unlink()
    again = pilot.one(plan, slot, tmp_path / "outputs")
    assert "never resend" in again["error"] and len(calls) == 1


def test_native_controls_are_isolated_complete_and_directionally_correct(tmp_path):
    from tools.repair.judge_controls import controls

    rows = controls(tmp_path / "controls", background_count=3)
    assert len(rows) == 5
    packets = {r["case"]: read(r["packet"]["path"]) for r in rows}
    assert all(p["study_split"] is None and p["native_verified"] for p in packets.values())
    domain = packets["range_context"]["payload"]
    assert domain["plan_a"]["query_outcomes"]["intended_range"] == "true"
    assert domain["plan_b"]["query_outcomes"]["intended_range"] == "false"
    assert domain["plan_b"]["query_outcomes"]["intended_domain"] == "true"
    assert packets["tie"]["payload"]["plan_a"] == packets["tie"]["payload"]["plan_b"]
