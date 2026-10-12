import copy
import json

import httpx
import pytest

from exact.llm.routing import OpenRouterClient
from tools.repair import decisions_pilot as pilot
from tools.repair.batch import read
from tools.repair.historical_regression import binding
from tools.repair.shared_release import immutable


def cases():
    return [
        dict(
            id="control",
            role="authored_control",
            gold=None,
            payload=dict(
                plan_a={"axioms": ["A subClassOf B"]},
                plan_b={"axioms": []},
                evidence={"D1": "A is a kind of B"},
            ),
        )
    ]


def response(wire, *, unknown=False):
    answers = {}
    for key, question in wire["questions"].items():
        if question["type"] == "noul":
            answers[key] = dict(type="noul", noul=0 if unknown else 1)
        elif question["type"] == "score":
            answers[key] = dict(
                type="score",
                score=4,
                confidence=1,
                probabilities={str(i): float(i == 4) for i in range(5)},
                legend={str(i): v for i, v in enumerate(question["criteria"])},
            )
        else:
            value = (
                ("abstain" if unknown else "tie")
                if key.endswith(":preference")
                else ("unknown" if unknown else "full")
            )
            answers[key] = dict(
                type="choice",
                choice=value,
                confidence=1,
                probabilities={k: float(k == value) for k in question["criteria"]},
            )
    return dict(
        model=pilot.MODEL,
        provider="OpenAI",
        answers=answers,
        usage=dict(input_tokens=200, output_tokens=0, cost=0.00002),
    )


def test_swap_changes_only_plan_presentation_and_never_sends_gold():
    original = cases()
    original[0]["gold"] = {"preference": "A"}
    a, b = pilot.payload(original), pilot.payload(original, True)
    assert a["state"]["control"]["plan_a"] == b["state"]["control"]["plan_b"]
    assert a["state"]["control"]["evidence"] == b["state"]["control"]["evidence"]
    assert a["questions"] == b["questions"]
    assert "gold" not in json.dumps(a)
    assert not {"messages", "reasoning", "max_tokens", "response_format"} & a.keys()


def test_unknown_never_becomes_numeric_training_label():
    wire = pilot.payload(cases())
    row = pilot.assess(response(wire, unknown=True), wire, cases())[0]
    assert row["derived_preference"] == "abstain" and row["preference_consistent"]
    assert row["scores"] == [None, None]
    assert all(v["expected_score"] is None for r in row["criteria"].values() for v in r.values())
    assert row["training_eligible"] is False and row["selection_eligible"] is False


def test_choice_only_followup_preserves_unknown_without_forcing_a_score():
    wire = pilot.payload(cases(), include_scores=False)
    assert all(q["type"] != "score" for q in wire["questions"].values())
    row = pilot.assess(response(wire, unknown=True), wire, cases())[0]
    assert row["scores"] == [None, None] and row["derived_preference"] == "abstain"
    assert row["preference_consistent"]


@pytest.mark.parametrize(
    "mutation", ["missing", "nan", "mass", "wrong_score", "wrong_model", "legend"]
)
def test_malformed_or_inconsistent_decisions_are_rejected(mutation):
    wire = pilot.payload(cases())
    result = response(wire)
    key = "control:meaning_retention:a:score"
    if mutation == "missing":
        result["answers"].pop(key)
    elif mutation == "nan":
        result["answers"][key]["probabilities"]["0"] = float("nan")
    elif mutation == "mass":
        result["answers"][key]["probabilities"]["0"] = 0.5
    elif mutation == "wrong_score":
        result["answers"][key]["score"] = 2
    elif mutation == "wrong_model":
        result["model"] = "openai/gpt-6-luna"
    else:
        result["answers"][key]["legend"]["0"] = "different rubric"
    with pytest.raises(ValueError):
        pilot.assess(result, wire, cases())


def test_preference_contradiction_is_reported_without_reversing_it():
    wire = pilot.payload(cases())
    result = response(wire)
    result["answers"]["control:preference"].update(
        choice="B", probabilities={"A": 0, "B": 1, "tie": 0, "abstain": 0}
    )
    row = pilot.assess(result, wire, cases())[0]
    assert row["preference"] == "B" and row["derived_preference"] == "tie"
    assert not row["preference_consistent"]


def test_live_wire_uses_decisions_and_replay_preserves_costs(tmp_path, monkeypatch):
    cs = cases()
    source = tmp_path / "source.json"
    gold = dict(
        preference="tie",
        allowed_anchors={k: dict(a=["full"], b=["full"]) for k in pilot.pilot.CRITERIA},
    )
    immutable(
        source, dict(payload=cs[0]["payload"], gold=gold, study_split=None, native_verified=True)
    )
    cs[0].update(source=binding(source), gold=gold)
    immutable(tmp_path / "cases.json", cs)
    route = tmp_path / "endpoint.json"
    immutable(
        route,
        dict(
            data=dict(
                endpoints=[
                    dict(
                        model_id=pilot.MODEL,
                        tag="openai",
                        context_length=1050000,
                        pricing=dict(prompt="0.0000001", completion="0"),
                    )
                ]
            )
        ),
    )
    ledger = tmp_path / "ledger"
    before = dict(
        schema="exact-repair/annotation-phase-budget/v1",
        cost_ceiling_usd=35,
        request_limits=dict(calibration=91, train=251, development=207, test=155),
        reservations={"historical": dict(phase="train", state="unresolved", reserved_cost_usd=19)},
    )
    immutable(ledger / "phase-reservations.json", before)
    immutable(tmp_path / "before.json", before)
    slots = []
    for swapped in (False, True):
        path = tmp_path / (str(swapped) + ".json")
        immutable(path, pilot.payload(cs, swapped))
        slots.append(
            dict(
                id=str(swapped),
                model="decisions",
                case="bundle",
                swapped=swapped,
                packet=binding(path),
            )
        )
    plan = dict(
        schema=pilot.VERSION,
        semantic_role=pilot.ROLE,
        budget_phase="development",
        max_requests=2,
        authorization=dict(user_text="Test Decisions"),
        profile=dict(
            api_base=pilot.BASE,
            model=pilot.MODEL,
            api_key_env="DECISIONS_TEST_KEY",
            revision=pilot.MODEL + "-20261006",
            timeout_secs=180,
        ),
        cases=binding(tmp_path / "cases.json"),
        endpoint=binding(route),
        slots=slots,
        pilot_cost_ceiling_usd=1,
        previous_phase_state=binding(tmp_path / "before.json"),
        ledger_directory=str(ledger),
        deadline_epoch=2000000000,
    )
    immutable(tmp_path / "plan.json", plan)
    calls = []

    def handler(request):
        assert str(request.url) == pilot.BASE + "/decisions"
        wire = json.loads(request.content)
        assert wire == pilot.payload(cs, bool(calls))
        calls.append(wire)
        return httpx.Response(200, json=response(wire))

    real_init = OpenRouterClient.__init__

    def init(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        self._client.close()
        self._client = httpx.Client(transport=httpx.MockTransport(handler))

    monkeypatch.setenv("DECISIONS_TEST_KEY", "test-key")
    monkeypatch.setattr(OpenRouterClient, "__init__", init)
    first = pilot.run(tmp_path / "plan.json", tmp_path / "output")
    charged = read(ledger / "phase-reservations.json")
    assert first["status"] == "complete" and len(calls) == 2
    assert first["swap_audits"][0]["anchors_stable"]
    assert charged["reservations"]["historical"] == before["reservations"]["historical"]
    assert pilot.run(tmp_path / "plan.json", tmp_path / "output") == first
    assert len(calls) == 2 and read(ledger / "phase-reservations.json") == charged
    changed = copy.deepcopy(plan)
    changed["authorization"]["user_text"] = "Changed protocol"
    with pytest.raises(ValueError, match="dependencies changed"):
        pilot.one(changed, slots[0], tmp_path / "output/presentations")
    # Unknown delivery is separately guarded by the shared ledger; removing a
    # local receipt must replay its durable raw response, not issue another call.
    (tmp_path / "output/presentations/False/receipt.json").unlink()
    recovered = pilot.one(plan, slots[0], tmp_path / "output/presentations")
    assert recovered["status"] == "received" and len(calls) == 2
