import copy
import json
import time

import httpx
import pytest

from exact.llm.routing import OpenRouterClient
from exact.repair.records import canonical_hash
from tools.repair import probabilistic_judge as p
from tools.repair.batch import read
from tools.repair.historical_regression import binding
from tools.repair.shared_release import immutable


def fixture(tmp_path):
    profile = dict(
        backend="openrouter",
        model=p.decisions.MODEL,
        api_base=p.decisions.BASE,
        revision=p.decisions.MODEL + "-20261006",
        api_key_env="PILOT_TEST_KEY",
        timeout_secs=180,
    )
    plan = dict(
        schema=p.VERSION,
        semantic_role=p.ROLE,
        budget_phase="development",
        ledger_directory=str(tmp_path / "ledger"),
        deadline_epoch=time.time() + 3600,
        models={"decisions": dict(profile=profile)},
    )
    case = dict(
        id="control",
        role="authored_control",
        gold=dict(
            preference="tie",
            allowed_anchors={
                k: dict(a=["mostly", "full"], b=["mostly", "full"]) for k in p.chat.CRITERIA
            },
        ),
        payload=dict(
            plan_a={"content": ["A subClassOf B"]},
            plan_b={"content": ["A subClassOf B"]},
            evidence={"D1": {"text": "A is a kind of B"}},
        ),
    )
    slots = []
    for stage in ("gate", "score"):
        path = tmp_path / (stage + ".json")
        immutable(path, p.packet(case, stage, False, plan["models"]))
        slots.append(
            dict(
                id=stage,
                stage=stage,
                model="decisions",
                case="control",
                swapped=False,
                repeat=0,
                packet=binding(path),
            )
        )
    slots[1]["gate_slot"] = "gate"
    plan["slots"] = slots
    identity = canonical_hash(plan)
    state = dict(
        request_limits=dict(calibration=91, train=251, development=280, test=154),
        total_request_limit=776,
        cost_ceiling_usd=35,
        reservations={},
    )
    for s in slots:
        state["reservations"][canonical_hash(("development", s["id"]))] = dict(
            request_basis=identity, state="reserved", reserved_cost_usd=0.1
        )
    immutable(tmp_path / "ledger/phase-reservations.json", state)
    return plan, case, slots


def response(wire):
    answers = {}
    for key, question in wire["questions"].items():
        if question["type"] == "score":
            answers[key] = dict(
                type="score",
                score=3.4,
                probabilities={"0": 0, "1": 0, "2": 0, "3": 0.6, "4": 0.4},
                legend={str(i): v for i, v in enumerate(question["criteria"])},
                confidence=0.8,
            )
        else:
            option = "tie" if key == "preference" else "sufficient"
            answers[key] = dict(
                type="choice",
                choice=option,
                probabilities={k: float(k == option) for k in question["criteria"]},
                confidence=0.8,
            )
    return dict(
        model=p.decisions.MODEL,
        provider="OpenAI",
        answers=answers,
        usage=dict(cost=0.0001, input_tokens=1000, output_tokens=0),
    )


def test_expected_score_uses_distribution_and_preserves_missing_criteria(tmp_path):
    plan, case, slots = fixture(tmp_path)
    gate_wire = read(slots[0]["packet"]["path"])
    gate = dict(valid=True, answers=p.validate_decisions(response(gate_wire), gate_wire))
    template = read(slots[1]["packet"]["path"])
    wire = p.score_packet(template, gate)
    rec = dict(valid=True, answers=p.validate_decisions(response(wire), wire))
    row = p.assessment(case, slots[1], rec, gate)
    assert row["scores"] == pytest.approx([0.85, 0.85])
    assert row["control_correct"]
    gate["answers"]["collateral_fidelity:a"].update(
        choice="missing", probabilities=dict(sufficient=0.1, missing=0.9, contradictory=0)
    )
    wire = p.score_packet(template, gate)
    assert "collateral_fidelity:a" not in wire["questions"]
    assert all(len(q["criteria"]) == 5 for q in wire["questions"].values() if q["type"] == "score")
    rec = dict(valid=True, answers=p.validate_decisions(response(wire), wire))
    row = p.assessment(case, slots[1], rec, gate)
    assert row["scores"][0] is None and row["scores"][1] == pytest.approx(0.85)
    assert row["derived_preference"] == "abstain"
    assert row["direct_preference"] == "tie"
    assert not row["selection_eligible"] and not row["training_eligible"]


def test_refusal_retains_other_scores_and_is_not_unknown_success(tmp_path):
    plan, case, slots = fixture(tmp_path)
    wire = read(slots[1]["packet"]["path"])
    result = response(wire)
    result["answers"]["meaning_retention:a"] = dict(type="refusal")
    parsed = p.validate_decisions(result, wire)
    row = p.assessment(case, slots[1], dict(valid=True, answers=parsed))
    assert row["criteria"]["meaning_retention:a"] is None
    assert row["criteria"]["assertion_fidelity:a"] == pytest.approx(0.85)
    assert not row["control_correct"]


def test_gate_does_not_force_score_or_use_arbitrary_high_confidence(tmp_path):
    plan, _, slots = fixture(tmp_path)
    template = read(slots[1]["packet"]["path"])
    gate_wire = read(slots[0]["packet"]["path"])
    parsed = response(gate_wire)["answers"]
    for answer in parsed.values():
        answer.update(
            probabilities=dict(sufficient=0.55, missing=0.45, contradictory=0), confidence=0.1
        )
    assert len(p.score_packet(template, dict(valid=True, answers=parsed))["questions"]) == 7
    for answer in parsed.values():
        answer["probabilities"] = dict(sufficient=0.5, missing=0.5, contradictory=0)
    assert list(p.score_packet(template, dict(valid=True, answers=parsed))["questions"]) == [
        "preference"
    ]
    assert p.score_packet(template, dict(valid=False)) is None


def test_live_wire_and_safe_replay_bind_gate_and_raw_probabilities(tmp_path, monkeypatch):
    plan, case, slots = fixture(tmp_path)
    calls = []

    def handler(request):
        assert str(request.url) == p.decisions.BASE + "/decisions"
        wire = json.loads(request.content)
        assert wire["state"] == case["payload"]
        assert "gold" not in json.dumps(wire)
        calls.append(wire)
        return httpx.Response(200, json=response(wire))

    real_init = OpenRouterClient.__init__

    def init(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        self._client.close()
        self._client = httpx.Client(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(OpenRouterClient, "__init__", init)
    monkeypatch.setenv("PILOT_TEST_KEY", "test-only-key")
    directory = tmp_path / "output"
    gate = p.one(plan, slots[0], directory)
    assert gate["valid"], gate
    score = p.one(plan, slots[1], directory, gate)
    assert score["valid"], score
    assert len(calls) == 2
    assert p.one(plan, slots[1], directory, gate) == score
    saved_ledger = read(tmp_path / "ledger/phase-reservations.json")
    (directory / "score/receipt.json").unlink()
    restored = p.one(plan, slots[1], directory, gate)
    assert restored["answers"] == score["answers"] and len(calls) == 2
    assert read(tmp_path / "ledger/phase-reservations.json") == saved_ledger
    changed = copy.deepcopy(gate)
    changed["answers"]["meaning_retention:a"]["probabilities"]["sufficient"] = 0.9
    with pytest.raises(ValueError, match="dependencies changed"):
        p.one(plan, slots[1], directory, changed)
    altered = copy.deepcopy(restored)
    altered["answers"]["meaning_retention:a"]["score"] = 0
    (directory / "score/receipt.json").write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="parsed judgment changed"):
        p.one(plan, slots[1], directory, gate)


def test_unknown_delivery_is_never_repeated(tmp_path, monkeypatch):
    plan, _, slots = fixture(tmp_path)
    calls = []

    def handler(request):
        calls.append(request)
        raise httpx.ReadTimeout("unknown delivery")

    real_init = OpenRouterClient.__init__

    def init(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        self._client.close()
        self._client = httpx.Client(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(OpenRouterClient, "__init__", init)
    monkeypatch.setenv("PILOT_TEST_KEY", "test-only-key")
    directory = tmp_path / "output"
    record = p.one(plan, slots[0], directory)
    assert not record["valid"] and len(calls) == 1
    (directory / "gate/receipt.json").unlink()
    recovered = p.one(plan, slots[0], directory)
    assert not recovered["valid"] and len(calls) == 1
    skipped = p.one(plan, slots[1], directory, recovered)
    assert skipped["status"] == "skipped_dependency" and len(calls) == 1


@pytest.mark.parametrize("delta,expected", [(0, "tie"), (0.049, "tie"), (0.051, "A")])
def test_material_ties(delta, expected):
    assert p.derived([0.5 + delta, 0.5]) == expected
    assert p.derived([None, 0.5]) == "abstain"
