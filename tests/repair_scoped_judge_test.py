import copy
import json
import time

import httpx
import pytest

from exact.llm.routing import OpenRouterClient
from exact.repair.records import canonical_hash
from tools.repair import scoped_judge as s
from tools.repair.batch import read
from tools.repair.historical_regression import binding
from tools.repair.shared_release import immutable


def case():
    plan = dict(
        plan_id="p",
        theory_hash="theory",
        content=["S subClassOf T"],
        query_outcomes={"focal": "true"},
        policy_hash="policy",
        query_basis_hash="queries",
    )
    return dict(
        id="control",
        repeats=2,
        role="authored_control",
        gold=dict(
            preference="tie",
            allowed_anchors={k: dict(a=["full"], b=["full"]) for k in s.chat.CRITERIA},
        ),
        scope={k: "The explicitly declared focal content" for k in s.chat.CRITERIA},
        payload=dict(
            plan_a=plan,
            plan_b=copy.deepcopy(plan),
            coverage=dict(complete=True, omissions=[]),
            context=dict(plan_a_id="p", plan_b_id="p", policy_hash="policy"),
            evidence={
                "D1": dict(
                    kind="authored_definition",
                    text="S is a kind of T in a fictional registry; not independent expert truth.",
                )
            },
        ),
    )


def models():
    return dict(
        chat=dict(
            profile=dict(model="openai/gpt-6-luna", provider={}),
            reasoning={"effort": "low"},
            accepted_model_ids=["openai/gpt-6-luna"],
        ),
        decisions=dict(
            profile=dict(
                backend="openrouter",
                model=s.old.decisions.MODEL,
                api_base=s.old.decisions.BASE,
                revision="test-v1",
                api_key_env="SCOPED_TEST_KEY",
                timeout_secs=180,
            )
        ),
    )


def native(wire):
    answers = {}
    for k, q in wire["questions"].items():
        if q["type"] == "choice":
            assert len(q["criteria"]) >= 2, "Provider requires at least two choices"
        if q["type"] == "score":
            answers[k] = dict(
                type="score",
                score=3.9,
                probabilities={str(i): p for i, p in enumerate([0, 0, 0, 0.1, 0.9])},
                legend={str(i): v for i, v in enumerate(q["criteria"])},
                confidence=0.8,
            )
        else:
            choice = (
                "tie"
                if k == "preference"
                else (
                    "supported"
                    if k.endswith(":finding")
                    else "D1" if k.endswith(":source") else "sufficient"
                )
            )
            answers[k] = dict(
                type="choice",
                choice=choice,
                probabilities={a: float(a == choice) for a in q["criteria"]},
                confidence=0.8,
            )
    return dict(
        model=s.old.decisions.MODEL, provider="OpenAI", answers=answers, usage=dict(cost=0.0001)
    )


def test_canonical_reuse_binds_full_context_and_preserves_repeats():
    c = case()
    assert len(s.units(c)) == 1
    slots = s.schedule([c])
    assert len(slots) == 14  # 2*(one canonical plan * 3 stages + 4 direct comparisons)
    assert len({r["id"] for r in slots}) == 14
    d = copy.deepcopy(c)
    d["scope"]["meaning_retention"] = "Changed meaning scope"
    assert set(s.units(c)).isdisjoint(s.units(d))
    d = copy.deepcopy(c)
    d["payload"]["plan_b"]["query_outcomes"]["focal"] = "false"
    assert len(s.units(d)) == 2


@pytest.mark.parametrize("swapped", [False, True])
@pytest.mark.parametrize("model", ["chat", "decisions"])
def test_shared_scope_and_orientation_reach_both_interfaces(swapped, model):
    c = case()
    c["payload"]["plan_b"]["plan_id"] = "other"
    c["payload"]["context"]["plan_b_id"] = "other"
    before = copy.deepcopy(c)
    slot = dict(
        kind="pair", swapped=swapped, model=model, stage="chat" if model == "chat" else "pair"
    )
    wire = s.packet(c, slot, models())
    p = json.loads(wire["messages"][1]["content"]) if model == "chat" else wire["state"]
    assert p["assessment_contract"]["scope"] == c["scope"]
    assert p["context"]["plan_a_id"] == p["plan_a"]["plan_id"]
    assert p["context"]["plan_b_id"] == p["plan_b"]["plan_id"]
    assert "gold" not in json.dumps(wire)
    assert c == before
    c["payload"]["context"]["plan_a_id"] = "wrong"
    with pytest.raises(ValueError, match="Source context"):
        s.packet(c, slot, models())


def test_missing_criterion_keeps_null_without_discarding_known_scores():
    c = case()
    slots = s.schedule([c])
    gs, ss = slots[1:3]
    gate_wire = s.packet(c, gs, models())
    gate = dict(
        valid=True, answers=s.parse(native(gate_wire), gate_wire, "gate", models()["decisions"])
    )
    gate["answers"]["collateral_fidelity"].update(
        choice="missing", probabilities=dict(sufficient=0, missing=1, contradictory=0)
    )
    wire = s.filter_scores(s.packet(c, ss, models()), gate)
    assert set(wire["questions"]["meaning_retention:source"]["criteria"]) == {
        "D1",
        "no_supported_source",
    }
    assert all(not k.startswith("collateral") for k in wire["questions"])
    answer = s.parse(native(wire), wire, "score", models()["decisions"])
    result = s.assessment(ss, dict(valid=True, answers=answer), gate)
    assert result["completed"] and result["total"] is None
    assert result["values"]["meaning_retention"] == pytest.approx(0.975)
    gate["answers"]["meaning_retention"] = dict(type="refusal")
    assert not s.assessment(ss, dict(valid=True, answers=answer), gate)["completed"]


def test_all_missing_can_complete_without_a_paid_score_but_failure_cannot():
    c = case()
    _, gs, ss = s.schedule([c])[:3]
    wire = s.packet(c, gs, models())
    gate = dict(valid=True, answers=native(wire)["answers"])
    for r in gate["answers"].values():
        r.update(choice="missing", probabilities=dict(sufficient=0, missing=1, contradictory=0))
    assert s.filter_scores(s.packet(c, ss, models()), gate) is None
    assert s.assessment(ss, dict(valid=False, status="skipped_dependency"), gate)["completed"]
    gate["valid"] = False
    assert not s.assessment(ss, dict(valid=False), gate)["completed"]


def test_chat_grounding_and_unknown_consistency_are_validated():
    c = case()
    wire = s.packet(c, s.schedule([c])[0], models())
    answer = {
        k: dict(
            anchor="full",
            finding="supported",
            reason="D1 supplies the intended inclusion.",
            evidence_ids=["D1"],
        )
        for k in s.chat.CRITERIA
    }
    response = dict(
        model="openai/gpt-6-luna",
        provider="OpenAI",
        choices=[dict(finish_reason="stop", message=dict(content=json.dumps(answer)))],
    )
    assert s.parse(response, wire, "chat", models()["chat"]) == answer
    for change in (dict(evidence_ids=["invented"]), dict(anchor="unknown"), dict(reason="")):
        bad = copy.deepcopy(answer)
        bad["meaning_retention"].update(change)
        response["choices"][0]["message"]["content"] = json.dumps(bad)
        with pytest.raises(ValueError):
            s.parse(response, wire, "chat", models()["chat"])


@pytest.mark.parametrize("embedded", [False, True])
def test_unary_citation_cannot_borrow_another_plans_entailment(embedded):
    c = case()
    fact = dict(plan_id="other", outcome="false")
    c["payload"]["evidence"]["F1"] = (
        dict(kind="native_entailment", text=json.dumps(fact))
        if embedded
        else dict(kind="native_query", **fact)
    )
    slot = s.schedule([c])[0]
    wire = s.packet(c, slot, models())
    answer = {
        k: dict(
            anchor="full", finding="supported", reason="A supplied fact", evidence_ids=["D1", "F1"]
        )
        for k in s.chat.CRITERIA
    }
    response = dict(
        model="openai/gpt-6-luna",
        provider="OpenAI",
        choices=[dict(finish_reason="stop", message=dict(content=json.dumps(answer)))],
    )
    with pytest.raises(ValueError, match="another plan"):
        s.parse(response, wire, "chat", models()["chat"])


def test_native_live_wire_replay_does_not_repay_and_binds_gate(tmp_path, monkeypatch):
    c, mm = case(), models()
    slots = s.schedule([c])[1:3]
    for slot in slots:
        p = tmp_path / (slot["stage"] + ".json")
        immutable(p, s.packet(c, slot, mm))
        slot["packet"] = binding(p)
    plan = dict(
        schema=s.VERSION,
        semantic_role=s.ROLE,
        budget_phase="development",
        slots=slots,
        models=mm,
        ledger_directory=str(tmp_path / "ledger"),
        deadline_epoch=time.time() + 1000,
    )
    immutable(
        tmp_path / "ledger/phase-reservations.json",
        dict(
            reservations={
                canonical_hash(("development", slot["id"])): dict(
                    request_basis=canonical_hash(plan), state="reserved"
                )
                for slot in slots
            }
        ),
    )
    calls = []

    def handler(request):
        wire = json.loads(request.content)
        assert str(request.url).endswith("/decisions")
        assert wire["state"] == s.state(c, side=slots[0]["side"])
        calls.append(wire)
        return httpx.Response(200, json=native(wire))

    original = OpenRouterClient.__init__

    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self._client.close()
        self._client = httpx.Client(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(OpenRouterClient, "__init__", init)
    monkeypatch.setenv("SCOPED_TEST_KEY", "test-only")
    kwargs = dict(parser=s.parse, score_filter=s.filter_scores)
    gate = s.old.one(plan, slots[0], tmp_path / "out", **kwargs)
    assert gate["valid"], gate
    result = s.old.one(plan, slots[1], tmp_path / "out", gate, **kwargs)
    assert result["valid"], result
    saved = read(tmp_path / "ledger/phase-reservations.json")
    assert s.old.one(plan, slots[1], tmp_path / "out", gate, **kwargs) == result
    (tmp_path / "out" / slots[1]["id"] / "receipt.json").unlink()
    assert (
        s.old.one(plan, slots[1], tmp_path / "out", gate, **kwargs)["answers"] == result["answers"]
    )
    assert len(calls) == 2 and read(tmp_path / "ledger/phase-reservations.json") == saved
    changed = copy.deepcopy(gate)
    changed["answers"]["meaning_retention"]["choice"] = "missing"
    with pytest.raises(ValueError, match="dependencies changed"):
        s.old.one(plan, slots[1], tmp_path / "out", changed, **kwargs)


def test_report_keeps_conflicts_and_duplicate_reuse_out_of_order_accuracy():
    c, mm = case(), models()
    slots, records = s.schedule([c]), []
    for slot in slots:
        wire = s.packet(c, slot, mm)
        if slot["model"] == "chat":
            ans = (
                dict(preference="A", reason="incorrect deliberate conflict", evidence_ids=["D1"])
                if slot["kind"] == "pair"
                else {
                    k: dict(anchor="full", finding="supported", reason="D1", evidence_ids=["D1"])
                    for k in s.chat.CRITERIA
                }
            )
        else:
            ans = native(wire)["answers"]
        records.append(
            dict(slot=slot["id"], valid=True, answers=ans, status="received", usage=dict(cost=0))
        )
    report = s.summarize(dict(slots=slots), {c["id"]: c}, records)
    assert report["methods"]["chat"]["direct_derived_conflicts"] == 4
    assert not report["methods"]["chat"]["automated_qualified"]
    assert all(
        r["score_order_test"].startswith("not independent") for r in report["repeat_order_audits"]
    )
    assert not any(p["eligible"] for p in report["pairs"])


def test_recovery_requires_confirmed_rejection_and_unchanged_gate(tmp_path, monkeypatch):
    from tools.repair.scoped_judge_recovery import replacement_wire

    c, mm = case(), models()
    slots = s.schedule([c])[1:3]
    for slot in slots:
        wire = s.packet(c, slot, mm)
        for key, q in wire["questions"].items():
            if key.endswith(":source"):
                q["criteria"].pop("no_supported_source", None)
        path = tmp_path / (slot["stage"] + ".json")
        immutable(path, wire)
        slot["packet"] = binding(path)
    original = dict(
        slots=slots,
        models=mm,
        ledger_directory=str(tmp_path / "ledger"),
        deadline_epoch=time.time() + 1000,
        budget_phase="development",
    )
    immutable(
        tmp_path / "ledger/phase-reservations.json",
        dict(
            reservations={
                canonical_hash(("development", slot["id"])): dict(
                    request_basis=canonical_hash(original), state="reserved"
                )
                for slot in slots
            }
        ),
    )

    def handler(request):
        wire = json.loads(request.content)
        if any(
            q["type"] == "choice" and len(q["criteria"]) == 1 for q in wire["questions"].values()
        ):
            return httpx.Response(
                400, json=dict(error=dict(message="OpenAI Decisions needs at least 2 choices"))
            )
        return httpx.Response(200, json=native(wire))

    real_init = OpenRouterClient.__init__

    def init(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        self._client.close()
        self._client = httpx.Client(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(OpenRouterClient, "__init__", init)
    monkeypatch.setenv("SCOPED_TEST_KEY", "test-only")
    kwargs = dict(parser=s.parse, score_filter=s.filter_scores)
    gate = s.old.one(original, slots[0], tmp_path / "out", **kwargs)
    failed = s.old.one(original, slots[1], tmp_path / "out", gate, **kwargs)
    assert gate["valid"] and not failed["valid"]
    request = read(tmp_path / "out" / slots[1]["id"] / "request.json")
    fixed = replacement_wire(original, slots[1], failed, gate, request, c)
    assert all(len(q["criteria"]) >= 2 for q in fixed["questions"].values())
    assert fixed == s.filter_scores(s.packet(c, slots[1], mm), gate)
    with pytest.raises(ValueError, match="resample"):
        replacement_wire(original, slots[1], {**failed, "valid": True}, gate, request, c)
    changed = copy.deepcopy(gate)
    changed["answers"]["meaning_retention"]["choice"] = "missing"
    with pytest.raises(ValueError, match="gate"):
        replacement_wire(original, slots[1], failed, changed, request, c)
    changed_case = copy.deepcopy(c)
    changed_case["scope"]["meaning_retention"] = "New semantic scope"
    with pytest.raises(ValueError, match="changes more"):
        replacement_wire(original, slots[1], failed, gate, request, changed_case)
