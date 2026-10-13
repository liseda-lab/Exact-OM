import copy
import json
import time

import httpx
import pytest

from exact.llm.routing import OpenRouterClient
from exact.repair.records import canonical_hash
from tools.repair import noisy_dev as n
from tools.repair.batch import read
from tools.repair.historical_regression import binding
from tools.repair.shared_release import immutable


def case(identifier="case", **changes):
    plan = dict(
        plan_id="p",
        theory_hash="t",
        policy_hash="policy",
        query_basis_hash="queries",
        content=["Complete reconstructed theory:\nOntology(\n SubClassOf(<urn:S> <urn:T>)\n)\n"],
        query_outcomes={"focal": "true"},
    )
    c = dict(
        id=identifier,
        case_id=identifier,
        parent=identifier,
        seed=13,
        condition="symbolic",
        family="hierarchy",
        audited=True,
        swapped=False,
        scope=n.SCOPE,
        payload=dict(
            case_id=identifier,
            plan_a=plan,
            plan_b=copy.deepcopy(plan),
            coverage=dict(complete=True, omissions=[]),
            context=dict(plan_a_id="p", plan_b_id="p", policy_hash="policy"),
            evidence={"D1": dict(kind="authored_definition", text="S is a kind of T.")},
        ),
    )
    return {**c, **changes}


def models():
    return dict(
        chat=dict(
            profile=dict(
                backend="openrouter",
                model="openai/gpt-6-luna",
                revision="test-v1",
                api_key_env="NOISY_TEST_KEY",
                timeout_secs=180,
                provider={},
            ),
            reasoning=dict(effort="low"),
            accepted_model_ids=["openai/gpt-6-luna"],
        )
    )


def answer(anchor="full"):
    return {
        k: dict(
            anchor=anchor,
            finding="unknown" if anchor == "unknown" else "supported",
            reason="Supplied definition",
            evidence_ids=["D1"],
        )
        for k in n.scoped.chat.CRITERIA
    }


def observations(aa, bb, votes):
    records, link = {}, dict(plans=dict(a=[], b=[]), pairs=[])
    for i, (a, b, vote) in enumerate(zip(aa, bb, votes)):
        for s, value in (("a", a), ("b", b)):
            key = s + str(i)
            records[key] = dict(valid=True, answers=answer(value))
            link["plans"][s].append(key)
        key = "pair" + str(i)
        records[key] = dict(valid=True, answers=dict(preference=vote))
        link["pairs"].append(dict(slot=key, swapped=False))
    return records, link


def test_unknown_is_masked_without_renormalizing_or_losing_known_components():
    records, link = observations(["full"], ["limited"], ["A"])
    records["a0"]["answers"]["collateral_fidelity"] = answer("unknown")["collateral_fidelity"]
    r = n.aggregate(case(), link, records)
    assert r["score"] is None and not r["ordinal_eligible"] and r["scheduled"]
    assert r["assessments"]["a"]["medians"]["meaning_retention"] == 1
    assert r["assessments"]["a"]["descriptive_total"] is None
    assert r["direct_preference"] == "A" and r["preference"] == "abstain"


def test_absolute_score_instability_can_retain_consistent_ordinal_information():
    records, link = observations(["mixed", "full"], ["none", "limited"], ["A", "A"])
    r = n.aggregate(case(), link, records)
    assert r["ordinal_eligible"] and r["preference"] == "A"
    assert not r["numeric_eligible"] and r["score"] is None
    assert r["assessments"]["a"]["criteria"]["meaning_retention"] == [0.5, 1]


@pytest.mark.parametrize("votes", [["A", "B"], ["A", "A"]])
def test_disagreement_or_numeric_contradiction_abstains_without_flipping(votes):
    records, link = observations(["limited"] * 2, ["mostly"] * 2, votes)
    r = n.aggregate(case(), link, records)
    assert r["canonical_votes"] == votes
    assert not r["ordinal_eligible"] and r["score"] is None
    assert r["preference"] == "abstain" and r["derived_preferences"] == ["B", "B"]


def test_a_stable_comparison_is_eligible_and_swap_is_canonicalized():
    records, link = observations(["mostly"] * 2, ["limited"] * 2, ["A", "B"])
    link["pairs"][1]["swapped"] = True
    r = n.aggregate(case(), link, records)
    assert r["score"] == 0.75 and r["preference"] == "A"
    assert r["canonical_votes"] == ["A", "A"]


def test_missing_observation_is_not_silently_dropped_from_quorum():
    records, link = observations(["full"] * 2, ["none"] * 2, ["A", "A"])
    del records["a1"]
    r = n.aggregate(case(), link, records)
    assert r["score"] is None and r["preference"] == "abstain"
    assert r["assessments"]["a"]["criteria"]["meaning_retention"] == [1, None]


def test_exact_wire_reuse_does_not_collapse_repeats_or_context():
    c, mm = case(), models()
    slots, links = n.schedule([c], mm)
    assert len(slots) == 4  # One shared identical plan and one pair, twice.
    assert links[c["id"]]["plans"]["a"] == links[c["id"]]["plans"]["b"]
    assert len(set(links[c["id"]]["plans"]["a"])) == 2
    d = copy.deepcopy(c)
    d["scope"] = {**d["scope"], "meaning_retention": "Different scope"}
    other, _ = n.schedule([d], mm)
    assert {s["id"] for s in slots}.isdisjoint(s["id"] for s in other)
    for s in slots:
        p = json.loads(n.packet(c, s, mm)["messages"][1]["content"])
        assert p["assessment_contract"]["version"] == n.VERSION
        assert p["assessment_kind"] == (
            "single_plan" if s["kind"] == "plan" else "pairwise_comparison"
        )


def test_hard_evidence_and_identity_checks_are_preserved():
    c, mm = case(), models()
    slot = dict(kind="plan", side="a", swapped=False, stage="chat", model="chat")
    wire = n.packet(c, slot, mm)
    p = json.loads(wire["messages"][1]["content"])
    p["evidence"]["F1"] = dict(kind="native_intended_entailed", symbolic_value="false", plan_id="p")
    wire["messages"][1]["content"] = json.dumps(p)
    response = dict(
        model="openai/gpt-6-luna",
        provider="OpenAI",
        choices=[dict(finish_reason="stop", message=dict(content=json.dumps(answer())))],
    )
    with pytest.raises(ValueError, match="Perfect retention"):
        n.parse(response, wire, "chat", mm["chat"])
    response["model"] = "foreign-model"
    with pytest.raises(ValueError, match="identity"):
        n.parse(response, wire, "chat", mm["chat"])


def test_parent_denominators_uncertainty_and_sensitivity_are_deterministic():
    rows = []
    for seed, parent, scores in [(13, "p1", (0.5, 1)), (37, "p1", (0.5, 1)), (73, "p2", (None, 1))]:
        for condition, score in zip(("symbolic", "symbolic_plus_llm"), scores):
            rows.append(
                dict(seed=seed, parent=parent, case_id=parent, condition=condition, score=score)
            )
    r = n.paired_summary(rows)
    assert r == n.paired_summary(rows)
    assert r["scheduled_pairs"] == 3 and r["scheduled_parents"] == 2
    assert r["evaluable_pairs"] == 2 and r["evaluable_parents"] == 1
    assert r["full_denominator_missing_label_bounds"] == [-0.25, 0.75]
    assert r["conclusion"] == "inconclusive" and r["descriptive_parent_bootstrap_95"] is None


def test_entrypoint_keeps_six_checkpoints_with_noise_and_replay_does_not_repay(
    tmp_path, monkeypatch
):
    mm, cases, model_rows = models(), [], []

    def save(name, value):
        p = tmp_path / (name + ".json")
        immutable(p, value)
        return binding(p)

    for seed in (13, 37, 73):
        for condition in ("symbolic", "symbolic_plus_llm"):
            c = case(
                condition + str(seed),
                parent="parent" + str(seed),
                case_id="pair" + str(seed),
                seed=seed,
                condition=condition,
            )
            cases.append(c)
            row = save(c["id"] + "-row", dict(result=dict(decoded=dict(status="verified"))))
            source = save(c["id"] + "-source", dict(rows=[row] * 32))
            weights = save(c["id"] + "-weights", dict(artifact=dict(path=c["id"], sha256="frozen")))
            model_rows.append(
                dict(seed=seed, condition=condition, weights=weights, source_report=source)
            )
    slots, links = n.schedule(cases, mm)
    by_case = {c["id"]: c for c in cases}
    for s in slots:
        s["packet"] = save(s["id"], n.packet(by_case[s["case"]], s, mm))
    prior = save("prior", dict(models=model_rows))
    phase = dict(reservations={}, request_limits=dict(development=len(slots)), cost_ceiling_usd=35)
    previous = save("previous", phase)
    save("ledger/phase-reservations", phase)
    plan = dict(
        slots=slots,
        models=mm,
        prior_development_plan=prior,
        previous_phase_state=previous,
        ledger_directory=str(tmp_path / "ledger"),
        semantic_role=n.ROLE,
        transport_role=n.ROLE,
        budget_phase="development",
        deadline_epoch=time.time() + 1000,
    )
    path = save("plan", plan)
    costs = {s["id"]: dict(reserved_cost_usd=0.001) for s in slots}
    monkeypatch.setattr(n, "validate", lambda p: (cases, links, costs))
    calls = []

    def handler(request):
        wire = json.loads(request.content)
        p = json.loads(wire["messages"][1]["content"])
        calls.append(wire)
        ans = (
            dict(preference="tie", reason="Identical", evidence_ids=["D1"])
            if "plan_a" in p
            else answer("unknown" if "symbolic13" in p["case_id"] else "full")
        )
        return httpx.Response(
            200,
            json=dict(
                model="openai/gpt-6-luna",
                provider="OpenAI",
                usage=dict(cost=0.0001),
                choices=[dict(finish_reason="stop", message=dict(content=json.dumps(ans)))],
            ),
        )

    original = OpenRouterClient.__init__

    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self._client.close()
        self._client = httpx.Client(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(OpenRouterClient, "__init__", init)
    monkeypatch.setenv("NOISY_TEST_KEY", "test-only")
    r = n.run(path["path"], tmp_path / "out")
    assert r["status"] == "complete_with_uncertainty"
    assert r["selected_models_available"] == 6 and r["logical_scheduled"] == 192
    assert r["scheduled_comparisons"] == 6 and r["numeric_available"] == 5
    assert not r["test_opened"] and not r["dev_outcomes_selected_checkpoints"]
    saved = read(tmp_path / "ledger/phase-reservations.json")
    assert n.run(path["path"], tmp_path / "out") == r
    assert len(calls) == len(slots)
    assert read(tmp_path / "ledger/phase-reservations.json") == saved
    bad = copy.deepcopy(r["records"])
    bad[-1]["request_id"] = bad[0]["request_id"]
    with pytest.raises(ValueError, match="independent"):
        n.summarize(plan, cases, links, bad)
    # Replay is bound to the whole policy, not just the request packet.
    with pytest.raises(ValueError, match="dependencies changed"):
        n.scoped.old.one(
            {**plan, "changed_policy": True},
            slots[0],
            tmp_path / "out/presentations",
            parser=n.parse,
        )
