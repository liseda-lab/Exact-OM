import copy
import json

import pytest

from tools.repair import scoped_judge as s
from tools.repair.judge_projection import INDEX_PREFIX, THEORY_PREFIX, project
from tests.repair_scoped_judge_test import case, models


def attributed_case():
    c = case()
    p = c["payload"]
    p["plan_b"]["plan_id"] = p["context"]["plan_b_id"] = "deleted"
    p["plan_b"]["theory_hash"] = "deleted-theory"
    p["plan_b"]["query_outcomes"]["focal"] = "false"
    p["plan_b"]["content"][0] = THEORY_PREFIX + "Ontology(\n)\n"
    for side in ("a", "b"):
        plan = p["plan_" + side]
        plan["content"].extend(
            [
                json.dumps(
                    dict(
                        object_id="m",
                        kind="mapping",
                        original="fallible original",
                        selected="Ontology(\n)\n" if side == "b" else "selected assertion",
                    )
                ),
                json.dumps(dict(consequence_report_id="proof-" + side)),
            ]
        )
        p["evidence"]["F" + side] = dict(
            kind="native_entailment",
            source_id="proof-" + side,
            symbolic_value=plan["query_outcomes"]["focal"],
            text=json.dumps(
                dict(
                    plan_id=plan["plan_id"],
                    theory_hash=plan["theory_hash"],
                    policy_hash="policy",
                    query_basis_hash="queries",
                    outcome=plan["query_outcomes"]["focal"],
                    query="S subClassOf T",
                )
            ),
        )
    p["evidence"]["DIFF"] = dict(
        kind="native_asserted_text_identity",
        text=json.dumps(
            dict(
                first_plan_id="p",
                second_plan_id="deleted",
                first_plan_theory_hash="theory",
                second_plan_theory_hash="deleted-theory",
                lines_only_in_first_plan=["selected assertion"],
            )
        ),
    )
    p["local_context"] = [INDEX_PREFIX + "Fa true; Fb false", "Shared, necessary domain context"]
    return c


def test_unary_projection_preserves_complete_current_and_original_content_without_foreign_facts():
    c = attributed_case()
    before = copy.deepcopy(c)
    p = s.state(c, side="b")
    assert p["assessment_kind"] == "single_plan"
    assert "unary" in p["projection"]["excluded_reason"]
    assert set(p["evidence"]) == {"D1", "Fb"}
    assert p["evidence"]["D1"] == c["payload"]["evidence"]["D1"]
    assert p["evidence"]["Fb"]["outcome"] == "false"
    assert p["plan"]["native_evidence_ids"] == ["Fb"]
    assert p["plan"]["current_theory"] == "Ontology(\n)\n"
    assert p["plan"]["current_revision_objects"][0]["selected_axioms"] == "Ontology(\n)\n"
    assert (
        p["plan"]["original_observations"]["objects"][0]["observed_axioms"] == "fallible original"
    )
    assert p["local_context"] == ["Shared, necessary domain context"]
    assert p["projection"]["excluded_evidence_ids"] == ["DIFF", "Fa"]
    assert "selected assertion" not in json.dumps(p)
    assert c == before


@pytest.mark.parametrize("swapped", [False, True])
def test_pair_projection_binds_facts_to_oriented_plan_and_retains_both_theories(swapped):
    c = attributed_case()
    p = s.state(c, swapped=swapped)
    assert p["assessment_kind"] == "pairwise_comparison"
    assert p["projection"]["excluded_reason"] is None
    assert set(p["evidence"]) == set(c["payload"]["evidence"])
    for side in ("a", "b"):
        plan = p["plan_" + side]
        assert p["context"]["plan_" + side + "_id"] == plan["plan_id"]
        for key in plan["native_evidence_ids"]:
            assert p["evidence"][key]["applies_to"] == ["plan_" + side]
            assert p["evidence"][key]["plan_id"] == plan["plan_id"]


@pytest.mark.parametrize("field", ["plan_id", "theory_hash", "policy_hash", "query_basis_hash"])
def test_projection_rejects_stale_native_context_before_excluding_other_plan(field):
    c = attributed_case()
    row = c["payload"]["evidence"]["Fa"]
    fact = json.loads(row["text"])
    fact[field] = "foreign"
    row["text"] = json.dumps(fact)
    with pytest.raises(ValueError, match="another plan|context mismatch"):
        s.state(c, side="b")


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_owner",
        "contradictory_owner",
        "contradictory_outcome",
        "duplicate_plan",
        "missing_theory",
    ],
)
def test_projection_rejects_unqualified_or_ambiguous_inputs(mutation):
    c = attributed_case()
    p, row = c["payload"], c["payload"]["evidence"]["Fa"]
    if mutation == "missing_owner":
        fact = json.loads(row["text"])
        del fact["plan_id"]
        row["text"] = json.dumps(fact)
    elif mutation == "contradictory_owner":
        row["plan_id"] = "deleted"
    elif mutation == "contradictory_outcome":
        row["symbolic_value"] = "false"
    elif mutation == "duplicate_plan":
        p["plan_b"]["plan_id"] = "p"
    else:
        p["plan_a"]["content"].pop(0)
    with pytest.raises(ValueError):
        project(p)


def chat_report(unknown=False, conflict=False):
    c, records = attributed_case(), []
    slots = s.schedule([c], ["chat"])
    for slot in slots:
        answer = (
            dict(
                preference="A" if conflict else "abstain" if unknown else "tie",
                reason="Evidence",
                evidence_ids=["D1"],
            )
            if slot["kind"] == "pair"
            else {
                k: dict(
                    anchor="unknown" if unknown and k == "collateral_fidelity" else "full",
                    finding="unknown" if unknown and k == "collateral_fidelity" else "supported",
                    reason="Evidence",
                    evidence_ids=["D1"],
                )
                for k in s.chat.CRITERIA
            }
        )
        records.append(
            dict(slot=slot["id"], valid=True, answers=answer, status="received", usage=dict(cost=0))
        )
    return s.summarize(dict(slots=slots, methods=["chat"]), {c["id"]: c}, records)


def test_chat_only_schedule_never_calls_decisions_and_report_rejects_unknown_numeric_targets():
    report = chat_report(unknown=True)
    assert set(report["methods"]) == {"chat"}
    assert all(r["stage"] == "chat" for r in report["records"] if "stage" in r)
    assert all(
        r["total"] is None and r["values"]["assertion_fidelity"] == 1 for r in report["assessments"]
    )
    assert all(
        p["reason"] == "required_evidence_unknown" and not p["admissible_comparison"]
        for p in report["pairs"]
    )
    assert all(p["direct"] == "abstain" for p in report["pairs"])
    assert {r["stage"] for r in s.schedule([attributed_case()], ["chat"])} == {"chat"}


def test_direct_preference_never_silently_replaced_by_numeric_winner():
    report = chat_report(conflict=True)
    assert all(
        p["direct"] == "A" and p["derived"] == "tie" for p in report["pairs"] if not p["swapped"]
    )
    assert all(
        p["reason"] == "direct_numeric_conflict" and not p["admissible_comparison"]
        for p in report["pairs"]
    )
    assert not report["selection_eligible"]


def test_full_anchor_cannot_claim_lost_or_unsupported_content():
    c = attributed_case()
    slot = next(x for x in s.schedule([c], ["chat"]) if x["kind"] == "plan")
    wire = s.packet(c, slot, models())
    answer = {
        k: dict(anchor="full", finding="lost", reason="Lost content", evidence_ids=["D1"])
        for k in s.chat.CRITERIA
    }
    response = dict(
        model="openai/gpt-6-luna",
        provider="OpenAI",
        choices=[dict(finish_reason="stop", message=dict(content=json.dumps(answer)))],
    )
    with pytest.raises(ValueError, match="inconsistent anchor"):
        s.parse(response, wire, "chat", models()["chat"])
