"""Explicit endpoint ablations preserve complete coherent observation bundles."""

from types import SimpleNamespace

import pytest
import pyowl_core as owl

from exact.repair.records import read_record
from tools.repair import scaling
from tools.repair.corpus import coherent_control
from tools.repair.expanded_corpus import bound
from tools.repair.expanded_profile import parent_case


def policy(count):
    return dict(
        schema="exact-repair/scaling-endpoint-policy/v1",
        endpoints_per_side=count,
        admission="complete_elementary_relation",
    )


@pytest.mark.parametrize("depth", [13, 14])
def test_complete_composite_control_supports_full_substitution_and_explicit_ablation(depth):
    original = scaling.extend_case(parent_case("papers", depth, 13), 4)
    clean = coherent_control(original)
    assert len(clean.problem.objects[0].original_axioms) == 3
    config = scaling.configurations()[0]
    for schedule in ({}, {"endpoint_policy": policy(2)}):
        # COR-05 keeps elementary recovery and substitutes the complete
        # explicitly bound composite, without guessing an elementary relation.
        _, observed, retrieval = scaling.generation_input(
            original.problem.to_dict(), config, schedule
        )
        assert any(m.endpoint_alternatives for m in retrieval.menus)
        assert observed.objects[0].original_axioms == original.problem.objects[0].original_axioms
        _, prepared, clean_retrieval = scaling.generation_input(
            clean.problem.to_dict(), config, schedule
        )
        obj, supplied = prepared.objects[0], clean.problem.objects[0]
        assert obj.original_axioms == supplied.original_axioms
        assert (
            obj.source_entity == supplied.source_entity
            and obj.target_entity == supplied.target_entity
        )
        by_id = {c.candidate_id: c for c in obj.candidates}
        for old in supplied.candidates:
            assert by_id[old.candidate_id].axioms == old.axioms
            assert by_id[old.candidate_id].active_expressions == old.active_expressions
        # The observed definition has both necessary directions and a jointly
        # sufficient condition. Check all emitted axioms independently.
        source, target = obj.source_entity, obj.target_entity
        condition = next(
            a.super_class
            for a in obj.original_axioms
            if a.sub_class == target and a.super_class != source
        )
        menu = clean_retrieval.for_object(obj.object_id)
        assert {side for side, _ in menu.endpoint_alternatives} == {"source", "target"}
        for side, endpoint in menu.endpoint_alternatives:
            candidate = next(
                c
                for c in obj.candidates
                if ("endpoint_contract", "complete-bound-substitution/v1") in c.provenance
                and ("endpoint_side", side) in c.provenance
                and ("bound_replacement", str(endpoint.iri.value)) in c.provenance
            )
            left = endpoint if side == "source" else source
            right = endpoint if side == "target" else target
            conjunction = (
                left
                if left == condition
                else owl.ObjectIntersectionOf(owl.CanonicalSet((left, condition)))
            )
            expected = {
                owl.SubClassOf(right, left),
                owl.SubClassOf(right, condition),
                owl.SubClassOf(conjunction, right),
            }
            assert set(candidate.axioms) == expected
            costs = dict(candidate.cost_features)
            assert costs["endpoint_change"] == 1
            assert (
                costs["new_constructor"]
                == costs["removed_direction"]
                == costs["necessary_condition"]
                == 0
            )
            assert ("cost_contract", "syntax-delta/v2") in candidate.provenance
            from exact.repair.owl import OwlVerifier, snapshot_from_axioms

            snapshot = snapshot_from_axioms(candidate.axioms)
            report = OwlVerifier("auto", backend="auto").check_theory(snapshot, (left, right))
            assert report.complete and report.logical_status == "VERIFIED_FEASIBLE"
    for case in (original, clean):
        before = case.problem.to_dict()
        observed, prepared, retrieval = scaling.generation_input(
            before, config, {"endpoint_policy": policy(0)}
        )
        assert observed == prepared == case.problem
        assert not any(m.endpoint_alternatives for m in retrieval.menus)
        assert case.problem.to_dict() == before


@pytest.mark.parametrize("control", ["corrupted", "coherent"])
def test_generate_entry_point_records_effective_policy_and_retains_supplied_inventory(
    tmp_path, monkeypatch, control
):
    from exact.repair import grammar, proposals

    case = scaling.extend_case(parent_case("papers", 13, 13), 4)
    if control == "coherent":
        case = coherent_control(case)
    # Exercise actual retrieval, complete-bundle materialization, grammar and
    # pool publication, while excluding native qualification from this unit test.
    monkeypatch.setattr(grammar, "with_immutable_context", lambda encoding, *a, **k: encoding)
    monkeypatch.setattr(
        proposals, "enumerate_grammar", lambda *a, **k: SimpleNamespace(candidates=())
    )
    schedule = dict(
        endpoint_policy=policy(0),
        context_checks=0,
        generation_seconds=60,
        max_expressions=10,
        seed=13,
        draws_per_object=0,
    )
    result = scaling.generate(
        case.problem.to_dict(),
        scaling.configurations()[0],
        "semantic_enumeration_decoder",
        schedule,
        tmp_path / "pool",
        tmp_path / "cache",
    )
    pool = bound(result)
    assert pool["endpoint_policy"] == policy(0)
    assert pool["completed_objects"] == pool["scheduled_objects"] == 4
    prepared = read_record(pool["input"])
    for old, new in zip(case.problem.objects, prepared.objects):
        assert old.original_axioms == new.original_axioms
        by_id = {candidate.candidate_id: candidate for candidate in new.candidates}
        for candidate in old.candidates:
            assert by_id[candidate.candidate_id].axioms == candidate.axioms
            assert by_id[candidate.candidate_id].active_expressions == candidate.active_expressions
    assert all(not row["effective_menu"]["endpoint_alternatives"] for row in pool["reports"])


@pytest.mark.parametrize(
    "value",
    [
        None,
        {},
        {"endpoints_per_side": 0},
        policy(-1),
        policy(True),
        dict(policy(0), admission="guess_equality"),
        dict(policy(0), silently_drop_axioms=True),
    ],
)
def test_malformed_or_unsupported_endpoint_policy_is_rejected(value):
    with pytest.raises(ValueError, match="captured scaling endpoint policy"):
        scaling.endpoint_policy({"endpoint_policy": value})


def test_legacy_policy_is_explicit_and_returned_without_mutating_schedule():
    assert scaling.endpoint_policy({}) == policy(2)
    schedule = {"endpoint_policy": policy(0)}
    recorded = scaling.endpoint_policy(schedule)
    recorded["endpoints_per_side"] = 2
    assert schedule["endpoint_policy"] == policy(0)
