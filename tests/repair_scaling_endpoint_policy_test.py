"""Explicit endpoint ablations preserve complete coherent observation bundles."""

from types import SimpleNamespace

import pytest

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
def test_complete_composite_control_requires_explicit_endpoint_ablation(depth):
    original = scaling.extend_case(parent_case("papers", depth, 13), 4)
    clean = coherent_control(original)
    assert len(clean.problem.objects[0].original_axioms) == 3
    config = scaling.configurations()[0]
    for schedule in ({}, {"endpoint_policy": policy(2)}):
        # The old contract still materializes elementary mappings, and still
        # rejects guessing a relation for the complete three-axiom control.
        _, observed, retrieval = scaling.generation_input(
            original.problem.to_dict(), config, schedule
        )
        assert any(m.endpoint_alternatives for m in retrieval.menus)
        assert observed.objects[0].original_axioms == original.problem.objects[0].original_axioms
        with pytest.raises(ValueError, match="cannot infer complete original relation"):
            scaling.generation_input(clean.problem.to_dict(), config, schedule)
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
