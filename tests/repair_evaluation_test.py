"""Matched study controls, partial labels and grouped uncertainty conformance."""

import dataclasses

import pytest

from exact.repair.comparisons import freeze_controls, research_arms
from exact.repair.evaluation import paired_group_effects, partial_reference_metrics
from exact.repair.study import StudyArmV2, filter_inventory
from tests.repair_study_test import frozen_case


def test_controls_keep_the_inventory_costs_and_do_not_require_labels():
    case = frozen_case()
    controls = freeze_controls(case, profile=(("delete", 1.0),))
    assert controls.problem is case.problem
    variants = dict(controls.objective_variants)
    assert {"uniform", "confidence", "symbolic"} <= variants.keys()
    assert variants["uniform"].costs == variants["symbolic"].costs
    assert not variants["uniform"].pairs
    assert any(any(value != 0 for value in row) for row in variants["confidence"].benefit)
    assert "confidence" not in dict(
        freeze_controls(dataclasses.replace(case, mapping_scores=())).objective_variants
    )
    arms = research_arms(omitted_families=("complex_equivalence",))
    assert len({a.arm_id for a in arms}) == len(arms)
    assert {"hgt_unary", "hgt_pairwise", "rgcn_unary", "none_unary"} <= {a.arm_id for a in arms}
    p, _, _ = filter_inventory(controls, StudyArmV2("uniform", objective_key="uniform"))
    assert p.objects == controls.problem.objects


def test_partial_reference_absence_stays_unlabelled_and_entity_kind_scoped():
    a = ("class", "a", "b", "=")
    b = ("class", "c", "d", "=")
    result = partial_reference_metrics([a, b, ("individual", "x", "y", "=")], [a])
    assert result["true_positive"] == 1 and result["false_positive"] == 0
    assert result["unlabelled_predictions"] == 1
    assert result["labelled_precision"] == 1
    assert partial_reference_metrics([b], [a], [b])["labelled_f1"] == 0
    with pytest.raises(ValueError, match="overlap"):
        partial_reference_metrics([a], [a], [a])


def test_paired_effects_weight_groups_equally_and_keep_unknown_denominator():
    rows = []
    for case, group, effect in [("a", "large", 2), ("b", "large", 4), ("c", "small", 9)]:
        for arm, value in [("left", 0), ("right", effect)]:
            rows.append(
                dict(
                    case_id=case,
                    group_id=group,
                    arm_id=arm,
                    value=value,
                    logical_status="VERIFIED_FEASIBLE",
                )
            )
    rows.extend(
        dict(
            case_id="unknown", group_id="missing", arm_id=arm, value=None, logical_status="UNKNOWN"
        )
        for arm in ("left", "right")
    )
    report = paired_group_effects(rows, "left", "right", "value", bootstrap_replicates=20)
    assert report["scheduled_cases"] == 4 and report["paired_verified_cases"] == 3
    assert report["per_group_effect"] == {"large": 3.0, "small": 9.0}
    assert report["mean_group_effect"] == 6.0
    assert report["all_scheduled_status"]["right"]["UNKNOWN"] == 1
    assert report == paired_group_effects(rows, "left", "right", "value", bootstrap_replicates=20)


def test_captured_scores_are_read_from_immutable_matching_evidence():
    from exact.repair.study import _captured_scores

    case = frozen_case()
    problem = dataclasses.replace(
        case.problem, evidence=(("mapping", {"mapping": {"Score": 0.83}}),)
    )
    captured = dataclasses.replace(case, problem=problem, mapping_scores=())
    assert _captured_scores(captured) == {"mapping": 0.83}
    assert "confidence" in dict(freeze_controls(captured).objective_variants)


def test_pair_degree_cap_handles_hubs_without_complete_pair_inventory():
    import pyowl_core as owl

    from exact.repair.candidates import mapping_candidates
    from exact.repair.graph import observable_interaction_pairs
    from exact.repair.records import PolicyV2, RepairInputV2, RevisionObjectV2

    hub = owl.Class(owl.IRI("urn:hub"))
    objects = []
    for index in range(120):
        left = owl.Class(owl.IRI(f"urn:left:{index}"))
        candidates = mapping_candidates(str(index), left, hub)
        objects.append(RevisionObjectV2(str(index), "mapping", candidates[0].axioms, candidates))
    problem = RepairInputV2((), tuple(objects), PolicyV2())
    pairs = observable_interaction_pairs(problem, per_object_limit=3)
    assert len(pairs) <= 180
    assert pairs == tuple(sorted(set(pairs)))
    assert all(sum(index in pair for pair in pairs) <= 3 for index in range(120))
    assert observable_interaction_pairs(problem, per_object_limit=0) == ()


def test_public_profile_features_and_legacy_aliases_have_identical_costs():
    from exact.repair.records import candidate_cost

    obj = frozen_case().problem.objects[0]
    candidate = dataclasses.replace(obj.candidates[0], cost_features=(("mapping_deletion", 1.0),))
    assert candidate_cost(obj, candidate, (("mapping_deletion", 0.1),)) == 0.1
    assert candidate_cost(obj, candidate, (("delete", 0.1),)) == 0.1
    candidate = dataclasses.replace(
        candidate, cost_features=(("mapping_deletion", 1.0), ("delete", 0.0))
    )
    with pytest.raises(ValueError, match="conflicting"):
        candidate_cost(obj, candidate, (("mapping_deletion", 0.1),))
