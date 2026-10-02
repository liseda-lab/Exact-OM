"""Observed coherent bundles and dependency-checked training rollback."""

import copy
from dataclasses import replace
from types import SimpleNamespace

import pyowl_core as owl
import pytest

from exact.repair.candidates import mapping_candidates, materialize_retrieved_endpoints
from exact.repair.records import (
    PolicyV3,
    RepairInputV3,
    RevisionObjectV3,
    canonical_hash,
)
from exact.repair.retrieval import ObjectMenus
from tools.repair import endpoint_recovery as recovery


def fixture(relation="<", side="target"):
    s, t, u, v = [owl.Class(owl.IRI("urn:endpoint:" + x)) for x in "stuv"]
    left, right = (s, u) if side == "target" else (u, t)
    candidates = mapping_candidates("m", left, right, relation)
    obj = RevisionObjectV3(
        "m", "mapping", candidates[0].axioms, candidates, source_entity=s, target_entity=t
    )
    problem = RepairInputV3((), (obj,), PolicyV3((s, t, u, v)))
    menu = ObjectMenus("m", endpoint_alternatives=(("source", v), ("target", v)))
    retrieval = SimpleNamespace(for_object=lambda _: menu)
    return problem, retrieval, left, right, v


@pytest.mark.parametrize("relation", ["<", ">", "="])
@pytest.mark.parametrize("side", ["source", "target"])
def test_endpoint_retrieval_preserves_complete_observed_relation(relation, side):
    problem, retrieval, left, right, alternative = fixture(relation, side)
    observed = problem.objects[0].original_axioms
    result = materialize_retrieved_endpoints(problem, retrieval)
    assert materialize_retrieved_endpoints(result, retrieval) == result
    expected = mapping_candidates(
        "m",
        left,
        right,
        relation,
        endpoint_alternatives=(("source", alternative), ("target", alternative)),
        enabled_actions=("keep", "replace_endpoint"),
    )
    additions = [c for c in result.objects[0].candidates if "replace_endpoint" in c.action_tags]
    assert {c.axioms for c in additions} == {
        c.axioms for c in expected if "replace_endpoint" in c.action_tags
    }
    assert result.objects[0].original_axioms == observed
    assert {c.axioms for c in result.objects[0].candidates if "keep" in c.action_tags} == {observed}
    # Frozen provenance is not rewritten to disguise input incompatibility.
    assert result.objects[0].source_entity == problem.objects[0].source_entity
    assert result.objects[0].target_entity == problem.objects[0].target_entity


def test_endpoint_retrieval_does_not_drop_extra_original_axioms():
    problem, retrieval, left, _, alternative = fixture()
    obj = problem.objects[0]
    bad = replace(obj, original_axioms=(*obj.original_axioms, owl.SubClassOf(left, alternative)))
    with pytest.raises(ValueError, match="complete original relation"):
        materialize_retrieved_endpoints(replace(problem, objects=(bad,)), retrieval)


def recovery_fixture():
    # canonical_hash requires serializable scientific inputs.
    from tools.repair.corpus import generate_corpus

    case = next(
        c
        for c in generate_corpus(
            split_counts={"train": 1, "development": 1, "test": 0},
            siblings_per_parent=1,
            families=("range",),
        )
        if c.split == "train"
    )
    options = dict(epochs=10, training=[(case, None)], seed=13)
    collection = dict(schema="exact-repair/collection-state/v3.2", collection_dependencies={})
    collection["collection_identity"] = canonical_hash({})
    baseline = dict(
        schema="exact-repair/training-state/v3",
        recovery_revision="exact-phase-resume/v3.1",
        identity=canonical_hash((options, None, recovery.INITIAL_IMPLEMENTATION)),
        phase="acquisition",
        next_epoch=0,
        next_offset=0,
        optimized=0,
        history=[],
        optimizer={"state": {}},
        best_state=None,
        elapsed_seconds=100,
        execution_count=1,
        model={"retained": [1, 2]},
        cpu_rng=[3, 4],
        pending_acquisition=dict(
            epoch=0, proposed=[], case_id=case.case_id, collection_state=collection
        ),
    )
    generated = {
        "dev": dict(
            status="generation_error",
            detail="ValueError: cannot infer complete original relation for endpoint retrieval",
        )
    }
    failed = dict(
        baseline,
        identity=canonical_hash((options, None, recovery.PREDECESSOR)),
        next_epoch=10,
        best_epoch=None,
        elapsed_seconds=4000,
        execution_count=2,
        history=[
            (
                dict(epoch=e, generated=generated, selection_criterion=None)
                if e in (1, 5, 10)
                else dict(epoch=e)
            )
            for e in range(1, 11)
        ],
    )
    return failed, baseline, options


def test_rollback_keeps_original_state_and_later_costs_without_mutating_evidence():
    failed, baseline, options = recovery_fixture()
    originals = copy.deepcopy((failed, baseline))
    restored = recovery.recover_endpoint_state(
        failed, baseline, options, None, recovery.UNCHANGED_DEPENDENCIES
    )
    assert (failed, baseline) == originals
    assert restored["model"] == baseline["model"]
    assert restored["optimizer"] == baseline["optimizer"]
    assert restored["cpu_rng"] == baseline["cpu_rng"]
    assert restored["next_epoch"] == 0 and restored["history"] == []
    assert restored["elapsed_seconds"] == 4000 and restored["execution_count"] == 2
    assert restored["pending_acquisition"]["case_deadline_exhausted"]
    assert restored["recovery_lineage"][-1]["budgets_reset"] is False


@pytest.mark.parametrize("changed", ["settings", "dependencies", "optimized", "selected", "cause"])
def test_rollback_rejects_changed_dependencies_or_unproven_boundary(changed):
    failed, baseline, options = recovery_fixture()
    dependencies = recovery.UNCHANGED_DEPENDENCIES
    if changed == "settings":
        options["seed"] = 37
    elif changed == "dependencies":
        dependencies = "different"
    elif changed == "optimized":
        baseline["optimizer"]["state"] = {"update": 1}
    elif changed == "selected":
        failed["best_epoch"] = 5
    else:
        failed["history"][-1]["generated"]["dev"]["detail"] = "unknown"
    with pytest.raises(ValueError, match="incompatible|diagnosed"):
        recovery.recover_endpoint_state(failed, baseline, options, None, dependencies)
