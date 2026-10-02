"""Case deadlines retain partial supervision without interrupting the training stage."""

import copy
from types import SimpleNamespace

import pytest

from exact.repair.learning import collect_sampled_repairs
from exact.repair.records import canonical_hash, make_objective
from exact.repair.workers import CallResult
from tests.repair_review2_labels_test import collector_options, partial_collection
from tests.repair_training_completion_test import cache_for
from tools.repair import train
from tools.repair.corpus import generate_corpus


def test_exhausted_case_retains_labels_and_all_slots_without_more_calls():
    state, calls, label = partial_collection()
    original = copy.deepcopy(state)
    result = collect_sampled_repairs(
        (3, 3), label, resume_state=state, **{**collector_options(), "deadline_seconds": 0}
    )
    assert len(calls) == 1
    assert state == original
    assert len(result.cache.labels) == 1
    assert result.cache.labels[0].benefit == 0.75
    assert len(result.attempts) == 4 and result.stop_reason == "deadline"
    assert [row["status"] for row in result.attempts] == ["verified"] + ["unvisited"] * 3
    assert result.collection_identity == state["collection_identity"]
    assert result.cache.elapsed_seconds >= state["elapsed_seconds"]


@pytest.mark.parametrize("stage_limited", [False, True])
def test_training_distinguishes_case_and_stage_deadlines(tmp_path, monkeypatch, stage_limited):
    import torch

    cases = generate_corpus(
        split_counts={"train": 1, "development": 1, "test": 0},
        siblings_per_parent=1,
        families=("range",),
    )
    training = [(c, cache_for(c)) for c in cases if c.split == "train"]
    development = [(c, cache_for(c)) for c in cases if c.split == "development"]
    monkeypatch.setattr(
        train,
        "_freeze_training_model",
        lambda problem, *a, **k: CallResult(
            "complete",
            SimpleNamespace(
                problem=problem, objective=make_objective(problem.objects), proposal_reports=()
            ),
        ),
    )
    monkeypatch.setattr(
        train,
        "collect_sampled_repairs",
        lambda *a, **k: collect_sampled_repairs(*a, **{**k, "deadline_seconds": 0}),
    )
    monkeypatch.setattr(train, "decoded_development", lambda *a, **k: {"exact_regret": 0.0})
    monkeypatch.setattr(
        train,
        "generated_development",
        lambda *a, **k: dict(
            useful_candidate_coverage=1.0,
            decoded=dict(status="verified", selected_utility=0.5, checks=1),
        ),
    )
    state = tmp_path / "state.pt"
    options = dict(
        epochs=1,
        hidden_dim=8,
        heads=2,
        layers=0,
        encoder="none",
        proposal_arm="bounded_enumeration",
        plan_risk=False,
        sampled_assignments=4,
        collection_options={
            "plan_attempts_per_case": 4,
            "plan_quotas": dict(utility=0, proposal=0, diversity=4, quartet=0, uniform=0),
        },
        decode_seconds=1000 if stage_limited else 1,
        deadline_seconds=60,
        checkpoint_path=state,
    )
    if stage_limited:
        with pytest.raises(TimeoutError, match="exact phase checkpoint"):
            train.train_cases(training, development, **options)
        saved = torch.load(state, weights_only=True)
        assert saved["phase"] == "acquisition" and saved["next_epoch"] == 0
        assert saved["pending_acquisition"]
    else:
        _, report = train.train_cases(training, development, **options)
        saved = torch.load(state, weights_only=True)
        assert saved["next_epoch"] == 1 and not saved["pending_acquisition"]
        assert report["acquisition_rounds"][0]["stop_reason"] == "deadline"
        assert len(report["acquisition_rounds"][0]["attempts"]) == 4


def test_legacy_migration_requires_all_dependencies_and_preoptimization_state():
    options = {"seed": 13, "split": "frozen"}
    collection_dependencies = {"input": "immutable", "query": "frozen"}
    saved = dict(
        schema="exact-repair/training-state/v3",
        recovery_revision="exact-phase-resume/v3.1",
        identity=canonical_hash((options, None, train._ACQUISITION_DEADLINE_PREDECESSOR)),
        phase="acquisition",
        next_epoch=0,
        next_offset=0,
        optimized=0,
        history=[],
        optimizer={"state": {}},
        best_state=None,
        pending_acquisition=dict(
            epoch=0,
            proposed=[],
            collection_state=dict(
                schema="exact-repair/collection-state/v3.2",
                collection_dependencies=collection_dependencies,
                collection_identity=canonical_hash(collection_dependencies),
            ),
        ),
    )
    dependencies = train._ACQUISITION_UNCHANGED_DEPENDENCIES
    assert train._acquisition_deadline_recovery(saved, options, None, dependencies)["migration"]
    for field, value in (
        ("identity", "renamed"),
        ("schema", "exact-repair/training-state/v2"),
        ("phase", "train"),
        ("next_epoch", 1),
        ("optimized", 1),
        ("optimizer", {"state": {"updated": True}}),
    ):
        with pytest.raises(ValueError, match="incompatible"):
            train._acquisition_deadline_recovery(
                {**saved, field: value}, options, None, dependencies
            )
    for changed_options, changed_dependencies in (
        ({**options, "seed": 37}, dependencies),
        (options, "changed"),
    ):
        with pytest.raises(ValueError, match="incompatible"):
            train._acquisition_deadline_recovery(saved, changed_options, None, changed_dependencies)
