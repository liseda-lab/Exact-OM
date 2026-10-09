"""Completed report recovery retains selection and rejects changed supervision."""

import copy

import pytest

from exact.repair.records import canonical_hash
from tools.repair.report_recovery import (
    PREDECESSOR,
    UNCHANGED_DEPENDENCIES,
    completed_report_recovery,
)


def fixture():
    options = dict(revision="v3", epochs=2, seed=13, training="train-A", development="dev-A")
    saved = dict(
        schema="exact-repair/training-state/v3",
        recovery_revision="exact-phase-resume/v3.1",
        identity=canonical_hash((options, None, PREDECESSOR)),
        next_epoch=2,
        next_offset=0,
        phase="acquisition",
        pending_acquisition={},
        development_progress={},
        history=[dict(epoch=1, selection_criterion=(-1, -1, 6)), dict(epoch=2)],
        best_epoch=1,
        best_criterion=(-1, -1, 6),
        best_state={"weights": [1, 2]},
        optimizer={"state": "retained"},
        elapsed_seconds=1234,
        execution_count=3,
    )
    return saved, options


def test_finalization_keeps_all_saved_work_and_selection():
    saved, options = fixture()
    before = copy.deepcopy(saved)
    result = completed_report_recovery(saved, options, None, UNCHANGED_DEPENDENCIES)
    assert saved == before
    assert result["additional_optimization_epochs"] == 0
    assert result["selected_epoch"] == 1 and result["budgets_reset"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("identity", "unrelated"),
        ("next_epoch", 1),
        ("next_offset", 1),
        ("phase", "development"),
        ("pending_acquisition", {"case": "pending"}),
        ("development_progress", {"case": "pending"}),
        ("best_state", None),
        ("best_epoch", 2),
        ("best_criterion", (-1, -1, 7)),
        ("history", []),
    ],
)
def test_incomplete_or_inconsistent_saved_work_is_rejected(field, value):
    saved, options = fixture()
    saved[field] = value
    with pytest.raises(ValueError, match="incompatible"):
        completed_report_recovery(saved, options, None, UNCHANGED_DEPENDENCIES)


@pytest.mark.parametrize(
    "field,value",
    [
        ("epochs", 3),
        ("seed", 37),
        ("training", "train-B"),
        ("development", "test-A"),
    ],
)
def test_changed_schedule_or_split_is_rejected(field, value):
    saved, options = fixture()
    options[field] = value
    with pytest.raises(ValueError, match="incompatible"):
        completed_report_recovery(saved, options, None, UNCHANGED_DEPENDENCIES)


def test_changed_repair_implementation_or_warm_start_is_rejected():
    saved, options = fixture()
    for warm, dependencies in [(None, "changed"), ("other-model", UNCHANGED_DEPENDENCIES)]:
        with pytest.raises(ValueError, match="incompatible"):
            completed_report_recovery(saved, options, warm, dependencies)


def test_report_recovery_cannot_start_fresh_or_combine_migrations(tmp_path):
    from tools.repair.train import train_cases

    path = tmp_path / "checkpoint.pt"
    with pytest.raises(ValueError, match="existing checkpoint"):
        train_cases([], [], checkpoint_path=path, resume_report_transport=True)
    path.touch()
    for option in ({"resume_acquisition_deadline": True}, {"resume_endpoint_retrieval": path}):
        with pytest.raises(ValueError, match="no other migration"):
            train_cases([], [], checkpoint_path=path, resume_report_transport=True, **option)


def test_completed_checkpoint_finalizes_without_training_or_reselection(tmp_path, monkeypatch):
    import inspect
    from pathlib import Path

    import torch

    from tests.repair_training_completion_test import cache_for
    from tools.repair import report_recovery, train
    from tools.repair.corpus import generate_corpus

    cases = generate_corpus(
        split_counts={"train": 1, "development": 1, "test": 0},
        siblings_per_parent=1,
        families=("range",),
    )
    training = [(c, cache_for(c)) for c in cases if c.split == "train"]
    development = [(c, cache_for(c)) for c in cases if c.split == "development"]
    path = tmp_path / "state.pt"
    options = dict(
        epochs=1,
        hidden_dim=8,
        heads=2,
        layers=0,
        encoder="none",
        mixtures=1,
        max_depth=1,
        max_constructors=1,
        decode_seconds=30,
        development_draws_per_object=0,
        sampled_assignments=4,
        deadline_seconds=180,
        total_training_seconds=180,
        checkpoint_path=path,
    )
    model, original_report = train.train_cases(training, development, **options)
    saved = torch.load(path, weights_only=True)
    # Model a pinned predecessor in the fixture only; live checkpoints are never rewritten.
    bound = inspect.signature(train.train_cases).bind(training, development, **options)
    bound.apply_defaults()
    identity_options = dict(bound.arguments)
    for key in ("graph_schema", "final_development_reserve_seconds"):
        if identity_options[key] is None:
            identity_options.pop(key)
    for key in (
        "checkpoint_path",
        "deadline_seconds",
        "warm_start_weights",
        "resume_acquisition_deadline",
        "resume_endpoint_retrieval",
        "resume_report_transport",
    ):
        identity_options.pop(key)
    monkeypatch.setattr(report_recovery, "PREDECESSOR", "fixture-completed-predecessor")
    saved["identity"] = canonical_hash((identity_options, None, report_recovery.PREDECESSOR))
    torch.save(saved, path)

    def forbidden(*args, **kwargs):
        raise AssertionError("Completed recovery must not optimize, collect labels or select again")

    monkeypatch.setattr(torch.optim.AdamW, "step", forbidden)
    monkeypatch.setattr(train, "collect_sampled_repairs", forbidden)
    monkeypatch.setattr(train, "generated_development", forbidden)
    with pytest.raises(ValueError, match="incompatible"):
        train.train_cases(training, development, **options)
    # The production migration remains pinned to its original dependency set;
    # changing only a predecessor label must not bypass that guard.
    with pytest.raises(ValueError, match="incompatible"):
        train.train_cases(training, development, resume_report_transport=True, **options)
    fixture_dependencies = canonical_hash(
        [
            (source.name, source.read_bytes().hex())
            for source in sorted(
                (Path(train.__file__).resolve().parents[2] / "exact/repair").glob("*.py")
            )
        ]
    )
    monkeypatch.setattr(report_recovery, "UNCHANGED_DEPENDENCIES", fixture_dependencies)
    resumed, report = train.train_cases(
        training, development, resume_report_transport=True, **options
    )
    assert report["status"] == "complete"
    assert report["history"] == original_report["history"]
    assert report["selected_epoch"] == original_report["selected_epoch"]
    for name, weights in model.state_dict().items():
        assert torch.equal(weights, resumed.state_dict()[name])
    restored = torch.load(path, weights_only=True)
    assert restored["elapsed_seconds"] >= saved["elapsed_seconds"]
    assert restored["execution_count"] == saved["execution_count"] + 1
    assert restored["recovery_lineage"][-1]["additional_optimization_epochs"] == 0
