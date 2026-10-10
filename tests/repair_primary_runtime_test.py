"""Process boundaries preserve optimizer/RNG and never execute a foreign phase."""

import random

import pytest
import torch

from exact.repair.records import canonical_hash
from tests.repair_training_completion_test import cache_for
from tools.repair import train
from tools.repair.corpus import generate_corpus
from tools.repair.primary_runtime import (
    DEV_GPU,
    FIT_GPU,
    PhaseBoundary,
    phase_job,
    remaining_dev_reserve,
)


def test_owned_phase_stages_charge_both_devices_to_common_caps():
    fit = phase_job("model-symbolic-13", "fit", 600)
    dev = phase_job("model-symbolic-13", "development", 600)
    oversized = phase_job("model-symbolic-13", "development", 600, oversized=True)
    assert fit["gpu_devices"] == oversized["gpu_devices"] == [FIT_GPU]
    assert dev["gpu_devices"] == [DEV_GPU]
    for job in (fit, dev, oversized):
        assert set(["primary_training", "model-symbolic-13", "learning"]) <= set(
            job["budget_stages"]
        )
    assert "secondary_learning" in dev["budget_stages"]
    assert "secondary_learning" not in oversized["budget_stages"]


def test_remaining_schedule_reserve_does_not_replenish_cases(monkeypatch):
    monkeypatch.setattr("tools.repair.primary_runtime.time.time", lambda: 100)
    limits = {
        canonical_hash(dict(seed=13, supervision_condition="symbolic", epoch=e, case_id=c)): 30
        for e in (2, 5)
        for c in ("a", "b")
    }
    args = ((2, 5), limits, 13, "symbolic", ("a", "b"))
    assert remaining_dev_reserve(*args, [], {}, 0) == 120
    assert remaining_dev_reserve(*args, [{"epoch": 2}], {}, 0) == 60
    assert (
        remaining_dev_reserve(
            *args,
            [{"epoch": 2}],
            dict(epoch=4, generated={"a": {}}, case_deadlines_epoch={"b": 109}),
            0,
        )
        == 9
    )


def test_fit_dev_boundaries_resume_exact_optimizer_and_fit_rng(tmp_path, monkeypatch):
    cases = generate_corpus(
        split_counts={"train": 2, "development": 1, "test": 0},
        siblings_per_parent=1,
        families=("range",),
        revision="v3",
    )
    fitting = [(c, cache_for(c)) for c in cases if c.split == "train"]
    dev = train.scheduled_development(cases, {})
    decoded = []
    monkeypatch.setattr(train, "decoded_development", lambda *a, **k: dict(exact_regret=None))

    def generated(model, case, *args, **kwargs):
        decoded.append(case.case_id)
        torch.rand(50)  # DEV randomness cannot advance the fitting generator.
        random.random()
        return dict(
            parent_group_id=case.structural_parent,
            decoded=dict(status="verified", selected_utility=0.5, checks=1),
        )

    monkeypatch.setattr(train, "generated_development", generated)
    options = dict(
        encoder="none",
        hidden_dim=8,
        heads=2,
        layers=0,
        dropout=0.2,
        epochs=3,
        development_epochs=[2, 3],
        max_full_development_evaluations=2,
        patience_enabled=False,
        sampled_assignments=0,
        proposal_arm="bounded_enumeration",
        deadline_seconds=90,
        total_training_seconds=180,
        decode_seconds=1,
    )

    def run(directory):
        path = directory / "state.pt"
        with pytest.raises(PhaseBoundary) as boundary:
            train.train_cases(fitting, dev, checkpoint_path=path, execution_phase="fit", **options)
        assert boundary.value.receipt["optimizer_updates"] == 4
        saved = torch.load(path, weights_only=True)
        assert saved["history"][0]["development_status"] == "not_scheduled"
        assert saved["phase"] == "development_loss" and saved["next_epoch"] == 1
        before = saved["model"]
        with pytest.raises(ValueError, match="pending DEV"):
            train.train_cases(fitting, dev, checkpoint_path=path, execution_phase="fit", **options)
        with pytest.raises(PhaseBoundary) as boundary:
            train.train_cases(
                fitting, dev, checkpoint_path=path, execution_phase="development", **options
            )
        saved = torch.load(path, weights_only=True)
        assert saved["optimizer_updates"] == 4 and saved["next_epoch"] == 2
        assert all(torch.equal(v, saved["model"][k]) for k, v in before.items())
        with pytest.raises(PhaseBoundary):
            train.train_cases(fitting, dev, checkpoint_path=path, execution_phase="fit", **options)
        model, report = train.train_cases(
            fitting, dev, checkpoint_path=path, execution_phase="development", **options
        )
        assert report["optimizer_updates"] == 6 and report["completed_epochs"] == 3
        assert report["execution_count"] == 4
        return torch.load(path, weights_only=True)

    first = run(tmp_path / "a")
    second = run(tmp_path / "b")
    assert len(decoded) == 4
    assert all(torch.equal(v, second["model"][k]) for k, v in first["model"].items())
    assert torch.equal(first["cpu_rng"], second["cpu_rng"])
    assert first["optimizer"]["state"].keys() == second["optimizer"]["state"].keys()


def evaluation_inputs():
    generated = dict(
        split="test",
        outcomes_opened=False,
        target_revision="test-revision",
        rows=[
            dict(
                case_id=f"g{i}",
                structural_parent=f"p{i//2}",
                family=f"f{i//8}",
                control="coherent" if i % 2 else "corrupted",
                input_hash=f"input{i}",
                observable=dict(path=f"/not-opened/{i}.json", sha256="sealed"),
            )
            for i in range(64)
        ],
    )
    conference = dict(
        rows=[
            dict(
                id=f"c{i}",
                group_id=f"pair{i}",
                split="test" if i < 7 else ("development" if i == 7 else "train"),
                ontology_names=[f"source{i}", "ekaw" if i < 6 else f"target{i}"],
                ontology_bindings=[dict(path="/not-opened/ontology", sha256="sealed")],
                matcher_archive=dict(path="/not-opened/archive", sha256="sealed"),
                matcher_member=f"LogMap-{i}.rdf",
                theory_scope="full",
                license_status="not_qualified",
            )
            for i in range(21)
        ]
    )
    return generated, conference


def test_input_only_schedule_balance_reuse_and_capacity():
    from collections import Counter

    from tools.repair.prepare_primary_runtime import (
        evaluation_schedule,
        validate_schedule,
    )

    generated, conference = evaluation_inputs()
    schedule = evaluation_schedule(
        generated, conference, dict(path="/protocol", sha256="frozen"), {}
    )
    assert validate_schedule(schedule)
    assert len(schedule["expected_runs"]) == 1535
    assert schedule["worker_reservations_seconds"] == dict(generated=61200, conference=52500)
    assert len(schedule["conference_all_pairs"]) == 21
    slots = schedule["semantic_slots"]
    assert Counter(r["seed"] for r in slots) == {13: 43, 37: 43, 73: 42}
    assert sum(r["cohort"] == "conference" for r in slots) == 43
    assert sum(r["contrast"] == "combined_vs_observable_control" for r in slots) == 1
    assert not any(r["experiment"].startswith("E6") for r in schedule["expected_runs"])
    native = [r for r in schedule["expected_runs"] if r["arm"] == "native_deletion"]
    assert all(r["settings"]["generation"] == "none" for r in native)
    after = [r for r in schedule["expected_runs"] if r["arm"] == "deletion_after_rich"]
    assert all(r["settings"]["generation"] == "rich" and r["settings"]["ablation"] for r in after)
    assert schedule["final_annotation_aggregation_reserve_seconds"] == 28800
    schedule["semantic_slots"][0]["left"] = "missing"
    with pytest.raises(ValueError, match="missing output"):
        validate_schedule(schedule)


def test_unresolved_primary_entry_point_cannot_run(tmp_path):
    import json

    from tools.repair.primary_runtime import run_phase

    path = tmp_path / "inert.json"
    path.write_text(
        json.dumps(dict(schema="exact-repair/primary-phase-runtime/v1", execution_authorized=False))
    )
    with pytest.raises(ValueError, match="not admitted"):
        run_phase(path, "fit", tmp_path / "output")
    assert not (tmp_path / "output").exists()
