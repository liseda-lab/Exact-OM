"""Recovery must preserve optimizer/RNG, scientific identity, and spent budgets."""

import json
from dataclasses import replace

import pytest

from exact.repair.checkpointing import CumulativeBudget


def test_cumulative_budget_never_resets_and_charges_lost_work(tmp_path, monkeypatch):
    import exact.repair.checkpointing as module

    now = [10.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    path = tmp_path / "budget.json"
    with CumulativeBudget(path, "fixed", 20) as budget:
        assert budget.begin(8) == 8
        now[0] += 3
        budget.finish()
    with CumulativeBudget(path, "fixed", 20) as budget:
        assert budget.remaining == 17
        budget.begin(5)
        # Simulate a lost process after reserving blocking work.
        budget.close()
        budget.started = None
    with CumulativeBudget(path, "fixed", 20) as budget:
        assert budget.remaining == 12
        assert budget.state["attempts"][-1]["status"] == "lost_reserved_cost"
    with pytest.raises(ValueError, match="identity/limit"):
        CumulativeBudget(path, "changed", 20)
    assert json.loads(path.read_text())["spent_seconds"] == 8


def test_training_recovers_exact_optimizer_rng_and_rejects_changed_split(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from tests.repair_training_completion_test import cache_for
    from tools.repair import train
    from tools.repair.corpus import generate_corpus

    cases = generate_corpus(
        split_counts={"train": 1, "development": 1, "test": 1},
        siblings_per_parent=1,
        families=("range",),
    )
    training = [(c, cache_for(c)) for c in cases if c.split == "train"]
    development = [(c, cache_for(c)) for c in cases if c.split == "development"]
    # Keep actual neural optimizer, dropout and selection; bypass only expensive
    # independent solver/proposal qualification already covered in their suites.
    monkeypatch.setattr(train, "decoded_development", lambda *a, **k: {"exact_regret": 0.0})
    monkeypatch.setattr(
        train, "generated_development", lambda *a, **k: {"useful_candidate_coverage": 1.0}
    )
    options = dict(
        revision="v2",
        epochs=2,
        hidden_dim=8,
        heads=2,
        layers=0,
        encoder="none",
        dropout=0.2,
        proposal_arm="bounded_enumeration",
        deadline_seconds=120,
    )
    baseline, expected = train.train_cases(training, development, **options)
    path = tmp_path / "state.pt"
    save = train.save_training_state

    def interrupt(destination, state):
        save(destination, state)
        if state["next_epoch"] == 1:
            raise InterruptedError("injected after committed epoch")

    monkeypatch.setattr(train, "save_training_state", interrupt)
    with pytest.raises(InterruptedError):
        train.train_cases(training, development, checkpoint_path=path, **options)
    monkeypatch.setattr(train, "save_training_state", save)
    resumed, report = train.train_cases(training, development, checkpoint_path=path, **options)
    assert report["history"] == expected["history"]
    assert report["model_hash"] == expected["model_hash"]
    assert all(
        torch.equal(value, resumed.state_dict()[key])
        for key, value in baseline.state_dict().items()
    )
    with pytest.raises(ValueError, match="incompatible"):
        train.train_cases(training, development, checkpoint_path=path, **{**options, "seed": 37})
    bad = [(replace(training[0][0], split="test"), training[0][1])]
    with pytest.raises(ValueError, match="splits"):
        train.train_cases(bad, development, checkpoint_path=path, **options)


def test_study_resume_cannot_replenish_campaign_budget(tmp_path, monkeypatch):
    import exact.repair.study as study
    from tests.repair_study_test import frozen_case

    monkeypatch.setattr(study, "runtime_manifest", lambda: {"fixture": "v1"})
    from exact.repair.kernel import baseline_identity
    from exact.repair.records import BaselineReportV3

    # This fixture isolates the interrupted campaign budget, not four-theory diagnosis.
    monkeypatch.setattr(
        study,
        "collect_baselines",
        lambda problem: BaselineReportV3(baseline_identity(problem), (), 0.0, 0),
    )
    case = frozen_case()
    arms = (study.StudyArmV2("deletion", "deletion"),)
    # Persist the immutable schedule but interrupt just before evaluation.
    monkeypatch.setattr(
        study, "evaluate_case", lambda *a, **k: (_ for _ in ()).throw(InterruptedError())
    )
    with pytest.raises(InterruptedError):
        study.run_study([case], tmp_path, arms=arms, campaign_seconds=1)
    path = tmp_path / "budget.json"
    state = json.loads(path.read_text())
    state["spent_seconds"] = 1
    path.write_text(json.dumps(state))
    result = study.run_study([case], tmp_path, arms=arms, campaign_seconds=1)
    assert result["rows"][0]["status"] == "timeout"
