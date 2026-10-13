import copy

import pytest

from tools.repair import fixed_endpoint_freeze as fixed


def test_early_opening_preserves_absolute_deadline_and_final_reserve(tmp_path):
    from tools.repair.historical_regression import binding
    from tools.repair.shared_release import immutable

    opening, deadline, reserve = 1000, 400000, 28800
    original = dict(
        measurements_deadline_epoch=deadline, final_annotation_aggregation_reserve_seconds=reserve
    )
    immutable(tmp_path / "old-schedule.json", original)
    immutable(tmp_path / "authorization.json", dict(decision="Start TEST as soon as ready"))
    immutable(
        tmp_path / "amendment.json",
        dict(
            schema="exact-repair/early-test-opening/v1",
            new_opening_epoch=opening,
            original_schedule=binding(tmp_path / "old-schedule.json"),
            authorization=binding(tmp_path / "authorization.json"),
        ),
    )
    immutable(tmp_path / "schedule.json", dict(original, models_freeze_epoch=opening))
    manifest = dict(
        not_before_epoch=opening,
        deadline_epoch=deadline,
        schedule=binding(tmp_path / "schedule.json"),
        opening_amendment=binding(tmp_path / "amendment.json"),
    )
    assert fixed.repair_deadline(manifest, opening) == deadline - reserve
    assert fixed.repair_deadline(manifest, opening + 100) == deadline - reserve
    assert fixed.repair_deadline({}, opening) == opening + 40 * 3600
    with pytest.raises(ValueError, match="authorization, deadline or final reserve"):
        fixed.repair_deadline(dict(manifest, deadline_epoch=deadline + 1), opening)


def fixture(monkeypatch):
    rules = dict(checkpoint_selection=fixed.RULE, epoch=6, optimizer_updates=756)
    records = {
        "plan": dict(authorization="authorization", rules=rules),
        "authorization": dict(rules=rules, training_weights_unchanged=True),
    }
    rows = []
    for condition, seed in sorted(fixed.MEMBERS):
        key = f"{condition}-{seed}"
        artifact = dict(path=key, sha256=key, model_hash=key)
        records[key] = dict(
            condition=condition,
            seed=seed,
            epoch=6,
            optimizer_updates=756,
            weights_unchanged=True,
            artifact=artifact,
        )
        rows.append(
            dict(
                condition=condition,
                seed=seed,
                weights=key,
                selected_checkpoint=artifact,
                selection_rule=fixed.RULE,
                selection_criterion=None,
                semantic_numeric_available=0,
            )
        )
    records["report"] = dict(
        models=rows, dev_outcomes_selected_checkpoints=False, test_opened=False
    )
    records["freeze"] = dict(
        schema="exact-repair/noise-aware-dev/v1",
        plan="plan",
        report="report",
        models=rows,
        selection_rule=fixed.RULE,
        outcome_selection=False,
        test_opened=False,
    )
    monkeypatch.setattr(fixed, "bound", lambda ref: records[ref])
    authenticated = []
    monkeypatch.setattr(fixed, "authenticate", authenticated.append)
    return records, authenticated


def test_all_six_unknown_dev_scores_keep_exact_weights(monkeypatch):
    records, checked = fixture(monkeypatch)
    result = fixed.models("freeze")
    assert len(result) == len(checked) == 6
    assert all(m["selection_criterion"] is None for m in records["freeze"]["models"])
    freeze = dict(
        schema=fixed.SCHEMA,
        noise_aware_dev_freeze="freeze",
        models=result,
        selection_rule=fixed.RULE,
        outcome_selection=False,
        test_outcomes_opened=False,
        final_freeze=True,
    )
    fixed.validate(freeze)
    changed = copy.deepcopy(freeze)
    changed["models"]["hgt-pair-symbolic-s13"]["sha256"] = "another checkpoint"
    with pytest.raises(ValueError, match="TEST model bindings"):
        fixed.validate(changed)


@pytest.mark.parametrize(
    "field,value", [("epoch", 5), ("optimizer_updates", 755), ("weights_unchanged", False)]
)
def test_rejects_different_training_endpoint(monkeypatch, field, value):
    records, _ = fixture(monkeypatch)
    records["symbolic-13"][field] = value
    with pytest.raises(ValueError, match="fixed training endpoint"):
        fixed.models("freeze")


@pytest.mark.parametrize("change", ["drop_model", "score", "opened", "authorization"])
def test_rejects_filtering_scores_and_unapproved_selection(monkeypatch, change):
    records, _ = fixture(monkeypatch)
    if change == "drop_model":
        records["freeze"]["models"].pop()
    elif change == "score":
        records["freeze"]["models"][0]["selection_criterion"] = 1.0
    elif change == "opened":
        records["freeze"]["test_opened"] = True
    else:
        records["authorization"]["rules"] = dict(checkpoint_selection="best_dev_score")
    with pytest.raises(ValueError):
        fixed.models("freeze")
