"""Selected-profile continuity without new calibration calls or history resets."""

import copy
from dataclasses import replace

import pytest

from exact.repair.records import canonical_hash
from tests.repair_annotation_controls_test import panel_fixture, save
from tests.repair_semantic_fidelity_test import packet
from tests.repair_training_completion_test import cache_for
from tools.repair import corrective_semantics as runner
from tools.repair import teacher_panel as panel
from tools.repair.teacher_selection import qualified_contract
from tools.repair.prepare_teacher_successor import (
    annotation_template,
    comparison_pairs,
    evidence_status,
)
from tools.repair.corpus import generate_corpus


def selected_fixture(tmp_path, monkeypatch):
    source, _ = panel_fixture(tmp_path, monkeypatch)
    source.update(
        request_limits=dict(panel.FOUR_MODEL_LIMITS),
        prompt_version="semantic-fidelity-prompt/v3.2",
        prompt_hash="frozen",
        rubric_version="frozen",
        criterion_weights={"meaning_retention": 1.0},
        request_profiles={"teacher": {"max_output_tokens": 2000}},
    )
    proposal = runner.read(source["teacher_panel"]["proposal"]["path"])
    proposal.update(
        request_limits=dict(panel.FOUR_MODEL_LIMITS),
        max_requests=32,
        max_reserved_usd=0.13296,
        calibration_comparison_identity_limit=56,
    )
    proposal["contracts"] = {"teacher": canonical_hash(panel.contract(source))}
    source["teacher_panel"]["proposal"] = save(tmp_path, "four-proposal.json", proposal)
    approval = runner.read(source["teacher_panel"]["authorization"]["path"])
    approval["proposal"] = source["teacher_panel"]["proposal"]
    source["teacher_panel"]["authorization"] = save(tmp_path, "four-approval.json", approval)
    runner._reserve_phase(source, "new-0-0", packet().content_hash, 0.0042, comparison_id="panel-0")
    state = runner.read(tmp_path / "ledger/phase-reservations.json")
    phase_ref = save(tmp_path, "closed-panel-state.json", state)
    gate = save(
        tmp_path,
        "gate.json",
        dict(
            status="qualified",
            selected_profile="teacher",
            metrics={"teacher": dict(valid=8, correct=8, order_consistent=4, eligible=True)},
        ),
    )
    review = save(
        tmp_path,
        "review.json",
        dict(
            status="qualified",
            selected_profile="teacher",
            scheduled=32,
            rows=[{}] * 32,
            gate=gate,
            phase_state=phase_ref,
        ),
    )
    selected = save(
        tmp_path,
        "selected.json",
        dict(
            schema="exact-repair/qualified-teacher/v1",
            selected_profile="teacher",
            review=review,
            gate=gate,
            manifest=save(tmp_path, "source.json", source),
            phase_state=phase_ref,
            request_limits=source["request_limits"],
        ),
    )
    return annotation_template(selected, source, "train", runner.time.time() + 3600), state


def test_successor_reserves_train_without_calibration_reset_or_window_extension(
    tmp_path, monkeypatch
):
    settings, before = selected_fixture(tmp_path, monkeypatch)
    # Calibration's admission window is over. TRAIN uses its own existing stage;
    # the calibration contract and original charges remain unchanged.
    monkeypatch.setattr(panel.time, "time", lambda: 2000000000)
    # This is a simulated future TRAIN stage, not an expiry of the Slurm worker
    # running the fixture. Keep the production bound-artifact deadline guard on.
    monkeypatch.setenv("EXACT_REPAIR_DEADLINE_EPOCH", "2000003600")
    settings["deadline_epoch"] = 2000003600
    assert runner._reserve_phase(settings, "train-0", "packet", 0.0048, comparison_id="c0") is None
    after = runner.read(tmp_path / "ledger/phase-reservations.json")
    assert after["request_limits"] == dict(calibration=91, train=325, development=96, test=192)
    assert all(after["reservations"][k] == v for k, v in before["reservations"].items())
    assert len(after["reservations"]) == len(before["reservations"]) + 1
    row = after["reservations"][canonical_hash(("train", "train-0"))]
    assert row["phase"] == "train" and "teacher_panel" not in row
    assert runner._reserve_phase(settings, "train-0", "packet", 0.0048, comparison_id="c0") is False
    assert runner.read(tmp_path / "ledger/phase-reservations.json") == after


@pytest.mark.parametrize(
    "defect", ["quota", "provider", "model", "controls", "gate", "phase", "rubric", "too_many_rows"]
)
def test_successor_rejects_changed_profile_or_authorization(tmp_path, monkeypatch, defect):
    settings, _ = selected_fixture(tmp_path, monkeypatch)
    if defect == "quota":
        settings["request_limits"]["train"] += 1
    elif defect == "provider":
        settings["profiles"]["teacher"]["provider"]["only"] = ["fallback"]
    elif defect == "model":
        settings["profiles"]["teacher"]["model"] = "different/model"
    elif defect == "controls":
        settings["request_profiles"]["teacher"]["max_output_tokens"] += 1
    elif defect == "gate":
        settings["calibration_gate"] = save(tmp_path, "other-gate.json", dict(status="qualified"))
    elif defect == "phase":
        settings["phase"] = "calibration"
    elif defect == "rubric":
        settings["rubric_version"] = "changed"
    else:
        settings["slots"] = [{}] * 257
    with pytest.raises(ValueError, match="successor changed"):
        qualified_contract(settings)


def test_successor_rejects_erased_calibration_reservation_before_train(tmp_path, monkeypatch):
    settings, prior = selected_fixture(tmp_path, monkeypatch)
    changed = copy.deepcopy(prior)
    # Includes the panel's post-amendment row, not merely the original32 receipts.
    changed["reservations"].pop(canonical_hash(("calibration", "new-0-0")))
    runner.write_artifact(tmp_path / "ledger/phase-reservations.json", changed)
    with pytest.raises(ValueError, match="reset calibration history"):
        runner._reserve_phase(settings, "train-0", "packet", 0.0048)
    assert runner.read(tmp_path / "ledger/phase-reservations.json") == changed


def test_successor_keeps_unique_comparison_cap_distinct_from_swaps(tmp_path, monkeypatch):
    settings, _ = selected_fixture(tmp_path, monkeypatch)
    state = runner.read(tmp_path / "ledger/phase-reservations.json")
    for i in range(256):
        state["reservations"]["train-fixture-" + str(i)] = dict(
            phase="train",
            slot=str(i),
            packet_hash=str(i),
            comparison_id=str(i),
            reserved_cost_usd=0.0048,
            state="completed",
        )
    runner.write_artifact(tmp_path / "ledger/phase-reservations.json", state)
    assert runner._reserve_phase(settings, "swap-0", "p", 0.0048, comparison_id="0") is None
    with pytest.raises(runner.AnnotationBudgetExhausted, match="unique comparison"):
        runner._reserve_phase(settings, "new-257", "p", 0.0048, comparison_id="new")


def test_pair_selection_is_shared_stable_and_not_sorted_by_value():
    case = generate_corpus(
        split_counts=dict(train=1, development=0, test=0),
        families=("papers",),
        siblings_per_parent=1,
        revision="v3",
    )[0]
    cache = cache_for(case)
    pairs = comparison_pairs(case, cache)
    permuted = replace(
        cache,
        labels=tuple(
            replace(label, benefit=1 - label.benefit) if label.benefit is not None else label
            for label in reversed(cache.labels)
        ),
    )
    assert comparison_pairs(case, permuted) == pairs
    assert len(pairs) == 2 and pairs[0] != pairs[1]
    assert comparison_pairs(case, None) == []


def test_noisy_matcher_text_does_not_manufacture_grounded_definitions():
    case = generate_corpus(
        split_counts=dict(train=1, development=0, test=0),
        families=("papers",),
        siblings_per_parent=1,
        revision="v3",
    )[0]
    evidence = evidence_status(case)
    assert evidence["noisy_matcher_records"] > 0
    assert not evidence["grounded_definition_available"]
    assert evidence["declared_consequence_queries"] > 0


def test_postprocessing_manifest_cannot_be_transmitted(tmp_path, monkeypatch):
    settings, _ = selected_fixture(tmp_path, monkeypatch)
    settings["postprocessing_only"] = True
    settings["authorized"] = True
    # The execution guard must act before network setup. Other validation still
    # rejects this fixture's deliberately minimal request controls.
    monkeypatch.setattr(runner, "_manifest_prompt", lambda m: "fixture")
    monkeypatch.setattr("tools.repair.annotation_profile.request_profile", lambda *a: {})
    with pytest.raises(PermissionError, match="Postprocessing"):
        runner.validate_manifest(settings)


def test_local_summary_uses_actual_worker_output_and_aggregate_uses_explicit_paths(
    tmp_path, monkeypatch
):
    import exact.llm.routing
    from tests.repair_semantic_fidelity_test import adapter
    from tests.repair_october_learning_annotation_runner_test import manifest
    from tools.repair.corrective_calibration import summarize

    client, calls = adapter(tmp_path / "ledger", monkeypatch)
    monkeypatch.setattr(exact.llm.routing, "LLMRouter", lambda *a, **k: client.router)
    settings = manifest(tmp_path)
    output = tmp_path / "actual-output"
    labels = runner.annotate_packet(packet(), settings, output / "s", slot_id="s", seconds=300)
    assert labels is not None and len(calls) == 1
    packet_ref = save(tmp_path, "packet.json", packet().to_dict())
    settings.update(
        calibration_profiles=["teacher"],
        gold={packet().case_id: "A"},
        slots=[
            dict(
                id="s",
                profile="teacher",
                packet=packet_ref,
                swapped=False,
                output_directory=str(tmp_path / "obsolete-output"),
            )
        ],
    )
    ref = save(tmp_path, "summary.json", settings)
    assert summarize(ref["path"], output, local_run=True)["metrics"]["teacher"]["valid"] == 1
    assert summarize(ref["path"], tmp_path / "summary")["metrics"]["teacher"]["valid"] == 0
    settings["slots"][0]["output_directory"] = str(output)
    ref = save(tmp_path, "summary-corrected.json", settings)
    assert summarize(ref["path"], tmp_path / "aggregate")["metrics"]["teacher"]["valid"] == 1
    assert len(calls) == 1
