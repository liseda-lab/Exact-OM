from dataclasses import replace

import pytest

from exact.repair.api import write_artifact
from exact.repair.learning import RepairLabel, TeacherCache
from tools.repair import generated_acquisition as ga, shared_release as release
from tools.repair.historical_regression import binding
from tools.repair.prepare import case_to_dict


def test_nonce_and_step_must_match_terminal_receipt(tmp_path):
    for name, value in {
        "completion": dict(
            dispatch_nonce="same",
            step_id="14451.40",
            status="complete",
            exit_code=0,
            batch=str(tmp_path / "batch.json"),
            job_id="job",
        ),
        "step": dict(dispatch_nonce="same", step_id="14451.40"),
        "outputs": {},
        "batch": dict(jobs=[dict(id="job")]),
    }.items():
        write_artifact(tmp_path / (name + ".json"), value)
    run = {
        name: binding(tmp_path / (name + ".json"))
        for name in ("completion", "step", "outputs", "batch")
    }
    run.update(dispatch_nonce="same", step_id="14451.40", expected_status="complete")
    release.validate_completion(run)
    with pytest.raises(ValueError, match="nonce/Slurm"):
        release.validate_completion(dict(run, dispatch_nonce="another"))
    with pytest.raises(ValueError, match="nonce/Slurm"):
        release.validate_completion(dict(run, step_id="14451.41"))


def test_interrupted_tail_is_retained_without_claiming_no_delivery():
    slots = release.reconcile_slots([dict(order=0, status="verified")], "error")
    assert len(slots) == 16 and slots[0]["status"] == "verified"
    assert all(s["status"] == "not_committed_error" for s in slots[1:])
    with pytest.raises(ValueError, match="order/denominator"):
        release.reconcile_slots([dict(order=1, status="verified")], "error")


def test_risk_semantic_pair_and_proposal_masks_are_distinct():
    labels = tuple(
        RepairLabel(assignment, True, value, 0)
        for assignment, value in (((0, 0), 0), ((1, 0), 1), ((0, 1), 1), ((1, 1), 3))
    ) + (
        RepairLabel((2, 0), False, None, 0),
        RepairLabel((2, 1), None, None, 0),
        RepairLabel((0, 2), True, None, 0),
    )
    cache = TeacherCache((3, 3), labels, False, "partial", (), 1)
    result = release.summarize_labels(cache, ((0, 1),))
    assert result["symbolic_value"] == 4 and result["risk"] == 6
    assert result["unknown_policy"] == result["unknown_semantics"] == 1
    assert result["quartet_positive"] == result["pair"] == 1
    assert result["quartet_targets"] == [1]
    assert result["weak_anchor"] == result["weak_comparison"] == 0
    assert result["proposal_candidate_labels"] == 4 and result["proposal_exact"] is False
    assert release.summarize_labels(cache, ())["pair"] == 0


def test_entry_point_audit_authenticates_generated_cache_and_preserves_missing_case(
    tmp_path, monkeypatch
):
    from tests.repair_generated_acquisition_test import fixture

    case, _, plan, protocol, *_ = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(ga, "_pairs", lambda *args: ((0, 1),))
    result = ga.case_worker(case_to_dict(case), plan, protocol, tmp_path, "case", 1300)
    write_artifact(tmp_path / "protocol.json", protocol)
    plan.update(protocol=binding(tmp_path / "protocol.json"), _path=str(tmp_path / "plan.json"))
    write_artifact(tmp_path / "plan.json", plan)
    write_artifact(tmp_path / "input.json", case_to_dict(case))
    declaration = dict(
        case_id=case.case_id,
        family=case.family,
        control=case.control,
        structural_parent=case.structural_parent,
        input_hash=case.problem.content_hash,
        evaluator=binding(tmp_path / "input.json"),
    )
    value = dict(
        case_id=case.case_id,
        family=case.family,
        control=case.control,
        parent=case.structural_parent,
        scheduled_attempts=16,
        cleanup_complete=True,
        status="complete",
        result=result,
        partial_state=binding(tmp_path / "state.json"),
    )
    write_artifact(tmp_path / "completion.json", value)
    ref = binding(tmp_path / "completion.json")
    row = release.audit_case(declaration, ref, plan, {"completion.json": ref["sha256"]}, tmp_path)
    assert len(row["attempts"]) == 16 and row["masks"]["symbolic_value"] > 0
    assert row["masks"]["quartet_zero"] > 0
    release.immutable(tmp_path / "row.json", row)
    release.immutable(tmp_path / "row.json", row)  # JSON tuple/list round-trip remains resumable.
    changed = dict(value, status="error", result=None)
    write_artifact(tmp_path / "completion.json", changed)
    with pytest.raises(ValueError, match="dependency changed"):
        release.audit_case(declaration, ref, plan, {"completion.json": ref["sha256"]}, tmp_path)
    ref = binding(tmp_path / "completion.json")
    failed = release.audit_case(
        declaration, ref, plan, {"completion.json": ref["sha256"]}, tmp_path
    )
    assert failed["masks"] is None and len(failed["attempts"]) == 16
    summary = release.aggregate([row, failed])
    assert summary["cases"] == 2 and summary["scheduled_slots"] == 32
    assert summary["usable_cases"] == 1 and summary["process_statuses"]["error"] == 1


def test_deadline_prevents_opening_dependencies(tmp_path, monkeypatch):
    monkeypatch.setenv("EXACT_REPAIR_DEADLINE_EPOCH", "1")
    with pytest.raises(TimeoutError, match="deadline"):
        release.authenticate(dict(path=str(tmp_path / "absent"), sha256="unused"))


def test_final_union_preserves_unknowns_and_counts_reused_labels_once():
    from tools.repair.shared_refinement_release import union_caches

    old = TeacherCache(
        (2, 2),
        (RepairLabel((0, 0), True, 1, 0), RepairLabel((1, 0), None, None, 0)),
        False,
        "old",
        (("input", "same"),),
        3,
    )
    new = replace(old, labels=(old.labels[0], RepairLabel((0, 1), True, 2, 0)), elapsed_seconds=2)
    merged = union_caches(old, new)
    assert len(merged.labels) == 3 and merged.labels[1].feasible is None
    assert merged.elapsed_seconds == 5
    with pytest.raises(ValueError, match="prior committed label"):
        union_caches(old, replace(new, labels=(RepairLabel((1, 0), True, 3, 0),)))
    with pytest.raises(ValueError, match="dependencies"):
        union_caches(old, replace(new, hashes=(("input", "different"),)))
