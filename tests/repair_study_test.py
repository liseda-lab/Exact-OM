"""Tiny frozen-case study conformance; no benchmark or scale experiments."""

import dataclasses
import json

import pyowl_core as owl
import pytest

import exact.repair.kernel as kernel
import exact.repair.study as study
from exact.repair.api import write_artifact
from exact.repair.candidates import (
    make_candidate,
    mapping_candidates,
    ontology_candidates,
)
from exact.repair.records import (
    BudgetsV2,
    ObjectiveV2,
    ObligationV2,
    PolicyV2,
    RepairInputV2,
    RevisionObjectV2,
    VerificationReportV2,
    canonical_hash,
    read_record,
)
from exact.repair.workers import CallResult


def cls(name):
    return owl.Class(owl.IRI(f"urn:study:{name}"))


def frozen_case(case_id="fixture", group="parent-1"):
    left, right, expression = map(cls, "SRE")
    pool = mapping_candidates("mapping", left, right, expressions=[expression])
    mapping = RevisionObjectV2(
        "mapping", "mapping", next(c.axioms for c in pool if "keep" in c.action_tags), pool
    )
    axiom = owl.SubClassOf(expression, left)
    keep = make_candidate("ontology", [axiom], ["keep"])
    ontology = RevisionObjectV2(
        "ontology", "ontology_axiom", (axiom,), (keep,), occurrence_id="document:7"
    )
    ontology = dataclasses.replace(
        ontology, candidates=ontology_candidates(ontology, expressions=[right])
    )
    problem = RepairInputV2(
        (),
        (mapping, ontology),
        PolicyV2(monitored_classes=(left, right, expression)),
        source_identity="source-v1",
        target_identity="target-v1",
        budgets=BudgetsV2(total_seconds=5, solver_seconds=1, verification_seconds=1, max_checks=50),
    )
    unary = tuple(
        tuple(0 if "keep" in c.action_tags else -1 for c in obj.candidates)
        for obj in problem.objects
    )
    objective = ObjectiveV2(unary, ((0, 1, 1, 1, 4),))
    return study.StudyCaseV2(
        case_id,
        "generated",
        "tiny-conformance-v1",
        group,
        problem,
        objective,
        capture_hash=canonical_hash(
            (problem.source_identity, problem.target_identity, mapping.original_axioms)
        ),
        mapping_scores=(("mapping", 0.7),),
        external_objective=objective,
    )


def synchronous(function, *args, timeout, **kwargs):
    if timeout <= 0:
        return CallResult("timeout")
    try:
        return CallResult("complete", function(*args, **kwargs))
    except Exception as error:
        return CallResult("error", detail=str(error))


def verifier(problem, assignment):
    # A finite controlled label checks harness routing, not reasoner soundness.
    safe = not all(
        "keep" in obj.candidates[a].action_tags for obj, a in zip(problem.objects, assignment)
    )
    return VerificationReportV2(
        canonical_hash(assignment),
        canonical_hash(kernel.materialize(problem, assignment)),
        problem.policy.content_hash,
        "VERIFIED_FEASIBLE" if safe else "VERIFIED_INFEASIBLE",
        "complete_supported_fragment",
        (ObligationV2("controlled", "pass" if safe else "fail", True),),
    )


def incomplete_verifier(problem, assignment):
    return VerificationReportV2(
        canonical_hash(assignment),
        canonical_hash(kernel.materialize(problem, assignment)),
        problem.policy.content_hash,
        "VERIFIED_FEASIBLE",
        "partial_detection",
        (ObligationV2("controlled", "pass", False),),
    )


@pytest.fixture
def local_workers(monkeypatch):
    pytest.importorskip("pysat")
    monkeypatch.setattr(study, "bounded_call", synchronous)
    monkeypatch.setattr(kernel, "bounded_call", synchronous)
    monkeypatch.setattr(
        study, "runtime_manifest", lambda: {"code": "test-only-v1", "dependencies": "installed"}
    )


def test_action_arms_filter_identical_inventory_without_refilling_and_remap_pairs():
    case = frozen_case()
    previous = set()
    for language in ("no_repair", "deletion", "directional", "rich_mapping", "ontology_edits"):
        problem, objective, indices = study.filter_inventory(
            case, study.StudyArmV2(language, language, pairwise=True)
        )
        ids = {c.candidate_id for obj in problem.objects for c in obj.candidates}
        assert previous <= ids
        previous = ids
        assert problem.policy is case.problem.policy and problem.evidence == case.problem.evidence
        for old_obj, new_obj, selected in zip(case.problem.objects, problem.objects, indices):
            assert tuple(old_obj.candidates[i] for i in selected) == new_obj.candidates
        for i, a, j, b, value in objective.pairs:
            assert (i, indices[i][a], j, indices[j][b], value) in case.objective.pairs
    no_repair, _, _ = study.filter_inventory(case, study.DEFAULT_ARMS[0])
    assert all(len(obj.candidates) == 1 for obj in no_repair.objects)
    deletion, _, _ = study.filter_inventory(case, study.StudyArmV2("deletion", "deletion"))
    assert len(deletion.objects[0].candidates) == 2 and len(deletion.objects[1].candidates) == 1
    unary = study.filter_inventory(case, study.StudyArmV2("unary"))[1]
    assert not unary.pairs
    assert study.filter_inventory(case, study.StudyArmV2("pairwise", pairwise=True))[1].pairs


def test_exact_and_greedy_controls_share_verification_and_do_not_authorize_partial_results(
    local_workers,
):
    case = frozen_case()
    exact = study.evaluate_case(
        case, study.StudyArmV2("pairwise", pairwise=True), split="test", verifier=verifier
    )
    assert exact.status == "completed" and exact.result.logical_status == "VERIFIED_FEASIBLE"
    assert exact.result.search_status == "OPTIMAL_IN_POOL"
    assert exact.external_value == case.external_objective.score(exact.common_assignment)
    assert exact.exact_external_regret is None
    greedy_arm = study.StudyArmV2("greedy", "deletion", "score_greedy")
    greedy = study.evaluate_case(case, greedy_arm, split="test", verifier=verifier)
    assert greedy.result.logical_status == "VERIFIED_FEASIBLE" and greedy.result.checks == 2
    assert greedy.result.ontology_patch == ()
    partial = study.evaluate_case(case, greedy_arm, split="test", verifier=incomplete_verifier)
    assert partial.result.assignment is None and partial.result.logical_status == "UNKNOWN"
    assert partial.result.pending and not partial.result.exclusions
    assert dict(partial.resources)["elapsed_seconds"] >= 0


def test_missing_scores_models_and_failed_cases_remain_scheduled(local_workers):
    case = dataclasses.replace(frozen_case(), mapping_scores=())
    missing_score = study.evaluate_case(
        case,
        study.StudyArmV2("greedy", "deletion", "score_greedy"),
        split="test",
        verifier=verifier,
    )
    assert missing_score.status == "unavailable" and "score" in missing_score.detail
    missing_model = study.evaluate_case(
        case,
        study.StudyArmV2("hgt", objective_key="hgt-checkpoint-v1"),
        split="test",
        verifier=verifier,
    )
    assert missing_model.status == "unavailable"
    missing_case = study.StudyCaseV2(
        "missing",
        "bioml_2026_whole",
        "2026-explicit-release",
        "doid-ordo",
        availability="unavailable",
        detail="licensed input absent",
    )
    result = study.evaluate_case(
        missing_case, study.DEFAULT_ARMS[0], split="test", verifier=verifier
    )
    assert result.status == "unavailable" and result.result is None
    assert "licensed" in result.detail


def test_frozen_model_variants_and_complete_teacher_claim_are_explicit(local_workers):
    case = frozen_case()
    variant = dataclasses.replace(
        case.objective, unary=tuple(tuple(-v for v in row) for row in case.objective.unary)
    )
    case = dataclasses.replace(case, objective_variants=(("rgcn-frozen-checkpoint", variant),))
    _, objective, _ = study.filter_inventory(
        case, study.StudyArmV2("rgcn", objective_key="rgcn-frozen-checkpoint", pairwise=True)
    )
    assert objective == variant
    with pytest.raises(ValueError, match="complete teacher cache"):
        dataclasses.replace(case, complete_teacher_optimum=10)
    with pytest.raises(ValueError, match="preserve elementary"):
        study.StudyArmV2("bad", omitted_actions=("delete",))


def test_grouped_splits_hold_all_corruptions_and_matcher_outputs_together():
    first = frozen_case("first", "shared-parent")
    sibling = dataclasses.replace(first, case_id="renamed-corruption", capture_hash="other-capture")
    splits = study.grouped_splits((first, sibling), seed=51)
    assert splits[first.case_id] == splits[sibling.case_id]
    assert splits == study.grouped_splits((sibling, first), seed=51)


def test_atomic_schedule_resume_preserves_all_statuses_and_checks_artifacts(
    tmp_path, local_workers
):
    case = frozen_case()
    missing = study.StudyCaseV2(
        "unavailable",
        "conference_2025",
        "2025",
        "paper-pair",
        availability="unavailable",
        detail="not captured",
    )
    arms = (
        study.StudyArmV2("no_repair", "no_repair"),
        study.StudyArmV2("exact_pairwise", pairwise=True),
    )
    first = study.run_study((case, missing), tmp_path, arms=arms, seed=11, verifier=verifier)
    assert first["scheduled"] == first["recorded"] == 4
    assert sum(row["status"] == "unavailable" for row in first["rows"]) == 2
    contents = {path: path.read_bytes() for path in (tmp_path / "outcomes").glob("*.json")}
    second = study.run_study((case, missing), tmp_path, arms=arms, seed=11, verifier=verifier)
    assert first == second
    assert all(path.read_bytes() == value for path, value in contents.items())
    assert all(
        isinstance(read_record(json.loads(value)), study.StudyOutcomeV2)
        for value in contents.values()
    )
    with pytest.raises(ValueError, match="resume plan differs"):
        study.run_study((case, missing), tmp_path, arms=arms, seed=12, verifier=verifier)
    path = next(iter(contents))
    payload = json.loads(path.read_text())
    payload["hash"] = "tampered"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="hash mismatch"):
        study.run_study((case, missing), tmp_path, arms=arms, seed=11, verifier=verifier)


def test_schedule_loading_records_missing_and_corrupt_cases_instead_of_dropping(tmp_path):
    case = frozen_case()
    write_artifact(tmp_path / "case.json", case.to_dict())
    entries = [
        {
            "case_id": case.case_id,
            "cohort": case.cohort,
            "source_version": case.source_version,
            "group_id": case.group_id,
            "artifact": "case.json",
        },
        {
            "case_id": "missing",
            "cohort": "conference_2025",
            "source_version": "2025",
            "group_id": "pair-1",
            "artifact": "missing.json",
        },
    ]
    path = tmp_path / "schedule.json"
    path.write_text(
        json.dumps(
            {
                "schema": "exact-repair/study-schedule/v2",
                "cases": entries,
                "arms": [{"arm_id": "no_repair", "language": "no_repair"}],
                "seed": 2,
            }
        )
    )
    cases, arms, seed = study.load_schedule(path)
    assert len(cases) == 2 and cases[0] == case and cases[1].availability == "unavailable"
    assert arms[0].language == "no_repair" and seed == 2


def test_same_parent_stays_together_across_cohorts_and_fixed_cost_uses_actual_deletions():
    generated = frozen_case("generated", "same-parent")
    adapted = dataclasses.replace(generated, case_id="adapted", cohort="real_structure")
    splits = study.grouped_splits((generated, adapted), seed=22)
    assert splits["generated"] == splits["adapted"]
    problem, objective, _ = study.filter_inventory(
        generated, study.StudyArmV2("fixed", "deletion", objective_key="fixed_cost")
    )
    for obj, row in zip(problem.objects, objective.unary):
        for candidate, value in zip(obj.candidates, row):
            assert value == (-generated.objective.scale if not candidate.axioms else 0)


def test_interrupted_attempts_have_a_finite_resume_limit(tmp_path, local_workers):
    case = frozen_case()
    arm = study.StudyArmV2("no_repair", "no_repair")
    result = study.run_study((case,), tmp_path, arms=(arm,), verifier=verifier, max_attempts=1)
    outcome_path = tmp_path / result["rows"][0]["artifact"]
    outcome_path.unlink()
    resumed = study.run_study((case,), tmp_path, arms=(arm,), verifier=verifier, max_attempts=1)
    outcome = read_record(json.loads(outcome_path.read_text()))
    assert resumed["recorded"] == 1 and outcome.status == "failed"
    assert "attempt limit" in outcome.detail


def test_cli_missing_scheduled_case_is_recorded_without_running_experiments(tmp_path, capsys):
    from tools.repair.run_study import main

    schedule = tmp_path / "schedule.json"
    schedule.write_text(
        json.dumps(
            {
                "schema": "exact-repair/study-schedule/v2",
                "cases": [
                    {
                        "case_id": "not-provided",
                        "cohort": "conference_2025",
                        "source_version": "2025",
                        "group_id": "held-out-pair",
                        "artifact": "missing.json",
                    }
                ],
                "arms": [{"arm_id": "no_repair", "language": "no_repair"}],
            }
        )
    )
    assert main([str(schedule), "--output", str(tmp_path / "outputs")]) == 0
    assert json.loads(capsys.readouterr().out)["recorded"] == 1
    assert json.loads((tmp_path / "outputs" / "loading.json").read_text())["loading_seconds"] >= 0
    rows = json.loads((tmp_path / "outputs" / "results.json").read_text())["rows"]
    assert rows[0]["status"] == "unavailable" and rows[0]["logical_status"] == "UNKNOWN"
