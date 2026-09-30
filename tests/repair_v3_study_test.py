"""Matched baseline accounting and strict v3 study evidence; no benchmark jobs."""

import dataclasses
import json

import pytest

from exact.repair import kernel, study
from exact.repair.records import (
    BaselineReportV3,
    ObjectiveV3,
    ObligationV2,
    RepairResultV3,
    VerificationReportV2,
    VerificationReportV3,
    canonical_hash,
    promote_input_v3,
    read_record,
)
from tests.repair_study_test import frozen_case, synchronous


def modern_case():
    case = frozen_case()
    problem = promote_input_v3(case.problem)
    objective = ObjectiveV3(
        **{f.name: getattr(case.objective, f.name) for f in dataclasses.fields(case.objective)},
        pool_hash=canonical_hash(problem.objects),
        model_hash="fixture-model",
        target_basis="symbolic-semantic-vector/v3"
    )
    return dataclasses.replace(case, problem=problem, objective=objective)


def report(problem, assignment, *, safe=None):
    if safe is None:
        safe = not all(
            "keep" in obj.candidates[a].action_tags for obj, a in zip(problem.objects, assignment)
        )
    theory, active = kernel.materialize(problem, assignment)
    expected = kernel.expected_queries(active, problem.policy)
    return VerificationReportV3(
        canonical_hash(assignment),
        canonical_hash((theory, active)),
        problem.policy.content_hash,
        "VERIFIED_FEASIBLE" if safe else "VERIFIED_INFEASIBLE",
        "complete_supported_fragment",
        tuple(
            ObligationV2(name, "fail" if not safe and i == 0 else "pass", True)
            for i, name in enumerate(expected)
        ),
        expected_obligations=expected,
        backend="controlled-local-test",
    )


def shared_baseline(problem):
    original = tuple(
        next(i for i, c in enumerate(o.candidates) if "keep" in c.action_tags)
        for o in problem.objects
    )
    alignment = report(problem, original)
    return BaselineReportV3(
        kernel.baseline_identity(problem),
        tuple((scope, alignment) for scope in ("source", "target", "union", "alignment")),
        elapsed_seconds=3.5,
        checks=4,
    )


@pytest.fixture
def local(monkeypatch):
    pytest.importorskip("pysat")
    monkeypatch.setattr(study, "bounded_call", synchronous)
    monkeypatch.setattr(kernel, "bounded_call", synchronous)
    monkeypatch.setattr(study, "runtime_manifest", lambda: {"code": "test-v3"})


def test_four_baselines_are_reused_by_greedy_exact_and_no_search(local):
    case = modern_case()
    baseline = shared_baseline(case.problem)
    arms = (
        study.StudyArmV2("no-repair", "no_repair"),
        study.StudyArmV2("greedy", "deletion", "score_greedy"),
        study.StudyArmV2("exact", "deletion"),
    )
    outcomes = [
        study.evaluate_case(case, arm, split="test", verifier=report, baseline_evidence=baseline)
        for arm in arms
    ]
    for outcome in outcomes:
        assert outcome.status == "completed", outcome.detail
        assert outcome.baseline_hash == baseline.content_hash
        assert outcome.result.baseline == baseline.reports
        assert dict(outcome.resources)["shared_baseline_checks"] == 4
        assert dict(outcome.resources)["shared_baseline_seconds"] == 3.5
        assert outcome.schema_version.endswith("/v3")
        assert isinstance(outcome.result, RepairResultV3)
        assert read_record(outcome.to_dict()) == outcome
    assert outcomes[0].result.assignment is None and outcomes[0].result.checks == 0
    assert outcomes[1].result.checks == 1  # Original diagnosis was reused, not repeated.
    assert outcomes[2].result.logical_status == "VERIFIED_FEASIBLE"


def test_campaign_baseline_is_charged_once_and_resume_uses_same_record(
    local, monkeypatch, tmp_path
):
    case = modern_case()
    calls = []
    reservations = []
    original_begin = study.CumulativeBudget.begin

    def track_budget(self, seconds):
        reservations.append(seconds)
        return original_begin(self, seconds)

    def collect(problem):
        calls.append(problem)
        return shared_baseline(problem)

    monkeypatch.setattr(study, "verify_assignment", report)
    monkeypatch.setattr(study, "collect_baselines", collect)
    monkeypatch.setattr(study, "_verify_with_exceptions", lambda p, a, e: report(p, a))
    monkeypatch.setattr(study.CumulativeBudget, "begin", track_budget)
    arms = (
        study.StudyArmV2("no-repair", "no_repair"),
        study.StudyArmV2("greedy", "deletion", "score_greedy"),
        study.StudyArmV2("exact", "deletion"),
    )
    first = study.run_study((case,), tmp_path, arms=arms, verifier=report)
    assert len(calls) == 1 and len(reservations) == 4  # One preparation, three selectors.
    assert first["scheduled"] == first["recorded"] == 3
    assert len(list((tmp_path / "baselines").glob("*.json"))) == 1
    second = study.run_study((case,), tmp_path, arms=arms, verifier=report)
    assert first == second and len(calls) == 1 and len(reservations) == 4
    assert json.loads((tmp_path / "results.json").read_text())["schema"].endswith("/v3")


@pytest.mark.parametrize("kind", ["legacy_report", "missing_obligations"])
def test_greedy_cannot_accept_report_rejected_by_v3_kernel(local, kind):
    case = modern_case()

    def incomplete(problem, assignment):
        full = report(problem, assignment, safe=True)
        if kind == "legacy_report":
            return VerificationReportV2(
                **{f.name: getattr(full, f.name) for f in dataclasses.fields(VerificationReportV2)}
            )
        return dataclasses.replace(
            full,
            obligations=(full.obligations[0],),
            expected_obligations=(full.expected_obligations[0],),
        )

    outcome = study.evaluate_case(
        case,
        study.StudyArmV2("greedy", "deletion", "score_greedy"),
        split="test",
        verifier=incomplete,
    )
    assert outcome.status == "completed"
    assert outcome.result.assignment is None
    assert outcome.result.logical_status == "UNKNOWN" and outcome.result.pending


def test_failed_resume_and_missing_case_are_explicit_v3_outcomes(local, tmp_path):
    case = modern_case()
    arm = study.StudyArmV2("no-repair", "no_repair")
    first = study.run_study((case,), tmp_path, arms=(arm,), verifier=report, max_attempts=1)
    destination = tmp_path / first["rows"][0]["artifact"]
    destination.unlink()
    study.run_study((case,), tmp_path, arms=(arm,), verifier=report, max_attempts=1)
    exhausted = read_record(json.loads(destination.read_text()))
    assert exhausted.status == "failed" and exhausted.schema_version.endswith("/v3")
    missing = study.StudyCaseV2(
        "absent", "generated", "v3-test", "absent-parent", availability="unavailable"
    )
    absent = study.evaluate_case(missing, arm, split="test", verifier=report)
    assert absent.status == "unavailable" and absent.schema_version.endswith("/v3")
    assert read_record(absent.to_dict()) == absent


def test_wrong_policy_baseline_fails_all_arms_and_model_target_identity_is_retained(local):
    case = modern_case()
    wrong = dataclasses.replace(shared_baseline(case.problem), identity="other-policy")
    for arm in (
        study.StudyArmV2("greedy", "deletion", "score_greedy"),
        study.StudyArmV2("exact", "deletion"),
    ):
        outcome = study.evaluate_case(
            case, arm, split="test", verifier=report, baseline_evidence=wrong
        )
        assert outcome.status == "failed" and outcome.result is None
    weak = dataclasses.replace(
        case.objective, target_basis="llm-weak-semantic-fidelity/v3", model_hash="weak-model"
    )
    labeled = dataclasses.replace(case, objective_variants=(("weak", weak),))
    problem, filtered, _ = study.filter_inventory(
        labeled, study.StudyArmV2("weak", "deletion", objective_key="weak")
    )
    assert isinstance(filtered, ObjectiveV3)
    assert filtered.target_basis != case.objective.target_basis
    assert filtered.model_hash == "weak-model" and filtered.pool_hash == canonical_hash(
        problem.objects
    )


def test_campaign_exhaustion_uses_v3_record_without_evaluation(local, monkeypatch, tmp_path):
    class EmptyBudget:
        def __init__(self, *args):
            self.remaining = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(study, "CumulativeBudget", EmptyBudget)
    case = modern_case()
    result = study.run_study(
        (case,), tmp_path, arms=(study.StudyArmV2("none", "no_repair"),), verifier=report
    )
    artifact = read_record(json.loads((tmp_path / result["rows"][0]["artifact"]).read_text()))
    assert artifact.status == "timeout" and artifact.schema_version.endswith("/v3")


def test_malformed_authorizing_baseline_cannot_replace_required_query_coverage(local):
    case = modern_case()
    baseline = shared_baseline(case.problem)
    original = tuple(
        next(i for i, c in enumerate(o.candidates) if "keep" in c.action_tags)
        for o in case.problem.objects
    )
    good = report(case.problem, original, safe=True)
    partial = dataclasses.replace(
        good,
        obligations=(good.obligations[0],),
        expected_obligations=(good.expected_obligations[0],),
    )
    malformed = dataclasses.replace(baseline, reports=(("alignment", partial),))

    def always_unknown(problem, assignment):
        full = report(problem, assignment, safe=True)
        return dataclasses.replace(
            full,
            verdict="UNKNOWN",
            scope="partial_detection",
            obligations=(ObligationV2("unavailable", "unknown", False),),
            expected_obligations=(),
        )

    for arm in (
        study.StudyArmV2("greedy", "deletion", "score_greedy"),
        study.StudyArmV2("exact", "deletion"),
    ):
        outcome = study.evaluate_case(
            case, arm, split="test", verifier=always_unknown, baseline_evidence=malformed
        )
        assert outcome.status == "completed", outcome.detail
        assert outcome.result.assignment is None
