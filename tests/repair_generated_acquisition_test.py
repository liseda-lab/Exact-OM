from dataclasses import replace, fields
from types import SimpleNamespace

import pytest

from exact.repair.api import write_artifact
from exact.repair.learning import RepairLabel
from exact.repair.records import (
    make_objective,
    FrozenMapping,
    ProposalRecordV3,
    ReplacementCandidateV3,
)
from exact.repair.workers import CallResult
from tools.repair import generated_acquisition as ga
from tools.repair.expanded_profile import parent_case, checked_checkpoint
from tools.repair.historical_regression import binding
from tools.repair.prepare import case_to_dict, read_label_cache


def fixture(tmp_path, monkeypatch):
    from exact.repair import pipeline
    from tools.repair import train, evaluate_campaign

    case = parent_case("overlap", 2, 13, split="train")
    obj = case.problem.objects[0]
    # Extra canonical candidate: acquisition must label the generated inventory.
    candidate = replace(obj.candidates[-1], candidate_id="generated-new-candidate")
    generated = replace(
        case.problem,
        objects=(replace(obj, candidates=obj.candidates + (candidate,)),)
        + case.problem.objects[1:],
    )
    objective = make_objective(
        generated.objects,
        tuple(tuple(float(i) for i in range(len(o.candidates))) for o in generated.objects),
    )
    sample_candidate = ReplacementCandidateV3(
        **{f.name: getattr(obj.candidates[0], f.name) for f in fields(obj.candidates[0])}
    )
    sample = ProposalRecordV3(
        dict(
            schema="exact-repair/proposal/v3",
            object_id=obj.object_id,
            candidate_id=sample_candidate.candidate_id,
            candidate=sample_candidate,
            assignment=(True,),
            component=0,
            log_probability=-1.0,
            grammar_hash="0" * 64,
            circuit_hash=None,
            retrieval_hash="0" * 64,
            context_hash="0" * 64,
            seed=13,
            family="fixture",
            probability_semantics="conditioned_bundle_probability",
            distribution_scope="declared_language",
        )
    )
    calls = []
    clock = [1000.0]
    monkeypatch.setattr(ga.time, "time", lambda: clock[0])
    monkeypatch.setattr(ga, "_pairs", lambda *args: ())
    monkeypatch.setattr(train, "_verify_intended", lambda *args, **kwargs: True)

    def label(case, assignment, *args, **kwargs):
        calls.append(assignment)
        assert case.problem.content_hash == generated.content_hash
        return RepairLabel(assignment, True, 1.0, 0.0)

    monkeypatch.setattr(train, "_assignment_label", label)
    monkeypatch.setattr(
        evaluate_campaign, "generation_options", lambda *args: dict(compile_seconds=20)
    )
    limits = []

    def bounded(function, *args, timeout, **kwargs):
        limits.append(timeout)
        clock[0] += 1
        kwargs.pop("memory_mb", None)
        kwargs.pop("cpu_seconds", None)
        return CallResult(
            "complete", function(*args, **kwargs), resource_usage=(("cpu_seconds", 1.0),)
        )

    monkeypatch.setattr(ga, "bounded_call", bounded)

    def freeze(*args, seconds, **kwargs):
        limits.append(seconds)
        clock[0] += 1
        return CallResult(
            "complete",
            SimpleNamespace(
                problem=generated,
                objective=objective,
                proposal_reports=(
                    dict(
                        samples=(sample,),
                        nested=FrozenMapping({"bound": True}),
                    ),
                ),
                model_hash="shared-untrained",
            ),
            resource_usage=(("cpu_seconds", 1.0),),
        )

    monkeypatch.setattr(pipeline, "bounded_freeze_checkpoint", freeze)
    plan = dict(
        resources=dict(
            case_seconds=300,
            cleanup_seconds=2,
            cpu_seconds=600,
            memory_mb=20480,
            initial_parent_seconds=20,
            generation_seconds=60,
            compile_seconds=20,
            full_check_seconds=30,
        ),
        checkpoint=dict(path="unused", sha256="unused"),
        seed=13,
        round_id="0",
    )
    protocol = dict(
        objective=dict(edit_weights={}), teacher=dict(family_weights=dict(desired=1, unwanted=1))
    )
    return case, generated, plan, protocol, calls, clock, limits


def test_generated_labels_shared_cache_and_completed_resume(tmp_path, monkeypatch):
    case, generated, plan, protocol, calls, clock, limits = fixture(tmp_path, monkeypatch)
    result = ga.case_worker(case_to_dict(case), plan, protocol, tmp_path, "same", 1300)
    cache = read_label_cache(result["cache"], tmp_path / "cache", replace(case, problem=generated))
    assert result["requested"] == 16
    assert any(
        label.assignment[0] == len(generated.objects[0].candidates) - 1 for label in cache.labels
    )
    assert len(calls) == len(set(calls))
    assert result["cpu_seconds"] >= len(calls) + 2
    count = len(calls)
    again = ga.case_worker(case_to_dict(case), plan, protocol, tmp_path, "same", 1300)
    assert len(calls) == count
    assert again["cache"]["sha256"] == result["cache"]["sha256"]
    with pytest.raises(ValueError, match="dependencies"):
        ga.case_worker(case_to_dict(case), plan, protocol, tmp_path, "other", 1300)


def test_every_nested_call_clips_to_one_case_and_stage_clock(tmp_path, monkeypatch):
    case, _, plan, protocol, calls, clock, limits = fixture(tmp_path, monkeypatch)
    result = ga.case_worker(case_to_dict(case), plan, protocol, tmp_path, "same", 1009)
    assert limits[0] <= 7 and max(limits) <= 7
    assert clock[0] <= 1009
    state = checked_checkpoint(tmp_path / "state.json", "same")
    assert state["case_deadline_epoch"] == 1009
    before = len(calls)
    clock[0] = 2000
    ga.case_worker(case_to_dict(case), plan, protocol, tmp_path, "same", 3000)
    assert len(calls) == before


def test_resume_partial_collection_reuses_committed_native_labels(tmp_path, monkeypatch):
    case, generated, plan, protocol, calls, clock, limits = fixture(tmp_path, monkeypatch)
    original = ga.collect_sampled_repairs

    def interrupt(*args, **kwargs):
        progress = kwargs["progress"]

        def save(state):
            progress(state)
            if len(state["labels"]) == 2:
                raise InterruptedError("controller interruption")

        kwargs["progress"] = save
        return original(*args, **kwargs)

    monkeypatch.setattr(ga, "collect_sampled_repairs", interrupt)
    with pytest.raises(InterruptedError):
        ga.case_worker(case_to_dict(case), plan, protocol, tmp_path, "same", 1300)
    before = checked_checkpoint(tmp_path / "state.json", "same")
    old_labels = set(calls)
    monkeypatch.setattr(ga, "collect_sampled_repairs", original)
    result = ga.case_worker(case_to_dict(case), plan, protocol, tmp_path, "same", 1300)
    assert result["cpu_seconds"] >= before["cpu_seconds"]
    assert len(calls) == len(set(calls)) and old_labels <= set(calls)
    assert result["labels"] == len(calls)


def test_cleanup_failure_is_not_scientific_unknown(tmp_path, monkeypatch):
    case, _, plan, protocol, *_ = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        ga, "bounded_call", lambda *args, **kwargs: CallResult("timeout", cleanup_complete=False)
    )
    for _ in range(2):
        with pytest.raises(RuntimeError, match="cleanup"):
            ga.case_worker(case_to_dict(case), plan, protocol, tmp_path, "same", 1300)


def test_unknown_delivery_is_charged_and_never_repeated(tmp_path, monkeypatch):
    case, _, plan, protocol, calls, clock, limits = fixture(tmp_path, monkeypatch)
    marker = tmp_path / "calls/parent.started.json"
    write_artifact(marker, dict(cpu_reservation=20, started_epoch=999))
    result = ga.case_worker(case_to_dict(case), plan, protocol, tmp_path, "same", 1300)
    assert not calls and not limits
    assert result["status"] == "unknown_intended_parent"
    state = checked_checkpoint(tmp_path / "state.json", "same")
    assert state["cpu_seconds"] == 20
    ga.case_worker(case_to_dict(case), plan, protocol, tmp_path, "same", 1300)
    assert checked_checkpoint(tmp_path / "state.json", "same")["cpu_seconds"] == 20


def test_schedule_preserves_both_denominators_and_rejects_test(tmp_path, monkeypatch):
    from tests.repair_training_acquisition_test import schedule

    train = schedule(tmp_path / "train", "train", 128)
    dev = schedule(tmp_path / "dev", "development", 32)
    index = dict(
        training_manifest=train["training_manifest"],
        development_manifest=dev["development_manifest"],
        training_rows=ga.verify_binding(train["training_manifest"])["rows"],
        development_rows=ga.verify_binding(dev["development_manifest"])["rows"],
    )
    # The independent fixture parents and IDs are synthetic; make DEV distinct.
    for row in index["development_rows"]:
        row["case_id"] = "dev-" + row["case_id"]
        row["structural_parent"] = "dev-" + row["structural_parent"]
    write_artifact(tmp_path / "dev/manifest.json", dict(rows=index["development_rows"]))
    index["development_manifest"] = binding(tmp_path / "dev/manifest.json")
    write_artifact(tmp_path / "index.json", index)
    write_artifact(tmp_path / "protocol.json", {})
    (tmp_path / "model").write_bytes(b"model")
    plan = dict(
        schema=ga.SCHEMA,
        heldout_outcomes_opened=False,
        model_updates=0,
        round_id="0",
        shared_conditions=["symbolic", "symbolic_plus_llm"],
        plan_quotas=ga.QUOTAS,
        inputs=binding(tmp_path / "index.json"),
        protocol=binding(tmp_path / "protocol.json"),
        checkpoint=binding(tmp_path / "model"),
        case_ids=["fixture-0"],
    )
    assert len(ga.validate(plan)[0]) == 1
    with pytest.raises(ValueError, match="TRAIN"):
        ga.validate(dict(plan, case_ids=["test-case"]))
    index["development_rows"].pop()
    write_artifact(tmp_path / "index.json", index)
    with pytest.raises(ValueError, match="denominator"):
        ga.validate(dict(plan, inputs=binding(tmp_path / "index.json")))
