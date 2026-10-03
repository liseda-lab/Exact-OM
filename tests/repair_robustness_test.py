"""Paired robustness boundaries and native exclusion after every producer."""

import dataclasses
import json
from pathlib import Path

import pytest

from exact.repair.records import canonical_hash, read_record
from exact.repair.workers import bounded_call
from tools.repair import fresh_evaluation as fresh
from tools.repair import robustness as robust
from tools.repair.corpus import generate_corpus
from tools.repair.prepare import case_to_dict, load_preparation

pytest_plugins = ["tests.repair_prepare_controls_test"]


def test_noise_and_missingness_preserve_theory_and_grouped_pairing(prepared):
    source, protocol_path = prepared
    cases, _, _ = load_preparation(source)
    problem = cases[0].problem
    protocol = fresh.read(protocol_path)
    for condition in robust.CONDITIONS[1:3]:
        first, manifest = robust.intervention(problem, "parent", condition, protocol)
        again, repeated = robust.intervention(problem, "parent", condition, protocol)
        assert first == again and manifest == repeated
        assert manifest["applicable"] and first.evidence != problem.evidence
        assert dataclasses.replace(first, evidence=problem.evidence) == problem
        other, _ = robust.intervention(problem, "different-parent", condition, protocol)
        assert other.evidence != first.evidence
        assert robust.generation_overrides({"intervention": manifest}, first) == {
            "final_candidate_removals": {},
            "omitted_generation_symbols": (),
        }
        with pytest.raises(ValueError, match="observable changed"):
            robust.generation_overrides({"intervention": manifest}, problem)


def test_removal_is_not_an_observable_feature_and_never_removes_keep_delete(prepared):
    source, protocol = prepared
    cases, _, _ = load_preparation(source)
    problem = cases[0].problem
    unchanged, manifest = robust.intervention(
        problem, "parent", robust.CONDITIONS[3], fresh.read(protocol)
    )
    assert unchanged == problem and manifest["applicable"]
    for obj in problem.objects:
        targets = set(manifest["final_candidate_removals"].get(obj.object_id, ()))
        assert targets
        assert all(
            c.axioms and (set(c.axioms) != set(obj.original_axioms) or c.active_expressions)
            for c in obj.candidates
            if c.candidate_id in targets
        )
    with pytest.raises(ValueError, match="exclusion violated"):
        robust.audit_final_pool(problem, {"intervention": manifest})
    assert "final_candidate_removals" not in json.dumps(problem.to_dict())


def test_empty_omission_targets_are_explicit_noops(prepared):
    source, protocol = prepared
    case = load_preparation(source)[0][0]
    problem = dataclasses.replace(
        case.problem,
        objects=tuple(
            dataclasses.replace(
                o,
                candidates=tuple(
                    c
                    for c in o.candidates
                    if not c.axioms
                    or (set(c.axioms) == set(o.original_axioms) and not c.active_expressions)
                ),
            )
            for o in case.problem.objects
        ),
    )
    same, manifest = robust.intervention(
        problem, "group", robust.CONDITIONS[3], fresh.read(protocol)
    )
    assert same == problem and not manifest["applicable"]
    assert robust.audit_final_pool(same, {"intervention": manifest})["removal_target_count"] == 0


def test_schedule_keeps_missing_rows_splits_models_and_exact_pair_denominators(
    prepared, tmp_path, monkeypatch
):
    source, protocol = prepared
    original = load_preparation(source)[0][0]
    item = dict(
        case_id=original.case_id,
        structural_parent=original.structural_parent,
        group_id="original-group",
        family=original.family,
        status="materialized",
        split="test",
        observable=fresh.immutable(tmp_path / "original.json", original.problem.to_dict()),
        evaluator=fresh.immutable(tmp_path / "sealed.json", case_to_dict(original)),
    )
    arms = [
        dict(id=str(i), kind="learned" if i < 6 else "control", protocol=fresh.binding(protocol))
        for i in range(9)
    ]
    base = dict(
        cases=[
            item,
            dict(case_id="missing", status="unavailable", group_id="missing-group", split="test"),
        ],
        arms=arms,
    )

    def prepared_base(campaign, selection, attempt, destination):
        fresh.immutable(destination, base)
        return base

    monkeypatch.setattr(fresh, "prepare", prepared_base)
    result = robust.prepare(tmp_path, {}, {}, tmp_path / "schedule")
    assert len(result["cases"]) == 10 and len(result["rows"]) == 90
    assert len({r["id"] for r in result["rows"]}) == 90
    assert sum(c["status"] == "unavailable" for c in result["cases"]) == 5
    assert all(
        c["group_id"] == "original-group"
        and c["split"] == "test"
        and c["evaluator"] == item["evaluator"]
        for c in result["cases"][:5]
    )
    assert result["arms"] == arms
    assert all(
        (r["seconds"], r["cpu_seconds"], r["memory_mb"]) == (300, 600, 8192) for r in result["rows"]
    )
    assert all(c["base_case_id"] == original.case_id for c in result["cases"][:5])


@pytest.mark.parametrize("condition", robust.CONDITIONS)
def test_native_uniform_robustness_pipeline(prepared, tmp_path, condition):
    source, protocol_path = prepared
    case = next(c for c in load_preparation(source)[0] if c.split == "development")
    problem, manifest = robust.intervention(
        case.problem, case.structural_parent, condition, fresh.read(protocol_path)
    )
    item = dict(
        case_id=case.case_id + ":" + condition[0],
        base_case_id=case.case_id,
        structural_parent=case.structural_parent,
        group_id=case.structural_parent,
        family=case.family,
        family_exposure="seen_family",
        status="materialized",
        condition=condition[0],
        intervention=manifest,
        observable=fresh.immutable(tmp_path / "input.json", problem.to_dict()),
        evaluator=fresh.immutable(tmp_path / "evaluator.json", case_to_dict(case)),
    )
    arms = [
        dict(
            id="uniform", kind="control", status="available", protocol=fresh.binding(protocol_path)
        )
    ]
    schedule = dict(cases=[item], arms=arms)
    row = fresh.make_rows([item], arms)[0]
    call = bounded_call(
        fresh.evaluate_row,
        schedule,
        row,
        str(tmp_path / "run"),
        timeout=300,
        cpu_seconds=600,
        memory_mb=8192,
    )
    assert call.status == "complete", call.detail
    result = fresh.bound(call.value)
    assert result["status"] == "evaluated", result
    assert result["condition"] == condition[0] and result["base_case_id"] == case.case_id
    assert result["intervention_audit"]["final_exclusion_verified"]
    pool = fresh.read(tmp_path / "run/pool.json")
    robust.audit_final_pool(read_record(pool["input"]), item)
    assert read_record(pool["objective"]).pool_hash == canonical_hash(
        read_record(pool["input"]).objects
    )
    assert fresh.bound(item["evaluator"]) == case_to_dict(case)


def test_learned_checkpoint_uses_same_final_exclusion(prepared, tmp_path):
    import torch
    from exact.repair.graph import EffectivePreparation
    from exact.repair.model import RepairModel
    from exact.repair.retrieval import retrieve_vocabulary
    from exact.repair.pipeline import bounded_freeze_checkpoint

    source, protocol_path = prepared
    case = load_preparation(source)[0][0]
    protocol = fresh.read(protocol_path)
    problem, manifest = robust.intervention(
        case.problem, case.structural_parent, robust.CONDITIONS[3], protocol
    )
    options = fresh.pilot.generation_options(protocol, tmp_path / "cache")
    prep = EffectivePreparation(
        max_graph_nodes=options["max_graph_nodes"],
        max_graph_edges=options["max_graph_edges"],
        max_explanations=options["max_explanations"],
        max_text_tokens=options["max_text_tokens"],
        pair_factor_limit_per_object=options["pair_factor_limit_per_object"],
        pair_max_pairs=options["pair_max_pairs"],
        pair_max_factors=options["pair_max_factors"],
        retrieval_config=options["retrieval_config"],
        revision="v3",
    )
    graph = prep.graph(problem, retrieve_vocabulary(problem, config=options["retrieval_config"]))
    model = RepairModel(
        graph.metadata,
        encoder="none",
        hidden_dim=8,
        layers=0,
        heads=2,
        dropout=0,
        pairwise=False,
        plan_risk=False,
        revision="v3",
    )
    path = tmp_path / "model.pt"
    torch.save(
        dict(
            metadata=model.metadata,
            config=model.config,
            model_schema="exact-repair/model/v3",
            state_dict=model.state_dict(),
        ),
        path,
    )
    digest = fresh.sha(path)
    call = bounded_freeze_checkpoint(
        problem,
        path,
        checkpoint_sha256=digest,
        seconds=60,
        memory_mb=8192,
        cpu_seconds=600,
        **options,
        **robust.generation_overrides({"intervention": manifest}, problem)
    )
    assert call.status == "complete", call.detail
    assert robust.audit_final_pool(call.value.problem, {"intervention": manifest})[
        "final_exclusion_verified"
    ]
    assert fresh.sha(path) == digest
