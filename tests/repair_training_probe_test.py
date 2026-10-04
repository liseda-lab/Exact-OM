"""Probe admission, honest gradients and observable-only execution."""

import json
from pathlib import Path

import pytest
import torch

from exact.repair.graph_schema import generic_graph_schema
from tools.repair.historical_regression import binding
from tools.repair.training_probe import ARMS, gradient_summary, probe, schedule


def plan_for(tmp_path):
    release = tmp_path / "release.json"
    rows = [
        dict(
            case_id=str(i),
            structural_parent=str(i // 2),
            family="fixture",
            control="corrupted",
            input_hash="unused",
            observable={"path": "unused"},
            split="development",
            status="materialized",
            evaluator={"path": "must-never-open"},
        )
        for i in range(32)
    ]
    release.write_text(json.dumps(dict(rows=rows)))
    return dict(
        schema="exact-repair/training-probe-plan/v1",
        split="development",
        expected_cases=32,
        heldout_use=False,
        model_fitting=False,
        arms=list(ARMS),
        release=binding(release),
    )


def test_schedule_strips_evaluator_and_preserves_all_rows(tmp_path):
    rows = schedule(plan_for(tmp_path))
    assert len(rows) == 32
    assert all("evaluator" not in row and "split" not in row for row in rows)


@pytest.mark.parametrize(
    "field,value",
    [
        ("split", "test"),
        ("heldout_use", True),
        ("model_fitting", True),
        ("expected_cases", 31),
        ("arms", ["none-unary"]),
    ],
)
def test_reject_boundary_before_loading_release(tmp_path, field, value):
    plan = plan_for(tmp_path)
    plan[field] = value
    plan["release"] = {"path": "never-open"}
    with pytest.raises(ValueError, match="boundary"):
        schedule(plan)


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "cross_split", "unknown", "tampered"])
def test_release_admission(tmp_path, mutation):
    plan = plan_for(tmp_path)
    path = tmp_path / "release.json"
    value = json.loads(path.read_text())
    if mutation == "duplicate":
        value["rows"][1]["case_id"] = "0"
    if mutation == "missing":
        value["rows"].pop()
    if mutation == "cross_split":
        value["rows"][0]["split"] = "test"
    if mutation == "unknown":
        value["rows"][0]["status"] = "unavailable"
    path.write_text(json.dumps(value) + "\n")
    if mutation != "tampered":
        plan["release"] = binding(path)
    with pytest.raises(ValueError):
        schedule(plan)


def test_gradient_probe_rejects_nonfinite_and_zero():
    model = torch.nn.Linear(2, 1)
    with pytest.raises(ValueError, match="zero"):
        gradient_summary(model, model.weight.sum() * 0)
    with pytest.raises(ValueError, match="Nonfinite"):
        gradient_summary(model, model.weight.sum() * float("nan"))
    result = gradient_summary(model, model.weight.square().sum())
    assert result["nonzero_parameters"] == ["weight"]
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize(
    "encoder,head",
    [("none", "unary"), ("none", "pairwise"), ("rgcn", "pairwise"), ("hgt", "pairwise")],
)
def test_actual_full_schema_proposal_backward_without_teacher_or_optimizer(
    tmp_path, monkeypatch, encoder, head
):
    from tools.repair.corpus import generate_corpus
    from tools.repair import train

    cases = generate_corpus(
        split_counts={"train": 0, "development": 1, "test": 0},
        siblings_per_parent=1,
        families=("overlap",),
        revision="v3",
    )
    monkeypatch.setattr(train, "label_case", lambda *a, **kw: pytest.fail("No teacher allowed"))
    monkeypatch.setattr(
        torch.optim.AdamW, "step", lambda *a, **kw: pytest.fail("No fitting allowed")
    )
    settings = dict(
        threads=1,
        seed=13,
        device="cpu",
        graph={},
        retrieval={},
        model=dict(hidden_dim=8, layers=1, heads=2, dropout=0, revision="v3"),
        proposal_seconds=60,
        compile_seconds=30,
        grammar=dict(max_depth=1, max_constructors=1),
        proposal=dict(mixtures=2, max_circuit_nodes=100000),
    )
    result = probe(
        cases[0].problem.to_dict(),
        settings,
        generic_graph_schema(),
        encoder + "-" + head,
        tmp_path,
    )
    assert result["readout_gradients"]["nonzero_parameters"]
    assert result["status"] == "complete"
    if head == "pairwise":
        assert result["selected_pairs"] > 0 and result["pair_factors"] > 0
    else:
        assert result["selected_pairs"] == result["pair_factors"] == 0
        # Disabled interactions are deliberate, even when the observable graph links objects.
        assert any(count > 0 for _, count in result["pair_omissions"])
    assert result["optimizer_steps"] == result["checkpoints_written"] == 0
    assert all(row["status"] == "complete" for row in result["proposals"])
    assert all(row["gradients"]["nonzero_parameters"] for row in result["proposals"])
    assert any(
        "proposal" in name
        for row in result["proposals"]
        for name in row["gradients"]["nonzero_parameters"]
    )


def test_resume_preserves_timeout_denominator_and_refuses_pending_owner(tmp_path, monkeypatch):
    from tools.repair import numerical_profile, training_probe
    from exact.repair.workers import CallResult
    from tools.repair.corpus import generate_corpus
    from exact.repair.api import write_artifact
    import exact.repair.workers as workers

    case = generate_corpus(
        split_counts={"train": 0, "development": 1, "test": 0},
        siblings_per_parent=1,
        families=("overlap",),
        revision="v3",
    )[0]
    observable = tmp_path / "observable.json"
    write_artifact(observable, case.problem.to_dict())
    declaration = tmp_path / "declaration.json"
    write_artifact(declaration, generic_graph_schema())
    plan = plan_for(tmp_path)
    release = json.loads((tmp_path / "release.json").read_text())
    for row in release["rows"]:
        row.update(observable=binding(observable), input_hash=case.problem.content_hash)
    write_artifact(tmp_path / "release.json", release)
    plan.update(
        release=binding(tmp_path / "release.json"),
        declaration=binding(declaration),
        settings={},
        row_budget={"timeout": 1},
    )
    plan_path = tmp_path / "plan.json"
    write_artifact(plan_path, plan)
    monkeypatch.setattr(numerical_profile, "probe_identity", lambda *a: "runtime")
    calls = []

    def timeout(*args, **kwargs):
        calls.append(args)
        return CallResult("timeout", detail="bounded fixture", cleanup_complete=True)

    monkeypatch.setattr(workers, "bounded_call", timeout)
    report = training_probe.run(plan_path, "none-unary", tmp_path / "output")
    assert len(calls) == report["recorded_rows"] == report["scheduled_rows"] == 32
    training_probe.run(plan_path, "none-unary", tmp_path / "output")
    assert len(calls) == 32
    # A saved intent without a settled row must never receive another budget.
    first = report["rows"][0]
    Path(first["path"]).unlink()
    with pytest.raises(RuntimeError, match="prior-owner"):
        training_probe.run(plan_path, "none-unary", tmp_path / "output")
    assert len(calls) == 32


@pytest.mark.parametrize("arm", ["hgt-unary", "hgt-pairwise"])
def test_probe_omission_interpretation_preserves_actual_missingness(arm):
    from copy import deepcopy
    from tools.repair.training_probe import probe_status

    result = dict(arm=arm, omitted_nodes=0, omitted_supports=0, omitted_evidence=[],
                  support_omissions=[], pair_omissions=[["fixed_axiom", 0]],
                  proposals=[dict(status="complete", missing_candidates=0)])
    original = deepcopy(result)
    assert probe_status(result) == "complete"
    assert result == original
    result["pair_omissions"] = [["fixed_axiom", 3]]
    assert probe_status(result) == ("partial" if arm.endswith("-pairwise") else "complete")
    for key, value in [("omitted_nodes", 1), ("omitted_supports", 1),
                       ("omitted_evidence", ["missing"]),
                       ("support_omissions", [["support", "graph_budget"]]),
                       ("proposals", [dict(status="proposal_deadline")]),
                       ("proposals", [dict(status="complete", missing_candidates=1)])]:
        missing = {**original, key: value}
        assert probe_status(missing) == "partial"
