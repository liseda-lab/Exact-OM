"""Resource schedules retain failures, isolate development, and reject stale receipts."""

import dataclasses
from pathlib import Path

import pytest

from exact.repair.records import canonical_hash
from exact.repair.workers import CallResult
from tools.repair import numerical_profile as profile
from tools.repair.expanded_profile import MECHANISMS_SELECTED, parent_case
from tools.repair.prepare import case_to_dict


def settings():
    return dict(
        graph=dict(
            max_graph_nodes=4096,
            max_graph_edges=32768,
            max_explanations=64,
            max_text_tokens=128,
            pair_factor_limit_per_object=16,
            pair_max_pairs=128,
            pair_max_factors=65536,
            revision="v3",
        ),
        retrieval=dict(classes_per_side=8, properties_per_side=4, endpoints_per_side=8),
        model=dict(
            feature_dim=128,
            hidden_dim=128,
            layers=3,
            heads=4,
            dropout=0.1,
            revision="v3",
            plan_risk=True,
            support_enabled=False,
        ),
        teacher=dict(max_assignments=64, deadline_seconds=60, call_seconds=10),
        seed=13,
        threads=4,
        iterations=3,
        optimizer=dict(lr=0.001, weight_decay=0.0001),
    )


@pytest.mark.parametrize("family", MECHANISMS_SELECTED)
def test_graph_uses_observable_preparation_and_respects_mandatory_caps(family):
    record = case_to_dict(parent_case(family, 2, 13))
    case, graph, pairs, report = profile.prepare_graph(record, settings())
    assert case.split == "development"
    assert report["graph_hash"] == canonical_hash(graph)
    assert report["graph_nodes"] > 0
    assert report["feature_values"] == len(graph.nodes) * 128
    assert report["selected_pairs"] == len(pairs.pairs)
    assert not {"probes", "intended_theory", "labels"} & set(report)
    clipped = settings()
    clipped["graph"]["max_graph_nodes"] = 4
    with pytest.raises(ValueError, match="mandatory"):
        profile.prepare_graph(record, clipped)


def test_graph_refuses_heldout_and_legacy_inputs():
    case = parent_case("range", 2, 13)
    for changed in (
        dataclasses.replace(case, split="test"),
        dataclasses.replace(case, schema_revision="v2"),
    ):
        with pytest.raises(ValueError, match="development"):
            profile.prepare_graph(case_to_dict(changed), settings())


@pytest.mark.parametrize("arm", profile.ARMS)
def test_actual_diagnostic_readouts_have_finite_gradients(arm):
    import torch
    from exact.repair.model import RepairModel

    torch.set_num_threads(1)
    torch.manual_seed(13)
    record = case_to_dict(parent_case("overlap", 2, 13))
    case, graph, pairs, _ = profile.prepare_graph(record, settings())
    encoder, head = arm.split("-")
    model = RepairModel(
        graph.metadata,
        hidden_dim=8,
        layers=1,
        heads=2,
        encoder=encoder,
        pairwise=head == "pairwise",
        revision="v3",
    )
    loss, factor_count = profile.diagnostic_loss(
        model, case, graph, pairs.pairs if head == "pairwise" else ()
    )
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert torch.isfinite(loss).item() and gradients
    assert all(torch.isfinite(g).all().item() for g in gradients)
    assert any(torch.count_nonzero(g).item() for g in gradients)
    if head == "pairwise":
        assert factor_count > 0
        assert any(p.grad is not None for p in model.pair_head.parameters())
    else:
        assert factor_count == 0


def fixture_plan(tmp_path, monkeypatch):
    source = profile.immutable(tmp_path / "case.json", case_to_dict(parent_case("range", 2, 13)))
    cases = [
        dict(key=str(i), case=source, case_hash=profile.bound(source)["hash"]) for i in range(32)
    ]
    cases[-1]["case"] = None
    frozen = dict(cases=cases, native_profile=dict(scheduled=64, outcomes={"timeout/unknown": 4}))
    plan = dict(
        settings=settings(),
        budgets={
            stage: dict(seconds=30, memory_mb=4096) for stage in ("graph", "teacher", "numerical")
        },
    )
    plan_path, schedule_path = tmp_path / "plan.json", tmp_path / "schedule.json"
    profile.immutable(plan_path, plan)
    profile.immutable(schedule_path, frozen)
    monkeypatch.setattr(profile, "load_schedule", lambda *args: (plan, frozen))
    monkeypatch.setattr(profile, "probe_identity", lambda *args: "fixed-identity")
    return plan_path, schedule_path, frozen


def test_full_denominators_and_resume_do_not_requery(tmp_path, monkeypatch):
    plan, schedule, frozen = fixture_plan(tmp_path, monkeypatch)
    calls = []

    def invoke(*args, **kwargs):
        calls.append(args)
        return CallResult("timeout", detail="recorded native timeout")

    monkeypatch.setattr("exact.repair.workers.bounded_call", invoke)
    outputs = {stage: str(tmp_path / stage) for stage in profile.STAGES}
    for stage in profile.STAGES:
        report = profile.run_stage(plan, schedule, stage, outputs[stage], outputs["graph"])
        assert report["scheduled"] == 32
        assert len(report["rows"]) == 32
        assert profile.run_stage(plan, schedule, stage, outputs[stage], outputs["graph"]) == report
    # Graph and teacher each try 31 present cases. Six GPU arms retain unavailable graphs.
    assert len(calls) == 62
    result = profile.summarize(plan, schedule, outputs, tmp_path / "completion.json")
    assert result["numerical_denominator"] == 192
    assert result["recommendations"]["full_training_runtime_estimate"] is None
    assert not result["gates_passed"] and not result["training_complete"]
    for arm in profile.ARMS:
        assert result["stages"][arm]["outcomes"] == {
            "unavailable_graph/unknown": 31,
            "unavailable_structural_parent/unknown": 1,
        }
    # Tampering with an outcome does not make it eligible for rerun.
    Path(outputs["graph"], "rows", "0.json").write_text("{}")
    with pytest.raises(ValueError, match="dependencies"):
        profile.run_stage(plan, schedule, "graph", outputs["graph"])


def test_unfinished_probe_requires_cleanup_reconciliation(tmp_path, monkeypatch):
    plan, schedule, frozen = fixture_plan(tmp_path, monkeypatch)
    output = tmp_path / "graph"
    profile.checkpoint(
        output / "pending/0.json", canonical_hash(("fixed-identity", "graph", "0")), budget={}
    )

    def forbidden(*args, **kwargs):
        pytest.fail("An unfinished native probe must not be repeated")

    monkeypatch.setattr("exact.repair.workers.bounded_call", forbidden)
    with pytest.raises(RuntimeError, match="prior owner"):
        profile.run_stage(plan, schedule, "graph", output)


def test_cleanup_failure_and_changed_payload_are_not_hidden(tmp_path, monkeypatch):
    plan, schedule, frozen = fixture_plan(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "exact.repair.workers.bounded_call",
        lambda *args, **kwargs: CallResult("timeout", cleanup_complete=False),
    )
    with pytest.raises(RuntimeError, match="cleanup incomplete"):
        profile.run_stage(plan, schedule, "graph", tmp_path / "graph")
    payload = tmp_path / "weights.pt"
    payload.write_bytes(b"profiling-only")
    row = dict(result=dict(artifacts=[profile.binding(payload)]))
    profile.validate_artifacts(row)
    payload.write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        profile.validate_artifacts(row)


def test_completion_rejects_missing_stages(tmp_path, monkeypatch):
    plan, schedule, frozen = fixture_plan(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="Every registered"):
        profile.summarize(plan, schedule, {}, tmp_path / "completion.json")


def test_freeze_preserves_missing_development_parents(tmp_path, monkeypatch):
    plan = tmp_path / "plan.json"
    profile.immutable(
        plan,
        dict(
            schema="exact-repair/numerical-profile-plan/v1",
            profile_report=dict(path=str(tmp_path / "report.json")),
        ),
    )
    inventory = dict(
        selected=[],
        missing=[dict(split="profile", family=f) for f in MECHANISMS_SELECTED for _ in range(2)],
    )
    monkeypatch.setattr(profile, "validate_gpu", lambda _: None)
    monkeypatch.setattr(profile, "validate_profile", lambda _: ({}, inventory, {}))
    frozen = profile.schedule(plan)
    assert len(frozen["cases"]) == 32
    assert all(row["case"] is None for row in frozen["cases"])
    assert frozen["numerical_rows"] == 192
    inventory["missing"].pop()
    with pytest.raises(ValueError, match="32-case"):
        profile.schedule(plan)


def test_frozen_schedule_cannot_be_reselected(tmp_path, monkeypatch):
    source = tmp_path / "schedule.json"
    profile.immutable(source, dict(cases=["old"]))
    monkeypatch.setattr(profile, "schedule", lambda _: dict(cases=["new"]))
    with pytest.raises(ValueError, match="schedule changed"):
        profile.load_schedule(tmp_path / "plan.json", source)
