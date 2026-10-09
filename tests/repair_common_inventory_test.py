"""Fixed inventories, frozen value heads, hidden evaluation and bounded receipts."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from exact.repair.model import RepairModel, ModelGraphSchemaError
from exact.repair.pipeline import model_digest
from exact.repair.records import canonical_hash, read_record
from exact.repair.workers import bounded_call
from tools.repair import common_inventory as inventory, fresh_evaluation as fresh
from tools.repair.prepare import load_preparation, case_to_dict

pytest_plugins = ["tests.repair_prepare_controls_test"]


def fixture(prepared, tmp_path):
    source, protocol_path = prepared
    cases, _, _ = load_preparation(source)
    case = next(c for c in cases if c.split == "development")
    protocol = fresh.read(protocol_path)
    item = dict(
        case_id=case.case_id,
        structural_parent=case.structural_parent,
        family=case.family,
        family_exposure="seen_family",
        status="materialized",
        observable=fresh.immutable(tmp_path / "observable.json", case.problem.to_dict()),
        evaluator=fresh.immutable(tmp_path / "evaluator.json", case_to_dict(case)),
    )
    return case, protocol, protocol_path, item


def model_arm(problem, protocol, protocol_path, tmp_path, *, pairwise=True, encoder="none"):
    _, graph = inventory.observable_graph(problem, protocol, tmp_path)
    model = RepairModel(
        graph.metadata,
        encoder=encoder,
        hidden_dim=8,
        heads=2,
        layers=1 if encoder != "none" else 0,
        dropout=0,
        revision="v3",
        pairwise=pairwise,
    )
    path = tmp_path / "model.pt"
    torch.save(
        dict(
            model_schema="exact-repair/model/v3",
            metadata=model.metadata,
            config=model.config,
            state_dict=model.state_dict(),
        ),
        path,
    )
    return (
        dict(
            id="fixture-model",
            kind="learned",
            status="available",
            model=fresh.binding(path),
            protocol=fresh.binding(protocol_path),
            model_hash=model_digest(model),
        ),
        model,
        graph,
    )


def test_separate_schedule_keeps_all_rows_budgets_and_primary_links():
    cases = [dict(case_id=str(i)) for i in range(64)]
    arms = [dict(id=str(i)) for i in range(9)]
    primary = dict(rows=fresh.make_rows(cases, arms))
    rows = inventory.make_rows(primary)
    assert len(rows) == len({r["id"] for r in rows}) == 576
    assert not {r["id"] for r in rows} & {r["id"] for r in primary["rows"]}
    assert [r["primary_row_id"] for r in rows] == [r["id"] for r in primary["rows"]]
    assert all((r["seconds"], r["cpu_seconds"], r["memory_mb"]) == (300, 600, 8192) for r in rows)


@pytest.mark.parametrize("pairwise", [False, True])
def test_frozen_readout_matches_direct_scores_without_candidate_changes(
    prepared, tmp_path, pairwise
):
    case, protocol, pp, _ = fixture(prepared, tmp_path)
    arm, model, graph = model_arm(case.problem, protocol, pp, tmp_path, pairwise=pairwise)
    original = Path(arm["model"]["path"]).read_bytes()
    frozen = inventory.freeze_inventory(case.problem, arm, protocol, tmp_path)
    prep, _ = inventory.observable_graph(case.problem, protocol, tmp_path)
    pairs = prep.pairs(case.problem, graph, enabled=pairwise)
    model.eval()
    with torch.no_grad():
        unary, factors = model.score_inventory(
            case.problem.objects, model.encode(graph), interaction_pairs=pairs.pairs
        )
    assert frozen.problem is case.problem and frozen.proposal_reports == ()
    assert frozen.objective.benefit == tuple(tuple(float(v) for v in row) for row in unary)
    assert frozen.objective.raw_pairs == tuple(
        (*key, float(v)) for key, v in sorted(factors.items())
    )
    assert frozen.objective.pool_hash == canonical_hash(case.problem.objects)
    assert frozen.model_hash == model_digest(model)
    assert frozen.risk_scorer.checkpoint_sha256 == arm["model"]["sha256"]
    assert Path(arm["model"]["path"]).read_bytes() == original
    assert not (tmp_path / "unused-compiler-cache").exists()


def test_model_binding_tampering_is_not_schema_unavailability(prepared, tmp_path):
    case, protocol, pp, _ = fixture(prepared, tmp_path)
    arm, _, _ = model_arm(case.problem, protocol, pp, tmp_path)
    Path(arm["model"]["path"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="dependency changed"):
        inventory.freeze_inventory(case.problem, arm, protocol, tmp_path)


@pytest.mark.parametrize("arm_id", fresh.CONTROLS)
def test_controls_share_supplied_inventory_and_explicit_costs(prepared, tmp_path, arm_id):
    case, protocol, _, _ = fixture(prepared, tmp_path)
    arm = dict(id=arm_id, kind="control")
    frozen = inventory.freeze_inventory(case.problem, arm, protocol, tmp_path)
    expected_problem, expected_objective = fresh.control_objective(case.problem, arm_id, protocol)
    assert frozen.problem == expected_problem and frozen.objective == expected_objective
    assert frozen.risk_scorer is None and not frozen.proposal_reports
    proof = fresh.read(tmp_path / "inventory-scoring.json")
    assert proof["generation"] is False and proof["evaluator_opened"] is False


def test_schema_rejection_preserves_checkpoint(prepared, tmp_path, monkeypatch):
    case, protocol, pp, _ = fixture(prepared, tmp_path)
    arm, _, graph = model_arm(case.problem, protocol, pp, tmp_path, encoder="rgcn")
    prep, _ = inventory.observable_graph(case.problem, protocol, tmp_path)
    from exact.repair.graph import GraphNode

    changed = replace(graph, nodes=graph.nodes + (GraphNode("new", "untrained", ()),))
    monkeypatch.setattr(inventory, "observable_graph", lambda *a: (prep, changed))
    before = Path(arm["model"]["path"]).read_bytes()
    with pytest.raises(ModelGraphSchemaError):
        inventory.freeze_inventory(case.problem, arm, protocol, tmp_path)
    assert Path(arm["model"]["path"]).read_bytes() == before


@pytest.mark.parametrize("arm_id", [*fresh.CONTROLS, "fixture-model"])
def test_native_fixed_inventory_selection_and_postselection_label(prepared, tmp_path, arm_id):
    case, protocol, pp, item = fixture(prepared, tmp_path)
    if arm_id == "fixture-model":
        arm, _, _ = model_arm(case.problem, protocol, pp, tmp_path)
    else:
        arm = dict(id=arm_id, kind="control", status="available", protocol=fresh.binding(pp))
    schedule = dict(cases=[item], arms=[arm])
    row = inventory.make_rows(dict(rows=fresh.make_rows([item], [arm])))[0]
    result = bounded_call(
        inventory.evaluate_row,
        schedule,
        row,
        str(tmp_path / "run"),
        timeout=300,
        cpu_seconds=600,
        memory_mb=8192,
    )
    assert result.status == "complete", result.detail
    payload = fresh.bound(result.value)
    assert payload["status"] == "evaluated", payload
    assert payload["study_kind"] == "common_inventory_diagnostic"
    assert payload["common_inventory_regret"] is None
    pool = fresh.read(tmp_path / "run/pool.json")
    assert pool["proposal_reports"] == []
    if arm_id != "deletion":
        assert read_record(pool["input"]).objects == case.problem.objects
    if payload["logical_status"] == "VERIFIED_FEASIBLE":
        assert payload["semantic_status"] == "known", payload
        if arm_id == "fixture-model":
            assert payload["selected_utility_error"] == pytest.approx(
                payload["selected_objective_utility"] - payload["selected_utility"]
            )


def test_unavailable_scoring_never_opens_hidden_evaluator(prepared, tmp_path, monkeypatch):
    import exact.repair.workers
    from exact.repair.workers import CallResult

    case, _, pp, item = fixture(prepared, tmp_path)
    Path(item["evaluator"]["path"]).unlink()
    arm = dict(id="fixture-model", kind="learned", status="available", protocol=fresh.binding(pp))
    schedule = dict(cases=[item], arms=[arm])
    row = fresh.make_rows([item], [arm])[0]
    calls = []

    def reject(*a, **k):
        calls.append(a[0])
        return CallResult("error", detail="ModelGraphSchemaError: untrained relation")

    monkeypatch.setattr(exact.repair.workers, "bounded_call", reject)
    payload = fresh.bound(inventory.evaluate_row(schedule, row, tmp_path / "run"))
    assert payload["status"] == "unavailable_model_schema"
    assert payload["semantic_status"] == "not_evaluated"
    assert calls == [inventory.freeze_inventory]
