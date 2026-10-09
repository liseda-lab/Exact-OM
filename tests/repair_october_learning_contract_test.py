"""Corrective learning contracts; no hosted calls or held-out fitting."""

import copy
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from exact.repair.graph import FEATURE_SCHEMA_V3, GraphNode, ObservableGraph
from exact.repair.graph_schema import declared_metadata, generic_graph_schema
from exact.repair.learning import RepairLabel, generated_checkpoint_criterion, interaction_loss
from exact.repair.model import RepairModel
from exact.repair.semantic_fidelity import aggregate_plan_ratings
from tests.repair_training_completion_test import cache_for
from tools.repair import train
from tools.repair.corpus import generate_corpus


def all_declared_graph():
    nodes, edges = declared_metadata(generic_graph_schema())
    return ObservableGraph(
        tuple(GraphNode(k, k, (("observed", 1.0),)) for k in nodes),
        edges,
        (),
        feature_schema=FEATURE_SCHEMA_V3,
    )


@pytest.mark.parametrize("encoder", ["hgt", "rgcn", "none"])
def test_all_declared_types_actual_adamw_and_resume(tmp_path, encoder):
    torch.set_num_threads(1)
    torch.manual_seed(13)
    graph = all_declared_graph()
    schema = generic_graph_schema()
    model = RepairModel(
        declared_metadata(schema),
        graph_schema=schema,
        encoder=encoder,
        hidden_dim=8,
        heads=2,
        layers=1,
        dropout=0,
        revision="v3",
        pairwise=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    before = copy.deepcopy(model.state_dict())

    def update(m, optim):
        optim.zero_grad()
        memory = m.encode(graph)
        loss = sum((row - 0.3).square().mean() for row in memory.rows.values())
        loss.backward()
        optim.step()
        return loss

    update(model, optimizer)
    assert optimizer.state and any(
        not torch.equal(v, model.state_dict()[k]) for k, v in before.items()
    )
    path = tmp_path / "actual-adamw.pt"
    torch.save(
        dict(model=model.state_dict(), optimizer=optimizer.state_dict(), rng=torch.get_rng_state()),
        path,
    )
    update(model, optimizer)
    saved = torch.load(path, weights_only=True)
    restored = RepairModel(
        declared_metadata(schema),
        graph_schema=schema,
        encoder=encoder,
        hidden_dim=8,
        heads=2,
        layers=1,
        dropout=0,
        revision="v3",
        pairwise=True,
    )
    restored.load_state_dict(saved["model"], strict=True)
    resumed_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.001)
    resumed_optimizer.load_state_dict(saved["optimizer"])
    torch.set_rng_state(saved["rng"])
    update(restored, resumed_optimizer)
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items())
    assert len(graph.metadata[0]) == 13 and len(graph.metadata[1]) == 553
    assert all(torch.isfinite(v).all() for v in restored.encode(graph).rows.values())


def test_entire_missing_dev_cache_keeps_schedule_and_criterion_denominator():
    cases = generate_corpus(
        split_counts={"train": 1, "development": 2, "test": 0},
        siblings_per_parent=1,
        families=("range",),
        revision="v3",
    )
    dev = [c for c in cases if c.split == "development"]
    schedule = train.scheduled_development(
        cases, {dev[0].case_id: cache_for(dev[0])}, expected_case_ids=[c.case_id for c in dev]
    )
    assert len(schedule) == 2
    assert schedule[1][1].stop_reason == "missing_development_cache"
    reports = [dict(case_id=c.case_id, parent_group_id=c.structural_parent) for c in dev]
    reports[0]["decoded"] = dict(status="verified", selected_utility=1, checks=1)
    ids = [c.case_id for c in dev]
    assert generated_checkpoint_criterion(reports, expected_case_ids=ids) is None
    assert (
        generated_checkpoint_criterion(reports, expected_case_ids=ids, minimum_coverage=0.5)[0]
        == -0.5
    )
    with pytest.raises(ValueError, match="case-ID"):
        generated_checkpoint_criterion(reports[:1], expected_case_ids=ids)
    assert (
        generated_checkpoint_criterion(
            [dict(case_id=c.case_id) for c in dev], expected_case_ids=ids
        )
        is None
    )


def test_valid_parent_does_not_consult_soft_queries(monkeypatch):
    # Actual installed native verifier: only the soft-label oracle is forbidden.
    case = generate_corpus(
        split_counts={"train": 1, "development": 0, "test": 0},
        siblings_per_parent=1,
        families=("range",),
        revision="v3",
    )[0]
    monkeypatch.setattr(
        train,
        "OwlTeacherOracle",
        lambda *a: (_ for _ in ()).throw(AssertionError("soft queries gate parent")),
    )
    assert train._verify_intended(case)


@pytest.mark.parametrize("effect", [-0.7, 0.0, 0.6])
def test_only_selected_factors_receive_verified_quartets(effect):
    values = [0.1, 0.1, 0.1, 0.1 + effect]
    labels = [RepairLabel(a, True, v, 0) for a, v in zip(((0, 0), (1, 0), (0, 1), (1, 1)), values)]
    predictions = torch.zeros(4, requires_grad=True)
    missing_factor = interaction_loss(predictions, labels, eligible_pairs=())
    assert missing_factor["eligible"] == 0
    selected = interaction_loss(predictions, labels, eligible_pairs=((0, 1),))
    assert selected["targets"] == pytest.approx((effect,))
    assert selected["positive"] == int(effect > 0)
    assert selected["negative"] == int(effect < 0)
    assert selected["zero"] == int(effect == 0)
    selected["loss"].backward()
    assert bool(predictions.grad.abs().sum() > 0) == (effect != 0)
    labels[-1] = replace(labels[-1], feasible=False, benefit=None)
    assert interaction_loss(predictions, labels, eligible_pairs=((0, 1),))["eligible"] == 0


def test_repeated_rating_medians_provenance_disagreement_and_incompatible_context():
    policy = dict(revision="per_criterion_median/v1", minimum_ratings=3, max_criterion_range=0.8)
    weights = dict(meaning_retention=0.75, assertion_fidelity=0.25)
    rows = [
        dict(
            context="exact-plan-and-evidence",
            provenance=dict(request_id=str(i), attempt=1, response_sha256=str(i)),
            criteria=dict(meaning_retention=a, assertion_fidelity=b),
        )
        for i, (a, b) in enumerate(((0.2, 0.8), (0.9, 0.1), (0.4, 0.4)))
    ]
    value = aggregate_plan_ratings(rows + [rows[0]], weights, policy)
    assert len(value["ratings"]) == 3
    assert value["score"] == pytest.approx(0.4)
    assert aggregate_plan_ratings(list(reversed(rows)), weights, policy) == value
    strict = aggregate_plan_ratings(rows, weights, {**policy, "max_criterion_range": 0.2})
    assert strict["status"] == "disagreement_abstention" and strict["score"] is None
    with pytest.raises(ValueError, match="incompatible"):
        aggregate_plan_ratings(
            [rows[0], {**rows[1], "context": "changed evidence"}], weights, policy
        )


def test_train_entry_full_schema_minibatch_exact_update_resume_and_two_passes(
    tmp_path, monkeypatch
):
    cases = generate_corpus(
        split_counts={"train": 3, "development": 1, "test": 0},
        siblings_per_parent=1,
        families=("range",),
        revision="v3",
    )
    fitting = [(c, cache_for(c)) for c in cases if c.split == "train"]
    dev = train.scheduled_development(cases, {})
    evaluated = []
    monkeypatch.setattr(train, "decoded_development", lambda *a, **k: dict(exact_regret=None))

    def generated(model, case, *a, **kw):
        evaluated.append(case.case_id)
        return dict(
            parent_group_id=case.structural_parent,
            decoded=dict(status="verified", selected_utility=0.5, checks=1),
        )

    monkeypatch.setattr(train, "generated_development", generated)
    options = dict(
        encoder="hgt",
        graph_schema=generic_graph_schema(),
        hidden_dim=8,
        heads=2,
        layers=1,
        dropout=0.2,
        pairwise=True,
        batch_cases=2,
        epochs=3,
        development_epochs=[2, 3],
        development_case_ids=[dev[0][0].case_id],
        max_full_development_evaluations=2,
        patience_enabled=False,
        proposal_arm="bounded_enumeration",
        sampled_assignments=0,
        deadline_seconds=90,
        total_training_seconds=180,
    )
    baseline, report = train.train_cases(fitting, dev, **options)
    assert report["optimizer_updates"] == 6 and len(evaluated) == 2
    evaluated.clear()
    path = tmp_path / "state.pt"
    original_save = train.save_training_state

    def interrupt(destination, state):
        original_save(destination, state)
        if state["optimizer_updates"] == 1:
            raise InterruptedError("after committed optimizer update")

    monkeypatch.setattr(train, "save_training_state", interrupt)
    with pytest.raises(InterruptedError):
        train.train_cases(fitting, dev, checkpoint_path=path, **options)
    saved = torch.load(path, weights_only=True)
    assert saved["optimizer_updates"] == 1 and saved["optimizer"]["state"]
    assert saved["next_offset"] == 2
    monkeypatch.setattr(train, "save_training_state", original_save)
    resumed, report = train.train_cases(fitting, dev, checkpoint_path=path, **options)
    assert report["optimizer_updates"] == 6 and len(evaluated) == 2
    assert all(torch.equal(v, resumed.state_dict()[k]) for k, v in baseline.state_dict().items())


def test_native_parent_with_unknown_soft_query_retains_risk_labels(monkeypatch):
    from exact.repair.workers import CallResult

    class UnknownSoft:
        def __init__(self, *args):
            pass

        def consistent(self):
            return True

        def entails(self, query):
            return None

        def satisfiable(self, query):
            return True

    case = generate_corpus(
        split_counts={"train": 1, "development": 0, "test": 0},
        siblings_per_parent=1,
        families=("range",),
        revision="v3",
    )[0]
    monkeypatch.setattr(train, "OwlTeacherOracle", UnknownSoft)

    def direct(function, *args, **kwargs):
        kwargs.pop("timeout", None)
        return CallResult("complete", function(*args, **kwargs))

    monkeypatch.setattr(train, "bounded_call", direct)
    cache = train.label_case(case, deadline_seconds=30, call_seconds=10)
    assert not cache.complete
    assert any(label.feasible is True for label in cache.labels)
    assert all(label.benefit is None for label in cache.labels)
    from exact.repair.learning import risk_loss

    logits = torch.zeros(len(cache.labels), requires_grad=True)
    loss = risk_loss(logits, cache.labels)
    assert loss["eligible"] == len(cache.labels)


def test_semantic_subset_keeps_full_logical_denominator_and_all_preselected_labels():
    reports = [
        dict(
            case_id="a",
            parent_group_id="p1",
            decoded=dict(
                logical_status="VERIFIED_FEASIBLE",
                status="verified",
                selected_utility=0.8,
                checks=2,
            ),
        ),
        dict(
            case_id="b",
            parent_group_id="p2",
            decoded=dict(
                logical_status="VERIFIED_FEASIBLE", status="unknown_selected_labels", checks=3
            ),
        ),
    ]
    criterion = generated_checkpoint_criterion(
        reports, expected_case_ids=["a", "b"], semantic_case_ids=["a"]
    )
    assert criterion[:2] == (-1.0, -1.0)
    assert (
        generated_checkpoint_criterion(
            reports, expected_case_ids=["a", "b"], semantic_case_ids=["a", "b"]
        )
        is None
    )
    assert (
        generated_checkpoint_criterion(reports, expected_case_ids=["a", "b"], semantic_case_ids=[])
        is None
    )


def test_optimizer_qualifier_sweeps_every_exposed_dev_case(tmp_path):
    from tools.repair.prepare import save_preparation
    from tools.repair.qualify_learning import run

    cases = generate_corpus(
        split_counts={"train": 0, "development": 2, "test": 0},
        siblings_per_parent=1,
        families=("range",),
        revision="v3",
    )
    preparation = tmp_path / "preparation.json"
    save_preparation(preparation, cases, dict(scope="qualification only"), {})
    result = run(
        tmp_path / "qualification",
        device="cpu",
        width=8,
        layers=1,
        heads=2,
        prepared=preparation,
        seconds=30,
    )
    assert result["status"] == "qualified" and result["exact_resume"]
    assert result["actual_cases_scheduled"] == result["actual_cases_completed"] == 2
    assert all(row["optimizer_updates"] == 1 for row in result["actual_case_rows"])


def test_final_dev_reserve_stops_next_optimizer_update_but_allows_scheduled_dev(
    tmp_path, monkeypatch
):
    cases = generate_corpus(
        split_counts={"train": 1, "development": 1, "test": 0},
        siblings_per_parent=1,
        families=("range",),
        revision="v3",
    )
    fitting = [(c, cache_for(c)) for c in cases if c.split == "train"]
    dev = train.scheduled_development(cases, {})
    elapsed = [0.0]
    monkeypatch.setattr(train.time, "monotonic", lambda: elapsed[0])
    monkeypatch.setattr(train, "decoded_development", lambda *a, **k: dict(exact_regret=None))
    evaluated = []

    def generated(model, case, *a, **kw):
        evaluated.append(kw["seconds"])
        return dict(
            parent_group_id=case.structural_parent,
            decoded=dict(status="verified", selected_utility=0.5, checks=1),
        )

    monkeypatch.setattr(train, "generated_development", generated)
    save = train.save_training_state

    def clock_after_update(path, state):
        save(path, state)
        if state["optimizer_updates"] == 1:
            elapsed[0] = 6.0

    monkeypatch.setattr(train, "save_training_state", clock_after_update)
    path = tmp_path / "reserved-state.pt"
    options = dict(
        encoder="none",
        hidden_dim=8,
        heads=2,
        layers=0,
        dropout=0,
        epochs=2,
        development_epochs=[1, 2],
        max_full_development_evaluations=2,
        patience_enabled=False,
        sampled_assignments=0,
        proposal_arm="bounded_enumeration",
        deadline_seconds=10,
        total_training_seconds=10,
        final_development_reserve_seconds=5,
        checkpoint_path=path,
    )
    _, report = train.train_cases(fitting, dev, **options)
    assert report["optimizer_updates"] == report["completed_epochs"] == 1
    assert evaluated == [4.0]
    assert report["status"] == "interrupted"
    assert report["interruption_reason"] == "final_development_reserve_reached"
    saved = torch.load(path, weights_only=True)
    assert saved["optimizer_updates"] == 1 and saved["elapsed_seconds"] == 6.0
    _, resumed = train.train_cases(fitting, dev, **options)
    assert resumed["optimizer_updates"] == 1 and evaluated == [4.0]


def test_final_dev_reserve_protocol_routing_preserves_legacy_absence():
    import json
    from pathlib import Path
    from exact.repair.protocol import RepairProtocolV3, training_projection_v3

    path = Path(__file__).parents[1] / "specs/exact-repair/protocol/xr21-review2-conformance.json"
    value = json.loads(path.read_text())
    original = RepairProtocolV3.model_validate(value)
    assert "final_development_reserve_seconds" not in original.model_dump()["training"]
    value["training"]["final_development_reserve_seconds"] = 40
    declared = RepairProtocolV3.model_validate(value)
    options = train._protocol_arguments(training_projection_v3(declared))
    assert options["final_development_reserve_seconds"] == 40
