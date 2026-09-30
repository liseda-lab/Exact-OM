"""Corrective REV03–06/09/12 regressions; no scientific campaign or live judge."""

import itertools
from dataclasses import replace

import pyowl_core as owl
import pytest

from exact.repair.graph import (
    FEATURE_SCHEMA_V3,
    EffectivePreparation,
    GraphExplanation,
    build_observable_graph,
)
from exact.repair.learning import (
    RepairLabel,
    SemanticTargetSpec,
    collect_sampled_repairs,
    support_loss,
)
from exact.repair.records import canonical_hash
from tests.repair_learning_v3_test import problem
from tests.repair_training_completion_test import cache_for
from tools.repair.corpus import generate_corpus


def test_hgt_risk_uses_complete_admitted_support_projection_and_omissions():
    import torch

    from exact.repair.model import RepairModel

    torch.set_num_threads(1)
    p = problem()
    absent = owl.Class(owl.IRI("urn:test:SupportOnly"))
    support = GraphExplanation(
        "late", ("o0",), (owl.SubClassOf(absent, p.policy.monitored_classes[0]),), absent
    )
    base = build_observable_graph(p.objects, feature_schema=FEATURE_SCHEMA_V3)
    for limits in (
        {"max_explanations": 0},
        {"max_nodes": len(base.nodes)},
        {"max_edges": len(base.edges)},
    ):
        graph = build_observable_graph(
            p.objects, explanations=(support,), feature_schema=FEATURE_SCHEMA_V3, **limits
        )
        assert graph.admitted_supports == () and graph.omitted_supports == ("late",)
        model = RepairModel(
            graph.metadata, hidden_dim=8, heads=2, layers=1, encoder="hgt", dropout=0, revision="v3"
        ).eval()
        memory = model.encode(graph)
        value = model.plan_risk_logit(
            p.objects, memory, (0, 0, 0), supports=graph.admitted_supports
        )
        assert torch.isfinite(value)
        changed = replace(support, witness=owl.Class(owl.IRI("urn:other:Excluded")))
        assert changed != support
        assert torch.equal(value, model.plan_risk_logit(p.objects, memory, (0, 0, 0), supports=()))
        with pytest.raises(ValueError, match="graph admission"):
            model.plan_risk_logit(p.objects, memory, (0, 0, 0), supports=(changed,))
    included = build_observable_graph(
        p.objects, explanations=(support,), feature_schema=FEATURE_SCHEMA_V3
    )
    assert included.admitted_supports == (support,)
    model = RepairModel(
        included.metadata, hidden_dim=8, heads=2, layers=1, encoder="hgt", dropout=0, revision="v3"
    )
    assert torch.isfinite(
        model.plan_risk_logit(
            p.objects, model.encode(included), (0, 0, 0), supports=included.admitted_supports
        )
    )


def test_effective_preparation_resolves_nondefault_limits_and_identity():
    from exact.repair.retrieval import RetrievalConfig, retrieve_vocabulary

    p = problem()
    retrieval = retrieve_vocabulary(p)
    settings = EffectivePreparation(300, 800, 0, 3, 1, 2, 12, RetrievalConfig(), "v3")
    one, two = settings.graph(p, retrieval), settings.graph(p, retrieval)
    assert canonical_hash(one) == canonical_hash(two)
    assert settings.pairs(p, one) == settings.pairs(p, two)
    assert one.preparation_identity == settings.content_hash
    assert replace(settings, max_text_tokens=4).content_hash != settings.content_hash
    assert settings.freeze_options()["max_explanations"] == 0


def test_complete_plan_quotas_preserve_every_origin_and_finite_denominator():
    quotas = dict(utility=32, proposal=32, diversity=32, quartet=32, uniform=0)
    calls = []

    def label(assignment):
        calls.append(assignment)
        return RepairLabel(assignment, None, None, 0)

    proposed = [((0,) * 8, "utility")] * 32 + [((0,) * 8, "proposal", 0.25)] * 32
    options = dict(
        case_id="case",
        parent_group_id="parent",
        split="train",
        hashes={"inventory": "pool"},
        model_hash="model",
        round_id="0",
        max_assignments=128,
        deadline_seconds=10,
        seed=11,
        proposed=proposed,
        plan_quotas=quotas,
        exploration_fraction=0,
    )
    result = collect_sampled_repairs((5,) * 8, label, **options)
    assert len(result.attempts) == 128
    stats = {name: dict(row) for name, row in result.strata}
    assert all(
        stats[name]["scheduled"] == 32 for name in ("utility", "proposal", "diversity", "quartet")
    )
    assert len(calls) == len(set(calls))
    assert len([row for row in result.attempts if row["assignment"] == (0,) * 8]) >= 64
    assert any(
        row["status"] == "duplicated" and row["stratum"] == "proposal" for row in result.attempts
    )
    again = collect_sampled_repairs((5,) * 8, label, **options)
    assert result.sampler_hash == again.sampler_hash and result.attempts == again.attempts
    altered = {**quotas, "utility": 31, "uniform": 1}
    variant = collect_sampled_repairs(
        (5,) * 8, label, **{**options, "plan_quotas": altered, "exploration_fraction": 1 / 128}
    )
    assert variant.sampler_hash != result.sampler_hash
    with pytest.raises(ValueError, match="fraction contradicts"):
        collect_sampled_repairs((5,) * 8, label, **{**options, "exploration_fraction": 0.5})
    exhausted = collect_sampled_repairs(
        (1,),
        label,
        **{
            **options,
            "proposed": (),
            "plan_quotas": dict(utility=32, proposal=32, diversity=32, quartet=32, uniform=0),
        },
    )
    assert all(row["status"] == "unavailable" for row in exhausted.attempts)


def test_nonunit_target_reuses_native_semantic_vector_without_changing_costs():
    from tools.repair.train import DEFAULT_PROFILE, _assignment_label

    case = generate_corpus(
        split_counts={"train": 1, "development": 0, "test": 0},
        siblings_per_parent=1,
        families=("range",),
    )[0]
    target = SemanticTargetSpec(canonical_hash(case.probes), 2.5, 3.5)
    labels = [
        _assignment_label(case, assignment, DEFAULT_PROFILE, semantic_target=target)
        for assignment in itertools.product(
            *(range(len(obj.candidates)) for obj in case.problem.objects)
        )
    ]
    label = next(row for row in labels if row.usable)
    unit = _assignment_label(case, label.assignment, DEFAULT_PROFILE)
    families = {}
    for outcome in label.semantic_vector:
        value = outcome.credit if outcome.desired else outcome.entailed
        families.setdefault((outcome.desired, outcome.family), []).append(float(value))
    expected = sum(
        (2.5 if desired else -3.5) * sum(values) / len(values)
        for (desired, _), values in families.items()
    )
    assert label.benefit == pytest.approx(expected) and label.cost == unit.cost
    assert label.semantic_vector == unit.semantic_vector
    assert target.content_hash != replace(target, false_positive_weight=3).content_hash


def test_native_three_action_witness_targets_train_support_head_without_proof_leakage():
    import torch

    from exact.repair.model import RepairModel
    from tools.repair.train import DEFAULT_PROFILE, _assignment_label

    case = next(
        c
        for c in generate_corpus(
            split_counts={"train": 1, "development": 0, "test": 0},
            siblings_per_parent=1,
            families=("conflicts_higher_order",),
        )
        if c.control != "clean"
    )
    label = _assignment_label(case, (0, 0, 0), DEFAULT_PROFILE)
    positives = [target for target in label.support_targets if target.violated]
    assert len(positives) == 1 and len(positives[0].asserted_axioms) >= 3
    assert positives[0].available_before_decision is False
    graph = build_observable_graph(
        case.problem.objects,
        fixed_axioms=case.problem.fixed_axioms,
        feature_schema=FEATURE_SCHEMA_V3,
    )
    model = RepairModel(
        graph.metadata,
        hidden_dim=8,
        heads=2,
        layers=1,
        encoder="hgt",
        dropout=0,
        revision="v3",
        support_enabled=True,
    )
    memory = model.encode(graph)
    target = positives[0]
    value = model.support_violation_logit(
        case.problem.objects,
        memory,
        target.assignment,
        owl.decode_canonical(bytes.fromhex(target.witness)),
    )
    terms = support_loss(value.reshape(1), (target,))
    terms["loss"].backward()
    assert terms["eligible"] == 1 and any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.support_head.parameters()
    )
    unknown = replace(target, violated=None, proof_json=None)
    parameter = torch.tensor([0.5], requires_grad=True)
    masked = support_loss(parameter, (unknown,))
    masked["loss"].backward()
    assert parameter.grad.item() == 0
    # Dropping a premise is verified independently; negatives aren't inferred
    # merely from missing support. No pairwise incompatibility labels are made.
    for i in range(3):
        assignment = [0, 0, 0]
        assignment[i] = 1
        pair = _assignment_label(case, tuple(assignment), DEFAULT_PROFILE)
        assert pair.feasible is True
        assert not any(t.violated for t in pair.support_targets)
    assert canonical_hash(graph) == canonical_hash(
        build_observable_graph(
            case.problem.objects,
            fixed_axioms=case.problem.fixed_axioms,
            feature_schema=FEATURE_SCHEMA_V3,
        )
    )


@pytest.mark.parametrize(
    "phase,offset",
    [
        ("train", 1),
        ("development_loss", 3),
        ("development_selection", 3),
        ("development_selection_midway", 3),
    ],
)
def test_exact_partial_phase_resume_matches_uninterrupted_model(
    tmp_path, monkeypatch, phase, offset
):
    import torch

    import tools.repair.train as train

    cases = generate_corpus(
        split_counts={"train": 3, "development": 2, "test": 0},
        siblings_per_parent=1,
        families=("range",),
    )
    training = [(c, cache_for(c)) for c in cases if c.split == "train"]
    development = [(c, cache_for(c)) for c in cases if c.split == "development"]
    monkeypatch.setattr(train, "decoded_development", lambda *a, **k: {"exact_regret": 0.0})

    def generated(model, case, cache, **options):
        return dict(
            status="generated",
            parent_group_id=case.structural_parent,
            useful_candidate_coverage=1.0,
            decoded=dict(status="verified", selected_utility=0.5, checks=1),
        )

    monkeypatch.setattr(train, "generated_development", generated)
    options = dict(
        epochs=1,
        hidden_dim=8,
        heads=2,
        layers=0,
        encoder="none",
        dropout=0.2,
        proposal_arm="bounded_enumeration",
        sampled_assignments=0,
        plan_risk=False,
        deadline_seconds=60,
        batch_cases=1,
    )
    baseline_path = tmp_path / "baseline.pt"
    baseline, expected = train.train_cases(
        training, development, checkpoint_path=baseline_path, **options
    )
    baseline_saved = torch.load(baseline_path, weights_only=True)
    path = tmp_path / "state.pt"
    save = train.save_training_state
    triggered = []

    def interrupt(destination, state):
        save(destination, state)
        desired_phase = (
            "development_selection" if phase == "development_selection_midway" else phase
        )
        midway = (
            phase != "development_selection_midway"
            or len(state["development_progress"].get("generated", {})) == 1
        )
        if (
            state["phase"] == desired_phase
            and state["next_offset"] == offset
            and midway
            and not triggered
        ):
            triggered.append(True)
            raise InterruptedError("after durable phase commit")

    monkeypatch.setattr(train, "save_training_state", interrupt)
    with pytest.raises(InterruptedError):
        train.train_cases(training, development, checkpoint_path=path, **options)
    saved = torch.load(path, weights_only=True)
    assert saved["next_epoch"] == 0 and saved["next_offset"] == offset
    assert set(saved["epoch_order"]) == {case.case_id for case, _ in training}
    monkeypatch.setattr(train, "save_training_state", save)
    resumed, actual = train.train_cases(training, development, checkpoint_path=path, **options)
    assert actual["history"] == expected["history"]
    resumed_saved = torch.load(path, weights_only=True)

    def equal_state(left, right):
        if isinstance(left, torch.Tensor):
            assert torch.equal(left, right)
        elif isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                equal_state(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            assert len(left) == len(right)
            for a, b in zip(left, right):
                equal_state(a, b)
        else:
            assert left == right

    for key in (
        "model",
        "optimizer",
        "cpu_rng",
        "cuda_rng",
        "python_rng",
        "best_state",
        "best_criterion",
        "next_epoch",
        "next_offset",
        "epoch_order",
        "stale_evaluations",
    ):
        equal_state(baseline_saved[key], resumed_saved[key])
    assert resumed_saved["execution_count"] == 2 and baseline_saved["execution_count"] == 1
    assert resumed_saved["elapsed_seconds"] >= saved["elapsed_seconds"]
    assert all(
        torch.equal(value, resumed.state_dict()[key])
        for key, value in baseline.state_dict().items()
    )


@pytest.mark.parametrize("limit", ["explanation", "nodes", "edges", "admitted"])
def test_frozen_hgt_risk_replay_preserves_admitted_projection(tmp_path, limit):
    from exact.repair.model import RepairModel
    from exact.repair.pipeline import (
        FrozenPlanRisk,
        _publish_risk_snapshot,
        model_digest,
    )
    from exact.repair.records import ObjectiveV3, promote_input_v3

    p = promote_input_v3(problem())
    support = GraphExplanation(
        "omitted", ("o0",), witness=owl.Class(owl.IRI("urn:outside:Support"))
    )
    base = build_observable_graph(p.objects, feature_schema=FEATURE_SCHEMA_V3)
    limits = {
        "admitted": {},
        "explanation": {"max_explanations": 0},
        "nodes": {"max_nodes": len(base.nodes)},
        "edges": {"max_edges": len(base.edges)},
    }[limit]
    graph = build_observable_graph(
        p.objects, explanations=(support,), feature_schema=FEATURE_SCHEMA_V3, **limits
    )
    model = RepairModel(
        graph.metadata, hidden_dim=8, heads=2, layers=1, encoder="hgt", dropout=0, revision="v3"
    ).eval()
    path, digest, size = _publish_risk_snapshot(model, str(tmp_path), 10000000)
    risk = FrozenPlanRisk(
        path,
        digest,
        size,
        graph,
        p.objects,
        graph.admitted_supports,
        model_digest(model),
        canonical_hash(graph),
        canonical_hash(p.objects),
    )
    p = replace(p, graph_identity=canonical_hash(graph))
    objective = ObjectiveV3(
        tuple(tuple(0 for _ in obj.candidates) for obj in p.objects),
        pool_hash=canonical_hash(p.objects),
        model_hash=model_digest(model),
    )
    restored = FrozenPlanRisk.from_dict(risk.to_dict(), problem=p, objective=objective)
    assert restored.risk_identity == risk.risk_identity
    assert restored((0, 0, 0)) == pytest.approx(risk((0, 0, 0)))
    changed_support = replace(support, witness=owl.Class(owl.IRI("urn:another:Excluded")))
    if limit == "admitted":
        assert restored.supports == (support,)
        assert restored.to_dict() == risk.to_dict()
        with pytest.raises(ValueError, match="graph admission"):
            replace(restored, supports=(changed_support,))
        return
    changed_graph = build_observable_graph(
        p.objects,
        explanations=(changed_support,),
        feature_schema=FEATURE_SCHEMA_V3,
        **limits,
    )
    assert (changed_graph.nodes, changed_graph.edges, changed_graph.admitted_supports) == (
        graph.nodes,
        graph.edges,
        graph.admitted_supports,
    )
    # Omitted node IDs are audit metadata; their names may differ while the
    # complete neural projection and effective omission features remain fixed.
    replayed = replace(restored, graph=changed_graph, graph_hash=canonical_hash(changed_graph))
    assert replayed((0, 0, 0)) == pytest.approx(restored((0, 0, 0)))
    with pytest.raises(ValueError, match="graph admission"):
        replace(risk, supports=(support,))


def test_disabled_auxiliaries_skip_readouts_and_keep_label_coverage(monkeypatch):
    import tools.repair.train as train
    from exact.repair.model import RepairModel

    cases = generate_corpus(
        split_counts={"train": 1, "development": 1, "test": 0},
        siblings_per_parent=1,
        families=("range",),
    )
    rows = []
    for case in cases:
        cache = cache_for(case)
        labels = tuple(
            train._assignment_label(case, label.assignment, train.DEFAULT_PROFILE)
            for label in cache.labels
        )
        rows.append((case, replace(cache, labels=labels)))

    def forbidden(*args, **kwargs):
        raise AssertionError("disabled auxiliary readout invoked")

    support_readout = RepairModel.support_violation_logit
    monkeypatch.setattr(RepairModel, "plan_risk_logit", forbidden)
    monkeypatch.setattr(RepairModel, "support_violation_logit", forbidden)
    monkeypatch.setattr(train, "decoded_development", lambda *a, **k: {"exact_regret": 0.0})
    monkeypatch.setattr(
        train,
        "generated_development",
        lambda *a, **k: dict(
            useful_candidate_coverage=1.0,
            decoded=dict(status="verified", selected_utility=0.5, checks=1),
        ),
    )
    _, report = train.train_cases(
        [row for row in rows if row[0].split == "train"],
        [row for row in rows if row[0].split == "development"],
        epochs=1,
        hidden_dim=8,
        heads=2,
        layers=1,
        encoder="hgt",
        dropout=0,
        proposal_arm="bounded_enumeration",
        sampled_assignments=0,
        plan_risk=False,
        support_enabled=False,
    )
    assert report["support_enabled"] is False
    assert any(row["support_available"] > 0 for row in report["loss_eligibility"].values())
    assert all(
        row["support_optimized"] == 0 and row["risk"] == 0
        for row in report["loss_eligibility"].values()
    )
    monkeypatch.setattr(RepairModel, "support_violation_logit", support_readout)
    _, enabled = train.train_cases(
        [row for row in rows if row[0].split == "train"],
        [row for row in rows if row[0].split == "development"],
        epochs=1,
        hidden_dim=8,
        heads=2,
        layers=1,
        encoder="hgt",
        dropout=0,
        proposal_arm="bounded_enumeration",
        sampled_assignments=0,
        plan_risk=False,
        support_enabled=True,
    )
    assert enabled["support_enabled"] is True
    assert any(row["support_optimized"] > 0 for row in enabled["loss_eligibility"].values())


def test_collection_committed_attempts_resume_without_reverification():
    import copy

    state = {}
    calls = []
    options = dict(
        case_id="case",
        parent_group_id="p",
        split="train",
        hashes={"inventory": "pool"},
        model_hash="model",
        round_id="1",
        max_assignments=8,
        deadline_seconds=5,
        seed=4,
        plan_quotas=dict(utility=0, proposal=0, diversity=4, quartet=4, uniform=0),
    )

    def label(assignment):
        calls.append(assignment)
        return RepairLabel(assignment, True, 1.0, 0)

    def save(value):
        state.update(copy.deepcopy(value))
        if len(value["attempts"]) == 2:
            raise InterruptedError("committed collection prefix")

    with pytest.raises(InterruptedError):
        collect_sampled_repairs((3, 3), label, progress=save, **options)
    committed = tuple(calls)
    resumed = collect_sampled_repairs((3, 3), label, resume_state=state, **options)
    assert len(calls) == len(set(calls)) and calls[: len(committed)] == list(committed)
    baseline = collect_sampled_repairs((3, 3), lambda a: RepairLabel(a, True, 1.0, 0), **options)
    assert resumed.attempts == baseline.attempts
    assert resumed.sampler_hash == baseline.sampler_hash


@pytest.mark.parametrize("cumulative_exhausted", [False, True])
def test_deadline_after_one_update_does_not_advance_epoch(
    tmp_path, monkeypatch, cumulative_exhausted
):
    from types import SimpleNamespace

    import torch

    import tools.repair.train as train

    cases = generate_corpus(
        split_counts={"train": 3, "development": 1, "test": 0},
        siblings_per_parent=1,
        families=("range",),
    )
    training = [(c, cache_for(c)) for c in cases if c.split == "train"]
    development = [(c, cache_for(c)) for c in cases if c.split == "development"]
    monkeypatch.setattr(train, "decoded_development", lambda *a, **k: {"exact_regret": 0.0})
    monkeypatch.setattr(
        train,
        "generated_development",
        lambda *a, **k: dict(
            useful_candidate_coverage=1.0,
            decoded=dict(status="verified", selected_utility=0.5, checks=1),
        ),
    )
    now = [0.0]
    monkeypatch.setattr(train, "time", SimpleNamespace(monotonic=lambda: now[0]))
    options = dict(
        epochs=1,
        hidden_dim=8,
        heads=2,
        layers=0,
        encoder="none",
        dropout=0.2,
        proposal_arm="bounded_enumeration",
        sampled_assignments=0,
        plan_risk=False,
        deadline_seconds=60,
        total_training_seconds=60 if cumulative_exhausted else 180,
        batch_cases=1,
    )
    baseline, _ = train.train_cases(training, development, **options)
    path = tmp_path / "state.pt"
    save = train.save_training_state

    def expire(destination, state):
        save(destination, state)
        if state["phase"] == "train" and state["next_offset"] == 1:
            now[0] = 100.0

    monkeypatch.setattr(train, "save_training_state", expire)
    with pytest.raises(TimeoutError, match="exact phase checkpoint"):
        train.train_cases(training, development, checkpoint_path=path, **options)
    state = torch.load(path, weights_only=True)
    assert (state["next_epoch"], state["next_offset"], state["optimized"], state["phase"]) == (
        0,
        1,
        1,
        "train",
    )
    assert not state["history"] and state["stale_evaluations"] == 0
    assert state["elapsed_seconds"] == 100.0
    assert state["total_training_seconds"] == options["total_training_seconds"]
    now[0] = 0.0
    monkeypatch.setattr(train, "save_training_state", save)
    if cumulative_exhausted:
        with pytest.raises(TimeoutError, match="exact phase checkpoint"):
            train.train_cases(training, development, checkpoint_path=path, **options)
        restored = torch.load(path, weights_only=True)
        assert restored["next_offset"] == 1 and restored["optimized"] == 1
        assert restored["elapsed_seconds"] >= state["elapsed_seconds"]
        with pytest.raises(ValueError, match="incompatible"):
            train.train_cases(
                training,
                development,
                checkpoint_path=path,
                **{**options, "total_training_seconds": 180},
            )
        return
    resumed, _ = train.train_cases(training, development, checkpoint_path=path, **options)
    assert all(
        torch.equal(value, resumed.state_dict()[key])
        for key, value in baseline.state_dict().items()
    )


def test_acquisition_resume_retains_preparation_target_and_committed_labels(tmp_path, monkeypatch):
    import tools.repair.train as train
    from exact.repair.pipeline import FrozenNeuralRound
    from exact.repair.records import ObjectiveV3
    from exact.repair.workers import CallResult

    cases = generate_corpus(
        split_counts={"train": 1, "development": 1, "test": 0},
        siblings_per_parent=1,
        families=("range",),
    )
    rows = []
    for case in cases:
        target = SemanticTargetSpec(canonical_hash(case.probes), 2.0, 3.0)
        cache = cache_for(case)
        labels = tuple(
            train._assignment_label(
                case, label.assignment, train.DEFAULT_PROFILE, semantic_target=target
            )
            for label in cache.labels
        )
        rows.append(
            (
                case,
                replace(
                    cache,
                    labels=labels,
                    hashes=tuple(
                        sorted(
                            {**dict(cache.hashes), "semantic_target": target.content_hash}.items()
                        )
                    ),
                ),
            )
        )
    frozen_calls = []
    label_calls = []

    def freeze(problem, model, **options):
        frozen_calls.append(options)
        assert {
            key: options[key]
            for key in ("max_graph_nodes", "max_graph_edges", "max_explanations", "max_text_tokens")
        } == dict(max_graph_nodes=500, max_graph_edges=2000, max_explanations=0, max_text_tokens=7)
        objective = ObjectiveV3(
            tuple(tuple(0 for _ in obj.candidates) for obj in problem.objects),
            pool_hash=canonical_hash(problem.objects),
        )
        return CallResult("complete", FrozenNeuralRound(problem, objective, "graph", "model", ()))

    monkeypatch.setattr(train, "_freeze_training_model", freeze)

    def bounded(function, *args, **kwargs):
        kwargs.pop("timeout", None)
        if function is train._assignment_label:
            label_calls.append((args[0].case_id, args[1]))
            assert kwargs["semantic_target"].desired_family_weight == 2.0
            assert kwargs["semantic_target"].false_positive_weight == 3.0
        return CallResult("complete", function(*args, **kwargs))

    monkeypatch.setattr(train, "bounded_call", bounded)
    monkeypatch.setattr(train, "decoded_development", lambda *a, **k: {"exact_regret": 0.0})

    def development(*args, **options):
        assert options["semantic_target"].desired_family_weight == 2.0
        return dict(
            useful_candidate_coverage=1.0,
            decoded=dict(status="verified", selected_utility=2.0, checks=1),
        )

    monkeypatch.setattr(train, "generated_development", development)
    options = dict(
        epochs=1,
        hidden_dim=8,
        heads=2,
        layers=0,
        encoder="none",
        dropout=0,
        proposal_arm="bounded_enumeration",
        sampled_assignments=4,
        plan_risk=False,
        max_graph_nodes=500,
        max_graph_edges=2000,
        max_explanations=0,
        max_text_tokens=7,
        desired_family_weight=2.0,
        false_positive_weight=3.0,
        deadline_seconds=60,
    )
    training = [row for row in rows if row[0].split == "train"]
    dev = [row for row in rows if row[0].split == "development"]
    path = tmp_path / "state.pt"
    save = train.save_training_state
    triggered = []

    def interrupt(destination, state):
        save(destination, state)
        collection = state.get("pending_acquisition", {}).get("collection_state", {})
        if len(collection.get("labels", ())) == 1 and not triggered:
            triggered.append(True)
            raise InterruptedError("durable acquisition label")

    monkeypatch.setattr(train, "save_training_state", interrupt)
    with pytest.raises(InterruptedError):
        train.train_cases(training, dev, checkpoint_path=path, **options)
    monkeypatch.setattr(train, "save_training_state", save)
    _, report = train.train_cases(training, dev, checkpoint_path=path, **options)
    assert len(frozen_calls) == 1 and len(label_calls) == len(set(label_calls))
    assert report["acquisition_rounds"]
    with pytest.raises(ValueError, match="incompatible|dependencies"):
        train.train_cases(
            training, dev, checkpoint_path=path, **{**options, "desired_family_weight": 4.0}
        )

    with pytest.raises(ValueError, match="incompatible"):
        train.train_cases(training, dev, checkpoint_path=path, **{**options, "max_text_tokens": 8})


def test_weighted_original_acquisition_and_generated_development_share_target(monkeypatch):
    import tools.repair.train as train
    from exact.repair.learning import generated_checkpoint_criterion
    from exact.repair.pipeline import FrozenNeuralRound
    from exact.repair.records import ObjectiveV3
    from exact.repair.workers import CallResult

    case = generate_corpus(
        split_counts={"train": 0, "development": 1, "test": 0},
        siblings_per_parent=1,
        families=("range",),
    )[0]
    from exact.repair.learning import TeacherProbe

    # A deliberately competing unwanted family makes both nonunit weights
    # observable in the same complete native semantic vector.
    case = replace(
        case,
        probes=case.probes
        + (TeacherProbe("unwanted-audit", case.probes[0].axiom, "unwanted_audit", desired=False),),
    )
    target = SemanticTargetSpec(canonical_hash(case.probes), 2.5, 3.5)

    # Use native OWL labels through the actual original teacher wrapper. Only
    # worker transport is made synchronous to isolate weight propagation.
    def bounded(function, *args, **options):
        options.pop("timeout", None)
        return CallResult("complete", function(*args, **options))

    monkeypatch.setattr(train, "bounded_call", bounded)
    cache = train.label_case(
        case, desired_family_weight=2.5, false_positive_weight=3.5, deadline_seconds=30
    )
    expected = next(
        row
        for row in cache.labels
        if row.usable and any(outcome.desired and outcome.credit for outcome in row.semantic_vector)
    )
    acquired = train._assignment_label(
        case, expected.assignment, train.DEFAULT_PROFILE, semantic_target=target
    )
    assert expected.semantic_vector == acquired.semantic_vector
    assert (expected.feasible, expected.benefit, expected.cost) == (
        acquired.feasible,
        acquired.benefit,
        acquired.cost,
    )
    assert {outcome.desired for outcome in expected.semantic_vector} == {True, False}

    objective = ObjectiveV3(
        tuple(tuple(0 for _ in obj.candidates) for obj in case.problem.objects),
        pool_hash=canonical_hash(case.problem.objects),
    )
    frozen = FrozenNeuralRound(case.problem, objective, "graph", "model", ())
    monkeypatch.setattr(
        train, "_freeze_training_model", lambda *a, **k: CallResult("complete", frozen)
    )
    from types import SimpleNamespace

    import exact.repair.pipeline as pipeline

    monkeypatch.setattr(
        pipeline,
        "repair_neural_round",
        lambda *a, **k: SimpleNamespace(
            assignment=expected.assignment,
            logical_status="VERIFIED_FEASIBLE",
            search_status="optimal",
            checks=1,
            solves=1,
            lower_bound=0,
            upper_bound=0,
            first_verified_seconds=0.0,
            resource_counters=(),
            failures=(),
            stage_seconds=(),
        ),
    )
    report = train.generated_development(
        None,
        case,
        cache,
        seconds=30,
        semantic_target=target,
        selection_options={"risk_ordering": False},
    )
    assert report["semantic_target_hash"] == target.content_hash
    assert report["decoded"]["semantic_benefit"] == acquired.benefit
    assert report["decoded"]["cost"] == acquired.cost
    assert report["decoded"]["selected_utility"] == acquired.benefit - acquired.cost
    other = train.generated_development(
        None,
        case,
        cache,
        seconds=30,
        semantic_target=replace(target, desired_family_weight=0.5),
        selection_options={"risk_ordering": False},
    )
    assert other["semantic_target_hash"] != report["semantic_target_hash"]
    assert other["decoded"]["selected_utility"] != report["decoded"]["selected_utility"]
    assert generated_checkpoint_criterion([other]) != generated_checkpoint_criterion([report])


@pytest.mark.parametrize("limit", ["explanation", "nodes", "edges"])
def test_enabled_hgt_training_preserves_truncated_support_projection(monkeypatch, limit):
    import torch

    import exact.repair.retrieval as retrieval_module
    import tools.repair.train as train
    from exact.repair.model import RepairModel

    cases = generate_corpus(
        split_counts={"train": 1, "development": 1, "test": 0},
        siblings_per_parent=1,
        families=("range",),
    )
    original_retrieve = retrieval_module.retrieve_vocabulary
    defaults = EffectivePreparation()
    graphs = [defaults.graph(case.problem, original_retrieve(case.problem)) for case in cases]
    graph_limits = {
        "explanation": {"max_explanations": 0},
        "nodes": {"max_graph_nodes": max(len(graph.nodes) for graph in graphs)},
        "edges": {"max_graph_edges": max(len(graph.edges) for graph in graphs)},
    }[limit]
    outside = tuple(owl.Class(owl.IRI(f"urn:excluded:{i}")) for i in range(64))

    def retrieve(problem, **options):
        result = original_retrieve(problem, **options)
        support = GraphExplanation(
            "over-budget",
            (problem.objects[0].object_id,),
            tuple(owl.SubClassOf(left, right) for left, right in zip(outside, outside[1:])),
            outside[0],
        )
        return replace(result, explanations=(support,))

    monkeypatch.setattr(retrieval_module, "retrieve_vocabulary", retrieve)
    observed = []
    readout = RepairModel.plan_risk_logit

    def risk(self, objects, memory, assignment, **options):
        assert options["supports"] == memory.graph.admitted_supports == ()
        assert memory.graph.omitted_supports == ("over-budget",)
        observed.append(memory.graph)
        return readout(self, objects, memory, assignment, **options)

    monkeypatch.setattr(RepairModel, "plan_risk_logit", risk)
    monkeypatch.setattr(train, "decoded_development", lambda *a, **k: {"exact_regret": 0.0})
    monkeypatch.setattr(
        train,
        "generated_development",
        lambda *a, **k: dict(
            useful_candidate_coverage=1.0,
            decoded=dict(status="verified", selected_utility=0.5, checks=1),
        ),
    )
    model, report = train.train_cases(
        [(case, cache_for(case)) for case in cases if case.split == "train"],
        [(case, cache_for(case)) for case in cases if case.split == "development"],
        epochs=1,
        hidden_dim=8,
        heads=2,
        layers=1,
        encoder="hgt",
        dropout=0,
        proposal_arm="bounded_enumeration",
        sampled_assignments=0,
        plan_risk=True,
        **graph_limits,
    )
    assert observed and report["status"] == "complete"
    assert any(row["risk"] > 0 for row in report["loss_eligibility"].values())
    assert any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and parameter.grad.abs().sum() > 0
        for parameter in model.risk_head.parameters()
    )


def test_sampler_deadline_retains_unvisited_denominators_and_masks_partial_quartets(monkeypatch):
    from types import SimpleNamespace

    import exact.repair.learning as learning

    now = [0.0]
    calls = []
    monkeypatch.setattr(learning, "time", SimpleNamespace(monotonic=lambda: now[0]))

    def label(assignment):
        calls.append(assignment)
        if len(calls) == 2:
            now[0] = 2.0
        return RepairLabel(assignment, True, 1.0, 0.0)

    result = collect_sampled_repairs(
        (3, 3),
        label,
        case_id="deadline",
        parent_group_id="p",
        split="train",
        hashes={"inventory": "pool"},
        model_hash="model",
        round_id="0",
        max_assignments=8,
        deadline_seconds=1.0,
        seed=13,
        plan_quotas=dict(utility=0, proposal=0, diversity=4, quartet=4, uniform=0),
    )
    counts = {name: dict(row) for name, row in result.strata}
    assert result.stop_reason == "deadline" and len(result.attempts) == 8
    assert sum(row["attempted"] for row in counts.values()) == 2
    assert sum(row["unvisited"] for row in counts.values()) == 6
    assert counts["diversity"]["scheduled"] == counts["quartet"]["scheduled"] == 4
    assert all(
        row["quartet_complete"] is False for row in result.attempts if row["stratum"] == "quartet"
    )
    assert len(calls) == len(set(calls)) == 2


def test_unavailable_complete_plan_proposals_keep_denominator_without_empty_training(monkeypatch):
    import tools.repair.train as train
    from exact.repair.pipeline import FrozenNeuralRound
    from exact.repair.records import ObjectiveV3
    from exact.repair.workers import CallResult

    cases = generate_corpus(
        split_counts={"train": 1, "development": 1, "test": 0},
        siblings_per_parent=1,
        families=("range",),
    )

    def freeze(problem, *args, **options):
        objective = ObjectiveV3(
            tuple(tuple(0 for _ in obj.candidates) for obj in problem.objects),
            pool_hash=canonical_hash(problem.objects),
        )
        return CallResult("complete", FrozenNeuralRound(problem, objective, "graph", "model", ()))

    def unexpected_label(*args, **kwargs):
        raise AssertionError("Unavailable proposals must not manufacture assignment labels")

    monkeypatch.setattr(train, "_freeze_training_model", freeze)
    monkeypatch.setattr(train, "_assignment_label", unexpected_label)
    monkeypatch.setattr(train, "decoded_development", lambda *a, **k: {"exact_regret": 0.0})
    monkeypatch.setattr(
        train,
        "generated_development",
        lambda *a, **k: dict(
            useful_candidate_coverage=1.0,
            decoded=dict(status="verified", selected_utility=0.5, checks=1),
        ),
    )
    _, report = train.train_cases(
        [(case, cache_for(case)) for case in cases if case.split == "train"],
        [(case, cache_for(case)) for case in cases if case.split == "development"],
        epochs=1,
        hidden_dim=8,
        heads=2,
        layers=0,
        encoder="none",
        dropout=0,
        proposal_arm="bounded_enumeration",
        sampled_assignments=4,
        plan_risk=True,
        collection_options={
            "plan_quotas": dict(utility=0, proposal=4, diversity=0, quartet=0, uniform=0)
        },
    )
    assert report["status"] == "complete"
    assert report["history"][0]["optimized_cases"] == 1
    acquired = report["acquisition_rounds"][0]
    counts = {name: dict(row) for name, row in acquired["strata"]}
    assert counts["proposal"]["scheduled"] == counts["proposal"]["unavailable"] == 4
    assert not acquired["cache"]["labels"]


def test_v3_freeze_requires_qualified_effective_preparation_for_supplied_graph():
    from exact.repair.model import RepairModel
    from exact.repair.pipeline import freeze_neural_round
    from exact.repair.records import promote_input_v3
    from exact.repair.retrieval import retrieve_vocabulary

    p = promote_input_v3(problem())
    settings = EffectivePreparation(
        4096, 32768, 64, 128, pair_max_pairs=None, pair_max_factors=None
    )
    graph = settings.graph(p, retrieve_vocabulary(p))
    model = RepairModel(
        graph.metadata,
        hidden_dim=8,
        heads=2,
        layers=0,
        dropout=0,
        revision="v3",
        plan_risk=False,
    )
    options = dict(draws_per_object=0, proposal_arm="bounded_enumeration")
    with pytest.raises(ValueError, match="Rebuild unqualified graph"):
        freeze_neural_round(p, model, graph=replace(graph, preparation_identity=""), **options)
    with pytest.raises(ValueError, match="incompatible effective preparation"):
        freeze_neural_round(p, model, graph=graph, max_text_tokens=1, **options)
    frozen = freeze_neural_round(p, model, graph=graph, **options)
    assert frozen.proposal_reports[0]["preparation_identity"] == settings.content_hash
