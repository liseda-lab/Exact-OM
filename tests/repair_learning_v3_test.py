"""XR-2.1 train/inference parity and label-routing conformance (no live studies)."""

import itertools
from dataclasses import replace

import pyowl_core as owl
import pytest

from exact.repair.candidates import make_candidate
from exact.repair.graph import (
    FEATURE_SCHEMA_V3,
    GraphExplanation,
    build_observable_graph,
    select_interaction_pairs,
)
from exact.repair.learning import (
    RepairLabel,
    TeacherCache,
    collect_sampled_repairs,
    covered_proposal_loss,
    generated_checkpoint_criterion,
    interaction_loss,
    risk_loss,
)
from exact.repair.records import PolicyV2, RepairInputV2, RevisionObjectV2


def problem():
    classes = [owl.Class(owl.IRI("urn:test:" + name)) for name in "ABCDEF"]
    objects = []
    for i in range(3):
        axiom = owl.SubClassOf(classes[2 * i], classes[2 * i + 1])
        name = f"o{i}"
        objects.append(
            RevisionObjectV2(
                name,
                "mapping",
                (axiom,),
                (make_candidate(name, (axiom,), ("keep",)), make_candidate(name, (), ("delete",))),
            )
        )
    return RepairInputV2((), tuple(objects), PolicyV2(tuple(classes)))


def test_shared_pair_selector_accounts_explanation_candidate_links_and_factor_caps():
    p = problem()
    bridge = make_candidate(
        "o1",
        (owl.SubClassOf(p.policy.monitored_classes[2], p.policy.monitored_classes[0]),),
        ("replace_endpoint",),
    )
    p = replace(
        p,
        objects=(
            p.objects[0],
            replace(p.objects[1], candidates=(*p.objects[1].candidates, bridge)),
            p.objects[2],
        ),
    )
    explanation = GraphExplanation("witness", ("o0", "o2"))
    graph = build_observable_graph(
        p.objects, explanations=(explanation,), feature_schema=FEATURE_SCHEMA_V3
    )
    first = select_interaction_pairs(
        p, graph=graph, explanations=(explanation,), per_object_limit=2
    )
    second = select_interaction_pairs(
        p, graph=graph, explanations=(explanation,), per_object_limit=2
    )
    assert first.content_hash == second.content_hash
    assert (0, 2) in first.pairs and (0, 1) in first.pairs
    assert "candidate_signature" in next(
        reasons for i, j, reasons in first.reasons if (i, j) == (0, 1)
    )
    capped = select_interaction_pairs(
        p, graph=graph, explanations=(explanation,), per_object_limit=1
    )
    assert capped.pairs == ((0, 2),)
    assert dict(capped.omissions)["candidate_signature"] > 0
    assert (
        select_interaction_pairs(p, graph=graph, explanations=(explanation,), max_factors=3).pairs
        == ()
    )


def test_v3_feature_states_identifier_mask_and_future_evidence_rejection():
    p = problem()

    def features(payload):
        g = build_observable_graph(
            p.objects, evidence={"o0": payload}, feature_schema=FEATURE_SCHEMA_V3
        )
        return next(n.features for n in g.nodes if n.node_id == "evidence:object:o0")

    states = [
        features({"score": value})
        for value in (0.0, None, {"status": "unsupported"}, {"status": "truncated"})
    ]
    assert len(set(states)) == 4
    assert features({"score": 0.0, "path": "/test/parent17"}) == features({"score": 0.0})
    for evidence in (
        {"nested": {"intended_assignment": [1]}},
        {"available_before_decision": False},
    ):
        with pytest.raises(ValueError):
            features(evidence)


def test_quartet_masks_and_higher_order_risk_do_not_make_pair_negatives():
    torch = pytest.importorskip("torch")
    assignments = list(itertools.product(range(2), repeat=2))
    for values, contrast in (([0, 0, 0, 1], 1), ([0, 1, 1, 1], -1)):
        labels = [RepairLabel(a, True, b, 0) for a, b in zip(assignments, values)]
        result = interaction_loss(
            torch.tensor(values, dtype=torch.float32, requires_grad=True), labels
        )
        assert result["targets"] == (contrast,) and result["loss"].item() == 0
        labels[-1] = RepairLabel(assignments[-1], False, None, 0)
        assert interaction_loss(torch.zeros(4), labels)["eligible"] == 0
    labels = [
        RepairLabel((1, 1, 1), False, None, 0),
        RepairLabel((1, 1, 0), True, 1, 0),
        RepairLabel((0, 0, 0), None, None, 0),
    ]
    logits = torch.zeros(3, requires_grad=True)
    loss = risk_loss(logits, labels)
    loss["loss"].backward()
    assert loss["eligible"] == 2 and logits.grad[0] < 0  # failure pushes logit upward
    assert logits.grad[2] == 0


def test_sample_proposal_mass_masks_are_explicit():
    torch = pytest.importorskip("torch")
    probabilities = torch.tensor([-0.2, float("-inf")], requires_grad=True)
    result = covered_proposal_loss(
        probabilities, [0.25, 0.75], target_kind="sample_conditioned", elementary=[True, False]
    )
    assert result["missing_target_mass"] == 0.75 and result["eligible"] == 0
    assert result["partitions"]["complex"]["missing_mass"] == 0.75
    assert result["loss"].isfinite() and not result["renormalized"]
    projected = covered_proposal_loss(
        probabilities, [0.25, 0.75], target_kind="sample_conditioned", reachable_subset=True
    )
    assert projected["renormalized"] and projected["loss"].item() == pytest.approx(0.2)


def test_sample_collection_retains_unknowns_split_and_new_inventory_identity():
    round = collect_sampled_repairs(
        (2, 2),
        lambda a: RepairLabel(a, None, None, 0),
        case_id="case",
        parent_group_id="parent",
        split="train",
        hashes={
            **{
                key: key
                for key in (
                    "input",
                    "patch",
                    "policy",
                    "query",
                    "backend",
                    "profile",
                    "semantic_target",
                )
            },
            "inventory": "new",
        },
        model_hash="frozen",
        round_id="0",
        max_assignments=6,
        deadline_seconds=1,
        seed=1,
        proposed=(((1, 1), "learned"),),
    )
    assert not round.cache.complete and round.cache.coverage["unknown_policy"] == len(
        round.cache.labels
    )
    assert round.inventory_hash == "new" and round.requested == 6
    assert all(prob is None for _, stratum, prob in round.selections if stratum != "uniform")
    with pytest.raises(ValueError, match="training-only"):
        collect_sampled_repairs(
            (2,),
            lambda a: None,
            case_id="c",
            parent_group_id="p",
            split="test",
            hashes={"inventory": "x"},
            model_hash="m",
            round_id="1",
            max_assignments=1,
            deadline_seconds=1,
        )


def test_generated_checkpoint_ignores_better_cached_regret_and_preserves_denominator():
    def report(quality, regret):
        return {
            "decoded": {"status": "verified", "selected_utility": quality, "checks": 2},
            "exact_regret": regret,
        }

    worse = generated_checkpoint_criterion([report(0.1, 0)])
    better = generated_checkpoint_criterion([report(0.9, 100)])
    assert better < worse
    assert generated_checkpoint_criterion([report(1, 0), {"status": "generation_timeout"}]) is None


def test_v3_model_gauge_risk_and_frozen_prefix_are_separate():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from exact.repair.model import RepairModel

    p = problem()
    graph = build_observable_graph(p.objects, feature_schema=FEATURE_SCHEMA_V3)
    model = RepairModel(
        graph.metadata,
        hidden_dim=8,
        heads=2,
        layers=0,
        dropout=0,
        encoder="none",
        pairwise=True,
        revision="v3",
    ).eval()
    memory = model.encode(graph)
    unary, pairs = model.score_inventory(p.objects, memory, interaction_pairs=((0, 1),))
    assert all(row[0].item() == 0 for row in unary)
    assert all(
        value.item() == pytest.approx(0)
        for (i, a, j, b), value in pairs.items()
        if a == 0 or b == 0
    )
    risk = model.plan_risk_logit(p.objects, memory, (0, 0, 0))
    risk.backward()
    assert model.risk_head[0].weight.grad is not None
    prefix = model.plan_context(p.objects, memory, (None, 1, None), target_object_id="o0")
    assert prefix.shape == (8,)


def test_conflict_families_have_actual_three_way_supports_and_coherent_cycle_control():
    from tools.repair.corpus import generate_corpus
    from tools.repair.train import DEFAULT_PROFILE, _assignment_label

    cases = generate_corpus(
        parents_per_family=1,
        siblings_per_parent=1,
        families=("conflicts_higher_order", "conflicts_overlap", "coherent_cycle"),
    )
    for case in cases:
        zero = tuple(0 for _ in case.problem.objects)
        label = _assignment_label(case, zero, DEFAULT_PROFILE)
        assert label.feasible is (case.family == "coherent_cycle")
        assert _assignment_label(case, case.intended_assignment, DEFAULT_PROFILE).feasible is True
        if case.family == "conflicts_higher_order":
            assert len(case.problem.objects) == 3
            for i in range(3):
                assignment = list(zero)
                assignment[i] = 1
                assert _assignment_label(case, tuple(assignment), DEFAULT_PROFILE).feasible is True


def test_v3_training_selects_generated_repairs_and_saves_resumable_acquisition(tmp_path):
    pytest.importorskip("torch_geometric")
    import torch

    from tests.repair_training_completion_test import cache_for
    from tools.repair.corpus import generate_corpus
    from tools.repair.train import train_cases

    cases = generate_corpus(
        split_counts={"train": 1, "development": 1, "test": 0},
        siblings_per_parent=1,
        families=("range",),
    )
    train = next(c for c in cases if c.split == "train")
    dev = next(c for c in cases if c.split == "development")
    checkpoint = tmp_path / "state.pt"
    _, report = train_cases(
        [(train, cache_for(train))],
        [(dev, cache_for(dev))],
        epochs=1,
        hidden_dim=8,
        heads=2,
        layers=0,
        encoder="none",
        mixtures=1,
        max_depth=1,
        max_constructors=1,
        decode_seconds=30,
        development_draws_per_object=0,
        sampled_assignments=4,
        deadline_seconds=180,
        checkpoint_path=checkpoint,
    )
    assert report["schema"] == "exact-repair/training/v3"
    assert report["checkpoint_criterion"] == "generated_pool_verified_quality_effort"
    assert report["acquisition_rounds"] and report["loss_eligibility"][train.case_id]["risk"] > 0
    saved = torch.load(checkpoint, weights_only=True)
    assert saved["schema"] == "exact-repair/training-state/v3"
    assert saved["sampled_training"] and saved["acquisition_epoch"] == 0


def test_conditional_joint_distinguishes_equal_marginal_teachers():
    torch = pytest.importorskip("torch")
    from exact.repair.learning import conditional_proposal_loss, teacher_marginals

    assignments = list(itertools.product(range(2), repeat=2))

    def cache(diagonal):
        return TeacherCache(
            (2, 2),
            tuple(RepairLabel(a, True, float((a[0] == a[1]) == diagonal), 0) for a in assignments),
            True,
            "complete",
            (),
            0,
        )

    diagonal, anti = cache(True), cache(False)
    assert teacher_marginals(diagonal) == teacher_marginals(anti)
    parameter = torch.tensor(0.4, requires_grad=True)

    def probability(i, prefix, choice):
        if not prefix:
            return parameter * 0 - torch.log(parameter.new_tensor(2.0))
        same = choice == prefix[0]
        return torch.nn.functional.logsigmoid(parameter if same else -parameter)

    left = conditional_proposal_loss(diagonal, probability)
    right = conditional_proposal_loss(anti, probability)
    assert left["target_kind"] == "exact_conditional_joint" and left["loss"] != right["loss"]
    grad_a = torch.autograd.grad(left["loss"], parameter, retain_graph=True)[0]
    grad_b = torch.autograd.grad(right["loss"], parameter)[0]
    assert grad_a < 0 and grad_b > 0


def test_generated_development_uses_independent_exact_plan_fidelity_label(tmp_path, monkeypatch):
    import tools.repair.train as train
    from exact.repair.pipeline import FrozenNeuralRound
    from exact.repair.records import ObjectiveV3, canonical_hash
    from exact.repair.workers import CallResult
    from tests.repair_review_semantic_test import qualified_offline_fixture
    from tools.repair.corpus import GeneratedCase

    observed, packet, comparison = qualified_offline_fixture(tmp_path, monkeypatch, "development")
    case = GeneratedCase(
        observed.case_id,
        observed.structural_parent,
        "fidelity_fixture",
        "development",
        observed.problem,
        observed.probes,
        (0,),
        13,
        False,
        schema_revision="v3",
    )
    objective = ObjectiveV3(((10, 0, 0),), pool_hash=canonical_hash(case.problem.objects))
    frozen = FrozenNeuralRound(case.problem, objective, "graph", "model", ())
    monkeypatch.setattr(
        train, "_freeze_training_model", lambda *args, **kwargs: CallResult("complete", frozen)
    )
    cache = TeacherCache(
        (3,),
        tuple(RepairLabel((i,), True, float(i == 0), 0) for i in range(3)),
        True,
        "complete",
        (),
        0,
    )
    report = train.generated_development(
        None,
        case,
        cache,
        seconds=15,
        target_basis="ai_weak",
        fidelity_evaluator_labels=((packet, comparison),),
        selection_options={"risk_ordering": False},
    )
    assert report["decoded"]["status"] == "verified"
    assert report["decoded"]["semantic_benefit"] == 0.75
    report = train.generated_development(
        None,
        case,
        cache,
        seconds=15,
        target_basis="ai_weak",
        fidelity_evaluator_labels=(),
        selection_options={"risk_ordering": False},
    )
    assert report["decoded"]["status"] == "unknown_selected_labels"
    assert report["decoded"]["selected_utility"] is None
