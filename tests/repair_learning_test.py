"""Small XR-2 teacher and graph/model acceptance tests, without research experiments."""

from dataclasses import replace

import pyowl_core as owl
import pytest

from exact.repair.graph import (
    GraphExplanation,
    build_observable_graph,
    structural_id,
    validate_observable_evidence,
)
from exact.repair.learning import (
    RepairLabel,
    TeacherProbe,
    benefit_losses,
    enumerate_teacher,
    evaluate_teacher,
    feasible_anchor,
    grouped_split,
    proposal_loss,
    teacher_marginals,
)
from exact.repair.records import ReplacementCandidateV2, RevisionObjectV2


def cls(name):
    return owl.Class(owl.IRI(f"urn:learning:{name}"))


class Oracle:
    def __init__(self, consistent=True, entailment=True, unsat=(), unknown=()):
        self.consistency = consistent
        self.entailment = entailment
        self.unsat = set(unsat)
        self.unknown = set(unknown)
        self.queries = []

    def consistent(self):
        return self.consistency

    def entails(self, axiom):
        return self.entailment

    def satisfiable(self, expression):
        self.queries.append(expression)
        return None if expression in self.unknown else expression not in self.unsat


def test_disjointness_requires_operands_and_whole_subclass_requires_whole_expression():
    a, b, c = map(cls, "ABC")
    conjunction = owl.ObjectIntersectionOf((a, b))
    oracle = Oracle(unsat=(conjunction,))
    probes = [TeacherProbe("disjoint", owl.DisjointClasses((a, b)), "disjoint")]
    assert evaluate_teacher(oracle, probes).benefit == 1
    assert set(oracle.queries) == {a, b}
    subclass = TeacherProbe("subclass", owl.SubClassOf(conjunction, c), "inclusion")
    assert evaluate_teacher(oracle, [subclass]).benefit == 0


def test_property_and_instance_nonvacuity_are_typed():
    prop = owl.ObjectProperty(owl.IRI("urn:learning:r"))
    domain = TeacherProbe("domain", owl.ObjectPropertyDomain(prop, cls("A")), "property")
    assert isinstance(domain.conditions()[0], owl.ObjectSomeValuesFrom)
    individual = owl.NamedIndividual(owl.IRI("urn:learning:i"))
    probe = TeacherProbe("instance", owl.ClassAssertion(cls("A"), individual), "instance")
    assert probe.conditions() == ()


def test_unknown_masks_entire_family_and_inconsistency_never_earns_credit():
    a, b = map(cls, "AB")
    probes = [TeacherProbe("wanted", owl.SubClassOf(a, b), "subclass")]
    result = evaluate_teacher(Oracle(unknown=(a,)), probes)
    assert result.benefit is None and not result.complete
    assert evaluate_teacher(Oracle(consistent=False), probes).benefit is None
    unwanted = replace(probes[0], desired=False)
    result = evaluate_teacher(Oracle(unsat=(a,)), [unwanted])
    assert result.benefit == -1 and result.outcomes[0].entailed is True


def make_cache(cap=4, unknown=False):
    benefit = {(0, 0): 0.0, (1, 0): 10.0, (0, 1): 10.0, (1, 1): 10.0}

    def label(assignment):
        missing = unknown and assignment == (1, 1)
        return RepairLabel(
            assignment, True, None if missing else benefit[assignment], sum(assignment)
        )

    return enumerate_teacher(
        (2, 2),
        label,
        hashes=dict.fromkeys(("input", "patch", "policy", "query", "inventory", "backend"), "test"),
        max_assignments=cap,
    )


def test_cache_complete_marginals_and_caps():
    cache = make_cache()
    assert cache.complete and cache.coverage["requested"] == 4
    marginal = teacher_marginals(cache)
    assert sum(marginal[0]) == pytest.approx(1)
    assert marginal[0] == marginal[1]
    for incomplete in (make_cache(2), make_cache(unknown=True)):
        assert not incomplete.complete
        with pytest.raises(ValueError, match="complete"):
            teacher_marginals(incomplete)


def test_feasible_anchor_unknown_mask_and_nonadditive_interaction():
    torch = pytest.importorskip("torch")
    cache = make_cache()
    # Semantic benefits 0,10,10,10 imply utility 0,9,9,8 after costs once.
    unary = torch.tensor([0.0, 10.0, 10.0, 20.0], requires_grad=True)
    pair = torch.tensor([0.0, 0.0, 0.0, -10.0])
    losses = benefit_losses(unary + pair, cache.labels)
    assert losses["value"].item() == 0
    assert benefit_losses(unary, cache.labels)["value"].item() > 0
    assert [label.benefit - label.cost for label in cache.labels] == [0, 9, 9, 8]
    labels = [
        RepairLabel((0,), False, None, 0),
        RepairLabel((1,), True, None, 0),
        RepairLabel((2,), True, 0.5, 3),
        RepairLabel((3,), True, 1, 4),
    ]
    assert feasible_anchor(labels) == 2
    predictions = torch.tensor([99.0, 99.0, 0.5, 1.0], requires_grad=True)
    losses = benefit_losses(predictions, labels)
    assert losses["usable"] == 2 and losses["pairs"] == 1
    (losses["value"] + losses["rank"]).backward()
    assert predictions.grad[:2].tolist() == [0.0, 0.0]


def test_proposal_loss_normalization_and_complete_cache_only():
    torch = pytest.importorskip("torch")
    logits = [torch.tensor([0.0, 1.0], requires_grad=True) for _ in range(2)]
    loss = proposal_loss([row.log_softmax(0) for row in logits], make_cache())
    loss.backward()
    assert all(row.grad is not None for row in logits)
    with pytest.raises(ValueError, match="normaliser"):
        proposal_loss(logits, make_cache())
    with pytest.raises(ValueError, match="complete"):
        proposal_loss([row.log_softmax(0) for row in logits], make_cache(2))


def test_splits_are_parent_grouped_deterministic_and_hold_out_mechanisms():
    parents = {f"parent-{i}": "heldout" if i < 2 else "normal" for i in range(40)}
    split = grouped_split(parents, heldout_families=("heldout",))
    assert split == grouped_split(
        dict(reversed(list(parents.items()))), heldout_families=("heldout",)
    )
    assert split["parent-0"] == split["parent-1"] == "test"
    assert len(set(split.values())) == 3


def observed_case():
    a, b, c = map(cls, "ABC")
    original = owl.SubClassOf(a, b)
    keep = ReplacementCandidateV2("m", "keep", (original,))
    changed = ReplacementCandidateV2("m", "changed", (owl.SubClassOf(a, c),), ("replace_endpoint",))
    obj = RevisionObjectV2("m", "mapping", (original,), (keep, changed))
    return obj, c


def test_graph_retains_all_evidence_channels_punning_and_explanation_support():
    obj, c = observed_case()
    prop = owl.ObjectProperty(c.iri)
    explanation = GraphExplanation("x", ("m",), (owl.SubClassOf(c, cls("D")),), cls("D"))
    graph = build_observable_graph(
        [obj],
        evidence={
            "m": {
                "score": 0.7,
                "hierarchy": {"up": [1, 2]},
                "llm": {"brief": "context", "reliability": 0.5},
                "embeddings": [0.1, 0.2],
            }
        },
        explanations=[explanation],
        retrieved_symbols=(c, prop),
    )
    assert structural_id(c) != structural_id(prop)
    features = dict(next(node for node in graph.nodes if node.kind == "evidence").features)
    assert {"observed/score", "observed/hierarchy/up/0", "observed/embeddings/1"} <= features.keys()
    assert ("explanation:x", "support", structural_id(explanation.support_axioms[0])) in graph.edges
    assert any(role == "reverse_sub_class" for _, role, _ in graph.edges)
    minimal = build_observable_graph([obj], retrieved_symbols=(c, prop))
    budgeted = build_observable_graph(
        [obj], retrieved_symbols=(c, prop), explanations=[explanation], max_nodes=len(minimal.nodes)
    )
    assert budgeted.omitted_supports == ("x",)
    assert all(node.node_id != "explanation:x" for node in budgeted.nodes)


@pytest.mark.parametrize(
    "key", ["corruption_trace", "teacher_labels", "reference_membership", "split_id"]
)
def test_nested_hidden_features_are_rejected(key):
    with pytest.raises(ValueError, match="Evaluator-only"):
        validate_observable_evidence({"channel": {key: True}})


@pytest.mark.parametrize("encoder", ["hgt", "rgcn", "none"])
def test_shared_model_candidate_sensitivity_gradients_and_cost_independence(encoder):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from exact.repair.model import RepairModel

    torch.manual_seed(13)
    obj, c = observed_case()
    graph = build_observable_graph([obj], retrieved_symbols=(c,))
    model = RepairModel(
        graph.metadata, hidden_dim=16, heads=2, layers=1, encoder=encoder, dropout=0, pairwise=True
    )
    memory = model.encode(graph)
    keep, changed = obj.candidates
    keep_value, keep_embedding = model.candidate_value(keep, memory)
    changed_value, changed_embedding = model.candidate_value(changed, memory)
    assert keep_embedding.shape == (16,)
    if encoder != "none":
        assert not torch.allclose(keep_embedding, changed_embedding)
        assert not torch.allclose(keep_value, changed_value)
    complete = replace(changed, axioms=keep.axioms + changed.axioms)
    assert not torch.allclose(model.candidate_embedding(complete, memory), keep_embedding)
    with_cost = replace(changed, cost_features=(("delete", 999.0),))
    assert torch.equal(model.candidate_value(with_cost, memory)[0], changed_value)
    context = model.object_context("m", memory)
    pair = model.interaction(keep_embedding, changed_embedding, context)
    assert torch.equal(pair, model.interaction(changed_embedding, keep_embedding, context))
    weights, logits = model.proposal_logits(
        context, torch.stack([keep_embedding, changed_embedding]), mixtures=4
    )
    assert weights.shape == (4,) and logits.shape == (4, 2)
    (keep_value + changed_value + pair + logits.sum()).backward()
    assert model.target_head[0].weight.grad is not None
    if encoder != "none":
        assert any(param.grad is not None for param in model.graph_layers.parameters())


def test_candidate_operand_permutation_and_unretrieved_symbol():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from exact.repair.model import RepairModel

    obj, c = observed_case()
    graph = build_observable_graph([obj], retrieved_symbols=(c,))
    model = RepairModel(graph.metadata, hidden_dim=16, heads=2, layers=0, dropout=0)
    memory = model.encode(graph)
    a = cls("A")
    left = owl.ObjectIntersectionOf((a, c))
    right = owl.ObjectIntersectionOf((c, a))
    assert torch.equal(model.encode_structure(left, memory), model.encode_structure(right, memory))
    with pytest.raises(ValueError, match="retrieved"):
        model.encode_structure(cls("unseen"), memory)


def test_actual_matcher_handoff_global_features_through_neural_freeze():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    pytest.importorskip("pysdd")
    from exact.repair.api import prepare_repair
    from exact.repair.model import RepairModel
    from exact.repair.owl import snapshot_from_axioms
    from exact.repair.pipeline import freeze_neural_round

    source = snapshot_from_axioms((owl.SubClassOf(cls("A"), cls("Parent")),))
    target = snapshot_from_axioms((owl.Declaration(cls("B")),))
    problem = prepare_repair(
        source,
        target,
        [{"Src": cls("A").iri.value, "Tgt": cls("B").iri.value, "Score": 0.7, "object_id": "m"}],
        evidence={
            "m": {"embeddings": [1.0, 2.0], "reference_alignment": ["hidden"]},
            "explanations": [{"text": "observable explanation", "confidence": 0.6}],
            "alternative_candidates": [{"Src": cls("A").iri.value, "Tgt": cls("B").iri.value}],
            "inference_artifacts": {"channel": [0.1, 0.2]},
        },
    )
    graph = build_observable_graph(
        problem.objects, fixed_axioms=problem.fixed_axioms, evidence=dict(problem.evidence)
    )
    assert graph.omitted_evidence == ("m.reference_alignment",)
    node = next(n for n in graph.nodes if n.node_id == "object:m")
    assert dict(node.features)["score_missing"] == 0
    assert any(n.node_id == "evidence:global:explanations" for n in graph.nodes)
    assert not any("reference_alignment" in key for n in graph.nodes for key, _ in n.features)
    torch.manual_seed(13)
    model = RepairModel(graph.metadata, hidden_dim=16, heads=2, layers=1, dropout=0)
    frozen = freeze_neural_round(
        problem, model, graph=graph, draws_per_object=2, candidate_cap=8, profile=(("delete", 0.1),)
    )
    assert model.training
    assert frozen.graph_hash and frozen.model_hash
    assert frozen.proposal_reports[0]["attempted_draws"] == 2
    tags = {
        tag for candidate in frozen.problem.objects[0].candidates for tag in candidate.action_tags
    }
    assert {"keep", "delete", "retain_subsumption"} <= tags
    from exact.repair.kernel import repair

    result = repair(frozen.problem, frozen.objective, diagnose=False, preserve_verified_input=False)
    assert result.logical_status == "VERIFIED_FEASIBLE"
    assert frozen.model_hash in result.model_status
    assert frozen.problem.graph_identity == frozen.graph_hash
    assert frozen.problem.proposal_provenance == frozen.proposal_reports
    assert result.assignment is not None
    for benefit, cost, integer in zip(
        frozen.objective.benefit[0], frozen.objective.costs[0], frozen.objective.unary[0]
    ):
        assert integer == pytest.approx((benefit - cost) * frozen.objective.scale, abs=0.5)


def test_normalised_disjointness_and_unknown_operand_are_not_vacuous_labels():
    a, b = cls("A"), cls("B")
    probe = TeacherProbe(
        "normalised", owl.SubClassOf(owl.ObjectIntersectionOf((a, b)), owl.OWL_NOTHING), "disjoint"
    )
    assert evaluate_teacher(Oracle(), [probe]).benefit == 1.0
    result = evaluate_teacher(Oracle(unsat=(a,), unknown=(b,)), [probe])
    assert result.benefit is None and result.outcomes[0].nonvacuous is None


def test_generated_controls_keep_parents_and_verify_intended_theories():
    pytest.importorskip("pyhermit")
    from tools.repair.corpus import FAMILIES, generate_corpus
    from tools.repair.train import _verify_intended

    cases = generate_corpus(
        parents_per_family=1,
        siblings_per_parent=1,
        coherent_controls=True,
        missing_candidate_controls=True,
    )
    assert len(cases) == 3 * len(FAMILIES)
    groups = {}
    for case in cases:
        groups.setdefault(case.structural_parent, set()).add(case.split)
        if case.control == "corrupted":
            assert _verify_intended(case), case.family
        for obj in case.problem.objects:
            if obj.kind == "ontology_axiom":
                for candidate in obj.candidates:
                    if "delete" in candidate.action_tags:
                        assert dict(candidate.cost_features)["ontology_edit"] == 1
    assert all(len(splits) == 1 for splits in groups.values())
    missing = next(case for case in cases if case.control == "missing_candidate")
    assert missing.intended_theory and missing.intended_assignment == (-1,) * len(
        missing.problem.objects
    )


def test_training_rejects_heldout_and_adaptation_without_weights():
    pytest.importorskip("torch_geometric")
    from tools.repair.corpus import generate_corpus
    from tools.repair.train import train_cases

    cases = generate_corpus(parents_per_family=3, siblings_per_parent=1)
    train = next(case for case in cases if case.split == "train")
    dev = next(case for case in cases if case.split == "development")
    test = next(case for case in cases if case.split == "test")
    cache = make_cache()
    with pytest.raises(ValueError, match="saved clean-parent"):
        train_cases([(test, cache)], [(dev, cache)], epochs=1)
    with pytest.raises(ValueError, match="weights only"):
        train_cases(
            [(replace(train, origin="training_side_real"), cache)],
            [(replace(dev, origin="training_side_real"), cache)],
            epochs=1,
            arm="training_side_adaptation",
        )
    with pytest.raises(ValueError, match="dependencies/profile"):
        train_cases([(train, cache)], [(dev, cache)], epochs=1)


def test_model_identity_includes_attention_heads_and_graph_indexes_are_immutable():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from exact.repair.model import RepairModel
    from exact.repair.pipeline import model_digest

    obj, c = observed_case()
    graph = build_observable_graph([obj], retrieved_symbols=(c,))
    a = RepairModel(graph.metadata, encoder="none", hidden_dim=16, heads=1, layers=0)
    b = RepairModel(graph.metadata, encoder="none", hidden_dim=16, heads=2, layers=0)
    b.load_state_dict(a.state_dict())
    assert model_digest(a) != model_digest(b)
    assert model_digest(a.to(dtype=torch.bfloat16))
    with pytest.raises(TypeError):
        graph.adjacency["object:m"] = frozenset()


def test_teacher_cache_cannot_claim_complete_for_a_partial_assignment_space():
    cache = make_cache(2)
    with pytest.raises(ValueError, match="every assignment"):
        replace(cache, complete=True)


def test_original_and_candidate_scalar_owl_syntax_is_preserved():
    pytest.importorskip("torch_geometric")
    import torch

    from exact.repair.model import RepairModel

    prop = owl.ObjectProperty(owl.IRI("urn:learning:p"))
    a, b = cls("A"), cls("B")
    two = owl.ObjectMinCardinality(2, prop, b)
    three = owl.ObjectMinCardinality(3, prop, b)
    keep = ReplacementCandidateV2("m", "keep", (owl.SubClassOf(a, two),))
    obj = RevisionObjectV2("m", "mapping", keep.axioms, (keep,))
    graph = build_observable_graph([obj])
    features = dict(next(n for n in graph.nodes if n.node_id == structural_id(two)).features)
    assert features["syntax/cardinality"] == 2
    model = RepairModel(graph.metadata, hidden_dim=16, heads=2, layers=0)
    memory = model.encode(graph)
    assert not torch.equal(
        model.encode_structure(two, memory), model.encode_structure(three, memory)
    )
