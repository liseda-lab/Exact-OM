"""Compiler qualification against finite truth tables and direct probability sums."""

from itertools import product

import pyowl_core as owl
import pytest

from exact.repair.candidates import intersection, make_candidate, mapping_candidates
from exact.repair.circuit import (
    ConditionedMixture,
    EmptyProposalSpace,
    ProposalEncoding,
    compile_encoding,
    encode_candidates,
)

pytest.importorskip("pysdd")
torch = pytest.importorskip("torch")


def binary_encoding():
    return ProposalEncoding((("choice", ("a", "b")),), ((True, False), (False, True)), ("a", "b"))


def direct_normalizers(encoding, logits):
    probabilities = torch.sigmoid(logits)
    terms = []
    for assignment in encoding.assignments:
        mask = torch.tensor(assignment, dtype=torch.bool)
        terms.append(torch.where(mask, probabilities, 1 - probabilities).prod(dim=1))
    return torch.stack(terms).sum(dim=0)


def test_sdd_models_exactly_match_canonical_boolean_assignments():
    encoding = binary_encoding()
    compiled = compile_encoding(encoding)
    assert compiled.root.global_model_count() == 2
    for bits in product((False, True), repeat=encoding.variable_count):
        node = compiled.root
        for variable, bit in enumerate(bits, 1):
            node = node.condition(variable if bit else -variable)
        assert node.is_true() == (bits in encoding.assignments)
    assert compiled.node_count > 0 and compiled.compilation_seconds >= 0
    assert compile_encoding(encoding) is compiled
    reordered = compile_encoding(encoding, variable_order=(2, 1))
    assert reordered.cache_key != compiled.cache_key
    dependent = ProposalEncoding(
        encoding.fields,
        encoding.assignments,
        encoding.candidate_ids,
        constraint_identity="new-fixed-premises",
    )
    assert compile_encoding(dependent).cache_key != compiled.cache_key


def test_conditioned_mixture_normalizer_posterior_and_probabilities_match_enumeration():
    encoding = binary_encoding()
    logits = torch.logit(torch.tensor([[0.9, 0.9], [0.1, 0.9]], dtype=torch.float64))
    mixture = torch.tensor([0.0, 0.0], dtype=torch.float64)
    conditioned = ConditionedMixture(compile_encoding(encoding), logits, mixture)
    expected = direct_normalizers(encoding, logits)
    assert torch.allclose(conditioned.component_log_normalizers.exp(), expected)
    assert torch.allclose(expected, torch.tensor([0.18, 0.82], dtype=torch.float64))
    assert torch.allclose(conditioned.component_posterior, expected / expected.sum())
    assert not torch.allclose(conditioned.component_posterior, mixture.softmax(dim=0))
    assert torch.allclose(conditioned.log_normalizer.exp(), expected.mean())
    probabilities = [
        conditioned.log_probability(bits).exp() for bits in product((False, True), repeat=2)
    ]
    assert torch.allclose(sum(probabilities), torch.tensor(1.0, dtype=torch.float64))
    assert probabilities[0] == 0 and probabilities[-1] == 0


def test_likelihood_gradient_includes_normalizer():
    encoding = binary_encoding()
    logits = torch.tensor([[0.4, -0.8], [1.4, 0.3]], dtype=torch.float64, requires_grad=True)
    components = torch.tensor([0.3, -0.2], dtype=torch.float64, requires_grad=True)
    conditioned = ConditionedMixture(compile_encoding(encoding), logits, components)
    loss = -conditioned.log_probability(encoding.assignments[0])
    gradients = torch.autograd.grad(loss, (logits, components))
    probabilities = torch.sigmoid(logits)
    pi = components.softmax(dim=0)
    numerator = (pi * probabilities[:, 0] * (1 - probabilities[:, 1])).sum()
    denominator = (pi * direct_normalizers(encoding, logits)).sum()
    brute_loss = -(numerator / denominator).log()
    expected_gradients = torch.autograd.grad(brute_loss, (logits, components))
    assert torch.allclose(loss, brute_loss)
    assert all(
        torch.allclose(actual, expected) for actual, expected in zip(gradients, expected_gradients)
    )
    assert torch.autograd.gradcheck(
        lambda literals, mixing: ConditionedMixture(
            compile_encoding(encoding), literals, mixing
        ).log_probability(encoding.assignments[0]),
        (logits, components),
    )


def test_sampling_support_distribution_and_seed():
    encoding = binary_encoding()
    logits = torch.logit(torch.tensor([[0.9, 0.9], [0.1, 0.9]], dtype=torch.float64))
    conditioned = ConditionedMixture(compile_encoding(encoding), logits)
    samples = conditioned.sample(3000, seed=177)
    assert {s.assignment for s in samples} == set(encoding.assignments)
    assert abs(sum(s.component == 0 for s in samples) / len(samples) - 0.18) < 0.035
    expected_a = float(conditioned.log_probability(encoding.assignments[0]).exp())
    assert abs(sum(s.candidate_id == "a" for s in samples) / len(samples) - expected_a) < 0.035
    assert conditioned.sample(10, seed=91) == conditioned.sample(10, seed=91)


def test_zero_mass_is_explicit_and_extreme_finite_logits_do_not_underflow():
    encoding = binary_encoding()
    compiled = compile_encoding(encoding)
    with pytest.raises(EmptyProposalSpace):
        ConditionedMixture(compiled, torch.tensor([[float("inf"), float("inf")]]))
    extreme = ConditionedMixture(compiled, torch.tensor([[10000.0, 10000.0]], dtype=torch.float64))
    assert torch.isfinite(extreme.log_normalizer)
    assert abs(float(extreme.log_probability(encoding.assignments[0]).exp()) - 0.5) < 1e-10
    empty = ProposalEncoding(encoding.fields, (), ())
    with pytest.raises(EmptyProposalSpace):
        ConditionedMixture(compile_encoding(empty), torch.zeros(1, 2))


def test_mixture_captures_property_filler_correlation_without_autoregression():
    fields = (("property", ("r", "s")), ("filler", ("A", "B")))
    assignments = (
        (True, False, True, False),
        (True, False, False, True),
        (False, True, True, False),
        (False, True, False, True),
    )
    encoding = ProposalEncoding(fields, assignments, ("rA", "rB", "sA", "sB"))
    circuit = compile_encoding(encoding)
    independent = ConditionedMixture(circuit, torch.zeros(1, 4, dtype=torch.float64))
    correlated = ConditionedMixture(
        circuit, torch.tensor([[4.0, -4.0, 4.0, -4.0], [-4.0, 4.0, -4.0, 4.0]], dtype=torch.float64)
    )
    assert torch.allclose(
        independent.log_probability(assignments[0]).exp(), torch.tensor(0.25, dtype=torch.float64)
    )
    assert correlated.log_probability(assignments[0]).exp() > 0.49
    assert correlated.log_probability(assignments[1]).exp() < 0.01


def test_full_pool_encoding_preserves_elementary_states_and_unique_unused_fields():
    source, target, condition = [owl.Class(owl.IRI(f"urn:{name}")) for name in ("S", "T", "E")]
    pool = mapping_candidates("mapping", source, target, expressions=[condition])
    encoding = encode_candidates(pool, max_depth=1, max_constructors=1)
    assert set(encoding.candidate_ids) == {c.candidate_id for c in pool}
    field_offsets = []
    offset = 0
    for name, values in encoding.fields:
        field_offsets.append((name, values, offset))
        offset += len(values)
    for candidate, assignment in zip(
        sorted(pool, key=lambda c: ("keep" not in c.action_tags, c.candidate_id)),
        encoding.assignments,
    ):
        if {"keep", "delete", "retain_subsumption"} & set(candidate.action_tags):
            for name, values, offset in field_offsets:
                if name.startswith("expression:"):
                    assert assignment[offset : offset + len(values)] == tuple(
                        v == "unused" for v in values
                    )
    assert compile_encoding(encoding).root.global_model_count() == len(pool)
    a = make_candidate(
        "m", (), ("specialise_subclass",), active_expressions=[intersection(source, condition)]
    )
    b = make_candidate(
        "m", (), ("specialise_subclass",), active_expressions=[intersection(condition, source)]
    )
    assert len(encode_candidates([a, b]).candidate_ids) == 1
    with pytest.raises(ValueError, match="bounds"):
        encode_candidates([a], max_depth=0, max_constructors=0)


def test_grammar_validity_does_not_claim_global_ontology_feasibility():
    source, target = [owl.Class(owl.IRI(f"urn:{name}")) for name in ("S", "T")]
    # This inclusion may contradict a separate selected ontology object. The
    # circuit must not filter it using that editable object's current assertions.
    candidate = make_candidate("mapping", [owl.SubClassOf(source, target)], ["keep"])
    encoding = encode_candidates([candidate], constraint_identity="fixed-background-v1")
    conditioned = ConditionedMixture(
        compile_encoding(encoding), torch.zeros(1, encoding.variable_count)
    )
    assert conditioned.sample()[0].candidate_id == candidate.candidate_id
    with pytest.raises(ValueError, match="one-hot"):
        ProposalEncoding((("choice", ("a", "b")),), ((True, True),), ("bad",))
