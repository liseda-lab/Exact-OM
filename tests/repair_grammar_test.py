"""Independent finite-oracle checks for direct grammar compilation and bundle mass."""

from dataclasses import replace
from itertools import product

import pyowl_core as owl
import pytest
import torch

from exact.repair.candidates import (
    expression_size,
    finite_expression_menu,
    intersection,
    make_candidate,
    mapping_candidates,
    ontology_candidates,
)
from exact.repair.circuit import ConditionedMixture
from exact.repair.grammar import CircuitBudgetExceeded, compile_grammar, mapping_grammar
from exact.repair.records import RevisionObjectV2


def cls(name):
    return owl.Class(owl.IRI(f"urn:{name}"))


def prop(name):
    return owl.ObjectProperty(owl.IRI(f"urn:{name}"))


def mapping(relation="="):
    source, target = cls("S"), cls("T")
    candidates = mapping_candidates("mapping", source, target, relation)
    return RevisionObjectV2(
        "mapping",
        "mapping",
        candidates[0].axioms,
        candidates,
        source_entity=source,
        target_entity=target,
    )


def oracle(encoding):
    expressions = finite_expression_menu(
        encoding.classes,
        encoding.properties,
        max_depth=encoding.max_depth,
        max_constructors=encoding.max_constructors,
    )
    values = {}
    for template in encoding.templates:
        for expression in (None,) if template.fixed else expressions:
            candidate = encoding.emit(template, expression)
            if any(
                expression_size(e)[0] > encoding.max_depth
                or expression_size(e)[1] > encoding.max_constructors
                for e in candidate.active_expressions
            ):
                continue
            values[encoding.assignment(template, expression)] = candidate
    return values


def distribution(encoding, logits=None, components=None):
    circuit = compile_grammar(encoding)
    if logits is None:
        logits = torch.zeros((1, encoding.variable_count), dtype=torch.float64)
    return ConditionedMixture(circuit, logits, components)


def test_every_tiny_boolean_assignment_matches_typed_canonical_oracle():
    encoding = mapping_grammar(mapping("<"), [cls("S")], max_depth=0, max_constructors=0)
    values = oracle(encoding)
    dist = distribution(encoding)
    assert encoding.variable_count < 16
    for bits in product((False, True), repeat=encoding.variable_count):
        assert dist.accepts(bits) == encoding.accepts(bits) == (bits in values)
    assert dist.circuit.root.model_count() == len(values)
    assert {sample.candidate_id for sample in dist.sample(50)} <= {
        c.candidate_id for c in values.values()
    }


@pytest.mark.parametrize("depth,size", [(1, 1), (2, 2), (2, 3)])
def test_symbolic_slots_match_finite_expression_and_template_enumeration(depth, size):
    obj = mapping()
    classes, properties = (cls("S"), cls("T"), cls("A")), (prop("r"),)
    encoding = mapping_grammar(obj, classes, properties, max_depth=depth, max_constructors=size)
    dist = distribution(encoding)
    values = oracle(encoding)
    assert dist.circuit.root.model_count() == len(values)
    assert all(dist.accepts(bits) for bits in values)
    expected = mapping_candidates(
        obj.object_id,
        cls("S"),
        cls("T"),
        expressions=finite_expression_menu(
            classes, properties, max_depth=depth, max_constructors=size
        ),
    )
    expected = {
        c.candidate_id
        for c in expected
        if all(
            expression_size(e)[0] <= depth and expression_size(e)[1] <= size
            for e in c.active_expressions
        )
    }
    assert {c.candidate_id for c in values.values()} == expected
    assert not any(name == "bundle" for name, _ in encoding.fields)
    assert len(encoding.templates) < len(values)


def test_probability_sums_template_and_absorbed_expression_aliases_and_gradients():
    encoding = mapping_grammar(
        mapping(), [cls("S"), cls("T"), cls("A")], max_depth=1, max_constructors=1
    )
    generator = torch.Generator().manual_seed(19)
    logits = torch.randn(
        (2, encoding.variable_count), generator=generator, dtype=torch.float64, requires_grad=True
    )
    mixture = torch.tensor([0.3, -0.5], dtype=torch.float64, requires_grad=True)
    dist = distribution(encoding, logits, mixture)
    values = oracle(encoding)
    bits = torch.tensor(list(values), dtype=torch.bool)
    logq = torch.where(
        bits[:, None, :],
        torch.nn.functional.logsigmoid(logits),
        torch.nn.functional.logsigmoid(-logits),
    ).sum(-1)
    component_logz = torch.logsumexp(logq, dim=0)
    logz = torch.logsumexp(torch.log_softmax(mixture, 0) + component_logz, dim=0)
    assert torch.allclose(dist.component_log_normalizers, component_logz)
    assert torch.allclose(dist.log_normalizer, logz)
    assert torch.allclose(
        dist.component_posterior, torch.softmax(torch.log_softmax(mixture, 0) + component_logz, 0)
    )
    probabilities = {}
    for row, candidate in enumerate(values.values()):
        probabilities.setdefault(candidate.candidate_id, []).append(row)
    assert max(map(len, probabilities.values())) > 1
    chosen_id, rows = max(probabilities.items(), key=lambda row: len(row[1]))
    candidate = next(c for c in values.values() if c.candidate_id == chosen_id)
    actual = dist.candidate_log_probability(candidate)
    expected = torch.logsumexp(logq[rows] + torch.log_softmax(mixture, 0), dim=(0, 1)) - logz
    assert torch.allclose(actual, expected)
    actual_grad = torch.autograd.grad(actual, (logits, mixture), retain_graph=True)
    expected_grad = torch.autograd.grad(expected, (logits, mixture), retain_graph=True)
    assert all(torch.allclose(a, b, atol=1e-10) for a, b in zip(actual_grad, expected_grad))
    total = sum(
        dist.candidate_log_probability(
            next(c for c in values.values() if c.candidate_id == identifier)
        ).exp()
        for identifier in probabilities
    )
    assert torch.allclose(total, torch.tensor(1.0, dtype=torch.float64))
    outsider = make_candidate("mapping", [owl.SubClassOf(cls("foreign"), cls("other"))], ["keep"])
    assert torch.isneginf(dist.candidate_log_probability(outsider))


def test_complex_ontology_left_obeys_bounds_after_flattening_and_absorption():
    a, b, c = map(cls, "ABC")
    role = prop("r")
    left = intersection(b, owl.ObjectSomeValuesFrom(role, a))
    axiom = owl.SubClassOf(left, c)
    keep = make_candidate("ontology", (axiom,), ("keep",))
    revision = RevisionObjectV2(
        "ontology", "ontology_axiom", (axiom,), (keep,), occurrence_id="source:7"
    )
    encoding = mapping_grammar(revision, [a, b, c], [role], max_depth=2, max_constructors=2)
    dist = distribution(encoding)
    values = oracle(encoding)
    assert dist.circuit.root.model_count() == len(values)
    assert all(dist.accepts(bits) for bits in values)
    expected = ontology_candidates(
        revision,
        expressions=finite_expression_menu([a, b, c], [role], max_depth=2, max_constructors=2),
    )
    expected = {
        c.candidate_id
        for c in expected
        if all(
            expression_size(e)[0] <= 2 and expression_size(e)[1] <= 2 for e in c.active_expressions
        )
    }
    assert {c.candidate_id for c in values.values()} == expected
    for candidate in values.values():
        encodings = encoding.candidate_assignments(candidate)
        assert all(dist.accepts(bits) for bits in encodings if bits in values)
        assert {bits for bits, c in values.items() if c.candidate_id == candidate.candidate_id} == {
            bits for bits in encodings if dist.accepts(bits)
        }


def test_side_restrictions_lock_controls_and_deterministic_representatives():
    obj = mapping()
    encoding = mapping_grammar(
        obj,
        [cls("S"), cls("T")],
        [prop("r")],
        max_depth=1,
        max_constructors=1,
        source_classes=[cls("S")],
        target_classes=[cls("T")],
        source_properties=[],
        target_properties=[],
    )
    dist = distribution(encoding)
    for template in encoding.templates:
        if template.fixed is None:
            allowed = cls("S") if template.index == 0 else cls("T")
            # Index order is canonical axiom order; identify the actual subclass.
            allowed = next(
                a.sub_class
                for i, a in enumerate(sorted(obj.original_axioms, key=owl.canonical_bytes))
                if i == template.index
            )
            assert dist.accepts(encoding.assignment(template, allowed))
            other = cls("T") if allowed == cls("S") else cls("S")
            assert not dist.accepts(encoding.assignment(template, other))
    representatives = encoding.representatives()
    assert {tag for candidate in representatives for tag in candidate.action_tags} >= {
        "keep",
        "delete",
        "retain_subsumption",
        "specialise_subclass",
        "add_necessary_condition",
        "complex_equivalence",
        "composite",
    }
    assert all(torch.isfinite(dist.candidate_log_probability(c)) for c in representatives)
    keep = next(c for c in obj.candidates if "keep" in c.action_tags)
    locked = replace(obj, locked=True, candidates=(keep,))
    locked_encoding = mapping_grammar(locked, [cls("S"), cls("T")], max_depth=1, max_constructors=1)
    assert distribution(locked_encoding).circuit.root.model_count() == 1


def test_compiler_resource_limits_fail_explicitly_and_cache_identity_includes_language():
    encoding = mapping_grammar(mapping(), [cls("S"), cls("T")], max_depth=1, max_constructors=1)
    with pytest.raises(CircuitBudgetExceeded, match="allocated"):
        compile_grammar(encoding, max_nodes=1)
    with pytest.raises(CircuitBudgetExceeded, match="wall"):
        compile_grammar(encoding, max_seconds=1e-12)
    first = compile_grammar(encoding)
    assert compile_grammar(encoding) is first
    expanded = mapping_grammar(
        mapping(), [cls("S"), cls("T"), cls("extra")], max_depth=1, max_constructors=1
    )
    assert expanded.variable_count - encoding.variable_count == encoding.slot_count
    changed = compile_grammar(replace(encoding, constraint_identity="different-fixed-premises"))
    assert first.cache_key != changed.cache_key


def test_constructor_limit_eliminates_impossible_deep_slots_without_changing_language():
    classes, properties = [cls("S"), cls("T"), cls("A")], [prop("r")]
    shallow = mapping_grammar(mapping(), classes, properties, max_depth=2, max_constructors=1)
    deep = mapping_grammar(mapping(), classes, properties, max_depth=8, max_constructors=1)
    assert shallow.slot_count == deep.slot_count == 3
    assert shallow.variable_count == deep.variable_count
    assert {c.candidate_id for c in oracle(shallow).values()} == {
        c.candidate_id for c in oracle(deep).values()
    }
    assert (
        distribution(shallow).circuit.root.model_count()
        == distribution(deep).circuit.root.model_count()
    )
