"""Conformance of complete replacement actions and finite canonical generation."""

import pyowl_core as owl
import pytest

from exact.repair.candidates import (
    budget_candidates,
    canonical_expression,
    deduplicate_candidates,
    expression_size,
    expression_tree,
    finite_expression_menu,
    intersection,
    make_candidate,
    mapping_candidates,
    normalise_axioms,
    ontology_candidates,
)
from exact.repair.records import RevisionObjectV2


def cls(name):
    return owl.Class(owl.IRI(f"urn:{name}"))


def prop(name):
    return owl.ObjectProperty(owl.IRI(f"urn:{name}"))


def select(pool, action):
    return [c for c in pool if action in c.action_tags]


def revision(axiom, **kwargs):
    keep = make_candidate("axiom", [axiom], ["keep"])
    return RevisionObjectV2(
        "axiom",
        "ontology_axiom",
        (axiom,),
        (keep,),
        occurrence_id="source/document:assertion:7",
        **kwargs,
    )


def test_intersections_are_flat_canonical_and_bounds_apply_to_binary_tree():
    a, b, c = map(cls, "ABC")
    nested = owl.ObjectIntersectionOf(owl.CanonicalSet((a, intersection(b, a, c))))
    assert canonical_expression(nested) == intersection(c, b, a)
    assert expression_tree(nested) == expression_tree(intersection(a, b, c))
    assert expression_size(nested) == (2, 2)
    assert expression_size(owl.ObjectSomeValuesFrom(prop("r"), nested)) == (3, 3)
    with pytest.raises(ValueError, match="outside"):
        canonical_expression(owl.ObjectComplementOf(a))


def test_bounded_menu_generates_new_existentials_and_intersections():
    a, b = map(cls, "AB")
    r = prop("r")
    menu = finite_expression_menu((a, b), (r,), max_depth=1, max_constructors=1)
    assert set(menu) == {
        a,
        b,
        intersection(a, b),
        owl.ObjectSomeValuesFrom(r, a),
        owl.ObjectSomeValuesFrom(r, b),
    }
    assert finite_expression_menu((b, a), (r,), max_depth=1, max_constructors=1) == menu
    larger = finite_expression_menu((a, b), (r,), max_depth=2, max_constructors=2)
    assert owl.ObjectSomeValuesFrom(r, intersection(a, b)) in larger
    assert intersection(a, owl.ObjectSomeValuesFrom(r, b)) in larger
    with pytest.raises(ValueError, match="exceeds"):
        finite_expression_menu((a, b), (r,), max_expressions=2)
    with pytest.raises(TypeError):
        finite_expression_menu((r,))


def test_mapping_bundles_are_complete_and_mirrored_with_independent_necessity():
    source, target, filler, alternative = map(cls, ("PaperS", "AcceptedT", "Acceptance", "PaperT"))
    expression = owl.ObjectSomeValuesFrom(prop("hasDecision"), filler)
    pool = mapping_candidates(
        "mapping",
        source,
        target,
        expressions=[expression],
        endpoint_alternatives=[("target", alternative)],
    )
    forward, backward = owl.SubClassOf(source, target), owl.SubClassOf(target, source)
    assert set(select(pool, "keep")[0].axioms) == {forward, backward}
    assert {frozenset(c.axioms) for c in select(pool, "retain_subsumption")} == {
        frozenset([forward]),
        frozenset([backward]),
    }
    for left, right, retained in ((source, target, backward), (target, source, forward)):
        specialised = intersection(left, expression)
        sufficient = owl.SubClassOf(specialised, right)
        necessary = owl.SubClassOf(right, expression)
        bundle = next(
            c
            for c in select(pool, "specialise_subclass")
            if set(c.axioms) == {sufficient, retained}
        )
        assert bundle.active_expressions == (specialised,)
        assert dict(bundle.provenance) == {
            "subclass": left.iri.value,
            "superclass": right.iri.value,
        }
        assert any(
            set(c.axioms) == {retained, necessary} for c in select(pool, "add_necessary_condition")
        )
        assert any(
            set(c.axioms) == {sufficient, retained, necessary}
            for c in select(pool, "complex_equivalence")
        )
        assert all(forward not in c.axioms for c in [bundle] if left == source)
    endpoint = select(pool, "replace_endpoint")[0]
    assert set(endpoint.axioms) == {
        owl.SubClassOf(source, alternative),
        owl.SubClassOf(alternative, source),
    }


def test_directional_originals_never_gain_reverse_controls():
    source, target, expression = map(cls, "STE")
    for relation, left, right in (("<", source, target), (">", target, source)):
        pool = mapping_candidates("mapping", source, target, relation, expressions=[expression])
        reverse = owl.SubClassOf(right, left)
        assert not select(pool, "retain_subsumption")
        assert not select(pool, "complex_equivalence")
        assert all(reverse not in c.axioms for c in pool)
        assert select(pool, "composite")


def test_typed_nonclass_controls_and_locks():
    source, target = prop("source"), prop("target")
    pool = mapping_candidates("property", source, target, entity_kind="object_property")
    assert {tag for c in pool for tag in c.action_tags} == {"keep", "delete"}
    assert set(select(pool, "keep")[0].axioms) == {
        owl.SubObjectPropertyOf(source, target),
        owl.SubObjectPropertyOf(target, source),
    }
    assert (
        len(
            mapping_candidates(
                "property", source, target, entity_kind="object_property", locked=True
            )
        )
        == 1
    )
    with pytest.raises(TypeError):
        mapping_candidates("bad", source, target)


def test_ontology_actions_preserve_exact_unedited_disjoint_pairs_and_conjuncts():
    a, b, c = map(cls, "ABC")
    original = owl.DisjointClasses(owl.CanonicalSet((a, b, c)))
    pool = ontology_candidates(revision(original))
    changed = select(pool, "remove_disjointness")
    assert len(changed) == 3
    pairs = {owl.DisjointClasses(owl.CanonicalSet(pair)) for pair in ((a, b), (a, c), (b, c))}
    assert all(len(item.axioms) == 2 and set(item.axioms) < pairs for item in changed)
    assert all(
        dict(item.provenance)["occurrence_id"] == "source/document:assertion:7" for item in changed
    )
    conjunction = owl.SubClassOf(a, intersection(b, c))
    changed = select(ontology_candidates(revision(conjunction)), "remove_superclass_conjunct")
    assert {frozenset(c.axioms) for c in changed} == {
        frozenset([owl.SubClassOf(a, b)]),
        frozenset([owl.SubClassOf(a, c)]),
    }
    assert len(ontology_candidates(revision(original, locked=True))) == 1


def test_generalisation_requires_fixed_asserted_premises():
    a, narrow, broad, expression = map(cls, ("A", "Person", "Agent", "Presenting"))
    role = prop("writer")
    fixed = (owl.SubClassOf(narrow, broad),)
    fixtures = [
        (owl.SubClassOf(a, narrow), owl.SubClassOf(a, broad), "generalise_superclass"),
        (
            owl.ObjectPropertyDomain(role, narrow),
            owl.ObjectPropertyDomain(role, broad),
            "generalise_domain",
        ),
        (
            owl.ObjectPropertyRange(role, narrow),
            owl.ObjectPropertyRange(role, broad),
            "generalise_range",
        ),
        (
            owl.SubClassOf(a, owl.ObjectSomeValuesFrom(role, narrow)),
            owl.SubClassOf(a, owl.ObjectSomeValuesFrom(role, broad)),
            "generalise_existential_filler",
        ),
    ]
    for original, expected, tag in fixtures:
        assert not select(ontology_candidates(revision(original)), tag)
        candidate = select(
            ontology_candidates(revision(original, authorship="human"), fixed_axioms=fixed), tag
        )[0]
        assert candidate.axioms == (expected,)
        assert dict(candidate.provenance)["fixed_premise"] == owl.structural_hexdigest(fixed[0])
        assert dict(candidate.cost_features)["human_authored_ontology_edit"] == 1
    specialised = select(
        ontology_candidates(revision(fixtures[0][0]), expressions=[expression]),
        "specialise_ontology_subclass",
    )[0]
    assert specialised.active_expressions == (intersection(a, expression),)
    assert specialised.axioms == (owl.SubClassOf(intersection(a, expression), narrow),)


def test_identity_retains_activation_and_merges_provenance_aliases():
    a, b = map(cls, "AB")
    axiom = owl.SubClassOf(a, b)
    first = make_candidate("o", [axiom], ["keep"], provenance=[("source", "a")])
    alias = make_candidate("o", [axiom, axiom], ["composite"], provenance=[("source", "b")])
    active = make_candidate("o", [axiom], ["composite"], active_expressions=[a])
    pool = deduplicate_candidates([first, alias, active])
    assert len(pool) == 2 and first.candidate_id != active.candidate_id
    merged = next(c for c in pool if c.candidate_id == first.candidate_id)
    assert set(merged.provenance) == {("source", "a"), ("source", "b")}
    assert set(merged.action_tags) == {"keep", "composite"}
    eq = owl.EquivalentClasses(owl.CanonicalSet((a, b)))
    assert set(normalise_axioms([eq])) == {axiom, owl.SubClassOf(b, a)}


def test_budget_preserves_controls_and_available_families_before_ranking():
    a, b, e = map(cls, "ABE")
    pool = mapping_candidates("m", a, b, expressions=[e])
    with pytest.raises(ValueError, match="mandatory"):
        budget_candidates(pool, 3)
    ranked = {c.candidate_id: 100 if "complex_equivalence" in c.action_tags else 0 for c in pool}
    retained = budget_candidates(pool, 8, scores=ranked)
    assert len(retained) == 8
    required_controls = {
        c.candidate_id
        for c in pool
        if set(c.action_tags) & {"keep", "delete", "retain_subsumption"}
    }
    assert required_controls <= {c.candidate_id for c in retained}
    assert {tag for c in retained for tag in c.action_tags} == {
        tag for c in pool for tag in c.action_tags
    }
    assert budget_candidates(pool, 100) == pool


def test_deletion_alias_keeps_deletion_cost_and_necessary_expression_has_size():
    a, b = map(cls, "AB")
    disjoint = owl.DisjointClasses(owl.CanonicalSet((a, b)))
    deletion = select(ontology_candidates(revision(disjoint)), "delete")[0]
    assert "remove_disjointness" in deletion.action_tags
    assert dict(deletion.cost_features)["delete"] == 1.0
    condition = owl.ObjectSomeValuesFrom(prop("r"), a)
    necessities = select(
        mapping_candidates("m", a, b, expressions=[condition]), "add_necessary_condition"
    )
    assert all(dict(candidate.cost_features)["expression_size"] > 0 for candidate in necessities)


def test_outside_grammar_assertions_remain_available_without_illegal_specialisation():
    a, b, e = map(cls, "ABE")
    original = owl.SubClassOf(owl.ObjectComplementOf(a), b)
    pool = ontology_candidates(revision(original), expressions=[e])
    assert select(pool, "keep")[0].axioms == (original,)
    assert select(pool, "delete")
    assert not select(pool, "specialise_ontology_subclass")


def test_protocol_costs_describe_emitted_edits_and_retain_constructor_credit():
    from exact.repair.candidates import replacement_cost_features

    source, target, filler = map(cls, ("S", "T", "A"))
    condition = owl.ObjectSomeValuesFrom(prop("r"), filler)
    original = (owl.SubClassOf(source, target), owl.SubClassOf(target, source))
    specialised = intersection(source, condition)
    emitted = (owl.SubClassOf(specialised, target), original[1])
    costs = dict(replacement_cost_features(original, emitted, active_expressions=(specialised,)))
    assert costs["removed_direction"] == 1
    assert costs["subclass_specialisation"] == 1
    assert costs["necessary_condition"] == 0
    assert costs["new_constructor"] == 2
    assert costs["mapping_deletion"] == costs["ontology_edit"] == 0
    composite = dict(
        replacement_cost_features(
            original,
            (*emitted, owl.SubClassOf(target, condition)),
            active_expressions=(specialised,),
        )
    )
    assert composite["necessary_condition"] == 1
    assert composite["new_constructor"] == 3
    reused = dict(
        replacement_cost_features(
            (owl.SubClassOf(source, condition),),
            (owl.SubClassOf(target, condition),),
            kind="ontology_axiom",
        )
    )
    assert reused["expression_size"] == 1
    assert reused["new_constructor"] == 0
    human_delete = dict(
        replacement_cost_features(original, (), kind="ontology_axiom", authorship="human")
    )
    assert human_delete["human_ontology_edit"] == human_delete["human_authored_ontology_edit"] == 1
    assert human_delete["mapping_deletion"] == 0
    assert all(value == 0 for _, value in replacement_cost_features(original, original))


def test_every_retrieved_endpoint_control_is_mandatory_even_with_low_score():
    source, target, x, y = map(cls, ("S", "T", "X", "Y"))
    pool = mapping_candidates(
        "m", source, target, endpoint_alternatives=(("target", x), ("target", y))
    )
    endpoint_ids = {c.candidate_id for c in pool if "replace_endpoint" in c.action_tags}
    with pytest.raises(ValueError, match="mandatory"):
        budget_candidates(pool, len(pool) - 1)
    assert endpoint_ids <= {c.candidate_id for c in budget_candidates(pool, len(pool))}
