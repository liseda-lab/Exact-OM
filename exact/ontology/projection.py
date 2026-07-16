"""Pure-Python taxonomy and OWL2Vec*-style projections."""

from __future__ import annotations

from collections.abc import Iterable

from exact.core.entities.graph import Edge
from exact.core.entities.kinds import EntityKind
from exact.ontology.expressions import intersection_operands, named_class_iri
from exact.ontology.parser import (
    ClassExpression,
    NamedClass,
    ObjectSomeValuesFrom,
    ParsedOntology,
)

SUBCLASS_OF = "http://subclassof"
RDF_TYPE = "http://type"
RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
XSD_BOOLEAN = "http://www.w3.org/2001/XMLSchema#boolean"


def _expression_edges(subject: str, expr: ClassExpression) -> Iterable[Edge]:
    target = named_class_iri(expr)
    if target is not None:
        yield Edge(subject, SUBCLASS_OF, target)
        return
    if isinstance(expr, ObjectSomeValuesFrom):
        filler = named_class_iri(expr.filler)
        if filler is not None:
            yield Edge(subject, expr.property_iri, filler)
        return
    for operand in intersection_operands(expr):
        if operand is expr:
            continue
        yield from _expression_edges(subject, operand)


def _inverse_expansions(parsed: ParsedOntology, edge: Edge) -> Iterable[Edge]:
    for inverse in parsed.inverse_object_properties:
        if edge.rel == inverse.first_property_iri:
            yield Edge(edge.dst, inverse.second_property_iri, edge.src)
        elif edge.rel == inverse.second_property_iri:
            yield Edge(edge.dst, inverse.first_property_iri, edge.src)


def project(
    parsed: ParsedOntology,
    method: str = "owl2vecstar",
    include_literals: bool = False,
) -> list[Edge]:
    """Project normalized axioms into a stable, duplicate-free edge list."""

    method = str(method).lower().replace("_", "").replace("-", "")
    if method not in {"taxonomy", "owl2vecstar"}:
        raise ValueError("method must be one of {'taxonomy', 'owl2vecstar'}")

    edges: set[Edge] = set()
    for subclass_axiom in parsed.subclass_axioms:
        subject = named_class_iri(subclass_axiom.sub)
        target = named_class_iri(subclass_axiom.sup)
        if subject is not None and target is not None:
            edges.add(Edge(subject, SUBCLASS_OF, target))

    if method == "taxonomy":
        return sorted(edges, key=Edge.astuple)

    expression_edges: set[Edge] = set()
    for subclass_axiom in parsed.subclass_axioms:
        subject = named_class_iri(subclass_axiom.sub)
        if subject is not None:
            expression_edges.update(_expression_edges(subject, subclass_axiom.sup))

    for equivalent_axiom in parsed.equivalent_class_axioms:
        anchors = sorted(
            expr.iri for expr in equivalent_axiom.expressions if isinstance(expr, NamedClass)
        )
        if anchors:
            anchor = anchors[0]
            for expression in equivalent_axiom.expressions:
                if isinstance(expression, NamedClass) and expression.iri == anchor:
                    continue
                expression_edges.update(_expression_edges(anchor, expression))

    # The legacy projector expands inverse properties for restriction-derived
    # edges, but not arbitrary assertions or symmetric-property declarations.
    expanded_expressions = set(expression_edges)
    for edge in expression_edges:
        expanded_expressions.update(_inverse_expansions(parsed, edge))
    edges.update(expanded_expressions)

    for class_assertion in parsed.class_assertions:
        target = named_class_iri(class_assertion.class_expression)
        if target is not None:
            edges.add(Edge(class_assertion.individual_iri, RDF_TYPE, target))
    for object_assertion in parsed.object_property_assertions:
        edges.add(
            Edge(
                object_assertion.subject_iri,
                object_assertion.property_iri,
                object_assertion.object_iri,
            )
        )

    domains: dict[str, set[str]] = {}
    ranges: dict[str, set[str]] = {}
    data_properties = frozenset(parsed.signature[EntityKind.DATA_PROPERTY])
    for domain_axiom in parsed.property_domains:
        if domain_axiom.property_iri in data_properties:
            continue
        domain = named_class_iri(domain_axiom.domain)
        if domain is not None:
            domains.setdefault(domain_axiom.property_iri, set()).add(domain)
    for range_axiom in parsed.property_ranges:
        if range_axiom.property_iri in data_properties:
            continue
        range_iri = named_class_iri(range_axiom.range)
        if range_iri is not None:
            ranges.setdefault(range_axiom.property_iri, set()).add(range_iri)
    for property_iri in domains.keys() & ranges.keys():
        for domain in domains[property_iri]:
            for range_iri in ranges[property_iri]:
                edges.add(Edge(domain, property_iri, range_iri))

    if include_literals:
        classes = frozenset(parsed.signature[EntityKind.CLASS])
        for annotation_assertion in parsed.annotation_assertions:
            value = annotation_assertion.value
            if (
                annotation_assertion.subject_iri not in classes
                or not value.is_literal
                or value.datatype == XSD_BOOLEAN
            ):
                continue
            relation = "rdfs:label" if value.property_iri == RDFS_LABEL else value.property_iri
            edges.add(Edge(annotation_assertion.subject_iri, relation, value.value))

    return sorted(edges, key=Edge.astuple)
