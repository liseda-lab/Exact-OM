"""Pure-Python taxonomy and OWL2Vec*-style projections."""

from __future__ import annotations

from collections.abc import Iterable

from exact.core.entities.graph import Edge
from exact.core.entities.kinds import EntityKind
from exact.ontology.expressions import intersection_operands, named_class_iri
from exact.ontology.parser import (
    ClassExpression,
    NamedClass,
    ObjectAllValuesFrom,
    ObjectCardinalityRestriction,
    ObjectIntersectionOf,
    ObjectSomeValuesFrom,
    ObjectUnionOf,
    ParsedOntology,
    SubPropertyOf,
)

SUBCLASS_OF = "http://subclassof"
RDF_TYPE = "http://type"
XSD_BOOLEAN = "http://www.w3.org/2001/XMLSchema#boolean"
OWL_THING = "http://www.w3.org/2002/07/owl#Thing"
RDFS_NAMESPACE = "http://www.w3.org/2000/01/rdf-schema#"


def _int32(value: int) -> int:
    """Return *value* with Java's signed 32-bit integer semantics."""

    value &= 0xFFFFFFFF
    return value - 0x100000000 if value & 0x80000000 else value


def _java_string_hash(value: str) -> int:
    result = 0
    for character in value:
        result = _int32(31 * result + ord(character))
    return result


def _owlapi_iri_hash(iri: str) -> int:
    """Reproduce the namespace-plus-remainder hash used by OWLAPI 4.5."""

    split_at = max(iri.rfind("#"), iri.rfind("/"), iri.rfind(":"))
    namespace = iri[: split_at + 1] if split_at >= 0 else ""
    remainder = iri[split_at + 1 :] if split_at >= 0 else iri
    return _int32(_java_string_hash(namespace) + _java_string_hash(remainder))


def _legacy_subproperty_hash(sub_property: str, super_property: str) -> int:
    """Hash an OWLSubObjectPropertyOf axiom like OWLAPI 4.5.22."""

    # HashCode.primes[56] and HashCode.primes[27] in OWLAPI 4.5.22.
    sub_hash = _int32(31 * 4153 + _owlapi_iri_hash(sub_property))
    super_hash = _int32(31 * 4153 + _owlapi_iri_hash(super_property))
    result = 1823
    for component in (sub_hash, super_hash, 0):  # no axiom annotations
        result = _int32(31 * result + component)
    return result


def _legacy_subroles(parsed: ParsedOntology) -> dict[str, tuple[str, ...]]:
    """Reproduce the historical projector's sub-role overwrite behavior.

    mOWL 1.0.1 iterated the OWLAPI RBox ``HashSet`` and accidentally looked up
    the existing children under the sub-role before assigning them to the
    super-role. Siblings therefore overwrote one another. Keeping this quirk
    isolated here preserves projection parity without leaking Java into the
    parser or the rest of the ontology model.
    """

    axioms = [
        axiom for axiom in parsed.subproperty_axioms if axiom.kind == EntityKind.OBJECT_PROPERTY
    ]
    inverse_count = len(
        {
            frozenset((axiom.first_property_iri, axiom.second_property_iri))
            for axiom in parsed.inverse_object_properties
        }
    )
    rbox_size = len(axioms) + inverse_count
    capacity = 16
    while rbox_size > int(capacity * 0.75):
        capacity *= 2

    def order(axiom: SubPropertyOf) -> tuple[int, int, str, str]:
        raw_hash = _legacy_subproperty_hash(
            axiom.sub_property_iri,
            axiom.super_property_iri,
        )
        unsigned = raw_hash & 0xFFFFFFFF
        spread = unsigned ^ (unsigned >> 16)
        return (
            spread & (capacity - 1),
            spread,
            axiom.sub_property_iri,
            axiom.super_property_iri,
        )

    subroles: dict[str, tuple[str, ...]] = {}
    for axiom in sorted(axioms, key=order):
        subroles[axiom.super_property_iri] = (
            axiom.sub_property_iri,
            *subroles.get(axiom.sub_property_iri, ()),
        )
    return subroles


def _expression_edges(
    subject: str,
    expr: ClassExpression,
    *,
    include_unions: bool = False,
    data_properties: frozenset[str] = frozenset(),
) -> Iterable[Edge]:
    target = named_class_iri(expr)
    if target is not None:
        yield Edge(subject, SUBCLASS_OF, target)
        return
    if isinstance(expr, (ObjectSomeValuesFrom, ObjectAllValuesFrom)):
        filler = named_class_iri(expr.filler)
        if filler is not None:
            yield Edge(subject, expr.property_iri, filler)
        return
    if isinstance(expr, ObjectCardinalityRestriction):
        if expr.cardinality_type not in {"min", "max"} or expr.property_iri in data_properties:
            return
        filler = named_class_iri(expr.filler) if expr.filler is not None else OWL_THING
        yield Edge(subject, expr.property_iri, filler or OWL_THING)
        return
    if isinstance(expr, ObjectUnionOf) and not include_unions:
        return
    operands = expr.operands if isinstance(expr, ObjectUnionOf) else intersection_operands(expr)
    for operand in operands:
        if operand is expr:
            continue
        yield from _expression_edges(
            subject,
            operand,
            # The legacy projector expands a union used directly as an
            # equivalent-class expression, not unions nested in intersections.
            include_unions=include_unions and isinstance(expr, ObjectUnionOf),
            data_properties=data_properties,
        )


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
    data_properties = frozenset(parsed.signature[EntityKind.DATA_PROPERTY])
    for subclass_axiom in parsed.subclass_axioms:
        subject = named_class_iri(subclass_axiom.sub)
        if subject is not None:
            expression_edges.update(
                _expression_edges(
                    subject,
                    subclass_axiom.sup,
                    data_properties=data_properties,
                )
            )

    for equivalent_axiom in parsed.equivalent_class_axioms:
        anchors = sorted(
            expr.iri for expr in equivalent_axiom.expressions if isinstance(expr, NamedClass)
        )
        if anchors:
            anchor = anchors[0]
            for expression in equivalent_axiom.expressions:
                if isinstance(expression, NamedClass) and expression.iri == anchor:
                    continue
                # mOWL's equivalence walker handled named classes and
                # top-level intersections/unions. A direct restriction fell
                # through without producing an edge.
                if not isinstance(
                    expression,
                    (NamedClass, ObjectIntersectionOf, ObjectUnionOf),
                ):
                    continue
                expression_edges.update(
                    _expression_edges(
                        anchor,
                        expression,
                        include_unions=True,
                        data_properties=data_properties,
                    )
                )

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
    subroles = _legacy_subroles(parsed)
    for property_iri in set(domains) & set(ranges):
        for domain in domains[property_iri]:
            for range_iri in ranges[property_iri]:
                base = Edge(domain, property_iri, range_iri)
                edges.add(base)
                edges.update(_inverse_expansions(parsed, base))
                for subrole in subroles.get(property_iri, ()):
                    edges.add(Edge(domain, subrole, range_iri))

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
            relation = (
                "rdfs:" + value.property_iri.removeprefix(RDFS_NAMESPACE)
                if value.property_iri.startswith(RDFS_NAMESPACE)
                else value.property_iri
            )
            edges.add(Edge(annotation_assertion.subject_iri, relation, value.value))

    return sorted(edges, key=Edge.astuple)
