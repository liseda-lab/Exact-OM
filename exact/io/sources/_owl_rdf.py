"""RDF graph normalization for the OWL source adapter.

This module owns Exact's RDFLib dependency. The ontology package supplies the
Horned-OWL document loader and backend-neutral records; this adapter translates
RDF graphs into those records without leaking RDFLib objects across the I/O
boundary.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable

try:
    from rdflib import BNode, Graph, Literal, URIRef
    from rdflib.namespace import OWL, RDF, RDFS
except ImportError as exc:  # pragma: no cover - core dependency failure.
    raise RuntimeError(
        "OWL parsing requires rdflib. Install Exact-OM with its core dependencies."
    ) from exc

from exact.core.entities.graph import AnnotationValue
from exact.core.entities.kinds import EntityKind
from exact.ontology.records import (
    AnnotationAssertion,
    AnonymousClassExpression,
    ClassAssertion,
    ClassExpression,
    DataOneOf,
    DataPropertyAssertion,
    EquivalentClasses,
    InverseObjectProperties,
    NamedClass,
    ObjectAllValuesFrom,
    ObjectCardinalityRestriction,
    ObjectIntersectionOf,
    ObjectPropertyAssertion,
    ObjectSomeValuesFrom,
    ObjectUnionOf,
    ParsedOntology,
    PropertyDomain,
    PropertyRange,
    SubClassOf,
    SubPropertyOf,
)

_BUILTIN_TYPES = {
    OWL.AllDifferent,
    OWL.AllDisjointClasses,
    OWL.AllDisjointProperties,
    OWL.AnnotationProperty,
    OWL.AsymmetricProperty,
    OWL.Class,
    OWL.DatatypeProperty,
    OWL.DeprecatedClass,
    OWL.DeprecatedProperty,
    OWL.FunctionalProperty,
    OWL.InverseFunctionalProperty,
    OWL.IrreflexiveProperty,
    OWL.NamedIndividual,
    OWL.NegativePropertyAssertion,
    OWL.ObjectProperty,
    OWL.Ontology,
    OWL.OntologyProperty,
    OWL.ReflexiveProperty,
    OWL.Restriction,
    OWL.SymmetricProperty,
    OWL.TransitiveProperty,
    RDF.Property,
    RDFS.Class,
    RDFS.Datatype,
}

_STRUCTURAL_PREDICATES = {
    RDF.first,
    RDF.rest,
    RDF.type,
    RDFS.domain,
    RDFS.range,
    RDFS.subClassOf,
    RDFS.subPropertyOf,
    OWL.equivalentClass,
    OWL.disjointWith,
    OWL.complementOf,
    OWL.equivalentProperty,
    OWL.propertyDisjointWith,
    OWL.imports,
    OWL.intersectionOf,
    OWL.inverseOf,
    OWL.members,
    OWL.distinctMembers,
    OWL.oneOf,
    OWL.onProperty,
    OWL.onClass,
    OWL.onDataRange,
    OWL.someValuesFrom,
    OWL.allValuesFrom,
    OWL.cardinality,
    OWL.minCardinality,
    OWL.maxCardinality,
    OWL.qualifiedCardinality,
    OWL.minQualifiedCardinality,
    OWL.maxQualifiedCardinality,
    OWL.unionOf,
}


def _iri(value: object) -> str:
    return sys.intern(str(value))


def _load_graph(
    path: Path,
    *,
    rdf_xml: str | None = None,
    horned_error: BaseException | None = None,
) -> Graph:
    graph = Graph()
    try:
        if rdf_xml is not None:
            graph.parse(data=rdf_xml, format="xml", publicID=path.resolve().as_uri())
        else:
            graph.parse(path.resolve().as_uri())
        return graph
    except Exception as exc:
        detail = f" py-horned-owl also failed: {horned_error}" if horned_error else ""
        raise ValueError(
            f"Could not parse OWL ontology {path}:{detail} rdflib failed: {exc}"
        ) from exc


def _rdf_list(graph: Graph, head: Any) -> list[Any]:
    values: list[Any] = []
    seen: set[Any] = set()
    current = head
    while current != RDF.nil and current not in seen:
        seen.add(current)
        first = graph.value(current, RDF.first)
        if first is None:
            break
        values.append(first)
        current = graph.value(current, RDF.rest)
        if current is None:
            break
    return values


def _expression(graph: Graph, node: Any) -> ClassExpression:
    if isinstance(node, URIRef):
        return NamedClass(_iri(node))

    intersection = graph.value(node, OWL.intersectionOf)
    if intersection is not None:
        operands = tuple(_expression(graph, item) for item in _rdf_list(graph, intersection))
        return ObjectIntersectionOf(operands)

    union = graph.value(node, OWL.unionOf)
    if union is not None:
        operands = tuple(_expression(graph, item) for item in _rdf_list(graph, union))
        return ObjectUnionOf(operands)

    one_of = graph.value(node, OWL.oneOf)
    if one_of is not None:
        values = tuple(
            _functional_literal(item)
            for item in _rdf_list(graph, one_of)
            if isinstance(item, Literal)
        )
        if values:
            return DataOneOf(values)

    prop = graph.value(node, OWL.onProperty)
    filler = graph.value(node, OWL.someValuesFrom)
    if isinstance(prop, URIRef) and filler is not None:
        return ObjectSomeValuesFrom(_iri(prop), _expression(graph, filler))

    filler = graph.value(node, OWL.allValuesFrom)
    if isinstance(prop, URIRef) and filler is not None:
        return ObjectAllValuesFrom(_iri(prop), _expression(graph, filler))

    cardinality_type = next(
        (
            kind
            for predicate, kind in (
                (OWL.minCardinality, "min"),
                (OWL.minQualifiedCardinality, "min"),
                (OWL.maxCardinality, "max"),
                (OWL.maxQualifiedCardinality, "max"),
                (OWL.cardinality, "exact"),
                (OWL.qualifiedCardinality, "exact"),
            )
            if graph.value(node, predicate) is not None
        ),
        None,
    )
    if isinstance(prop, URIRef) and cardinality_type is not None:
        qualified_filler = graph.value(node, OWL.onClass)
        return ObjectCardinalityRestriction(
            _iri(prop),
            cardinality_type,
            _expression(graph, qualified_filler) if qualified_filler is not None else None,
        )

    return AnonymousClassExpression(_iri(node))


def _named_iris(expr: ClassExpression) -> Iterable[str]:
    if isinstance(expr, NamedClass):
        yield expr.iri
    elif isinstance(expr, (ObjectSomeValuesFrom, ObjectAllValuesFrom)):
        yield from _named_iris(expr.filler)
    elif isinstance(expr, ObjectCardinalityRestriction) and expr.filler is not None:
        yield from _named_iris(expr.filler)
    elif isinstance(expr, (ObjectIntersectionOf, ObjectUnionOf)):
        for operand in expr.operands:
            yield from _named_iris(operand)


def _object_property_iris(expr: ClassExpression) -> Iterable[str]:
    if isinstance(expr, (ObjectSomeValuesFrom, ObjectAllValuesFrom)):
        yield expr.property_iri
        yield from _object_property_iris(expr.filler)
    elif isinstance(expr, ObjectCardinalityRestriction):
        yield expr.property_iri
        if expr.filler is not None:
            yield from _object_property_iris(expr.filler)
    elif isinstance(expr, (ObjectIntersectionOf, ObjectUnionOf)):
        for operand in expr.operands:
            yield from _object_property_iris(operand)


def _literal_value(property_iri: str, value: Literal) -> AnnotationValue:
    datatype = str(value.datatype) if value.datatype is not None else None
    return AnnotationValue(
        property_iri=property_iri,
        value=str(value),
        is_literal=True,
        lang=value.language,
        datatype=datatype,
    )


def _functional_literal(value: Literal) -> str:
    """Render an RDF literal like the OWL API functional-syntax printer."""

    lexical = str(value).replace("\\", "\\\\").replace('"', '\\"')
    rendered = f'"{lexical}"'
    if value.language:
        return f"{rendered}@{value.language}"
    if value.datatype is None:
        return rendered
    datatype = str(value.datatype)
    xsd = "http://www.w3.org/2001/XMLSchema#"
    compact = f"xsd:{datatype[len(xsd):]}" if datatype.startswith(xsd) else f"<{datatype}>"
    return f"{rendered}^^{compact}"


def _annotation_value(property_iri: str, value: object) -> AnnotationValue:
    if isinstance(value, Literal):
        return _literal_value(property_iri, value)
    return AnnotationValue(property_iri, _iri(value), False)


def _sorted_unique(items: Iterable[object], key) -> tuple:
    return tuple(sorted(set(items), key=key))


def normalize_owl_document(
    path: Path,
    *,
    rdf_xml: str | None = None,
    parser_backend: str = "rdflib",
    horned_error: BaseException | None = None,
) -> ParsedOntology:
    """Normalize an OWL document or Horned-OWL RDF/XML rendering."""

    path = Path(path)
    graph = _load_graph(path, rdf_xml=rdf_xml, horned_error=horned_error)

    signatures: dict[EntityKind, set[str]] = {kind: set() for kind in EntityKind}
    declaration_types = {
        OWL.Class: EntityKind.CLASS,
        OWL.DeprecatedClass: EntityKind.CLASS,
        RDFS.Class: EntityKind.CLASS,
        OWL.ObjectProperty: EntityKind.OBJECT_PROPERTY,
        OWL.DatatypeProperty: EntityKind.DATA_PROPERTY,
        OWL.AnnotationProperty: EntityKind.ANNOTATION_PROPERTY,
        OWL.NamedIndividual: EntityKind.INDIVIDUAL,
    }
    for rdf_type, kind in declaration_types.items():
        signatures[kind].update(
            _iri(subject)
            for subject in graph.subjects(RDF.type, rdf_type)
            if isinstance(subject, URIRef)
        )
    for rdf_type in (
        OWL.AsymmetricProperty,
        OWL.InverseFunctionalProperty,
        OWL.IrreflexiveProperty,
        OWL.ReflexiveProperty,
        OWL.SymmetricProperty,
        OWL.TransitiveProperty,
    ):
        signatures[EntityKind.OBJECT_PROPERTY].update(
            _iri(subject)
            for subject in graph.subjects(RDF.type, rdf_type)
            if isinstance(subject, URIRef)
        )

    ontology_nodes = sorted(
        (_iri(subject) for subject in graph.subjects(RDF.type, OWL.Ontology)), key=str
    )
    ontology_iri = ontology_nodes[0] if ontology_nodes else None

    subclasses: list[SubClassOf] = []
    for sub_node, sup_node in graph.subject_objects(RDFS.subClassOf):
        sub, sup = _expression(graph, sub_node), _expression(graph, sup_node)
        subclasses.append(SubClassOf(sub, sup))
        signatures[EntityKind.CLASS].update(_named_iris(sub))
        signatures[EntityKind.CLASS].update(_named_iris(sup))
        signatures[EntityKind.OBJECT_PROPERTY].update(_object_property_iris(sub))
        signatures[EntityKind.OBJECT_PROPERTY].update(_object_property_iris(sup))

    equivalences: list[EquivalentClasses] = []
    seen_equivalences: set[frozenset[ClassExpression]] = set()
    for first_node, second_node in graph.subject_objects(OWL.equivalentClass):
        first, second = _expression(graph, first_node), _expression(graph, second_node)
        group = frozenset((first, second))
        if len(group) < 2 or group in seen_equivalences:
            continue
        seen_equivalences.add(group)
        expressions = tuple(sorted(group, key=repr))
        equivalences.append(EquivalentClasses(expressions))
        for expression in expressions:
            signatures[EntityKind.CLASS].update(_named_iris(expression))
            signatures[EntityKind.OBJECT_PROPERTY].update(_object_property_iris(expression))

    domains: list[PropertyDomain] = []
    ranges: list[PropertyRange] = []
    for prop_node, domain_node in graph.subject_objects(RDFS.domain):
        if not isinstance(prop_node, URIRef):
            continue
        prop_iri, domain = _iri(prop_node), _expression(graph, domain_node)
        domains.append(PropertyDomain(prop_iri, domain))
        signatures[EntityKind.CLASS].update(_named_iris(domain))
    for prop_node, range_node in graph.subject_objects(RDFS.range):
        if not isinstance(prop_node, URIRef):
            continue
        prop_iri, range_expression = _iri(prop_node), _expression(graph, range_node)
        ranges.append(PropertyRange(prop_iri, range_expression))
        if prop_iri not in signatures[EntityKind.DATA_PROPERTY]:
            signatures[EntityKind.CLASS].update(_named_iris(range_expression))

    subproperties: list[SubPropertyOf] = []
    for sub_node, sup_node in graph.subject_objects(RDFS.subPropertyOf):
        if not isinstance(sub_node, URIRef) or not isinstance(sup_node, URIRef):
            continue
        sub_iri, sup_iri = _iri(sub_node), _iri(sup_node)
        if (
            sub_iri in signatures[EntityKind.DATA_PROPERTY]
            or sup_iri in signatures[EntityKind.DATA_PROPERTY]
        ):
            kind = EntityKind.DATA_PROPERTY
        elif (
            sub_iri in signatures[EntityKind.ANNOTATION_PROPERTY]
            or sup_iri in signatures[EntityKind.ANNOTATION_PROPERTY]
        ):
            kind = EntityKind.ANNOTATION_PROPERTY
        else:
            kind = EntityKind.OBJECT_PROPERTY
        signatures[kind].update((sub_iri, sup_iri))
        subproperties.append(SubPropertyOf(sub_iri, sup_iri, kind))

    declared_properties = set().union(
        signatures[EntityKind.OBJECT_PROPERTY],
        signatures[EntityKind.DATA_PROPERTY],
        signatures[EntityKind.ANNOTATION_PROPERTY],
    )
    for domain_record in domains:
        if domain_record.property_iri not in declared_properties:
            signatures[EntityKind.OBJECT_PROPERTY].add(domain_record.property_iri)
    for range_record in ranges:
        if range_record.property_iri not in declared_properties:
            signatures[EntityKind.OBJECT_PROPERTY].add(range_record.property_iri)

    annotation_properties = set(signatures[EntityKind.ANNOTATION_PROPERTY])
    annotation_properties.update((_iri(RDFS.label), _iri(OWL.deprecated)))
    annotations: list[AnnotationAssertion] = []
    data_assertions: list[DataPropertyAssertion] = []
    object_assertions: list[ObjectPropertyAssertion] = []
    class_assertions: list[ClassAssertion] = []

    individuals = signatures[EntityKind.INDIVIDUAL]
    for subject, obj in graph.subject_objects(RDF.type):
        if not isinstance(subject, URIRef) or obj in _BUILTIN_TYPES:
            continue
        if isinstance(obj, (URIRef, BNode)):
            subject_iri = _iri(subject)
            asserted_expression = _expression(graph, obj)
            individuals.add(subject_iri)
            class_assertions.append(ClassAssertion(subject_iri, asserted_expression))
            signatures[EntityKind.CLASS].update(_named_iris(asserted_expression))

    for subject, predicate, obj in graph:
        if not isinstance(subject, URIRef) or not isinstance(predicate, URIRef):
            continue
        subject_iri, predicate_iri = _iri(subject), _iri(predicate)

        if predicate == RDF.type or predicate in _STRUCTURAL_PREDICATES:
            continue

        if predicate_iri in signatures[EntityKind.DATA_PROPERTY] and isinstance(obj, Literal):
            data_assertions.append(
                DataPropertyAssertion(subject_iri, _literal_value(predicate_iri, obj))
            )
            individuals.add(subject_iri)
            continue

        if (
            subject_iri in individuals
            and isinstance(obj, Literal)
            and predicate_iri not in annotation_properties
        ):
            signatures[EntityKind.DATA_PROPERTY].add(predicate_iri)
            data_assertions.append(
                DataPropertyAssertion(subject_iri, _literal_value(predicate_iri, obj))
            )
            continue

        if predicate_iri in signatures[EntityKind.OBJECT_PROPERTY] and isinstance(obj, URIRef):
            object_assertions.append(ObjectPropertyAssertion(subject_iri, predicate_iri, _iri(obj)))
            individuals.add(subject_iri)
            individuals.add(_iri(obj))
            continue

        if (
            subject_iri in individuals
            and isinstance(obj, URIRef)
            and predicate_iri not in annotation_properties
        ):
            signatures[EntityKind.OBJECT_PROPERTY].add(predicate_iri)
            object_assertions.append(ObjectPropertyAssertion(subject_iri, predicate_iri, _iri(obj)))
            individuals.add(_iri(obj))
            continue

        named_schema_subject = any(
            subject_iri in signatures[kind]
            for kind in (
                EntityKind.CLASS,
                EntityKind.OBJECT_PROPERTY,
                EntityKind.DATA_PROPERTY,
                EntityKind.ANNOTATION_PROPERTY,
            )
        )
        if (
            predicate_iri in annotation_properties
            or isinstance(obj, Literal)
            or named_schema_subject
        ):
            annotation_properties.add(predicate_iri)
            signatures[EntityKind.ANNOTATION_PROPERTY].add(predicate_iri)
            annotations.append(
                AnnotationAssertion(subject_iri, _annotation_value(predicate_iri, obj))
            )

    inverses: list[InverseObjectProperties] = []
    for inverse_first, inverse_second in graph.subject_objects(OWL.inverseOf):
        if isinstance(inverse_first, URIRef) and isinstance(inverse_second, URIRef):
            first_iri, second_iri = _iri(inverse_first), _iri(inverse_second)
            signatures[EntityKind.OBJECT_PROPERTY].update((first_iri, second_iri))
            inverses.append(InverseObjectProperties(first_iri, second_iri))

    symmetric = frozenset(
        _iri(prop)
        for prop in graph.subjects(RDF.type, OWL.SymmetricProperty)
        if isinstance(prop, URIRef)
    )
    signatures[EntityKind.OBJECT_PROPERTY].update(symmetric)

    signature = MappingProxyType(
        {kind: tuple(sorted(values)) for kind, values in signatures.items()}
    )
    return ParsedOntology(
        origin=path.resolve(),
        ontology_iri=ontology_iri,
        signature=signature,
        subclass_axioms=_sorted_unique(subclasses, repr),
        equivalent_class_axioms=_sorted_unique(equivalences, repr),
        annotation_assertions=_sorted_unique(annotations, repr),
        property_domains=_sorted_unique(domains, repr),
        property_ranges=_sorted_unique(ranges, repr),
        subproperty_axioms=_sorted_unique(subproperties, repr),
        class_assertions=_sorted_unique(class_assertions, repr),
        object_property_assertions=_sorted_unique(object_assertions, repr),
        data_property_assertions=_sorted_unique(data_assertions, repr),
        inverse_object_properties=_sorted_unique(inverses, repr),
        symmetric_object_properties=symmetric,
        parser_backend=parser_backend,
    )


__all__ = ["normalize_owl_document"]
