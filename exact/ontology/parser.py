"""OWL parsing and normalization.

This is deliberately the only module which imports parser implementations.  The
rest of :mod:`exact.ontology` consumes the immutable records below.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping

try:  # py-horned-owl's import package is named ``pyhornedowl``.
    import pyhornedowl  # type: ignore[import-not-found]
except ImportError:  # Intel macOS currently has no upstream wheel.
    pyhornedowl = None

try:
    from rdflib import BNode, Graph, Literal, URIRef
    from rdflib.namespace import OWL, RDF, RDFS
except ImportError as exc:  # pragma: no cover - dependency error is tested via import isolation.
    raise RuntimeError(
        "OWL parsing requires rdflib. Install Exact-OM with its core dependencies."
    ) from exc

from exact.core.entities.graph import AnnotationValue
from exact.core.entities.kinds import EntityKind


def _iri(value: object) -> str:
    return sys.intern(str(value))


@dataclass(frozen=True, slots=True)
class NamedClass:
    iri: str


@dataclass(frozen=True, slots=True)
class ObjectSomeValuesFrom:
    property_iri: str
    filler: "ClassExpression"


@dataclass(frozen=True, slots=True)
class ObjectIntersectionOf:
    operands: tuple["ClassExpression", ...]


@dataclass(frozen=True, slots=True)
class AnonymousClassExpression:
    """An unsupported expression retained without leaking parser-native objects."""

    identifier: str


ClassExpression = (
    NamedClass | ObjectSomeValuesFrom | ObjectIntersectionOf | AnonymousClassExpression
)


@dataclass(frozen=True, slots=True)
class SubClassOf:
    sub: ClassExpression
    sup: ClassExpression


@dataclass(frozen=True, slots=True)
class EquivalentClasses:
    expressions: tuple[ClassExpression, ...]


@dataclass(frozen=True, slots=True)
class AnnotationAssertion:
    subject_iri: str
    value: AnnotationValue


@dataclass(frozen=True, slots=True)
class PropertyDomain:
    property_iri: str
    domain: ClassExpression


@dataclass(frozen=True, slots=True)
class PropertyRange:
    property_iri: str
    range: ClassExpression


@dataclass(frozen=True, slots=True)
class SubPropertyOf:
    sub_property_iri: str
    super_property_iri: str
    kind: EntityKind


@dataclass(frozen=True, slots=True)
class ClassAssertion:
    individual_iri: str
    class_expression: ClassExpression


@dataclass(frozen=True, slots=True)
class ObjectPropertyAssertion:
    subject_iri: str
    property_iri: str
    object_iri: str


@dataclass(frozen=True, slots=True)
class DataPropertyAssertion:
    subject_iri: str
    value: AnnotationValue


@dataclass(frozen=True, slots=True)
class InverseObjectProperties:
    first_property_iri: str
    second_property_iri: str


@dataclass(frozen=True, slots=True)
class ParsedOntology:
    origin: Path
    ontology_iri: str | None
    signature: Mapping[EntityKind, tuple[str, ...]]
    subclass_axioms: tuple[SubClassOf, ...] = ()
    equivalent_class_axioms: tuple[EquivalentClasses, ...] = ()
    annotation_assertions: tuple[AnnotationAssertion, ...] = ()
    property_domains: tuple[PropertyDomain, ...] = ()
    property_ranges: tuple[PropertyRange, ...] = ()
    subproperty_axioms: tuple[SubPropertyOf, ...] = ()
    class_assertions: tuple[ClassAssertion, ...] = ()
    object_property_assertions: tuple[ObjectPropertyAssertion, ...] = ()
    data_property_assertions: tuple[DataPropertyAssertion, ...] = ()
    inverse_object_properties: tuple[InverseObjectProperties, ...] = ()
    symmetric_object_properties: frozenset[str] = frozenset()
    parser_backend: str = field(default="rdflib", compare=False)


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
    OWL.imports,
    OWL.intersectionOf,
    OWL.inverseOf,
    OWL.onProperty,
    OWL.someValuesFrom,
    OWL.unionOf,
}


def _load_graph(path: Path) -> tuple[Graph, str]:
    """Load through Horned-OWL when available, then normalize its RDF/XML."""

    horned_error: BaseException | None = None
    if pyhornedowl is not None:
        try:
            ontology = pyhornedowl.open_ontology_from_file(str(path))
            rdf_xml = ontology.save_to_string("owl")
            graph = Graph()
            graph.parse(data=rdf_xml, format="xml", publicID=path.resolve().as_uri())
            return graph, "pyhornedowl"
        except BaseException as exc:  # Rust parser panics are not always Exception subclasses.
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            horned_error = exc

    graph = Graph()
    try:
        graph.parse(path.resolve().as_uri())
        return graph, "rdflib"
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

    prop = graph.value(node, OWL.onProperty)
    filler = graph.value(node, OWL.someValuesFrom)
    if isinstance(prop, URIRef) and filler is not None:
        return ObjectSomeValuesFrom(_iri(prop), _expression(graph, filler))

    return AnonymousClassExpression(_iri(node))


def _named_iris(expr: ClassExpression) -> Iterable[str]:
    if isinstance(expr, NamedClass):
        yield expr.iri
    elif isinstance(expr, ObjectSomeValuesFrom):
        yield from _named_iris(expr.filler)
    elif isinstance(expr, ObjectIntersectionOf):
        for operand in expr.operands:
            yield from _named_iris(operand)


def _object_property_iris(expr: ClassExpression) -> Iterable[str]:
    if isinstance(expr, ObjectSomeValuesFrom):
        yield expr.property_iri
        yield from _object_property_iris(expr.filler)
    elif isinstance(expr, ObjectIntersectionOf):
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


def _annotation_value(property_iri: str, value: object) -> AnnotationValue:
    if isinstance(value, Literal):
        return _literal_value(property_iri, value)
    return AnnotationValue(property_iri, _iri(value), False)


def _sorted_unique(items: Iterable[object], key) -> tuple:
    return tuple(sorted(set(items), key=key))


def parse(path: Path) -> ParsedOntology:
    """Parse an OWL document into backend-neutral, deterministic records."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    graph, backend = _load_graph(path)

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
        for expr in expressions:
            signatures[EntityKind.CLASS].update(_named_iris(expr))
            signatures[EntityKind.OBJECT_PROPERTY].update(_object_property_iris(expr))

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
        prop_iri, range_expr = _iri(prop_node), _expression(graph, range_node)
        ranges.append(PropertyRange(prop_iri, range_expr))
        if prop_iri not in signatures[EntityKind.DATA_PROPERTY]:
            signatures[EntityKind.CLASS].update(_named_iris(range_expr))

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

    # Domain/range predicates inherit their declared property kind, defaulting to object.
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
    # OWL declarations are optional.  Harvest class assertions first so the
    # classification of subsequent property assertions does not depend on
    # rdflib's graph iteration order.
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

        if predicate == RDF.type:
            continue

        if predicate in _STRUCTURAL_PREDICATES:
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

        # Literal predicates on named classes are annotation properties even when a
        # producer omitted their declaration (common in Bio-ML ontologies).
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
        parser_backend=backend,
    )
