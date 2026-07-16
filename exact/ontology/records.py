"""Backend-neutral records produced by ontology document parsers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from exact.core.entities.graph import AnnotationValue
from exact.core.entities.kinds import EntityKind


@dataclass(frozen=True, slots=True)
class NamedClass:
    """A named OWL class expression."""

    iri: str


@dataclass(frozen=True, slots=True)
class ObjectSomeValuesFrom:
    """An object-property existential restriction."""

    property_iri: str
    filler: "ClassExpression"


@dataclass(frozen=True, slots=True)
class ObjectIntersectionOf:
    """A conjunction of class expressions."""

    operands: tuple["ClassExpression", ...]


@dataclass(frozen=True, slots=True)
class AnonymousClassExpression:
    """An unsupported expression retained without parser-native objects."""

    identifier: str


ClassExpression = (
    NamedClass | ObjectSomeValuesFrom | ObjectIntersectionOf | AnonymousClassExpression
)


@dataclass(frozen=True, slots=True)
class SubClassOf:
    """A normalized subclass axiom."""

    sub: ClassExpression
    sup: ClassExpression


@dataclass(frozen=True, slots=True)
class EquivalentClasses:
    """A normalized equivalent-classes axiom."""

    expressions: tuple[ClassExpression, ...]


@dataclass(frozen=True, slots=True)
class AnnotationAssertion:
    """An annotation assertion on a named entity."""

    subject_iri: str
    value: AnnotationValue


@dataclass(frozen=True, slots=True)
class PropertyDomain:
    """A normalized property-domain axiom."""

    property_iri: str
    domain: ClassExpression


@dataclass(frozen=True, slots=True)
class PropertyRange:
    """A normalized property-range axiom."""

    property_iri: str
    range: ClassExpression


@dataclass(frozen=True, slots=True)
class SubPropertyOf:
    """A normalized subproperty axiom with its entity kind."""

    sub_property_iri: str
    super_property_iri: str
    kind: EntityKind


@dataclass(frozen=True, slots=True)
class ClassAssertion:
    """A normalized class assertion for a named individual."""

    individual_iri: str
    class_expression: ClassExpression


@dataclass(frozen=True, slots=True)
class ObjectPropertyAssertion:
    """A normalized object-property assertion."""

    subject_iri: str
    property_iri: str
    object_iri: str


@dataclass(frozen=True, slots=True)
class DataPropertyAssertion:
    """A normalized data-property assertion."""

    subject_iri: str
    value: AnnotationValue


@dataclass(frozen=True, slots=True)
class InverseObjectProperties:
    """A normalized inverse-object-properties axiom."""

    first_property_iri: str
    second_property_iri: str


@dataclass(frozen=True, slots=True)
class ParsedOntology:
    """Immutable, backend-neutral representation of an ontology document."""

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


__all__ = [
    "AnnotationAssertion",
    "AnonymousClassExpression",
    "ClassAssertion",
    "ClassExpression",
    "DataPropertyAssertion",
    "EquivalentClasses",
    "InverseObjectProperties",
    "NamedClass",
    "ObjectIntersectionOf",
    "ObjectPropertyAssertion",
    "ObjectSomeValuesFrom",
    "ParsedOntology",
    "PropertyDomain",
    "PropertyRange",
    "SubClassOf",
    "SubPropertyOf",
]
