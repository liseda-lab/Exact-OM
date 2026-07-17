"""Deprecated structural-name imports backed exactly by :mod:`pyowl_core`.

Exact 2.1 owns no OWL records.  This module remains for one minor release so
imports fail gradually; every exported object is the authoritative core type.
New code must import these names from :mod:`pyowl_core` directly.
"""

from pyowl_core import (
    AnnotationAssertion,
    Class,
    ClassAssertion,
    ClassExpression,
    DataOneOf,
    DataPropertyAssertion,
    EquivalentClasses,
    InverseObjectProperties,
    ObjectAllValuesFrom,
    ObjectExactCardinality,
    ObjectIntersectionOf,
    ObjectMaxCardinality,
    ObjectMinCardinality,
    ObjectPropertyAssertion,
    ObjectPropertyDomain,
    ObjectPropertyRange,
    ObjectSomeValuesFrom,
    ObjectUnionOf,
    OntologySnapshot,
    SubClassOf,
    SubObjectPropertyOf,
)

# Historical spellings whose authoritative core counterparts are exact.
NamedClass = Class
ParsedOntology = OntologySnapshot
PropertyDomain = ObjectPropertyDomain
PropertyRange = ObjectPropertyRange
SubPropertyOf = SubObjectPropertyOf

__all__ = [
    "AnnotationAssertion",
    "Class",
    "ClassAssertion",
    "ClassExpression",
    "DataOneOf",
    "DataPropertyAssertion",
    "EquivalentClasses",
    "InverseObjectProperties",
    "NamedClass",
    "ObjectAllValuesFrom",
    "ObjectExactCardinality",
    "ObjectIntersectionOf",
    "ObjectMaxCardinality",
    "ObjectMinCardinality",
    "ObjectPropertyAssertion",
    "ObjectPropertyDomain",
    "ObjectPropertyRange",
    "ObjectSomeValuesFrom",
    "ObjectUnionOf",
    "OntologySnapshot",
    "ParsedOntology",
    "PropertyDomain",
    "PropertyRange",
    "SubClassOf",
    "SubObjectPropertyOf",
    "SubPropertyOf",
]
