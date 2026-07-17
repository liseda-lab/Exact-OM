"""Deprecated thin loading/coercion shim over :mod:`pyowl_core`."""

from __future__ import annotations

from os import PathLike
from typing import BinaryIO

import pyowl_core
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

NamedClass = Class
ParsedOntology = OntologySnapshot
PropertyDomain = ObjectPropertyDomain
PropertyRange = ObjectPropertyRange
SubPropertyOf = SubObjectPropertyOf


def parse(
    source: OntologySnapshot | str | PathLike[str] | bytes | bytearray | memoryview | BinaryIO,
    *,
    document_iri: pyowl_core.IRI | str | None = None,
    options: pyowl_core.LoadOptions | None = None,
    resolver: pyowl_core.ImportResolver | None = None,
) -> OntologySnapshot:
    """Return one concrete snapshot, preserving existing identity when supplied."""

    if isinstance(source, OntologySnapshot):
        return source
    if document_iri is None and not isinstance(
        source, (str, PathLike, bytes, bytearray, memoryview)
    ):
        document_iri = "urn:exact-om:stream-root"
    snapshot = pyowl_core.load_snapshot(
        source,
        document_iri=document_iri,
        options=options,
        resolver=resolver,
    )
    if not isinstance(snapshot, OntologySnapshot):  # pragma: no cover - core contract
        raise TypeError("pyowl_core.load_snapshot did not return OntologySnapshot")
    return snapshot


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
    "parse",
]
