"""OWL document loading behind backend-neutral ontology records.

Horned-OWL is isolated here. RDF graph normalization belongs to the I/O layer,
so importing :mod:`exact.ontology` never imports RDFLib and parser API drift is
contained at this seam.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

try:  # py-horned-owl's import package is named ``pyhornedowl``.
    import pyhornedowl  # type: ignore[import-not-found]
except ImportError:  # Intel macOS currently has no upstream wheel.
    pyhornedowl = None

from exact.ontology.records import (
    AnnotationAssertion,
    AnonymousClassExpression,
    ClassAssertion,
    ClassExpression,
    DataPropertyAssertion,
    EquivalentClasses,
    InverseObjectProperties,
    NamedClass,
    ObjectIntersectionOf,
    ObjectPropertyAssertion,
    ObjectSomeValuesFrom,
    ParsedOntology,
    PropertyDomain,
    PropertyRange,
    SubClassOf,
    SubPropertyOf,
)


def _normalizer() -> Callable[..., ParsedOntology]:
    """Load the RDF normalization adapter only when an ontology is parsed."""

    from exact.io.sources._owl_rdf import normalize_owl_document

    return normalize_owl_document


def parse(path: Path) -> ParsedOntology:
    """Parse an OWL document into backend-neutral, deterministic records."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    normalize = _normalizer()
    horned_error: BaseException | None = None
    if pyhornedowl is not None:
        try:
            ontology = pyhornedowl.open_ontology_from_file(str(path))
            rdf_xml = ontology.save_to_string("owl")
            return normalize(path, rdf_xml=rdf_xml, parser_backend="pyhornedowl")
        except BaseException as exc:  # Rust parser panics are not always Exception subclasses.
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            horned_error = exc

    return normalize(path, horned_error=horned_error)


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
    "parse",
]
