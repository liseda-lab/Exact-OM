"""Exact facade over the shared Java-free OWL stack."""

from os import PathLike
from typing import BinaryIO, Sequence

import pyowl_core
from pyowl_core import OntologySnapshot

from exact.ontology.store import OwlOntologySource


def load_ontology(
    source: OntologySnapshot | str | PathLike[str] | bytes | bytearray | memoryview | BinaryIO,
    *,
    label_properties: Sequence[str] | None = None,
    options: pyowl_core.LoadOptions | None = None,
    resolver: pyowl_core.ImportResolver | None = None,
    document_iri: pyowl_core.IRI | str | None = None,
) -> OwlOntologySource:
    """Wrap an existing snapshot or load one ontology closure exactly once."""

    if isinstance(source, OntologySnapshot):
        return OwlOntologySource(source, label_properties=label_properties)
    return OwlOntologySource.load(
        source,
        label_properties=label_properties,
        options=options,
        resolver=resolver,
        document_iri=document_iri,
    )


__all__ = ["OwlOntologySource", "load_ontology"]
