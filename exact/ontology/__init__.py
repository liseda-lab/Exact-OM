"""Exact facade over the shared Java-free OWL stack."""

from os import PathLike
from typing import BinaryIO, Sequence, TypeAlias, cast

import pyowl_core
from pyowl_core import OntologyView

from exact.ontology.store import OwlOntologySource
from exact.ontology.view_contract import is_ontology_view

_DocumentSource: TypeAlias = str | PathLike[str] | bytes | bytearray | memoryview | BinaryIO


def load_ontology(
    source: OntologyView | pyowl_core.SnapshotProvider | _DocumentSource,
    *,
    label_properties: Sequence[str] | None = None,
    options: pyowl_core.LoadOptions | None = None,
    resolver: pyowl_core.ImportResolver | None = None,
    document_iri: pyowl_core.IRI | str | None = None,
) -> OwlOntologySource:
    """Retain an existing view/provider or load one ontology closure exactly once."""

    if is_ontology_view(source):
        view = pyowl_core.coerce_snapshot(
            source,
            options=options,
            resolver=resolver,
            document_iri=document_iri,
        )
        return OwlOntologySource(view, label_properties=label_properties)
    if isinstance(source, pyowl_core.SnapshotProvider):
        view = pyowl_core.coerce_snapshot(
            source,
            options=options,
            resolver=resolver,
            document_iri=document_iri,
        )
        return OwlOntologySource(view, label_properties=label_properties)
    return OwlOntologySource.load(
        cast(_DocumentSource, source),
        label_properties=label_properties,
        options=options,
        resolver=resolver,
        document_iri=document_iri,
    )


__all__ = ["OwlOntologySource", "load_ontology"]
