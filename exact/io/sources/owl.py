"""OWL source adapter backed by :mod:`exact.ontology`."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from exact.core.contracts.knowledge import KnowledgeSource
from exact.core.entities.graph import AnnotationValue, Edge
from exact.core.entities.kinds import EntityKind
from exact.io.sources import SourceOptionsError
from exact.ontology import OwlOntologySource, load_ontology


class _SchemaOnlyOwlSource(KnowledgeSource):
    """A view that omits ABox entities and projected ABox assertions."""

    def __init__(self, source: OwlOntologySource) -> None:
        self._source = source
        self._individuals = frozenset(source.entities(EntityKind.INDIVIDUAL))

    @property
    def origin(self) -> Path | None:
        return self._source.origin

    def entities(self, kind: EntityKind = EntityKind.CLASS) -> Sequence[str]:
        return () if EntityKind(kind) == EntityKind.INDIVIDUAL else self._source.entities(kind)

    def labels(self, iri: str) -> list[str]:
        return [] if iri in self._individuals else self._source.labels(iri)

    def annotations(
        self, iri: str, properties: Sequence[str] | None = None
    ) -> list[AnnotationValue]:
        return [] if iri in self._individuals else self._source.annotations(iri, properties)

    def attributes(self, iri: str) -> list[AnnotationValue]:
        return [] if iri in self._individuals else self._source.attributes(iri)

    def direct_parents(self, iri: str, kind: EntityKind = EntityKind.CLASS) -> list[str]:
        if EntityKind(kind) == EntityKind.INDIVIDUAL:
            return []
        return self._source.direct_parents(iri, kind)

    def direct_children(self, iri: str, kind: EntityKind = EntityKind.CLASS) -> list[str]:
        if EntityKind(kind) == EntityKind.INDIVIDUAL:
            return []
        return self._source.direct_children(iri, kind)

    def hierarchy_bundle(
        self, iri: str, families: Mapping[str, Sequence[str]]
    ) -> dict[str, list[str]]:
        return self._source.hierarchy_bundle(iri, families)

    def projection_edges(
        self, *, method: str = "owl2vecstar", include_literals: bool = False
    ) -> list[Edge]:
        return [
            edge
            for edge in self._source.projection_edges(
                method=method, include_literals=include_literals
            )
            if edge.src not in self._individuals and edge.dst not in self._individuals
        ]

    def property_domains(self, prop_iri: str) -> list[str]:
        return self._source.property_domains(prop_iri)

    def property_ranges(self, prop_iri: str) -> list[str]:
        return self._source.property_ranges(prop_iri)

    def excluded_from_alignment(self) -> frozenset[str]:
        return self._source.excluded_from_alignment() - self._individuals

    def short_form(self, iri: str) -> str:
        return self._source.short_form(iri)


def create_source(path: Path, *, options: Mapping[str, Any] | None = None) -> KnowledgeSource:
    """Load an OWL source with optional labels and ABox filtering."""

    normalized = dict(options or {})
    unknown = sorted(set(normalized) - {"include_abox", "label_properties"})
    if unknown:
        raise SourceOptionsError(f"Unknown OWL source option(s): {', '.join(unknown)}")
    label_properties = normalized.get("label_properties")
    if label_properties is not None:
        if isinstance(label_properties, str):
            label_properties = (label_properties,)
        elif not isinstance(label_properties, Sequence):
            raise SourceOptionsError("label_properties must be a string or sequence")
        label_properties = tuple(str(item) for item in label_properties)
    include_abox = normalized.get("include_abox", True)
    if not isinstance(include_abox, bool):
        raise SourceOptionsError("include_abox must be a boolean")
    source = load_ontology(Path(path), label_properties=label_properties)
    return source if include_abox else _SchemaOnlyOwlSource(source)


__all__ = ["create_source"]
