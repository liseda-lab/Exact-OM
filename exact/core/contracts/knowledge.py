"""Backend-neutral read-only knowledge-source protocol."""

from pathlib import Path
from typing import Mapping, Protocol, Sequence, runtime_checkable

from exact.core.entities.graph import AnnotationValue, Edge
from exact.core.entities.kinds import EntityKind


@runtime_checkable
class KnowledgeSource(Protocol):
    """Read-only backend seam consumed by datasets and graph views."""

    @property
    def origin(self) -> Path | None: ...

    def entities(self, kind: EntityKind = EntityKind.CLASS) -> Sequence[str]:
        """Return entity identifiers in the requested signature."""

        ...

    def labels(self, iri: str) -> list[str]:
        """Return preferred and alternate labels for an entity."""

        ...

    def annotations(
        self, iri: str, properties: Sequence[str] | None = None
    ) -> list[AnnotationValue]:
        """Return annotations, optionally restricted to property identifiers."""

        ...

    def attributes(self, iri: str) -> list[AnnotationValue]:
        """Return non-label literal attributes for an entity."""

        ...

    def direct_parents(self, iri: str, kind: EntityKind = EntityKind.CLASS) -> list[str]:
        """Return asserted direct parents for an entity and kind."""

        ...

    def direct_children(self, iri: str, kind: EntityKind = EntityKind.CLASS) -> list[str]:
        """Return asserted direct children for an entity and kind."""

        ...

    def hierarchy_bundle(
        self, iri: str, families: Mapping[str, Sequence[str]]
    ) -> dict[str, list[str]]:
        """Collect hierarchy targets grouped by configured relation family."""

        ...

    def projection_edges(
        self, *, method: str = "owl2vecstar", include_literals: bool = False
    ) -> list[Edge]:
        """Project the knowledge source into graph-search edges."""

        ...

    def property_domains(self, prop_iri: str) -> list[str]:
        """Return named domains declared for a property."""

        ...

    def property_ranges(self, prop_iri: str) -> list[str]:
        """Return named ranges declared for a property."""

        ...

    def excluded_from_alignment(self) -> frozenset[str]:
        """Return entity identifiers explicitly excluded from matching."""

        ...

    def short_form(self, iri: str) -> str:
        """Return a compact display form for an entity identifier."""

        ...
