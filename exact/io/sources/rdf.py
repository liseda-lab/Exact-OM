"""RDF-backed implementation of the KnowledgeSource protocol."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    from rdflib import OWL, RDF, RDFS, SKOS, Graph, Literal, URIRef
except ImportError as exc:  # pragma: no cover - exercised in minimal-install smoke tests.
    raise ImportError(
        'RDF inputs require rdflib. Reinstall with `pip install "exact-om"`.'
    ) from exc

from exact.core.contracts.knowledge import KnowledgeSource
from exact.core.entities.graph import AnnotationValue, Edge
from exact.core.entities.kinds import EntityKind
from exact.core.values import ANNOTATION_IRI
from exact.io._common import short_form
from exact.io._hierarchy import HierarchyIndex
from exact.io.sources import SourceOptionsError

RDF_TYPE = str(RDF.type)
RDFS_LABEL = str(RDFS.label)
RDFS_SUBCLASS = str(RDFS.subClassOf)
RDFS_SUBPROPERTY = str(RDFS.subPropertyOf)
RDFS_DOMAIN = str(RDFS.domain)
RDFS_RANGE = str(RDFS.range)
OWL_DEPRECATED = str(OWL.deprecated)

DEFAULT_LABEL_PREDICATES = (
    RDFS_LABEL,
    str(SKOS.prefLabel),
    str(SKOS.altLabel),
)
DEFAULT_HIERARCHY_PREDICATES = (RDFS_SUBCLASS, str(SKOS.broader))
DEFAULT_TYPE_PREDICATES = (RDF_TYPE,)

_STANDARD_TYPES = {
    str(OWL.Class): EntityKind.CLASS,
    str(OWL.DeprecatedClass): EntityKind.CLASS,
    str(RDFS.Class): EntityKind.CLASS,
    str(OWL.ObjectProperty): EntityKind.OBJECT_PROPERTY,
    str(OWL.DatatypeProperty): EntityKind.DATA_PROPERTY,
    str(OWL.AnnotationProperty): EntityKind.ANNOTATION_PROPERTY,
    str(OWL.NamedIndividual): EntityKind.INDIVIDUAL,
    str(RDF.Property): EntityKind.OBJECT_PROPERTY,
}


def _strings(value: Any, *, option: str, default: Sequence[str]) -> tuple[str, ...]:
    if value is None:
        return tuple(default)
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        raise SourceOptionsError(f"{option} must be a string or sequence of strings")
    normalized = tuple(str(item) for item in value)
    if any(not item for item in normalized):
        raise SourceOptionsError(f"{option} cannot contain empty values")
    return normalized


class RdfSource(KnowledgeSource):
    """An eagerly indexed, read-only RDF knowledge source.

    Plain RDF subjects are treated as individuals unless their declarations or
    hierarchy participation identify them as schema entities. The
    ``entity_selector`` option may instead be a type IRI, a sequence of type
    IRIs (selected as classes), or a mapping from ``EntityKind`` values to type
    IRIs.
    """

    def __init__(
        self,
        graph: Graph,
        *,
        origin: Path | None = None,
        label_predicates: Sequence[str] = DEFAULT_LABEL_PREDICATES,
        hierarchy_predicates: Sequence[str] = DEFAULT_HIERARCHY_PREDICATES,
        type_predicates: Sequence[str] = DEFAULT_TYPE_PREDICATES,
        entity_selector: Any = "subjects",
    ) -> None:
        self.graph = graph
        self._origin = origin
        self.label_predicates = tuple(dict.fromkeys(str(item) for item in label_predicates))
        self.hierarchy_predicates = tuple(dict.fromkeys(str(item) for item in hierarchy_predicates))
        self.type_predicates = tuple(dict.fromkeys(str(item) for item in type_predicates))
        self._label_predicate_set = frozenset(self.label_predicates)
        self._hierarchy_predicate_set = frozenset(self.hierarchy_predicates)
        self._type_predicate_set = frozenset(self.type_predicates)

        annotations: dict[str, set[AnnotationValue]] = defaultdict(set)
        relation_targets: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        iri_edges: set[Edge] = set()
        literal_edges: set[Edge] = set()
        subjects: set[str] = set()
        hierarchy_edges: set[tuple[str, str]] = set()
        property_edges: set[tuple[str, str]] = set()
        type_assertions: dict[str, set[str]] = defaultdict(set)
        domains: dict[str, set[str]] = defaultdict(set)
        ranges: dict[str, set[str]] = defaultdict(set)
        predicate_has_iri: set[str] = set()
        predicate_has_literal: set[str] = set()

        for subject, predicate, obj in graph:
            if not isinstance(subject, URIRef) or not isinstance(predicate, URIRef):
                continue
            src, rel = str(subject), str(predicate)
            subjects.add(src)
            if isinstance(obj, URIRef):
                dst = str(obj)
                iri_edges.add(Edge(src, rel, dst))
                relation_targets[rel][src].add(dst)
                predicate_has_iri.add(rel)
                if rel in self._hierarchy_predicate_set:
                    hierarchy_edges.add((src, dst))
                if rel == RDFS_SUBPROPERTY:
                    property_edges.add((src, dst))
                if rel in self._type_predicate_set:
                    type_assertions[src].add(dst)
                if rel == RDFS_DOMAIN:
                    domains[src].add(dst)
                elif rel == RDFS_RANGE:
                    ranges[src].add(dst)
            elif isinstance(obj, Literal):
                value = AnnotationValue(
                    property_iri=rel,
                    value=str(obj),
                    is_literal=True,
                    lang=str(obj.language) if obj.language else None,
                    datatype=str(obj.datatype) if obj.datatype else None,
                )
                annotations[src].add(value)
                literal_edges.add(Edge(src, rel, str(obj)))
                predicate_has_literal.add(rel)

        signature: dict[EntityKind, set[str]] = {entity_kind: set() for entity_kind in EntityKind}
        selected, configured_kinds = self._select_entities(
            subjects, type_assertions, entity_selector
        )
        selects_all_subjects = entity_selector is None or entity_selector == "subjects"
        for configured_kind, configured_values in configured_kinds.items():
            signature[configured_kind].update(configured_values)
        for selected_subject in selected:
            for type_iri in type_assertions.get(selected_subject, ()):
                declared_kind = _STANDARD_TYPES.get(type_iri)
                if declared_kind is not None:
                    signature[declared_kind].add(selected_subject)

        if selects_all_subjects:
            indexed_hierarchy_edges = hierarchy_edges
            signature[EntityKind.CLASS].update(iri for edge in hierarchy_edges for iri in edge)
        else:
            indexed_hierarchy_edges = {
                (child, parent)
                for child, parent in hierarchy_edges
                if child in selected and parent in selected
            }

        for relation in predicate_has_iri:
            if relation in self._label_predicate_set:
                signature[EntityKind.ANNOTATION_PROPERTY].add(relation)
            else:
                signature[EntityKind.OBJECT_PROPERTY].add(relation)
        signature[EntityKind.DATA_PROPERTY].update(
            predicate_has_literal - self._label_predicate_set
        )
        signature[EntityKind.ANNOTATION_PROPERTY].update(
            predicate_has_literal & self._label_predicate_set
        )

        schema_entities = set().union(
            signature[EntityKind.CLASS],
            signature[EntityKind.OBJECT_PROPERTY],
            signature[EntityKind.DATA_PROPERTY],
            signature[EntityKind.ANNOTATION_PROPERTY],
        )
        for untyped_subject in selected - schema_entities:
            signature[EntityKind.INDIVIDUAL].add(untyped_subject)
        for asserted_subject, asserted_types in type_assertions.items():
            if asserted_subject in selected and any(
                type_iri not in _STANDARD_TYPES for type_iri in asserted_types
            ):
                if asserted_subject not in schema_entities:
                    signature[EntityKind.INDIVIDUAL].add(asserted_subject)

        self._signature = {
            entity_kind: tuple(sorted(entity_values))
            for entity_kind, entity_values in signature.items()
        }
        self.hierarchy = HierarchyIndex(
            self._signature[EntityKind.CLASS],
            indexed_hierarchy_edges,
            filter_owl_bounds=True,
        )
        property_entities = set().union(
            self._signature[EntityKind.OBJECT_PROPERTY],
            self._signature[EntityKind.DATA_PROPERTY],
            self._signature[EntityKind.ANNOTATION_PROPERTY],
        )
        self._property_hierarchy = HierarchyIndex(
            property_entities, property_edges, filter_owl_bounds=False
        )
        self._annotations = {
            iri: tuple(sorted(annotation_values, key=self._annotation_key))
            for iri, annotation_values in annotations.items()
        }
        self._labels = {
            iri: tuple(
                value.value
                for value in sorted(annotation_values, key=self._annotation_key)
                if value.property_iri in self._label_predicate_set
            )
            for iri, annotation_values in annotations.items()
        }
        self._attributes = {
            iri: tuple(
                value
                for value in sorted(annotation_values, key=self._annotation_key)
                if value.property_iri not in self._label_predicate_set
            )
            for iri, annotation_values in annotations.items()
        }
        self._relation_targets = {
            relation: {subject: tuple(sorted(targets)) for subject, targets in subject_map.items()}
            for relation, subject_map in relation_targets.items()
        }
        self._type_assertions = {
            iri: tuple(sorted(asserted_types)) for iri, asserted_types in type_assertions.items()
        }
        self._individual_parents = {
            iri: tuple(
                type_iri
                for type_iri in asserted_types
                if _STANDARD_TYPES.get(type_iri) in {None, EntityKind.CLASS}
            )
            for iri, asserted_types in self._type_assertions.items()
        }
        children_by_type: dict[str, set[str]] = defaultdict(set)
        for iri in self._signature[EntityKind.INDIVIDUAL]:
            for type_iri in self._individual_parents.get(iri, ()):
                children_by_type[type_iri].add(iri)
        self._children_by_type = {
            iri: tuple(sorted(children)) for iri, children in children_by_type.items()
        }
        self._domains = {iri: tuple(sorted(items)) for iri, items in domains.items()}
        self._ranges = {iri: tuple(sorted(items)) for iri, items in ranges.items()}
        self._iri_edges = tuple(sorted(iri_edges, key=Edge.astuple))
        self._literal_edges = tuple(sorted(literal_edges, key=Edge.astuple))
        excluded: set[str] = set()
        for iri, annotation_values in self._annotations.items():
            for annotation in annotation_values:
                lexical = annotation.value.strip().lower()
                if annotation.property_iri == ANNOTATION_IRI and lexical in {"false", "0"}:
                    excluded.add(iri)
                elif annotation.property_iri == OWL_DEPRECATED and lexical in {"true", "1"}:
                    excluded.add(iri)
        self._excluded = frozenset(excluded)

    @staticmethod
    def _annotation_key(value: AnnotationValue) -> tuple[str, str, str, str]:
        return (
            value.property_iri,
            value.value,
            value.lang or "",
            value.datatype or "",
        )

    @staticmethod
    def _select_entities(
        subjects: set[str],
        type_assertions: Mapping[str, set[str]],
        selector: Any,
    ) -> tuple[set[str], dict[EntityKind, set[str]]]:
        configured: dict[EntityKind, set[str]] = {kind: set() for kind in EntityKind}
        if selector == "subjects" or selector is None:
            return set(subjects), configured
        if isinstance(selector, Mapping):
            selected: set[str] = set()
            for raw_kind, raw_types in selector.items():
                try:
                    kind = EntityKind(str(raw_kind))
                except ValueError as exc:
                    raise SourceOptionsError(
                        f"entity_selector contains unknown kind {raw_kind!r}"
                    ) from exc
                type_iris = _strings(
                    raw_types,
                    option=f"entity_selector.{kind.value}",
                    default=(),
                )
                matches = {
                    subject
                    for subject, asserted in type_assertions.items()
                    if set(type_iris) & asserted
                }
                configured[kind].update(matches)
                selected.update(matches)
            return selected, configured
        type_iris = _strings(selector, option="entity_selector", default=())
        selected = {
            subject for subject, asserted in type_assertions.items() if set(type_iris) & asserted
        }
        configured[EntityKind.CLASS].update(selected)
        return selected, configured

    @classmethod
    def from_path(
        cls,
        path: Path,
        *,
        label_predicates: Sequence[str] = DEFAULT_LABEL_PREDICATES,
        hierarchy_predicates: Sequence[str] = DEFAULT_HIERARCHY_PREDICATES,
        type_predicates: Sequence[str] = DEFAULT_TYPE_PREDICATES,
        entity_selector: Any = "subjects",
        parser_format: str | None = None,
    ) -> "RdfSource":
        """Parse and index an RDF document."""

        source_path = Path(path)
        graph = Graph()
        graph.parse(source_path, format=parser_format)
        return cls(
            graph,
            origin=source_path,
            label_predicates=label_predicates,
            hierarchy_predicates=hierarchy_predicates,
            type_predicates=type_predicates,
            entity_selector=entity_selector,
        )

    @property
    def origin(self) -> Path | None:
        return self._origin

    def entities(self, kind: EntityKind = EntityKind.CLASS) -> Sequence[str]:
        """Return the indexed identifiers for ``kind``."""

        try:
            return self._signature[EntityKind(kind)]
        except ValueError as exc:
            raise ValueError(f"Unknown entity kind: {kind!r}") from exc

    def labels(self, iri: str) -> list[str]:
        """Return labels selected by the configured label predicates."""

        return list(self._labels.get(str(iri), ()))

    def annotations(
        self, iri: str, properties: Sequence[str] | None = None
    ) -> list[AnnotationValue]:
        """Return literal annotations, optionally filtered by predicate."""

        values = self._annotations.get(str(iri), ())
        if properties is None:
            return list(values)
        selected = frozenset(str(item) for item in properties)
        return [value for value in values if value.property_iri in selected]

    def attributes(self, iri: str) -> list[AnnotationValue]:
        """Return literal annotations that are not labels."""

        return list(self._attributes.get(str(iri), ()))

    def direct_parents(self, iri: str, kind: EntityKind = EntityKind.CLASS) -> list[str]:
        """Return direct parents using the kind-appropriate hierarchy."""

        normalized_kind = EntityKind(kind)
        if normalized_kind == EntityKind.CLASS:
            return self.hierarchy.direct_parents(str(iri))
        if normalized_kind == EntityKind.INDIVIDUAL:
            return list(self._individual_parents.get(str(iri), ()))
        return self._property_hierarchy.direct_parents(str(iri))

    def direct_children(self, iri: str, kind: EntityKind = EntityKind.CLASS) -> list[str]:
        """Return direct children using the kind-appropriate hierarchy."""

        normalized_kind = EntityKind(kind)
        if normalized_kind == EntityKind.CLASS:
            return self.hierarchy.direct_children(str(iri))
        if normalized_kind == EntityKind.INDIVIDUAL:
            return list(self._children_by_type.get(str(iri), ()))
        return self._property_hierarchy.direct_children(str(iri))

    def hierarchy_bundle(
        self, iri: str, families: Mapping[str, Sequence[str]]
    ) -> dict[str, list[str]]:
        """Collect configured hierarchy-predicate targets by family."""

        result: dict[str, list[str]] = {}
        for family, predicates in families.items():
            if family == "is_a":
                result[family] = self.direct_parents(iri)
                continue
            values: set[str] = set()
            for predicate in predicates:
                values.update(self._relation_targets.get(str(predicate), {}).get(str(iri), ()))
            result[family] = sorted(values)
        return result

    def projection_edges(
        self, *, method: str = "owl2vecstar", include_literals: bool = False
    ) -> list[Edge]:
        """Return RDF graph edges, optionally including literal objects."""

        del method
        edges = self._iri_edges + (self._literal_edges if include_literals else ())
        return sorted(edges, key=Edge.astuple)

    def property_domains(self, prop_iri: str) -> list[str]:
        """Return named ``rdfs:domain`` targets for a property."""

        return list(self._domains.get(str(prop_iri), ()))

    def property_ranges(self, prop_iri: str) -> list[str]:
        """Return named ``rdfs:range`` targets for a property."""

        return list(self._ranges.get(str(prop_iri), ()))

    def excluded_from_alignment(self) -> frozenset[str]:
        """Return deprecated or explicitly disabled identifiers."""

        return self._excluded

    def short_form(self, iri: str) -> str:
        """Return a compact display form for ``iri``."""

        return short_form(iri)

    def ancestors(self, iri: str) -> set[str]:
        """Return all hierarchy ancestors for relation-typing consumers."""

        return self.hierarchy.ancestors(str(iri))

    def descendants(self, iri: str) -> set[str]:
        """Return all hierarchy descendants for relation-typing consumers."""

        return self.hierarchy.descendants(str(iri))


def create_source(path: Path, *, options: Mapping[str, Any] | None = None) -> RdfSource:
    """Create an :class:`RdfSource` from registry options."""

    normalized = dict(options or {})
    allowed = {
        "entity_selector",
        "hierarchy_predicates",
        "label_predicates",
        "parser_format",
        "type_predicates",
    }
    unknown = sorted(set(normalized) - allowed)
    if unknown:
        raise SourceOptionsError(f"Unknown RDF source option(s): {', '.join(unknown)}")
    return RdfSource.from_path(
        path,
        label_predicates=_strings(
            normalized.get("label_predicates"),
            option="label_predicates",
            default=DEFAULT_LABEL_PREDICATES,
        ),
        hierarchy_predicates=_strings(
            normalized.get("hierarchy_predicates"),
            option="hierarchy_predicates",
            default=DEFAULT_HIERARCHY_PREDICATES,
        ),
        type_predicates=_strings(
            normalized.get("type_predicates"),
            option="type_predicates",
            default=DEFAULT_TYPE_PREDICATES,
        ),
        entity_selector=normalized.get("entity_selector", "subjects"),
        parser_format=(
            str(normalized["parser_format"])
            if normalized.get("parser_format") is not None
            else None
        ),
    )


__all__ = [
    "DEFAULT_HIERARCHY_PREDICATES",
    "DEFAULT_LABEL_PREDICATES",
    "DEFAULT_TYPE_PREDICATES",
    "RdfSource",
    "create_source",
]
