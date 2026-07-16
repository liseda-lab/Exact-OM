"""Indexed OWL implementation of the :class:`KnowledgeSource` contract."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from threading import RLock
from urllib.parse import unquote, urlsplit

from exact.core.contracts.knowledge import KnowledgeSource
from exact.core.entities.graph import AnnotationValue, Edge
from exact.core.entities.kinds import EntityKind
from exact.core.values import ANNOTATION_IRI
from exact.ontology.expressions import (
    existential_targets,
    intersection_operands,
    named_class_iri,
    named_class_iris,
    render_class_expression,
)
from exact.ontology.hierarchy import HierarchyIndex
from exact.ontology.parser import NamedClass, ObjectUnionOf, ParsedOntology, parse
from exact.ontology.projection import project

RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
OWL_DEPRECATED = "http://www.w3.org/2002/07/owl#deprecated"


class OwlOntologySource(KnowledgeSource):
    """A read-only, eagerly indexed OWL knowledge source."""

    def __init__(
        self,
        parsed: ParsedOntology,
        *,
        label_properties: Sequence[str] | None = None,
    ) -> None:
        self.parsed = parsed
        self._origin = parsed.origin
        self.ontology_iri = parsed.ontology_iri
        self.label_properties = tuple(
            (RDFS_LABEL,) if label_properties is None else label_properties
        )
        self._label_property_set = frozenset(self.label_properties)
        self._projection_cache: dict[tuple[str, bool], tuple[Edge, ...]] = {}
        self._projection_lock = RLock()

        annotations: dict[str, list[AnnotationValue]] = defaultdict(list)
        for annotation_assertion in parsed.annotation_assertions:
            annotations[annotation_assertion.subject_iri].append(annotation_assertion.value)
        self._annotations = {
            iri: tuple(
                sorted(
                    set(values),
                    key=lambda value: (
                        value.property_iri,
                        value.value,
                        value.lang or "",
                        value.datatype or "",
                        value.is_literal,
                    ),
                )
            )
            for iri, values in annotations.items()
        }

        labels: dict[str, list[AnnotationValue]] = defaultdict(list)
        attributes: dict[str, list[AnnotationValue]] = defaultdict(list)
        for iri, values in self._annotations.items():
            for value in values:
                if value.property_iri in self._label_property_set and value.is_literal:
                    labels[iri].append(value)
                elif value.is_literal:
                    attributes[iri].append(value)
        for data_assertion in parsed.data_property_assertions:
            attributes[data_assertion.subject_iri].append(data_assertion.value)
        self._labels = {
            iri: tuple(
                value.value
                for value in sorted(
                    set(values),
                    key=lambda value: (
                        value.value,
                        value.lang or "",
                    ),
                )
            )
            for iri, values in labels.items()
        }
        self._attributes = {
            iri: tuple(
                sorted(
                    set(values),
                    key=lambda value: (
                        value.property_iri,
                        value.value,
                        value.lang or "",
                        value.datatype or "",
                    ),
                )
            )
            for iri, values in attributes.items()
        }

        class_edges: set[tuple[str, str]] = set()
        restriction_expressions: dict[str, list] = defaultdict(list)
        for subclass_axiom in parsed.subclass_axioms:
            child = named_class_iri(subclass_axiom.sub)
            parent = named_class_iri(subclass_axiom.sup)
            if child is None:
                continue
            restriction_expressions[child].append(subclass_axiom.sup)
            if parent is not None:
                class_edges.add((child, parent))

        for equivalent_axiom in parsed.equivalent_class_axioms:
            anchors = [
                expr.iri for expr in equivalent_axiom.expressions if isinstance(expr, NamedClass)
            ]
            for anchor in anchors:
                for other in equivalent_axiom.expressions:
                    if isinstance(other, NamedClass):
                        if other.iri != anchor:
                            class_edges.add((anchor, other.iri))
                        continue
                    restriction_expressions[anchor].append(other)
                    if isinstance(other, ObjectUnionOf):
                        # A ≡ (B ⊔ C) entails B ⊑ A and C ⊑ A. ELK exposed
                        # these as direct parents in the historical baseline.
                        for operand in other.operands:
                            child = named_class_iri(operand)
                            if child is not None:
                                class_edges.add((child, anchor))
                        continue
                    for operand in intersection_operands(other):
                        parent = named_class_iri(operand)
                        if parent is not None:
                            class_edges.add((anchor, parent))

        self.hierarchy = HierarchyIndex(
            parsed.signature[EntityKind.CLASS], class_edges, filter_owl_bounds=True
        )
        self._restriction_expressions = {
            iri: tuple(dict.fromkeys(expressions))
            for iri, expressions in restriction_expressions.items()
        }

        property_edges: dict[EntityKind, set[tuple[str, str]]] = {
            EntityKind.OBJECT_PROPERTY: set(),
            EntityKind.DATA_PROPERTY: set(),
            EntityKind.ANNOTATION_PROPERTY: set(),
        }
        for subproperty_axiom in parsed.subproperty_axioms:
            property_edges[subproperty_axiom.kind].add(
                (
                    subproperty_axiom.sub_property_iri,
                    subproperty_axiom.super_property_iri,
                )
            )
        self._property_hierarchies = {
            kind: HierarchyIndex(parsed.signature[kind], edges, filter_owl_bounds=False)
            for kind, edges in property_edges.items()
        }

        individual_parents: dict[str, list[str]] = defaultdict(list)
        class_individuals: dict[str, list[str]] = defaultdict(list)
        for class_assertion in parsed.class_assertions:
            for class_iri in named_class_iris(class_assertion.class_expression):
                individual_parents[class_assertion.individual_iri].append(class_iri)
                class_individuals[class_iri].append(class_assertion.individual_iri)
        self._individual_parents = {
            iri: tuple(sorted(set(values))) for iri, values in individual_parents.items()
        }
        self._class_individuals = {
            iri: tuple(sorted(set(values))) for iri, values in class_individuals.items()
        }

        domains: dict[str, list[str]] = defaultdict(list)
        ranges: dict[str, list[str]] = defaultdict(list)
        for domain_axiom in parsed.property_domains:
            named = named_class_iri(domain_axiom.domain)
            domains[domain_axiom.property_iri].append(
                named if named is not None else render_class_expression(domain_axiom.domain)
            )
        for range_axiom in parsed.property_ranges:
            named = named_class_iri(range_axiom.range)
            ranges[range_axiom.property_iri].append(
                named if named is not None else render_class_expression(range_axiom.range)
            )
        self._domains = {iri: tuple(sorted(set(values))) for iri, values in domains.items()}
        self._ranges = {iri: tuple(sorted(set(values))) for iri, values in ranges.items()}

        excluded: set[str] = set()
        for iri, values in self._annotations.items():
            for value in values:
                lexical = value.value.strip().lower()
                if value.property_iri == ANNOTATION_IRI and lexical in {"false", "0"}:
                    excluded.add(iri)
                elif value.property_iri == OWL_DEPRECATED and lexical in {"true", "1"}:
                    excluded.add(iri)
        self._excluded = frozenset(excluded)

    @classmethod
    def from_path(
        cls,
        path: Path,
        *,
        label_properties: Sequence[str] | None = None,
    ) -> "OwlOntologySource":
        return cls(parse(Path(path)), label_properties=label_properties)

    @property
    def origin(self) -> Path | None:
        return self._origin

    def entities(self, kind: EntityKind = EntityKind.CLASS) -> tuple[str, ...]:
        try:
            normalized_kind = EntityKind(kind)
        except ValueError as exc:
            raise ValueError(f"Unknown entity kind: {kind!r}") from exc
        return self.parsed.signature[normalized_kind]

    def labels(self, iri: str) -> list[str]:
        return list(self._labels.get(str(iri), ()))

    def annotations(
        self, iri: str, properties: Sequence[str] | None = None
    ) -> list[AnnotationValue]:
        values = self._annotations.get(str(iri), ())
        if properties is None:
            return list(values)
        selected = frozenset(str(prop) for prop in properties)
        return [value for value in values if value.property_iri in selected]

    def attributes(self, iri: str) -> list[AnnotationValue]:
        return list(self._attributes.get(str(iri), ()))

    def direct_parents(self, iri: str, kind: EntityKind = EntityKind.CLASS) -> list[str]:
        normalized_kind = EntityKind(kind)
        if normalized_kind == EntityKind.CLASS:
            return self.hierarchy.direct_parents(str(iri))
        if normalized_kind == EntityKind.INDIVIDUAL:
            return list(self._individual_parents.get(str(iri), ()))
        return self._property_hierarchies[normalized_kind].direct_parents(str(iri))

    def direct_children(self, iri: str, kind: EntityKind = EntityKind.CLASS) -> list[str]:
        normalized_kind = EntityKind(kind)
        if normalized_kind == EntityKind.CLASS:
            return self.hierarchy.direct_children(str(iri))
        if normalized_kind == EntityKind.INDIVIDUAL:
            return list(self._class_individuals.get(str(iri), ()))
        return self._property_hierarchies[normalized_kind].direct_children(str(iri))

    def hierarchy_bundle(
        self, iri: str, families: Mapping[str, Sequence[str]]
    ) -> dict[str, list[str]]:
        iri = str(iri)
        bundle: dict[str, list[str]] = {}
        expressions = self._restriction_expressions.get(iri, ())
        for family, property_iris in families.items():
            if family == "is_a":
                bundle[family] = self.direct_parents(iri)
                continue
            values: list[str] = []
            for expression in expressions:
                values.extend(existential_targets(expression, property_iris))
            bundle[family] = list(dict.fromkeys(values))
        return bundle

    def projection_edges(
        self, *, method: str = "owl2vecstar", include_literals: bool = False
    ) -> list[Edge]:
        key = (str(method), bool(include_literals))
        with self._projection_lock:
            cached = self._projection_cache.get(key)
            if cached is None:
                cached = tuple(project(self.parsed, method, include_literals))
                self._projection_cache[key] = cached
        return list(cached)

    def property_domains(self, prop_iri: str) -> list[str]:
        return list(self._domains.get(str(prop_iri), ()))

    def property_ranges(self, prop_iri: str) -> list[str]:
        return list(self._ranges.get(str(prop_iri), ()))

    def excluded_from_alignment(self) -> frozenset[str]:
        return self._excluded

    def short_form(self, iri: str) -> str:
        text = str(iri)
        parsed = urlsplit(text)
        if parsed.fragment:
            return unquote(parsed.fragment)
        path = parsed.path.rstrip("/")
        if path:
            return unquote(path.rsplit("/", 1)[-1])
        if ":" in text:
            return text.rsplit(":", 1)[-1]
        return text
