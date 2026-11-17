"""Shared-snapshot implementation of Exact's :class:`KnowledgeSource` facade."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import cached_property
from os import PathLike
from pathlib import Path
from typing import BinaryIO
from urllib.parse import unquote, urlsplit

import pyowl_core
from pyowl2vec_star_projector import REFERENCE_PROFILE, Projector
from pyowl_core import (
    IRI,
    RDF_PLAIN_LITERAL_IRI,
    XSD_STRING_IRI,
    AnnotationAssertionIndex,
    AnnotationProperty,
    AssertedPropertyHierarchyView,
    AxiomTypeIndex,
    Class,
    ClassAssertion,
    ClassExpression,
    DataProperty,
    DataPropertyAssertion,
    Datatype,
    Entity,
)
from pyowl_core import EntityKind as CoreEntityKind
from pyowl_core import (
    InvalidIRIError,
    Literal,
    NamedIndividual,
    ObjectProperty,
    ObjectSomeValuesFrom,
    OntologySnapshot,
    OntologyView,
    PropertyDomainRangeView,
    SubAnnotationPropertyOf,
    walk,
)
from pyowl_core.index import PropertyComponent

from exact.core.contracts.knowledge import KnowledgeSource
from exact.core.entities.graph import AnnotationValue, Edge
from exact.core.entities.kinds import EntityKind
from exact.core.values import ANNOTATION_IRI
from exact.ontology.native_projection import require_native_support
from exact.ontology.projection import ProjectorSettings, SharedProjectionAdapter
from exact.ontology.view_contract import native_load_options, retain_ontology_view

RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
OWL_DEPRECATED = "http://www.w3.org/2002/07/owl#deprecated"
OWL_THING = "http://www.w3.org/2002/07/owl#Thing"
OWL_NOTHING = "http://www.w3.org/2002/07/owl#Nothing"

_CORE_KINDS = {
    EntityKind.CLASS: CoreEntityKind.CLASS,
    EntityKind.OBJECT_PROPERTY: CoreEntityKind.OBJECT_PROPERTY,
    EntityKind.DATA_PROPERTY: CoreEntityKind.DATA_PROPERTY,
    EntityKind.ANNOTATION_PROPERTY: CoreEntityKind.ANNOTATION_PROPERTY,
    EntityKind.INDIVIDUAL: CoreEntityKind.NAMED_INDIVIDUAL,
}


def _named_classes(value: object) -> tuple[str, ...]:
    if not isinstance(value, pyowl_core.StructuralNode):
        return ()
    return tuple(dict.fromkeys(node.iri.value for node in walk(value) if isinstance(node, Class)))


def _object_property_iri(value: object) -> str | None:
    return value.iri.value if isinstance(value, ObjectProperty) else None


def _annotation_value(property_iri: str, value: object) -> AnnotationValue | None:
    if isinstance(value, Literal):
        datatype: str | None = value.datatype.iri.value
        if value.language is not None or datatype in {
            RDF_PLAIN_LITERAL_IRI,
            XSD_STRING_IRI,
        }:
            datatype = None
        return AnnotationValue(
            property_iri,
            value.lexical_form,
            True,
            lang=value.language,
            datatype=datatype,
        )
    if isinstance(value, IRI):
        return AnnotationValue(property_iri, value.value, False)
    return None


class _ClassHierarchy:
    """IRI wrappers around native structural feature queries."""

    def __init__(self, view: pyowl_core.ClassFeatureView) -> None:
        self._view = view

    def _query(self, method: str, iri: str) -> list[str]:
        try:
            value = Class(IRI(str(iri)))
        except InvalidIRIError:
            return []
        return [item.iri.value for item in getattr(self._view, method)(value)]

    def direct_parents(self, iri: str) -> list[str]:
        return self._query("direct_parents", iri)

    def direct_children(self, iri: str) -> list[str]:
        return self._query("direct_children", iri)

    def ancestors(self, iri: str) -> set[str]:
        return set(self._query("ancestors", iri))

    def descendants(self, iri: str) -> set[str]:
        return set(self._query("descendants", iri))


class _PropertyHierarchy:
    """Requested IRI rows over native component reduction."""

    def __init__(self, view: AssertedPropertyHierarchyView) -> None:
        self._view = view

    def _query(self, method: str, iri: str) -> list[str]:
        try:
            value = IRI(str(iri))
        except InvalidIRIError:
            return []
        result: set[str] = set()
        # Exact's IRI presentation accepts either named property kind. The source
        # rejects ambiguous object/data punning before constructing this adapter.
        for constructor in (ObjectProperty, DataProperty):
            for node in getattr(self._view, method)(constructor(value)):
                members = node.members if isinstance(node, PropertyComponent) else (node,)
                result.update(member.iri.value for member in members)
        return sorted(result)

    def direct_parents(self, iri: str) -> list[str]:
        return self._query("direct_parents", iri)

    def direct_children(self, iri: str) -> list[str]:
        return self._query("direct_children", iri)


class OwlOntologySource(KnowledgeSource):
    """Read-only Exact facade owning one shared ontology view by identity."""

    def __init__(
        self,
        snapshot: OntologyView,
        *,
        label_properties: Sequence[str] | None = None,
        origin: Path | None = None,
        projector_backend: str = "native",
        projector_profile: str = REFERENCE_PROFILE,
    ) -> None:
        snapshot = retain_ontology_view(snapshot)
        self._snapshot = snapshot
        self._origin = Path(origin) if origin is not None else None
        root_snapshot: OntologyView = snapshot
        while isinstance(root_snapshot, pyowl_core.OntologyOverlay):
            root_snapshot = root_snapshot.base
        ontology_iri = (
            root_snapshot.document(root_snapshot.root_document_key).ontology_id.ontology_iri
            if isinstance(root_snapshot, OntologySnapshot)
            else None
        )
        self.ontology_iri = None if ontology_iri is None else ontology_iri.value
        self.label_properties = tuple(
            (RDFS_LABEL,) if label_properties is None else map(str, label_properties)
        )
        self._label_property_set = frozenset(self.label_properties)

        # Shared-core and Exact feature indexes are intentionally lazy. Large projection-only
        # or coherence runs must not pay for annotations, class/property hierarchy, ABox, and
        # domain/range indexes merely by constructing the source facade.
        self._entity_cache: dict[EntityKind, tuple[str, ...]] = {}
        self._projection = SharedProjectionAdapter(
            snapshot,
            ProjectorSettings.from_value(
                {"backend": projector_backend, "profile": projector_profile}
            ),
        )
        # The default remains the structural view and therefore imports no optional
        # reasoner.  Explicit dataset selection installs one narrow adapter lazily.
        self._reasoner: object | None = None

    @classmethod
    def load(
        cls,
        source: str | PathLike[str] | bytes | bytearray | memoryview | BinaryIO,
        *,
        options: pyowl_core.LoadOptions | None = None,
        resolver: pyowl_core.ImportResolver | None = None,
        document_iri: pyowl_core.IRI | str | None = None,
        label_properties: Sequence[str] | None = None,
        projector_backend: str = "native",
        projector_profile: str = REFERENCE_PROFILE,
    ) -> "OwlOntologySource":
        """Load one closure exactly once and retain the resulting snapshot."""

        if document_iri is None and not isinstance(
            source, (str, PathLike, bytes, bytearray, memoryview)
        ):
            document_iri = "urn:exact-om:stream-root"
        selected_options = native_load_options(options)
        require_native_support()
        snapshot = pyowl_core.load_snapshot(
            source,
            document_iri=document_iri,
            options=selected_options,
            resolver=resolver,
        )
        origin: Path | None = None
        if isinstance(source, (str, PathLike)):
            candidate = Path(source)
            if candidate.exists():
                origin = candidate
        return cls(
            snapshot,
            label_properties=label_properties,
            origin=origin,
            projector_backend=projector_backend,
            projector_profile=projector_profile,
        )

    @classmethod
    def from_path(
        cls,
        path: Path,
        *,
        label_properties: Sequence[str] | None = None,
    ) -> "OwlOntologySource":
        """Compatibility spelling for :meth:`load`."""

        return cls.load(Path(path), label_properties=label_properties)

    @cached_property
    def _signature(self) -> tuple[Entity, ...]:
        # Enumeration needs no reference-count index; retain the public typed closure.
        return self._snapshot.signature(include_builtins=True)

    @cached_property
    def _annotation_index(self) -> AnnotationAssertionIndex:
        return self._snapshot.view(
            AnnotationAssertionIndex,
            include_origins=False,
            include_nested=False,
            require_native_pipeline=True,
        )

    @cached_property
    def _axioms(self) -> AxiomTypeIndex:
        return self._snapshot.view(
            AxiomTypeIndex, include_origins=False, require_native_pipeline=True
        )

    @cached_property
    def _class_view(self) -> pyowl_core.ClassFeatureView:
        return self._snapshot.view(
            pyowl_core.ClassFeatureView,
            equivalent_operands=True,
            include_builtins=False,
            require_native_pipeline=True,
        )

    @cached_property
    def _property_view(self) -> AssertedPropertyHierarchyView:
        return self._snapshot.view(
            AssertedPropertyHierarchyView,
            include_origins=False,
            equivalence_handling="component",
            require_native_pipeline=True,
        )

    @cached_property
    def _domain_range(self) -> PropertyDomainRangeView:
        return self._snapshot.view(
            PropertyDomainRangeView, include_origins=False, require_native_pipeline=True
        )

    @cached_property
    def hierarchy(self) -> _ClassHierarchy:
        """Return the lazily constructed native structural hierarchy adapter."""
        return _ClassHierarchy(self._class_view)

    @cached_property
    def _property_hierarchy(self) -> _PropertyHierarchy:
        if self._axioms.native_report["has_object_data_property_punning"]:
            raise pyowl_core.BackendProtocolError(
                "Exact native property features cannot preserve ambiguous object/data IRI punning",
                code="NATIVE_VIEW_REQUIRED",
            )
        return _PropertyHierarchy(self._property_view)

    def _individual_parents(self, iri: str) -> list[str]:
        try:
            individual = NamedIndividual(IRI(iri))
        except InvalidIRIError:
            return []
        return sorted(
            {
                value
                for row in self._axioms.iter(ClassAssertion, referencing=individual)
                if row.individual == individual
                for value in _named_classes(row.class_expression)
            }
        )

    def _class_individuals(self, iri: str) -> list[str]:
        try:
            entity = Class(IRI(iri))
        except InvalidIRIError:
            return []
        return sorted(
            {
                row.individual.iri.value
                for row in self._axioms.iter(ClassAssertion, referencing=entity)
                if isinstance(row.individual, NamedIndividual)
                and iri in _named_classes(row.class_expression)
            }
        )

    def _data_values(self, iri: str) -> tuple[AnnotationValue, ...]:
        try:
            individual = NamedIndividual(IRI(iri))
        except InvalidIRIError:
            return ()
        return tuple(
            converted
            for row in self._axioms.iter(DataPropertyAssertion, referencing=individual)
            if row.source == individual
            and (converted := _annotation_value(row.property.iri.value, row.value)) is not None
        )

    def _annotation_property_relations(self, iri: str, *, upward: bool) -> list[str]:
        try:
            entity = AnnotationProperty(IRI(iri))
        except InvalidIRIError:
            return []
        return sorted(
            {
                (row.super_property if upward else row.sub_property).iri.value
                for row in self._axioms.iter(SubAnnotationPropertyOf, referencing=entity)
                if (row.sub_property if upward else row.super_property) == entity
            }
        )

    @cached_property
    def _excluded(self) -> frozenset[str]:
        return self._build_exclusions()

    def owl_snapshot(self) -> OntologyView:
        """Return the exact shared view instance; never rebuild or reparse."""

        return self._snapshot

    @property
    def origin(self) -> Path | None:
        return self._origin

    @property
    def projector(self) -> Projector:
        """Expose the projector's identity diagnostic without transferring ownership."""

        return self._projection.projector

    @property
    def projector_settings(self) -> ProjectorSettings:
        return self._projection.settings

    def configure_projector(
        self,
        *,
        backend: str = "native",
        profile: str = REFERENCE_PROFILE,
    ) -> None:
        """Select semantics before use while retaining the exact snapshot identity."""

        self._projection = SharedProjectionAdapter(
            self.owl_snapshot(),
            ProjectorSettings.from_value({"backend": backend, "profile": profile}),
        )

    @property
    def reasoner(self) -> object:
        """Return the selected narrow hierarchy adapter, creating asserted lazily."""

        if self._reasoner is None:
            from exact.ontology.reasoning import AssertedHierarchyReasoner

            self._reasoner = AssertedHierarchyReasoner(self)
        return self._reasoner

    @property
    def reasoner_provenance(self) -> dict[str, object]:
        """Return path-free core/reasoner identity for run-manifest consumers."""

        return dict(getattr(self.reasoner, "provenance"))

    def ontology_stack_provenance(self) -> dict[str, object]:
        """Return path-free provenance for the single shared snapshot and consumers."""

        from exact.ontology.provenance import ontology_stack_provenance

        return ontology_stack_provenance(
            self.owl_snapshot(),
            projector_settings=self.projector_settings,
            projector=self.projector,
            reasoner=self.reasoner_provenance,
        )

    def configure_reasoner(self, name: str = "asserted", **settings: object) -> None:
        """Select one explicit hierarchy reasoner over this exact snapshot."""

        from exact.ontology.reasoning import load_reasoner

        previous = self._reasoner
        selected = load_reasoner(name, self, settings=settings or None)
        self._reasoner = selected
        if previous is not None and previous is not selected:
            close = getattr(previous, "close", None)
            if callable(close):
                close()

    def entities(self, kind: EntityKind = EntityKind.CLASS) -> tuple[str, ...]:
        try:
            normalized_kind = EntityKind(kind)
        except ValueError as exc:
            raise ValueError(f"Unknown entity kind: {kind!r}") from exc
        cached = self._entity_cache.get(normalized_kind)
        if cached is None:
            core_kind = _CORE_KINDS[normalized_kind]
            cached = tuple(
                sorted(entity.iri.value for entity in self._signature if entity.kind is core_kind)
            )
            self._entity_cache[normalized_kind] = cached
        return cached

    def _annotation_rows(
        self, iri: str, properties: tuple[str, ...] | None = None
    ) -> tuple[AnnotationValue, ...]:
        try:
            subject = IRI(iri)
        except InvalidIRIError:
            return ()
        selected = None
        if properties is not None:
            selected = []
            for name in properties:
                try:
                    selected.append(AnnotationProperty(IRI(name)))
                except InvalidIRIError:
                    continue
        values: set[AnnotationValue] = set()
        for page in self._annotation_index.iter_columns(subjects=(subject,), properties=selected):
            for prop, value in zip(page.properties, page.values):
                converted = _annotation_value(prop.iri.value, value)
                if converted is not None:
                    values.add(converted)
        return tuple(
            sorted(
                values,
                key=lambda value: (
                    value.property_iri,
                    value.value,
                    value.lang or "",
                    value.datatype or "",
                    value.is_literal,
                ),
            )
        )

    def labels(self, iri: str) -> list[str]:
        selected = {
            value
            for value in self._annotation_rows(str(iri), self.label_properties)
            if value.is_literal and value.property_iri in self._label_property_set
        }
        return [
            value.value
            for value in sorted(selected, key=lambda item: (item.value, item.lang or ""))
        ]

    def annotations(
        self, iri: str, properties: Sequence[str] | None = None
    ) -> list[AnnotationValue]:
        selected = None if properties is None else tuple(sorted(set(map(str, properties))))
        return list(self._annotation_rows(str(iri), selected))

    def attributes(self, iri: str) -> list[AnnotationValue]:
        values = {
            value
            for value in self._annotation_rows(str(iri))
            if value.is_literal and value.property_iri not in self._label_property_set
        }
        values.update(self._data_values(str(iri)))
        return sorted(
            values,
            key=lambda value: (
                value.property_iri,
                value.value,
                value.lang or "",
                value.datatype or "",
            ),
        )

    def direct_parents(self, iri: str, kind: EntityKind = EntityKind.CLASS) -> list[str]:
        normalized_kind = EntityKind(kind)
        if normalized_kind is EntityKind.CLASS:
            if self._reasoner is not None:
                return list(getattr(self._reasoner, "direct_parents")(str(iri)))
            return self.hierarchy.direct_parents(str(iri))
        if normalized_kind is EntityKind.INDIVIDUAL:
            return self._individual_parents(str(iri))
        if normalized_kind is EntityKind.ANNOTATION_PROPERTY:
            return self._annotation_property_relations(str(iri), upward=True)
        return self._property_hierarchy.direct_parents(str(iri))

    def direct_children(self, iri: str, kind: EntityKind = EntityKind.CLASS) -> list[str]:
        normalized_kind = EntityKind(kind)
        if normalized_kind is EntityKind.CLASS:
            if self._reasoner is not None:
                return list(getattr(self._reasoner, "direct_children")(str(iri)))
            return self.hierarchy.direct_children(str(iri))
        if normalized_kind is EntityKind.INDIVIDUAL:
            return self._class_individuals(str(iri))
        if normalized_kind is EntityKind.ANNOTATION_PROPERTY:
            return self._annotation_property_relations(str(iri), upward=False)
        return self._property_hierarchy.direct_children(str(iri))

    def hierarchy_bundle(
        self, iri: str, families: Mapping[str, Sequence[str]]
    ) -> dict[str, list[str]]:
        selected_iri = str(iri)
        result: dict[str, list[str]] = {}
        expressions: tuple[ClassExpression, ...]
        try:
            expression_owner = Class(IRI(selected_iri))
        except InvalidIRIError:
            expressions = ()
        else:
            expressions = (
                tuple(self._class_view.restrictions(expression_owner))
                if any(family != "is_a" for family in families)
                else ()
            )
        for family, property_iris in families.items():
            if family == "is_a":
                result[family] = self.direct_parents(selected_iri)
                continue
            properties = frozenset(map(str, property_iris))
            targets: list[str] = []
            for expression in expressions:
                for node in walk(expression):
                    if not isinstance(node, ObjectSomeValuesFrom):
                        continue
                    property_iri = _object_property_iri(node.property)
                    if property_iri in properties:
                        targets.extend(_named_classes(node.filler))
            result[family] = list(dict.fromkeys(targets))
        return result

    def projection_edges(
        self, *, method: str = "owl2vecstar", include_literals: bool = False
    ) -> list[Edge]:
        return self._projection.edges(
            method=method,
            include_literals=include_literals,
        )

    def _property(self, iri: str) -> ObjectProperty | DataProperty | AnnotationProperty:
        if iri in self.entities(EntityKind.DATA_PROPERTY):
            return DataProperty(IRI(iri))
        if iri in self.entities(EntityKind.ANNOTATION_PROPERTY):
            return AnnotationProperty(IRI(iri))
        return ObjectProperty(IRI(iri))

    @staticmethod
    def _domain_range_value(value: object) -> str:
        if isinstance(value, IRI):
            return str(value.value)
        if isinstance(value, (Class, Datatype)):
            return str(value.iri.value)
        # Complex results are rare in Exact's schema channel.  The core value is
        # retained canonically; no local structural model is constructed.
        if isinstance(value, pyowl_core.StructuralNode):
            return f"{type(value).__name__}({pyowl_core.structural_hexdigest(value)})"
        return str(value)

    def property_domains(self, prop_iri: str) -> list[str]:
        try:
            prop = self._property(str(prop_iri))
        except InvalidIRIError:
            return []
        return sorted(
            {self._domain_range_value(record.value) for record in self._domain_range.domains(prop)}
        )

    def property_ranges(self, prop_iri: str) -> list[str]:
        try:
            prop = self._property(str(prop_iri))
        except InvalidIRIError:
            return []
        return sorted(
            {self._domain_range_value(record.value) for record in self._domain_range.ranges(prop)}
        )

    def _build_exclusions(self) -> frozenset[str]:
        excluded: set[str] = set()
        properties = tuple(AnnotationProperty(IRI(iri)) for iri in (ANNOTATION_IRI, OWL_DEPRECATED))
        for page in self._annotation_index.iter_columns(properties=properties):
            for subject, prop, value in zip(page.subjects, page.properties, page.values):
                if not isinstance(subject, IRI) or not isinstance(value, Literal):
                    continue
                lexical = value.lexical_form.strip().lower()
                if prop.iri.value == ANNOTATION_IRI and lexical in {"false", "0"}:
                    excluded.add(subject.value)
                elif prop.iri.value == OWL_DEPRECATED and lexical in {"true", "1"}:
                    excluded.add(subject.value)
        return frozenset(excluded)

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
