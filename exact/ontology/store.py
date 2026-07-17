"""Shared-snapshot implementation of Exact's :class:`KnowledgeSource` facade."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache
from os import PathLike
from pathlib import Path
from typing import BinaryIO, TypeAlias
from urllib.parse import unquote, urlsplit

import pyowl_core
from pyowl_core import (
    AnnotationAssertionIndex,
    AnnotationProperty,
    AssertedClassHierarchyView,
    AssertedPropertyHierarchyView,
    AxiomTypeIndex,
    Class,
    ClassAssertion,
    DataProperty,
    DataPropertyAssertion,
    Datatype,
    Entity,
    EquivalentClasses,
    IRI,
    InvalidIRIError,
    Literal,
    NamedIndividual,
    ObjectIntersectionOf,
    ObjectProperty,
    ObjectSomeValuesFrom,
    ObjectUnionOf,
    OntologySnapshot,
    PropertyDomainRangeView,
    SignatureView,
    SubAnnotationPropertyOf,
    SubClassOf,
    walk,
)
from pyowl_core import EntityKind as CoreEntityKind
from pyowl_core import RDF_PLAIN_LITERAL_IRI, XSD_STRING_IRI
from pyowl_core.index import ClassComponent, PropertyComponent
from pyowl2vec_star_projector import Projector, REFERENCE_PROFILE

from exact.core.contracts.knowledge import KnowledgeSource
from exact.core.entities.graph import AnnotationValue, Edge
from exact.core.entities.kinds import EntityKind
from exact.core.values import ANNOTATION_IRI
from exact.ontology.projection import ProjectorSettings, SharedProjectionAdapter

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
_OWL_BOUNDS = frozenset({OWL_THING, OWL_NOTHING})
_ClassNode: TypeAlias = Class | ClassComponent
_PropertyNode: TypeAlias = ObjectProperty | DataProperty | PropertyComponent


def _named_classes(value: object) -> tuple[str, ...]:
    if not isinstance(value, pyowl_core.StructuralNode):
        return ()
    return tuple(
        dict.fromkeys(node.iri.value for node in walk(value) if isinstance(node, Class))
    )


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
    """Exact directness semantics over the shared asserted class view.

    The core deliberately exposes asserted endpoints, not transitive reduction.  This
    adapter performs reduction only for queried rows and keeps no second edge graph.
    """

    def __init__(
        self,
        view: AssertedClassHierarchyView,
        components: Mapping[str, tuple[str, ...]],
        extra_parents: Mapping[str, tuple[str, ...]],
    ) -> None:
        self._view = view
        self._components = dict(components)
        self._extra_parents = dict(extra_parents)
        children: dict[str, set[str]] = defaultdict(set)
        for child, parents in extra_parents.items():
            for parent in parents:
                children[parent].add(child)
        self._extra_children = {
            parent: tuple(sorted(values)) for parent, values in children.items()
        }

    def _component(self, iri: str) -> tuple[str, ...]:
        return self._components.get(str(iri), (str(iri),))

    def _raw_parents(self, component: tuple[str, ...]) -> set[tuple[str, ...]]:
        parents: set[tuple[str, ...]] = set()
        for member in component:
            try:
                entity = Class(IRI(member))
            except InvalidIRIError:
                continue
            parents.update(
                self._component(parent.iri.value)
                for parent in self._view.asserted_parents(entity)
                if isinstance(parent, Class)
            )
            parents.update(
                self._component(parent) for parent in self._extra_parents.get(member, ())
            )
        parents.discard(component)
        return parents

    def _raw_children(self, component: tuple[str, ...]) -> set[tuple[str, ...]]:
        children: set[tuple[str, ...]] = set()
        for member in component:
            try:
                entity = Class(IRI(member))
            except InvalidIRIError:
                continue
            children.update(
                self._component(child.iri.value)
                for child in self._view.asserted_children(entity)
                if isinstance(child, Class)
            )
            children.update(
                self._component(child) for child in self._extra_children.get(member, ())
            )
        children.discard(component)
        return children

    def _is_ancestor(
        self, descendant: tuple[str, ...], ancestor: tuple[str, ...]
    ) -> bool:
        seen: set[tuple[str, ...]] = set()
        stack = list(self._raw_parents(descendant))
        while stack:
            current = stack.pop()
            if current == ancestor:
                return True
            if current in seen:
                continue
            seen.add(current)
            stack.extend(self._raw_parents(current) - seen)
        return False

    @lru_cache(maxsize=None)
    def _direct_parent_components(self, iri: str) -> tuple[tuple[str, ...], ...]:
        component = self._component(iri)
        candidates = self._raw_parents(component)
        return tuple(
            sorted(
                (
                    parent
                    for parent in candidates
                    if not any(
                        parent != other and self._is_ancestor(other, parent)
                        for other in candidates
                    )
                ),
                key=lambda values: tuple(item.encode("utf-8") for item in values),
            )
        )

    def direct_parents(self, iri: str) -> list[str]:
        return sorted(
            member
            for component in self._direct_parent_components(str(iri))
            for member in component
            if member not in _OWL_BOUNDS
        )

    @lru_cache(maxsize=None)
    def _direct_child_components(self, iri: str) -> tuple[tuple[str, ...], ...]:
        component = self._component(iri)
        return tuple(
            sorted(
                (
                    child
                    for child in self._raw_children(component)
                    if component in self._direct_parent_components(child[0])
                ),
                key=lambda values: tuple(item.encode("utf-8") for item in values),
            )
        )

    def direct_children(self, iri: str) -> list[str]:
        return sorted(
            member
            for component in self._direct_child_components(str(iri))
            for member in component
            if member not in _OWL_BOUNDS
        )

    def ancestors(self, iri: str) -> set[str]:
        seen: set[str] = set()
        stack = self.direct_parents(iri)
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(self.direct_parents(current))
        return seen

    def descendants(self, iri: str) -> set[str]:
        seen: set[str] = set()
        stack = self.direct_children(iri)
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(self.direct_children(current))
        return seen

    def equivalent_entities(self, iri: str) -> frozenset[str]:
        return frozenset(member for member in self._component(iri) if member != iri)


class _PropertyHierarchy:
    """Directness and equivalence normalization over a shared property view."""

    def __init__(
        self,
        view: AssertedPropertyHierarchyView,
        components: Mapping[str, tuple[str, ...]],
        constructors: Mapping[str, type[ObjectProperty] | type[DataProperty]],
    ) -> None:
        self._view = view
        self._components = dict(components)
        self._constructors = dict(constructors)

    def _component(self, iri: str) -> tuple[str, ...]:
        return self._components.get(str(iri), (str(iri),))

    def _entity(self, iri: str) -> ObjectProperty | DataProperty | None:
        constructor = self._constructors.get(iri)
        if constructor is None:
            return None
        try:
            return constructor(IRI(iri))
        except InvalidIRIError:
            return None

    def _raw_parents(self, component: tuple[str, ...]) -> set[tuple[str, ...]]:
        parents: set[tuple[str, ...]] = set()
        for member in component:
            entity = self._entity(member)
            if entity is None:
                continue
            parents.update(
                self._component(parent.iri.value)
                for parent in self._view.asserted_parents(entity)
                if isinstance(parent, (ObjectProperty, DataProperty))
            )
        parents.discard(component)
        return parents

    def _raw_children(self, component: tuple[str, ...]) -> set[tuple[str, ...]]:
        children: set[tuple[str, ...]] = set()
        for member in component:
            entity = self._entity(member)
            if entity is None:
                continue
            children.update(
                self._component(child.iri.value)
                for child in self._view.asserted_children(entity)
                if isinstance(child, (ObjectProperty, DataProperty))
            )
        children.discard(component)
        return children

    def _is_ancestor(
        self, descendant: tuple[str, ...], ancestor: tuple[str, ...]
    ) -> bool:
        seen: set[tuple[str, ...]] = set()
        stack = list(self._raw_parents(descendant))
        while stack:
            current = stack.pop()
            if current == ancestor:
                return True
            if current in seen:
                continue
            seen.add(current)
            stack.extend(self._raw_parents(current) - seen)
        return False

    @lru_cache(maxsize=None)
    def _direct_parents(self, iri: str) -> tuple[tuple[str, ...], ...]:
        candidates = self._raw_parents(self._component(iri))
        return tuple(
            sorted(
                parent
                for parent in candidates
                if not any(
                    parent != other and self._is_ancestor(other, parent)
                    for other in candidates
                )
            )
        )

    def direct_parents(self, iri: str) -> list[str]:
        return sorted(member for node in self._direct_parents(str(iri)) for member in node)

    def direct_children(self, iri: str) -> list[str]:
        component = self._component(str(iri))
        return sorted(
            member
            for child in self._raw_children(component)
            if component in self._direct_parents(child[0])
            for member in child
        )


def _equivalence_components(
    groups: Iterable[Iterable[Entity]],
) -> dict[str, tuple[str, ...]]:
    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for group in groups:
        values = [item.iri.value for item in group]
        for value in values[1:]:
            union(values[0], value)
    members: dict[str, list[str]] = defaultdict(list)
    for value in parent:
        members[find(value)].append(value)
    result: dict[str, tuple[str, ...]] = {}
    for values in members.values():
        component = tuple(sorted(values))
        result.update((value, component) for value in component)
    return result


class OwlOntologySource(KnowledgeSource):
    """Read-only Exact facade owning one concrete shared ontology snapshot."""

    def __init__(
        self,
        snapshot: OntologySnapshot,
        *,
        label_properties: Sequence[str] | None = None,
        origin: Path | None = None,
        projector_backend: str = "auto",
        projector_profile: str = REFERENCE_PROFILE,
    ) -> None:
        if not isinstance(snapshot, OntologySnapshot):
            raise TypeError("snapshot must be a concrete pyowl_core.OntologySnapshot")
        self._snapshot = snapshot
        self._origin = Path(origin) if origin is not None else None
        root = snapshot.document(snapshot.root_document_key)
        ontology_iri = root.ontology_id.ontology_iri
        self.ontology_iri = None if ontology_iri is None else ontology_iri.value
        self.label_properties = tuple(
            (RDFS_LABEL,) if label_properties is None else map(str, label_properties)
        )
        self._label_property_set = frozenset(self.label_properties)

        # These are lazy shared-core indexes; the snapshot cache preserves their identity
        # for projector and future reasoner consumers.
        self._signature = snapshot.view(SignatureView, include_builtins=True)
        self._annotations = snapshot.view(AnnotationAssertionIndex)
        self._axioms = snapshot.view(AxiomTypeIndex)
        self._class_view = snapshot.view(AssertedClassHierarchyView)
        self._property_view = snapshot.view(AssertedPropertyHierarchyView)
        self._domain_range = snapshot.view(PropertyDomainRangeView)

        class_components = _equivalence_components(
            record.classes for record in self._class_view.equivalence_sets()
        )
        property_components = _equivalence_components(
            record.properties for record in self._property_view.equivalence_sets()
        )
        extra_parents: dict[str, set[str]] = defaultdict(set)
        restrictions: dict[str, list[pyowl_core.StructuralNode]] = defaultdict(list)
        for subclass_axiom in self._axioms.iter(SubClassOf):
            if isinstance(subclass_axiom.sub_class, Class):
                restrictions[subclass_axiom.sub_class.iri.value].append(
                    subclass_axiom.super_class
                )
        for equivalent_axiom in self._axioms.iter(EquivalentClasses):
            anchors = tuple(
                expression
                for expression in equivalent_axiom.expressions
                if isinstance(expression, Class)
            )
            for anchor in anchors:
                anchor_iri = anchor.iri.value
                for expression in equivalent_axiom.expressions:
                    if isinstance(expression, Class):
                        continue
                    restrictions[anchor_iri].append(expression)
                    if isinstance(expression, ObjectUnionOf):
                        for operand in expression.operands:
                            if isinstance(operand, Class):
                                extra_parents[operand.iri.value].add(anchor_iri)
                    elif isinstance(expression, ObjectIntersectionOf):
                        for operand in expression.operands:
                            if isinstance(operand, Class):
                                extra_parents[anchor_iri].add(operand.iri.value)
        self._restriction_expressions = {
            iri: tuple(dict.fromkeys(values)) for iri, values in restrictions.items()
        }
        self.hierarchy = _ClassHierarchy(
            self._class_view,
            class_components,
            {iri: tuple(sorted(values)) for iri, values in extra_parents.items()},
        )

        constructors = {
            entity.iri.value: (
                ObjectProperty if entity.kind is CoreEntityKind.OBJECT_PROPERTY else DataProperty
            )
            for entity in self._signature.iter()
            if entity.kind in {CoreEntityKind.OBJECT_PROPERTY, CoreEntityKind.DATA_PROPERTY}
        }
        self._property_hierarchy = _PropertyHierarchy(
            self._property_view, property_components, constructors
        )

        individual_parents: dict[str, set[str]] = defaultdict(set)
        class_individuals: dict[str, set[str]] = defaultdict(set)
        for class_assertion in self._axioms.iter(ClassAssertion):
            if not isinstance(class_assertion.individual, NamedIndividual):
                continue
            individual = class_assertion.individual.iri.value
            for class_iri in _named_classes(class_assertion.class_expression):
                individual_parents[individual].add(class_iri)
                class_individuals[class_iri].add(individual)
        self._individual_parents = {
            iri: tuple(sorted(values)) for iri, values in individual_parents.items()
        }
        self._class_individuals = {
            iri: tuple(sorted(values)) for iri, values in class_individuals.items()
        }

        data_values: dict[str, list[AnnotationValue]] = defaultdict(list)
        for data_assertion in self._axioms.iter(DataPropertyAssertion):
            if not isinstance(data_assertion.source, NamedIndividual):
                continue
            converted = _annotation_value(
                data_assertion.property.iri.value, data_assertion.value
            )
            if converted is not None:
                data_values[data_assertion.source.iri.value].append(converted)
        self._data_values = {
            iri: tuple(values) for iri, values in data_values.items()
        }

        self._annotation_property_parents: dict[str, tuple[str, ...]] = {}
        self._annotation_property_children: dict[str, tuple[str, ...]] = {}
        annotation_parents: dict[str, set[str]] = defaultdict(set)
        annotation_children: dict[str, set[str]] = defaultdict(set)
        for subproperty_axiom in self._axioms.iter(SubAnnotationPropertyOf):
            child = subproperty_axiom.sub_property.iri.value
            parent = subproperty_axiom.super_property.iri.value
            annotation_parents[child].add(parent)
            annotation_children[parent].add(child)
        self._annotation_property_parents = {
            iri: tuple(sorted(values)) for iri, values in annotation_parents.items()
        }
        self._annotation_property_children = {
            iri: tuple(sorted(values)) for iri, values in annotation_children.items()
        }

        self._excluded = self._build_exclusions()
        self._entity_cache: dict[EntityKind, tuple[str, ...]] = {}
        self._projection = SharedProjectionAdapter(
            snapshot,
            ProjectorSettings.from_value(
                {"backend": projector_backend, "profile": projector_profile}
            ),
        )

    @classmethod
    def load(
        cls,
        source: str | PathLike[str] | bytes | bytearray | memoryview | BinaryIO,
        *,
        options: pyowl_core.LoadOptions | None = None,
        resolver: pyowl_core.ImportResolver | None = None,
        document_iri: pyowl_core.IRI | str | None = None,
        label_properties: Sequence[str] | None = None,
        projector_backend: str = "auto",
        projector_profile: str = REFERENCE_PROFILE,
    ) -> "OwlOntologySource":
        """Load one closure exactly once and retain the resulting snapshot."""

        if (
            document_iri is None
            and not isinstance(source, (str, PathLike, bytes, bytearray, memoryview))
        ):
            document_iri = "urn:exact-om:stream-root"
        snapshot = pyowl_core.load_snapshot(
            source,
            document_iri=document_iri,
            options=options,
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

    def owl_snapshot(self) -> OntologySnapshot:
        """Return the exact shared snapshot instance; never rebuild or reparse."""

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
        backend: str = "auto",
        profile: str = REFERENCE_PROFILE,
    ) -> None:
        """Select semantics before use while retaining the exact snapshot identity."""

        self._projection = SharedProjectionAdapter(
            self.owl_snapshot(),
            ProjectorSettings.from_value({"backend": backend, "profile": profile}),
        )

    def entities(self, kind: EntityKind = EntityKind.CLASS) -> tuple[str, ...]:
        try:
            normalized_kind = EntityKind(kind)
        except ValueError as exc:
            raise ValueError(f"Unknown entity kind: {kind!r}") from exc
        cached = self._entity_cache.get(normalized_kind)
        if cached is None:
            core_kind = _CORE_KINDS[normalized_kind]
            cached = tuple(
                entity.iri.value
                for entity in self._signature.iter()
                if entity.kind is core_kind
            )
            self._entity_cache[normalized_kind] = cached
        return cached

    @lru_cache(maxsize=None)
    def _annotation_rows(self, iri: str) -> tuple[AnnotationValue, ...]:
        try:
            subject = IRI(iri)
        except InvalidIRIError:
            return ()
        values: set[AnnotationValue] = set()
        for posting in self._annotations.iter_subject(subject):
            assertion = posting.assertion
            converted = _annotation_value(assertion.property.iri.value, assertion.value)
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
            for value in self._annotation_rows(str(iri))
            if value.is_literal and value.property_iri in self._label_property_set
        }
        return [
            value.value
            for value in sorted(selected, key=lambda item: (item.value, item.lang or ""))
        ]

    def annotations(
        self, iri: str, properties: Sequence[str] | None = None
    ) -> list[AnnotationValue]:
        values = self._annotation_rows(str(iri))
        if properties is None:
            return list(values)
        selected = frozenset(map(str, properties))
        return [value for value in values if value.property_iri in selected]

    def attributes(self, iri: str) -> list[AnnotationValue]:
        values = {
            value
            for value in self._annotation_rows(str(iri))
            if value.is_literal and value.property_iri not in self._label_property_set
        }
        values.update(self._data_values.get(str(iri), ()))
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
            return self.hierarchy.direct_parents(str(iri))
        if normalized_kind is EntityKind.INDIVIDUAL:
            return list(self._individual_parents.get(str(iri), ()))
        if normalized_kind is EntityKind.ANNOTATION_PROPERTY:
            return list(self._annotation_property_parents.get(str(iri), ()))
        return self._property_hierarchy.direct_parents(str(iri))

    def direct_children(self, iri: str, kind: EntityKind = EntityKind.CLASS) -> list[str]:
        normalized_kind = EntityKind(kind)
        if normalized_kind is EntityKind.CLASS:
            return self.hierarchy.direct_children(str(iri))
        if normalized_kind is EntityKind.INDIVIDUAL:
            return list(self._class_individuals.get(str(iri), ()))
        if normalized_kind is EntityKind.ANNOTATION_PROPERTY:
            return list(self._annotation_property_children.get(str(iri), ()))
        return self._property_hierarchy.direct_children(str(iri))

    def hierarchy_bundle(
        self, iri: str, families: Mapping[str, Sequence[str]]
    ) -> dict[str, list[str]]:
        selected_iri = str(iri)
        result: dict[str, list[str]] = {}
        expressions = self._restriction_expressions.get(selected_iri, ())
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
            return value.value
        if isinstance(value, (Class, Datatype)):
            return value.iri.value
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
            {
                self._domain_range_value(record.value)
                for record in self._domain_range.domains(prop)
            }
        )

    def property_ranges(self, prop_iri: str) -> list[str]:
        try:
            prop = self._property(str(prop_iri))
        except InvalidIRIError:
            return []
        return sorted(
            {
                self._domain_range_value(record.value)
                for record in self._domain_range.ranges(prop)
            }
        )

    def _build_exclusions(self) -> frozenset[str]:
        excluded: set[str] = set()
        for subject in self._annotations.subjects():
            if not isinstance(subject, IRI):
                continue
            for posting in self._annotations.iter_subject(subject):
                assertion = posting.assertion
                if not isinstance(assertion.value, Literal):
                    continue
                lexical = assertion.value.lexical_form.strip().lower()
                property_iri = assertion.property.iri.value
                if property_iri == ANNOTATION_IRI and lexical in {"false", "0"}:
                    excluded.add(subject.value)
                elif property_iri == OWL_DEPRECATED and lexical in {"true", "1"}:
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
