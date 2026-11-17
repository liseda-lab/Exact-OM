"""Reusable conformance assertions for KnowledgeSource implementations."""

from pathlib import Path

from exact.core.contracts.knowledge import KnowledgeSource
from exact.core.entities.graph import AnnotationValue, Edge
from exact.core.entities.kinds import EntityKind


def assert_knowledge_source_conformance(source: KnowledgeSource) -> None:
    assert source.origin is None or isinstance(source.origin, Path)

    for kind in EntityKind:
        first = list(source.entities(kind))
        second = list(source.entities(kind))
        assert first == second
        assert first == sorted(set(first))
        assert all(isinstance(iri, str) and iri for iri in first)

    for iri in source.entities(EntityKind.CLASS):
        assert all(isinstance(label, str) for label in source.labels(iri))
        assert all(isinstance(value, AnnotationValue) for value in source.annotations(iri))
        assert all(isinstance(value, AnnotationValue) for value in source.attributes(iri))
        parents = source.direct_parents(iri)
        children = source.direct_children(iri)
        assert parents == sorted(set(parents))
        assert children == sorted(set(children))
        for parent in parents:
            assert iri in source.direct_children(parent)
        for child in children:
            assert iri in source.direct_parents(child)
        assert source.short_form(iri)

    for method in ("taxonomy", "owl2vecstar"):
        edges = source.projection_edges(method=method)
        assert all(isinstance(edge, Edge) for edge in edges)
        assert edges == sorted(set(edges), key=Edge.astuple)

    # Results are defensive copies: callers cannot mutate store state.
    classes = source.entities(EntityKind.CLASS)
    if classes:
        iri = classes[0]
        labels = source.labels(iri)
        labels.append("mutation")
        assert source.labels(iri) != labels

    assert isinstance(source.excluded_from_alignment(), frozenset)
