"""Declared observable graph vocabulary, independent of corpus observations.

The public OWL structural AST supplies syntax roles. Graph-builder rules supply
occurrence, evidence, explanation and reverse/self relations. This is an input
language declaration, not a claim that a native reasoner supports that language.
"""

from __future__ import annotations

from dataclasses import MISSING, fields, is_dataclass
from typing import get_args, get_type_hints

import pyowl_core as owl

from .records import canonical_hash

SCHEMA = "exact-repair/declared-graph-schema/v1"
LANGUAGE = "pyowl-core/public-structural-ast/v1"


def _kinds(cls):
    if issubclass(cls, owl.Entity):
        kind = next(f for f in fields(cls) if f.name == "kind").default
        return {item.value for item in owl.EntityKind} if kind is MISSING else {kind.value}
    if issubclass(cls, owl.Literal):
        return {"literal"}
    return {"axiom" if issubclass(cls, owl.Axiom) else "constructor"}


def _structural_types(annotation):
    if isinstance(annotation, type) and issubclass(annotation, owl.StructuralNode):
        return {annotation}
    return {cls for argument in get_args(annotation) for cls in _structural_types(argument)}


def generic_graph_schema():
    """Resolve the generic language, without accepting any case or split inputs."""
    classes = sorted(
        {cls for cls in vars(owl).values() if isinstance(cls, type)
         and issubclass(cls, owl.StructuralNode) and is_dataclass(cls)},
        key=lambda cls: (cls.__module__, cls.__name__),
    )
    structural = {kind for cls in classes for kind in _kinds(cls)}
    nodes = structural | {"mapping", "statement", "evidence", "explanation"}
    edges = set()
    language = []

    def connect(source, role, target):
        edges.add((source, role, target))
        edges.add((target, "reverse_" + role, source))

    for cls in classes:
        hints = get_type_hints(cls)
        roles = []
        for field in fields(cls):
            if field.name in {"annotations", "iri", "kind"}:
                continue
            children = _structural_types(hints[field.name])
            roles.append((field.name, sorted(c.__name__ for c in children)))
            for source in _kinds(cls):
                for child in children:
                    for target in _kinds(child):
                        connect(source, field.name, target)
        language.append((cls.__name__, sorted(_kinds(cls)), roles))
    for occurrence in ("mapping", "statement"):
        connect(occurrence, "asserts", "axiom")
        for entity in owl.EntityKind:
            for role in ("source", "target"):
                connect(occurrence, role, entity.value)
        connect("evidence", "global_context", occurrence)
        connect("explanation", "support", occurrence)
    # Evidence owners exist before explanations are added by the builder.
    for owner in structural | {"mapping", "statement", "evidence"}:
        connect("evidence", "supports", owner)
    connect("explanation", "support", "axiom")
    for witness in structural:
        connect("explanation", "witness", witness)
    edges.update((node, "self", node) for node in nodes)
    return dict(schema=SCHEMA, language=LANGUAGE, language_hash=canonical_hash(language),
                node_types=sorted(nodes), edge_types=[list(edge) for edge in sorted(edges)])


def declared_metadata(declaration):
    """Reject changed, partial or unsupported declarations before allocating weights."""
    if declaration != generic_graph_schema():
        raise ValueError("Declared graph schema does not match the generic input language")
    return (tuple(declaration["node_types"]),
            tuple(tuple(edge) for edge in declaration["edge_types"]))


def training_metadata(observed, declaration=None, warm_start_metadata=None):
    """Admit observed train/development types without expanding a declaration."""
    nodes, edges = set(observed[0]), set(map(tuple, observed[1]))
    if declaration is not None:
        expected = declared_metadata(declaration)
        if not nodes <= set(expected[0]) or not edges <= set(expected[1]):
            raise ValueError("Training graph has types outside the declared graph schema")
        if warm_start_metadata is not None and (
            tuple(sorted(warm_start_metadata[0])),
            tuple(sorted(map(tuple, warm_start_metadata[1]))),
        ) != expected:
            raise ValueError("Warm-start graph schema differs from the declared graph schema")
        return expected
    if warm_start_metadata is not None:
        nodes.update(warm_start_metadata[0])
        edges.update(map(tuple, warm_start_metadata[1]))
    return tuple(sorted(nodes)), tuple(sorted(edges))
