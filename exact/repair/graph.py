"""Observable, role-labelled repair graphs; no matching or neural runtime required."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields
from functools import cached_property
from typing import Any

import pyowl_core as owl

from .records import FrozenMapping

# Reject evaluator-only input at the boundary, including inside matcher channels.
_FORBIDDEN = frozenset(
    {
        "hidden_clean_theory",
        "clean_theory",
        "corruption_trace",
        "corruption_position",
        "reference_membership",
        "reference_alignment",
        "teacher_optimum",
        "teacher_labels",
        "split_id",
        "split_ids",
        "post_repair_explanations",
        "correctness",
        "ground_truth",
        "reference",
        "gold",
        "gold_label",
        "target_label",
        "corruption",
        "latent_parent",
        "teacher",
        "teacher_label",
        "future_explanation",
        "test_statistics",
        "oracle",
    }
)


def validate_observable_evidence(value: Any) -> None:
    """Reject reserved latent labels and non-finite/non-JSON evidence recursively."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("Evidence keys must be strings")
            if key.lower().replace("-", "_") in _FORBIDDEN:
                raise ValueError(f"Evaluator-only feature is forbidden: {key}")
            validate_observable_evidence(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            validate_observable_evidence(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Evidence must contain finite numbers")
    elif value is not None and not isinstance(value, (str, bool, int, float)):
        raise TypeError(f"Evidence must be JSON-compatible, got {type(value).__name__}")


def structural_id(value: Any) -> str:
    """Identify structural OWL nodes with kind-sensitive shared-core hashing."""
    return f"owl:{owl.structural_hexdigest(value)}"


def structural_arguments(value: Any) -> tuple[tuple[str, Any], ...]:
    """Expose public OWL dataclass fields, preserving roles and unordered operands."""
    result: list[tuple[str, Any]] = []
    for field in fields(value):
        if field.name in {"annotations", "iri", "kind"}:
            continue
        child = getattr(value, field.name)
        if isinstance(child, owl.StructuralNode):
            result.append((field.name, child))
        elif isinstance(child, (tuple, list, set, frozenset, owl.CanonicalSet)):
            result.extend(
                (field.name, item) for item in child if isinstance(item, owl.StructuralNode)
            )
    return tuple(sorted(result, key=lambda row: (row[0], structural_id(row[1]))))


@dataclass(frozen=True)
class GraphNode:
    """A public node and its sparse observable features."""

    node_id: str
    kind: str
    features: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class GraphExplanation:
    """A bounded diagnosis support and witnessed query available before selection."""

    explanation_id: str
    support_object_ids: tuple[str, ...] = ()
    support_axioms: tuple[Any, ...] = ()
    witness: Any | None = None
    available_before_decision: bool = True


@dataclass(frozen=True)
class ObservableGraph:
    """Immutable observable graph, independent of the complete reasoning input."""

    nodes: tuple[GraphNode, ...]
    edges: tuple[tuple[str, str, str], ...]
    object_nodes: tuple[tuple[str, str], ...]
    omitted_nodes: tuple[str, ...] = ()
    omitted_supports: tuple[str, ...] = ()
    omitted_evidence: tuple[str, ...] = ()

    @property
    def metadata(self) -> tuple[tuple[str, ...], tuple[tuple[str, str, str], ...]]:
        """Return the node/edge type schema expected by standard HGT layers."""
        kinds = {node.node_id: node.kind for node in self.nodes}
        return (
            tuple(sorted(set(kinds.values()))),
            tuple(sorted({(kinds[src], role, kinds[dst]) for src, role, dst in self.edges})),
        )

    @cached_property
    def adjacency(self) -> Mapping[str, frozenset[str]]:
        """Cache graph topology once for all shared per-object readouts."""
        result: dict[str, set[str]] = {}
        for source, _, target in self.edges:
            result.setdefault(source, set()).add(target)
        return FrozenMapping({key: frozenset(value) for key, value in result.items()})

    @cached_property
    def object_lookup(self) -> Mapping[str, str]:
        """Index editable targets once for all candidates."""
        return FrozenMapping(dict(self.object_nodes))

    @cached_property
    def explanation_ids(self) -> frozenset[str]:
        """Index included explanations independently of the readout count."""
        return frozenset(node.node_id for node in self.nodes if node.kind == "explanation")

    @cached_property
    def outgoing(self) -> Mapping[str, tuple[tuple[str, str], ...]]:
        """Index directed argument roles for per-object aggregation."""
        result: dict[str, list[tuple[str, str]]] = {}
        for source, role, target in self.edges:
            result.setdefault(source, []).append((role, target))
        return FrozenMapping({key: tuple(value) for key, value in result.items()})

    def context_ids(self, object_id: str, hops: int = 2) -> tuple[str, ...]:
        """Return bounded object context, including all included explanation supports."""
        selected = {self.object_lookup[object_id]}
        adjacency = self.adjacency
        frontier = selected.copy()
        for _ in range(hops):
            frontier = {dst for src in frontier for dst in adjacency.get(src, ())} - selected
            selected.update(frontier)
        explanations = self.explanation_ids
        selected.update(
            dst for src in tuple(selected & explanations) for dst in adjacency.get(src, ())
        )
        return tuple(sorted(selected))


def _features(value: Any, prefix: str = "") -> dict[str, float]:
    """Flatten arbitrary observed channels; text uses tokens, never IRI-specific weights."""
    result: dict[str, float] = {}
    if isinstance(value, Mapping):
        for key, item in sorted(value.items()):
            if key in {"omitted", "evidence_omissions"}:
                continue
            result.update(_features(item, f"{prefix}/{key}"))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            result.update(_features(item, f"{prefix}/{index}"))
    elif isinstance(value, (int, float)):
        result[prefix] = float(value)
    elif isinstance(value, str):
        for token in re.findall(r"[\w]+", value.lower()):
            key = f"{prefix}/token:{token}"
            result[key] = result.get(key, 0.0) + 1.0
    elif value is None:
        result[f"{prefix}/missing"] = 1.0
    return result


def build_observable_graph(
    objects: Iterable[Any],
    *,
    fixed_axioms: Iterable[Any] = (),
    source_axioms: Iterable[Any] = (),
    target_axioms: Iterable[Any] = (),
    evidence: Mapping[str, Any] | None = None,
    explanations: Iterable[GraphExplanation] = (),
    retrieved_symbols: Iterable[Any] = (),
    max_nodes: int | None = None,
) -> ObservableGraph:
    """Build syntax, occurrence, diagnosis and evidence nodes before encoding a round.

    Editable objects and retrieved symbols are mandatory. A context budget admits
    explanations atomically with their complete supports, then fixed axioms. Any
    omitted context is recorded; callers still verify against the full theory.
    """
    evidence = evidence or {}
    validate_observable_evidence(evidence)
    if max_nodes is not None and max_nodes < 1:
        raise ValueError("max_nodes must be positive")
    entity_sides: dict[str, set[str]] = {}
    for side, axioms in (("source", source_axioms), ("target", target_axioms)):
        for axiom in axioms:
            for entity in owl.signature(axiom):
                entity_sides.setdefault(structural_id(entity), set()).add(side)
    nodes: dict[str, GraphNode] = {}
    edges: set[tuple[str, str, str]] = set()
    object_nodes: dict[str, str] = {}
    omitted_nodes: list[str] = []
    omitted_supports: list[str] = []
    transaction_nodes: list[str] | None = None
    transaction_edges: list[tuple[str, str, str]] | None = None

    def add_node(node_id: str, kind: str, features: Mapping[str, float]) -> None:
        if node_id not in nodes and transaction_nodes is not None:
            transaction_nodes.append(node_id)
        nodes[node_id] = GraphNode(node_id, kind, tuple(sorted(features.items())))

    def connect(src: str, role: str, dst: str) -> None:
        for edge in ((src, role, dst), (dst, f"reverse_{role}", src)):
            if edge not in edges and transaction_edges is not None:
                transaction_edges.append(edge)
            edges.add(edge)

    def add_structure(value: Any) -> str:
        node_id = structural_id(value)
        if node_id in nodes:
            return node_id
        name = type(value).__name__
        if isinstance(value, owl.Entity):
            kind = value.kind.value
        elif isinstance(value, owl.Literal):
            kind = "literal"
        elif isinstance(value, owl.Axiom):
            kind = "axiom"
        else:
            kind = "constructor"
        feature = {f"syntax:{name}": 1.0}
        if isinstance(value, owl.Entity):
            feature.update({f"side:{side}": 1.0 for side in entity_sides.get(node_id, {"unknown"})})
        for attribute in fields(value):
            if attribute.name in {"iri", "kind", "annotations"}:
                continue
            scalar = getattr(value, attribute.name)
            if isinstance(scalar, (str, int, float, bool)):
                feature.update(_features(scalar, f"syntax/{attribute.name}"))
        add_node(node_id, str(kind), feature)
        for role, child in structural_arguments(value):
            connect(node_id, role, add_structure(child))
        return node_id

    def add_evidence(owner: str, payload: Any) -> None:
        node_id = f"evidence:{owner}"
        add_node(node_id, "evidence", _features(payload, "observed"))
        connect(node_id, "supports", owner)

    for obj in sorted(objects, key=lambda item: item.object_id):
        node_id = f"object:{obj.object_id}"
        if obj.object_id in object_nodes:
            raise ValueError(f"Duplicate object ID: {obj.object_id}")
        object_nodes[obj.object_id] = node_id
        payload = evidence.get(obj.object_id)
        has_score = False
        if isinstance(payload, Mapping):
            if "score_missing" in payload:
                has_score = not bool(payload["score_missing"])
            else:
                mapping = payload.get("mapping", payload)
                has_score = isinstance(mapping, Mapping) and any(
                    key in mapping and mapping[key] is not None
                    for key in ("Score", "score", "confidence", "scores")
                )
        add_node(
            node_id,
            "mapping" if obj.kind == "mapping" else "statement",
            {
                "eligible": float(obj.eligible and not obj.locked),
                "score_missing": float(not has_score),
                f"authorship:{obj.authorship}": 1.0,
                f"side:{obj.source or 'unknown'}": 1.0,
            },
        )
        for axiom in obj.original_axioms:
            connect(node_id, "asserts", add_structure(axiom))
        for role in ("source", "target"):
            endpoint = getattr(obj, f"{role}_entity", None)
            if endpoint is not None:
                connect(node_id, role, add_structure(endpoint))
        if payload is not None:
            add_evidence(node_id, payload)
    for symbol in retrieved_symbols:
        add_structure(symbol)
    for owner, payload in evidence.items():
        if owner in nodes:
            add_evidence(owner, payload)
        elif owner not in object_nodes and owner != "evidence_omissions":
            node_id = f"evidence:global:{owner}"
            add_node(node_id, "evidence", _features(payload, f"observed/{owner}"))
            for object_node in object_nodes.values():
                connect(node_id, "global_context", object_node)
    if max_nodes is not None and len(nodes) > max_nodes:
        raise ValueError("Context budget cannot fit mandatory object/evidence/retrieval nodes")

    def try_include(build: Any, label: str, *, support: bool = False) -> None:
        nonlocal transaction_nodes, transaction_edges
        if max_nodes is None:
            build()
            return
        transaction_nodes, transaction_edges = [], []
        build()
        if len(nodes) > max_nodes:
            omitted_nodes.extend(transaction_nodes)
            if support:
                omitted_supports.append(label)
            for key in transaction_nodes:
                del nodes[key]
            edges.difference_update(transaction_edges)
        transaction_nodes, transaction_edges = None, None

    seen_explanations: set[str] = set()
    for explanation in sorted(explanations, key=lambda item: item.explanation_id):
        if explanation.explanation_id in seen_explanations:
            raise ValueError("Explanation IDs must identify unique supports and witnesses")
        seen_explanations.add(explanation.explanation_id)
        if not explanation.available_before_decision:
            raise ValueError("Post-decision explanations cannot be model input")

        def add_explanation() -> None:
            node_id = f"explanation:{explanation.explanation_id}"
            add_node(
                node_id,
                "explanation",
                {
                    "explanation_missing": float(
                        not explanation.support_object_ids and not explanation.support_axioms
                    )
                },
            )
            for object_id in explanation.support_object_ids:
                if object_id not in object_nodes:
                    raise ValueError(f"Unknown explanation support: {object_id}")
                connect(node_id, "support", object_nodes[object_id])
            for axiom in explanation.support_axioms:
                connect(node_id, "support", add_structure(axiom))
            if explanation.witness is not None:
                connect(node_id, "witness", add_structure(explanation.witness))

        try_include(add_explanation, explanation.explanation_id, support=True)
    for axiom in sorted(fixed_axioms, key=structural_id):

        def add_fixed() -> None:
            occurrence = f"fixed:{structural_id(axiom)}"
            add_node(occurrence, "statement", {"eligible": 0.0})
            connect(occurrence, "asserts", add_structure(axiom))

        try_include(add_fixed, structural_id(axiom))
    # Every node has a typed self relation, including isolated retrieved vocabulary.
    edges.update((node.node_id, "self", node.node_id) for node in nodes.values())
    return ObservableGraph(
        tuple(nodes[key] for key in sorted(nodes)),
        tuple(sorted(edges)),
        tuple(sorted(object_nodes.items())),
        tuple(sorted(set(omitted_nodes) - nodes.keys())),
        tuple(omitted_supports),
        tuple(str(item) for item in evidence.get("evidence_omissions", ())),
    )


def feature_vector(node: GraphNode, dimension: int = 128) -> list[float]:
    """Feature-hash all observed channels to fixed dimensions without fitting on tests."""
    if dimension < 1:
        raise ValueError("Feature dimension must be positive")
    result = [0.0] * dimension
    for name, value in ((f"type:{node.kind}", 1.0), *node.features):
        digest = hashlib.blake2b(name.encode(), digest_size=8).digest()
        index = int.from_bytes(digest, "big") % dimension
        result[index] += value if digest[0] & 1 else -value
    return result
