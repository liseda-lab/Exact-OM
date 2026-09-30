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

from .records import FrozenMapping, canonical_hash

FEATURE_SCHEMA_V3 = "exact-repair/observable-features/v3"
_STATES = frozenset(
    {
        "observed",
        "missing",
        "truncated",
        "unsupported",
        "timeout",
        "error",
        "not_applicable",
        "unvisited",
    }
)
_IDENTIFIERS = frozenset(
    {
        "iri",
        "uri",
        "path",
        "file",
        "hash",
        "id",
        "candidate_id",
        "object_id",
        "source_id",
        "parent_id",
        "seed",
        "namespace",
        "content_hash",
        "document_id",
        "srcentity",
        "tgtentity",
    }
)
_FEATURE_KEYS = frozenset(
    {
        "score",
        "confidence",
        "scores",
        "channels",
        "lexical",
        "structural",
        "semantic",
        "matcher",
        "matcher_identity",
        "source_text",
        "target_text",
        "description",
        "label",
        "labels",
        "definition",
        "definitions",
        "text",
        "provenance",
        "authorship",
        "mapping",
        "observed",
        "value",
        "values",
        "status",
        "score_missing",
        "relation",
        "evidence_kind",
        "kind",
        "source",
        "target",
        "reliability",
        "availability",
        "available_before_decision",
        "truncated",
        "missing",
        "unsupported",
        "omitted",
        "evidence_omissions",
        "calibration",
        "calibration_status",
        "license",
        "access_status",
        "origin",
        "observed_at",
        "local_name",
        "annotations",
        "property",
        "sides",
        "Kind",
        "SrcKind",
        "TgtKind",
        "Source",
        "eligible",
        "locked",
        "matching_features",
        "decomposition",
        "attributes",
        "embedding",
        "explanations",
        "Src",
        "Tgt",
        "Relation",
        "Score",
    }
)

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
        "intended_assignment",
        "intended_action",
        "assignment_label",
        "semantic_label",
        "comparison_label",
        "rationale",
        "label_confidence",
        "parent_group_id",
        "structural_parent",
        "generation_seed",
        "semantic_preference",
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
            if key == "available_before_decision" and item is not True:
                raise ValueError("Evidence must be available before the decision")
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
    support_status: str = "unknown"
    obligation_kind: str = "unspecified"
    theory_hash: str = ""
    policy_hash: str = ""

    def __post_init__(self):
        if self.support_status not in {
            "sufficient",
            "minimal",
            "partial",
            "unavailable",
            "unknown",
        }:
            raise ValueError("Unknown explanation support status")


@dataclass(frozen=True)
class ObservableGraph:
    """Immutable observable graph, independent of the complete reasoning input."""

    nodes: tuple[GraphNode, ...]
    edges: tuple[tuple[str, str, str], ...]
    object_nodes: tuple[tuple[str, str], ...]
    omitted_nodes: tuple[str, ...] = ()
    omitted_supports: tuple[str, ...] = ()
    omitted_evidence: tuple[str, ...] = ()
    feature_schema: str = "exact-repair/observable-features/v2"
    admitted_supports: tuple[GraphExplanation, ...] = ()
    support_omissions: tuple[tuple[str, str], ...] = ()
    admission_policy: str = "atomic-complete-support/v3.1"
    preparation_identity: str = ""

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


def validate_admitted_supports(graph: ObservableGraph) -> None:
    """Reject a replay whose purported admitted support cannot fit its memory."""
    nodes = {node.node_id for node in graph.nodes}
    ids = set()
    for support in graph.admitted_supports:
        if support.explanation_id in ids or not support.available_before_decision:
            raise ValueError("Invalid admitted support identity/availability")
        ids.add(support.explanation_id)
        if f"explanation:{support.explanation_id}" not in nodes:
            raise ValueError("Admitted support is absent from graph memory")
        if not set(support.support_object_ids) <= dict(graph.object_nodes).keys():
            raise ValueError("Admitted support names unavailable objects")
        structures = (
            *support.support_axioms,
            *((support.witness,) if support.witness is not None else ()),
        )
        for value in structures:
            if structural_id(value) not in nodes or any(
                structural_id(entity) not in nodes for entity in owl.signature(value)
            ):
                raise ValueError("Admitted support structure is not encodable by graph memory")


def _features(
    value: Any,
    prefix: str = "",
    *,
    max_text_tokens: int | None = None,
    omitted_text: list[str] | None = None,
    typed: bool = False,
) -> dict[str, float]:
    """Flatten arbitrary observed channels; text uses tokens, never IRI-specific weights."""
    result: dict[str, float] = {}
    if isinstance(value, Mapping):
        if typed and "status" in value:
            if value["status"] not in _STATES:
                raise ValueError("Unknown evidence channel status")
            result[f"{prefix}/state:{value['status']}"] = 1.0
            if value["status"] != "observed":
                return result
        for key, item in sorted(value.items()):
            if typed and (
                key.lower() in _IDENTIFIERS
                or key.endswith(("_hash", "_id", "_path"))
                or key in {"status", "Src", "Tgt"}
            ):
                continue
            if typed and key not in _FEATURE_KEYS:
                raise ValueError(f"Unregistered v3 inference feature channel: {key}")
            if key in {"omitted", "evidence_omissions"}:
                continue
            result.update(
                _features(
                    item,
                    f"{prefix}/{key}",
                    max_text_tokens=max_text_tokens,
                    omitted_text=omitted_text,
                    typed=typed,
                )
            )
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            result.update(
                _features(
                    item,
                    f"{prefix}/{index}",
                    max_text_tokens=max_text_tokens,
                    omitted_text=omitted_text,
                    typed=typed,
                )
            )
    elif isinstance(value, (int, float)):
        result[prefix] = float(value)
        if typed:
            result[f"{prefix}/state:observed"] = 1.0
    elif isinstance(value, str):
        if (
            typed
            and prefix.endswith("/local_name")
            and re.fullmatch(
                r"(?:[A-Za-z]|[A-Za-z]*[0-9][\w-]*|[0-9a-fA-F]{16,}|(?:Node|Core|Noise)[\w-]*)",
                value,
            )
        ):
            result[f"{prefix}/state:missing"] = 1.0
            return result
        if typed and ("://" in value or value.startswith(("urn:", "/"))):
            return result
        for index, match in enumerate(re.finditer(r"[\w]+", value.lower())):
            if max_text_tokens is not None and index >= max_text_tokens:
                if omitted_text is not None:
                    omitted_text.append(f"text_tokens:{prefix}")
                if typed:
                    result[f"{prefix}/state:truncated"] = 1.0
                break
            key = f"{prefix}/token:{match.group(0)}"
            result[key] = result.get(key, 0.0) + 1.0
    elif value is None:
        result[f"{prefix}/state:missing" if typed else f"{prefix}/missing"] = 1.0
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
    max_edges: int | None = None,
    max_explanations: int | None = None,
    max_text_tokens: int | None = None,
    feature_schema: str = "exact-repair/observable-features/v2",
) -> ObservableGraph:
    """Build syntax, occurrence, diagnosis and evidence nodes before encoding a round.

    Editable objects and retrieved symbols are mandatory. A context budget admits
    explanations atomically with their complete supports, then fixed axioms. Any
    omitted context is recorded; callers still verify against the full theory.
    """
    evidence = evidence or {}
    if feature_schema not in {FEATURE_SCHEMA_V3, "exact-repair/observable-features/v2"}:
        raise ValueError("Unknown observable feature schema")
    typed = feature_schema == FEATURE_SCHEMA_V3
    validate_observable_evidence(evidence)
    for name, limit, minimum in (
        ("max_nodes", max_nodes, 1),
        ("max_edges", max_edges, 1),
        ("max_explanations", max_explanations, 0),
        ("max_text_tokens", max_text_tokens, 1),
    ):
        if limit is not None and (type(limit) is not int or limit < minimum):
            raise ValueError(f"{name} must be an integer >= {minimum}")
    omitted_text: list[str] = []
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
    admitted_supports: list[GraphExplanation] = []
    support_omissions: list[tuple[str, str]] = []
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
        add_node(
            node_id,
            "evidence",
            _features(
                payload,
                "observed",
                max_text_tokens=max_text_tokens,
                omitted_text=omitted_text,
                typed=typed,
            ),
        )
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
        if typed:
            for candidate in obj.candidates:
                for expression in (*candidate.axioms, *candidate.active_expressions):
                    for entity in owl.signature(expression):
                        add_structure(entity)
    for symbol in retrieved_symbols:
        add_structure(symbol)
    for owner, payload in evidence.items():
        if owner in nodes:
            add_evidence(owner, payload)
        elif owner not in object_nodes and owner != "evidence_omissions":
            node_id = f"evidence:global:{owner}"
            add_node(
                node_id,
                "evidence",
                _features(
                    payload,
                    "observed/global" if typed else f"observed/{owner}",
                    max_text_tokens=max_text_tokens,
                    omitted_text=omitted_text,
                    typed=typed,
                ),
            )
            for object_node in object_nodes.values():
                connect(node_id, "global_context", object_node)

    def exceeds_budget() -> bool:
        return (max_nodes is not None and len(nodes) > max_nodes) or (
            max_edges is not None and len(edges) + len(nodes) > max_edges
        )

    if exceeds_budget():
        raise ValueError(
            "Context budget cannot fit mandatory object/evidence/retrieval nodes and edges"
        )

    def try_include(build: Any, label: str, *, support: bool = False) -> None:
        nonlocal transaction_nodes, transaction_edges
        if max_nodes is None and max_edges is None:
            build()
            return
        transaction_nodes, transaction_edges = [], []
        build()
        if exceeds_budget():
            omitted_nodes.extend(transaction_nodes)
            if support:
                omitted_supports.append(label)
                support_omissions.append((label, "graph_budget"))
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
        if max_explanations is not None and len(seen_explanations) > max_explanations:
            omitted_supports.append(explanation.explanation_id)
            support_omissions.append((explanation.explanation_id, "explanation_cap"))
            continue

        def add_explanation() -> None:
            node_id = f"explanation:{explanation.explanation_id}"
            add_node(
                node_id,
                "explanation",
                {
                    "explanation_missing": float(
                        not explanation.support_object_ids and not explanation.support_axioms
                    ),
                    **(
                        {
                            f"support_status:{explanation.support_status}": 1.0,
                            f"obligation_kind:{explanation.obligation_kind}": 1.0,
                        }
                        if typed
                        else {}
                    ),
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
        if f"explanation:{explanation.explanation_id}" in nodes:
            admitted_supports.append(explanation)
    for axiom in sorted(fixed_axioms, key=structural_id):

        def add_fixed() -> None:
            occurrence = f"fixed:{structural_id(axiom)}"
            add_node(occurrence, "statement", {"eligible": 0.0})
            connect(occurrence, "asserts", add_structure(axiom))

        try_include(add_fixed, structural_id(axiom))
    if typed:
        coverage = {
            "context_nodes_truncated": float(bool(omitted_nodes)),
            "context_supports_truncated": float(bool(omitted_supports)),
            "text_truncated": float(bool(omitted_text)),
        }
        for node_id in object_nodes.values():
            node = nodes[node_id]
            nodes[node_id] = GraphNode(
                node.node_id, node.kind, tuple(sorted((*node.features, *coverage.items())))
            )
    # Every node has a typed self relation, including isolated retrieved vocabulary.
    edges.update((node.node_id, "self", node.node_id) for node in nodes.values())
    return ObservableGraph(
        tuple(nodes[key] for key in sorted(nodes)),
        tuple(sorted(edges)),
        tuple(sorted(object_nodes.items())),
        tuple(sorted(set(omitted_nodes) - nodes.keys())),
        tuple(omitted_supports),
        tuple(str(item) for item in evidence.get("evidence_omissions", ()))
        + tuple(sorted(set(omitted_text))),
        feature_schema,
        tuple(admitted_supports),
        tuple(support_omissions),
    )


@dataclass(frozen=True)
class EffectivePreparation:
    """One resolved observable-preparation request, shared by every neural path."""

    max_graph_nodes: int | None = None
    max_graph_edges: int | None = None
    max_explanations: int | None = None
    max_text_tokens: int | None = None
    pair_factor_limit_per_object: int = 16
    pair_max_pairs: int | None = 128
    pair_max_factors: int | None = 65536
    retrieval_config: Any = None
    revision: str = "v3"
    schema: str = "exact-repair/effective-preparation/v3.1"

    def __post_init__(self) -> None:
        from .retrieval import RetrievalConfig

        if self.retrieval_config is None:
            object.__setattr__(self, "retrieval_config", RetrievalConfig())
        if (
            self.revision not in {"v2", "v3"}
            or self.schema != "exact-repair/effective-preparation/v3.1"
        ):
            raise ValueError("Unknown effective preparation revision")
        for name, minimum in (
            ("max_graph_nodes", 1),
            ("max_graph_edges", 1),
            ("max_explanations", 0),
            ("max_text_tokens", 1),
            ("pair_factor_limit_per_object", 0),
            ("pair_max_pairs", 0),
            ("pair_max_factors", 0),
        ):
            limit = getattr(self, name)
            if limit is not None and (type(limit) is not int or limit < minimum):
                raise ValueError(f"{name} must be an integer >= {minimum}")

    @property
    def content_hash(self) -> str:
        from .records import canonical_hash

        return canonical_hash(self)

    def freeze_options(self) -> dict[str, Any]:
        return {
            key: getattr(self, key)
            for key in (
                "max_graph_nodes",
                "max_graph_edges",
                "max_explanations",
                "max_text_tokens",
                "pair_factor_limit_per_object",
                "pair_max_pairs",
                "pair_max_factors",
                "retrieval_config",
            )
        }

    def graph(
        self, problem: Any, retrieval: Any, *, retrieved_symbols: Iterable[Any] = ()
    ) -> ObservableGraph:
        from dataclasses import replace

        graph = build_observable_graph(
            problem.objects,
            fixed_axioms=problem.fixed_axioms,
            source_axioms=problem.source_axioms,
            target_axioms=problem.target_axioms,
            evidence=retrieval.graph_evidence(problem.evidence),
            retrieved_symbols=tuple(retrieved_symbols) + tuple(retrieval.symbols),
            explanations=retrieval.explanations,
            max_nodes=self.max_graph_nodes,
            max_edges=self.max_graph_edges,
            max_explanations=self.max_explanations,
            max_text_tokens=self.max_text_tokens,
            feature_schema=f"exact-repair/observable-features/{self.revision}",
        )
        return replace(graph, preparation_identity=self.content_hash)

    def pairs(self, problem: Any, graph: ObservableGraph, *, enabled: bool = True) -> Any:
        return select_interaction_pairs(
            problem,
            graph=graph,
            explanations=graph.admitted_supports,
            per_object_limit=self.pair_factor_limit_per_object if enabled else 0,
            max_pairs=self.pair_max_pairs,
            max_factors=self.pair_max_factors,
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


@dataclass(frozen=True)
class InteractionSelection:
    """Replayable sparse semantic pair index; hyperedges remain graph records."""

    pairs: tuple[tuple[int, int], ...]
    reasons: tuple[tuple[int, int, tuple[str, ...]], ...]
    omissions: tuple[tuple[str, int], ...]
    considered_channels: tuple[str, ...]
    input_hash: str
    candidate_factors: int
    schema: str = "exact-repair/interaction-selection/v3"

    @property
    def content_hash(self) -> str:
        return canonical_hash(self)


def select_interaction_pairs(
    problem: Any,
    *,
    graph: ObservableGraph | None = None,
    per_object_limit: int = 16,
    explanations: Iterable[GraphExplanation] = (),
    max_pairs: int | None = None,
    max_factors: int | None = None,
    visible_queries: Iterable[Any] = (),
) -> InteractionSelection:
    """Select explanation-first pairs from the same final inventory on every path.

    Index signatures once, then stream support links. Caps are applied before
    candidate matrices exist. Omitted links retain their channel and count.
    """
    from collections import defaultdict

    for name, value in (
        ("per_object_limit", per_object_limit),
        ("max_pairs", max_pairs),
        ("max_factors", max_factors),
    ):
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError(f"{name} must be a nonnegative integer")
    objects = problem.objects
    explanations = tuple(sorted(explanations, key=lambda e: e.explanation_id))
    visible_queries = tuple(sorted(visible_queries, key=structural_id))
    object_index = {obj.object_id: index for index, obj in enumerate(objects)}
    original, candidate = defaultdict(set), defaultdict(set)
    for index, obj in enumerate(objects):
        for axiom in obj.original_axioms:
            for entity in owl.signature(axiom):
                original[entity].add(index)
        for action in obj.candidates:
            for axiom in (*action.axioms, *action.active_expressions):
                for entity in owl.signature(axiom):
                    candidate[entity].add(index)
    selected: dict[tuple[int, int], set[str]] = {}
    omitted: dict[str, int] = defaultdict(int)
    degrees = [0] * len(objects)
    factor_count = 0
    channels = (
        "explanation",
        "original_signature",
        "fixed_axiom",
        "visible_query",
        "candidate_signature",
    )

    def include(indices: Iterable[int], channel: str) -> None:
        nonlocal factor_count
        indices = sorted(set(indices))
        for offset, first in enumerate(indices):
            rest = indices[offset + 1 :]
            known = {second for (left, second) in selected if left == first}
            for second in known.intersection(rest):
                selected[first, second].add(channel)
            if degrees[first] >= per_object_limit or (
                max_pairs is not None and len(selected) >= max_pairs
            ):
                omitted[channel] += len(rest) - len(known.intersection(rest))
                continue
            for position, second in enumerate(rest):
                pair = first, second
                if pair in selected:
                    continue
                if degrees[first] >= per_object_limit or (
                    max_pairs is not None and len(selected) >= max_pairs
                ):
                    omitted[channel] += (
                        len(rest) - position - len(known.intersection(rest[position:]))
                    )
                    break
                factors = len(objects[first].candidates) * len(objects[second].candidates)
                if degrees[second] >= per_object_limit or (
                    max_factors is not None and factor_count + factors > max_factors
                ):
                    omitted[channel] += 1
                    continue
                selected[pair] = {channel}
                degrees[first] += 1
                degrees[second] += 1
                factor_count += factors

    for explanation in explanations:
        if not explanation.available_before_decision:
            raise ValueError("Post-decision explanations cannot select interactions")
        if (
            graph is not None
            and f"explanation:{explanation.explanation_id}" not in graph.explanation_ids
        ):
            omitted["unadmitted_explanation"] += 1
            continue
        if not set(explanation.support_object_ids) <= object_index.keys():
            raise ValueError("Unknown explanation support object")
        indices = {object_index[key] for key in explanation.support_object_ids}
        for axiom in explanation.support_axioms:
            indices.update(i for entity in owl.signature(axiom) for i in candidate.get(entity, ()))
        include(indices, "explanation")
    for entity in sorted(original, key=structural_id):
        include(original[entity], "original_signature")
    for axiom in sorted(problem.fixed_axioms, key=structural_id):
        include(
            (i for entity in owl.signature(axiom) for i in original.get(entity, ())), "fixed_axiom"
        )
    for query in visible_queries:
        include(
            (i for entity in owl.signature(query) for i in candidate.get(entity, ())),
            "visible_query",
        )
    for entity in sorted(candidate, key=structural_id):
        include(candidate[entity], "candidate_signature")
    return InteractionSelection(
        tuple(sorted(selected)),
        tuple((i, j, tuple(sorted(selected[i, j]))) for i, j in sorted(selected)),
        tuple(sorted(omitted.items())),
        channels,
        canonical_hash(
            (
                problem.objects,
                problem.fixed_axioms,
                graph,
                explanations,
                visible_queries,
                per_object_limit,
                max_pairs,
                max_factors,
            )
        ),
        factor_count,
    )


def observable_interaction_pairs(
    problem: Any,
    *,
    per_object_limit: int = 16,
    explanations: Iterable[GraphExplanation] = (),
    graph: ObservableGraph | None = None,
    max_pairs: int | None = None,
    max_factors: int | None = None,
) -> tuple[tuple[int, int], ...]:
    """Compatibility projection of the shared, report-producing v3 selector."""
    return select_interaction_pairs(
        problem,
        graph=graph,
        per_object_limit=per_object_limit,
        explanations=explanations,
        max_pairs=max_pairs,
        max_factors=max_factors,
    ).pairs
