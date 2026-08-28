"""Deterministic, backend-neutral relation typing for accepted mappings."""

from __future__ import annotations

import heapq
import json
import time
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import pandas as pd

from exact.core.contracts.knowledge import KnowledgeSource
from exact.core.entities.kinds import EntityKind
from exact.io.writers._frames import validated_mapping_frame
from exact.io.writers.base import WriterOptionsError
from exact.utils.candidate_generation import normalize_candidate_text

_Node = tuple[str, str, str]
_SUPPORTED_SEMANTIC_KINDS = frozenset(
    {
        EntityKind.CLASS,
        EntityKind.OBJECT_PROPERTY,
        EntityKind.DATA_PROPERTY,
    }
)


class _SemanticTimeout(RuntimeError):
    pass


def _closure(
    source: KnowledgeSource,
    iri: str,
    *,
    direct: Callable[[str], list[str]],
    optimized_name: str,
) -> set[str]:
    optimized = getattr(source, optimized_name, None)
    if callable(optimized):
        return {str(value) for value in optimized(str(iri))}
    seen: set[str] = set()
    pending = list(direct(str(iri)))
    while pending:
        current = str(pending.pop())
        if current in seen:
            continue
        seen.add(current)
        pending.extend(direct(current))
    return seen


def _ancestors(source: KnowledgeSource, iri: str) -> set[str]:
    return _closure(
        source,
        iri,
        direct=source.direct_parents,
        optimized_name="ancestors",
    )


def _descendants(source: KnowledgeSource, iri: str) -> set[str]:
    return _closure(
        source,
        iri,
        direct=source.direct_children,
        optimized_name="descendants",
    )


def _anchor_frame(candidates: pd.DataFrame, anchors: Any | None) -> pd.DataFrame:
    if anchors is not None:
        return validated_mapping_frame(anchors)
    for marker in ("Anchor", "anchor", "is_anchor", "protected"):
        if marker in candidates.columns:
            marked = candidates[candidates[marker].fillna(False).astype(bool)]
            if not marked.empty:
                return marked.copy()
    return (
        candidates.sort_values(
            ["Score", "TgtEntity"],
            ascending=[False, True],
            kind="mergesort",
        )
        .drop_duplicates("SrcEntity", keep="first")
        .reset_index(drop=True)
    )


def _hierarchy_heuristic(
    frame: pd.DataFrame,
    source: KnowledgeSource,
    target: KnowledgeSource,
    anchors: Any | None,
) -> pd.DataFrame:
    anchor_frame = _anchor_frame(frame, anchors)
    images: dict[str, list[tuple[str, float]]] = {}
    for row in anchor_frame.itertuples(index=False):
        weight = max(float(row.Score), 1e-12)
        images.setdefault(str(row.SrcEntity), []).append((str(row.TgtEntity), weight))

    source_ancestor_cache: dict[str, set[str]] = {}
    source_descendant_cache: dict[str, set[str]] = {}
    target_ancestor_cache: dict[str, set[str]] = {}
    target_descendant_cache: dict[str, set[str]] = {}

    def ancestors_of(knowledge: KnowledgeSource, iri: str, *, target_side: bool) -> set[str]:
        cache = target_ancestor_cache if target_side else source_ancestor_cache
        if iri not in cache:
            cache[iri] = _ancestors(knowledge, iri)
        return cache[iri]

    def descendants_of(knowledge: KnowledgeSource, iri: str, *, target_side: bool) -> set[str]:
        cache = target_descendant_cache if target_side else source_descendant_cache
        if iri not in cache:
            cache[iri] = _descendants(knowledge, iri)
        return cache[iri]

    relations: list[str] = []
    confidences: list[float] = []
    for row in frame.itertuples(index=False):
        src, tgt = str(row.SrcEntity), str(row.TgtEntity)
        subsumed_by = 0.0
        subsumes = 0.0
        equivalent = 0.0

        for image, weight in images.get(src, ()):
            if tgt == image:
                equivalent += weight
            elif tgt in ancestors_of(target, image, target_side=True):
                subsumed_by += weight
            elif tgt in descendants_of(target, image, target_side=True):
                subsumes += weight

        for ancestor in ancestors_of(source, src, target_side=False):
            for image, weight in images.get(ancestor, ()):
                if tgt == image:
                    subsumed_by += weight
        for descendant in descendants_of(source, src, target_side=False):
            for image, weight in images.get(descendant, ()):
                if tgt == image:
                    subsumes += weight

        total = subsumed_by + subsumes + equivalent
        if subsumed_by > max(subsumes, equivalent):
            relation, dominant = "<", subsumed_by
        elif subsumes > max(subsumed_by, equivalent):
            relation, dominant = ">", subsumes
        else:
            relation, dominant = "=", equivalent
        relations.append(relation)
        confidences.append(0.0 if total == 0.0 else dominant / total)

    result = frame.copy()
    result["Relation"] = relations
    result["relation_confidence"] = confidences
    return result


def _entity_kind(value: Any) -> EntityKind:
    try:
        return EntityKind(str(getattr(value, "value", value)).strip().lower())
    except ValueError as exc:
        raise WriterOptionsError(f"Unsupported semantic relation entity kind: {value!r}") from exc


def _row_kinds(frame: pd.DataFrame, index: Any) -> tuple[EntityKind, EntityKind]:
    row = frame.loc[index]
    source_value = row.get("SrcKind", row.get("Kind", EntityKind.CLASS.value))
    target_value = row.get("TgtKind", source_value)
    return _entity_kind(source_value), _entity_kind(target_value)


def _normalized_labels(source: KnowledgeSource, iri: str) -> set[str]:
    return {
        normalized
        for label in source.labels(str(iri))
        if (normalized := normalize_candidate_text(str(label)))
    }


def _rank_summary(
    frame: pd.DataFrame,
    *,
    group_column: str,
    candidate_column: str,
) -> dict[str, tuple[str, float, float]]:
    summaries: dict[str, tuple[str, float, float]] = {}
    grouped = frame.groupby(group_column, sort=True, dropna=False)
    for group_value, group in grouped:
        best_by_candidate = (
            group.groupby(candidate_column, sort=True, dropna=False)["Score"].max().reset_index()
        )
        ordered = best_by_candidate.sort_values(
            ["Score", candidate_column],
            ascending=[False, True],
            kind="mergesort",
        )
        first = ordered.iloc[0]
        second_score = float(ordered.iloc[1]["Score"]) if len(ordered) > 1 else 0.0
        summaries[str(group_value)] = (
            str(first[candidate_column]),
            float(first["Score"]),
            float(first["Score"]) - second_score,
        )
    return summaries


def _semantic_anchor_rows(
    frame: pd.DataFrame,
    source: KnowledgeSource,
    target: KnowledgeSource,
    *,
    anchors: Any | None,
    threshold: float,
    margin: float,
) -> list[dict[str, Any]]:
    source_kinds = {
        str(frame.loc[index, "SrcEntity"]): _row_kinds(frame, index)[0] for index in frame.index
    }
    target_kinds = {
        str(frame.loc[index, "TgtEntity"]): _row_kinds(frame, index)[1] for index in frame.index
    }
    candidates: list[dict[str, Any]] = []
    if anchors is not None:
        explicit = validated_mapping_frame(anchors)
        for row in explicit.itertuples(index=False):
            src = str(row.SrcEntity)
            tgt = str(row.TgtEntity)
            candidates.append(
                {
                    "src": src,
                    "tgt": tgt,
                    "score": 1.0,
                    "candidate_score": float(row.Score),
                    "origin": "explicit",
                    "src_kind": source_kinds.get(src, EntityKind.CLASS),
                    "tgt_kind": target_kinds.get(tgt, EntityKind.CLASS),
                }
            )

    source_top = _rank_summary(
        frame,
        group_column="SrcEntity",
        candidate_column="TgtEntity",
    )
    target_top = _rank_summary(
        frame,
        group_column="TgtEntity",
        candidate_column="SrcEntity",
    )
    label_cache_source: dict[str, set[str]] = {}
    label_cache_target: dict[str, set[str]] = {}
    for index, row in frame.iterrows():
        src = str(row["SrcEntity"])
        tgt = str(row["TgtEntity"])
        src_kind, tgt_kind = _row_kinds(frame, index)
        if src_kind != tgt_kind or src_kind not in _SUPPORTED_SEMANTIC_KINDS:
            continue
        src_labels = label_cache_source.setdefault(src, _normalized_labels(source, src))
        tgt_labels = label_cache_target.setdefault(tgt, _normalized_labels(target, tgt))
        exact = bool(src_labels & tgt_labels)
        source_best = source_top[src]
        target_best = target_top[tgt]
        reciprocal = source_best[0] == tgt and target_best[0] == src
        high_margin = source_best[2] >= margin and target_best[2] >= margin
        high_score = float(row["Score"]) >= threshold
        if not exact and not (reciprocal and high_margin and high_score):
            continue
        candidates.append(
            {
                "src": src,
                "tgt": tgt,
                "score": 1.0 if exact else float(row["Score"]),
                "candidate_score": float(row["Score"]),
                "origin": "exact_label" if exact else "reciprocal_top1",
                "src_kind": src_kind,
                "tgt_kind": tgt_kind,
            }
        )

    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for candidate in sorted(
        candidates,
        key=lambda value: (
            0 if value["origin"] in {"explicit", "exact_label"} else 1,
            -float(value["score"]),
            value["src"],
            value["tgt"],
        ),
    ):
        key = (candidate["src"], candidate["tgt"])
        previous = selected.get(key)
        if previous is None or float(candidate["score"]) > float(previous["score"]):
            selected[key] = candidate

    source_images: dict[str, str] = {}
    target_images: dict[str, str] = {}
    for candidate in selected.values():
        src = candidate["src"]
        tgt = candidate["tgt"]
        if candidate["src_kind"] != candidate["tgt_kind"]:
            raise WriterOptionsError(f"Semantic equivalence anchor crosses kinds: {src} -> {tgt}")
        if src in source_images and source_images[src] != tgt:
            raise WriterOptionsError(
                f"Conflicting semantic equivalence anchors for source {src!r}: "
                f"{source_images[src]!r} and {tgt!r}"
            )
        if tgt in target_images and target_images[tgt] != src:
            raise WriterOptionsError(
                f"Conflicting semantic equivalence anchors for target {tgt!r}: "
                f"{target_images[tgt]!r} and {src!r}"
            )
        source_images[src] = tgt
        target_images[tgt] = src
    return sorted(selected.values(), key=lambda value: (value["src"], value["tgt"]))


def _check_deadline(deadline: float) -> None:
    if time.monotonic() > deadline:
        raise _SemanticTimeout


def _semantic_graph(
    source: KnowledgeSource,
    target: KnowledgeSource,
    anchor_rows: list[dict[str, Any]],
    *,
    deadline: float,
) -> dict[_Node, list[tuple[_Node, float, str]]]:
    adjacency: dict[_Node, list[tuple[_Node, float, str]]] = defaultdict(list)
    for side, knowledge in (("src", source), ("tgt", target)):
        for kind in sorted(_SUPPORTED_SEMANTIC_KINDS, key=lambda value: value.value):
            for iri_value in knowledge.entities(kind):
                _check_deadline(deadline)
                iri = str(iri_value)
                node = (side, kind.value, iri)
                adjacency.setdefault(node, [])
                for parent_value in knowledge.direct_parents(iri, kind):
                    parent = str(parent_value)
                    parent_node = (side, kind.value, parent)
                    adjacency[node].append((parent_node, 1.0, "hierarchy"))
                    adjacency.setdefault(parent_node, [])

    for anchor in anchor_rows:
        _check_deadline(deadline)
        kind = anchor["src_kind"]
        src_node = ("src", kind.value, anchor["src"])
        tgt_node = ("tgt", kind.value, anchor["tgt"])
        weight = max(0.0, min(1.0, float(anchor["score"])))
        evidence = f"anchor:{anchor['origin']}"
        adjacency[src_node].append((tgt_node, weight, evidence))
        adjacency[tgt_node].append((src_node, weight, evidence))
    for node in adjacency:
        adjacency[node].sort(key=lambda edge: (edge[0], edge[2], -edge[1]))
    return dict(adjacency)


def _widest_path(
    adjacency: dict[_Node, list[tuple[_Node, float, str]]],
    start: _Node,
    goal: _Node,
    *,
    deadline: float,
) -> tuple[float, list[_Node], list[str]]:
    if start == goal:
        return 1.0, [start], []
    best: dict[_Node, float] = {start: 1.0}
    predecessor: dict[_Node, tuple[_Node, str]] = {}
    pending: list[tuple[float, _Node]] = [(-1.0, start)]
    while pending:
        _check_deadline(deadline)
        neg_width, node = heapq.heappop(pending)
        width = -neg_width
        if width + 1.0e-15 < best.get(node, 0.0):
            continue
        if node == goal:
            break
        for neighbor, edge_weight, evidence in adjacency.get(node, ()):
            candidate = min(width, edge_weight)
            if candidate <= best.get(neighbor, -1.0) + 1.0e-15:
                continue
            best[neighbor] = candidate
            predecessor[neighbor] = (node, evidence)
            heapq.heappush(pending, (-candidate, neighbor))
    if goal not in best:
        return 0.0, [], []
    path = [goal]
    evidence_path: list[str] = []
    current = goal
    while current != start:
        parent, evidence = predecessor[current]
        path.append(parent)
        evidence_path.append(evidence)
        current = parent
    path.reverse()
    evidence_path.reverse()
    return best[goal], path, evidence_path


def _path_payload(
    width: float,
    path: list[_Node],
    evidence: list[str],
) -> dict[str, Any]:
    return {
        "confidence": width,
        "nodes": [{"side": side, "kind": kind, "iri": iri} for side, kind, iri in path],
        "edges": evidence,
    }


def _semantic_entailment(
    frame: pd.DataFrame,
    source: KnowledgeSource,
    target: KnowledgeSource,
    *,
    anchors: Any | None,
    anchor_threshold: float,
    anchor_margin: float,
    relation_threshold: float,
    timeout_seconds: float,
) -> pd.DataFrame:
    deadline = time.monotonic() + max(0.001, float(timeout_seconds))
    anchor_rows = _semantic_anchor_rows(
        frame,
        source,
        target,
        anchors=anchors,
        threshold=anchor_threshold,
        margin=anchor_margin,
    )
    try:
        adjacency = _semantic_graph(source, target, anchor_rows, deadline=deadline)
    except _SemanticTimeout:
        empty = frame.iloc[0:0].copy()
        empty["relation_confidence"] = pd.Series(dtype=float)
        empty["relation_semantic_backend"] = pd.Series(dtype=str)
        empty["relation_evidence"] = pd.Series(dtype=str)
        empty.attrs["relation_abstentions"] = [
            {
                "source": str(row.SrcEntity),
                "target": str(row.TgtEntity),
                "reason": "reasoning_timeout",
            }
            for row in frame.itertuples(index=False)
        ]
        empty.attrs["relation_anchor_count"] = len(anchor_rows)
        return empty

    kept_indices: list[Any] = []
    relations: list[str] = []
    confidences: list[float] = []
    evidence_rows: list[str] = []
    timed_out = False
    abstentions: list[dict[str, Any]] = []
    for index, row in frame.iterrows():
        src = str(row["SrcEntity"])
        tgt = str(row["TgtEntity"])
        if timed_out:
            abstentions.append(
                {
                    "source": src,
                    "target": tgt,
                    "reason": "reasoning_timeout",
                }
            )
            continue
        try:
            _check_deadline(deadline)
        except _SemanticTimeout:
            timed_out = True
            abstentions.append(
                {
                    "source": src,
                    "target": tgt,
                    "reason": "reasoning_timeout",
                }
            )
            continue
        src_kind, tgt_kind = _row_kinds(frame, index)
        if src_kind != tgt_kind or src_kind not in _SUPPORTED_SEMANTIC_KINDS:
            abstentions.append(
                {
                    "source": src,
                    "target": tgt,
                    "reason": "unsupported_kind",
                }
            )
            continue
        src_node = ("src", src_kind.value, src)
        tgt_node = ("tgt", tgt_kind.value, tgt)
        try:
            forward_conf, forward_path, forward_edges = _widest_path(
                adjacency, src_node, tgt_node, deadline=deadline
            )
            reverse_conf, reverse_path, reverse_edges = _widest_path(
                adjacency, tgt_node, src_node, deadline=deadline
            )
        except _SemanticTimeout:
            timed_out = True
            abstentions.append(
                {
                    "source": src,
                    "target": tgt,
                    "reason": "reasoning_timeout",
                }
            )
            continue
        forward = forward_conf >= relation_threshold
        reverse = reverse_conf >= relation_threshold
        if forward and reverse:
            relation = "="
            confidence = min(forward_conf, reverse_conf)
        elif forward:
            relation = "<"
            confidence = forward_conf
        elif reverse:
            relation = ">"
            confidence = reverse_conf
        else:
            abstentions.append(
                {
                    "source": src,
                    "target": tgt,
                    "reason": "not_entailed",
                    "forward_confidence": forward_conf,
                    "reverse_confidence": reverse_conf,
                }
            )
            continue
        kept_indices.append(index)
        relations.append(relation)
        confidences.append(confidence)
        evidence_rows.append(
            json.dumps(
                {
                    "backend": "graph_closure",
                    "relation": relation,
                    "forward": _path_payload(forward_conf, forward_path, forward_edges),
                    "reverse": _path_payload(reverse_conf, reverse_path, reverse_edges),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )

    result = frame.loc[kept_indices].copy()
    result["Relation"] = relations
    result["relation_confidence"] = confidences
    result["relation_semantic_backend"] = "graph_closure"
    result["relation_evidence"] = evidence_rows
    result.attrs["relation_abstentions"] = abstentions
    result.attrs["relation_anchor_count"] = len(anchor_rows)
    result.attrs["relation_anchors"] = [
        {
            "source": row["src"],
            "target": row["tgt"],
            "confidence": row["score"],
            "candidate_score": row.get("candidate_score", row["score"]),
            "origin": row["origin"],
            "kind": row["src_kind"].value,
        }
        for row in anchor_rows
    ]
    return result


def predict_relations(
    candidates: Any,
    source: KnowledgeSource,
    target: KnowledgeSource,
    *,
    mode: str = "none",
    anchors: Any | None = None,
    semantic_backend: str = "graph_closure",
    equivalence_anchor_threshold: float = 0.95,
    equivalence_anchor_margin: float = 0.10,
    relation_confidence_threshold: float = 0.5,
    timeout_seconds: float = 60.0,
) -> pd.DataFrame:
    """Type accepted pairs while preserving the shipped all-equivalent default.

    semantic_entailment implements the clarification's portable named
    class/property graph closure. Unresolved or unsupported rows abstain and are
    omitted; their reasons remain available in DataFrame.attrs. The optional
    bridge reasoner and fitted learned relation head fail closed until a real
    implementation/artifact is supplied.
    """

    frame = validated_mapping_frame(candidates)
    normalized_mode = str(mode).strip().lower()
    if normalized_mode == "none":
        frame["Relation"] = "="
        frame["relation_confidence"] = 1.0
        return frame
    if normalized_mode == "hierarchy_heuristic":
        return _hierarchy_heuristic(frame, source, target, anchors)
    if normalized_mode == "semantic_entailment":
        backend = str(semantic_backend).strip().lower()
        if backend == "bridge_reasoner":
            raise WriterOptionsError(
                "relation semantic backend 'bridge_reasoner' is deferred; "
                "use graph_closure or provide the optional reasoner bridge"
            )
        if backend != "graph_closure":
            raise WriterOptionsError(
                "relation semantic backend must be graph_closure or bridge_reasoner"
            )
        if source is None or target is None:
            raise WriterOptionsError("semantic relation typing requires source and target graphs")
        if not 0.0 <= float(equivalence_anchor_threshold) <= 1.0:
            raise WriterOptionsError("equivalence anchor threshold must be in [0, 1]")
        if not 0.0 <= float(equivalence_anchor_margin) <= 1.0:
            raise WriterOptionsError("equivalence anchor margin must be in [0, 1]")
        if not 0.0 <= float(relation_confidence_threshold) <= 1.0:
            raise WriterOptionsError("relation confidence threshold must be in [0, 1]")
        return _semantic_entailment(
            frame,
            source,
            target,
            anchors=anchors,
            anchor_threshold=float(equivalence_anchor_threshold),
            anchor_margin=float(equivalence_anchor_margin),
            relation_threshold=float(relation_confidence_threshold),
            timeout_seconds=float(timeout_seconds),
        )
    if normalized_mode in {"learned_three_way", "semantic_then_learned"}:
        raise WriterOptionsError(
            f"relation mode {normalized_mode!r} requires a fitted multinomial relation artifact; "
            "refusing to execute hierarchy_heuristic as a substitute"
        )
    raise WriterOptionsError(
        "relation prediction must be none, hierarchy_heuristic, semantic_entailment, "
        "learned_three_way, or semantic_then_learned"
    )


__all__ = ["predict_relations"]
