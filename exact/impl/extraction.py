"""Deterministic post-selector extraction strategies.

The helpers in this module are intentionally independent of the trainer.  They
operate on scored :class:`~exact.core.entities.mappings.entity.EntityMapping`
objects so experiment orchestration can apply the score threshold at the
strategy-appropriate point and keep exact matches protected.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from exact.core.entities.mappings.entity import EntityMapping

Node = Tuple[str, str]
EdgeKey = Tuple[Node, Node]
Pair = Tuple[str, str]


@dataclass(frozen=True)
class ExtractionResult:
    """Selected mappings plus small, serialisable extraction diagnostics."""

    mappings: List[EntityMapping]
    diagnostics: Mapping[str, object]


def _kind_text(value: object) -> str:
    return str(getattr(value, "value", value))


def _source_node(mapping: EntityMapping) -> Node:
    return str(mapping.head), _kind_text(mapping.src_kind)


def _target_node(mapping: EntityMapping) -> Node:
    return str(mapping.tail), _kind_text(mapping.tgt_kind)


def _edge_key(mapping: EntityMapping) -> EdgeKey:
    return _source_node(mapping), _target_node(mapping)


def _mapping_order_key(mapping: EntityMapping) -> Tuple[object, ...]:
    return (
        _kind_text(mapping.src_kind),
        str(mapping.head),
        _kind_text(mapping.tgt_kind),
        str(mapping.tail),
        -float(mapping.score),
        str(mapping.relation),
    )


def _deduplicate(mappings: Iterable[EntityMapping]) -> List[EntityMapping]:
    """Keep one deterministic highest-scoring object for each typed edge."""

    best: Dict[EdgeKey, EntityMapping] = {}
    for mapping in mappings:
        key = _edge_key(mapping)
        incumbent = best.get(key)
        if incumbent is None or float(mapping.score) > float(incumbent.score):
            best[key] = mapping
        elif float(mapping.score) == float(incumbent.score):
            if _mapping_order_key(mapping) < _mapping_order_key(incumbent):
                best[key] = mapping
    return sorted(best.values(), key=_mapping_order_key)


def _preassign_exact(
    mappings: Sequence[EntityMapping],
    protected_pairs: Set[Pair],
) -> Tuple[List[EntityMapping], List[EntityMapping]]:
    protected = _deduplicate(
        mapping for mapping in mappings if (str(mapping.head), str(mapping.tail)) in protected_pairs
    )
    occupied_sources = {_source_node(mapping) for mapping in protected}
    occupied_targets = {_target_node(mapping) for mapping in protected}
    residual = [
        mapping
        for mapping in _deduplicate(mappings)
        if (str(mapping.head), str(mapping.tail)) not in protected_pairs
        and _source_node(mapping) not in occupied_sources
        and _target_node(mapping) not in occupied_targets
    ]
    return protected, residual


def _validate_protected_exact_constraints(
    mappings: Sequence[EntityMapping],
    protected_pairs: Set[Pair],
) -> None:
    """Reject contradictory exact-match hard constraints before extraction.

    E01 treats protected exact matches as pre-assigned one-to-one edges. Letting
    two protected edges occupy the same typed source or target would make the
    optimization infeasible and used to leak a non-one-to-one result out of
    every global extraction mode. Only protected pairs present in the input
    graph participate: a protected pair may have been consumed by an earlier
    exact-match stage and therefore need not be present here.
    """

    present = _deduplicate(
        mapping for mapping in mappings if (str(mapping.head), str(mapping.tail)) in protected_pairs
    )
    targets_by_source: Dict[Node, Set[Node]] = defaultdict(set)
    sources_by_target: Dict[Node, Set[Node]] = defaultdict(set)
    for mapping in present:
        source = _source_node(mapping)
        target = _target_node(mapping)
        targets_by_source[source].add(target)
        sources_by_target[target].add(source)

    source_conflicts = {
        source: tuple(sorted(targets))
        for source, targets in targets_by_source.items()
        if len(targets) > 1
    }
    target_conflicts = {
        target: tuple(sorted(sources))
        for target, sources in sources_by_target.items()
        if len(sources) > 1
    }
    if not source_conflicts and not target_conflicts:
        return

    details: List[str] = []
    for source, targets in sorted(source_conflicts.items()):
        details.append(f"source {source!r} -> {list(targets)!r}")
    for target, sources in sorted(target_conflicts.items()):
        details.append(f"target {target!r} <- {list(sources)!r}")
    raise ValueError(
        "Conflicting protected exact matches violate one-to-one extraction: " + "; ".join(details)
    )


def _threshold_selected(
    selected: Sequence[EntityMapping],
    protected_pairs: Set[Pair],
    threshold: Optional[float],
) -> Tuple[List[EntityMapping], int]:
    if threshold is None:
        return list(selected), 0
    kept: List[EntityMapping] = []
    removed = 0
    for mapping in selected:
        pair = (str(mapping.head), str(mapping.tail))
        if pair in protected_pairs or float(mapping.score) >= float(threshold):
            kept.append(mapping)
        else:
            removed += 1
    return kept, removed


def _mutual_best(mappings: Sequence[EntityMapping]) -> List[EntityMapping]:
    source_edges: Dict[Node, List[EntityMapping]] = defaultdict(list)
    target_edges: Dict[Node, List[EntityMapping]] = defaultdict(list)
    for mapping in mappings:
        source_edges[_source_node(mapping)].append(mapping)
        target_edges[_target_node(mapping)].append(mapping)

    source_best = {
        _edge_key(
            min(
                edges,
                key=lambda mapping: (
                    -float(mapping.score),
                    _target_node(mapping),
                    str(mapping.relation),
                ),
            )
        )
        for edges in source_edges.values()
    }
    target_best = {
        _edge_key(
            min(
                edges,
                key=lambda mapping: (
                    -float(mapping.score),
                    _source_node(mapping),
                    str(mapping.relation),
                ),
            )
        )
        for edges in target_edges.values()
    }
    return sorted(
        [mapping for mapping in mappings if _edge_key(mapping) in source_best & target_best],
        key=_mapping_order_key,
    )


def _stable_marriage(mappings: Sequence[EntityMapping]) -> List[EntityMapping]:
    edge_lookup = {_edge_key(mapping): mapping for mapping in mappings}
    source_preferences: Dict[Node, List[Node]] = defaultdict(list)
    target_preferences: Dict[Node, List[Node]] = defaultdict(list)
    for mapping in mappings:
        source_preferences[_source_node(mapping)].append(_target_node(mapping))
        target_preferences[_target_node(mapping)].append(_source_node(mapping))

    for source, targets in source_preferences.items():
        targets.sort(
            key=lambda target: (
                -float(edge_lookup[(source, target)].score),
                target,
            )
        )
    target_rank: Dict[Node, Dict[Node, int]] = {}
    for target, sources in target_preferences.items():
        sources.sort(
            key=lambda source: (
                -float(edge_lookup[(source, target)].score),
                source,
            )
        )
        target_rank[target] = {source: rank for rank, source in enumerate(sources)}

    free = deque(sorted(source_preferences))
    next_choice = {source: 0 for source in source_preferences}
    target_match: Dict[Node, Node] = {}
    while free:
        source = free.popleft()
        preferences = source_preferences[source]
        offset = next_choice[source]
        if offset >= len(preferences):
            continue
        target = preferences[offset]
        next_choice[source] = offset + 1
        incumbent = target_match.get(target)
        if incumbent is None:
            target_match[target] = source
            continue
        ranks = target_rank[target]
        if ranks[source] < ranks[incumbent]:
            target_match[target] = source
            if next_choice[incumbent] < len(source_preferences[incumbent]):
                free.append(incumbent)
        elif next_choice[source] < len(preferences):
            free.append(source)

    selected = [edge_lookup[(source, target)] for target, source in target_match.items()]
    return sorted(selected, key=_mapping_order_key)


def _connected_components(mappings: Sequence[EntityMapping]) -> List[List[EntityMapping]]:
    edge_lookup: Dict[EdgeKey, EntityMapping] = {
        _edge_key(mapping): mapping for mapping in mappings
    }
    source_to_targets: Dict[Node, Set[Node]] = defaultdict(set)
    target_to_sources: Dict[Node, Set[Node]] = defaultdict(set)
    for source, target in edge_lookup:
        source_to_targets[source].add(target)
        target_to_sources[target].add(source)

    components: List[List[EntityMapping]] = []
    seen_sources: Set[Node] = set()
    seen_targets: Set[Node] = set()
    for initial_source in sorted(source_to_targets):
        if initial_source in seen_sources:
            continue
        queue = deque([("source", initial_source)])
        component_sources: Set[Node] = set()
        component_targets: Set[Node] = set()
        while queue:
            side, node = queue.popleft()
            if side == "source":
                if node in seen_sources:
                    continue
                seen_sources.add(node)
                component_sources.add(node)
                for target in sorted(source_to_targets.get(node, set())):
                    if target not in seen_targets:
                        queue.append(("target", target))
            else:
                if node in seen_targets:
                    continue
                seen_targets.add(node)
                component_targets.add(node)
                for source in sorted(target_to_sources.get(node, set())):
                    if source not in seen_sources:
                        queue.append(("source", source))
        components.append(
            sorted(
                [
                    mapping
                    for (source, target), mapping in edge_lookup.items()
                    if source in component_sources and target in component_targets
                ],
                key=_mapping_order_key,
            )
        )
    return components


def _assignment_component(mappings: Sequence[EntityMapping]) -> List[EntityMapping]:
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:  # pragma: no cover - exercised when optional SciPy is absent
        raise RuntimeError(
            "matching.extraction.mode='assignment' requires SciPy; install scipy or choose "
            "mutual_best/stable_marriage."
        ) from exc

    edge_lookup = {_edge_key(mapping): mapping for mapping in mappings}
    sources = sorted({source for source, _ in edge_lookup})
    targets = sorted({target for _, target in edge_lookup})
    if not sources or not targets:
        return []

    # One private zero-valued dummy target per source permits a maximum-weight
    # partial matching instead of forcing nonexistent or negative-score edges.
    missing_cost = 1.0e12
    costs = np.full((len(sources), len(targets) + len(sources)), missing_cost, dtype=float)
    costs[:, len(targets) :] = 0.0
    for source_idx, source in enumerate(sources):
        for target_idx, target in enumerate(targets):
            mapping = edge_lookup.get((source, target))
            if mapping is not None:
                costs[source_idx, target_idx] = -float(mapping.score)

    row_indices, column_indices = linear_sum_assignment(costs)
    selected: List[EntityMapping] = []
    for row_idx, column_idx in zip(row_indices.tolist(), column_indices.tolist()):
        if column_idx >= len(targets):
            continue
        mapping = edge_lookup.get((sources[row_idx], targets[column_idx]))
        if mapping is not None:
            selected.append(mapping)
    return sorted(selected, key=_mapping_order_key)


def extract_global_alignment(
    mappings: Sequence[EntityMapping],
    *,
    mode: str = "greedy",
    threshold: Optional[float] = None,
    protected_pairs: Optional[Set[Pair]] = None,
    source_cardinality: Optional[int] = 1,
    target_cardinality: Optional[int] = 1,
    assignment_component_cap: int = 500,
) -> ExtractionResult:
    """Extract a global alignment while preserving protected exact pairs.

    Non-greedy modes are one-to-one strategies.  They select on the complete
    score graph and apply ``threshold`` afterwards, which is important for the
    assignment arm.  ``greedy`` retains the shipped threshold-then-cascade
    order.
    """

    normalized_mode = str(mode or "greedy").strip().lower()
    if normalized_mode not in {"greedy", "mutual_best", "stable_marriage", "assignment"}:
        raise ValueError(f"Unknown extraction mode: {mode!r}")
    if assignment_component_cap < 1:
        raise ValueError("assignment_component_cap must be at least one")

    protected_set = {(str(source), str(target)) for source, target in (protected_pairs or set())}
    _validate_protected_exact_constraints(mappings, protected_set)
    protected, residual = _preassign_exact(list(mappings), protected_set)
    threshold_removed = 0
    component_count = 0
    fallback_components = 0

    if normalized_mode == "greedy":
        eligible, threshold_removed = _threshold_selected(residual, set(), threshold)
        selected = list(eligible)
        if source_cardinality is not None:
            selected = EntityMapping.filter_top_n_entity_mappings(
                selected, max(1, int(source_cardinality))
            )
        if target_cardinality is not None:
            selected = EntityMapping.filter_top_n_target_entity_mappings(
                selected, max(1, int(target_cardinality))
            )
    elif normalized_mode == "mutual_best":
        selected = _mutual_best(residual)
        selected, threshold_removed = _threshold_selected(selected, set(), threshold)
    elif normalized_mode == "stable_marriage":
        selected = _stable_marriage(residual)
        selected, threshold_removed = _threshold_selected(selected, set(), threshold)
    else:
        selected = []
        components = _connected_components(residual)
        component_count = len(components)
        for component in components:
            source_count = len({_source_node(mapping) for mapping in component})
            target_count = len({_target_node(mapping) for mapping in component})
            if source_count + target_count > int(assignment_component_cap):
                selected.extend(_mutual_best(component))
                fallback_components += 1
            else:
                selected.extend(_assignment_component(component))
        selected, threshold_removed = _threshold_selected(selected, set(), threshold)

    output = _deduplicate([*protected, *selected])
    diagnostics: Dict[str, object] = {
        "mode": normalized_mode,
        "input_mappings": len(mappings),
        "deduplicated_mappings": len(_deduplicate(mappings)),
        "protected_mappings": len(protected),
        "selected_mappings": len(output),
        "threshold": threshold,
        "threshold_removed": threshold_removed,
        "assignment_components": component_count,
        "assignment_fallback_components": fallback_components,
        "assignment_component_cap": int(assignment_component_cap),
    }
    return ExtractionResult(mappings=output, diagnostics=diagnostics)


def extract_mappings(
    mappings: Sequence[EntityMapping],
    **kwargs: Any,
) -> List[EntityMapping]:
    """List-only convenience wrapper around :func:`extract_global_alignment`."""

    return extract_global_alignment(mappings, **kwargs).mappings


__all__ = ["ExtractionResult", "extract_global_alignment", "extract_mappings"]
