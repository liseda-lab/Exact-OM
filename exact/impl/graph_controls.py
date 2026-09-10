"""Deterministic graph perturbations shared by the bounded E12/E23 screens."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from typing import Iterable, Mapping

from exact.core.entities.graph import Edge


def graph_fingerprint(edges: Iterable[Edge]) -> str:
    rows = sorted(set(edge.astuple() for edge in edges))
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def shuffle_relations(
    edges: Iterable[Edge],
    *,
    seed: int,
    kinds: Mapping[str, str] | None = None,
    swaps_per_edge: int = 5,
) -> tuple[list[Edge], dict]:
    """Swap endpoints within predicates and endpoint kinds, preserving node degrees.

    Graphs use set semantics. No duplicate edge or new self-loop is admitted.
    Constrained graphs may resist swaps; the observed change fraction is reported
    rather than claiming that every graph loses its alignment signal.
    """
    if swaps_per_edge < 0:
        raise ValueError("swaps_per_edge must be non-negative")
    original = sorted(set(edges), key=Edge.astuple)
    current = set(original)
    by_predicate = defaultdict(list)
    kinds = kinds or {}
    for edge in original:
        by_predicate[(edge.rel, kinds.get(edge.src), kinds.get(edge.dst))].append(edge)
    rng = random.Random(int(seed))
    swaps = attempts = 0
    for key in sorted(by_predicate, key=str):
        group = by_predicate[key]
        if len(group) < 2:
            continue
        for _ in range(swaps_per_edge * len(group)):
            attempts += 1
            first, second = rng.sample(range(len(group)), 2)
            left, right = group[first], group[second]
            a, b = Edge(left.src, left.rel, right.dst), Edge(right.src, right.rel, left.dst)
            if a.src == a.dst or b.src == b.dst or a in current or b in current:
                continue
            current.difference_update((left, right))
            current.update((a, b))
            group[first], group[second] = a, b
            swaps += 1
    output = sorted(current, key=Edge.astuple)
    return output, {
        "mode": "predicate_degree_preserving",
        "seed": int(seed),
        "input_sha256": graph_fingerprint(original),
        "output_sha256": graph_fingerprint(output),
        "edges": len(original),
        "attempts": attempts,
        "accepted_swaps": swaps,
        "changed_edges": len(set(output) - set(original)),
        "preserved": [
            "predicate_count",
            "per_predicate_in_degree",
            "per_predicate_out_degree",
            "endpoint_kinds",
        ],
    }


def remove_hierarchy(
    edges: Iterable[Edge],
    *,
    fraction: float,
    seed: int,
    hierarchy_predicates: Iterable[str],
) -> tuple[list[Edge], dict]:
    """Remove a nested hash-ranked subset; labels and entity inventories are external."""
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("hierarchy removal fraction must be in [0, 1]")
    original = sorted(set(edges), key=Edge.astuple)
    predicates = set(hierarchy_predicates)
    hierarchy = [edge for edge in original if edge.rel in predicates]
    hierarchy.sort(
        key=lambda edge: (
            hashlib.sha256(
                json.dumps([int(seed), *edge.astuple()], separators=(",", ":")).encode()
            ).digest(),
            edge.astuple(),
        )
    )
    removed = set(hierarchy[: math.ceil(fraction * len(hierarchy))])
    output = [edge for edge in original if edge not in removed]
    return output, {
        "mode": "hierarchy_removal",
        "fraction": fraction,
        "seed": int(seed),
        "hierarchy_predicates": sorted(predicates),
        "hierarchy_edges": len(hierarchy),
        "removed": [edge.astuple() for edge in sorted(removed, key=Edge.astuple)],
        "input_sha256": graph_fingerprint(original),
        "output_sha256": graph_fingerprint(output),
    }


def hierarchy_control_view(dataset, *, fraction: float, seed: int, hierarchy_predicates):
    """Isolate scoring evidence from retrieval while removing the same asserted edges."""
    from copy import copy

    class SourceView:
        def __init__(self, source, removed):
            self.source, self.removed = source, removed

        def __getattr__(self, name):
            return getattr(self.source, name)

        def direct_parents(self, iri, kind):
            return [
                parent
                for parent in self.source.direct_parents(iri, kind)
                if (iri, parent) not in self.removed
            ]

        def direct_children(self, iri, kind):
            return [
                child
                for child in self.source.direct_children(iri, kind)
                if (child, iri) not in self.removed
            ]

        def hierarchy_bundle(self, iri, families):
            return {
                family: [target for target in targets if (iri, target) not in self.removed]
                for family, targets in self.source.hierarchy_bundle(iri, families).items()
            }

    view = copy(dataset)
    for field in (
        "_entity_feature_cache",
        "_direct_superclass_cache",
        "_hierarchy_axiom_targets_cache",
        "_relation_family_cache",
        "_property_extra_cache",
        "_shuffled_graph_cache",
    ):
        setattr(view, field, {})
    view.graph_control_manifests = {}
    for side in ("src", "tgt"):
        original = dataset.source_graph if side == "src" else dataset.target_graph
        edges, manifest = remove_hierarchy(
            original.edges or [],
            fraction=fraction,
            seed=seed,
            hierarchy_predicates=hierarchy_predicates,
        )
        graph = copy(original)
        graph.edges = edges
        graph.out_edges = graph._build_out_edges()
        graph.graph = graph._build_graph(edges)
        for field in (
            "_node_ic_cache",
            "_edge_ic_cache",
            "_edge_ic_max_cache",
            "_incident_edge_cache",
            "_rel_adj_cache",
            "_edge_cost_cache",
            "_example_triples_cache",
        ):
            setattr(graph, field, None)
        graph._raw_neighborhood_cache = {}
        graph._context_subgraph_cache = {}
        removed = {(src, tgt) for src, _, tgt in manifest["removed"]}
        source = dataset.source if side == "src" else dataset.target
        setattr(view, "_source" if side == "src" else "_target", SourceView(source, removed))
        setattr(view, "_source_graph" if side == "src" else "_target_graph", graph)
        view.graph_control_manifests[side + ":hierarchy_removal"] = manifest
    return view
