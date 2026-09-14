"""Feature budgets must retain the same evidence across processes and projections."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from exact.core.entities.graph import Edge
from exact.core.entities.ontology import OntologyGraph
from exact.impl.datasets.pair_adaptive_context import PairAdaptiveContextDataset


class _Source:
    def __init__(self, edges):
        self.edges = edges

    def projection_edges(self, **kwargs):
        return self.edges

    def labels(self, iri):
        return [iri]

    def attributes(self, iri):
        return []

    def direct_parents(self, iri, kind):
        return []

    def hierarchy_bundle(self, iri, families):
        return {family: [] for family in families}


def _feature_row(reverse=False):
    edges = [Edge("urn:root", "urn:relation", f"urn:leaf{i:02}") for i in range(12)]
    source = _Source(list(reversed(edges)) if reverse else edges)
    graph = OntologyGraph(source)
    # Exercise the complete feature assembly with a pure source: no loader/model.
    dataset = object.__new__(PairAdaptiveContextDataset)
    settings = {
        "_source": source,
        "_source_graph": graph,
        "all_labels": True,
        "n_hops": 1,
        "hierarchy_max_depth": 1,
        "hierarchical_relation_families": {},
        "_normalized_relation_families": {},
        "_relation_family_cache": {},
        "_entity_feature_cache": {},
        "_direct_superclass_cache": {},
        "_hierarchy_axiom_targets_cache": {},
        "projection_include_literals": False,
        "max_object_triples": 4,
        "max_attr_items": 12,
        "max_hierarchy_triples_per_family": 6,
    }
    for name, value in settings.items():
        setattr(dataset, name, value)
    features = dataset.get_entity_features("urn:root", "src", "class")
    return ["class", "urn:root", features]


def test_full_features_are_identical_across_hash_seeds_and_edge_insertion():
    root = Path(__file__).resolve().parents[1]
    rows = []
    for seed in (1, 2, 3):
        process = subprocess.run(
            [sys.executable, str(Path(__file__).resolve())],
            cwd=root,
            env={
                **os.environ,
                "PYTHONHASHSEED": str(seed),
                "PYTHONPATH": os.pathsep.join([str(root), os.environ.get("PYTHONPATH", "")]),
            },
            text=True,
            capture_output=True,
            check=True,
        )
        rows.extend(json.loads(process.stdout))
    assert all(row == rows[0] for row in rows)
    assert [item["object_iri"] for item in rows[0][2]["object_triples"]] == [
        f"urn:leaf{i:02}" for i in range(4)
    ]
    assert {item["score"] for item in rows[0][2]["object_triples"]} == {1.0}
    # Match the benchmark's exact row serialization; do not normalize list order.
    assert (
        len(
            {
                hashlib.sha256(
                    (json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n").encode()
                ).hexdigest()
                for row in rows
            }
        )
        == 1
    )


def test_ordering_preserves_neighborhood_population_direction_and_cache_isolation():
    ab = Edge("urn:A", "urn:p", "urn:B")
    bc = Edge("urn:B", "urn:q", "urn:C")
    cd = Edge("urn:C", "urn:r", "urn:D")
    graph = OntologyGraph(_Source([cd, bc, ab, ab]))
    assert graph.get_raw_neighborhood("urn:A", 1) == [ab.astuple()]
    expected = [ab.astuple(), bc.astuple()]
    assert graph.get_raw_neighborhood("urn:A", 2) == expected
    assert graph.get_raw_neighborhood("urn:A", 2, include_reverse=False) == [ab.astuple()]
    returned = graph.get_raw_neighborhood("urn:A", 2)
    returned.clear()
    assert graph.get_raw_neighborhood("urn:A", 2) == expected
    assert graph.get_raw_neighborhood("urn:missing", 2) == []


def test_tie_order_correctness_change_invalidates_evidence_schema_2(monkeypatch):
    from exact.impl.datasets.contextgraph import ContextDataset

    monkeypatch.setattr(ContextDataset, "_cache_fingerprint_payload", lambda self: {})
    dataset = object.__new__(PairAdaptiveContextDataset)
    for name, value in {
        "projection_include_literals": False,
        "hierarchical_relation_families": {},
        "hierarchy_max_depth": 1,
        "max_hierarchy_triples_per_family": 6,
        "max_object_triples": 4,
        "max_diff_triples": 24,
        "max_attr_items": 12,
    }.items():
        setattr(dataset, name, value)
    current = dataset._cache_fingerprint_payload()
    prior = {**current, "evidence_schema": 2}
    assert current["evidence_schema"] == 3
    current_fingerprint = dataset.cache_fingerprint
    monkeypatch.setattr(dataset, "_cache_fingerprint_payload", lambda: prior)
    assert dataset.cache_fingerprint != current_fingerprint


if __name__ == "__main__":
    print(json.dumps([_feature_row(), _feature_row(reverse=True)]))
