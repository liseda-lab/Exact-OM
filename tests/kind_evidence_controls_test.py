from collections import Counter
from pathlib import Path

import pytest
import torch

from exact.core.entities.graph import Edge
from exact.core.entities.kinds import EntityKind
from exact.impl.datasets.pair_adaptive_context import PairAdaptiveContextDataset
from exact.impl.graph_controls import remove_hierarchy, shuffle_relations
from tests.pair_adaptive_experiments_test import _scorer

SRC = "http://example.org/mini/src#"
FIXTURES = Path(__file__).parent / "fixtures" / "ontologies"


@pytest.fixture
def dataset(tmp_path):
    data = PairAdaptiveContextDataset(
        output_path=tmp_path,
        cache_ok=False,
        verbaliser_name=None,
        projection_include_literals=True,
        entity_kinds=["class", "object_property", "data_property", "individual"],
    )
    data.load_ontologies(FIXTURES / "mini_src.owl", FIXTURES / "mini_tgt.owl")
    return data


def test_property_signature_and_usage_are_controlled_without_mutating_cached_evidence(dataset):
    label_only = _scorer(
        property={
            "enabled": True,
            "annotations": False,
            "signature": False,
            "hierarchy": False,
            "characteristics": False,
            "usage": False,
        }
    )
    label_only.attach_dataset(dataset)
    labels = label_only._experiment_entity_features(SRC + "participatesIn", "src")
    assert labels["labels"] and labels["attributes"] == [] and labels["object_triples"] == []
    assert labels["hierarchy"] == {}
    signatures = _scorer(property={"enabled": True, "usage": False})
    signatures.attach_dataset(dataset)
    features = signatures._experiment_entity_features(SRC + "hasPart", "src")
    relations = {item["rel_iri"] for item in features["object_triples"]}
    assert {
        "http://www.w3.org/2000/01/rdf-schema#domain",
        "http://www.w3.org/2000/01/rdf-schema#range",
        "http://www.w3.org/2002/07/owl#inverseOf",
    } <= relations
    inverse = next(
        item for item in features["object_triples"] if item["rel_iri"].endswith("inverseOf")
    )
    assert inverse["axiom_id"] and inverse["property_schema"]
    full = _scorer(property={"enabled": True})
    full.attach_dataset(dataset)
    usage = full._experiment_entity_features(SRC + "participatesIn", "src")["object_triples"]
    assert any(item.get("evidence_group") == "usage" for item in usage)
    assert dataset.get_entity_features(SRC + "participatesIn", "src")["hierarchy"]


def test_instance_types_literals_relations_and_annotations_keep_provenance(dataset):
    scorer = _scorer(request_seed=17, instance={"enabled": True, "relations": False})
    scorer.attach_dataset(dataset)
    features = scorer._experiment_entity_features(SRC + "alice", "src")
    assert features["hierarchy"] and features["attributes"] and features["object_triples"] == []
    code = next(item for item in features["attributes"] if item["value"] == "P-001")
    assert code["prop_iri"] == SRC + "hasCode"
    scorer.instance_config.update(types=False, literals=False, relations=True)
    relations = scorer._experiment_entity_features(SRC + "alice", "src")
    assert relations["hierarchy"] == {} and relations["attributes"] == []
    assert relations["object_triples"]
    scorer.instance_config["relations_shuffled"] = True
    shuffled = scorer._experiment_entity_features(SRC + "alice", "src")
    assert shuffled["experiment_evidence"]["relations_shuffled"]
    assert dataset.graph_control_manifests
    assert dataset.entity_kind_for(SRC + "alice", "src") == EntityKind.INDIVIDUAL


def test_property_support_does_not_exchange_domain_and_range(monkeypatch):
    scorer = _scorer(property={"enabled": True})
    monkeypatch.setattr(
        scorer, "_encode_label_matrix", lambda left, right: torch.ones((len(left), len(right)))
    )
    source = [{"triple": ["p", "domain", "Organ"], "rel_iri": "domain", "property_schema": True}]
    target = [{"triple": ["q", "range", "Organ"], "rel_iri": "range", "property_schema": True}]
    assert scorer._object_support_matrix(source, target).item() == 0.0


def test_shuffle_preserves_predicate_node_degrees_and_removal_is_nested():
    edges = [Edge(f"s{i}", relation, f"t{i}") for i in range(12) for relation in ("r", "is_a")]
    shuffled, manifest = shuffle_relations(edges, seed=17)
    assert shuffled == shuffle_relations(reversed(edges), seed=17)[0]
    assert Counter((e.src, e.rel) for e in shuffled) == Counter((e.src, e.rel) for e in edges)
    assert Counter((e.dst, e.rel) for e in shuffled) == Counter((e.dst, e.rel) for e in edges)
    assert len(set(shuffled)) == len(edges) and manifest["changed_edges"] > 0
    half, removed_half = remove_hierarchy(
        edges, fraction=0.5, seed=17, hierarchy_predicates=["is_a"]
    )
    all_removed, removed_all = remove_hierarchy(
        edges, fraction=1.0, seed=17, hierarchy_predicates=["is_a"]
    )
    assert len(half) == 18 and len(all_removed) == 12
    assert set(removed_half["removed"]) <= set(removed_all["removed"])
    assert {edge for edge in edges if edge.rel == "r"} <= set(half)


def test_representation_inventory_detects_literal_metadata_loss(dataset):
    from dataclasses import replace

    from exact.experiments.evidence_inventory import (
        compare_evidence_inventories,
        evidence_inventory,
    )

    class LostLanguage:
        def __getattr__(self, name):
            return getattr(dataset.source, name)

        def attributes(self, iri):
            return [replace(value, lang="zxx") for value in dataset.source.attributes(iri)]

    original = evidence_inventory(dataset.source)
    changed = evidence_inventory(LostLanguage())
    assert compare_evidence_inventories(original, original)["evidence_parity"]
    difference = compare_evidence_inventories(original, changed)
    assert not difference["evidence_parity"]
    assert difference["differences"]["attributes"]["left_only"]
    assert difference["differences"]["projected_edges"] == {"left_only": [], "right_only": []}
    assert original["owl_axiom_counts"]
