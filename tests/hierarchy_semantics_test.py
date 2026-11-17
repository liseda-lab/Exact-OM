"""E09 controls change hierarchy evidence without inventing logical contradictions."""

from copy import deepcopy

import pandas as pd
import pytest
import torch

from exact.core.entities.kinds import EntityKind
from exact.impl.datasets.pair_adaptive_context import PairAdaptiveContextDataset
from tests import kind_evidence_controls_test
from tests.pair_adaptive_experiments_test import _scorer, _TinyDataset

dataset = kind_evidence_controls_test.dataset
SRC = kind_evidence_controls_test.SRC
TGT = "http://example.org/mini/tgt#"


def _fixed_embeddings(monkeypatch, scorer):
    scorer.use_context = scorer.use_lexical = True
    monkeypatch.setattr(scorer, "encode_labels_batch", lambda values: torch.ones((len(values), 2)))
    for name in ("_encode_label_matrix", "_encode_context_matrix"):
        monkeypatch.setattr(
            scorer, name, lambda left, right: torch.full((len(left), len(right)), 0.8)
        )
    monkeypatch.setattr(scorer, "_context_similarity_from_sentences", lambda *args: 0.8)


def test_sibling_mismatch_requires_anchored_siblings_and_is_not_disjointness(monkeypatch):
    data = _TinyDataset()
    data.target.parents["tt"] = ["other-parent"]
    data.target._entities.append("other-parent")
    scorer = _scorer(hier={"enabled": True, "mode": "labels_overlap", "siblings": True})
    _fixed_embeddings(monkeypatch, scorer)
    scorer.attach_dataset(data)
    source = [{"triple": ("source", "is_a", "source parent"), "specificity": 1.0}]
    target = [{"triple": ("target", "is_a", "target parent"), "specificity": 1.0}]
    mismatch = scorer._score_hierarchy_family("is_a", source, target, "s", "t")
    assert mismatch["sibling_conflict"] == mismatch["sibling_coverage"] == 1.0
    scorer.hier_config["siblings"] = False
    ancestor_only = scorer._score_hierarchy_family("is_a", source, target, "s", "t")
    assert 0.0 < mismatch["score"] < ancestor_only["score"]
    assert data.target.parents["t"] == ["ta"]  # No inferred contradiction or graph mutation.

    data.exact_matches = pd.DataFrame({"Src": ["sa"], "Tgt": ["ta"]})
    scorer.attach_dataset(data)
    scorer.hier_config["siblings"] = True
    unanchored = scorer._score_hierarchy_family("is_a", source, target, "s", "t")
    assert unanchored["sibling_coverage"] == unanchored["sibling_conflict"] == 0.0
    assert unanchored["score"] == ancestor_only["score"]


def test_hierarchy_removal_preserves_selected_labels_and_attribute_evidence(dataset, monkeypatch):
    original = deepcopy(dataset.get_entity_features(SRC + "Heart", "src"))
    records = []
    for mode in ("labels", "off"):
        scorer = _scorer(return_explanations=True, hier={"enabled": True, "mode": mode})
        _fixed_embeddings(monkeypatch, scorer)
        scorer.attach_dataset(dataset)
        result = scorer(
            src_iris=[SRC + "Heart"],
            tgt_iris=[TGT + "CardiacOrgan"],
            src_label_lists=[original["labels"]],
            tgt_label_lists=[dataset.get_entity_features(TGT + "CardiacOrgan", "tgt")["labels"]],
        )
        records.append(result["explanations"][0])
    baseline, removed = records
    assert baseline["contributions"]["C_hier"] > 0.0
    assert removed["contributions"]["C_hier"] == 0.0
    for section, keys in (
        ("confidences", ["s_label", "s_attr"]),
        ("qualities", ["q_label", "q_attr"]),
    ):
        for key in keys:
            assert baseline[section][key] == removed[section][key]
    assert baseline["selected_labels"] == removed["selected_labels"]
    assert baseline["attributes"] == removed["attributes"]
    assert (
        baseline["triple_attributions"]["hierarchy"] == removed["triple_attributions"]["hierarchy"]
    )
    assert dataset.get_entity_features(SRC + "Heart", "src") == original


@pytest.mark.parametrize("wrong_alias", [False, True])
def test_part_of_reversal_and_wrong_alias_are_distinguishable(dataset, tmp_path, wrong_alias):
    part_of = "http://purl.obolibrary.org/obo/BFO_0000050"
    families = {
        "part_of": {"iri_aliases": [SRC + "hasPart" if wrong_alias else part_of]},
        "has_part": {"iri_aliases": [part_of if wrong_alias else SRC + "hasPart"]},
    }
    controlled = PairAdaptiveContextDataset(
        output_path=tmp_path / "families",
        cache_ok=False,
        verbaliser_name=None,
        hierarchical_relation_families=families,
    )
    controlled._source = controlled._target = dataset.source
    forward = controlled.get_entity_features(SRC + "ChestPain", "src", EntityKind.CLASS)[
        "hierarchy"
    ]
    reversed_query = controlled.get_entity_features(SRC + "Heart", "src", EntityKind.CLASS)[
        "hierarchy"
    ]
    expected_family = "has_part" if wrong_alias else "part_of"
    targets = {item["object_iri"] for item in forward[expected_family]}
    assert SRC + "Heart" in targets
    assert not any(
        item["object_iri"] == SRC + "ChestPain" for item in reversed_query.get(expected_family, [])
    )
    assert not forward.get("part_of" if wrong_alias else "has_part")
