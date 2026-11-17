import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from exact.core.entities.graph import Edge
from exact.impl.models.graph_head import (
    FEATURE_SCHEMA,
    fit_graph_artifact,
    graph_pair_features,
    graph_predictions,
    structural_profiles,
)
from tests import kind_evidence_controls_test
from tests.pair_adaptive_experiments_test import _scorer

dataset = kind_evidence_controls_test.dataset
SRC = kind_evidence_controls_test.SRC


def training_rows():
    rows = []
    for source in range(4):
        for matched in (False, True):
            rows.append(
                {
                    "Src": f"s{source}",
                    "Tgt": f"t{source}-{int(matched)}",
                    "graph_features": {
                        "values": [0.8 if matched else 0.2] * len(FEATURE_SCHEMA),
                        "active": True,
                        "graph_fingerprints": {"src": "a", "tgt": "b"},
                    },
                }
            )
    return pd.DataFrame(rows)


def test_graph_fit_is_grouped_replayable_and_exact_on_logit_scale(tmp_path, monkeypatch):
    rows = training_rows()
    refs = {(f"s{i}", f"t{i}-1") for i in range(4)}
    application = {
        "dataset_signature": "tiny",
        "source_ids": ["report"],
        "negative_label_policy": "complete_reference",
    }
    path = tmp_path / "head.json"
    artifact = fit_graph_artifact(rows, refs, path, application=application, seed=17)
    assert len(artifact["oof_predictions"]) == len(rows)
    assert all(
        not set(fold["training_sources"]) & set(fold["heldout_sources"])
        for fold in artifact["folds"]
    )
    probabilities, contributions, logits = graph_predictions(rows.graph_features, artifact)
    assert np.allclose(contributions.sum(axis=1) + artifact["bias"], logits)
    assert probabilities[1] > probabilities[0]
    assert len(list((tmp_path / "head.json.folds").glob("*.json"))) == 3
    original_minimize = __import__("exact.impl.models.graph_head", fromlist=["minimize"]).minimize
    calls = []

    def count_calls(*args, **kwargs):
        calls.append(True)
        return original_minimize(*args, **kwargs)

    monkeypatch.setattr("exact.impl.models.graph_head.minimize", count_calls)
    path.unlink()
    assert fit_graph_artifact(rows, refs, path, application=application, seed=17) == artifact
    assert len(calls) == 1  # Completed source folds survive final-head repair.
    bad = json.loads(json.dumps(artifact))
    bad["graph_fingerprints"]["src"] = "changed"
    with pytest.raises(ValueError, match="fingerprint"):
        graph_predictions(rows.graph_features, bad)
    with pytest.raises(ValueError, match="explicit"):
        fit_graph_artifact(
            rows,
            refs,
            tmp_path / "bad.json",
            application={**application, "negative_label_policy": None},
            seed=17,
        )


def test_structural_features_are_identifier_invariant_and_controlled():
    edges = [Edge(f"s{i}", "r", f"s{(i + 1) % 8}") for i in range(8)]
    renamed = [
        Edge(edge.src.replace("s", "x"), "other-predicate", edge.dst.replace("s", "x"))
        for edge in edges
    ]
    profiles, other = structural_profiles(edges), structural_profiles(renamed)
    assert all(
        np.array_equal(profile, other[iri.replace("s", "x")]) for iri, profile in profiles.items()
    )
    data = SimpleNamespace(
        source_graph=SimpleNamespace(edges=edges), target_graph=SimpleNamespace(edges=renamed)
    )
    real = graph_pair_features(data, ["s0", "missing"], ["x0", "x0"], {}, 17)
    assert real[0]["values"] == [1.0] * len(FEATURE_SCHEMA)
    assert not real[1]["active"]
    shuffled = graph_pair_features(data, ["s0"], ["x0"], {"shuffled": True}, 17)
    assert shuffled[0]["graph_fingerprints"]["src"]["shuffle"]["changed_edges"] > 0


def test_hierarchy_removal_changes_scoring_view_without_changing_frozen_inputs(dataset):
    original = dataset.get_entity_features(SRC + "Organism", "src")
    assert original["hierarchy"]
    scorer = _scorer(request_seed=17, graph={"mode": "off", "hierarchy_removal": 1.0})
    scorer.attach_dataset(dataset)
    removed = scorer._experiment_entity_features(SRC + "Organism", "src")
    assert not removed["hierarchy"]
    assert removed["labels"] == original["labels"]
    assert dataset.get_entity_features(SRC + "Organism", "src") == original
    assert scorer._attached_dataset.graph_control_manifests["src:hierarchy_removal"]["removed"]


def test_inductive_graph_channel_runs_and_reconstructs(dataset, tmp_path):
    targets = "http://example.org/mini/tgt#"
    source_iris = [SRC + "Organism", SRC + "Person", SRC + "Organ", SRC + "Heart"]
    target_iris = [
        targets + "LivingThing",
        targets + "Human",
        targets + "BodyOrgan",
        targets + "CardiacOrgan",
    ]
    candidates = [(source, target) for source in source_iris for target in target_iris]
    src, tgt = zip(*candidates)
    features = graph_pair_features(dataset, src, tgt, {}, 17)
    rows = pd.DataFrame({"Src": src, "Tgt": tgt, "graph_features": features})
    path = tmp_path / "graph.json"
    fit_graph_artifact(
        rows,
        set(zip(source_iris, target_iris)),
        path,
        application={
            "dataset_signature": dataset.dataset_signature,
            "source_ids": [],
            "negative_label_policy": "complete_reference",
        },
        seed=17,
    )
    scorer = _scorer(
        request_seed=17,
        return_explanations=True,
        graph={"mode": "inductive", "artifact": str(path)},
    )
    scorer.attach_dataset(dataset)
    out = scorer(
        src_iris=[source_iris[0]],
        tgt_iris=[target_iris[0]],
        src_label_lists=[["Organism"]],
        tgt_label_lists=[["Organism"]],
    )
    assert len(out["graph_head"]) == 1
    head = out["graph_head"][0]
    assert sum(head["feature_contributions"]) + head["bias"] == pytest.approx(head["logit"])
    assert out["s_graph"].shape == (1,)
    assert out["fusion_channels"]["graph"]["active"].tolist() == [True]
