"""Graph-control serialization preserves numerical features and fitted predictions."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from exact.core.entities.graph import Edge
from exact.impl.graph_controls import graph_fingerprint, remove_hierarchy
from exact.impl.models.graph_head import (
    HIERARCHY_PREDICATES,
    compact_graph_fingerprints,
    fit_graph_artifact,
    graph_pair_features,
    graph_predictions,
    structural_profiles,
    verify_graph_manifests,
)
from exact.utils.fitted_artifacts import fingerprint
from tests import kind_evidence_controls_test

dataset = kind_evidence_controls_test.dataset


def _dataset(size=12):
    def edges(prefix, shift):
        return [
            edge
            for index in range(size)
            for edge in (
                Edge(f"{prefix}{index}", "is_a", f"{prefix}{(index + 1) % size}"),
                Edge(f"{prefix}{index}", "part", f"{prefix}{(index * shift + 3) % size}"),
            )
        ]

    return SimpleNamespace(
        source_graph=SimpleNamespace(edges=edges("s", 2)),
        target_graph=SimpleNamespace(edges=edges("t", 3)),
    )


def _legacy_profiles_and_manifest(data, fraction):
    profiles, manifests = {}, {}
    for side in ("src", "tgt"):
        original = getattr(data, "source_graph" if side == "src" else "target_graph").edges
        selected, removal = remove_hierarchy(
            original, fraction=fraction, seed=17, hierarchy_predicates=HIERARCHY_PREDICATES
        )
        profiles[side] = structural_profiles(selected)
        manifests[side] = {
            "input_sha256": graph_fingerprint(original),
            "hierarchy_removal": removal,
            "output_sha256": graph_fingerprint(selected),
        }
    return profiles, manifests


@pytest.mark.parametrize("fraction", [0.5, 1.0])
def test_compaction_keeps_features_fits_predictions_and_full_provenance(tmp_path, fraction):
    data = _dataset()
    src = [f"s{index}" for index in range(6) for _ in range(2)]
    tgt = [f"t{index + offset}" for index in range(6) for offset in (0, 1)]
    directory = tmp_path / "manifests"
    features = graph_pair_features(
        data, src, tgt, {"hierarchy_removal": fraction}, 17, manifest_directory=directory
    )
    profiles, manifests = _legacy_profiles_and_manifest(data, fraction)
    compact = compact_graph_fingerprints(manifests)
    assert compact_graph_fingerprints(compact) == compact
    for source, target, row in zip(src, tgt, features):
        expected = (1 / (1 + abs(profiles["src"][source] - profiles["tgt"][target]))).tolist()
        assert row["values"] == expected
        assert row["active"] is True
        assert row["graph_fingerprints"] == compact
    assert len(list(directory.glob("*.json"))) == 2
    for side, full in manifests.items():
        digest = fingerprint(full)
        assert compact[side]["manifest_sha256"] == digest
        assert fingerprint(json.loads((directory / f"{digest}.json").read_text())) == digest
        assert "removed" not in compact[side]["hierarchy_removal"]
        assert compact[side]["hierarchy_removal"]["removed_count"] == len(
            full["hierarchy_removal"]["removed"]
        )
    verify_graph_manifests(compact, directory)

    legacy = [{**row, "graph_fingerprints": manifests} for row in features]
    refs = {(f"s{index}", f"t{index}") for index in range(6)}
    application = {
        "dataset_signature": "same-input-graphs",
        "source_ids": ["heldout-report-source"],
        "negative_label_policy": "complete_reference",
    }
    rows = pd.DataFrame({"Src": src, "Tgt": tgt, "graph_features": features})
    old_rows = pd.DataFrame({"Src": src, "Tgt": tgt, "graph_features": legacy})
    old = fit_graph_artifact(
        old_rows, refs, tmp_path / "old.json", application=application, seed=17
    )
    new = fit_graph_artifact(rows, refs, tmp_path / "new.json", application=application, seed=17)
    assert old["weights"] == new["weights"]
    assert old["bias"] == new["bias"]
    assert old["folds"] == new["folds"]
    assert old["oof_predictions"] == new["oof_predictions"]
    for predicted in (graph_predictions(legacy, old), graph_predictions(features, old)):
        for actual, expected in zip(predicted, graph_predictions(features, new)):
            np.testing.assert_array_equal(actual, expected)
    # Storage representation changes are not silently relabelled as old fits.
    assert old["fit_identity"] != new["fit_identity"]
    with pytest.raises(ValueError, match="does not match"):
        fit_graph_artifact(rows, refs, tmp_path / "old.json", application=application, seed=17)

    changed = copy.deepcopy(compact)
    changed["src"]["hierarchy_removal"]["removed_count"] += 1
    with pytest.raises(ValueError, match="compact binding"):
        verify_graph_manifests(changed, directory)
    with pytest.raises(ValueError, match="fingerprint"):
        graph_predictions([{**features[0], "graph_fingerprints": changed}], old)


@pytest.mark.parametrize("fraction", [0.5, 1.0])
def test_each_row_stays_bounded_as_graph_grows_and_sidecars_write_once(
    tmp_path, monkeypatch, fraction
):
    import exact.impl.models.graph_head as graph_head

    monkeypatch.setenv("EXACT_EXPERIMENT_RUNTIME", str(tmp_path / "cell/recovery-runtime.json"))
    data = _dataset(2000)
    config = {"hierarchy_removal": fraction}
    row = graph_pair_features(data, ["s0"], ["t0"], config, 17)[0]
    assert len(json.dumps(row)) < 2200
    assert len(json.dumps(data._inductive_graph_manifests)) > 20 * len(json.dumps(row))
    directory = tmp_path / "cell/fitting/graph-manifests"
    verify_graph_manifests(row["graph_fingerprints"], directory)
    times = {path: path.stat().st_mtime_ns for path in directory.glob("*.json")}
    assert len(times) == 2

    def unexpected_write(*args, **kwargs):
        raise AssertionError("Unchanged graph sidecar must not be serialized every batch")

    monkeypatch.setattr(graph_head, "freeze_json", unexpected_write)
    repeated = graph_pair_features(data, ["s0"] * 100, ["t0"] * 100, config, 17)
    assert all(item == row for item in repeated)
    assert {path: path.stat().st_mtime_ns for path in times} == times
    assert len(json.dumps(repeated)) < 220_000


def test_unperturbed_graph_bindings_remain_unchanged():
    data = _dataset()
    row = graph_pair_features(data, ["s0"], ["t0"], {}, 17)[0]
    assert row["graph_fingerprints"] == {
        side: {
            "input_sha256": graph_fingerprint(graph.edges),
            "output_sha256": graph_fingerprint(graph.edges),
        }
        for side, graph in (("src", data.source_graph), ("tgt", data.target_graph))
    }
    assert not hasattr(data, "_inductive_graph_manifests")


@pytest.mark.parametrize("fraction", [0.5, 1.0])
def test_inference_explanations_use_compact_controls_with_graph_channel_off(
    tmp_path, monkeypatch, dataset, fraction
):
    from tests.kind_evidence_controls_test import SRC
    from tests.pair_adaptive_experiments_test import _scorer

    monkeypatch.setenv("EXACT_EXPERIMENT_RUNTIME", str(tmp_path / "cell/recovery-runtime.json"))
    scorer = _scorer(
        request_seed=17,
        return_explanations=True,
        property={"enabled": True},
        graph={"mode": "off", "hierarchy_removal": fraction},
    )
    scorer.attach_dataset(dataset)
    output = scorer(
        src_iris=[SRC + "hasPart", SRC + "participatesIn"],
        tgt_iris=["http://example.org/mini/tgt#containsPart", "http://example.org/mini/tgt#enrolledIn"],
        src_label_lists=[["has part"], ["participates in"]],
        tgt_label_lists=[["contains part"], ["enrolled in"]],
    )
    original = scorer._attached_dataset.graph_control_manifests
    compact = compact_graph_fingerprints(original)
    assert output["kind_evidence"]["graph_controls"] == compact
    assert all("removed" not in value for value in compact.values())
    assert any("removed" in value for value in original.values())
    for explanation in output["explanations"]:
        assert explanation["experiment_diagnostics"]["kind_evidence"]["graph_controls"] == compact
    verify_graph_manifests(compact, tmp_path / "cell/fitting/graph-manifests")
