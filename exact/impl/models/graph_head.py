"""Small inductive structural head; no identifiers, labels or seed links are features."""

from __future__ import annotations

import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from exact.impl.graph_controls import (
    graph_fingerprint,
    remove_hierarchy,
    shuffle_relations,
)
from exact.impl.models.selector.fitting import (
    fingerprint,
    freeze_json,
    safe_training_labels,
)

FEATURE_SCHEMA = [
    "in_degree_agreement",
    "out_degree_agreement",
    "predicate_count_agreement",
    "neighbor_count_agreement",
    "neighbor_degree_mean_agreement",
    "neighbor_degree_std_agreement",
    "reciprocity_agreement",
    "self_loop_agreement",
]
HIERARCHY_PREDICATES = [
    "http://subclassof",
    "http://subpropertyof",
    "http://www.w3.org/2000/01/rdf-schema#subClassOf",
    "http://www.w3.org/2000/01/rdf-schema#subPropertyOf",
    "is_a",
    "subClassOf",
    "subPropertyOf",
]


def compact_graph_fingerprints(manifests):
    """Bind complete graph controls without repeating their edge lists per pair.

    Manifests without removed-edge lists retain their original representation.
    The full control is kept separately, addressed by its canonical checksum.
    """
    result = {}
    for side, manifest in manifests.items():
        removal = manifest.get("hierarchy_removal") if isinstance(manifest, dict) else None
        direct = isinstance(manifest, dict) and "removed" in manifest
        if direct:
            removal = manifest
        if not isinstance(removal, dict) or "removed" not in removal:
            result[side] = manifest
            continue
        compact = {key: value for key, value in removal.items() if key != "removed"}
        compact.update(
            removed_count=len(removal["removed"]),
            removed_sha256=fingerprint(removal["removed"]),
        )
        result[side] = {
            **(compact if direct else {**manifest, "hierarchy_removal": compact}),
            "manifest_sha256": fingerprint(manifest),
        }
    return result


def verify_graph_manifests(manifests, directory):
    """Verify relocated full controls against their compact per-pair bindings."""
    for side, manifest in manifests.items():
        digest = manifest.get("manifest_sha256") if isinstance(manifest, dict) else None
        if digest is None:
            continue
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError("Invalid graph control manifest checksum")
        full = json.loads((Path(directory) / (digest + ".json")).read_text())
        if (
            fingerprint(full) != digest
            or compact_graph_fingerprints({side: full})[side] != manifest
        ):
            raise ValueError("Graph control manifest does not match its compact binding")


def _persist_graph_manifests(dataset, directory):
    # The experiment recovery contract already includes the entire fitting tree.
    # Other callers can supply a directory, or inspect this registry in memory.
    if directory is None:
        runtime = os.environ.get("EXACT_EXPERIMENT_RUNTIME")
        if runtime:
            directory = Path(runtime).parent / "fitting" / "graph-manifests"
    if directory is None:
        return
    written = getattr(dataset, "_inductive_graph_manifest_paths", None)
    if written is None:
        written = dataset._inductive_graph_manifest_paths = set()
    for digest, manifest in getattr(dataset, "_inductive_graph_manifests", {}).items():
        path = Path(directory) / (digest + ".json")
        if path not in written:
            freeze_json(path, manifest)
            written.add(path)


def graph_control_evidence(dataset, *, manifest_directory=None):
    """Compact the scoring view's immutable controls once, including graph-off arms."""
    controls = getattr(dataset, "graph_control_manifests", {})
    cache = getattr(dataset, "_compact_graph_control_cache", None)
    if cache is None:
        cache = dataset._compact_graph_control_cache = {}
    result = {}
    for name, full in controls.items():
        # Control views publish complete manifests, replacing rather than mutating
        # them. Retain the object reference so Python cannot recycle its identity.
        if name not in cache or cache[name][0] is not full:
            compact = compact_graph_fingerprints({name: full})[name]
            cache[name] = (full, compact)
            if "manifest_sha256" in compact:
                registry = getattr(dataset, "_inductive_graph_manifests", None)
                if registry is None:
                    registry = dataset._inductive_graph_manifests = {}
                registry[compact["manifest_sha256"]] = full
        result[name] = cache[name][1]
    _persist_graph_manifests(dataset, manifest_directory)
    return result


def structural_profiles(edges):
    """One linear graph pass plus one neighbor pass, shared across every candidate."""
    incoming, outgoing = Counter(), Counter()
    neighbors, predicates, successors = defaultdict(set), defaultdict(set), defaultdict(set)
    loops = Counter()
    for edge in edges:
        incoming[edge.dst] += 1
        outgoing[edge.src] += 1
        neighbors[edge.src].add(edge.dst)
        neighbors[edge.dst].add(edge.src)
        predicates[edge.src].add(edge.rel)
        predicates[edge.dst].add(edge.rel)
        successors[edge.src].add(edge.dst)
        loops[edge.src] += int(edge.src == edge.dst)
    profiles = {}
    for iri, adjacent in neighbors.items():
        degrees = [incoming[node] + outgoing[node] for node in adjacent]
        reciprocal = sum(iri in successors[node] for node in successors[iri]) / max(
            1, len(successors[iri])
        )
        profiles[iri] = np.array(
            [
                *np.log1p(
                    [
                        incoming[iri],
                        outgoing[iri],
                        len(predicates[iri]),
                        len(adjacent),
                        np.mean(degrees),
                        np.std(degrees),
                    ]
                ),
                reciprocal,
                float(loops[iri] > 0),
            ]
        )
    return profiles


def graph_pair_features(dataset, src_iris, tgt_iris, config, seed, *, manifest_directory=None):
    """Use ontology-wide unsupervised profiles, with a separate lock for each control."""
    if seed is None and (config.get("hierarchy_removal") or config.get("shuffled")):
        raise ValueError("graph controls require an explicit request_seed")
    cache = getattr(dataset, "_inductive_graph_cache", None)
    if cache is None:
        cache = dataset._inductive_graph_cache = {}
    side_profiles, manifests = {}, {}
    for side in ("src", "tgt"):
        graph = dataset.source_graph if side == "src" else dataset.target_graph
        key = (
            side,
            float(config.get("hierarchy_removal", 0.0)),
            bool(config.get("shuffled")),
            seed,
        )
        if key not in cache:
            edges = list(graph.edges or [])
            manifest = {"input_sha256": graph_fingerprint(edges)}
            if config.get("hierarchy_removal", 0.0):
                edges, manifest["hierarchy_removal"] = remove_hierarchy(
                    edges,
                    fraction=config["hierarchy_removal"],
                    seed=seed,
                    hierarchy_predicates=config.get("hierarchy_predicates", HIERARCHY_PREDICATES),
                )
            if config.get("shuffled"):
                edges, manifest["shuffle"] = shuffle_relations(edges, seed=seed)
            manifest["output_sha256"] = graph_fingerprint(edges)
            compact = compact_graph_fingerprints({side: manifest})[side]
            if "manifest_sha256" in compact:
                registry = getattr(dataset, "_inductive_graph_manifests", None)
                if registry is None:
                    registry = dataset._inductive_graph_manifests = {}
                registry[compact["manifest_sha256"]] = manifest
            cache[key] = (structural_profiles(edges), compact)
        side_profiles[side], manifests[side] = cache[key]
    _persist_graph_manifests(dataset, manifest_directory)
    rows = []
    for src, tgt in zip(src_iris, tgt_iris):
        source, target = side_profiles["src"].get(src), side_profiles["tgt"].get(tgt)
        active = source is not None and target is not None
        values = (
            (1.0 / (1.0 + abs(source - target))).tolist() if active else [0.0] * len(FEATURE_SCHEMA)
        )
        rows.append({"values": values, "active": active, "graph_fingerprints": manifests})
    return rows


def fit_graph_artifact(training, reference_pairs, path, *, application, seed):
    """Fit one fixed regularized logistic recipe on explicitly labeled train sources."""
    policy = application.get("negative_label_policy")
    if policy not in {"complete_reference", "confirmed_negatives"}:
        raise ValueError(
            "graph fitting requires an explicit complete-reference or confirmed-negative policy"
        )
    frame, reference = safe_training_labels(training, reference_pairs, application)
    if frame.empty or set(frame.Src.astype(str)) & set(application.get("source_ids", [])):
        raise ValueError("graph fitting requires disjoint labeled training source groups")
    features = np.asarray([row["values"] for row in frame.graph_features], dtype=float)
    if features.shape[1:] != (len(FEATURE_SCHEMA),) or not np.isfinite(features).all():
        raise ValueError("graph training feature schema mismatch")
    if any(
        row["graph_fingerprints"] != frame.graph_features.iloc[0]["graph_fingerprints"]
        for row in frame.graph_features
    ):
        raise ValueError("graph training rows mix graph/profile fingerprints")
    labels = np.asarray(
        [float((str(row.Src), str(row.Tgt)) in reference) for row in frame.itertuples()]
    )
    if len(set(labels)) != 2:
        raise ValueError("graph fitting requires both positive and permitted negative examples")
    provenance = {
        "training_sources": sorted(set(frame.Src.astype(str))),
        "negative_label_policy": policy,
        "training_features_sha256": fingerprint(
            frame[["Src", "Tgt", "graph_features"]].to_dict("records")
        ),
        "training_reference_sha256": fingerprint(sorted(reference)),
        "seed": int(seed),
        "application": application,
        "recipe": "inductive_graph_statistics_logistic_l2_0.01_v1",
    }
    identity = fingerprint(provenance)
    path = Path(path)
    if path.exists():
        payload = json.loads(path.read_text())
        if payload.get("fit_identity") != identity:
            raise ValueError("graph artifact does not match training features/recipe")
        return payload
    # Equal total contribution per source, so a large pool does not acquire extra supervision.
    counts = frame.groupby("Src").Src.transform("size").to_numpy(dtype=float)
    weights = 1.0 / counts
    weights /= weights.sum()

    def fit_rows(indices):
        matrix, truth, sample_weight = features[indices], labels[indices], weights[indices]
        sample_weight = sample_weight / sample_weight.sum()

        def objective(parameters):
            logits = matrix @ parameters[:-1] + parameters[-1]
            error = (expit(logits) - truth) * sample_weight
            value = np.sum(
                sample_weight * (np.logaddexp(0.0, logits) - truth * logits)
            ) + 0.01 * np.sum(parameters[:-1] ** 2)
            gradient = np.r_[matrix.T @ error + 0.02 * parameters[:-1], error.sum()]
            return float(value), gradient

        fit = minimize(
            objective,
            np.zeros(len(FEATURE_SCHEMA) + 1),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 200},
        )
        if not fit.success:
            raise ValueError(f"graph head optimization failed: {fit.message}")
        return fit

    source_ids = sorted(set(frame.Src.astype(str)))
    if len(source_ids) < 2:
        raise ValueError(
            "graph fitting requires at least two source groups for out-of-fold predictions"
        )
    random.Random(int(seed)).shuffle(source_ids)
    oof, folds = [], []
    source_array = frame.Src.astype(str).to_numpy()
    for fold in range(min(3, len(source_ids))):
        heldout = set(source_ids[fold :: min(3, len(source_ids))])
        held_mask = np.array([source in heldout for source in source_array])
        checkpoint = path.parent / (path.name + ".folds") / f"{identity}-{fold}.json"
        if checkpoint.exists():
            record = json.loads(checkpoint.read_text())
        else:
            fitted = fit_rows(np.flatnonzero(~held_mask))
            probabilities = expit(features[held_mask] @ fitted.x[:-1] + fitted.x[-1])
            record = {
                "fold": fold,
                "training_sources": sorted(set(source_ids) - heldout),
                "heldout_sources": sorted(heldout),
                "predictions": [
                    {"Src": str(row.Src), "Tgt": str(row.Tgt), "score": float(probability)}
                    for row, probability in zip(frame[held_mask].itertuples(), probabilities)
                ],
            }
            freeze_json(checkpoint, record)
        oof.extend(record["predictions"])
        folds.append({key: value for key, value in record.items() if key != "predictions"})
    result = fit_rows(np.arange(len(frame)))
    payload = {
        "schema_version": 1,
        "mode": "inductive",
        "kind": "graph_head",
        "dataset_signature": application["dataset_signature"],
        "feature_schema": FEATURE_SCHEMA,
        "weights": result.x[:-1].tolist(),
        "bias": float(result.x[-1]),
        "fit_identity": identity,
        "fit_provenance": provenance,
        "graph_fingerprints": frame.graph_features.iloc[0]["graph_fingerprints"],
        "folds": folds,
        "oof_predictions": oof,
        "explanation_schema": "feature_additive_logit",
    }
    return freeze_json(Path(path), payload)


def graph_predictions(features, artifact):
    if artifact.get("feature_schema") != FEATURE_SCHEMA:
        raise ValueError("graph artifact feature schema mismatch")
    weights = np.asarray(artifact["weights"], dtype=float)
    if weights.shape != (len(FEATURE_SCHEMA),) or not np.isfinite(weights).all():
        raise ValueError("graph artifact weights are invalid")
    # Existing fitted heads retain the complete legacy manifest. Normalize only
    # the provenance binding; weights, features and predictions remain unchanged.
    expected = compact_graph_fingerprints(artifact["graph_fingerprints"])
    if any(compact_graph_fingerprints(row["graph_fingerprints"]) != expected for row in features):
        raise ValueError("graph artifact graph/profile fingerprint mismatch")
    if not np.isfinite(float(artifact["bias"])):
        raise ValueError("graph artifact bias must be finite")
    values = np.asarray([row["values"] for row in features])
    contributions = values * weights
    logits = contributions.sum(axis=1) + float(artifact["bias"])
    return expit(logits), contributions, logits
