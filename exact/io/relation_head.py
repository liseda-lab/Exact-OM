"""Small source-grouped multinomial head for explicitly typed positive pairs."""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

from exact.core.entities.kinds import EntityKind
from exact.utils.candidate_generation import normalize_candidate_text
from exact.utils.fitted_artifacts import fingerprint, freeze_json

RELATIONS = ["=", "<", ">"]
FEATURE_SCHEMA = [
    "exact_label",
    "best_token_jaccard",
    "label_length_log_ratio",
    "parent_count_log_ratio",
    "child_count_log_ratio",
    "parent_count_agreement",
    "child_count_agreement",
]


def _source_profiles(source):
    cached = getattr(source, "_exact_relation_profiles", None)
    if cached is not None:
        return cached
    rows = {}
    for kind in EntityKind:
        if kind not in {EntityKind.CLASS, EntityKind.OBJECT_PROPERTY, EntityKind.DATA_PROPERTY}:
            continue
        for iri in source.entities(kind):
            labels = sorted(
                {normalize_candidate_text(label) for label in source.labels(iri)} - {""}
            )
            rows[(kind.value, iri)] = (
                labels,
                len(source.direct_parents(iri, kind)),
                len(source.direct_children(iri, kind)),
            )
    identity = fingerprint([[*key, value] for key, value in sorted(rows.items())])
    source._exact_relation_profiles = (rows, identity)
    return rows, identity


def relation_features(frame, source, target):
    source_rows, source_sha = _source_profiles(source)
    target_rows, target_sha = _source_profiles(target)
    features = []
    for row in frame.itertuples():
        src = str(getattr(row, "Src", getattr(row, "SrcEntity", "")))
        tgt = str(getattr(row, "Tgt", getattr(row, "TgtEntity", "")))
        kind = str(getattr(row, "SrcKind", "class"))
        target_kind = str(getattr(row, "TgtKind", kind))
        if kind != target_kind or (kind, src) not in source_rows or (kind, tgt) not in target_rows:
            raise ValueError("Relation head requires known, same-kind class/property pairs")
        lhs, sp, sc = source_rows[kind, src]
        rhs, tp, tc = target_rows[kind, tgt]
        jaccard = max(
            (
                len(set(a.split()) & set(b.split())) / max(1, len(set(a.split()) | set(b.split())))
                for a in lhs
                for b in rhs
            ),
            default=0.0,
        )
        features.append(
            [
                float(bool(set(lhs) & set(rhs))),
                jaccard,
                np.log1p(max(map(len, lhs), default=0)) - np.log1p(max(map(len, rhs), default=0)),
                np.log1p(sp) - np.log1p(tp),
                np.log1p(sc) - np.log1p(tc),
                1.0 / (1 + abs(sp - tp)),
                1.0 / (1 + abs(sc - tc)),
            ]
        )
    return np.asarray(features, dtype=float).reshape((-1, len(FEATURE_SCHEMA))), {
        "source": source_sha,
        "target": target_sha,
    }


def typed_reference_frame(path):
    """Normalize public BioKG aligned lists and ordinary typed pair tables."""
    from ast import literal_eval

    from exact.utils.data import read_table

    frame = read_table(Path(path))
    labels = {
        "equivalent": "=",
        "subsumed_by": "<",
        "subsumes": ">",
        "source_subsumed_by_target": "<",
        "source_subsumes_target": ">",
        "=": "=",
        "<": "<",
        ">": ">",
    }
    if {"SrcEntity", "TgtEntities", "Relations"} <= set(frame):
        rows = []
        for row in frame.itertuples():
            targets, relations = literal_eval(str(row.TgtEntities)), literal_eval(
                str(row.Relations)
            )
            if len(targets) != len(relations):
                raise ValueError("Typed target and relation lists have different lengths")
            rows.extend(
                (str(row.SrcEntity), str(tgt), labels.get(str(relation), str(relation)))
                for tgt, relation in zip(targets, relations)
            )
        frame = pd.DataFrame(rows, columns=["Src", "Tgt", "Relation"])
    else:
        frame = frame.rename(columns={"SrcEntity": "Src", "TgtEntity": "Tgt"})
        if not {"Src", "Tgt", "Relation"} <= set(frame):
            raise ValueError("Typed training table requires Src/Tgt/Relation")
        frame = frame.copy()
        frame["Relation"] = frame.Relation.map(lambda value: labels.get(str(value), str(value)))
    if not frame.Relation.isin(RELATIONS).all():
        raise ValueError(
            "Typed training relations must be =,<,> with < meaning source-subsumed-by-target"
        )
    if frame.groupby(["Src", "Tgt"]).Relation.nunique().max() > 1:
        raise ValueError("Typed training pair has conflicting relation labels")
    return frame.drop_duplicates(["Src", "Tgt"]).reset_index(drop=True)


def fit_relation_artifact(training, typed_reference, source, target, path, *, application, seed):
    frame = training.drop(columns=["Relation"], errors="ignore").merge(
        typed_reference, on=["Src", "Tgt"], how="inner", validate="one_to_one"
    )
    if frame.empty or set(frame.Src.astype(str)) & set(application.get("source_ids", [])):
        raise ValueError("Typed head requires disjoint labeled training source groups")
    if set(frame.Relation) != set(RELATIONS):
        raise ValueError("Three-way fitting needs real =,<,> training labels")
    x, profiles = relation_features(frame, source, target)
    y = np.asarray([RELATIONS.index(value) for value in frame.Relation])
    provenance = {
        "recipe": "multinomial_l2_0.01_v1",
        "seed": seed,
        "application": application,
        "training": frame[["Src", "Tgt", "Relation"]].to_dict("records"),
        "features": fingerprint(x.tolist()),
        "profiles": profiles,
    }
    identity, path = fingerprint(provenance), Path(path)
    if path.exists():
        artifact = json.loads(path.read_text())
        if artifact.get("fit_identity") != identity:
            raise ValueError("Typed head training identity mismatch")
        return artifact
    source_ids = sorted(frame.Src.astype(str).unique())
    if len(source_ids) < 2:
        raise ValueError("Typed fitting needs multiple source groups")
    random.Random(seed).shuffle(source_ids)
    weight = 1.0 / frame.groupby("Src").Src.transform("size").to_numpy()

    def fit(indices):
        if set(y[indices]) != {0, 1, 2}:
            raise ValueError(
                "Each typed training fold needs all three relations; insufficient grouped labels"
            )
        matrix = np.c_[x[indices], np.ones(len(indices))]
        truth = np.eye(3)[y[indices]]
        w = weight[indices] / weight[indices].sum()

        def objective(parameters):
            parameters = parameters.reshape((len(FEATURE_SCHEMA) + 1, 3))
            logits = matrix @ parameters
            loss = np.sum(
                w * (logsumexp(logits, axis=1) - np.sum(logits * truth, axis=1))
            ) + 0.01 * np.sum(parameters[:-1] ** 2)
            gradient = matrix.T @ ((softmax(logits, axis=1) - truth) * w[:, None])
            gradient[:-1] += 0.02 * parameters[:-1]
            return float(loss), gradient.ravel()

        result = minimize(
            objective,
            np.zeros(matrix.shape[1] * 3),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 200},
        )
        if not result.success:
            raise ValueError(f"Typed head optimization failed: {result.message}")
        return result.x.reshape((matrix.shape[1], 3))

    folds, oof = [], []
    for fold in range(min(3, len(source_ids))):
        held = set(source_ids[fold :: min(3, len(source_ids))])
        mask = frame.Src.astype(str).isin(held).to_numpy()
        checkpoint = path.parent / (path.name + ".folds") / f"{identity}-{fold}.json"
        if checkpoint.exists():
            record = json.loads(checkpoint.read_text())
        else:
            parameters = fit(np.flatnonzero(~mask))
            probabilities = softmax(np.c_[x[mask], np.ones(sum(mask))] @ parameters, axis=1)
            record = {
                "fold": fold,
                "training_sources": sorted(set(source_ids) - held),
                "heldout_sources": sorted(held),
                "predictions": [
                    {
                        "Src": row.Src,
                        "Tgt": row.Tgt,
                        "true_relation": row.Relation,
                        "probabilities": p.tolist(),
                    }
                    for row, p in zip(frame[mask].itertuples(), probabilities)
                ],
            }
            freeze_json(checkpoint, record)
        folds.append({key: value for key, value in record.items() if key != "predictions"})
        oof.extend(record["predictions"])
    parameters = fit(np.arange(len(frame)))
    return freeze_json(
        path,
        {
            "schema_version": 1,
            "mode": "learned_three_way",
            "feature_schema": FEATURE_SCHEMA,
            "relations": RELATIONS,
            "weights": parameters[:-1].tolist(),
            "bias": parameters[-1].tolist(),
            "dataset_signature": application.get("dataset_signature"),
            "profile_fingerprints": profiles,
            "fit_identity": identity,
            "fit_provenance": provenance,
            "folds": folds,
            "oof_predictions": oof,
            "relation_counts": frame.Relation.value_counts().to_dict(),
            "explanation_schema": "feature_additive_categorical_logits",
        },
    )


def predict_relation_head(frame, source, target, artifact):
    artifact = (
        json.loads(Path(artifact).read_text()) if isinstance(artifact, (str, Path)) else artifact
    )
    if (
        not artifact
        or artifact.get("feature_schema") != FEATURE_SCHEMA
        or artifact.get("relations") != RELATIONS
    ):
        raise ValueError("Relation prediction requires a fitted multinomial relation artifact")
    features, profiles = relation_features(frame, source, target)
    if profiles != artifact["profile_fingerprints"]:
        raise ValueError("Relation head ontology/profile fingerprint mismatch")
    weights, bias = np.asarray(artifact["weights"]), np.asarray(artifact["bias"])
    if (
        weights.shape != (len(FEATURE_SCHEMA), 3)
        or bias.shape != (3,)
        or not np.isfinite(weights).all()
        or not np.isfinite(bias).all()
    ):
        raise ValueError("Malformed relation head parameters")
    contributions = features[:, :, None] * weights[None, :, :]
    logits = contributions.sum(axis=1) + bias
    return softmax(logits, axis=1), contributions, logits, artifact
