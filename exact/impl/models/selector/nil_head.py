"""Explicit source labels separate ontology NIL, candidate misses and unknowns."""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch

from .fitting import fingerprint, freeze_json
from .llm_learning import source_features

STATUSES = ("in_pool", "ontology_nil", "pool_miss")
BENCHMARK_STATUSES = ("in_pool", "benchmark_nil", "pool_miss")
FEATURES = ("top_score", "top_two_margin", "candidate_entropy", "displayed_count")


def benchmark_candidate_labels(frame):
    """Only author-confirmed candidates define negatives in a fixed benchmark pool."""
    if not {"Src", "Tgt", "confirmed_label"} <= set(frame):
        raise ValueError("Benchmark NIL requires explicit confirmed candidate labels")
    if frame[["Src", "Tgt"]].duplicated().any() or not set(frame.confirmed_label) <= {0, 1}:
        raise ValueError("Benchmark candidate labels must be unique, complete binary annotations")
    return {(str(row.Src), str(row.Tgt)): int(row.confirmed_label) for row in frame.itertuples()}


def nil_source_features(frame, sources):
    grouped = {str(source): group for source, group in frame.groupby("Src", sort=False)}
    return [
        (
            source_features(grouped[source].S_base)
            if source in grouped and not grouped[source].empty
            else [0.0, 0.0, 0.0, 0.0]
        )
        for source in sources
    ]


def _fit(features, labels):
    x = torch.tensor(features, dtype=torch.float64)
    mean = x.mean(0)
    scale = x.std(0, unbiased=False).clamp_min(1e-6)
    x = (x - mean) / scale
    with torch.enable_grad():
        weights = torch.zeros(
            (len(FEATURES), len(STATUSES)), dtype=torch.float64, requires_grad=True
        )
        bias = torch.zeros(len(STATUSES), dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.Adam([weights, bias], lr=0.05)
        for _ in range(100):
            optimizer.zero_grad()
            loss = (
                torch.nn.functional.cross_entropy(x @ weights + bias, torch.tensor(labels))
                + 0.001 * weights.square().mean()
            )
            loss.backward()
            optimizer.step()
    return {
        "weights": weights.detach().tolist(),
        "bias": bias.detach().tolist(),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
    }


def nil_probabilities(features, model):
    x = torch.tensor(features, dtype=torch.float64)
    standardized = (x - torch.tensor(model["mean"], dtype=torch.float64)) / torch.tensor(
        model["scale"], dtype=torch.float64
    )
    contributions = standardized[:, :, None] * torch.tensor(model["weights"], dtype=torch.float64)
    logits = contributions.sum(1) + torch.tensor(model["bias"], dtype=torch.float64)
    return torch.softmax(logits, dim=-1).tolist(), contributions.tolist(), logits.tolist()


def fit_nil_artifact(frame, source_labels, reference_pairs, path, *, application, seed=17):
    """Fit explicit source labels under their declared absence scope."""
    benchmark = application.get("nil_label_semantics") == "benchmark_pool"
    statuses = BENCHMARK_STATUSES if benchmark else STATUSES
    nil_status = statuses[1]
    required = {"Src", "Status"}
    if not required.issubset(source_labels):
        raise ValueError("NIL training requires explicit Src/Status source labels")
    labels = source_labels.copy()
    labels["Src"] = labels.Src.astype(str)
    if labels.Src.duplicated().any() or not set(labels.Status).issubset({*statuses, "unknown"}):
        raise ValueError(f"NIL source labels must be unique with {statuses} or unknown status")
    labels = labels[labels.Status != "unknown"].sort_values("Src").reset_index(drop=True)
    if application.get("nil_label_semantics") not in {
        "natural",
        "benchmark_pool",
    } or nil_status not in set(labels.Status):
        raise ValueError(
            "NIL fitted arm requires independently annotated examples with declared absence semantics"
        )
    if "in_pool" not in set(labels.Status) or len(labels) < 3:
        raise ValueError(
            "NIL fitting needs mapped and explicitly annotated rejection source groups"
        )
    sources = labels.Src.tolist()
    if set(sources) & set(application.get("source_ids", [])):
        raise ValueError("NIL training overlaps reporting source groups")
    reference = {(str(source), str(target)) for source, target in reference_pairs}
    pool = set(zip(frame.Src.astype(str), frame.Tgt.astype(str)))
    confirmed = benchmark_candidate_labels(frame) if benchmark else None
    if benchmark and {pair for pair, value in confirmed.items() if value == 1} != reference:
        raise ValueError("Benchmark training positives differ from confirmed candidate labels")
    for row in labels.itertuples():
        positives = {pair for pair in reference if pair[0] == row.Src}
        if row.Status == nil_status and positives:
            raise ValueError("NIL label conflicts with a known positive mapping")
        if row.Status == "in_pool" and not positives & pool:
            raise ValueError(
                "in_pool NIL training status has no verified positive in the candidate pool"
            )
        if row.Status == "pool_miss" and (not positives or positives & pool):
            raise ValueError(
                "pool_miss requires a known outside-pool target and no known positive in the pool"
            )
    features = nil_source_features(frame, sources)
    targets = [statuses.index(status) for status in labels.Status]
    identity = fingerprint(
        {
            "application": application,
            "source_labels": labels.to_dict("records"),
            "features": features,
            "references": sorted(reference),
            "seed": seed,
            "recipe": "three_class_source_softmax_v1",
        }
    )
    path = Path(path)
    if path.exists():
        payload = json.loads(path.read_text())
        if payload.get("fit_identity") != identity:
            raise ValueError("NIL artifact data/recipe identity mismatch")
        return payload
    order = list(range(len(sources)))
    random.Random(seed).shuffle(order)
    records, oof = [], []
    for fold in range(min(5, len(sources))):
        test = set(order[fold :: min(5, len(sources))])
        train = [index for index in order if index not in test]
        checkpoint = path.parent / (path.name + ".folds") / f"{identity}-{fold}.json"
        if checkpoint.exists():
            record = json.loads(checkpoint.read_text())
        else:
            model = _fit([features[index] for index in train], [targets[index] for index in train])
            indices = sorted(test)
            probabilities, _, _ = nil_probabilities([features[index] for index in indices], model)
            record = {
                "fold": fold,
                "training_sources": sorted(sources[index] for index in train),
                "heldout_sources": sorted(sources[index] for index in test),
                "predictions": [
                    {
                        "Src": sources[index],
                        "Status": statuses[targets[index]],
                        "probabilities": dict(zip(statuses, prediction)),
                    }
                    for index, prediction in zip(indices, probabilities)
                ],
            }
            freeze_json(checkpoint, record)
        records.append(
            {key: record[key] for key in ("fold", "training_sources", "heldout_sources")}
        )
        oof.extend(record["predictions"])
    payload = {
        "schema_version": 1,
        "kind": "benchmark_nil_head" if benchmark else "natural_nil_head",
        "label_semantics": "benchmark_pool" if benchmark else "natural",
        "ontology_nil_claim": not benchmark,
        "fit_identity": identity,
        "feature_schema": list(FEATURES),
        "statuses": list(statuses),
        "application": application,
        "training_sources": sources,
        "source_label_counts": labels.Status.value_counts().to_dict(),
        "model": _fit(features, targets),
        "folds": records,
        "oof_predictions": oof,
        "probability_scale": "hierarchical_joint_source_status_times_conditional_candidate",
        "unknown_labels_excluded": int((source_labels.Status == "unknown").sum()),
    }
    return freeze_json(path, payload)


def validate_nil_application(dataset, path, artifact, config):
    """Use the same strict donor manifest as other heads; never relax its dataset guard."""
    from exact.utils.artifact_transfer import validate_transferred_artifact

    transferred = validate_transferred_artifact(dataset, path, kind="nil", features=list(FEATURES))
    if transferred and config.get("training_source_labels"):
        raise ValueError("Transferred NIL cannot consume recipient training source labels")
    if not transferred and artifact["application"].get("dataset_signature") != getattr(
        dataset, "dataset_signature", None
    ):
        raise ValueError("NIL artifact application dataset mismatch")
    if artifact["application"].get("nil_label_semantics") != config.get("label_semantics"):
        raise ValueError("NIL artifact label-semantics mismatch")


def source_decision_records(frame, source_universe, *, artifact=None):
    """Return one record per frozen source, including sources with no candidates."""
    sources = sorted(set(str(value) for value in source_universe))
    if not sources:
        return []
    groups = {str(source): group for source, group in frame.groupby("Src", sort=False)}
    probabilities = None
    statuses = STATUSES
    if artifact:
        statuses = BENCHMARK_STATUSES if artifact.get("kind") == "benchmark_nil_head" else STATUSES
        if (
            artifact.get("kind") not in {"natural_nil_head", "benchmark_nil_head"}
            or artifact.get("feature_schema") != list(FEATURES)
            or artifact.get("statuses") != list(statuses)
        ):
            raise ValueError("Invalid NIL feature/schema artifact")
        if set(sources) & set(artifact["training_sources"]):
            raise ValueError("NIL inference overlaps training source groups")
        probabilities, contributions, logits = nil_probabilities(
            nil_source_features(frame, sources), artifact["model"]
        )
    records = []
    for index, source in enumerate(sources):
        group = groups.get(source)
        count = 0 if group is None else len(group)
        row = {
            "Src": source,
            "SrcKind": str(group.iloc[0].get("SrcKind", "class")) if count else "class",
            "candidate_count": count,
            "absence_semantics": "unknown",
            "ontology_nil_probability": None,
            "benchmark_nil_probability": None,
            "pool_miss_probability": None,
        }
        if probabilities is not None:
            prediction = dict(zip(statuses, probabilities[index]))
            status = statuses[
                max(range(len(STATUSES)), key=lambda item: probabilities[index][item])
            ]
            if max(probabilities[index]) < 0.5 or (count == 0 and status == "in_pool"):
                status = "unknown"
            row.update(
                absence_semantics=status,
                **{f"{statuses[1]}_probability": prediction[statuses[1]]},
                pool_miss_probability=prediction["pool_miss"],
                in_pool_probability=prediction["in_pool"],
                feature_contributions=contributions[index],
                logits=logits[index],
                probability_scale=artifact["probability_scale"],
            )
        elif count:
            abstained = bool(group.iloc[0].get("selection_nil_winner", False)) or bool(
                group.iloc[0].get("selection_abstained", False)
            )
            row["absence_semantics"] = "unknown" if abstained else "in_pool"
            row["probability_scale"] = "candidate_set_abstention_only"
        row["status_origin"] = "fitted_source_prediction" if artifact else "candidate_set_heuristic"
        row["action"] = (
            "rank_candidates" if row["absence_semantics"] == "in_pool" and count else "abstain"
        )
        records.append(row)
    return records


def remove_development_positives(frame, reference_pairs, *, role, negative_label_policy):
    if role not in {"development", "diagnostic"} or negative_label_policy != "complete_reference":
        raise ValueError(
            "Gold-removal diagnostics require a declared development role and complete reference"
        )
    reference = {(str(source), str(target)) for source, target in reference_pairs}
    before = list(zip(frame.Src.astype(str), frame.Tgt.astype(str)))
    keep = [pair not in reference for pair in before]
    return frame.loc[keep].copy(), {
        "role": "diagnostic",
        "source_universe": sorted(set(frame.Src.astype(str))),
        "removed_pairs": sorted(set(before) & reference),
        "input_pool_sha256": fingerprint(before),
        "ontology_nil_claim": False,
        "absence_semantics": "synthetic_pool_miss",
    }


def nil_metrics(
    records, source_labels, reference_pairs, emitted_pairs, *, nil_status="ontology_nil"
):
    """Evaluate after scoring; unknown source labels never become negative outcomes."""
    labels = {str(row.Src): str(row.Status) for row in source_labels.itertuples()}
    known = {source for source, status in labels.items() if status != "unknown"}
    predictions = {row["Src"]: row for row in records}
    if len(predictions) != len(records) or known - set(predictions):
        raise ValueError("NIL evaluation requires one output for every labeled source")
    emitted_sources = {str(source) for source, _ in emitted_pairs}
    predicted_nil = {
        source
        for source in known
        if (
            source not in emitted_sources
            if nil_status == "benchmark_nil"
            else predictions[source]["absence_semantics"] == nil_status
        )
    }
    gold_nil = {source for source in known if labels[source] == nil_status}
    reference = {
        (str(source), str(target)) for source, target in reference_pairs if str(source) in known
    }
    if any(source in gold_nil for source, _ in reference):
        raise ValueError("NIL evaluation labels conflict with positive mappings")
    truth = reference | {(source, "__NIL__") for source in gold_nil}
    emitted = {
        (str(source), str(target)) for source, target in emitted_pairs if str(source) in known
    }
    predicted = emitted | {(source, "__NIL__") for source in predicted_nil}

    def counts(actual, expected):
        tp, fp, fn = len(actual & expected), len(actual - expected), len(expected - actual)
        return {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "F1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        }

    return {
        "nil_aware": counts(predicted, truth),
        ("benchmark_nil" if nil_status == "benchmark_nil" else "natural_nil"): counts(
            predicted_nil, gold_nil
        ),
        "labeled_sources": len(known),
        "unknown_sources_excluded": len(labels) - len(known),
        "pool_miss_as_nil": len(
            {source for source in predicted_nil if labels[source] == "pool_miss"}
        ),
    }


def prepare_pool_miss_diagnostic(dataset, reference_path, *, negative_label_policy, seed):
    """Named development-only intervention; preserve the frozen source population."""
    from exact.utils.data import read_table

    table = read_table(Path(reference_path))
    reference = {
        (str(source), str(target))
        for source, target in table.iloc[:, :2].itertuples(index=False, name=None)
    }
    frame, manifest = remove_development_positives(
        dataset.candidates,
        reference,
        role="development",
        negative_label_policy=negative_label_policy,
    )
    if not hasattr(dataset, "eligible_source_iris"):
        dataset.freeze_source_universe(manifest["source_universe"], cap=None, seed=seed)
    dataset._candidates = frame
    exact = getattr(dataset, "_exact_matches", None)
    if exact is not None and {"Src", "Tgt"}.issubset(exact):
        dataset._exact_matches = exact.loc[
            [
                (str(source), str(target)) not in reference
                for source, target in exact[["Src", "Tgt"]].itertuples(index=False, name=None)
            ]
        ].copy()
    dataset._active_candidate_config["gold_removal_diagnostic"] = manifest
    dataset._refresh_candidate_pool_manifest(origin="development_gold_removal")
    return manifest
