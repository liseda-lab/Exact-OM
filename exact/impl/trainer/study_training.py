"""Optional component-study reuse of native evidence and frozen TRAIN judgments."""

from __future__ import annotations

import json
import hashlib
import fcntl
import logging
import math
import os
import sqlite3
import shutil
from contextlib import nullcontext
from pathlib import Path

import pandas as pd

from exact.utils.fitted_artifacts import fingerprint
from exact.utils.provenance import sha256_file


def structured_training_packet(record):
    """Render exactly the selected native facts, without scores or reporting gold."""
    from exact.runtime.evidence import structured_facts

    labels = record.get("selected_labels", {})
    if set(labels) != {"source", "target"}:
        raise ValueError("Structured TRAIN facts require the actual selected labels")
    triples = record.get("triple_attributions")
    if not isinstance(triples, dict) or not {"hierarchy", "similarity", "difference"} <= set(
        triples
    ):
        raise ValueError("Structured TRAIN facts require full native selected attributions")
    groups = {}
    for group, sides in [
        *(("hierarchy:" + family, sides) for family, sides in triples["hierarchy"].items()),
        ("similarity", triples["similarity"]),
        ("difference", triples["difference"]),
        ("attribute", record.get("attributes", {})),
    ]:
        if set(sides) != {"source", "target"}:
            raise ValueError("Native selected attribution sides are incomplete")
        groups[group] = {"src_selected": sides["source"], "tgt_selected": sides["target"]}
    return structured_facts(labels["source"], labels["target"], groups)


def prepare_native_features(view, primitive):
    """Reuse native per-entity objects, then run the ordinary pair feature API.

    Cache entries share the numerical cache's checksum, atomic writes and disk
    allowance. They contain each entity once per chunk, never repeated pair
    contexts or fitted heads. A miss always uses the normal native implementation.
    """
    if primitive is None or os.getenv("EXACT_COMPONENT_FEATURE_CACHE") != "1":
        return view.get_features(view._candidates)
    from exact.runtime.numerical_cache import NumericalCache

    record = json.loads(Path(os.environ["EXACT_EXPERIMENT_RUNTIME"]).read_text())
    scope = fingerprint(
        {
            "schema": "exact-native-train-features-v1",
            **{
                key: primitive[key]
                for key in (
                    "dataset",
                    "inputs",
                    "implementation",
                    "dependencies",
                    "kind",
                    "namespace",
                )
            },
        }
    )
    keys = []
    for side, iri, kind in (("src", "Src", "SrcKind"), ("tgt", "Tgt", "TgtKind")):
        kinds = view._candidates.get(kind, pd.Series("class", index=view._candidates.index))
        keys.extend(sorted({(side, str(k), str(i)) for i, k in zip(view._candidates[iri], kinds)}))
    hits = misses = 0
    try:
        cache = NumericalCache(Path(os.getenv("EXACT_NUMERICAL_CACHE_ROOT", record["root"])))
    except (OSError, sqlite3.Error) as error:
        logging.getLogger(__name__).warning("Native TRAIN feature cache unavailable: %s", error)
        return view.get_features(view._candidates)
    try:
        for start in range(0, len(keys), 512):
            chunk = keys[start : start + 512]
            address = fingerprint(chunk)
            try:
                saved = cache.get(scope, "native_train_entities", address, "cpu")
            except (OSError, sqlite3.Error):
                saved = None
            if (
                isinstance(saved, list)
                and len(saved) == len(chunk)
                and all(
                    isinstance(item, (list, tuple))
                    and len(item) == 2
                    and tuple(item[0]) == key
                    and isinstance(item[1], dict)
                    for item, key in zip(saved, chunk)
                )
            ):
                view._entity_feature_cache.update((tuple(key), value) for key, value in saved)
                hits += len(chunk)
            else:
                # Native errors are science failures, not cache misses.
                saved = [(key, view.get_entity_features(key[2], key[0], key[1])) for key in chunk]
                try:
                    cache.put(scope, "native_train_entities", address, saved)
                except (OSError, sqlite3.Error):
                    pass
                misses += len(chunk)
    finally:
        try:
            cache.close()
        except (OSError, sqlite3.Error):
            pass
    view._training_feature_cache_report = {"entity_hits": hits, "entity_misses": misses}
    return view.get_features(view._candidates)


def load_train_judgments(model, raw, application, *, pool_path, reference_path):
    """Validate a frozen, label-independent TRAIN bank before any fitting starts."""
    path = os.getenv("EXACT_COMPONENT_TRAIN_JUDGMENTS")
    digest = os.getenv("EXACT_COMPONENT_TRAIN_JUDGMENTS_SHA256")
    if path is None:
        if digest is not None:
            raise ValueError("TRAIN judgment checksum requires a manifest")
        return None
    contents = Path(path).read_bytes()
    if not digest or hashlib.sha256(contents).hexdigest() != digest:
        raise ValueError("TRAIN judgment manifest checksum mismatch")
    payload = json.loads(contents)
    if payload.get("schema") != "exact-component-train-judgments-v1":
        raise ValueError("Unsupported TRAIN judgment manifest")
    expected_binding = {
        "training_pool_sha256": sha256_file(Path(pool_path)),
        "reference_sha256": sha256_file(Path(reference_path)),
        "dataset_signature": application["dataset_signature"],
    }
    if payload.get("binding") != expected_binding:
        raise ValueError("TRAIN judgment input binding mismatch")
    expected_model = os.getenv("EXACT_COMPONENT_JUDGE_MODEL") or getattr(
        model, "llm_model_name", None
    )
    if not payload.get("model_id") or payload["model_id"] != expected_model:
        raise ValueError("TRAIN judgment model mismatch")
    provenance = payload.get("provenance", {})
    # This implementation admits no label-trained prompts, routing, or exemplars.
    # Such a bank needs explicit per-fold dependency validation before admission.
    if provenance != {
        "role": "train",
        "labels_used": False,
        "request_support": "frozen_label_blind",
        "fold_safe": True,
        "evidence": "train_independent_ontology",
        "upstream_fit_dependencies": [],
    }:
        raise ValueError("TRAIN judgment requests need train-independent, fold-safe provenance")
    sources = payload.get("source_ids", [])
    if (
        not sources
        or len(sources) != len(set(sources))
        or not set(sources).issubset(set(raw.Src.astype(str)))
        or set(sources) & set(application.get("source_ids", []))
    ):
        raise ValueError("TRAIN judgment source support is invalid or overlaps reporting")
    expected = [(str(row["Src"]), str(row["Tgt"])) for row in payload.get("expected_pairs", [])]
    available = set(zip(raw.Src.astype(str), raw.Tgt.astype(str)))
    predictions = payload.get("predictions", [])
    pairs = [(str(row["Src"]), str(row["Tgt"])) for row in predictions]
    if (
        len(expected) != len(set(expected))
        or len(pairs) != len(set(pairs))
        or set(pairs) != set(expected)
        or not set(expected).issubset(available)
        or not {source for source, _ in expected}.issubset(sources)
    ):
        raise ValueError("TRAIN judgment bank is incomplete or differs from frozen request support")
    if any(
        isinstance(row.get("probability"), bool)
        or not isinstance(row.get("probability"), (float, int))
        or not math.isfinite(row["probability"])
        or not 0 <= row["probability"] <= 1
        for row in predictions
    ):
        raise ValueError("TRAIN judgment probabilities must be finite and in [0, 1]")
    policy = payload.get("policy", {})
    if policy.get("fusion_weight") not in {"beta_u", "constant"}:
        raise ValueError("Unsupported TRAIN judgment fusion policy")
    scalar = (
        policy.get("constant_weight")
        if policy["fusion_weight"] == "constant"
        else policy.get("beta")
    )
    if (
        isinstance(scalar, bool)
        or not isinstance(scalar, (float, int))
        or not math.isfinite(scalar)
        or scalar < 0
        or (policy["fusion_weight"] == "constant" and scalar > 1)
    ):
        raise ValueError("Invalid TRAIN judgment fusion weight")
    config = getattr(model, "llm_experiment_config", {})
    # Offline replay intentionally disables live experimental routing; its
    # explicitly configured mixing policy still identifies the cached treatment.
    actual_policy = "constant" if config.get("fusion_weight") == "constant" else "beta_u"
    actual_scalar = (
        config.get("constant_weight")
        if actual_policy == "constant"
        else getattr(model, "beta", None)
    )
    if policy["fusion_weight"] != actual_policy or scalar != actual_scalar:
        raise ValueError("TRAIN judgment fusion policy differs from application model")
    return {**payload, "sha256": digest}


def replay_train_judgments(training, bank):
    """Fuse real saved probabilities with current upstream scores, never pseudo-labels."""
    if bank is None:
        return training
    probabilities = {(row["Src"], row["Tgt"]): row["probability"] for row in bank["predictions"]}
    result = training.copy()
    if not {"S_base", "U"}.issubset(result):
        raise ValueError("TRAIN judgment replay requires current S_base and uncertainty")
    if not set(probabilities).issubset(set(zip(result.Src.astype(str), result.Tgt.astype(str)))):
        raise ValueError("An upstream fit removed required TRAIN judgment support")
    policy = bank["policy"]
    # None is an absent observation, never a fabricated negative probability.
    result["p_llm"] = pd.Series([None] * len(result), index=result.index, dtype=object)
    result["llm_used"] = False
    result["S_final"] = result.S_base.astype(float)
    for index, source, target in zip(result.index, result.Src.astype(str), result.Tgt.astype(str)):
        probability = probabilities.get((source, target))
        if probability is None:
            continue
        if not all(math.isfinite(float(result.at[index, field])) for field in ("U", "S_base")):
            raise ValueError("TRAIN judgment replay requires finite current scores")
        weight = (
            policy["constant_weight"]
            if policy["fusion_weight"] == "constant"
            else min(1.0, max(0.0, policy["beta"] * float(result.at[index, "U"])))
        )
        result.at[index, "p_llm"] = probability
        result.at[index, "llm_used"] = True
        result.at[index, "S_final"] = (1 - weight) * float(
            result.at[index, "S_base"]
        ) + weight * probability
    result["S_pair_final"] = result.S_final
    return result


def export_component_training(
    training, *, output_dir, cache_dir, application, pool_path, reference_path
):
    """Publish compact post-fusion/pre-selector rows once, referenced by each consumer."""
    if not any(
        os.getenv(name) == "1"
        for name in ("EXACT_COMPONENT_TRAIN_FACTS", "EXACT_COMPONENT_EXPORT_TRAIN")
    ):
        return None
    import zstandard as zstd

    from exact.runtime.numerical_cache import publish_training_reference
    from exact.impl.models.selector.feature_contract import feature_contract
    from exact.utils.fitted_artifacts import _json_chunks

    excluded = {
        "src_label_text",
        "tgt_label_text",
        "llm_evidence_packet",
        "llm_structured_evidence_packet",
        "selected_labels",
        "graph_features",
        "fusion_channels",
    }
    compact = training[[name for name in training if name not in excluded]].copy()
    if "S_pair_final" not in compact:
        compact["S_pair_final"] = compact.S_final
    if "p_llm" in compact:
        missing = compact.p_llm.isna()
        if "llm_used" in compact and (missing & compact.llm_used.astype(bool)).any():
            raise ValueError("A judged TRAIN pair has no observed probability")
        compact["p_llm"] = compact.p_llm.astype(object).where(~missing, None)
    payload = {
        "schema": "exact-component-training-v1",
        "application": application,
        "feature_contract": feature_contract(compact),
        "source_ids": sorted(compact.Src.astype(str).unique()),
        "inputs": {
            key: {"path": str(Path(path).resolve()), "sha256": sha256_file(Path(path))}
            for key, path in (("training_pool", pool_path), ("reference", reference_path))
        },
        "rows": compact.to_dict("records"),
    }
    identity = fingerprint(payload)
    root = os.getenv("EXACT_NUMERICAL_CACHE_ROOT")
    if root is None and (runtime := os.getenv("EXACT_EXPERIMENT_RUNTIME")):
        root = json.loads(Path(runtime).read_text())["root"]
    index = (
        (Path(root) / "numerical-cache/component-training-references" / (identity + ".json"))
        if root
        else None
    )
    if index is not None:
        index.parent.mkdir(parents=True, exist_ok=True)
    with index.with_suffix(".lock").open("a") if index else nullcontext() as lock:
        if lock is not None:
            fcntl.flock(lock, fcntl.LOCK_EX)
        reference = json.loads(index.read_text()) if index and index.is_file() else None
        if (
            reference is None
            or not Path(reference["aggregate_uri"]).is_file()
            or sha256_file(Path(reference["aggregate_uri"])) != reference["sha256"]
        ):
            path = Path(cache_dir) / "component_training_scores.json.zst"
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".partial")
            limit = int(os.getenv("EXACT_COMPONENT_TRAIN_MAX_BYTES", str(512 * 1024**2)))
            reserve = int(os.getenv("EXACT_STORAGE_MIN_FREE_BYTES", str(1024**3)))
            if limit <= 0 or reserve < 0:
                raise ValueError("Invalid component TRAIN storage allowance")
            if path.exists():
                with zstd.open(path, "rt") as stream:
                    previous = json.load(stream)
                if (
                    previous.get("identity") != identity
                    or fingerprint(
                        {key: value for key, value in previous.items() if key != "identity"}
                    )
                    != identity
                ):
                    raise ValueError("Component TRAIN artifact identity changed")
            try:
                size = 0
                with temporary.open("wb") as stream:
                    with zstd.ZstdCompressor(level=3).stream_writer(
                        stream, closefd=False
                    ) as compressed:
                        for chunk in _json_chunks({"identity": identity, **payload}):
                            size += len(chunk)
                            if size > limit or shutil.disk_usage(path.parent).free < reserve + len(
                                chunk
                            ):
                                raise OSError(
                                    "Component TRAIN export exceeds its size or storage allowance"
                                )
                            compressed.write(chunk)
                    stream.flush()
                    os.fsync(stream.fileno())
                temporary.replace(path)
                descriptor = os.open(path.parent, os.O_DIRECTORY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            finally:
                temporary.unlink(missing_ok=True)
            reference = {
                "schema": "exact-component-training-reference-v1",
                "identity": identity,
                "aggregate_uri": str(path.resolve()),
                "sha256": sha256_file(path),
                "row_count": len(compact),
                "source_count": len(payload["source_ids"]),
                "application": application,
                "inputs": payload["inputs"],
                "feature_contract": payload["feature_contract"],
                "retention": "retain_until_all_consumers_complete",
            }
            if index:
                publish_training_reference(index, reference)
        publish_training_reference(Path(output_dir) / "component_training.json", reference)
    return reference


def publish_component_folds(output_dir, *, components=None, local_selector=None):
    """Append fit provenance atomically, never replacing an existing fit record."""
    from exact.runtime.encoders import _write

    destination = Path(output_dir) / "component-fit-folds.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.with_suffix(".json.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        report = (
            json.loads(destination.read_text())
            if destination.exists()
            else {
                "schema": "exact-component-native-folds-v1",
                "membership_policy": "native_sorted_sources_seeded_shuffle_not_planner_hash_order",
                "components": {},
            }
        )
        if report.get("schema") != "exact-component-native-folds-v1" or not isinstance(
            report.get("components"), dict
        ):
            raise ValueError("Invalid native component OOF provenance")
        for name, item in (components or {}).items():
            if name in report["components"] and report["components"][name] != item:
                raise ValueError("Recorded native component OOF provenance changed")
            report["components"][name] = item
        if local_selector is not None:
            if "local_selector" in report and report["local_selector"] != local_selector:
                raise ValueError("Recorded local selector ownership changed")
            report["local_selector"] = local_selector
        _write(destination, report)
    return report


def record_component_fit(output_dir, name, path, payload, *, reference):
    """Record actual native OOF groups, without claiming the planner's other split."""
    if not any(
        os.getenv(key) == "1"
        for key in ("EXACT_COMPONENT_TRAIN_FACTS", "EXACT_COMPONENT_EXPORT_TRAIN")
    ):
        return
    membership = {}
    folds = payload.get("folds", payload.get("fit_provenance", {}).get("folds"))
    if folds is not None:
        for fold in folds:
            train, heldout = set(fold["train_sources"]), set(fold["heldout_sources"])
            if train & heldout or set(membership) & heldout:
                raise ValueError("Native component OOF groups overlap")
            membership.update({source: fold["fold"] for source in heldout})
        if any(
            set(fold["train_sources"]) | set(fold["heldout_sources"]) != set(membership)
            for fold in folds
        ):
            raise ValueError("Native component OOF groups are incomplete")
    else:
        for row in payload["oof_predictions"]:
            source, fold = row["source"], row["fold"]
            if membership.setdefault(source, fold) != fold:
                raise ValueError("Native component OOF split divides a source group")
    if not membership:
        raise ValueError("Native component OOF membership is missing")
    targets = {}
    for source, target in reference:
        if source in membership:
            targets.setdefault(target, set()).add(source)
    crossing = {
        target: sorted(sources)
        for target, sources in targets.items()
        if len({membership[source] for source in sources}) > 1
    }
    item = {
        "artifact": {"path": str(Path(path).resolve()), "sha256": sha256_file(Path(path))},
        "actual_fold_map": membership,
        "actual_fold_sha256": fingerprint(membership),
        "reference_component_overlap": {
            "scope": "train_positive_reference_only",
            "targets_crossing_folds": len(crossing),
            "affected_sources": sorted(
                {source for sources in crossing.values() for source in sources}
            ),
            "crossing_targets_sha256": fingerprint(crossing),
            "interpretation": "source_grouped_oof_does_not_establish_target_component_independence",
        },
    }
    publish_component_folds(output_dir, components={name: item})
