"""Portable indexed candidate decisions adapted through the stable RunReader seam."""

from __future__ import annotations

import json
import math
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

from .artifacts import read_metadata, relative_path
from .contracts import (
    CONTRACT_VERSION,
    DomainError,
    EntityRef,
    canonical_hash,
    decode_cursor,
    encode_cursor,
    file_hash,
)
from .evidence_metadata import exact_match_basis, pair_evidence_metadata
from .sqlite_safety import DECISION_TABLES, readonly_connection

_SCORE_MEANINGS = {
    "P_rank": "Candidate-set probability-like ranking value; not an ordinal rank.",
    "P_match": "Producer acceptance score; calibration requires an identified validation artifact.",
    "S_final": "Final matcher decision score; not a probability of semantic equivalence.",
    "P_nil": "Explicit NIL value with the producer bounded-pool absence semantics.",
    "Q_nil": "NIL mass on the saved joint candidate/NIL scale.",
    "Q_match": "Candidate mass on the saved joint candidate/NIL scale.",
    "w_llm": "Mixture weight; neither correctness probability nor counterfactual necessity.",
}


ADAPTER_REVISION = "decision-adapter/7"
DEPENDENCY_INVENTORY_SCHEMA = "exact-decision-dependency-inventory/1"


def dependency_inventory_hash(dependencies):
    """Content address the complete verified run dependency map, without repeating it."""
    return canonical_hash({"schema": DEPENDENCY_INVENTORY_SCHEMA, "dependencies": dependencies})


def _source_candidates(document, artifact_ref):
    """Retain saved typed candidates without treating their array order as a rank.

    The source-decision writer sorts its candidate array by target IRI. It can
    contain candidates absent from both the explanation and final mapping files.
    This fixed scalar allowlist never copies private/reference fields or nested
    producer explanations, even when their names resemble score prefixes.
    """
    from exact.runs.decisions import pair_key

    numeric = {
        "U",
        "U_ind",
        "U_dis",
        "p_llm",
        "S_base",
        "S_pair_final",
        "S_pair_pre_calibration",
        "S_select",
        "P_select",
        "P_rank",
        "P_match",
        "S_final",
        "S_before_source_choice",
        "Q_pool_miss",
        "Q_match",
        "Q_nil",
        "P_nil",
        "P_match_pre_nil",
        "P_rank_without_nil",
        "nil_rank",
        "candidate_joint_rank",
        "selection_accept_threshold",
        "selection_margin",
        "selection_entropy",
        "selection_no_match_prob",
        "selection_evidence_support",
        "selection_distinctive",
        "selection_utility",
        "selection_target_cardinality",
        "cand_rank",
    }
    controls = {
        "nil_mode",
        "nil_ranking_scale",
        "nil_absence_semantics",
        "cand_channels",
        "cand_ordering",
        "cand_tie_rule",
        "cand_rank_provenance",
        "source_choice_reason",
    }
    flags = {"threshold_positive", "selection_nil_winner", "selection_target_conflict_enabled"}
    output = {}
    for source_index, source in enumerate(document.get("records", [])):
        candidates = source.get("candidates", [])
        if not isinstance(candidates, list):
            raise DomainError(
                "invalid_source_candidates", "Saved source candidates are invalid.", 422
            )
        for index, candidate in enumerate(candidates):
            if (
                not isinstance(candidate, dict)
                or not isinstance(source.get("Src"), str)
                or not source["Src"]
                or not isinstance(candidate.get("target"), str)
                or not candidate["target"]
                or (
                    source.get("SrcKind") is not None
                    and candidate.get("SrcKind") is not None
                    and source["SrcKind"] != candidate["SrcKind"]
                )
            ):
                raise DomainError(
                    "invalid_source_candidates", "Saved typed candidate identity is invalid.", 422
                )
            record = {
                "src_iri": source["Src"],
                "src_kind": candidate.get("SrcKind") or source.get("SrcKind") or "class",
                "tgt_iri": candidate["target"],
                "tgt_kind": candidate.get("TgtKind") or "class",
            }
            key = pair_key(record)
            if key in output:
                raise DomainError(
                    "invalid_source_candidates", "Duplicate saved typed candidate identity.", 422
                )
            values = {
                name: value
                for name, value in candidate.items()
                if (name in numeric and type(value) in {int, float} and math.isfinite(value))
                or (name in controls and isinstance(value, str))
                or (name in flags and type(value) is bool)
            }
            output[key] = {
                "record": record,
                "values": values,
                "reference": {
                    "artifact_ref": artifact_ref,
                    "source_record_index": source_index,
                    "candidate_index": index,
                    "ordering": "source_decisions_candidates_array",
                    "scores": _scores(values, "saved_source_decision"),
                },
            }
    return output


def _global_mapping_path(reader, manifest):
    """Honor one hash-bound manifest mapping in the producer's named TSV format."""
    canonical = reader.layout.mapping_path("global")
    declared = []
    for artifact in manifest.get("artifacts", []):
        name = artifact.get("path", "")
        if (
            artifact.get("kind") == "alignment"
            and isinstance(name, str)
            and (
                Path(name).name == "maps_global.tsv" or Path(name).name.endswith(".maps_global.tsv")
            )
        ):
            declared.append(artifact)
    if len(declared) > 1:
        raise DomainError(
            "ambiguous_alignment", "Multiple saved global mappings are declared.", 409
        )
    if not declared:
        return canonical
    artifact = declared[0]
    path = relative_path(reader.layout.root, artifact["path"])
    raw_hash = artifact.get("sha256")
    if raw_hash is None and canonical.is_file() and canonical.resolve() == path.resolve():
        # The stable reader's pre-existing canonical seam supports historical
        # mappings without a manifest hash. Import binds their actual bytes.
        return canonical
    expected = raw_hash.removeprefix("sha256:") if isinstance(raw_hash, str) else ""
    if (
        not re.fullmatch(r"[a-f0-9]{64}", expected)
        or not path.is_file()
        or file_hash(path) != "sha256:" + expected
        or ("bytes" in artifact and artifact["bytes"] != path.stat().st_size)
    ):
        raise DomainError(
            "artifact_conflict", "Saved global mapping differs from its manifest.", 409
        )
    if canonical.is_file() and canonical.resolve() != path.resolve():
        raise DomainError(
            "ambiguous_alignment", "Multiple saved global mappings are available.", 409
        )
    return path


def _score(name, value, stage):
    return {
        "name": name,
        "stage": stage,
        "value": value,
        "meaning": _SCORE_MEANINGS.get(
            name, "Producer stage value; no established correctness calibration."
        ),
        "range": None,
        "calibration_status": "not_established",
        "calibration_artifact": None,
    }


def _scores(values, stage):
    return [
        _score(key, value, stage)
        for key, value in sorted(values.items())
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and key not in {"nil_rank", "candidate_joint_rank", "cand_rank"}
    ]


def _summary_scores(values, events):
    producing_stage = {}
    for item in events:
        if item["stage"] in {"retrieval", "pair_scoring", "llm_signal", "selection"}:
            for name, value in item.get("values", {}).items():
                if values.get(name) == value:
                    producing_stage[name] = item["stage"]
    scores = _scores(values, "saved_artifact")
    for score in scores:
        score["stage"] = producing_stage.get(score["name"], "saved_artifact")
    return scores


def _event(raw, key_map, dependencies, *, source=None, target=None):
    result = {
        "stage": raw["stage"],
        "status": raw["status"],
        "outcome": raw["outcome"],
        "reason_code": raw["reason_code"],
        "scores": _scores(raw.get("values", {}), raw["stage"]),
        "competitor_pair_ids": [
            key_map[key] for key in raw.get("competitor_pair_keys", []) if key in key_map
        ],
        "artifact_refs": list(dependencies)
        + ([raw["config_hash"]] if raw.get("config_hash") else []),
        "implementation_id": raw.get("implementation_id"),
        "provenance_status": raw.get("provenance_status", "unavailable"),
    }

    if raw["stage"] == "prefilter":
        result["exact_basis"] = exact_match_basis(raw, source=source, target=target)
    return result


def _missing_event(stage):
    result = {
        "stage": stage,
        "status": "not_recorded",
        "outcome": "unknown",
        "reason_code": "historical_stage_record_not_exported",
        "scores": [],
        "competitor_pair_ids": [],
        "artifact_refs": [],
        "implementation_id": None,
        "provenance_status": "unavailable",
    }

    if stage == "prefilter":
        result["exact_basis"] = exact_match_basis(result)
    return result


def _feature_items(record):
    for channel, payload in (record.get("triple_attributions") or {}).items():
        if channel == "hierarchy":
            for family, sides in payload.items():
                for side in ("source", "target"):
                    for item in sides.get(side, []):
                        yield channel, side, family, item
        elif isinstance(payload, dict):
            for side in ("source", "target"):
                for item in payload.get(side, []):
                    yield channel, side, None, item
    for side in ("source", "target"):
        for item in (record.get("attributes") or {}).get(side, []):
            yield "attributes", side, None, item


def selected_evidence(record, source, target, *, resolver=None):
    from exact.runs.decisions import feature_terms, pair_key

    items = []
    for ordinal, (channel, side, family, raw) in enumerate(_feature_items(record)):
        item = dict(raw)
        entity = source if side == "source" else target
        # Legacy display-only records never acquire a fabricated semantic identity.
        semantic = feature_terms(item)
        has_identity = bool(
            (semantic.get("subject_iri") and semantic.get("object_iri"))
            or (semantic.get("property_iri", semantic.get("prop_iri")) and "value" in semantic)
        )
        if semantic.get("rel_iri") in ("domain", "range"):
            semantic["role"] = "property_" + semantic["rel_iri"]
            semantic["rel_iri"] = "http://www.w3.org/2000/01/rdf-schema#" + semantic["rel_iri"]
        feature_id = (
            canonical_hash(
                {
                    "ontology_version_id": entity["ontology_version_id"],
                    "entity_kind": entity["kind"],
                    "interpretation": item.get("interpretation", "projected"),
                    "family": family,
                    "terms": semantic,
                }
            )
            if has_identity
            else None
        )
        resolution = (
            resolver({**item, "family": family}, entity) if resolver and has_identity else {}
        )
        fact_ids = list(resolution.get("fact_ids") or [])
        for feature in item.get("grouped_semantic_features", []):
            if resolver:
                fact_ids.extend(resolver({**feature, "family": family}, entity).get("fact_ids", []))
        fact_ids = sorted(set(fact_ids))
        status = "available" if fact_ids else "not_exported"
        items.append(
            {
                "evidence_id": canonical_hash(
                    {
                        "side": side,
                        "channel": channel,
                        "feature_id": feature_id,
                        "legacy": (
                            {
                                "record": pair_key(record),
                                "ordinal": ordinal,
                                "alias": item.get("item_id"),
                            }
                            if not feature_id
                            else None
                        ),
                    }
                ),
                "channel": channel,
                "family": family,
                "side": side,
                "role": semantic.get("role", family or channel),
                "entity": entity,
                "feature_id": feature_id,
                "fact_ids": fact_ids,
                "source_axiom_refs": list(item.get("source_axiom_refs") or []),
                "axiom_origins": list(item.get("axiom_origins") or []),
                "semantic_terms": semantic if has_identity else {},
                "historical_item_alias": (
                    "legacy:" + str(item["item_id"]) if item.get("item_id") else None
                ),
                "display": {
                    key: item[key] for key in ("triple", "text", "property", "value") if key in item
                },
                "values": {
                    key: item[key]
                    for key in ("support", "importance", "weight", "edge_ic", "unsupported_mass")
                    if key in item
                },
                "interpretation": item.get("interpretation", "projected"),
                "derivation": item.get("derivation"),
                "provenance_status": resolution.get(
                    "provenance_status",
                    "derived_from_saved_artifacts" if fact_ids else "provenance_unavailable",
                ),
                "status": status,
                "origin_resolution": (
                    "resolved"
                    if fact_ids
                    else "unresolved" if resolver and has_identity else "not_exported"
                ),
                "graph_projection": (
                    item.get("graph_projection")
                    if item.get("graph_projection") in {"supported", "unsupported"}
                    else "not_recorded"
                ),
                "reason": (
                    None
                    if fact_ids
                    else "Original ontology facts were not recoverable from the saved record."
                ),
            }
        )
    return items


def _ontology_input_binding(reader, manifest, producer, dependencies, expected):
    """Check recorded root bytes; shared IRIs never establish an ontology version."""
    observed = {"source": set(), "target": set()}

    def add(side, value):
        if value is None:
            return
        value = str(value).removeprefix("sha256:")
        if not re.fullmatch(r"[a-f0-9]{64}", value):
            raise DomainError(
                "invalid_ontology_provenance", "Invalid recorded ontology digest", 409
            )
        observed[side].add("sha256:" + value)

    pools = [producer.get("candidate_pool") or {}]
    pool_path = reader.layout.dataset_dir / "candidate_pool_manifest.json"
    if pool_path.is_file():
        dependencies[reader.layout.relative(pool_path)] = file_hash(pool_path)
        pools.append(json.loads(pool_path.read_bytes()))
    for pool in pools:
        for side in observed:
            add(side, ((pool.get("inputs") or {}).get(side) or {}).get("sha256"))
    if reader.layout.config_path.is_file():
        import yaml

        config = yaml.safe_load(reader.layout.config_path.read_text()) or {}
        if isinstance(config, dict):
            for side in observed:
                add(side, config.get(side + "_hash"))
    for side in observed:
        stack = (manifest.get("ontology_stack") or {}).get(side) or {}
        closure = (stack.get("core") or {}).get("closure") or {}
        documents = closure.get("source_documents") or []
        # Multi-document manifests without an explicit root key are ambiguous.
        if closure.get("view_kind") == "snapshot" and len(documents) == 1:
            add(side, documents[0].get("source_sha256"))
    result = {}
    for side, digests in observed.items():
        if len(digests) > 1:
            raise DomainError("ontology_binding_mismatch", "Run ontology digests disagree", 409)
        actual = next(iter(digests), None)
        wanted = expected[side]
        if actual and wanted and actual != wanted:
            raise DomainError(
                "ontology_binding_mismatch", "Run and context ontology bytes differ", 409
            )
        result[side] = {
            "producer_sha256": actual,
            "context_sha256": wanted,
            "status": (
                "verified"
                if actual and wanted
                else "legacy_unverified" if not actual else "context_unverified"
            ),
        }
    result["status"] = (
        "verified" if all(result[s]["status"] == "verified" for s in observed) else "unverified"
    )
    return result


def import_run(
    run_dir: Path,
    output_dir: Path,
    *,
    source_ontology_version_id: str,
    target_ontology_version_id: str,
    run_id: str | None = None,
    context_revision: str | None = None,
    source_root_sha256: str | None = None,
    target_root_sha256: str | None = None,
    expected_artifacts: dict[str, str] | None = None,
    evidence_resolver: Callable | None = None,
) -> dict:
    """Freeze a relocatable gold-free indexed view; never rerun scoring or NLP."""
    from exact.runs.decisions import STAGES, candidate_values, pair_key, typed_pair
    from exact.runs.reader import RunReader

    reader = RunReader.open(Path(run_dir))
    manifest = reader.manifest()
    dependencies = {}
    for path in (
        reader.layout.manifest_path,
        reader.layout.config_path,
        reader.layout.explanation_index_path,
        reader.layout.full_explanations_path,
    ):
        if path.is_file():
            dependencies[reader.layout.relative(path)] = file_hash(path)
    # Explanation-index bindings include every shard, not just a mutable locator.
    if reader.layout.explanation_index_path.is_file():
        index = json.loads(reader.layout.explanation_index_path.read_text())
        for shard in [*index.get("shards", {}).values(), *index.get("overlays", [])]:
            path = reader.layout.explanations_dir / shard["path"]
            dependencies[reader.layout.relative(path)] = file_hash(path)
    raw_decisions = {}
    decision_path = reader.layout.candidate_decisions_path
    producer = {}
    if decision_path.is_file():
        dependencies[reader.layout.relative(decision_path)] = file_hash(decision_path)
        producer = reader.candidate_decisions() or {}
        if producer.get("schema_version") != 1:
            raise DomainError(
                "unsupported_decision_schema", "Unsupported candidate decision artifact."
            )
        raw_decisions = {entry["pair_key"]: entry for entry in producer.get("records", [])}
    ontology_binding = _ontology_input_binding(
        reader,
        manifest,
        producer,
        dependencies,
        {"source": source_root_sha256, "target": target_root_sha256},
    )
    source_decisions = reader.source_decisions() or {}
    if reader.layout.source_decisions_path.is_file():
        dependencies[reader.layout.relative(reader.layout.source_decisions_path)] = file_hash(
            reader.layout.source_decisions_path
        )
    saved_candidates = _source_candidates(
        source_decisions,
        dependencies.get(reader.layout.relative(reader.layout.source_decisions_path)),
    )
    run_id = run_id or str(manifest.get("run_id") or canonical_hash(dependencies))
    records: dict[str, dict[str, Any]] = {}
    try:
        for record in reader.iter_explanations():
            records[pair_key(record)] = record
    except FileNotFoundError:
        pass
    explained_keys = set(records)
    for key, candidate in saved_candidates.items():
        records.setdefault(key, candidate["record"])
    for key, decision in raw_decisions.items():
        records.setdefault(
            key,
            {
                "src_iri": decision["source"]["iri"],
                "src_kind": decision["source"]["kind"],
                "tgt_iri": decision["target"]["iri"],
                "tgt_kind": decision["target"]["kind"],
            },
        )
    memberships: dict[str, dict[str, Any]] = {}
    alignment_path = _global_mapping_path(reader, manifest)
    if alignment_path.is_file():
        dependencies[reader.layout.relative(alignment_path)] = file_hash(alignment_path)
        from exact.utils.data import read_table

        for raw in read_table(alignment_path).to_dict("records"):
            memberships[pair_key(raw)] = raw
            source_iri, sk, target_iri, tk = typed_pair(raw)
            records.setdefault(
                pair_key(raw),
                {"src_iri": source_iri, "src_kind": sk, "tgt_iri": target_iri, "tgt_kind": tk},
            )
    if expected_artifacts is not None:
        expected_paths = {
            str(Path(path).resolve()): sha for path, sha in expected_artifacts.items()
        }
        for relative, sha in dependencies.items():
            consumed_path = str((reader.layout.root / relative).resolve())
            if expected_paths.get(consumed_path) != sha:
                raise DomainError(
                    "unverified_run_input",
                    "Consumed run artifact is absent from the verified inventory",
                    409,
                )
    key_map = {}
    bindings = {}
    for key, record in records.items():
        source_iri, sk, target_iri, tk = typed_pair(record)
        refs = (
            EntityRef.model_validate(
                {"ontology_version_id": source_ontology_version_id, "iri": source_iri, "kind": sk}
            ).model_dump(),
            EntityRef.model_validate(
                {"ontology_version_id": target_ontology_version_id, "iri": target_iri, "kind": tk}
            ).model_dump(),
        )
        bindings[key] = refs
        key_map[key] = canonical_hash({"run_id": run_id, "source": refs[0], "target": refs[1]})
    revision = canonical_hash(
        {
            "implementation": ADAPTER_REVISION,
            "dependencies": dependencies,
            "source": source_ontology_version_id,
            "target": target_ontology_version_id,
            "context_revision": context_revision,
            "ontology_input_binding": ontology_binding,
        }
    )
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    database = directory / "decisions.sqlite"
    if database.exists():
        raise DomainError("artifact_conflict", "Decision package already exists.", 409)
    counts = {
        "pairs": 0,
        "selected_evidence": 0,
        "provenance_unavailable": 0,
        "source_decision_candidates": len(saved_candidates),
        "source_candidates_without_explanations": len(set(saved_candidates) - explained_keys),
    }
    inventory_hash = dependency_inventory_hash(dependencies)
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """CREATE TABLE pairs (id TEXT PRIMARY KEY, source TEXT NOT NULL,
            source_kind TEXT NOT NULL, target TEXT NOT NULL, target_kind TEXT NOT NULL,
            candidate TEXT NOT NULL, trace TEXT NOT NULL, evidence TEXT NOT NULL);
            CREATE INDEX pair_source ON pairs(source, source_kind, id);
            CREATE TABLE sources(id TEXT PRIMARY KEY, iri TEXT, kind TEXT, payload TEXT);
            CREATE INDEX source_order ON sources(id);"""
        )
        for key, record in records.items():
            source, target = bindings[key]
            raw = raw_decisions.get(key, record.get("candidate_decision") or {})
            saved_candidate = saved_candidates.get(key, {})
            values = {
                **saved_candidate.get("values", {}),
                **(raw.get("values") or candidate_values(record.get("confidences") or {})),
            }
            recorded = raw.get("events") or []
            retrieval: dict[str, Any] = next(
                (
                    item.get("values", {})
                    for item in reversed(recorded)
                    if item["stage"] == "retrieval"
                ),
                {},
            )
            selection: dict[str, Any] = next(
                (item for item in reversed(recorded) if item["stage"] == "selection"), {}
            )
            nil_config = (selection.get("config") or {}).get("nil") or {}
            ordinal_values = {**retrieval, **(selection.get("values") or {}), **values}
            observed_stages = {item["stage"] for item in recorded}
            events = [
                _event(item, key_map, dependencies.values(), source=source, target=target)
                for item in recorded
            ]
            events.extend(_missing_event(stage) for stage in STAGES if stage not in observed_stages)
            events.sort(key=lambda item: STAGES.index(item["stage"]))
            member = memberships.get(key)
            relation_raw = (member or {}).get(
                "Relation", (raw.get("relation") or {}).get("Relation")
            )
            relation = {
                "=": "equivalent",
                "<": "source_narrower",
                ">": "source_broader",
                "!=": "not_equivalent",
                "?": "unresolved",
            }.get(str(relation_raw))
            # A saved '=' alone cannot prove whether relation inference was enabled.
            basis = "unavailable"
            relation_config = (producer.get("policy") or {}).get("relation_prediction")
            if relation_config in ("off", "none", "default"):
                basis = "default_convention"
            elif relation_config in ("semantic", "semantic_graph", "semantic_entailment"):
                basis = "graph_closure_with_anchors"
            elif relation_config:
                basis = "recorded_other"
            trace = {
                "artifact_type": "pair_decision_trace",
                "contract_version": CONTRACT_VERSION,
                "run_id": run_id,
                "pair_id": key_map[key],
                "source": source,
                "target": target,
                "events": events,
                "saved_alignment_member": key in memberships if alignment_path.is_file() else None,
                "relation": relation,
                "relation_basis": basis,
                "assumption_refs": (
                    [canonical_hash(anchor) for anchor in producer.get("relation_anchors", [])]
                    if basis == "graph_closure_with_anchors"
                    else []
                ),
                "fixture_provenance": "derived_from_saved_artifacts",
            }
            evidence = selected_evidence(record, source, target, resolver=evidence_resolver)
            trace["evidence_metadata"] = pair_evidence_metadata(
                record, evidence, source, target, [inventory_hash], recorded
            )
            counts["pairs"] += 1
            counts["selected_evidence"] += len(evidence)
            counts["provenance_unavailable"] += sum(
                item["status"] != "available" for item in evidence
            )
            candidate = {
                "pair_id": key_map[key],
                "source": source,
                "target": target,
                "values": values,
                "scores": _summary_scores(values, recorded),
                "retrieval_channels": (
                    str(values.get("cand_channels") or "").split("|")
                    if values.get("cand_channels")
                    else []
                ),
                "ordinal_ranks": {
                    "candidate_joint_rank": ordinal_values.get("candidate_joint_rank"),
                    "nil_rank": ordinal_values.get("nil_rank"),
                    "joint_ordering": ordinal_values.get("nil_ranking_scale"),
                    "joint_tie_rule": nil_config.get("tie_rule"),
                    "retrieval_rank": ordinal_values.get("cand_rank"),
                    "retrieval_ordering": ordinal_values.get("cand_ordering"),
                    "retrieval_tie_rule": ordinal_values.get("cand_tie_rule"),
                    "retrieval_provenance": ordinal_values.get("cand_rank_provenance"),
                },
                "evidence_counts": record.get("evidence_counts", []),
                "nil": {
                    name: value
                    for name, value in values.items()
                    if "nil" in name.lower() or name == "Q_pool_miss"
                },
                "saved_alignment_member": trace["saved_alignment_member"],
                "membership_provenance": {
                    "status": (
                        "derived_from_saved_artifacts"
                        if alignment_path.is_file()
                        else "unavailable"
                    ),
                    "artifact_hash": dependencies.get(reader.layout.relative(alignment_path)),
                },
                "status": "available",
                "decision_status": "available" if recorded else "not_exported",
                "source_decision_record": saved_candidate.get("reference"),
            }
            connection.execute(
                "INSERT INTO pairs VALUES (?,?,?,?,?,?,?,?)",
                (
                    key_map[key],
                    source["iri"],
                    source["kind"],
                    target["iri"],
                    target["kind"],
                    json.dumps(candidate),
                    json.dumps(trace),
                    json.dumps(evidence),
                ),
            )
            source_id = canonical_hash(source)
            connection.execute(
                "INSERT OR IGNORE INTO sources VALUES (?,?,?,?)",
                (
                    source_id,
                    source["iri"],
                    source["kind"],
                    json.dumps({"entity_id": source_id, "entity": source}),
                ),
            )
        source_fields = {
            "candidate_count",
            "absence_semantics",
            "ontology_nil_probability",
            "benchmark_nil_probability",
            "pool_miss_probability",
            "in_pool_probability",
            "action",
            "empty_candidate_pool",
            "probability_scale",
        }
        for raw_source in source_decisions.get("records", []):
            entity = EntityRef.model_validate(
                {
                    "ontology_version_id": source_ontology_version_id,
                    "iri": raw_source["Src"],
                    "kind": raw_source.get("SrcKind", "class"),
                }
            ).model_dump()
            source_id = canonical_hash(entity)
            summary = {key: raw_source[key] for key in source_fields if key in raw_source}
            connection.execute(
                "INSERT OR REPLACE INTO sources VALUES (?,?,?,?)",
                (
                    source_id,
                    entity["iri"],
                    entity["kind"],
                    json.dumps(
                        {"entity_id": source_id, "entity": entity, "source_decision": summary}
                    ),
                ),
            )
    for relative, expected_hash in dependencies.items():
        artifact = reader.layout.root / relative
        if not artifact.is_file() or file_hash(artifact) != expected_hash:
            database.unlink(missing_ok=True)
            raise DomainError("artifact_conflict", "Run artifacts changed during import.", 409)
    result = {
        "contract_version": CONTRACT_VERSION,
        "artifact_type": "run_decision_package",
        "implementation": ADAPTER_REVISION,
        "run_id": run_id,
        "revision": revision,
        "source_ontology_version_id": source_ontology_version_id,
        "target_ontology_version_id": target_ontology_version_id,
        "context_revision": context_revision,
        "candidate_pool": producer.get("candidate_pool"),
        "ontology_input_binding": ontology_binding,
        "input_binding": {
            "status": "verified" if expected_artifacts is not None else "partial",
            "scope": "all_consumed_run_artifacts",
        },
        "relation_assumptions": {
            canonical_hash(anchor): anchor for anchor in producer.get("relation_anchors", [])
        },
        "dependencies": dependencies,
        "dependency_inventory_schema": DEPENDENCY_INVENTORY_SCHEMA,
        "dependency_inventory_hash": inventory_hash,
        "comparison_artifact_ref_semantics": {
            "legacy_and_default": "dependency_inventory_hash",
            "explicit_producer_refs": "as_recorded",
            "inventory": "Complete verified dependencies map in this manifest; not a new producer file.",
        },
        "counts": counts,
        "database": "decisions.sqlite",
        "database_hash": file_hash(database),
        "path": ".",
        "status": "available" if records else "not_exported",
    }
    (directory / "decisions.manifest.json").write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n"
    )
    return result


class DecisionStore:
    """Read-only SQLite keyset pages; a request never scans raw run files."""

    def __init__(self, directory: Path, *, verified_database=None):
        self.directory = Path(directory)
        self._manifest = read_metadata(self.directory / "decisions.manifest.json")
        if (
            self._manifest.get("implementation") in {"decision-adapter/6", "decision-adapter/7"}
            or "dependency_inventory_hash" in self._manifest
        ) and (
            self._manifest.get("dependency_inventory_schema") != DEPENDENCY_INVENTORY_SCHEMA
            or self._manifest.get("dependency_inventory_hash")
            != dependency_inventory_hash(self._manifest.get("dependencies"))
        ):
            raise DomainError("artifact_conflict", "Decision dependency inventory changed.", 409)
        if self._manifest.get("database") != "decisions.sqlite":
            raise DomainError("invalid_manifest", "Unsupported decision database locator", 422)
        self.database = relative_path(self.directory, self._manifest["database"])
        self._verified_stat = self._database_stat()
        # A containing service may have checked this exact inode and byte version
        # during readiness. Never reuse a bare digest across a file replacement.
        digest = (
            verified_database[0]
            if verified_database is not None and verified_database[1] == self._verified_stat
            else file_hash(self.database)
        )
        if (
            digest != self._manifest["database_hash"]
            or self._database_stat() != self._verified_stat
        ):
            raise DomainError("artifact_conflict", "Decision package hash changed.", 409)
        with self._connect():
            pass

    def manifest(self):
        return dict(self._manifest)

    def _database_stat(self):
        stat = self.database.stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

    @contextmanager
    def _connect(self):
        if self._database_stat() != self._verified_stat:
            raise DomainError(
                "artifact_conflict", "Decision package changed after verification.", 409
            )
        with readonly_connection(self.database, DECISION_TABLES) as connection:
            yield connection

    def _page(self, table, column, where, params, *, limit, cursor, policy_hash):
        if limit < 1 or limit > 100:
            raise DomainError("invalid_limit", "Page limit must be between 1 and 100.")
        scope = {
            "ontology_version_id": self._manifest["source_ontology_version_id"],
            "context_revision": self._manifest["revision"],
            "basis": "run_selected",
            "visibility_policy_hash": policy_hash,
            "filter_id": canonical_hash([table, where, params]),
        }
        after = decode_cursor(cursor, scope)
        if after is not None and not isinstance(after, str):
            raise DomainError("invalid_cursor", "Cursor key must be a string.", 422)
        with self._connect() as connection:
            total = connection.execute(
                f"SELECT count(*) FROM {table} WHERE {where}", params
            ).fetchone()[0]
            rows = connection.execute(
                f"SELECT id,{column} FROM {table} WHERE {where} AND id>? ORDER BY id LIMIT ?",
                (*params, after or "", limit + 1),
            ).fetchall()
        more = len(rows) > limit
        rows = rows[:limit]
        return {
            "items": [json.loads(row[1]) for row in rows],
            "returned_count": len(rows),
            "total_count": total,
            "next_cursor": encode_cursor(scope, rows[-1][0]) if more else None,
            "truncated": more,
            "scope": scope,
            "status": "available" if total else "absent_in_scope",
            "reason": None,
            "order": "pair_id_ascending" if table == "pairs" else "source_id_ascending",
        }

    def sources(self, *, limit=20, cursor=None, policy_hash=None):
        return self._page(
            "sources",
            "payload",
            "1=1",
            (),
            limit=limit,
            cursor=cursor,
            policy_hash=policy_hash or canonical_hash("exploration"),
        )

    def candidates(
        self, source_iri: str, source_kind="class", *, limit=20, cursor=None, policy_hash=None
    ):
        return self._page(
            "pairs",
            "candidate",
            "source=? AND source_kind=?",
            (source_iri, source_kind),
            limit=limit,
            cursor=cursor,
            policy_hash=policy_hash or canonical_hash("exploration"),
        )

    def _get(self, pair_id, column):
        with self._connect() as connection:
            row = connection.execute(
                f"SELECT {column} FROM pairs WHERE id=?", (pair_id,)
            ).fetchone()
        if row is None:
            raise DomainError("unknown_pair", "Unknown candidate pair.", 404)
        return json.loads(row[0])

    def pair(self, pair_id: str):
        return self._get(pair_id, "trace")

    def evidence(self, pair_id: str):
        return self._get(pair_id, "evidence")
