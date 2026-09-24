"""Bounded, gold-free observations captured at matcher stage boundaries.

This module never ranks, thresholds, or changes a candidate. Events contain the
actual inputs at the boundary, so consumers need not invent historic competitors.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

SCHEMA_VERSION = 1
STAGES = (
    "retrieval",
    "prefilter",
    "pair_scoring",
    "llm_signal",
    "selection",
    "threshold",
    "cardinality",
    "extraction",
    "relation_typing",
    "repair",
)


def digest(value: Any) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest()
    )


def scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return None


def typed_pair(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    def kind(value):
        return str(scalar(getattr(value, "value", value)) or "class")

    return (
        str(row.get("Src", row.get("src_iri", row.get("SrcEntity", "")))),
        kind(row.get("SrcKind", row.get("src_kind"))),
        str(row.get("Tgt", row.get("tgt_iri", row.get("TgtEntity", "")))),
        kind(row.get("TgtKind", row.get("tgt_kind"))),
    )


def pair_key(row: Mapping[str, Any]) -> str:
    return digest(typed_pair(row))


def feature_terms(item: Mapping[str, Any]) -> dict[str, Any]:
    """Canonical semantic inputs, excluding display syntax and availability metadata."""
    terms = {
        key: item[key]
        for key in (
            "subject_iri",
            "object_iri",
            "rel_iri",
            "property_iri",
            "prop_iri",
            "entity_iri",
            "value",
            "literal_terms",
            "literal_lexical_form",
            "datatype",
            "language",
            "type_closure",
            "role",
            "source_axiom_refs",
            "derivation",
        )
        if item.get(key) is not None
    }
    if "grouped_semantic_features" in item:
        terms["grouped_semantic_features"] = sorted(
            [feature_terms(feature) for feature in item["grouped_semantic_features"]], key=digest
        )
    return terms


def candidate_values(row: Mapping[str, Any]) -> dict[str, Any]:
    """Publish numeric stage families and explicit controls, excluding reference labels."""
    explicit = {
        "cand_channels",
        "candidate_joint_rank",
        "nil_rank",
        "nil_mode",
        "nil_ranking_scale",
        "nil_absence_semantics",
        "extraction_selected",
        "threshold_positive",
        "saved_alignment_member",
        "source_choice_reason",
    }
    prefixes = ("S_", "P_", "Q_", "s_", "q_", "w_", "cand_", "selection_", "llm_gate_")
    explicit.add("p_llm")
    return {
        str(key): scalar(value)
        for key, value in row.items()
        if (key in explicit or str(key).startswith(prefixes)) and scalar(value) is not None
    }


def payload(row: Mapping[str, Any]) -> dict[str, Any]:
    existing = row.get("candidate_decision")
    if isinstance(existing, str):
        existing = json.loads(existing)
    existing = existing if isinstance(existing, dict) else {}
    source, sk, target, tk = typed_pair(row)
    return {
        "schema_version": SCHEMA_VERSION,
        "pair_key": pair_key(row),
        "source": {"iri": source, "kind": sk},
        "target": {"iri": target, "kind": tk},
        "values": candidate_values(row),
        "events": list(existing.get("events", [])),
    }


def event(
    stage: str,
    *,
    outcome: str = "retained",
    reason: str,
    implementation: str,
    values: Mapping[str, Any] | None = None,
    competitors=(),
    config=None,
    status: str = "completed",
) -> dict[str, Any]:
    if stage not in STAGES:
        raise ValueError(f"Unknown decision stage: {stage}")
    result = {
        "stage": stage,
        "status": status,
        "outcome": outcome,
        "reason_code": reason,
        "values": dict(values or {}),
        "competitor_pair_keys": sorted(set(competitors)),
        "implementation_id": implementation,
        "config_hash": digest(config or {}),
        "config": config or {},
        "provenance_status": "recorded",
    }
    result["event_id"] = digest(result)
    return dict(json.loads(json.dumps(result, ensure_ascii=False, allow_nan=False)))


def append_event(row: Mapping[str, Any], observation: dict[str, Any]) -> dict[str, Any]:
    result = payload(row)
    if observation["event_id"] not in {item["event_id"] for item in result["events"]}:
        result["events"].append(observation)
    return result


def observe_dataset(dataset, *, restored=False):
    """Freeze retrieval/prefilter inputs at the dataset preparation boundary.

    Keep this separate from model features and retain candidates removed before
    scoring. A restored dataset cannot establish unrecorded historic events.
    """
    if restored:
        dataset._candidate_stage_observations = {}
        return
    candidates = getattr(dataset, "_candidates", None)
    if candidates is None:
        return
    output = {pair_key(row): row for row in dataset.dataframe.to_dict("records")}
    pool = dataset.candidate_pool_manifest
    generated = pool.get("origin") == "generated"
    prefilter_enabled = dataset.filter_exact_matches or (
        dataset.exact_matches is not None and not dataset.exact_matches.empty
    )
    config = {
        "filter_exact_matches": bool(prefilter_enabled),
        "drop_exact_match_sources": dataset.drop_exact_match_sources,
    }
    counts = {}
    observations = {}
    for row in candidates.to_dict("records"):
        source, sk, _, _ = typed_pair(row)
        group = (source, sk)
        counts[group] = counts.get(group, 0) + 1
        current = {
            **{
                key: value
                for key, value in row.items()
                if key in {"Src", "Tgt", "SrcKind", "TgtKind"}
            },
            **candidate_values(row),
            "cand_rank": counts[group],
            "cand_ordering": "saved_candidate_pool_order",
            "cand_tie_rule": "original_saved_order",
            "cand_rank_provenance": "derived_from_saved_artifacts",
        }
        current["candidate_decision"] = append_event(
            current,
            event(
                "retrieval",
                status="completed" if generated else "not_run",
                reason="candidate_pool_generated" if generated else "provided_candidate_pool",
                implementation="exact.impl.datasets.base.BaseAlignmentDataset.process",
                values=candidate_values(current),
                config={"pool_fingerprint": pool.get("fingerprint"), "origin": pool.get("origin")},
            ),
        )
        retained = pair_key(row) in output
        protected = bool(output.get(pair_key(row), {}).get("prefiltered", False))
        current["candidate_decision"] = append_event(
            current,
            event(
                "prefilter",
                status="completed" if prefilter_enabled else "not_run",
                outcome="retained" if retained else "rejected",
                reason=(
                    (
                        "exact_match_protected"
                        if protected
                        else "prefilter_retained" if retained else "exact_source_prefiltered"
                    )
                    if prefilter_enabled
                    else "prefilter_disabled"
                ),
                implementation="exact.impl.datasets.base.BaseAlignmentDataset.process",
                config=config,
            ),
        )
        if protected:
            for stage in ("pair_scoring", "llm_signal", "selection"):
                current["candidate_decision"] = append_event(
                    current,
                    event(
                        stage,
                        status="not_run",
                        outcome="not_applicable",
                        reason="exact_match_protected",
                        implementation="exact.pipeline",
                    ),
                )
        if not retained:
            for stage in (
                "pair_scoring",
                "llm_signal",
                "selection",
                "threshold",
                "cardinality",
                "extraction",
            ):
                current["candidate_decision"] = append_event(
                    current,
                    event(
                        stage,
                        status="not_run",
                        outcome="not_applicable",
                        reason="prefilter_removed",
                        implementation="exact.pipeline",
                    ),
                )
        observations[pair_key(row)] = current
    dataset._candidate_stage_observations = observations


def observe_selection(frame, *, implementation: str, config: dict[str, Any]) -> None:
    """Capture the listwise comparison pool at the selector return boundary."""
    if frame.empty:
        return
    groups = ["Src"] + (["SrcKind"] if "SrcKind" in frame else [])
    pools = {}
    for _, group in frame.groupby(groups, sort=False, dropna=False):
        keys = [pair_key(row) for row in group.to_dict("records")]
        for index in group.index:
            pools[index] = keys
    observations = []
    for index, row in frame.iterrows():
        values = candidate_values(row)
        outcome = (
            "abstained"
            if values.get("selection_abstained")
            else "selected" if values.get("selection_winner") else "not_selected"
        )
        observations.append(
            append_event(
                row,
                event(
                    "selection",
                    outcome=outcome,
                    reason=str(values.get("selection_reason") or "selector_returned"),
                    implementation=implementation,
                    values=values,
                    competitors=[key for key in pools[index] if key != pair_key(row)],
                    config=config,
                ),
            )
        )
    frame["candidate_decision"] = observations


def _competition_pools(rows):
    source_pools, target_pools = {}, {}
    for row in rows:
        source, sk, target, tk = typed_pair(row)
        key = pair_key(row)
        source_pools.setdefault((source, sk), set()).add(key)
        target_pools.setdefault((target, tk), set()).add(key)
    return source_pools, target_pools


def _competitors(row, pools):
    source, sk, target, tk = typed_pair(row)
    return (pools[0].get((source, sk), set()) | pools[1].get((target, tk), set())) - {pair_key(row)}


def observe_extraction(frame, selected, *, threshold, config, implementation):
    """Record threshold and extraction input/output at the actual return boundary."""
    chosen = {
        pair_key(
            {
                "Src": m.head,
                "Tgt": m.tail,
                "SrcKind": str(getattr(m.src_kind, "value", m.src_kind)),
                "TgtKind": str(getattr(m.tgt_kind, "value", m.tgt_kind)),
            }
        )
        for m in selected
    }
    rows = frame.to_dict("records")
    pools = _competition_pools(rows)
    observations = []
    for row in rows:
        key = pair_key(row)
        score = scalar(row.get("S_final"))
        eligible = score is not None and (threshold is None or score >= threshold)
        current = dict(row)
        current["candidate_decision"] = append_event(
            current,
            event(
                "threshold",
                status="not_run" if threshold is None else "completed",
                outcome="retained" if eligible else "rejected",
                reason="threshold_passed" if eligible else "below_threshold",
                implementation=implementation,
                values={"S_final": score, "threshold": threshold},
                config=config,
            ),
        )
        competitors = _competitors(row, pools)
        current["candidate_decision"] = append_event(
            current,
            event(
                "cardinality",
                status="completed" if config.get("source_cardinality") is not None else "not_run",
                outcome="retained" if key in chosen else "not_selected",
                reason=(
                    "extraction_cardinality"
                    if config.get("source_cardinality") is not None
                    else "cardinality_unbounded"
                ),
                implementation=implementation,
                competitors=competitors,
                config=config,
            ),
        )
        observations.append(
            append_event(
                current,
                event(
                    "extraction",
                    outcome="selected" if key in chosen else "not_selected",
                    reason=(
                        "selected"
                        if key in chosen
                        else "below_threshold" if not eligible else "extraction_omission"
                    ),
                    implementation=implementation,
                    values={"S_final": score},
                    competitors=competitors,
                    config=config,
                ),
            )
        )
    frame["candidate_decision"] = observations
    frame["extraction_selected"] = [pair_key(row) in chosen for row in rows]


def write_candidate_decisions(
    directory: Path, frame, typed, *, policy=None, pool=None, initial=None
) -> Path:
    """Export final stage observations; persisted mapping membership is joined by readers."""
    emitted = {pair_key(row): row for row in typed.to_dict("records")}
    inputs = dict(initial or {})
    for raw in frame.to_dict("records"):
        previous = inputs.get(pair_key(raw), {})
        merged = {**previous, **raw}
        prior_events = payload(previous)["events"] if previous else []
        current = payload(raw)
        current["events"] = list(
            {item["event_id"]: item for item in prior_events + current["events"]}.values()
        )
        merged["candidate_decision"] = current
        inputs[pair_key(raw)] = merged
    rows = []
    for raw in inputs.values():
        row = dict(raw)
        typed_row = emitted.get(pair_key(row))
        row["candidate_decision"] = append_event(
            row,
            event(
                "relation_typing",
                status=(
                    "not_run"
                    if (policy or {}).get("relation_prediction") == "none"
                    else "completed"
                ),
                outcome="retained" if typed_row else "not_applicable",
                reason=(
                    "default_equivalence_convention"
                    if (policy or {}).get("relation_prediction") == "none"
                    else "typed_output" if typed_row else "not_in_typed_output"
                ),
                implementation="exact.io.relations.predict_relations",
                values={},
                config=policy or {},
            ),
        )
        row["candidate_decision"] = append_event(
            row,
            event(
                "repair",
                status="not_run",
                outcome="not_applicable",
                reason="production_repair_not_enabled",
                implementation="exact.pipeline",
            ),
        )
        observation = payload(row)
        observation["events"].sort(key=lambda item: STAGES.index(item["stage"]))
        observation["relation"] = (
            {
                key: scalar(value)
                for key, value in typed_row.items()
                if key in {"Relation", "relation_confidence", "relation_evidence"}
            }
            if typed_row
            else None
        )
        rows.append(observation)
    result = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "candidate_decisions",
        "policy": policy or {},
        "candidate_pool": pool,
        "relation_anchors": typed.attrs.get("relation_anchors", []),
        "relation_abstentions": typed.attrs.get("relation_abstentions", []),
        "records": rows,
        "reference_labels_used": False,
    }
    destination = Path(directory) / "candidate_decisions.json"
    temporary = destination.with_suffix(".json.partial")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False, default=str) + "\n"
    )
    temporary.replace(destination)
    return destination


def observe_cardinality(
    frame, inputs, outputs, *, source_cardinality=None, target_cardinality=None, protected_pairs=()
):
    """Record real conflict inputs at the prefilter/cardinality return boundary."""
    if frame is None or frame.empty:
        return

    def row_for(mapping):
        return {
            "Src": mapping.head,
            "Tgt": mapping.tail,
            "SrcKind": getattr(mapping.src_kind, "value", mapping.src_kind),
            "TgtKind": getattr(mapping.tgt_kind, "value", mapping.tgt_kind),
        }

    input_rows = [row_for(mapping) for mapping in inputs]
    input_keys = {pair_key(row) for row in input_rows}
    pools = _competition_pools(input_rows)
    protected_pairs = set(protected_pairs)
    output_keys = {pair_key(row_for(mapping)) for mapping in outputs}
    config = {
        "source_cardinality": source_cardinality,
        "target_cardinality": target_cardinality,
        "protected_exact_pairs_hash": digest(sorted([list(pair) for pair in protected_pairs])),
    }
    observations = []
    for row in frame.to_dict("records"):
        key = pair_key(row)
        src, _, tgt, _ = typed_pair(row)
        competitors = _competitors(row, pools)
        current = dict(row)
        if (src, tgt) in protected_pairs:
            current["candidate_decision"] = append_event(
                current,
                event(
                    "prefilter",
                    outcome="retained",
                    reason="exact_match_protected",
                    implementation="exact.core.contracts.trainer.apply_prefilter",
                    config=config,
                ),
            )
        observations.append(
            append_event(
                current,
                event(
                    "cardinality",
                    outcome=(
                        "retained"
                        if key in output_keys
                        else "rejected" if key in input_keys else "not_applicable"
                    ),
                    reason=(
                        "cardinality_retained"
                        if key in output_keys
                        else (
                            "target_or_source_conflict"
                            if key in input_keys
                            else "not_in_cardinality_input"
                        )
                    ),
                    implementation="exact.core.contracts.trainer.apply_prefilter",
                    competitors=competitors,
                    config=config,
                ),
            )
        )
    frame["candidate_decision"] = observations
