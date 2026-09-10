"""Post-scoring NIL evaluation from separately bound source annotations."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any


from exact.impl.models.selector.nil_head import STATUSES, nil_metrics
from exact.utils.data import read_table
from exact.utils.provenance import sha256_file


def _reciprocal_rank(record: dict[str, Any], targets: set[str]) -> float | None:
    candidates = record.get("candidates", [])
    relevant = [row for row in candidates if str(row["target"]) in targets]
    if not relevant:
        return 0.0
    if all(row.get("candidate_joint_rank") is not None for row in candidates):
        ranks = [float(row["candidate_joint_rank"]) for row in relevant]
        if any(not value.is_integer() or value < 1 for value in ranks):
            raise ValueError("Invalid persisted joint candidate rank")
        return 1.0 / min(ranks)
    ranked = []
    for row in candidates:
        value = next(
            (
                row[key]
                for key in ("Q_match", "P_rank", "S_select", "S_pair_final", "S_final", "S_base")
                if row.get(key) is not None
            ),
            None,
        )
        if value is None or not math.isfinite(float(value)):
            return None
        ranked.append((-float(value), str(row["target"])))
    for key, source_key in (
        ("Q_nil", "ontology_nil_probability"),
        ("Q_pool_miss", "pool_miss_probability"),
    ):
        value = candidates[0].get(key, record.get(source_key))
        if value is not None:
            if not math.isfinite(float(value)):
                raise ValueError("Nonfinite NIL ranking probability")
            ranked.append((-float(value), "\uffff" + key))
    # Canonical real target IRIs precede absence items at an exact probability tie.
    return 1.0 / min(
        index for index, (_, target) in enumerate(sorted(ranked), 1) if target in targets
    )


def evaluate_source_labels(cell: Any) -> dict[str, Any] | None:
    """Evaluate natural NIL without exposing source annotations to scoring or fitting."""
    diagnostics = getattr(cell, "diagnostics", None) or {}
    binding = diagnostics.get("evaluation_source_labels")
    if binding is None:
        return None
    if (
        diagnostics.get("role") != cell.split_role
        or diagnostics.get("reference_role") != cell.reference_role
        or cell.split_role not in {"development", "reporting"}
    ):
        raise ValueError("NIL evaluation source-label role differs from the scored cell")
    allowed = {
        "train",
        "valid",
        "validation",
        "dev",
        "development",
        "internal_check",
        "research_development",
    }
    if cell.split_role == "reporting":
        allowed = {"test", "reporting", "final"}
    if cell.reference_role not in allowed or cell.reference_role == "train":
        raise ValueError(
            "Training/final source labels cannot be reused as development evaluation labels"
        )
    trace_path = Path(cell.output_dir) / "source_decisions.json"
    trace = json.loads(trace_path.read_text())
    if (
        trace.get("schema_version") != 2
        or trace.get("source_universe_status") != "declared"
        or trace.get("stage") != "after_cardinality_and_relation_typing"
    ):
        raise ValueError("NIL evaluation requires a complete final source decision trace")
    universe = set(map(str, trace["source_universe"]))
    records = [dict(row) for row in trace["records"]]
    if {str(row["Src"]) for row in records} != universe or len(records) != len(universe):
        raise ValueError("NIL evaluation requires one final decision per eligible source")
    data = cell.resolved_config.get("data", {})
    root = Path(data.get("root") or ".")
    path = Path(binding["path"])
    if not path.is_absolute():
        raise ValueError("NIL evaluation source-label binding must have a resolved absolute path")
    training = cell.resolved_config.get("matching", {}).get("nil", {}).get("training_source_labels")
    if training and path.resolve() == (root / training).resolve():
        raise ValueError("NIL evaluation cannot consume training_source_labels")
    # All role and completed-output checks precede reading either evaluation input.
    if sha256_file(path) != binding["sha256"]:
        raise ValueError("NIL evaluation source-label file changed after binding")
    labels = read_table(path).rename(columns={"SrcEntity": "Src"})
    if not {"Src", "Status"} <= set(labels):
        raise ValueError("NIL evaluation labels require Src/Status columns")
    labels = labels[["Src", "Status"]].copy()
    labels["Src"] = labels.Src.astype(str)
    if labels.Src.duplicated().any() or not set(labels.Status) <= {*STATUSES, "mapped", "unknown"}:
        raise ValueError(
            "NIL source annotations must be unique canonical statuses or mapped/unknown"
        )
    outside = int((~labels.Src.isin(universe)).sum())
    labels = labels.loc[labels.Src.isin(universe)].set_index("Src")
    labels = labels.reindex(sorted(universe)).fillna({"Status": "unknown"}).reset_index()
    reference_name = data.get("refs", {}).get(cell.reference_role)
    reference_path = root / reference_name if reference_name else None
    reference = set()
    if reference_path is not None:
        frame = read_table(reference_path)
        source_column = "SrcEntity" if "SrcEntity" in frame else "Src"
        target_column = "TgtEntity" if "TgtEntity" in frame else "Tgt"
        if source_column not in frame or target_column not in frame:
            raise ValueError("NIL evaluation references require canonical source/target columns")
        reference = {
            (str(source), str(target))
            for source, target in frame[[source_column, target_column]].itertuples(
                index=False, name=None
            )
            if str(source) in universe
        }
    pool = {
        (str(row["Src"]), str(candidate["target"]))
        for row in records
        for candidate in row.get("candidates", [])
    }
    reference_by_source: dict[str, set[str]] = {}
    for source, target in reference:
        reference_by_source.setdefault(source, set()).add(target)
    pool_sources = {source for source, _ in pool}
    synthetic = bool(
        cell.resolved_config.get("matching", {})
        .get("nil", {})
        .get("pool_miss_development_reference")
    )
    if synthetic and cell.split_role != "development":
        raise ValueError("Synthetic pool-miss interventions are development-only")
    unresolved_pool_status = []
    complete_reference = getattr(cell, "reference_completeness", "unknown") == "complete"
    for index, row in labels.iterrows():
        positives = {
            (str(row.Src), target) for target in reference_by_source.get(str(row.Src), set())
        }
        if row.Status == "ontology_nil" and positives:
            raise ValueError("Natural NIL source annotation conflicts with a positive mapping")
        if row.Status in {"mapped", "in_pool", "pool_miss"}:
            if not positives:
                raise ValueError(
                    "Mapped source status requires an independently verified positive reference"
                )
            observed_in_pool = bool(positives & pool)
            empty_pool = str(row.Src) not in pool_sources
            if row.Status == "mapped" or synthetic:
                actual = (
                    "in_pool"
                    if observed_in_pool
                    else "pool_miss" if complete_reference or empty_pool else "mapped"
                )
                labels.at[index, "Status"] = actual
                if actual == "mapped":
                    unresolved_pool_status.append(str(row.Src))
            elif (row.Status == "pool_miss" and observed_in_pool) or (
                row.Status == "in_pool"
                and not observed_in_pool
                and (complete_reference or empty_pool)
            ):
                raise ValueError("NIL source status conflicts with the frozen candidate pool")
    emitted = {
        (str(row["Src"]), str(target))
        for row in records
        for target in row.get("emitted_targets", [])
    }
    for row in records:
        row.setdefault("absence_semantics", "unknown")
    metrics = nil_metrics(records, labels, reference, emitted)
    known = set(labels.loc[labels.Status != "unknown", "Src"])
    gold_nil = set(labels.loc[labels.Status == "ontology_nil", "Src"])
    unassessed = {pair for pair in emitted if pair[0] in known - gold_nil and pair not in reference}
    if getattr(cell, "reference_completeness", "unknown") != "complete" and unassessed:
        metrics["nil_aware"] = {
            "status": "unavailable",
            "reason": "incomplete pair reference cannot label emitted alternatives false",
            "unassessed_emitted_pairs": len(unassessed),
        }
    non_nil = known - gold_nil
    reciprocal_ranks = {
        str(row["Src"]): _reciprocal_rank(
            row, {target for source, target in reference if source == str(row["Src"])}
        )
        for row in records
        if str(row["Src"]) in non_nil
    }
    metrics["non_nil_MRR"] = (
        sum(value for value in reciprocal_ranks.values() if value is not None) / len(non_nil)
        if non_nil and all(value is not None for value in reciprocal_ranks.values())
        else None
    )
    metrics["non_nil_sources"] = len(non_nil)
    metrics["applicability"] = {
        "natural_nil": bool(gold_nil),
        "nil_aware": bool(known) and metrics["nil_aware"].get("status") != "unavailable",
        "non_nil_MRR": metrics["non_nil_MRR"] is not None,
        "pool_miss_as_nil": bool((labels.Status == "pool_miss").any()),
    }
    report = {
        "schema_version": 1,
        "evaluation_only": True,
        "label_semantics": "natural",
        "metric_scope": "explicit_source_statuses_and_supplied_positive_reference",
        "role": cell.split_role,
        "reference_role": cell.reference_role,
        "reference_completeness": getattr(cell, "reference_completeness", "unknown"),
        "source_universe": sorted(universe),
        "outside_universe_source_labels": outside,
        "synthetic_pool_miss_diagnostic": synthetic,
        "pool_status_unresolved_sources": sorted(unresolved_pool_status),
        "non_nil_reciprocal_ranks": reciprocal_ranks,
        "ranking_stage": "joint_candidate_ranking_before_global_acceptance",
        "metrics": metrics,
        "source_labels": {"path": str(path), "sha256": binding["sha256"]},
        "trace": {"path": str(trace_path), "sha256": sha256_file(trace_path)},
        "reference": (
            {"path": str(reference_path), "sha256": sha256_file(reference_path)}
            if reference_path
            else None
        ),
    }
    destination = Path(cell.output_dir) / "diagnostics/nil_metrics.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.partial")
    with temporary.open("w") as stream:
        json.dump(report, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(destination)
    return {
        "artifact": {"path": str(destination), "sha256": sha256_file(destination)},
        "source_labels": report["source_labels"],
        "reference": report["reference"],
        "evaluation_only": True,
        "metrics": metrics,
    }
