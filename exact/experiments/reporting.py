"""Paper-facing inventory, metric summaries, and paired source inference."""

from __future__ import annotations

import hashlib
import math
from ast import literal_eval
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Optional, Sequence

from exact.core.actions.alignment import resolve_alignment_inputs
from exact.core.entities.configs.config import ConfigModel
from exact.core.entities.kinds import MATCHABLE_ENTITY_KINDS
from exact.core.entities.mappings import EntityMapping, ReferenceMapping
from exact.io.sources import infer_format
from exact.io.sources import resolve as resolve_source
from exact.io.sources.datalog import read_facts
from exact.utils.data import read_table
from exact.utils.provenance import sha256_file

from .statistics import paired_bootstrap

_RELATION_NAMES = {
    "=": "equivalence",
    "equivalence": "equivalence",
    "equivalent": "equivalence",
    "<": "source_subsumed_by_target",
    "subsumed_by": "source_subsumed_by_target",
    "source_subsumed_by_target": "source_subsumed_by_target",
    ">": "source_subsumes_target",
    "subsumes": "source_subsumes_target",
    "source_subsumes_target": "source_subsumes_target",
}
_NIL_VALUES = {"nil", "no_match", "nomatch", "unmatched"}
_MANDATORY_REPORTING_ENDPOINTS = (
    ("candidate_recall", "candidate_recall", "proportion"),
    ("coverage", "coverage", "proportion"),
    ("abstention", "abstention_rate", "proportion"),
    ("ranking", "MRR", "proportion"),
    ("ranking", "Hits@1", "proportion"),
    ("wall_time", "wall_seconds", "seconds"),
    ("peak_memory", "peak_memory_kb", "KiB"),
    ("llm_usage", "llm.calls", "calls"),
    ("llm_usage", "llm.input_tokens", "tokens"),
    ("llm_usage", "llm.output_tokens", "tokens"),
    ("llm_usage", "llm.total_tokens", "tokens"),
)
_LLM_COUNTER_ALIASES = {
    "llm.calls": ("calls",),
    "llm.input_tokens": ("input_tokens", "prompt_tokens"),
    "llm.output_tokens": ("output_tokens", "completion_tokens"),
    "llm.total_tokens": ("total_tokens",),
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "value"):
        return str(value.value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    return value


def _reference_inventory(path: Optional[Path]) -> dict[str, Any]:
    if path is None:
        return {
            "total": None,
            "by_kind": {},
            "by_relation": {},
            "by_kind_relation": {},
            "nil_count": None,
        }
    mappings = ReferenceMapping.read_table_mappings(Path(path))
    by_kind: dict[str, int] = defaultdict(int)
    by_relation: dict[str, int] = defaultdict(int)
    by_kind_relation: dict[str, int] = defaultdict(int)
    nil_count = 0
    for mapping in mappings:
        kind = str(mapping.src_kind.value)
        relation = _RELATION_NAMES.get(str(mapping.relation).strip().lower(), str(mapping.relation))
        by_kind[kind] += 1
        by_relation[relation] += 1
        by_kind_relation[f"{kind}|{relation}"] += 1
        if str(mapping.tail).strip().lower() in _NIL_VALUES:
            nil_count += 1
    return {
        "total": len(mappings),
        "by_kind": dict(sorted(by_kind.items())),
        "by_relation": dict(sorted(by_relation.items())),
        "by_kind_relation": dict(sorted(by_kind_relation.items())),
        "nil_count": nil_count,
    }


def _candidate_inventory(path: Optional[Path], expected_sources: int) -> dict[str, Any]:
    if path is None:
        return {
            "origin": "generated_at_runtime",
            "status": "pending",
            "candidate_pairs": None,
            "covered_sources": None,
            "coverage": None,
        }
    frame = read_table(Path(path))
    if frame.empty:
        return {
            "origin": "declared_file",
            "status": "declared",
            "candidate_pairs": 0,
            "covered_sources": 0,
            "coverage": 0.0 if expected_sources else None,
        }
    source_column = frame.columns[0]
    covered = int(frame[source_column].astype(str).nunique())
    pairs = len(frame)
    if len(frame.columns) >= 3:
        parsed_pairs = 0
        parseable = True
        for value in frame.iloc[:, 2]:
            try:
                candidates = literal_eval(str(value))
            except (SyntaxError, ValueError):
                parseable = False
                break
            if not isinstance(candidates, (list, tuple)):
                parseable = False
                break
            parsed_pairs += len(candidates)
        if parseable:
            pairs = parsed_pairs
    return {
        "origin": "declared_file",
        "status": "declared",
        "candidate_pairs": int(pairs),
        "covered_sources": covered,
        "coverage": (covered / expected_sources if expected_sources else None),
    }


def _datalog_fact_count(source: Any) -> int:
    descriptor = getattr(source, "descriptor", None)
    origin = getattr(source, "origin", None)
    files = tuple(getattr(descriptor, "datalog_files", ()) or ())
    if origin is None or not files:
        return 0
    return sum(len(read_facts(Path(origin) / str(relative))) for relative in files)


def inspect_dataset_task(
    config: ConfigModel,
    *,
    experiment_id: str,
    task_id: str,
    stage: str,
    split_role: str,
    reference_role: str,
    reference_completeness: str,
    capabilities: Sequence[str],
    split_availability: Sequence[str] = (),
) -> dict[str, Any]:
    """Materialize and inspect one declared task without constructing matcher models."""

    resolved = resolve_alignment_inputs(configs=config)
    training_reference = getattr(resolved, "training_reference", None)
    source_format = infer_format(resolved.source, config.io.source_options)
    target_format = infer_format(resolved.target, config.io.target_options)
    source = resolve_source(
        resolved.source,
        format=config.io.input_format,
        options=config.io.source_options,
    )
    target = resolve_source(
        resolved.target,
        format=config.io.input_format,
        options=config.io.target_options,
    )
    entity_counts = {
        "source": {kind.value: len(source.entities(kind)) for kind in MATCHABLE_ENTITY_KINDS},
        "target": {kind.value: len(target.entities(kind)) for kind in MATCHABLE_ENTITY_KINDS},
    }
    selected_kinds = [str(kind) for kind in config.matching.entity_kinds]
    expected_sources = sum(entity_counts["source"].get(kind, 0) for kind in selected_kinds)
    hierarchy_families = {
        str(name): [str(item) for item in values]
        for name, values in config.dataset.hierarchical_relation_families.items()
    }
    return {
        "experiment_id": experiment_id,
        "task_id": task_id,
        "stage": stage,
        "split_role": split_role,
        "reference_role": reference_role,
        "reference_completeness": reference_completeness,
        "capabilities": sorted(set(str(item) for item in capabilities)),
        "split_availability": sorted(set(str(item) for item in split_availability)),
        "track": config.data.track,
        "task": config.data.task,
        "representation": {"source": source_format, "target": target_format},
        "inputs": {
            "source": {"path": str(resolved.source), "sha256": sha256_file(resolved.source)},
            "target": {"path": str(resolved.target), "sha256": sha256_file(resolved.target)},
            "reference": (
                {
                    "path": str(resolved.full_reference),
                    "sha256": sha256_file(resolved.full_reference),
                }
                if resolved.full_reference is not None
                else None
            ),
            "training_reference": (
                {
                    "path": str(training_reference),
                    "sha256": sha256_file(training_reference),
                }
                if training_reference is not None
                else None
            ),
            "candidates": (
                {
                    "path": str(resolved.candidates),
                    "sha256": sha256_file(resolved.candidates),
                }
                if resolved.candidates is not None
                else None
            ),
        },
        "entity_kind_support": selected_kinds,
        "entity_counts": entity_counts,
        "reference_counts": _reference_inventory(resolved.full_reference),
        "training_reference_counts": _reference_inventory(training_reference),
        "candidate_pool": _candidate_inventory(resolved.candidates, expected_sources),
        "hierarchy_predicates": hierarchy_families,
        "datalog_fact_count": {
            "source": _datalog_fact_count(source),
            "target": _datalog_fact_count(target),
        },
    }


def enrich_inventory_from_manifests(
    rows: Sequence[Mapping[str, Any]],
    manifests: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Overlay every actual ranked pool while preserving preflight declarations."""

    pools: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for manifest in manifests:
        if manifest.get("status") != "complete":
            continue
        pool = manifest.get("candidate_pool")
        if not isinstance(pool, Mapping):
            continue
        fingerprint = pool.get("fingerprint")
        if not isinstance(fingerprint, str) or not fingerprint:
            continue
        key = (str(manifest.get("experiment_id")), str(manifest.get("task_id")))
        entry = pools[key].setdefault(
            fingerprint,
            {
                "fingerprint": fingerprint,
                "origin": pool.get("origin"),
                "summary": pool.get("gold_free_summary"),
                "by_kind": pool.get("per_kind"),
                "entity_counts": ((pool.get("ontology_inventory") or {}).get("per_kind")),
                "arms": set(),
                "seeds": set(),
            },
        )
        entry["arms"].add(str(manifest.get("arm_id")))
        entry["seeds"].add(int(manifest.get("seed") or 0))

    enriched: list[dict[str, Any]] = []
    for original in rows:
        row = dict(_jsonable(original))
        key = (str(row.get("experiment_id")), str(row.get("task_id")))
        runtime_pools = []
        for fingerprint, raw_entry in sorted(pools.get(key, {}).items()):
            entry = dict(raw_entry)
            entry["arms"] = sorted(raw_entry["arms"])
            entry["seeds"] = sorted(raw_entry["seeds"])
            runtime_pools.append(entry)
        if runtime_pools:
            declared = dict(row.get("candidate_pool") or {})
            declared["status"] = "materialized" if len(runtime_pools) == 1 else "arm_specific"
            row["candidate_pool"] = declared
            row["candidate_pools"] = runtime_pools
        enriched.append(row)
    return enriched


def _metric_family_endpoint(metric_key: str) -> tuple[str, str]:
    """Classify an explicit metric key without treating global recall as retrieval recall."""

    key = str(metric_key).strip()
    lowered = key.lower().replace("-", "_")
    leaf = lowered.replace("/", ".").rsplit(".", 1)[-1]
    if "candidate_recall" in lowered:
        return "candidate_recall", "candidate_recall"
    if leaf == "coverage":
        return "coverage", "coverage"
    if "abstention" in leaf:
        return "abstention", "abstention_rate"
    if leaf == "mrr":
        return "ranking", "MRR"
    if leaf == "hits@1" or leaf == "hits_1":
        return "ranking", "Hits@1"
    if leaf.startswith("hits@") or leaf.startswith("hits_"):
        return "ranking", key
    if leaf in {"p", "precision", "r", "recall", "f1"}:
        return "quality", key
    if leaf in {"ece", "brier", "brier_score"}:
        return "calibration", key
    return "other", key


def _finite_metric_value(value: Any, *, label: str, non_negative: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric, not boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    if non_negative and result < 0.0:
        raise ValueError(f"{label} must be non-negative")
    return result


def _proportion_metric_value(value: Any, *, label: str) -> float:
    result = _finite_metric_value(value, label=label)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be between zero and one")
    return result


def _metric_applicability(
    record: Mapping[str, Any],
    *,
    family: str,
    endpoint: str,
) -> tuple[str, str]:
    declared = record.get("metric_applicability")
    if declared is not None and not isinstance(declared, Mapping):
        raise ValueError("metric_applicability must be a mapping when provided")
    value = None
    if isinstance(declared, Mapping):
        if endpoint in declared:
            value = declared[endpoint]
        elif family in declared:
            value = declared[family]
    if value is False or value == "not_applicable":
        return "not_applicable", "declared_not_applicable"
    if value not in (None, True, "applicable", "required"):
        raise ValueError(
            f"metric_applicability[{endpoint!r}] must be boolean or an applicability label"
        )
    if family == "llm_usage" and value is None and record.get("llm_required") is False:
        return "not_applicable", "llm_not_used"
    if record.get("status") != "complete":
        return "unavailable", "cell_not_complete"
    return "unavailable", "not_recorded"


def _llm_usage_root(value: Any) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("llm_usage must be a mapping when provided")
    summary = value.get("summary")
    if isinstance(summary, Mapping):
        has_top_level_counter = any(
            alias in value for aliases in _LLM_COUNTER_ALIASES.values() for alias in aliases
        )
        if not has_top_level_counter:
            return summary
    return value


def _llm_counter(
    usage: Mapping[str, Any] | None,
    endpoint: str,
) -> tuple[float | None, str | None, Any]:
    if usage is None:
        return None, None, None
    aliases = _LLM_COUNTER_ALIASES[endpoint]
    availability = usage.get("counter_availability")
    availability = availability if isinstance(availability, Mapping) else {}
    reported: list[tuple[str, float]] = []
    for alias in aliases:
        if usage.get(alias) is not None:
            value = _finite_metric_value(
                usage[alias],
                label=f"llm_usage.{alias}",
                non_negative=True,
            )
            if not value.is_integer():
                raise ValueError(f"llm_usage.{alias} must be an integer counter")
            reported.append((alias, value))
    if reported and len({value for _alias, value in reported}) != 1:
        raise ValueError(f"LLM counter aliases disagree for {endpoint}: {reported}")
    if reported:
        alias, value = reported[0]
        for counter_alias in aliases:
            detail = availability.get(counter_alias)
            if not isinstance(detail, Mapping):
                continue
            if detail.get("status") == "unavailable":
                raise ValueError(f"llm_usage.{counter_alias} is reported but marked unavailable")
            if detail.get("value") is not None and float(detail["value"]) != value:
                raise ValueError(f"llm_usage.{counter_alias} disagrees with counter availability")
        return value, alias, None
    for alias in aliases:
        if alias in availability:
            detail = availability[alias]
            if isinstance(detail, Mapping) and detail.get("status") == "available":
                raise ValueError(
                    f"llm_usage.{alias} is marked available but has no reported counter"
                )
            return None, None, _jsonable(detail)
    return None, None, None


def cell_metric_rows(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Represent raw, runtime, and explicitly unavailable mandatory cell metrics."""

    experiment = str(record.get("experiment_id"))
    arm = str(record.get("arm_id"))
    task = str(record.get("task_id"))
    seed = int(record.get("seed") or 0)
    identity = {
        "experiment_id": experiment,
        "arm_id": arm,
        "task_id": task,
        "seed": seed,
        "entity_kind": "all",
        "relation": "all",
    }
    raw_metrics = record.get("metrics") or {}
    if not isinstance(raw_metrics, Mapping):
        raise ValueError(f"metrics must be a mapping for {experiment}/{arm}/{task}/seed-{seed}")
    rows: list[dict[str, Any]] = []
    represented: set[tuple[str, str]] = set()

    def add_available(
        metric: str,
        raw_value: Any,
        *,
        family: str,
        endpoint: str,
        unit: str,
        source: str,
        source_metric: str | None = None,
    ) -> None:
        value = _finite_metric_value(
            raw_value,
            label=metric,
            non_negative=unit in {"seconds", "KiB", "calls", "tokens"},
        )
        if unit == "proportion" or family in {
            "candidate_recall",
            "coverage",
            "abstention",
            "ranking",
        }:
            value = _proportion_metric_value(raw_value, label=metric)
        row = {
            **identity,
            "metric": metric,
            "metric_family": family,
            "metric_endpoint": endpoint,
            "value": value,
            "unit": unit,
            "availability": "available",
            "availability_reason": None,
            "source": source,
        }
        if source_metric is not None:
            row["source_metric"] = source_metric
        rows.append(row)
        represented.add((family, endpoint))

    for raw_key, raw_value in raw_metrics.items():
        metric = str(raw_key)
        family, endpoint = _metric_family_endpoint(metric)
        unit = (
            "proportion"
            if family
            in {
                "candidate_recall",
                "coverage",
                "abstention",
                "ranking",
                "quality",
                "calibration",
            }
            else "value"
        )
        add_available(
            metric,
            raw_value,
            family=family,
            endpoint=endpoint,
            unit=unit,
            source="authoritative_evaluator",
        )

    supplemental = (
        ("candidate_recall", "candidate_recall", "candidate_recall", "proportion"),
        ("coverage", "coverage", "coverage", "proportion"),
        ("abstention_rate", "abstention", "abstention_rate", "proportion"),
        ("abstention", "abstention", "abstention_rate", "proportion"),
        ("wall_seconds", "wall_time", "wall_seconds", "seconds"),
        ("peak_memory_kb", "peak_memory", "peak_memory_kb", "KiB"),
    )
    for field, family, endpoint, unit in supplemental:
        if (family, endpoint) in represented or record.get(field) is None:
            continue
        add_available(
            endpoint,
            record[field],
            family=family,
            endpoint=endpoint,
            unit=unit,
            source="run_manifest",
            source_metric=field,
        )

    usage = _llm_usage_root(record.get("llm_usage"))
    llm_availability_details: dict[str, Any] = {}
    for endpoint in _LLM_COUNTER_ALIASES:
        value, source_metric, detail = _llm_counter(usage, endpoint)
        if detail is not None:
            llm_availability_details[endpoint] = detail
        if value is not None:
            add_available(
                endpoint,
                value,
                family="llm_usage",
                endpoint=endpoint,
                unit="calls" if endpoint == "llm.calls" else "tokens",
                source="run_manifest",
                source_metric=f"llm_usage.{source_metric}",
            )

    for family, endpoint, unit in _MANDATORY_REPORTING_ENDPOINTS:
        if (family, endpoint) in represented:
            continue
        availability, reason = _metric_applicability(
            record,
            family=family,
            endpoint=endpoint,
        )
        row = {
            **identity,
            "metric": endpoint,
            "metric_family": family,
            "metric_endpoint": endpoint,
            "value": None,
            "unit": unit,
            "availability": availability,
            "availability_reason": reason,
            "source": "availability_declaration",
        }
        if endpoint in llm_availability_details:
            row["availability_detail"] = llm_availability_details[endpoint]
            row["availability_reason"] = "runtime_counter_unavailable"
        rows.append(row)
    rows.sort(
        key=lambda row: (
            str(row["metric_family"]),
            str(row["metric_endpoint"]),
            str(row["metric"]),
        )
    )
    return rows


def metric_reports(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return availability-aware cell metrics and completeness-aware task macros."""

    long_rows: list[dict[str, Any]] = []
    expected_cells: dict[tuple[str, str], set[tuple[str, int]]] = defaultdict(set)
    observed: dict[
        tuple[str, str, str, str, str, str, str, str],
        dict[tuple[str, int], float],
    ] = defaultdict(dict)
    for record in records:
        experiment = str(record.get("experiment_id"))
        arm = str(record.get("arm_id"))
        task = str(record.get("task_id"))
        seed = int(record.get("seed") or 0)
        expected_cells[(experiment, arm)].add((task, seed))
        for row in cell_metric_rows(record):
            long_rows.append(row)
            if row["availability"] != "available":
                continue
            metric_key = str(row["metric"])
            kind = str(row["entity_kind"])
            relation = str(row["relation"])
            observed[
                (
                    experiment,
                    arm,
                    metric_key,
                    str(row["metric_family"]),
                    str(row["metric_endpoint"]),
                    str(row["unit"]),
                    kind,
                    relation,
                )
            ][(task, seed)] = float(row["value"])

    macros: list[dict[str, Any]] = []
    for key, values in sorted(observed.items()):
        experiment, arm, metric_key, family, endpoint, unit, kind, relation = key
        expected = expected_cells[(experiment, arm)]
        missing = sorted(expected.difference(values))
        by_task: dict[str, list[float]] = defaultdict(list)
        for (task, _seed), value in values.items():
            by_task[task].append(value)
        task_means = {task: fmean(entries) for task, entries in by_task.items()}
        complete = not missing
        macros.append(
            {
                "experiment_id": experiment,
                "arm_id": arm,
                "entity_kind": kind,
                "relation": relation,
                "metric": metric_key,
                "metric_family": family,
                "metric_endpoint": endpoint,
                "unit": unit,
                "macro_over_tasks": (
                    fmean(task_means.values()) if complete and task_means else None
                ),
                "observed_task_means": dict(sorted(task_means.items())),
                "complete": complete,
                "missing_cells": [f"{task}/seed-{seed}" for task, seed in missing],
            }
        )
    long_rows.sort(
        key=lambda row: (
            row["experiment_id"],
            row["arm_id"],
            row["task_id"],
            row["seed"],
            row["metric_family"],
            row["metric"],
        )
    )
    return long_rows, macros


def _evaluation_reference_paths(output_dir: Path) -> tuple[Path, Optional[Path], str]:
    path = output_dir / "evaluation" / "evaluation_results.json"
    if not path.is_file():
        raise FileNotFoundError(f"global evaluation report is missing: {path}")
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    refs = (payload.get("meta") or {}).get("refs") or {}
    full = refs.get("full_reference")
    if not isinstance(full, Mapping) or not full.get("path") or not full.get("sha256"):
        raise ValueError(f"global evaluation report lacks full-reference provenance: {path}")
    full_path = Path(str(full["path"])).expanduser().resolve()
    if not full_path.is_file() or sha256_file(full_path) != str(full["sha256"]):
        raise ValueError(f"global evaluation reference changed after evaluation: {full_path}")
    train = refs.get("train_reference")
    train_path: Optional[Path] = None
    train_sha256: Optional[str] = None
    if isinstance(train, Mapping) and train.get("path"):
        train_path = Path(str(train["path"])).expanduser().resolve()
        if train.get("sha256"):
            train_sha256 = str(train["sha256"])
    if train_path is not None:
        if not train_path.is_file() or not train_sha256 or sha256_file(train_path) != train_sha256:
            raise ValueError(
                f"evaluation training reference changed after evaluation: {train_path}"
            )
    return full_path, train_path, str(full["sha256"])


def _source_f1(output_dir: Path) -> tuple[dict[str, float], str]:
    alignment = output_dir / "alignment" / "maps_global.tsv"
    if not alignment.is_file():
        raise FileNotFoundError(f"global per-source alignment is missing: {alignment}")
    full_reference, train_reference, reference_hash = _evaluation_reference_paths(output_dir)
    predictions = EntityMapping.read_table_mappings(alignment)
    references = ReferenceMapping.read_table_mappings(full_reference)
    null_pairs = {
        mapping.to_tuple()
        for mapping in (
            ReferenceMapping.read_table_mappings(train_reference)
            if train_reference is not None
            else []
        )
    }
    predicted: dict[str, set[str]] = defaultdict(set)
    expected: dict[str, set[str]] = defaultdict(set)
    for mapping in predictions:
        if mapping.to_tuple() not in null_pairs:
            predicted[str(mapping.head)].add(str(mapping.tail))
    for mapping in references:
        if mapping.to_tuple() not in null_pairs:
            expected[str(mapping.head)].add(str(mapping.tail))
    scores: dict[str, float] = {}
    for source in sorted(set(predicted) | set(expected)):
        left = predicted.get(source, set())
        right = expected.get(source, set())
        true_positive = len(left & right)
        precision = true_positive / len(left) if left else 0.0
        recall = true_positive / len(right) if right else 0.0
        scores[source] = (
            2.0 * precision * recall / (precision + recall) if precision + recall > 0.0 else 0.0
        )
    return scores, reference_hash


def paired_bootstrap_reports(
    records: Sequence[Mapping[str, Any]],
    comparisons: Mapping[str, Sequence[tuple[str, str, str]]],
    *,
    resamples: int = 10_000,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Run declared arm contrasts over per-source global F1 decisions."""

    index: dict[tuple[str, str, str, int], Mapping[str, Any]] = {}
    for record in records:
        if record.get("status") != "complete":
            continue
        index[
            (
                str(record.get("experiment_id")),
                str(record.get("arm_id")),
                str(record.get("task_id")),
                int(record.get("seed") or 0),
            )
        ] = record

    reports: list[dict[str, Any]] = []
    for experiment_id, declared in sorted(comparisons.items()):
        tasks = sorted(
            {task for experiment, _arm, task, _seed in index if experiment == experiment_id}
        )
        for decision_id, baseline_arm, candidate_arm in declared:
            for task_id in tasks:
                baseline_seeds = {
                    seed_value
                    for experiment, arm, task, seed_value in index
                    if experiment == experiment_id and arm == baseline_arm and task == task_id
                }
                candidate_seeds = {
                    seed_value
                    for experiment, arm, task, seed_value in index
                    if experiment == experiment_id and arm == candidate_arm and task == task_id
                }
                if not baseline_seeds or baseline_seeds != candidate_seeds:
                    reports.append(
                        {
                            "experiment_id": experiment_id,
                            "decision_id": decision_id,
                            "baseline_arm": baseline_arm,
                            "candidate_arm": candidate_arm,
                            "task_id": task_id,
                            "metric": "per_source_global_f1",
                            "status": "unavailable",
                            "reason": "arms do not have identical non-empty paired seeds",
                            "baseline_seeds": sorted(baseline_seeds),
                            "candidate_seeds": sorted(candidate_seeds),
                        }
                    )
                    continue
                common_seeds = sorted(baseline_seeds)
                baseline_by_source: dict[str, list[float]] = defaultdict(list)
                candidate_by_source: dict[str, list[float]] = defaultdict(list)
                reference_hashes: set[str] = set()
                try:
                    for seed_value in common_seeds:
                        baseline_record = index[(experiment_id, baseline_arm, task_id, seed_value)]
                        candidate_record = index[
                            (experiment_id, candidate_arm, task_id, seed_value)
                        ]
                        baseline_scores, baseline_ref = _source_f1(
                            Path(str(baseline_record["output_dir"]))
                        )
                        candidate_scores, candidate_ref = _source_f1(
                            Path(str(candidate_record["output_dir"]))
                        )
                        reference_hashes.update((baseline_ref, candidate_ref))
                        if len(reference_hashes) != 1:
                            raise ValueError(
                                "paired arms or seeds used different reporting-reference hashes"
                            )
                        units = sorted(set(baseline_scores) | set(candidate_scores))
                        for source in units:
                            baseline_by_source[source].append(baseline_scores.get(source, 0.0))
                            candidate_by_source[source].append(candidate_scores.get(source, 0.0))
                    baseline_mean = {
                        source: fmean(values) for source, values in baseline_by_source.items()
                    }
                    candidate_mean = {
                        source: fmean(candidate_by_source[source]) for source in baseline_mean
                    }
                    result = paired_bootstrap(
                        baseline_mean,
                        candidate_mean,
                        resamples=resamples,
                        seed=seed,
                    )
                except (KeyError, OSError, TypeError, ValueError) as exc:
                    reports.append(
                        {
                            "experiment_id": experiment_id,
                            "decision_id": decision_id,
                            "baseline_arm": baseline_arm,
                            "candidate_arm": candidate_arm,
                            "task_id": task_id,
                            "metric": "per_source_global_f1",
                            "status": "unavailable",
                            "reason": str(exc),
                            "paired_seeds": common_seeds,
                        }
                    )
                    continue
                reports.append(
                    {
                        "experiment_id": experiment_id,
                        "decision_id": decision_id,
                        "baseline_arm": baseline_arm,
                        "candidate_arm": candidate_arm,
                        "task_id": task_id,
                        "metric": "per_source_global_f1",
                        "status": "complete",
                        "paired_seeds": common_seeds,
                        "reference_sha256": next(iter(reference_hashes)),
                        **result.as_dict(),
                    }
                )
    return reports


def e17_interaction_bootstrap_reports(
    records: Sequence[Mapping[str, Any]],
    interactions: Mapping[str, Sequence[tuple[str, str, str, str, str]]],
    *,
    resamples: int = 10_000,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Bootstrap predeclared 2x2 residuals over paired source decisions."""

    index: dict[tuple[str, str, str, int], Mapping[str, Any]] = {
        (
            str(record.get("experiment_id")),
            str(record.get("arm_id")),
            str(record.get("task_id")),
            int(record.get("seed") or 0),
        ): record
        for record in records
        if record.get("status") == "complete"
    }
    reports: list[dict[str, Any]] = []
    for experiment_id, declared in sorted(interactions.items()):
        tasks = sorted(
            {task for experiment, _arm, task, _seed in index if experiment == experiment_id}
        )
        for interaction_id, arm_00, arm_10, arm_01, arm_11 in declared:
            arms = (arm_00, arm_10, arm_01, arm_11)
            for task_id in tasks:
                seed_sets = [
                    {
                        seed_value
                        for experiment, arm, task, seed_value in index
                        if experiment == experiment_id and arm == arm_id and task == task_id
                    }
                    for arm_id in arms
                ]
                if not all(seed_sets) or any(values != seed_sets[0] for values in seed_sets[1:]):
                    reports.append(
                        {
                            "experiment_id": experiment_id,
                            "decision_id": f"interaction_{interaction_id}",
                            "task_id": task_id,
                            "metric": "per_source_global_f1_interaction_residual",
                            "status": "unavailable",
                            "reason": "2x2 cells do not have identical non-empty paired seeds",
                            "cell_seed_sets": {
                                arm: sorted(values) for arm, values in zip(arms, seed_sets)
                            },
                        }
                    )
                    continue
                residuals: dict[str, list[float]] = defaultdict(list)
                reference_hashes: set[str] = set()
                try:
                    for seed_value in sorted(seed_sets[0]):
                        scores: dict[str, dict[str, float]] = {}
                        for arm_id in arms:
                            arm_scores, reference_hash = _source_f1(
                                Path(
                                    str(
                                        index[
                                            (
                                                experiment_id,
                                                arm_id,
                                                task_id,
                                                seed_value,
                                            )
                                        ]["output_dir"]
                                    )
                                )
                            )
                            scores[arm_id] = arm_scores
                            reference_hashes.add(reference_hash)
                        if len(reference_hashes) != 1:
                            raise ValueError("2x2 cells used different reporting-reference hashes")
                        units = sorted(set().union(*(set(values) for values in scores.values())))
                        for source in units:
                            residuals[source].append(
                                scores[arm_11].get(source, 0.0)
                                - scores[arm_10].get(source, 0.0)
                                - scores[arm_01].get(source, 0.0)
                                + scores[arm_00].get(source, 0.0)
                            )
                    residual_mean = {source: fmean(values) for source, values in residuals.items()}
                    result = paired_bootstrap(
                        {source: 0.0 for source in residual_mean},
                        residual_mean,
                        resamples=resamples,
                        seed=seed,
                    )
                except (KeyError, OSError, TypeError, ValueError) as exc:
                    reports.append(
                        {
                            "experiment_id": experiment_id,
                            "decision_id": f"interaction_{interaction_id}",
                            "task_id": task_id,
                            "metric": "per_source_global_f1_interaction_residual",
                            "status": "unavailable",
                            "reason": str(exc),
                            "paired_seeds": sorted(seed_sets[0]),
                        }
                    )
                    continue
                reports.append(
                    {
                        "experiment_id": experiment_id,
                        "decision_id": f"interaction_{interaction_id}",
                        "task_id": task_id,
                        "metric": "per_source_global_f1_interaction_residual",
                        "contrast": "score(11)-score(10)-score(01)+score(00)",
                        "cells": list(arms),
                        "status": "complete",
                        "paired_seeds": sorted(seed_sets[0]),
                        "reference_sha256": next(iter(reference_hashes)),
                        **result.as_dict(),
                    }
                )
    return reports


def stable_bootstrap_seed(suite_id: str, stage: str) -> int:
    """Return a stable 32-bit resampling seed independent of Python hashing."""

    digest = hashlib.sha256(f"{suite_id}\x1f{stage}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big", signed=False)


__all__ = [
    "cell_metric_rows",
    "e17_interaction_bootstrap_reports",
    "enrich_inventory_from_manifests",
    "inspect_dataset_task",
    "metric_reports",
    "paired_bootstrap_reports",
    "stable_bootstrap_seed",
]
