"""Authoritative paper metrics and source-paired global-F1 inference.

This module deliberately reads only evaluator JSON reports and verified global
alignment/reference artifacts.  It does not inspect the experiment harness's
derived CSV or run-statistics outputs.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping, Optional, Sequence

from exact.core.entities.kinds import EntityKind
from exact.utils.data import read_table
from exact.utils.provenance import sha256_file

_REPORT_NAME = "evaluation_results.json"
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


@dataclass(frozen=True)
class SourceConfusion:
    """TP/FP/FN counts for one independent source entity."""

    tp: int = 0
    fp: int = 0
    fn: int = 0

    def __post_init__(self) -> None:
        for name, value in (("tp", self.tp), ("fp", self.fp), ("fn", self.fn)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

    def __add__(self, other: "SourceConfusion") -> "SourceConfusion":
        return SourceConfusion(
            tp=self.tp + other.tp,
            fp=self.fp + other.fp,
            fn=self.fn + other.fn,
        )

    def as_dict(self) -> dict[str, int]:
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn}


@dataclass(frozen=True)
class PRFMetrics:
    """Unrounded precision, recall, and F1 with their sufficient counts."""

    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float

    @classmethod
    def from_counts(cls, counts: SourceConfusion) -> "PRFMetrics":
        precision = counts.tp / (counts.tp + counts.fp) if counts.tp + counts.fp else 0.0
        recall = counts.tp / (counts.tp + counts.fn) if counts.tp + counts.fn else 0.0
        denominator = 2 * counts.tp + counts.fp + counts.fn
        f1 = 2 * counts.tp / denominator if denominator else 0.0
        return cls(
            tp=counts.tp,
            fp=counts.fp,
            fn=counts.fn,
            precision=precision,
            recall=recall,
            f1=f1,
        )

    def as_dict(self) -> dict[str, float | int]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "P": self.precision,
            "R": self.recall,
            "F1": self.f1,
        }


@dataclass(frozen=True)
class SourceEvaluation:
    """Verified evaluation counts keyed by source entity."""

    reference_sha256: str
    alignment_sha256: str
    by_source: Mapping[str, SourceConfusion]
    null_reference_sha256: Optional[str] = None
    entity_kind: Optional[str] = None
    relation: Optional[str] = None
    source_universe_sha256: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.reference_sha256:
            raise ValueError("reference_sha256 must not be empty")
        if not self.alignment_sha256:
            raise ValueError("alignment_sha256 must not be empty")
        normalized: dict[str, SourceConfusion] = {}
        for source, counts in self.by_source.items():
            key = str(source)
            if not key:
                raise ValueError("source entity identifiers must not be empty")
            if not isinstance(counts, SourceConfusion):
                raise TypeError("by_source values must be SourceConfusion instances")
            normalized[key] = counts
        object.__setattr__(self, "by_source", dict(sorted(normalized.items())))

    @property
    def counts(self) -> SourceConfusion:
        total = SourceConfusion()
        for counts in self.by_source.values():
            total = total + counts
        return total

    @property
    def metrics(self) -> PRFMetrics:
        return PRFMetrics.from_counts(self.counts)

    def as_dict(self, *, include_sources: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "reference_sha256": self.reference_sha256,
            "source_universe_sha256": self.source_universe_sha256,
            "null_reference_sha256": self.null_reference_sha256,
            "alignment_sha256": self.alignment_sha256,
            "entity_kind": self.entity_kind,
            "relation": self.relation,
            "metrics": self.metrics.as_dict(),
        }
        if include_sources:
            result["by_source"] = {
                source: counts.as_dict() for source, counts in self.by_source.items()
            }
        return result


@dataclass(frozen=True)
class RecomputedEvaluation:
    """Relation-insensitive overall metrics plus requested typed slices."""

    overall: SourceEvaluation
    slices: Mapping[str, SourceEvaluation]

    def as_dict(self, *, include_sources: bool = True) -> dict[str, Any]:
        return {
            "overall": self.overall.as_dict(include_sources=include_sources),
            "slices": {
                name: evaluation.as_dict(include_sources=include_sources)
                for name, evaluation in sorted(self.slices.items())
            },
        }

    def metric_values(self) -> dict[str, float]:
        """Return flat P/R/F1 values without inferring dimensions from names."""

        result = {
            "P": self.overall.metrics.precision,
            "R": self.overall.metrics.recall,
            "F1": self.overall.metrics.f1,
        }
        for name, evaluation in sorted(self.slices.items()):
            result[f"{name}.P"] = evaluation.metrics.precision
            result[f"{name}.R"] = evaluation.metrics.recall
            result[f"{name}.F1"] = evaluation.metrics.f1
        return result


@dataclass(frozen=True)
class F1BootstrapResult:
    """Task-macro global-F1 contrast and its source-paired percentile interval."""

    contrast: str
    arm_scores: Mapping[str, float]
    coefficients: Mapping[str, float]
    delta: float
    ci_low: float
    ci_high: float
    p_value: float
    confidence: float
    resamples: int
    seed: int
    n_tasks: int
    n_cells: int
    n_units: int
    source_units_by_task: Mapping[str, int]
    paired_seeds: tuple[int, ...]
    seeds_by_task: Mapping[str, tuple[int, ...]]
    reference_sha256: Optional[str]
    reference_hashes_by_task: Mapping[str, Mapping[str, Optional[str]]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "contrast": self.contrast,
            "arm_scores": dict(sorted(self.arm_scores.items())),
            "coefficients": dict(sorted(self.coefficients.items())),
            "delta": self.delta,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "interval_method": "percentile_bootstrap",
            "p_value": self.p_value,
            "p_value_method": "two_sided_centered_bootstrap_plus_one",
            "p_value_adjustment": "none",
            "interval_adjustment": "none",
            "confidence": self.confidence,
            "resamples": self.resamples,
            "seed": self.seed,
            "n_tasks": self.n_tasks,
            "n_cells": self.n_cells,
            "n_units": self.n_units,
            "source_units_by_task": dict(sorted(self.source_units_by_task.items())),
            "paired_seeds": list(self.paired_seeds),
            "seeds_by_task": {
                task: list(seeds) for task, seeds in sorted(self.seeds_by_task.items())
            },
            "reference_sha256": self.reference_sha256,
            "reference_hashes_by_task": {
                task: dict(sorted(hashes.items()))
                for task, hashes in sorted(self.reference_hashes_by_task.items())
            },
        }


@dataclass(frozen=True)
class _TypedRow:
    source: str
    target: str
    relation: Optional[str]
    source_kind: Optional[str]
    target_kind: Optional[str]


def normalize_relation(value: str) -> str:
    """Normalize the relation spellings used by Exact's experiment reports."""

    token = str(value).strip().lower()
    if not token:
        raise ValueError("relation must not be empty")
    return _RELATION_NAMES.get(token, token)


def _read_json_object(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid evaluation report at {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"Evaluation report must contain a JSON object: {path}")
    return payload


def _primary_report_path(location: Path) -> Path:
    location = Path(location)
    if location.is_file():
        if location.name != _REPORT_NAME:
            raise ValueError(f"Expected {_REPORT_NAME}, got {location.name}")
        return location
    candidates = (
        location / "evaluation" / _REPORT_NAME,
        location / _REPORT_NAME,
        location / "model" / "alignment" / "default" / _REPORT_NAME,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No authoritative {_REPORT_NAME} under {location}")


def _report_scope(payload: Mapping[str, Any], path: Path) -> str:
    meta = payload.get("meta")
    refs = meta.get("refs") if isinstance(meta, Mapping) else None
    has_full = isinstance(refs, Mapping) and isinstance(refs.get("full_reference"), Mapping)
    has_local = isinstance(refs, Mapping) and isinstance(refs.get("reference_candidates"), Mapping)
    if has_full and has_local:
        raise ValueError(f"Evaluation report has both global and local references: {path}")
    if has_full:
        return "global"
    if has_local or path.parent.name == "local":
        return "local"
    raise ValueError(
        f"Cannot establish evaluation scope from meta.refs in authoritative report: {path}"
    )


def _direct_backend_metrics(
    payload: Mapping[str, Any], *, path: Path, scope: str
) -> dict[str, float]:
    backend_names = sorted(
        str(name)
        for name, value in payload.items()
        if name != "meta" and isinstance(value, Mapping)
    )
    if not backend_names:
        raise ValueError(f"Evaluation report has no backend metric objects: {path}")
    single_builtin = backend_names == ["builtin"]
    result: dict[str, float] = {}
    for backend in backend_names:
        metrics = payload[backend]
        assert isinstance(metrics, Mapping)
        for raw_name, raw_value in metrics.items():
            if raw_value is None or isinstance(raw_value, (Mapping, list, tuple)):
                continue
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                continue
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError(f"Non-finite metric {backend}.{raw_name} in {path}")
            metric_name = str(raw_name) if single_builtin else f"{backend}.{raw_name}"
            key = metric_name if scope == "global" else f"local.{metric_name}"
            if key in result:
                raise ValueError(f"Duplicate authoritative metric key {key!r} in {path}")
            result[key] = value
    if not result:
        raise ValueError(f"Evaluation report has no finite direct backend metrics: {path}")
    return result


def extract_evaluation_metrics(location: Path) -> dict[str, float]:
    """Extract metrics only from canonical evaluator JSON reports.

    Global keys retain the exact evaluator return namespace (for example,
    ``F1`` for builtin-only evaluation or ``builtin.F1`` with multiple
    backends).  Every local key is additionally prefixed with ``local.``.
    Nested backend details, ``meta``, CSV files, ``metrics.json``, and
    ``run_stats.json`` are never traversed.
    """

    primary = _primary_report_path(Path(location))
    reports = [primary]
    primary_payload = _read_json_object(primary)
    primary_scope = _report_scope(primary_payload, primary)
    if primary_scope == "global":
        local = primary.parent / "local" / _REPORT_NAME
        if local.is_file():
            reports.append(local)

    result: dict[str, float] = {}
    for report in reports:
        payload = primary_payload if report == primary else _read_json_object(report)
        scope = primary_scope if report == primary else _report_scope(payload, report)
        discovered = _direct_backend_metrics(payload, path=report, scope=scope)
        duplicates = sorted(set(result).intersection(discovered))
        if duplicates:
            raise ValueError(
                f"Duplicate authoritative metric keys across evaluator reports: {duplicates}"
            )
        result.update(discovered)
    return result


def _metadata_refs(payload: Mapping[str, Any], path: Path) -> Mapping[str, Any]:
    meta = payload.get("meta")
    refs = meta.get("refs") if isinstance(meta, Mapping) else None
    if not isinstance(refs, Mapping):
        raise ValueError(f"Evaluation report has no verifiable meta.refs object: {path}")
    return refs


def _provenance_path(
    provenance: Any,
    *,
    label: str,
    report_path: Path,
    override: Optional[Path] = None,
) -> tuple[Path, str]:
    if not isinstance(provenance, Mapping):
        raise ValueError(f"Evaluation report is missing {label} provenance")
    expected_hash = provenance.get("sha256")
    if not isinstance(expected_hash, str) or not expected_hash:
        raise ValueError(f"Evaluation report has no {label} SHA-256")
    if override is not None:
        path = Path(override).expanduser().resolve()
    else:
        raw_path = provenance.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"Evaluation report has no {label} path")
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = (report_path.parent / path).resolve()
        else:
            path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Verified {label} artifact does not exist: {path}")
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise ValueError(f"{label} SHA-256 mismatch for {path}: {actual_hash} != {expected_hash}")
    return path, expected_hash


def _default_alignment_path(location: Path, provenance: Any) -> Optional[Path]:
    if isinstance(provenance, Mapping) and isinstance(provenance.get("path"), str):
        recorded = Path(str(provenance["path"])).expanduser()
        if recorded.is_file():
            return recorded
    if location.is_dir():
        candidates = (
            location / "alignment" / "paper.maps_global.tsv",
            location / "alignment" / "maps_global.tsv",
            location / "alignment" / "src2tgt.maps_global.tsv",
            location / "model" / "alignment" / "src2tgt.maps_global.tsv",
            location / "model" / "alignment" / "maps_global.tsv",
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate
    return None


def _column_name(columns: Sequence[Any], *preferred: str) -> Optional[Any]:
    by_lower = {str(column).strip().lower(): column for column in columns}
    for name in preferred:
        found = by_lower.get(name.lower())
        if found is not None:
            return found
    return None


def _read_mapping_rows(path: Path, *, typed: bool, label: str) -> list[_TypedRow]:
    frame = read_table(path)
    columns = list(frame.columns)
    source_column = _column_name(columns, "SrcEntity", "Src", "source")
    target_column = _column_name(columns, "TgtEntity", "Tgt", "target")
    if source_column is None or target_column is None:
        if len(columns) < 2:
            raise ValueError(f"{label} mapping table must contain source and target columns")
        source_column, target_column = columns[:2]
    relation_column = _column_name(columns, "Relation", "relation")
    source_kind_column = _column_name(columns, "SrcKind", "Kind", "kind")
    target_kind_column = _column_name(columns, "TgtKind")
    if typed and relation_column is None:
        raise ValueError(f"Typed slices require an explicit Relation column in {label}: {path}")
    if typed and source_kind_column is None:
        raise ValueError(
            f"Typed slices require an explicit SrcKind or Kind column in {label}: {path}"
        )

    rows: list[_TypedRow] = []
    for row in frame.to_dict(orient="records"):
        source = str(row[source_column])
        target = str(row[target_column])
        source_kind = (
            str(row[source_kind_column]).strip().lower() if source_kind_column is not None else None
        )
        target_kind = (
            str(row[target_kind_column]).strip().lower()
            if target_kind_column is not None
            else source_kind
        )
        rows.append(
            _TypedRow(
                source=source,
                target=target,
                relation=(
                    normalize_relation(str(row[relation_column]))
                    if relation_column is not None
                    else None
                ),
                source_kind=source_kind,
                target_kind=target_kind,
            )
        )
    return rows


def _source_counts(
    predictions: Iterable[tuple[str, str]],
    references: Iterable[tuple[str, str]],
    null_pairs: Iterable[tuple[str, str]],
) -> dict[str, SourceConfusion]:
    null_set = set(null_pairs)
    prediction_set = set(predictions).difference(null_set)
    reference_set = set(references).difference(null_set)
    true_positives = prediction_set.intersection(reference_set)
    false_positives = prediction_set.difference(reference_set)
    false_negatives = reference_set.difference(prediction_set)
    sources = sorted({source for source, _target in prediction_set.union(reference_set)})
    return {
        source: SourceConfusion(
            tp=sum(pair[0] == source for pair in true_positives),
            fp=sum(pair[0] == source for pair in false_positives),
            fn=sum(pair[0] == source for pair in false_negatives),
        )
        for source in sources
    }


def _typed_pairs(
    rows: Iterable[_TypedRow], *, entity_kind: str, relation: str
) -> set[tuple[str, str]]:
    return {
        (row.source, row.target)
        for row in rows
        if row.source_kind == entity_kind
        and row.target_kind == entity_kind
        and row.relation == relation
    }


def _validate_builtin_parity(
    payload: Mapping[str, Any], metrics: PRFMetrics, report_path: Path
) -> None:
    builtin = payload.get("builtin")
    if not isinstance(builtin, Mapping):
        raise ValueError(f"Global evaluation report has no builtin metrics: {report_path}")
    expected = {"P": metrics.precision, "R": metrics.recall, "F1": metrics.f1}
    mismatches: dict[str, tuple[Any, float]] = {}
    for name, value in expected.items():
        reported = builtin.get(name)
        if isinstance(reported, bool) or not isinstance(reported, (int, float)):
            raise ValueError(f"Global builtin report is missing numeric {name}: {report_path}")
        if not math.isclose(float(reported), round(value, 3), rel_tol=0.0, abs_tol=1e-12):
            mismatches[name] = (reported, round(value, 3))
    if mismatches:
        raise ValueError(
            "Recomputed pair metrics do not match builtin evaluation; the artifact may not "
            f"represent the evaluated filtering pool ({mismatches})"
        )


def _frozen_population(report_path: Path) -> tuple[set[str], Optional[str]]:
    """Recover source membership from the verified-pool artifact, including empty groups."""
    import hashlib

    evaluation_dir = next(
        (parent for parent in report_path.parents if parent.name == "evaluation"), None
    )
    if evaluation_dir is None:
        return set(), None
    directory = evaluation_dir.parent / "dataset"
    for name in ("candidate_pool_sample_manifest.json", "candidate_pool_manifest.json"):
        path = directory / name
        if not path.is_file():
            continue
        sample = (_read_json_object(path).get("retrieval_config") or {}).get("source_sample")
        if not sample:
            continue
        groups = sorted(tuple(group) for group in sample["source_kind_groups"])
        identity = hashlib.sha256(
            "\n".join(f"{source}\t{kind}" for source, kind in groups).encode()
        ).hexdigest()
        sources = {source for source, _ in groups}
        if identity != sample["sha256"] or sources != set(sample["eligible_source_iris"]):
            raise ValueError("Frozen source population integrity check failed")
        return sources, identity
    return set(), None


def recompute_global_prf(
    location: Path,
    *,
    slices: Sequence[tuple[EntityKind | str, str]] = (),
    alignment_path: Optional[Path] = None,
    reference_path: Optional[Path] = None,
    train_reference_path: Optional[Path] = None,
    validate_builtin: bool = True,
) -> RecomputedEvaluation:
    """Recompute overall and explicit kind×relation P/R/F1 from verified artifacts.

    The overall endpoint intentionally discards relation and kind, matching the
    builtin evaluator's pair-set semantics.  Requested slices require explicit
    relation and kind columns in both the alignment and full reference.  A
    training/null reference is subtracted pair-wise from every endpoint.
    """

    location = Path(location)
    report_path = _primary_report_path(location)
    payload = _read_json_object(report_path)
    if _report_scope(payload, report_path) != "global":
        raise ValueError("P/R/F1 recomputation requires a global evaluation report")
    refs = _metadata_refs(payload, report_path)

    default_alignment = _default_alignment_path(location, refs.get("alignment"))
    alignment_override = alignment_path if alignment_path is not None else default_alignment
    resolved_alignment, alignment_hash = _provenance_path(
        refs.get("alignment"),
        label="alignment",
        report_path=report_path,
        override=alignment_override,
    )
    if not resolved_alignment.name.endswith("maps_global.tsv"):
        raise ValueError(
            "Paper P/R/F1 recomputation requires the canonical maps_global.tsv artifact, "
            f"got {resolved_alignment.name}"
        )
    resolved_reference, reference_hash = _provenance_path(
        refs.get("full_reference"),
        label="full reference",
        report_path=report_path,
        override=reference_path,
    )

    train_provenance = refs.get("train_reference")
    if train_reference_path is not None and not isinstance(train_provenance, Mapping):
        raise ValueError(
            "A train_reference_path cannot be supplied without matching evaluation provenance"
        )
    resolved_train: Optional[Path] = None
    train_hash: Optional[str] = None
    if isinstance(train_provenance, Mapping):
        resolved_train, train_hash = _provenance_path(
            train_provenance,
            label="training/null reference",
            report_path=report_path,
            override=train_reference_path,
        )

    typed = bool(slices)
    predictions = _read_mapping_rows(resolved_alignment, typed=typed, label="alignment")
    references = _read_mapping_rows(resolved_reference, typed=typed, label="full reference")
    null_references = (
        _read_mapping_rows(resolved_train, typed=False, label="training/null reference")
        if resolved_train is not None
        else []
    )
    null_pairs = {(row.source, row.target) for row in null_references}
    overall_counts = _source_counts(
        ((row.source, row.target) for row in predictions),
        ((row.source, row.target) for row in references),
        null_pairs,
    )
    population, population_hash = _frozen_population(report_path)
    if population_hash is not None:
        if set(overall_counts) - population:
            raise ValueError("Evaluated sources fall outside the frozen source population")
        for source in population:
            overall_counts.setdefault(source, SourceConfusion())
    overall = SourceEvaluation(
        source_universe_sha256=population_hash,
        reference_sha256=reference_hash,
        null_reference_sha256=train_hash,
        alignment_sha256=alignment_hash,
        by_source=overall_counts,
    )
    if validate_builtin:
        _validate_builtin_parity(payload, overall.metrics, report_path)

    computed_slices: dict[str, SourceEvaluation] = {}
    for raw_kind, raw_relation in slices:
        kind = EntityKind(raw_kind).value
        relation = normalize_relation(raw_relation)
        name = f"{kind}|{relation}"
        if name in computed_slices:
            raise ValueError(f"Duplicate requested typed slice: {name}")
        slice_counts = _source_counts(
            _typed_pairs(predictions, entity_kind=kind, relation=relation),
            _typed_pairs(references, entity_kind=kind, relation=relation),
            null_pairs,
        )
        for source in overall_counts:
            slice_counts.setdefault(source, SourceConfusion())
        computed_slices[name] = SourceEvaluation(
            source_universe_sha256=population_hash,
            reference_sha256=reference_hash,
            null_reference_sha256=train_hash,
            alignment_sha256=alignment_hash,
            entity_kind=kind,
            relation=relation,
            by_source=slice_counts,
        )
    return RecomputedEvaluation(overall=overall, slices=computed_slices)


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot compute a quantile of an empty sequence")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _aggregate_sources(
    evaluation: SourceEvaluation, sampled_sources: Iterable[str]
) -> SourceConfusion:
    total = SourceConfusion()
    empty = SourceConfusion()
    for source in sampled_sources:
        total = total + evaluation.by_source.get(source, empty)
    return total


def _task_macro_scores(
    arms: Mapping[str, Mapping[tuple[str, int], SourceEvaluation]],
    *,
    tasks: Sequence[str],
    seeds_by_task: Mapping[str, tuple[int, ...]],
    draws_by_task: Optional[Mapping[str, Sequence[str]]] = None,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for arm, cells in sorted(arms.items()):
        task_scores: list[float] = []
        for task in tasks:
            seed_scores: list[float] = []
            for seed in seeds_by_task[task]:
                evaluation = cells[(task, seed)]
                counts = (
                    evaluation.counts
                    if draws_by_task is None
                    else _aggregate_sources(evaluation, draws_by_task[task])
                )
                seed_scores.append(PRFMetrics.from_counts(counts).f1)
            task_scores.append(fmean(seed_scores))
        result[arm] = fmean(task_scores)
    return result


def bootstrap_f1_contrast(
    arms: Mapping[str, Mapping[tuple[str, int], SourceEvaluation]],
    coefficients: Mapping[str, float],
    *,
    resamples: int = 10_000,
    seed: int = 0,
    confidence: float = 0.95,
    contrast: str = "custom",
) -> F1BootstrapResult:
    """Bootstrap a linear contrast of task-macro global F1 values.

    Source IDs are sampled with replacement independently within each task.
    The same task draw is reused for every arm and seed.  TP/FP/FN counts are
    aggregated first and global F1 is recomputed in every replicate; source-F1
    values are never averaged.
    """

    if resamples < 1:
        raise ValueError("resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    if not arms:
        raise ValueError("at least one arm is required")
    if set(coefficients) != set(arms):
        raise ValueError("coefficient keys must exactly match arm keys")
    normalized_coefficients: dict[str, float] = {}
    for arm, raw_value in coefficients.items():
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(f"non-finite coefficient for arm {arm!r}")
        normalized_coefficients[str(arm)] = value

    ordered_arms = sorted(arms)
    first_keys = set(arms[ordered_arms[0]])
    if not first_keys:
        raise ValueError("each arm must contain at least one task/seed cell")
    for arm in ordered_arms[1:]:
        keys = set(arms[arm])
        if keys != first_keys:
            missing = sorted(first_keys.difference(keys))
            extra = sorted(keys.difference(first_keys))
            raise ValueError(
                f"arm {arm!r} has unequal task/seed cells (missing={missing}, extra={extra})"
            )

    tasks = sorted({str(task) for task, _seed in first_keys})
    seeds_by_task = {
        task: tuple(sorted(seed for cell_task, seed in first_keys if cell_task == task))
        for task in tasks
    }
    seed_sets = {seeds for seeds in seeds_by_task.values()}
    if len(seed_sets) != 1:
        raise ValueError(f"tasks must have identical paired seed sets: {seeds_by_task}")

    source_units: dict[str, tuple[str, ...]] = {}
    reference_hashes: dict[str, dict[str, Optional[str]]] = {}
    for task in tasks:
        evaluations = [
            arms[arm][(task, cell_seed)]
            for arm in ordered_arms
            for cell_seed in seeds_by_task[task]
        ]
        populations = {evaluation.source_universe_sha256 for evaluation in evaluations}
        if len(populations) != 1:
            raise ValueError(f"task {task!r} has unequal frozen source populations")
        identities = {
            (evaluation.reference_sha256, evaluation.null_reference_sha256)
            for evaluation in evaluations
        }
        if len(identities) != 1:
            raise ValueError(f"task {task!r} has unequal evaluation reference hashes")
        reference_sha, null_sha = next(iter(identities))
        reference_hashes[task] = {
            "full_reference": reference_sha,
            "train_reference": null_sha,
        }

        units = tuple(
            sorted({source for evaluation in evaluations for source in evaluation.by_source})
        )
        if not units:
            raise ValueError(f"task {task!r} has no source-level decisions")
        for source in units:
            expected_positives = {
                evaluation.by_source.get(source, SourceConfusion()).tp
                + evaluation.by_source.get(source, SourceConfusion()).fn
                for evaluation in evaluations
            }
            if len(expected_positives) != 1:
                raise ValueError(
                    f"task {task!r} source {source!r} has unequal reference-positive counts"
                )
        dimensions = {(evaluation.entity_kind, evaluation.relation) for evaluation in evaluations}
        if len(dimensions) != 1:
            raise ValueError(f"task {task!r} mixes overall and/or different typed slices")
        source_units[task] = units

    point_scores = _task_macro_scores(
        arms,
        tasks=tasks,
        seeds_by_task=seeds_by_task,
    )
    point = sum(normalized_coefficients[arm] * point_scores[arm] for arm in ordered_arms)
    if not math.isfinite(point):
        raise ValueError("F1 contrast is non-finite")
    generator = random.Random(seed)
    samples: list[float] = []
    for _index in range(resamples):
        draws = {
            task: tuple(units[generator.randrange(len(units))] for _item in range(len(units)))
            for task, units in source_units.items()
        }
        scores = _task_macro_scores(
            arms,
            tasks=tasks,
            seeds_by_task=seeds_by_task,
            draws_by_task=draws,
        )
        samples.append(sum(normalized_coefficients[arm] * scores[arm] for arm in ordered_arms))
    if any(not math.isfinite(sample) for sample in samples):
        raise ValueError("bootstrap F1 contrast contains non-finite replicates")
    alpha = (1.0 - confidence) / 2.0
    # Approximate the sampling distribution under H0 by centering each
    # bootstrap replicate on the observed contrast.  The +1 correction keeps
    # the finite Monte Carlo p-value strictly positive.
    extreme = sum(abs(sample - point) >= abs(point) for sample in samples)
    p_value = (extreme + 1) / (resamples + 1)
    full_reference_hashes = {hashes["full_reference"] for hashes in reference_hashes.values()}
    return F1BootstrapResult(
        contrast=contrast,
        arm_scores=point_scores,
        coefficients=normalized_coefficients,
        delta=point,
        ci_low=_quantile(samples, alpha),
        ci_high=_quantile(samples, 1.0 - alpha),
        p_value=p_value,
        confidence=confidence,
        resamples=resamples,
        seed=seed,
        n_tasks=len(tasks),
        n_cells=len(first_keys),
        n_units=sum(len(units) for units in source_units.values()),
        source_units_by_task={task: len(units) for task, units in source_units.items()},
        paired_seeds=next(iter(seed_sets)),
        seeds_by_task=seeds_by_task,
        reference_sha256=(
            next(iter(full_reference_hashes)) if len(full_reference_hashes) == 1 else None
        ),
        reference_hashes_by_task=reference_hashes,
    )


def holm_adjust_p_values(p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm-adjust a frozen family of raw p-values in deterministic order."""

    checked: list[tuple[str, float]] = []
    for name, raw_value in p_values.items():
        value = float(raw_value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"p-value for {name!r} must be finite and between zero and one")
        checked.append((str(name), value))
    ordered = sorted(checked, key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running_max = 0.0
    family_size = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running_max = max(running_max, (family_size - rank) * value)
        adjusted[name] = min(1.0, running_max)
    return dict(sorted(adjusted.items()))


def paired_global_f1_bootstrap(
    baseline: Mapping[tuple[str, int], SourceEvaluation],
    candidate: Mapping[tuple[str, int], SourceEvaluation],
    *,
    resamples: int = 10_000,
    seed: int = 0,
    confidence: float = 0.95,
) -> F1BootstrapResult:
    """Return candidate-minus-baseline task-macro global-F1 inference."""

    return bootstrap_f1_contrast(
        {"baseline": baseline, "candidate": candidate},
        {"baseline": -1.0, "candidate": 1.0},
        resamples=resamples,
        seed=seed,
        confidence=confidence,
        contrast="candidate - baseline",
    )


def e17_interaction_bootstrap(
    cell_00: Mapping[tuple[str, int], SourceEvaluation],
    cell_10: Mapping[tuple[str, int], SourceEvaluation],
    cell_01: Mapping[tuple[str, int], SourceEvaluation],
    cell_11: Mapping[tuple[str, int], SourceEvaluation],
    *,
    resamples: int = 10_000,
    seed: int = 0,
    confidence: float = 0.95,
) -> F1BootstrapResult:
    """Return E17's ``F1_11 - F1_10 - F1_01 + F1_00`` residual."""

    return bootstrap_f1_contrast(
        {
            "cell_00": cell_00,
            "cell_10": cell_10,
            "cell_01": cell_01,
            "cell_11": cell_11,
        },
        {"cell_00": 1.0, "cell_10": -1.0, "cell_01": -1.0, "cell_11": 1.0},
        resamples=resamples,
        seed=seed,
        confidence=confidence,
        contrast="F1_11 - F1_10 - F1_01 + F1_00",
    )


__all__ = [
    "F1BootstrapResult",
    "PRFMetrics",
    "RecomputedEvaluation",
    "SourceConfusion",
    "SourceEvaluation",
    "bootstrap_f1_contrast",
    "e17_interaction_bootstrap",
    "extract_evaluation_metrics",
    "holm_adjust_p_values",
    "normalize_relation",
    "paired_global_f1_bootstrap",
    "recompute_global_prf",
]
