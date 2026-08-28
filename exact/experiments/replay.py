"""Fail-closed replay validation for the frozen E00 production baseline."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NoReturn, cast

from exact.experiments.paper_metrics import extract_evaluation_metrics
from exact.utils.provenance import file_provenance

CPU_SCORE_TOLERANCE = 1e-6
GPU_SCORE_TOLERANCE = 1e-5
METRIC_TOLERANCE = 1e-4

_SOURCE_COLUMNS = ("SrcEntity", "source", "source_id", "src_iri")
_TARGET_COLUMNS = ("TgtEntity", "target", "target_id", "tgt_iri")
_SCORE_COLUMNS = ("Score", "score", "confidence")
_RELATION_COLUMNS = ("Relation", "relation", "relation_label")


class ReplayValidationError(ValueError):
    """Replay failure carrying a JSON-persistable result record."""

    def __init__(self, message: str, record: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.record = dict(record)
        json.dumps(self.record, allow_nan=False)

    def as_record(self) -> dict[str, Any]:
        """Return a defensive copy suitable for atomic JSON persistence."""

        return cast(
            dict[str, Any],
            json.loads(json.dumps(self.record, allow_nan=False)),
        )


def _failure_record(
    code: str,
    message: str,
    *,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_id": "E00",
        "status": "failed",
        "failure": {
            "type": "ReplayValidationError",
            "code": str(code),
            "message": str(message),
            "details": dict(details or {}),
        },
    }


def _fail(
    code: str,
    message: str,
    *,
    details: Mapping[str, Any] | None = None,
) -> NoReturn:
    raise ReplayValidationError(message, _failure_record(code, message, details=details))


def _column(fieldnames: Sequence[str], aliases: Sequence[str], label: str) -> str:
    accepted = {alias.lower() for alias in aliases}
    matches = [str(name) for name in fieldnames if str(name).lower() in accepted]
    if not matches:
        raise ValueError(f"canonical alignment is missing its {label} column")
    if len(matches) > 1:
        raise ValueError(f"canonical alignment has ambiguous {label} columns: {matches}")
    return matches[0]


def _alignment_path(output_dir: Path) -> Path:
    candidates = (
        output_dir / "alignment" / "paper.maps_global.tsv",
        output_dir / "alignment" / "maps_global.tsv",
        output_dir / "alignment" / "src2tgt.maps_global.tsv",
        output_dir / "model" / "alignment" / "src2tgt.maps_global.tsv",
        output_dir / "model" / "alignment" / "maps_global.tsv",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no canonical maps_global.tsv under {output_dir}")


def canonical_alignment(path: Path) -> dict[tuple[str, str, str], float]:
    """Read a canonical alignment keyed by exact source, target, and relation IDs."""

    resolved = Path(path).expanduser().resolve()
    try:
        handle = resolved.open("r", encoding="utf-8", newline="")
    except OSError as exc:
        raise ValueError(f"cannot read canonical alignment {resolved}: {exc}") from exc
    with handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = [str(name) for name in (reader.fieldnames or ())]
        lowered = [name.lower() for name in fields]
        if len(lowered) != len(set(lowered)):
            raise ValueError(f"canonical alignment has duplicate columns in {resolved}")
        source_column = _column(fields, _SOURCE_COLUMNS, "source ID")
        target_column = _column(fields, _TARGET_COLUMNS, "target ID")
        score_column = _column(fields, _SCORE_COLUMNS, "score")
        relation_column = _column(fields, _RELATION_COLUMNS, "relation label")
        rows: dict[tuple[str, str, str], float] = {}
        pair_relations: dict[tuple[str, str], str] = {}
        for row_number, row in enumerate(reader, start=2):
            source = str(row.get(source_column) or "")
            target = str(row.get(target_column) or "")
            relation = str(row.get(relation_column) or "")
            if not source or not target or not relation:
                raise ValueError(f"empty canonical mapping ID at {resolved}:{row_number}")
            if any(value != value.strip() for value in (source, target, relation)):
                raise ValueError(
                    f"canonical mapping IDs/relations contain surrounding whitespace "
                    f"at {resolved}:{row_number}"
                )
            try:
                score = float(row.get(score_column) or "")
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid mapping score at {resolved}:{row_number}") from exc
            if not math.isfinite(score):
                raise ValueError(f"non-finite mapping score at {resolved}:{row_number}")
            key = (source, target, relation)
            if key in rows:
                raise ValueError(f"duplicate canonical mapping {key!r} in {resolved}")
            pair = (source, target)
            previous_relation = pair_relations.get(pair)
            if previous_relation is not None and previous_relation != relation:
                raise ValueError(
                    f"canonical mapping {pair!r} has multiple relation labels "
                    f"{previous_relation!r} and {relation!r} in {resolved}"
                )
            pair_relations[pair] = relation
            rows[key] = score
    return dict(sorted(rows.items()))


def normalize_execution_device(value: Any, *, label: str = "execution device") -> dict[str, Any]:
    """Normalize declared or observed CPU/CUDA metadata without guessing."""

    if value is None:
        _fail("missing_execution_device", f"{label} metadata is missing")
    if isinstance(value, Mapping):
        raw_kind = (
            value.get("kind")
            or value.get("device_kind")
            or value.get("device_type")
            or value.get("type")
        )
        raw_device = value.get("device")
        raw_index = value.get("device_index", value.get("index"))
        raw_name = value.get("device_name", value.get("name"))
    else:
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            return normalize_execution_device(model_dump(mode="python"), label=label)
        raw_kind = getattr(value, "kind", None)
        raw_device = getattr(value, "device", None)
        raw_index = getattr(value, "device_index", None)
        raw_name = getattr(value, "device_name", None)
        if raw_kind is None and raw_device is None:
            raw_kind = value

    kind_text = str(raw_kind or "").strip().lower()
    device_text = str(raw_device or "").strip().lower()

    def hinted_kind(text: str) -> str | None:
        if text == "cpu":
            return "cpu"
        if text in {"gpu", "cuda"} or text.startswith(("gpu:", "cuda:")):
            return "gpu"
        return None

    kind_hint = hinted_kind(kind_text)
    device_hint = hinted_kind(device_text)
    if kind_hint is not None and device_hint is not None and kind_hint != device_hint:
        _fail(
            "conflicting_execution_device_metadata",
            f"{label} has contradictory kind/device metadata",
            details={"kind": kind_text, "device": device_text},
        )
    kind = kind_hint or device_hint
    if kind is None:
        _fail(
            "unsupported_execution_device",
            f"{label} must identify cpu or cuda/gpu, got {raw_kind or raw_device!r}",
        )
    if kind == "cpu":
        device = "cpu"
    else:
        candidate = device_text or kind_text
        try:
            if candidate in {"gpu", "cuda", ""}:
                candidate = f"cuda:{int(raw_index)}" if raw_index is not None else "cuda"
            elif candidate.isdigit():
                candidate = f"cuda:{int(candidate)}"
            elif candidate.startswith("gpu:"):
                candidate = f"cuda:{int(candidate.split(':', 1)[1])}"
            elif candidate.startswith("cuda:"):
                candidate = f"cuda:{int(candidate.split(':', 1)[1])}"
            else:
                _fail(
                    "invalid_execution_device",
                    f"{label} has invalid GPU device {candidate!r}",
                )
            if raw_index is not None:
                indexed_device = f"cuda:{int(raw_index)}"
                if candidate != "cuda" and candidate != indexed_device:
                    _fail(
                        "conflicting_execution_device_metadata",
                        f"{label} has contradictory device/index metadata",
                        details={"device": candidate, "device_index": int(raw_index)},
                    )
                candidate = indexed_device
        except ReplayValidationError:
            raise
        except (TypeError, ValueError) as exc:
            _fail(
                "invalid_execution_device_index",
                f"{label} has invalid GPU device/index metadata",
                details={"device": candidate, "device_index": raw_index},
            )
        device = candidate
    result: dict[str, Any] = {"kind": kind, "device": device}
    if raw_name is not None and str(raw_name).strip():
        result["name"] = str(raw_name).strip()
    return result


def _guard_declared_observed(
    declared: Mapping[str, Any],
    observed: Mapping[str, Any],
    *,
    label: str,
) -> None:
    if declared["kind"] != observed["kind"]:
        _fail(
            "declared_observed_device_mismatch",
            f"{label} declared {declared['kind']!r} but observed {observed['kind']!r}",
            details={"declared": dict(declared), "observed": dict(observed)},
        )
    declared_device = str(declared.get("device") or "")
    observed_device = str(observed.get("device") or "")
    if (
        declared["kind"] == "gpu"
        and declared_device not in {"", "cuda"}
        and declared_device != observed_device
    ):
        _fail(
            "declared_observed_device_mismatch",
            f"{label} declared device {declared_device!r} but observed {observed_device!r}",
            details={"declared": dict(declared), "observed": dict(observed)},
        )


def compare_replay_outputs(
    baseline_output: Path,
    replay_output: Path,
    *,
    execution_kind: str | None = None,
    observed_execution: Any = None,
    declared_execution: Any = None,
) -> dict[str, Any]:
    """Compare one paired E00 cell and return auditable tolerance evidence."""

    observed = normalize_execution_device(
        observed_execution if observed_execution is not None else execution_kind,
        label="observed replay execution device",
    )
    declared = (
        normalize_execution_device(declared_execution, label="declared replay execution device")
        if declared_execution is not None
        else None
    )
    if declared is not None:
        _guard_declared_observed(declared, observed, label="E00 replay")
    kind = str(observed["kind"])
    score_tolerance = GPU_SCORE_TOLERANCE if kind == "gpu" else CPU_SCORE_TOLERANCE
    try:
        baseline_path = _alignment_path(Path(baseline_output))
        replay_path = _alignment_path(Path(replay_output))
        baseline_rows = canonical_alignment(baseline_path)
        replay_rows = canonical_alignment(replay_path)
    except (FileNotFoundError, OSError, ValueError) as exc:
        _fail(
            "invalid_replay_alignment",
            str(exc),
            details={
                "baseline_output": str(Path(baseline_output)),
                "replay_output": str(Path(replay_output)),
            },
        )
    if baseline_rows.keys() != replay_rows.keys():
        missing = sorted(set(baseline_rows).difference(replay_rows))
        unexpected = sorted(set(replay_rows).difference(baseline_rows))
        _fail(
            "canonical_mapping_mismatch",
            "E00 replay changed canonical IDs, selected mappings, or relation labels: "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}",
            details={"missing": missing[:10], "unexpected": unexpected[:10]},
        )
    score_deltas = {key: abs(baseline_rows[key] - replay_rows[key]) for key in baseline_rows}
    max_score_delta = max(score_deltas.values(), default=0.0)
    if max_score_delta > score_tolerance:
        worst = max(score_deltas, key=lambda key: score_deltas[key])
        _fail(
            "score_tolerance_exceeded",
            f"E00 replay score drift {max_score_delta} exceeds {score_tolerance} for {worst!r}",
            details={
                "max_score_delta": max_score_delta,
                "score_tolerance": score_tolerance,
                "mapping": list(worst),
            },
        )

    try:
        baseline_metrics = extract_evaluation_metrics(Path(baseline_output))
        replay_metrics = extract_evaluation_metrics(Path(replay_output))
    except (FileNotFoundError, OSError, ValueError) as exc:
        _fail("invalid_replay_metrics", str(exc))
    if baseline_metrics.keys() != replay_metrics.keys():
        _fail(
            "metric_schema_mismatch",
            "E00 replay aggregate metric keys changed: "
            f"{sorted(baseline_metrics)} != {sorted(replay_metrics)}",
            details={
                "baseline_keys": sorted(baseline_metrics),
                "replay_keys": sorted(replay_metrics),
            },
        )
    metric_deltas = {
        key: abs(float(baseline_metrics[key]) - float(replay_metrics[key]))
        for key in baseline_metrics
    }
    max_metric_delta = max(metric_deltas.values(), default=0.0)
    if max_metric_delta > METRIC_TOLERANCE:
        worst_metric = max(metric_deltas, key=lambda key: metric_deltas[key])
        _fail(
            "metric_tolerance_exceeded",
            f"E00 replay metric drift {max_metric_delta} exceeds {METRIC_TOLERANCE} "
            f"for {worst_metric!r}",
            details={
                "max_metric_delta": max_metric_delta,
                "metric_tolerance": METRIC_TOLERANCE,
                "metric": worst_metric,
            },
        )
    canonical_rows = [
        {"source": source, "target": target, "relation": relation}
        for source, target, relation in baseline_rows
    ]
    return {
        "status": "passed",
        "execution_kind": kind,
        "observed_execution": observed,
        "declared_execution": declared,
        "mapping_count": len(baseline_rows),
        "canonical_order": canonical_rows,
        "score_tolerance": score_tolerance,
        "max_score_delta": max_score_delta,
        "metric_tolerance": METRIC_TOLERANCE,
        "max_metric_delta": max_metric_delta,
        "metric_deltas": dict(sorted(metric_deltas.items())),
        "baseline_alignment": file_provenance(baseline_path),
        "replay_alignment": file_provenance(replay_path),
    }


def validate_e00_replay(
    manifests: Sequence[Mapping[str, Any]],
    *,
    baseline_arm: str = "R_0",
    replay_arm: str = "baseline_replay",
) -> dict[str, Any]:
    """Validate every paired, completed E00 baseline/replay cell."""

    by_arm: dict[str, dict[tuple[str, int], Mapping[str, Any]]] = {
        baseline_arm: {},
        replay_arm: {},
    }
    for manifest in manifests:
        if manifest.get("experiment_id") != "E00" or manifest.get("status") != "complete":
            continue
        arm = str(manifest.get("arm_id"))
        if arm not in by_arm:
            continue
        task_id = str(manifest.get("task_id") or "").strip()
        if not task_id:
            _fail("invalid_replay_manifest", "completed E00 replay manifest has no task_id")
        try:
            seed = int(str(manifest.get("seed")))
        except (TypeError, ValueError) as exc:
            _fail(
                "invalid_replay_manifest",
                f"completed E00 replay manifest has invalid seed {manifest.get('seed')!r}",
            )
        key = (task_id, seed)
        if key in by_arm[arm]:
            _fail(
                "duplicate_replay_cell",
                f"duplicate E00 replay cell for {arm}/{key[0]}/seed-{key[1]}",
            )
        by_arm[arm][key] = manifest
    baseline_cells = set(by_arm[baseline_arm])
    replay_cells = set(by_arm[replay_arm])
    if not baseline_cells or baseline_cells != replay_cells:
        _fail(
            "incomplete_replay_pairs",
            "E00 replay requires complete paired cells: "
            f"{baseline_arm}={sorted(baseline_cells)}, {replay_arm}={sorted(replay_cells)}",
            details={
                "baseline_arm": baseline_arm,
                "replay_arm": replay_arm,
                "baseline_cells": [list(cell) for cell in sorted(baseline_cells)],
                "replay_cells": [list(cell) for cell in sorted(replay_cells)],
            },
        )
    comparisons: list[dict[str, Any]] = []
    for task_id, seed in sorted(baseline_cells):
        baseline = by_arm[baseline_arm][(task_id, seed)]
        replay = by_arm[replay_arm][(task_id, seed)]
        baseline_resource = baseline.get("resource")
        replay_resource = replay.get("resource")

        def output_path(manifest: Mapping[str, Any]) -> Path:
            fingerprint = manifest.get("fingerprint_payload")
            value = fingerprint.get("output_dir") if isinstance(fingerprint, Mapping) else None
            if not value:
                _fail(
                    "missing_replay_output",
                    f"completed E00 manifest has no output path for "
                    f"{manifest.get('arm_id')}/{task_id}/seed-{seed}",
                    details={
                        "arm_id": str(manifest.get("arm_id")),
                        "task_id": task_id,
                        "seed": seed,
                    },
                )
            return Path(str(value)).expanduser().resolve()

        def observed_device(manifest: Mapping[str, Any]) -> dict[str, Any]:
            value = next(
                (
                    manifest.get(key)
                    for key in (
                        "observed_execution",
                        "execution_device",
                        "runtime_device",
                    )
                    if manifest.get(key) is not None
                ),
                None,
            )
            if value is None:
                _fail(
                    "missing_observed_execution_device",
                    f"completed E00 manifest has no observed execution-device metadata for "
                    f"{manifest.get('arm_id')}/{task_id}/seed-{seed}",
                )
            return normalize_execution_device(
                value,
                label=f"observed {manifest.get('arm_id')} execution device",
            )

        baseline_declared = normalize_execution_device(
            baseline_resource, label=f"declared {baseline_arm} execution device"
        )
        replay_declared = normalize_execution_device(
            replay_resource, label=f"declared {replay_arm} execution device"
        )
        baseline_observed = observed_device(baseline)
        replay_observed = observed_device(replay)
        _guard_declared_observed(
            baseline_declared,
            baseline_observed,
            label=f"{baseline_arm}/{task_id}/seed-{seed}",
        )
        _guard_declared_observed(
            replay_declared,
            replay_observed,
            label=f"{replay_arm}/{task_id}/seed-{seed}",
        )
        if (
            baseline_observed["kind"] != replay_observed["kind"]
            or baseline_observed["device"] != replay_observed["device"]
        ):
            _fail(
                "paired_observed_device_mismatch",
                f"E00 paired cells changed observed execution device for " f"{task_id}/seed-{seed}",
                details={
                    "baseline_observed": baseline_observed,
                    "replay_observed": replay_observed,
                },
            )
        if (
            baseline_observed.get("name")
            and replay_observed.get("name")
            and baseline_observed["name"] != replay_observed["name"]
        ):
            _fail(
                "paired_observed_device_mismatch",
                f"E00 paired cells changed observed device name for {task_id}/seed-{seed}",
                details={
                    "baseline_observed": baseline_observed,
                    "replay_observed": replay_observed,
                },
            )
        try:
            comparison = compare_replay_outputs(
                output_path(baseline),
                output_path(replay),
                observed_execution=baseline_observed,
                declared_execution=baseline_declared,
            )
        except ReplayValidationError as exc:
            record = exc.as_record()
            record.update(
                {
                    "baseline_arm": baseline_arm,
                    "replay_arm": replay_arm,
                    "task_id": task_id,
                    "seed": seed,
                }
            )
            raise ReplayValidationError(str(exc), record) from exc
        comparison["baseline_observed_execution"] = baseline_observed
        comparison["replay_observed_execution"] = replay_observed
        comparison["baseline_declared_execution"] = baseline_declared
        comparison["replay_declared_execution"] = replay_declared
        comparison.update({"task_id": task_id, "seed": seed})
        comparisons.append(comparison)
    return {
        "schema_version": 1,
        "experiment_id": "E00",
        "status": "passed",
        "baseline_arm": baseline_arm,
        "replay_arm": replay_arm,
        "comparisons": comparisons,
    }


__all__ = [
    "CPU_SCORE_TOLERANCE",
    "GPU_SCORE_TOLERANCE",
    "METRIC_TOLERANCE",
    "ReplayValidationError",
    "canonical_alignment",
    "compare_replay_outputs",
    "normalize_execution_device",
    "validate_e00_replay",
]
