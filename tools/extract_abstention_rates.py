#!/usr/bin/env python3
"""Harvest abstention counters and committed-mapping quality from existing runs.

The script is deliberately inference-free.  It walks every ``exact.log`` below
``exp/test``, keeps the last completed candidate-set selector record in each
log, and writes one row per log (including logs without a selector record).

For completed selector runs, per-source abstentions are recovered from the
saved ``summary_metrics.csv`` candidate records when possible.  If those
annotations are unavailable, the fallback required by WP3 is used and clearly
labelled: a reference source with no committed mapping is treated as derived
abstained for the restricted-recall calculation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXP_ROOT = REPO_ROOT / "exp" / "test"
DEFAULT_OUTPUT = DEFAULT_EXP_ROOT / "review_response" / "analysis" / "e3_abstention.csv"

FINAL_CANDIDATE_MARKER = " - INFO - Candidate-set selector:"
FINAL_CALIBRATED_MARKER = " - INFO - Calibrated selector:"

NEW_COUNTER_RE = re.compile(
    r"\bsources=(?P<processed>\d+)/(?P<sources>\d+),\s*"
    r"abstained_sources=(?P<abstained>\d+),\s*"
    r"llm_sources=(?P<llm_sources>\d+)\b"
)
LEGACY_COUNTER_RE = re.compile(
    r"\b(?P<processed>\d+)\s+source groups/(?P<sources>\d+)\s+unique sources,\s*"
    r"abstained=(?P<abstained>\d+),\s*"
    r"llm_groups=(?P<llm_sources>\d+)\b"
)
KEY_VALUE_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9_]*)=([^,\s]+)")

SOURCE_COLUMNS = ("SrcEntity", "src_iri", "source", "source_iri")
TARGET_COLUMNS = ("TgtEntity", "tgt_iri", "target", "target_iri")


CSV_COLUMNS = (
    "run_dir",
    "task",
    "setting",
    "selector_status",
    "sources",
    "sources_processed",
    "abstained",
    "abstention_rate",
    "llm_sources",
    "llm_invocation_rate",
    "selector_P",
    "selector_R",
    "selector_F1",
    "selector_TP",
    "selector_FP",
    "selector_FN",
    "selector_metric_scope",
    "selector_line_format",
    "selector_line_number",
    "calibrated_line_number",
    "selector_annotation_status",
    "abstention_source_method",
    "annotation_sources",
    "annotation_abstained",
    "annotation_llm_sources",
    "annotation_conflicting_sources",
    "selector_counter_consistent",
    "n_committed_rows",
    "n_committed",
    "n_reference_rows",
    "n_reference",
    "n_reference_sources",
    "n_true_positive",
    "P_committed",
    "P_committed_numerator",
    "P_committed_denominator",
    "R_all",
    "R_all_numerator",
    "R_all_denominator",
    "R_committed",
    "R_committed_numerator",
    "R_committed_denominator",
    "R_committed_minus_R_all",
    "n_abstained_sources_with_reference",
    "selector_minus_P_committed",
    "selector_minus_R_committed",
    "quality_status",
    "run_config",
    "alignment_path",
    "reference_path",
    "selector_annotation_path",
    "log_mtime",
    "alignment_mtime",
    "reference_mtime",
    "selector_annotation_mtime",
    "notes",
)


@dataclass(frozen=True)
class SelectorRecord:
    processed: int
    sources: int
    abstained: int
    llm_sources: int
    line_number: int
    line_format: str
    calibrated_line_number: int | None
    calibrated_values: dict[str, str]


@dataclass(frozen=True)
class RunContext:
    log_path: Path
    run_dir: Path
    run_config_path: Path | None
    run_config: dict[str, Any]
    task: str
    setting: str
    alignment_path: Path | None
    reference_path: Path | None
    annotation_path: Path | None
    selector: SelectorRecord | None
    discovery_notes: tuple[str, ...]


def _strip_terminal_period(value: str) -> str:
    return value[:-1] if value.endswith(".") else value


def _parse_candidate_payload(
    payload: str,
    *,
    line_number: int,
    calibrated_line_number: int | None,
    calibrated_values: dict[str, str],
) -> SelectorRecord | None:
    """Parse either historical final candidate-selector counter format."""
    match = NEW_COUNTER_RE.search(payload)
    line_format = "sources"
    if match is None:
        match = LEGACY_COUNTER_RE.search(payload)
        line_format = "legacy_source_groups"
    if match is None:
        return None

    values = {key: int(value) for key, value in match.groupdict().items()}
    return SelectorRecord(
        processed=values["processed"],
        sources=values["sources"],
        abstained=values["abstained"],
        llm_sources=values["llm_sources"],
        line_number=line_number,
        line_format=line_format,
        calibrated_line_number=calibrated_line_number,
        calibrated_values=dict(calibrated_values),
    )


def _parse_key_values(payload: str) -> dict[str, str]:
    return {key: _strip_terminal_period(value) for key, value in KEY_VALUE_RE.findall(payload)}


def parse_final_selector(log_path: Path) -> tuple[SelectorRecord | None, list[str]]:
    """Return the last completed selector record, never a progress record."""
    latest_calibrated_values: dict[str, str] = {}
    latest_calibrated_line: int | None = None
    final_record: SelectorRecord | None = None
    unparsed_final_lines = 0

    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            if FINAL_CALIBRATED_MARKER in line:
                payload = line.split(FINAL_CALIBRATED_MARKER, 1)[1].strip()
                latest_calibrated_values = _parse_key_values(payload)
                latest_calibrated_line = line_number
                continue
            if FINAL_CANDIDATE_MARKER not in line:
                continue
            payload = line.split(FINAL_CANDIDATE_MARKER, 1)[1].strip()
            parsed = _parse_candidate_payload(
                payload,
                line_number=line_number,
                calibrated_line_number=latest_calibrated_line,
                calibrated_values=latest_calibrated_values,
            )
            if parsed is None:
                unparsed_final_lines += 1
            else:
                final_record = parsed

    notes: list[str] = []
    if unparsed_final_lines:
        notes.append(f"unparsed final candidate-selector lines={unparsed_final_lines}")
    if final_record is not None:
        if final_record.processed != final_record.sources:
            notes.append(
                "final selector was incomplete: "
                f"processed={final_record.processed}, sources={final_record.sources}"
            )
        if final_record.abstained > final_record.sources:
            notes.append("logged abstained count exceeds logged sources")
        if final_record.llm_sources > final_record.sources:
            notes.append("logged LLM count exceeds logged sources")
    return final_record, notes


def _resolve_repo_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else REPO_ROOT / path


def _normalise_path(path: Path) -> Path:
    return path.resolve(strict=False)


def _load_run_yaml(path: Path) -> dict[str, Any] | None:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError):
        return None
    if not isinstance(loaded, dict):
        return None
    if not isinstance(loaded.get("dataset"), dict):
        return None
    if not isinstance(loaded.get("job"), dict):
        return None
    return loaded


def discover_run_config(run_dir: Path) -> tuple[Path | None, dict[str, Any], list[str]]:
    """Choose the immediate run YAML whose output_dir best matches ``run_dir``."""
    options: list[tuple[int, Path, dict[str, Any]]] = []
    for yaml_path in sorted(run_dir.glob("*.yaml")):
        loaded = _load_run_yaml(yaml_path)
        if loaded is None:
            continue
        score = 0
        raw_output = loaded.get("job", {}).get("output_dir")
        if raw_output:
            output_path = _normalise_path(_resolve_repo_path(str(raw_output)))
            if output_path == _normalise_path(run_dir):
                score += 100
        if loaded.get("dataset", {}).get("full_reference"):
            score += 10
        options.append((score, yaml_path, loaded))

    if not options:
        return None, {}, ["no dataset/job run YAML found"]
    options.sort(key=lambda item: (-item[0], str(item[1])))
    score, path, loaded = options[0]
    notes: list[str] = []
    equally_ranked = [item for item in options if item[0] == score]
    if len(equally_ranked) > 1:
        notes.append(
            "multiple equally ranked run YAMLs; selected "
            f"{path.name} from {', '.join(item[1].name for item in equally_ranked)}"
        )
    return path, loaded, notes


def _resolve_reference(dataset: dict[str, Any]) -> Path | None:
    raw_reference = dataset.get("full_reference")
    if not raw_reference:
        return None
    reference = Path(str(raw_reference))
    if reference.is_absolute():
        return reference

    data_dir_raw = dataset.get("data_dir")
    candidates: list[Path] = []
    if data_dir_raw:
        candidates.append(_resolve_repo_path(str(data_dir_raw)) / reference)
    candidates.append(REPO_ROOT / reference)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else None


def _find_alignment(run_dir: Path, setting_hint: str) -> tuple[Path | None, str]:
    global_path = run_dir / "model" / "alignment" / "src2tgt.maps_global.tsv"
    local_path = run_dir / "model" / "alignment" / "src2tgt.maps_local.tsv"
    if global_path.exists():
        return global_path, "global"
    if local_path.exists():
        return local_path, "local"
    expected = global_path if setting_hint == "global" else local_path
    return expected, setting_hint


def _find_annotation(run_dir: Path) -> Path | None:
    preferred = run_dir / "model" / "alignment" / "default" / "summary_metrics.csv"
    if preferred.exists():
        return preferred
    candidates = sorted((run_dir / "model" / "alignment").glob("*/summary_metrics.csv"))
    return candidates[0] if candidates else None


def build_context(log_path: Path, exp_root: Path) -> RunContext:
    run_dir = log_path.parent
    selector, selector_notes = parse_final_selector(log_path)
    run_config_path, run_config, config_notes = discover_run_config(run_dir)
    dataset = run_config.get("dataset", {})

    data_dir_raw = dataset.get("data_dir")
    task = Path(str(data_dir_raw)).name if data_dir_raw else run_dir.name
    candidates = dataset.get("candidates", "__missing__")
    setting_hint = "global" if candidates in (None, "", "__missing__") else "local"
    alignment_path, setting = _find_alignment(run_dir, setting_hint)
    reference_path = _resolve_reference(dataset)
    annotation_path = _find_annotation(run_dir)

    notes = selector_notes + config_notes
    if selector is not None and setting != "global":
        notes.append("selector line found in a run inferred as local")
    if alignment_path is not None and not alignment_path.exists():
        notes.append("committed alignment artifact is missing")
    if reference_path is not None and not reference_path.exists():
        notes.append("full_reference artifact is missing")

    return RunContext(
        log_path=log_path,
        run_dir=run_dir,
        run_config_path=run_config_path,
        run_config=run_config,
        task=task,
        setting=setting,
        alignment_path=alignment_path,
        reference_path=reference_path,
        annotation_path=annotation_path,
        selector=selector,
        discovery_notes=tuple(notes),
    )


def _first_column(header: Iterable[str], candidates: tuple[str, ...]) -> str | None:
    lookup = {column.lstrip("\ufeff"): column for column in header}
    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]
    return None


def read_pairs(path: Path) -> dict[str, Any]:
    """Read a mapping/reference TSV into a unique pair set."""
    pairs: set[tuple[str, str]] = set()
    sources: set[str] = set()
    row_count = 0
    malformed_rows = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError(f"{path} has no TSV header")
        source_column = _first_column(reader.fieldnames, SOURCE_COLUMNS)
        target_column = _first_column(reader.fieldnames, TARGET_COLUMNS)
        if source_column is None or target_column is None:
            raise ValueError(f"{path} lacks source/target columns; header={reader.fieldnames!r}")
        for row in reader:
            row_count += 1
            source = (row.get(source_column) or "").strip()
            target = (row.get(target_column) or "").strip()
            if not source or not target:
                malformed_rows += 1
                continue
            pairs.add((source, target))
            sources.add(source)
    return {
        "pairs": pairs,
        "sources": sources,
        "row_count": row_count,
        "malformed_rows": malformed_rows,
    }


def _parse_bool(raw_value: str) -> bool | None:
    value = raw_value.strip().lower()
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no"}:
        return False
    return None


def scan_selector_annotations(path: Path) -> dict[str, Any]:
    """Stream selector flags without loading the multi-GB candidate CSV."""
    csv.field_size_limit(sys.maxsize)
    source_states: dict[str, bool] = {}
    llm_states: dict[str, bool] = {}
    conflicting_sources: set[str] = set()
    malformed_rows = 0
    unresolved_abstention_rows = 0
    unresolved_llm_rows = 0
    row_count = 0

    stat_before = path.stat()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        # Despite its historical ``.csv`` suffix this artifact is tab-delimited.
        reader = csv.reader(handle, delimiter="\t")
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"{path} is empty") from exc

        source_column = _first_column(header, SOURCE_COLUMNS)
        if source_column is None:
            raise ValueError(f"{path} has no source column")
        if "selector_abstained" not in header:
            raise ValueError(f"{path} has no selector_abstained column")
        source_index = header.index(source_column)
        abstained_index = header.index("selector_abstained")
        llm_index = header.index("selector_llm_used") if "selector_llm_used" in header else None
        required_index = max(
            source_index,
            abstained_index,
            llm_index if llm_index is not None else 0,
        )

        for row in reader:
            row_count += 1
            if len(row) <= required_index:
                malformed_rows += 1
                continue
            source = row[source_index].strip()
            if not source:
                malformed_rows += 1
                continue

            abstained = _parse_bool(row[abstained_index])
            if abstained is None:
                unresolved_abstention_rows += 1
            else:
                previous = source_states.setdefault(source, abstained)
                if previous != abstained:
                    conflicting_sources.add(source)

            if llm_index is not None:
                llm_used = _parse_bool(row[llm_index])
                if llm_used is None:
                    unresolved_llm_rows += 1
                else:
                    previous_llm = llm_states.setdefault(source, llm_used)
                    if previous_llm != llm_used:
                        conflicting_sources.add(source)

    stat_after = path.stat()
    changed_during_scan = (
        stat_before.st_size != stat_after.st_size
        or stat_before.st_mtime_ns != stat_after.st_mtime_ns
    )
    return {
        "abstained_sources": {source for source, abstained in source_states.items() if abstained},
        "llm_sources": {source for source, used in llm_states.items() if used},
        "unique_sources": len(source_states),
        "row_count": row_count,
        "malformed_rows": malformed_rows,
        "unresolved_abstention_rows": unresolved_abstention_rows,
        "unresolved_llm_rows": unresolved_llm_rows,
        "conflicting_sources": conflicting_sources,
        "changed_during_scan": changed_during_scan,
    }


def _file_fingerprint(path: Path) -> tuple[int, int, str]:
    """Cheap content fingerprint used only to avoid rescanning copied artifacts."""
    stat = path.stat()
    chunk_size = 64 * 1024
    with path.open("rb") as handle:
        first = handle.read(chunk_size)
        if stat.st_size > chunk_size:
            handle.seek(max(0, stat.st_size - chunk_size))
            last = handle.read(chunk_size)
        else:
            last = b""
    digest = hashlib.sha256(first + b"\0" + last).hexdigest()
    return stat.st_size, stat.st_mtime_ns, digest


def scan_all_annotations(
    contexts: list[RunContext],
    *,
    workers: int,
) -> tuple[dict[Path, dict[str, Any]], dict[Path, str]]:
    """Scan unique selector annotation artifacts, in parallel when requested."""
    path_to_fingerprint: dict[Path, tuple[int, int, str]] = {}
    fingerprint_to_path: dict[tuple[int, int, str], Path] = {}
    errors: dict[Path, str] = {}
    for context in contexts:
        path = context.annotation_path
        if context.selector is None or path is None or not path.exists():
            continue
        try:
            fingerprint = _file_fingerprint(path)
        except OSError as exc:
            errors[path] = f"annotation fingerprint failed: {exc}"
            continue
        path_to_fingerprint[path] = fingerprint
        fingerprint_to_path.setdefault(fingerprint, path)

    results_by_fingerprint: dict[tuple[int, int, str], dict[str, Any]] = {}
    unique_items = list(fingerprint_to_path.items())
    if workers <= 1 or len(unique_items) <= 1:
        for fingerprint, path in unique_items:
            try:
                results_by_fingerprint[fingerprint] = scan_selector_annotations(path)
            except Exception as exc:  # keep all logs represented in the CSV
                errors[path] = f"annotation scan failed: {exc}"
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(scan_selector_annotations, path): (fingerprint, path)
                for fingerprint, path in unique_items
            }
            for future in as_completed(futures):
                fingerprint, path = futures[future]
                try:
                    results_by_fingerprint[fingerprint] = future.result()
                except Exception as exc:  # keep all logs represented in the CSV
                    errors[path] = f"annotation scan failed: {exc}"

    results: dict[Path, dict[str, Any]] = {}
    for path, fingerprint in path_to_fingerprint.items():
        if fingerprint in results_by_fingerprint:
            results[path] = results_by_fingerprint[fingerprint]
    return results, errors


def _ratio(numerator: int, denominator: int) -> str:
    return "" if denominator == 0 else f"{numerator / denominator:.12f}"


def _float_or_blank(values: dict[str, str], key: str) -> str:
    value = values.get(key)
    if value is None:
        return ""
    try:
        return f"{float(value):.12f}"
    except ValueError:
        return ""


def _difference(left: str, right: str) -> str:
    if left == "" or right == "":
        return ""
    return f"{float(left) - float(right):.12f}"


def _relative(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return path.resolve(strict=False).relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _mtime(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    timestamp = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return timestamp.isoformat(timespec="seconds").replace("+00:00", "Z")


def _selector_metric_values(selector: SelectorRecord) -> dict[str, str]:
    values = selector.calibrated_values
    if any(key in values for key in ("selected_P", "selected_R", "selected_F1")):
        return {
            "P": _float_or_blank(values, "selected_P"),
            "R": _float_or_blank(values, "selected_R"),
            "F1": _float_or_blank(values, "selected_F1"),
            "TP": _float_or_blank(values, "selected_TP"),
            "FP": _float_or_blank(values, "selected_FP"),
            "FN": _float_or_blank(values, "selected_FN"),
            "scope": "selected",
        }
    return {"P": "", "R": "", "F1": "", "TP": "", "FP": "", "FN": "", "scope": ""}


def _empty_quality_fields() -> dict[str, str | int]:
    return {
        "n_committed_rows": "",
        "n_committed": "",
        "n_reference_rows": "",
        "n_reference": "",
        "n_reference_sources": "",
        "n_true_positive": "",
        "P_committed": "",
        "P_committed_numerator": "",
        "P_committed_denominator": "",
        "R_all": "",
        "R_all_numerator": "",
        "R_all_denominator": "",
        "R_committed": "",
        "R_committed_numerator": "",
        "R_committed_denominator": "",
        "R_committed_minus_R_all": "",
        "n_abstained_sources_with_reference": "",
    }


def make_row(
    context: RunContext,
    annotation_results: dict[Path, dict[str, Any]],
    pair_cache: dict[tuple[int, int, str], dict[str, Any]],
    *,
    annotation_scan_enabled: bool,
) -> dict[str, Any]:
    notes = list(context.discovery_notes)
    selector = context.selector
    selector_status = "found" if selector is not None else "no_selector_line"
    selector_metrics = (
        _selector_metric_values(selector)
        if selector is not None
        else {"P": "", "R": "", "F1": "", "TP": "", "FP": "", "FN": "", "scope": ""}
    )

    row: dict[str, Any] = {column: "" for column in CSV_COLUMNS}
    row.update(
        {
            "run_dir": _relative(context.run_dir),
            "task": context.task,
            "setting": context.setting,
            "selector_status": selector_status,
            "selector_P": selector_metrics["P"],
            "selector_R": selector_metrics["R"],
            "selector_F1": selector_metrics["F1"],
            "selector_TP": selector_metrics["TP"],
            "selector_FP": selector_metrics["FP"],
            "selector_FN": selector_metrics["FN"],
            "selector_metric_scope": selector_metrics["scope"],
            "quality_status": "unavailable",
            "run_config": _relative(context.run_config_path),
            "alignment_path": _relative(context.alignment_path),
            "reference_path": _relative(context.reference_path),
            "selector_annotation_path": _relative(context.annotation_path),
            "log_mtime": _mtime(context.log_path),
            "alignment_mtime": _mtime(context.alignment_path),
            "reference_mtime": _mtime(context.reference_path),
            "selector_annotation_mtime": _mtime(context.annotation_path),
        }
    )

    if selector is not None:
        row.update(
            {
                "sources": selector.sources,
                "sources_processed": selector.processed,
                "abstained": selector.abstained,
                "abstention_rate": _ratio(selector.abstained, selector.sources),
                "llm_sources": selector.llm_sources,
                "llm_invocation_rate": _ratio(selector.llm_sources, selector.sources),
                "selector_line_format": selector.line_format,
                "selector_line_number": selector.line_number,
                "calibrated_line_number": selector.calibrated_line_number or "",
            }
        )

    annotation: dict[str, Any] | None = None
    if selector is not None and context.annotation_path is not None:
        annotation = annotation_results.get(context.annotation_path)
    if annotation is not None:
        annotation_abstained = annotation["abstained_sources"]
        annotation_llm = annotation["llm_sources"]
        conflicts = annotation["conflicting_sources"]
        row.update(
            {
                "selector_annotation_status": "available",
                "abstention_source_method": "annotated_summary_metrics",
                "annotation_sources": annotation["unique_sources"],
                "annotation_abstained": len(annotation_abstained),
                "annotation_llm_sources": len(annotation_llm),
                "annotation_conflicting_sources": len(conflicts),
            }
        )
        counter_consistent = (
            selector is not None
            and annotation["unique_sources"] == selector.sources
            and len(annotation_abstained) == selector.abstained
            and len(annotation_llm) == selector.llm_sources
            and not conflicts
            and not annotation["changed_during_scan"]
        )
        row["selector_counter_consistent"] = str(counter_consistent).lower()
        if annotation["malformed_rows"]:
            notes.append(f"annotation malformed rows={annotation['malformed_rows']}")
        if annotation["unresolved_abstention_rows"]:
            notes.append(
                "annotation rows with unresolved abstention flag="
                f"{annotation['unresolved_abstention_rows']}"
            )
        if annotation["unresolved_llm_rows"]:
            notes.append(
                f"annotation rows with unresolved LLM flag={annotation['unresolved_llm_rows']}"
            )
        if conflicts:
            notes.append(f"annotation sources with conflicting flags={len(conflicts)}")
        if annotation["changed_during_scan"]:
            notes.append("selector annotation changed during scan")
        if not counter_consistent:
            notes.append("selector log counters disagree with candidate-record annotations")
    elif selector is not None:
        if (
            not annotation_scan_enabled
            and context.annotation_path is not None
            and context.annotation_path.exists()
        ):
            row["selector_annotation_status"] = "available_not_scanned"
            notes.append(
                "saved selector annotations were not scanned; use "
                "--scan-selector-annotations to read the large candidate artifacts"
            )
        else:
            row["selector_annotation_status"] = "unavailable"

    quality = _empty_quality_fields()
    alignment_path = context.alignment_path
    reference_path = context.reference_path
    if (
        alignment_path is not None
        and alignment_path.exists()
        and reference_path is not None
        and reference_path.exists()
    ):
        try:
            alignment_key = _file_fingerprint(alignment_path)
            reference_key = _file_fingerprint(reference_path)
            if alignment_key not in pair_cache:
                pair_cache[alignment_key] = read_pairs(alignment_path)
            if reference_key not in pair_cache:
                pair_cache[reference_key] = read_pairs(reference_path)
            committed = pair_cache[alignment_key]
            reference = pair_cache[reference_key]

            committed_pairs = committed["pairs"]
            reference_pairs = reference["pairs"]
            true_positive_pairs = committed_pairs & reference_pairs
            quality.update(
                {
                    "n_committed_rows": committed["row_count"],
                    "n_committed": len(committed_pairs),
                    "n_reference_rows": reference["row_count"],
                    "n_reference": len(reference_pairs),
                    "n_reference_sources": len(reference["sources"]),
                    "n_true_positive": len(true_positive_pairs),
                    "P_committed": _ratio(len(true_positive_pairs), len(committed_pairs)),
                    "P_committed_numerator": len(true_positive_pairs),
                    "P_committed_denominator": len(committed_pairs),
                    "R_all": _ratio(len(true_positive_pairs), len(reference_pairs)),
                    "R_all_numerator": len(true_positive_pairs),
                    "R_all_denominator": len(reference_pairs),
                }
            )
            if committed["malformed_rows"]:
                notes.append(f"committed alignment malformed rows={committed['malformed_rows']}")
            if reference["malformed_rows"]:
                notes.append(f"reference malformed rows={reference['malformed_rows']}")

            abstained_sources: set[str] | None = None
            if annotation is not None:
                abstained_sources = annotation["abstained_sources"]
            elif selector is not None:
                abstained_sources = reference["sources"] - committed["sources"]
                row["abstention_source_method"] = (
                    "derived_reference_sources_without_committed_mapping"
                )
                notes.append(
                    "per-source selector flags unavailable; restricted recall uses the "
                    "explicitly labelled no-committed-mapping derivation"
                )

            if abstained_sources is not None:
                eligible_reference_pairs = {
                    pair for pair in reference_pairs if pair[0] not in abstained_sources
                }
                eligible_true_positives = true_positive_pairs & eligible_reference_pairs
                abstained_reference_sources = reference["sources"] & abstained_sources
                quality.update(
                    {
                        "R_committed": _ratio(
                            len(eligible_true_positives),
                            len(eligible_reference_pairs),
                        ),
                        "R_committed_numerator": len(eligible_true_positives),
                        "R_committed_denominator": len(eligible_reference_pairs),
                        "n_abstained_sources_with_reference": len(abstained_reference_sources),
                    }
                )
                quality["R_committed_minus_R_all"] = _difference(
                    str(quality["R_committed"]),
                    str(quality["R_all"]),
                )
            row["quality_status"] = "computed"
        except Exception as exc:
            row["quality_status"] = "error"
            notes.append(f"committed-quality computation failed: {exc}")
    else:
        missing = []
        if alignment_path is None or not alignment_path.exists():
            missing.append("alignment")
        if reference_path is None or not reference_path.exists():
            missing.append("reference")
        notes.append(f"committed-quality unavailable: missing {', '.join(missing)}")

    row.update(quality)
    row["selector_minus_P_committed"] = _difference(
        str(row["selector_P"]),
        str(row["P_committed"]),
    )
    row["selector_minus_R_committed"] = _difference(
        str(row["selector_R"]),
        str(row["R_committed"]),
    )
    row["notes"] = "; ".join(notes)
    return row


def extract(
    exp_root: Path,
    output_path: Path,
    *,
    workers: int,
    scan_annotations: bool,
) -> list[dict[str, Any]]:
    log_paths = sorted(exp_root.rglob("exact.log"))
    contexts = [build_context(log_path, exp_root) for log_path in log_paths]
    if scan_annotations:
        annotation_results, annotation_errors = scan_all_annotations(
            contexts,
            workers=workers,
        )
    else:
        annotation_results, annotation_errors = {}, {}

    pair_cache: dict[tuple[int, int, str], dict[str, Any]] = {}
    rows = [
        make_row(
            context,
            annotation_results,
            pair_cache,
            annotation_scan_enabled=scan_annotations,
        )
        for context in contexts
    ]
    if annotation_errors:
        for row, context in zip(rows, contexts):
            annotation_path = context.annotation_path
            error = annotation_errors.get(annotation_path) if annotation_path else None
            if error:
                row["notes"] = "; ".join(filter(None, (row["notes"], error)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def run_self_test() -> None:
    calibrated = {
        "selected_P": "0.904",
        "selected_R": "0.651",
        "selected_F1": "0.757",
    }
    new_record = _parse_candidate_payload(
        "sources=8599/8599, abstained_sources=4442, llm_sources=0, elapsed=1.0s.",
        line_number=10,
        calibrated_line_number=9,
        calibrated_values=calibrated,
    )
    assert new_record is not None
    assert (new_record.sources, new_record.abstained, new_record.llm_sources) == (
        8599,
        4442,
        0,
    )
    assert new_record.line_format == "sources"

    legacy_record = _parse_candidate_payload(
        "15762 source groups/15762 unique sources, abstained=16, "
        "llm_groups=11297, elapsed=1.0s.",
        line_number=2,
        calibrated_line_number=None,
        calibrated_values={},
    )
    assert legacy_record is not None
    assert (
        legacy_record.sources,
        legacy_record.abstained,
        legacy_record.llm_sources,
    ) == (15762, 16, 11297)
    assert legacy_record.line_format == "legacy_source_groups"
    assert _parse_bool("True") is True
    assert _parse_bool("false") is False
    assert _parse_bool("") is None
    assert _ratio(1, 4) == "0.250000000000"
    print("self-test: passed")


def _print_summary(rows: list[dict[str, Any]], output_path: Path) -> None:
    selector_rows = [row for row in rows if row["selector_status"] == "found"]
    annotated_rows = [
        row for row in rows if row["abstention_source_method"] == "annotated_summary_metrics"
    ]
    derived_rows = [
        row
        for row in rows
        if row["abstention_source_method"] == "derived_reference_sources_without_committed_mapping"
    ]
    quality_rows = [row for row in rows if row["quality_status"] == "computed"]
    consistent_rows = [
        row for row in annotated_rows if row["selector_counter_consistent"] == "true"
    ]
    print(f"Wrote {_relative(output_path)}")
    print(f"logs represented: {len(rows)}")
    print(f"logs with final selector: {len(selector_rows)}")
    print(f"logs without final selector: {len(rows) - len(selector_rows)}")
    print(f"committed quality computed: {len(quality_rows)}")
    print(f"selector runs with saved per-source flags: {len(annotated_rows)}")
    print(f"selector runs using labelled derivation: {len(derived_rows)}")
    print("annotation/log counter checks passed: " f"{len(consistent_rows)}/{len(annotated_rows)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exp-root",
        type=Path,
        default=DEFAULT_EXP_ROOT,
        help="experiment tree to scan (default: exp/test)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="CSV path to write",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="parallel workers for large selector annotation CSVs (default: up to 4)",
    )
    parser.add_argument(
        "--scan-selector-annotations",
        action="store_true",
        help=(
            "scan saved per-pair candidate records for exact abstained-source flags; "
            "these artifacts currently total several GB"
        ),
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run parser unit checks and exit",
    )
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    exp_root = _resolve_repo_path(args.exp_root)
    output_path = _resolve_repo_path(args.output)
    rows = extract(
        exp_root,
        output_path,
        workers=args.workers,
        scan_annotations=args.scan_selector_annotations,
    )
    _print_summary(rows, output_path)


if __name__ == "__main__":
    main()
