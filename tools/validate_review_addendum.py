#!/usr/bin/env python3
"""Validate the WP5/WP6 review-response addendum artifacts.

The validator is intentionally read-only apart from its validation ledger.  A normal invocation
records missing in-flight WP6 artifacts and exits successfully; ``--strict`` turns every missing
or failed acceptance check into a non-zero exit status.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    import yaml
except ImportError:  # pragma: no cover - the repository already depends on PyYAML.
    yaml = None


CHANNELS = ("lex", "hier", "sim", "diff", "attr")
STRUCTURAL_CHANNELS = ("hier", "sim", "diff", "attr")
ALLOWED_RELIABILITY_STATUSES = {"ok", "constant_quality", "fully_inactive"}
BLOCKING_STATUSES = {"fail", "missing"}

EXPECTED_BOOTSTRAP_RESAMPLES = 2_000
EXPECTED_SEED = 42
ACTIVE_SIGMA_THRESHOLD = 1e-6
IDENTITY_TOLERANCE = 1e-5
STRUCT_ACTIVE_THRESHOLD = 1e-8

EXPECTED_OMIM_CANDIDATE_HASH = "faeda0c74a63ed216480ea142af2b2b77c6257f452c31bb920f5ea05a338a86d"
EXPECTED_OMIM_REFERENCE_HASH = "26e0e12d94b58e152fd61ccc0326d98acceb6980659dfd58a670393d435279d0"
EXPECTED_FMA_CANDIDATE_HASH = "3f1c34b4adbdb7da5038f0772bf4bb0d8e730e5cac87096977af2e34f8acf8f1"
EXPECTED_FMA_REFERENCE_HASH = "0c3f601da5b5910774b7f24e77acf81d1a47bf73e1e78666118df60c9f1acfbb"
EXPECTED_FMA_SOURCES = 300
EXPECTED_FMA_ANCHORS = 324
EXPECTED_FMA_RAW_PAIRS = 32_724
EXPECTED_FMA_FIRST_SOURCE = "http://snomed.info/id/10013000"
EXPECTED_FMA_LAST_SOURCE = "http://snomed.info/id/1425000"

CANONICAL_DUMP_COLUMNS = (
    "src",
    "tgt",
    "is_correct",
    "s_lex",
    "s_hier",
    "s_sim",
    "s_diff",
    "s_attr",
    "q_lex",
    "q_hier",
    "q_sim",
    "q_diff",
    "q_attr",
    "sigma_lex",
    "sigma_hier",
    "sigma_sim",
    "sigma_diff",
    "sigma_attr",
    "sigma_struct",
    "omega_hier",
    "omega_sim",
    "omega_diff",
    "omega_attr",
    "w_c",
    "S_struct",
    "Q_struct",
    "S_lctx",
    "U",
    "p_llm",
    "llm_invoked",
    "llm_gated",
    "S_pair",
    "S_final",
    "src_label_text",
    "tgt_label_text",
)
DUMP_TEXT_COLUMNS = {"src", "tgt", "src_label_text", "tgt_label_text"}
DUMP_BOOLEAN_COLUMNS = {"llm_invoked", "llm_gated"}
DUMP_NUMERIC_COLUMNS = set(CANONICAL_DUMP_COLUMNS) - DUMP_TEXT_COLUMNS - DUMP_BOOLEAN_COLUMNS

E5_EXPECTED_ACTIVITY = {
    "lex": (30_404, 30_404, 0),
    "hier": (30_404, 5_248, 25_156),
    "sim": (30_404, 0, 30_404),
    "diff": (30_404, 0, 30_404),
    "attr": (30_404, 30_404, 0),
}
E5_EXPECTED_STATUSES = {
    "lex": "constant_quality",
    "hier": "ok",
    "sim": "fully_inactive",
    "diff": "fully_inactive",
    "attr": "ok",
}

METRICS = ("MRR", "Hits@1", "Hits@5", "Hits@10")

FATAL_LOG_PATTERNS = (
    re.compile(r"Traceback \(most recent call last\)", re.IGNORECASE),
    re.compile(r"\bCUDA out of memory\b", re.IGNORECASE),
    re.compile(r"\bAlignment failed\b", re.IGNORECASE),
    re.compile(r"\brun failed\b", re.IGNORECASE),
    re.compile(r"\bsegmentation fault\b", re.IGNORECASE),
    re.compile(r"\b(?:AssertionError|RuntimeError|MemoryError):", re.IGNORECASE),
    re.compile(r"slurmstepd:\s*error:.*\bKilled\b", re.IGNORECASE),
    re.compile(r"(?m)^\s*(?:srun: error:.*|Killed)\s*$", re.IGNORECASE),
)

RELIABILITY_KINDS = {
    "calibration": {
        "correlation": "spearman_bin_brier",
        "n": "spearman_n_bins",
        "binned": True,
    },
    "discrimination": {
        "correlation": "spearman_quality_auc",
        "n": "spearman_n_sources",
        "binned": False,
    },
    "pick_rate": {
        "correlation": "spearman_quality_pick",
        "n": "spearman_n_sources",
        "binned": True,
    },
}


@dataclass
class Validation:
    wp: str
    area: str
    check: str
    status: str
    detail: str = ""
    path: str = ""

    def as_row(self) -> dict[str, str]:
        return {
            "wp": self.wp,
            "area": self.area,
            "check": self.check,
            "status": self.status,
            "detail": self.detail,
            "path": self.path,
        }


@dataclass
class Ledger:
    root: Path
    checks: list[Validation] = field(default_factory=list)

    def add(
        self,
        wp: str,
        area: str,
        check: str,
        status: str,
        detail: str = "",
        path: Path | str | None = None,
    ) -> Validation:
        if status not in {"pass", "fail", "missing", "warn"}:
            raise ValueError(f"unsupported validation status: {status!r}")
        record = Validation(
            wp=wp,
            area=area,
            check=check,
            status=status,
            detail=detail,
            path=_display_path(self.root, path) if path else "",
        )
        self.checks.append(record)
        return record

    def artifact(
        self,
        wp: str,
        area: str,
        path: Path,
        *,
        required: bool = True,
        check: str | None = None,
    ) -> bool:
        label = check or f"artifact:{path.name}"
        if not path.exists():
            self.add(
                wp,
                area,
                label,
                "missing" if required else "pass",
                "required artifact is absent" if required else "optional artifact is absent",
                path,
            )
            return False
        if not path.is_file():
            self.add(wp, area, label, "fail", "path is not a regular file", path)
            return False
        try:
            size = path.stat().st_size
        except OSError as exc:
            self.add(wp, area, label, "fail", f"cannot stat artifact: {exc}", path)
            return False
        if size == 0:
            self.add(wp, area, label, "fail", "artifact is empty", path)
            return False
        self.add(wp, area, label, "pass", f"{size} bytes", path)
        return True

    @property
    def blocking(self) -> list[Validation]:
        return [check for check in self.checks if check.status in BLOCKING_STATUSES]


@dataclass
class DumpStats:
    rows: int
    sources: set[str]
    sources_with_correct: set[str]
    active_counts: dict[str, int]
    sigma_inert_counts: dict[str, int]
    omega_inert_counts: dict[str, int]
    q_first: dict[str, float | None]
    q_constant: dict[str, bool]
    invalid_rows: int
    invalid_numeric: dict[str, int]
    omega_max_error: float
    omega_mismatches: int
    pair_max_error: float
    pair_mismatches: int
    pair_checked: int

    @property
    def expected_statuses(self) -> dict[str, str]:
        statuses: dict[str, str] = {}
        for channel in CHANNELS:
            if self.active_counts[channel] == 0:
                statuses[channel] = "fully_inactive"
            elif self.q_constant[channel]:
                statuses[channel] = "constant_quality"
            else:
                statuses[channel] = "ok"
        return statuses

    @property
    def activity(self) -> dict[str, tuple[int, int, int]]:
        return {
            channel: (
                self.rows,
                self.active_counts[channel],
                self.rows - self.active_counts[channel],
            )
            for channel in CHANNELS
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repository root; all default input and output paths are relative to it.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Validation ledger path. Defaults to "
            "exp/test/review_response/analysis/addendum_validation.csv under --repo-root."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero after writing the ledger if any required check is missing or failed.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run small parser/hash unit checks and exit without inspecting experiment artifacts.",
    )
    return parser.parse_args()


def _display_path(root: Path, path: Path | str | None) -> str:
    if path is None:
        return ""
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(root.resolve()))
    except (OSError, ValueError):
        return str(candidate)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _float(value: Any) -> float:
    number = float(str(value).strip())
    if not math.isfinite(number):
        raise ValueError(f"non-finite number {value!r}")
    return number


def _int(value: Any) -> int:
    number = _float(value)
    integer = int(number)
    if number != integer:
        raise ValueError(f"not an integer: {value!r}")
    return integer


def _binary(value: Any) -> int:
    text = str(value).strip().lower()
    if text in {"true", "yes", "y"}:
        return 1
    if text in {"false", "no", "n"}:
        return 0
    number = _int(value)
    if number not in {0, 1}:
        raise ValueError(f"not binary: {value!r}")
    return number


def _pipe_floats(value: str) -> list[float]:
    text = str(value).strip()
    if not text:
        return []
    return [_float(part) for part in text.split("|")]


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("CSV contains duplicate column names")
        rows = list(reader)
    return list(reader.fieldnames), rows


def _write_ledger(path: Path, checks: Iterable[Validation]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("wp", "area", "check", "status", "detail", "path"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(check.as_row() for check in checks)
    temporary.replace(path)


def _validate_plot(ledger: Ledger, wp: str, path: Path) -> None:
    if not ledger.artifact(wp, "plots", path):
        return
    signature = path.read_bytes()[:8]
    if path.suffix.lower() == ".png":
        valid = signature == b"\x89PNG\r\n\x1a\n"
        expected = "PNG signature"
    elif path.suffix.lower() == ".pdf":
        valid = signature.startswith(b"%PDF-")
        expected = "PDF signature"
    else:  # pragma: no cover - callers only pass PNG/PDF.
        valid = False
        expected = "known plot format"
    ledger.add(
        wp,
        "plots",
        f"format:{path.name}",
        "pass" if valid else "fail",
        expected if valid else f"artifact does not have a valid {expected}",
        path,
    )


def _required_reliability_columns(kind: str) -> set[str]:
    common = {
        "record_type",
        "channel",
        "status",
        "status_reason",
        "total_rows",
        "active_rows",
        "inactive_rows",
        "inactive_fraction",
        "spearman_ci_low",
        "spearman_ci_high",
        "bootstrap_resamples",
        "bootstrap_valid",
        "seed",
        "correlation_status",
    }
    specific = {
        "calibration": {
            "n_bins",
            "bin",
            "bin_edges",
            "n",
            "mean_q_k",
            "positive_rate",
            "mean_s_k",
            "brier",
            "mae",
            "spearman_bin_brier",
            "spearman_n_bins",
        },
        "discrimination": {
            "total_sources",
            "eligible_sources",
            "excluded_sources_without_scored_correct",
            "sources_all_inactive",
            "sources_without_auc",
            "sources_with_auc",
            "src",
            "source_status",
            "active_candidates",
            "inactive_candidates",
            "auc",
            "spearman_quality_auc",
            "spearman_n_sources",
        },
        "pick_rate": {
            "total_sources",
            "eligible_sources",
            "excluded_sources_without_scored_correct",
            "sources_all_inactive",
            "sources_in_pick_rate",
            "sources_tied_at_active_argmax",
            "n_bins",
            "bin",
            "bin_edges",
            "n",
            "mean_q_k",
            "pick_rate",
            "src",
            "source_status",
            "active_candidates",
            "inactive_candidates",
            "picked_correct",
            "tied_at_argmax",
            "spearman_quality_pick",
            "spearman_n_sources",
        },
    }
    return common | specific[kind]


def _validate_reliability_table(
    ledger: Ledger,
    *,
    wp: str,
    prefix: str,
    kind: str,
    path: Path,
    expected_statuses: Mapping[str, str] | None,
    expected_activity: Mapping[str, tuple[int, int, int]] | None,
    expected_sources: int | None,
    expected_eligible_sources: int | None,
) -> dict[str, str] | None:
    area = f"{prefix}_{kind}"
    if not ledger.artifact(wp, area, path):
        return None
    try:
        header, rows = _read_csv(path)
    except Exception as exc:
        ledger.add(wp, area, "csv:read", "fail", f"{type(exc).__name__}: {exc}", path)
        return None

    missing_columns = sorted(_required_reliability_columns(kind) - set(header))
    ledger.add(
        wp,
        area,
        "csv:schema",
        "pass" if not missing_columns else "fail",
        (
            f"{len(header)} columns; required schema present"
            if not missing_columns
            else "missing columns: " + ", ".join(missing_columns)
        ),
        path,
    )
    if missing_columns:
        return None

    summaries = [row for row in rows if row.get("record_type", "").strip() == "summary"]
    by_channel: dict[str, list[dict[str, str]]] = {}
    for row in summaries:
        by_channel.setdefault(row.get("channel", "").strip(), []).append(row)
    exact_channels = set(by_channel) == set(CHANNELS) and all(
        len(by_channel[channel]) == 1 for channel in CHANNELS
    )
    ledger.add(
        wp,
        area,
        "summary:five_channels",
        "pass" if exact_channels else "fail",
        (
            "exactly one summary row for lex, hier, sim, diff, attr"
            if exact_channels
            else f"summary row counts={{{', '.join(f'{k!r}: {len(v)}' for k, v in sorted(by_channel.items()))}}}"
        ),
        path,
    )
    if not exact_channels:
        return None

    observed_statuses: dict[str, str] = {}
    for channel in CHANNELS:
        row = by_channel[channel][0]
        status = row["status"].strip()
        observed_statuses[channel] = status
        expected_status = expected_statuses.get(channel) if expected_statuses else None
        status_ok = status in ALLOWED_RELIABILITY_STATUSES and (
            expected_status is None or status == expected_status
        )
        status_detail = f"status={status!r}"
        if expected_status is not None:
            status_detail += f"; expected={expected_status!r}"
        if not row["status_reason"].strip():
            status_ok = False
            status_detail += "; status_reason is blank"
        ledger.add(
            wp,
            area,
            f"status:{channel}",
            "pass" if status_ok else "fail",
            status_detail,
            path,
        )

        try:
            total = _int(row["total_rows"])
            active = _int(row["active_rows"])
            inactive = _int(row["inactive_rows"])
            inactive_fraction = _float(row["inactive_fraction"])
            count_ok = (
                total > 0
                and active >= 0
                and inactive >= 0
                and active + inactive == total
                and math.isclose(
                    inactive_fraction,
                    inactive / total,
                    rel_tol=0.0,
                    abs_tol=1e-8,
                )
            )
            expected_counts = expected_activity.get(channel) if expected_activity else None
            if expected_counts is not None:
                count_ok = count_ok and (total, active, inactive) == expected_counts
            detail = (
                f"total={total}; active={active}; inactive={inactive}; "
                f"inactive_fraction={inactive_fraction:.10g}"
            )
            if expected_counts is not None:
                detail += f"; expected={expected_counts}"
        except Exception as exc:
            count_ok = False
            detail = f"{type(exc).__name__}: {exc}"
            total = active = inactive = 0
        ledger.add(
            wp,
            area,
            f"inactive_counts:{channel}",
            "pass" if count_ok else "fail",
            detail,
            path,
        )

        try:
            resamples = _int(row["bootstrap_resamples"])
            seed = _int(row["seed"])
            bootstrap_config_ok = (
                resamples == EXPECTED_BOOTSTRAP_RESAMPLES and seed == EXPECTED_SEED
            )
            bootstrap_detail = f"resamples={resamples}; seed={seed}"
        except Exception as exc:
            bootstrap_config_ok = False
            bootstrap_detail = f"{type(exc).__name__}: {exc}"
        ledger.add(
            wp,
            area,
            f"bootstrap_config:{channel}",
            "pass" if bootstrap_config_ok else "fail",
            bootstrap_detail,
            path,
        )

        correlation_column = str(RELIABILITY_KINDS[kind]["correlation"])
        n_column = str(RELIABILITY_KINDS[kind]["n"])
        correlation_text = row[correlation_column].strip()
        try:
            correlation_n = _int(row[n_column])
            correlation_ok = correlation_n >= 0
            correlation_detail = f"n={correlation_n}"
            if correlation_text:
                correlation = _float(correlation_text)
                low = _float(row["spearman_ci_low"])
                high = _float(row["spearman_ci_high"])
                bootstrap_valid = _int(row["bootstrap_valid"])
                correlation_ok = (
                    correlation_ok
                    and correlation_n >= 2
                    and -1.0 <= correlation <= 1.0
                    and -1.0 <= low <= high <= 1.0
                    and 0 < bootstrap_valid <= EXPECTED_BOOTSTRAP_RESAMPLES
                )
                correlation_detail += (
                    f"; rho={correlation:.10g}; CI=[{low:.10g},{high:.10g}]; "
                    f"bootstrap_valid={bootstrap_valid}"
                )
            else:
                low_blank = not row["spearman_ci_low"].strip()
                high_blank = not row["spearman_ci_high"].strip()
                correlation_ok = correlation_ok and low_blank and high_blank
                correlation_detail += "; correlation is undefined and no CI is reported"
                if status == "ok" and row["correlation_status"].strip() == "ok":
                    correlation_ok = False
                    correlation_detail += "; correlation_status incorrectly says ok"
        except Exception as exc:
            correlation_ok = False
            correlation_detail = f"{type(exc).__name__}: {exc}"
        ledger.add(
            wp,
            area,
            f"correlation_n_ci:{channel}",
            "pass" if correlation_ok else "fail",
            correlation_detail,
            path,
        )

        if expected_sources is not None and kind in {"discrimination", "pick_rate"}:
            try:
                total_sources = _int(row["total_sources"])
                eligible_sources = _int(row["eligible_sources"])
                excluded_sources = _int(row["excluded_sources_without_scored_correct"])
                source_ok = (
                    total_sources == expected_sources
                    and eligible_sources + excluded_sources == total_sources
                    and (
                        expected_eligible_sources is None
                        or eligible_sources == expected_eligible_sources
                    )
                )
                source_detail = (
                    f"total={total_sources}; eligible={eligible_sources}; "
                    f"excluded_without_scored_correct={excluded_sources}"
                )
                if expected_eligible_sources is not None:
                    source_detail += f"; expected_eligible={expected_eligible_sources}"
            except Exception as exc:
                source_ok = False
                source_detail = f"{type(exc).__name__}: {exc}"
            ledger.add(
                wp,
                area,
                f"source_scope:{channel}",
                "pass" if source_ok else "fail",
                source_detail,
                path,
            )

        if kind in {"calibration", "pick_rate"}:
            channel_bins = [
                item
                for item in rows
                if item.get("record_type", "").strip() == "bin"
                and item.get("channel", "").strip() == channel
            ]
            try:
                n_bins = _int(row["n_bins"])
                edges = _pipe_floats(row["bin_edges"])
                if status == "ok":
                    bins_ok = (
                        1 <= n_bins <= 10
                        and len(channel_bins) == n_bins
                        and len(edges) == n_bins + 1
                        and all(left < right for left, right in zip(edges, edges[1:]))
                    )
                else:
                    bins_ok = n_bins == 0 and not channel_bins and not edges
                n_sum = 0
                seen_bins: set[int] = set()
                for bin_row in channel_bins:
                    bin_number = _int(bin_row["bin"])
                    bin_n = _int(bin_row["n"])
                    if bin_n <= 0:
                        bins_ok = False
                    n_sum += bin_n
                    seen_bins.add(bin_number)
                    row_edges = _pipe_floats(bin_row["bin_edges"])
                    if row_edges != edges:
                        bins_ok = False
                    if kind == "calibration":
                        for numeric_column in (
                            "mean_q_k",
                            "positive_rate",
                            "mean_s_k",
                            "brier",
                            "mae",
                        ):
                            _float(bin_row[numeric_column])
                        if not 0.0 <= _float(bin_row["positive_rate"]) <= 1.0:
                            bins_ok = False
                        if _float(bin_row["brier"]) < 0.0 or _float(bin_row["mae"]) < 0.0:
                            bins_ok = False
                    else:
                        _float(bin_row["mean_q_k"])
                        if not 0.0 <= _float(bin_row["pick_rate"]) <= 1.0:
                            bins_ok = False
                if channel_bins and seen_bins != set(range(1, n_bins + 1)):
                    bins_ok = False
                expected_bin_population = (
                    active if kind == "calibration" else _int(row["sources_in_pick_rate"])
                )
                if channel_bins and n_sum != expected_bin_population:
                    bins_ok = False
                bins_detail = (
                    f"n_bins={n_bins}; edge_count={len(edges)}; "
                    f"bin_rows={len(channel_bins)}; bin_n_sum={n_sum}"
                )
            except Exception as exc:
                bins_ok = False
                bins_detail = f"{type(exc).__name__}: {exc}"
            ledger.add(
                wp,
                area,
                f"visible_bins_edges:{channel}",
                "pass" if bins_ok else "fail",
                bins_detail,
                path,
            )

        if kind in {"discrimination", "pick_rate"}:
            source_rows = [
                item
                for item in rows
                if item.get("record_type", "").strip() == "source"
                and item.get("channel", "").strip() == channel
            ]
            source_rows_ok = True
            try:
                eligible = _int(row["eligible_sources"])
                source_rows_ok = len(source_rows) == eligible
                for source_row in source_rows:
                    active_candidates = _int(source_row["active_candidates"])
                    inactive_candidates = _int(source_row["inactive_candidates"])
                    if active_candidates < 0 or inactive_candidates < 0:
                        source_rows_ok = False
                    if kind == "pick_rate" and source_row["tied_at_argmax"].strip():
                        tied = _binary(source_row["tied_at_argmax"])
                        picked_text = source_row["picked_correct"].strip()
                        if tied and picked_text and _binary(picked_text) != 0:
                            source_rows_ok = False
                source_rows_detail = f"source_rows={len(source_rows)}; eligible_sources={eligible}"
            except Exception as exc:
                source_rows_ok = False
                source_rows_detail = f"{type(exc).__name__}: {exc}"
            ledger.add(
                wp,
                area,
                f"active_only_source_rows:{channel}",
                "pass" if source_rows_ok else "fail",
                source_rows_detail,
                path,
            )

    return observed_statuses


def _validate_reliability_bundle(
    ledger: Ledger,
    *,
    wp: str,
    prefix: str,
    analysis_dir: Path,
    expected_statuses: Mapping[str, str] | None,
    expected_activity: Mapping[str, tuple[int, int, int]] | None,
    expected_sources: int | None,
    expected_eligible_sources: int | None,
) -> None:
    observed: dict[str, dict[str, str] | None] = {}
    for kind in RELIABILITY_KINDS:
        csv_path = analysis_dir / f"{prefix}_{kind}.csv"
        observed[kind] = _validate_reliability_table(
            ledger,
            wp=wp,
            prefix=prefix,
            kind=kind,
            path=csv_path,
            expected_statuses=expected_statuses,
            expected_activity=expected_activity,
            expected_sources=expected_sources,
            expected_eligible_sources=expected_eligible_sources,
        )
        for suffix in (".png", ".pdf"):
            _validate_plot(ledger, wp, analysis_dir / f"{prefix}_{kind}{suffix}")

    available = [statuses for statuses in observed.values() if statuses is not None]
    if not available:
        ledger.add(
            wp,
            "reliability",
            f"{prefix}:cross_table_statuses",
            "missing",
            "no readable reliability summary tables",
            analysis_dir,
        )
    else:
        consistent = len(available) == len(RELIABILITY_KINDS) and all(
            statuses == available[0] for statuses in available[1:]
        )
        ledger.add(
            wp,
            "reliability",
            f"{prefix}:cross_table_statuses",
            "pass" if consistent else "fail",
            (
                "all three tables carry identical five-channel statuses"
                if consistent
                else f"observed={json.dumps(observed, sort_keys=True)}"
            ),
            analysis_dir,
        )


def _validate_wp5_results(ledger: Ledger, path: Path) -> None:
    if not ledger.artifact("WP5", "results", path):
        return
    text = path.read_text(encoding="utf-8", errors="replace")
    lower = text.lower()
    verdicts: dict[str, list[str]] = {}
    for channel in CHANNELS:
        pattern = re.compile(
            rf"\*\*\s*{re.escape(channel)}\s*:\s*" r"(supported|not supported|untestable)\b",
            re.IGNORECASE,
        )
        verdicts[channel] = [match.lower() for match in pattern.findall(text)]
    verdict_ok = all(len(verdicts[channel]) == 1 for channel in CHANNELS)
    for channel, expected_status in E5_EXPECTED_STATUSES.items():
        if (
            expected_status != "ok"
            and verdicts.get(channel)
            and verdicts[channel][0] != "untestable"
        ):
            verdict_ok = False
    ledger.add(
        "WP5",
        "results",
        "per_channel_verdicts",
        "pass" if verdict_ok else "fail",
        json.dumps(verdicts, sort_keys=True),
        path,
    )
    method_ok = (
        re.search(r"\b2[,\s]?000\b", text) is not None
        and re.search(r"\bseed\s+42\b", lower) is not None
        and "sigma_k > 1e-06" in lower
        and "204/300" in lower
    )
    ledger.add(
        "WP5",
        "results",
        "method_and_scope",
        "pass" if method_ok else "fail",
        "reports active-only rule, 2,000/seed 42 bootstrap, and 204/300 source restriction",
        path,
    )
    difference_ok = (
        "difference from wp1" in lower
        and re.search(r"threshold[-\s]agreement", lower) is not None
        and all(channel in lower for channel in CHANNELS)
    )
    ledger.add(
        "WP5",
        "results",
        "wp1_difference_paragraph",
        "pass" if difference_ok else "fail",
        "explicit WP1 threshold-agreement comparison covering all channels",
        path,
    )


def _manifest_candidates(analysis_dir: Path, prefix: str) -> list[Path]:
    candidates = []
    if not analysis_dir.is_dir():
        return candidates
    for path in sorted(analysis_dir.iterdir(), key=lambda item: item.name):
        if not path.is_file():
            continue
        name = path.name.lower()
        if path.suffix.lower() == ".sha256" or (
            ("hash" in name or "manifest" in name)
            and ("e5" in name or "wp5" in name or "reliability" in name)
            and path.suffix.lower() in {".csv", ".json", ".txt", ".sha256"}
        ):
            if prefix.lower() in name:
                candidates.append(path)
    return candidates


def _validate_reliability_manifests(
    ledger: Ledger,
    analysis_dir: Path,
    *,
    wp: str,
    prefix: str,
) -> None:
    manifests = _manifest_candidates(analysis_dir, prefix)
    if not manifests:
        ledger.add(
            wp,
            "determinism",
            "hash_manifest_if_present",
            "pass",
            f"no optional {prefix} deterministic hash manifest is present",
            analysis_dir,
        )
        return
    required = [analysis_dir / f"{prefix}_{kind}.csv" for kind in RELIABILITY_KINDS]
    for manifest in manifests:
        try:
            text = manifest.read_text(encoding="utf-8", errors="replace")
            covered = []
            mismatches = []
            nondeterministic = []
            for artifact in required:
                if not artifact.is_file():
                    continue
                actual = _sha256(artifact)
                relevant_lines = [line for line in text.splitlines() if artifact.name in line]
                if not relevant_lines:
                    continue
                covered.append(artifact.name)
                line_hashes = {
                    match.lower()
                    for line in relevant_lines
                    for match in re.findall(r"\b[0-9a-fA-F]{64}\b", line)
                }
                # Pretty-printed JSON may put the filename and hash on adjacent lines.
                all_hashes = {match.lower() for match in re.findall(r"\b[0-9a-fA-F]{64}\b", text)}
                candidate_hashes = line_hashes or all_hashes
                if actual not in candidate_hashes:
                    mismatches.append(f"{artifact.name}:{actual}")
                if len(line_hashes) > 1:
                    nondeterministic.append(f"{artifact.name}:{','.join(sorted(line_hashes))}")
            ok = (
                set(covered) == {path.name for path in required if path.is_file()}
                and not mismatches
                and not nondeterministic
            )
            detail = (
                f"covered={covered}; mismatches={mismatches}; "
                f"nonidentical_recorded_hashes={nondeterministic}"
            )
        except Exception as exc:
            ok = False
            detail = f"{type(exc).__name__}: {exc}"
        ledger.add(
            wp,
            "determinism",
            f"hash_manifest:{manifest.name}",
            "pass" if ok else "fail",
            detail,
            manifest,
        )


def _validate_wp5(ledger: Ledger, analysis_dir: Path) -> None:
    _validate_reliability_bundle(
        ledger,
        wp="WP5",
        prefix="e5",
        analysis_dir=analysis_dir,
        expected_statuses=E5_EXPECTED_STATUSES,
        expected_activity=E5_EXPECTED_ACTIVITY,
        expected_sources=300,
        expected_eligible_sources=204,
    )
    _validate_wp5_results(ledger, analysis_dir / "WP5_RESULTS.md")
    _validate_reliability_manifests(ledger, analysis_dir, wp="WP5", prefix="e5")


def _validate_subset(ledger: Ledger, root: Path) -> None:
    dataset_dir = root / "data/snomed-fma.body"
    candidates_path = dataset_dir / "review300.test.cands.tsv"
    reference_path = dataset_dir / "review300.test.tsv"

    paths_and_hashes = (
        (candidates_path, EXPECTED_FMA_CANDIDATE_HASH),
        (reference_path, EXPECTED_FMA_REFERENCE_HASH),
    )
    existing = True
    for path, expected_hash in paths_and_hashes:
        if not ledger.artifact("WP6", "subset", path):
            existing = False
            continue
        try:
            actual_hash = _sha256(path)
            ok = actual_hash == expected_hash
            detail = f"sha256={actual_hash}; expected={expected_hash}"
        except Exception as exc:
            ok = False
            detail = f"{type(exc).__name__}: {exc}"
        ledger.add(
            "WP6",
            "subset",
            f"sha256:{path.name}",
            "pass" if ok else "fail",
            detail,
            path,
        )
    if not existing:
        return

    try:
        with candidates_path.open("r", encoding="utf-8", newline="") as handle:
            candidate_reader = csv.DictReader(handle, delimiter="\t")
            if candidate_reader.fieldnames != ["SrcEntity", "TgtEntity", "TgtCandidates"]:
                raise ValueError(f"unexpected candidate header: {candidate_reader.fieldnames!r}")
            candidate_rows = list(candidate_reader)
        with reference_path.open("r", encoding="utf-8", newline="") as handle:
            reference_reader = csv.DictReader(handle, delimiter="\t")
            if reference_reader.fieldnames != ["SrcEntity", "TgtEntity", "Score"]:
                raise ValueError(f"unexpected reference header: {reference_reader.fieldnames!r}")
            reference_rows = list(reference_reader)

        unique_sources = sorted({row["SrcEntity"] for row in candidate_rows})
        reference_sources = {row["SrcEntity"] for row in reference_rows}
        candidate_pairs = 0
        contained = 0
        candidate_anchors: list[tuple[str, str]] = []
        for row_number, row in enumerate(candidate_rows, start=2):
            try:
                choices = ast.literal_eval(row["TgtCandidates"])
            except (SyntaxError, ValueError) as exc:
                raise ValueError(f"row {row_number}: invalid TgtCandidates literal: {exc}") from exc
            if not isinstance(choices, (tuple, list)):
                raise ValueError(f"row {row_number}: TgtCandidates is not a tuple/list")
            if not choices or not all(isinstance(choice, str) for choice in choices):
                raise ValueError(
                    f"row {row_number}: candidate list is empty or contains non-strings"
                )
            candidate_pairs += len(choices)
            contained += int(row["TgtEntity"] in choices)
            candidate_anchors.append((row["SrcEntity"], row["TgtEntity"]))
        reference_anchors = [(row["SrcEntity"], row["TgtEntity"]) for row in reference_rows]
        invariants = {
            "unique_sources": len(unique_sources),
            "candidate_rows": len(candidate_rows),
            "reference_rows": len(reference_rows),
            "candidate_pairs": candidate_pairs,
            "first_source": unique_sources[0] if unique_sources else "",
            "last_source": unique_sources[-1] if unique_sources else "",
            "reference_in_candidate_list": contained,
            "source_sets_equal": set(unique_sources) == reference_sources,
            "anchor_multisets_equal": sorted(candidate_anchors) == sorted(reference_anchors),
        }
        expected = {
            "unique_sources": EXPECTED_FMA_SOURCES,
            "candidate_rows": EXPECTED_FMA_ANCHORS,
            "reference_rows": EXPECTED_FMA_ANCHORS,
            "candidate_pairs": EXPECTED_FMA_RAW_PAIRS,
            "first_source": EXPECTED_FMA_FIRST_SOURCE,
            "last_source": EXPECTED_FMA_LAST_SOURCE,
            "reference_in_candidate_list": EXPECTED_FMA_ANCHORS,
            "source_sets_equal": True,
            "anchor_multisets_equal": True,
        }
        ok = invariants == expected
        detail = (
            f"actual={json.dumps(invariants, sort_keys=True)}; "
            f"expected={json.dumps(expected, sort_keys=True)}"
        )
    except Exception as exc:
        ok = False
        detail = f"{type(exc).__name__}: {exc}"
    ledger.add(
        "WP6",
        "subset",
        "static_invariants",
        "pass" if ok else "fail",
        detail,
        candidates_path,
    )

    subset_record = root / "specs/review-response/SUBSET.md"
    if ledger.artifact("WP6", "subset", subset_record):
        text = subset_record.read_text(encoding="utf-8", errors="replace")
        record_ok = all(
            value in text
            for value in (
                "data/snomed-fma.body/review300.test.cands.tsv",
                "data/snomed-fma.body/review300.test.tsv",
                EXPECTED_FMA_CANDIDATE_HASH,
                EXPECTED_FMA_REFERENCE_HASH,
            )
        )
        ledger.add(
            "WP6",
            "subset",
            "subset_record_hashes",
            "pass" if record_ok else "fail",
            "SUBSET.md records both FMA artifact paths and expected SHA-256 hashes",
            subset_record,
        )


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is not installed")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("top-level YAML value is not a mapping")
    return payload


def _expected_e6_run_yaml() -> dict[str, Any]:
    return {
        "dataset": {
            "data_dir": "data/snomed-fma.body",
            "source": "snomed.body.owl",
            "target": "fma.body.owl",
            "full_reference": "test.tsv",
            "candidates": "review300.test.cands.tsv",
        },
        "job": {
            "name": "e6_fma_dump",
            "output_dir": "exp/test/review_response/e6_fma_dump/",
            "config_file": "exp/test/review_response/e6_fma_dump/config.yaml",
            "memory": "60G",
            "device": 0,
            "save_logs": True,
            "run_eval": True,
        },
    }


def _validate_e6_configs(ledger: Ledger, root: Path, run_dir: Path) -> None:
    e1_config = root / "exp/test/review_response/e1_dump/config.yaml"
    e6_config = run_dir / "config.yaml"
    e1_exists = ledger.artifact("WP6", "config", e1_config, check="baseline:e1_config.yaml")
    e6_exists = ledger.artifact("WP6", "config", e6_config)
    if e1_exists and e6_exists:
        try:
            e1_bytes = e1_config.read_bytes()
            e6_bytes = e6_config.read_bytes()
            identical = e1_bytes == e6_bytes
            detail = (
                f"e1_sha256={_sha256(e1_config)}; e6_sha256={_sha256(e6_config)}; "
                f"bytes={len(e6_bytes)}"
            )
        except Exception as exc:
            identical = False
            detail = f"{type(exc).__name__}: {exc}"
        ledger.add(
            "WP6",
            "config",
            "byte_identical_to_e1",
            "pass" if identical else "fail",
            detail,
            e6_config,
        )

    run_yaml_path = run_dir / "snomed-fma-body.yaml"
    if ledger.artifact("WP6", "config", run_yaml_path):
        try:
            actual = _load_yaml_mapping(run_yaml_path)
            expected = _expected_e6_run_yaml()
            exact = actual == expected
            detail = (
                "run YAML exactly matches WP6 D2"
                if exact
                else (
                    f"actual={json.dumps(actual, sort_keys=True)}; "
                    f"expected={json.dumps(expected, sort_keys=True)}"
                )
            )
        except Exception as exc:
            exact = False
            detail = f"{type(exc).__name__}: {exc}"
        ledger.add(
            "WP6",
            "config",
            "run_yaml:exact",
            "pass" if exact else "fail",
            detail,
            run_yaml_path,
        )


def _find_fatal_marker(text: str) -> str:
    for pattern in FATAL_LOG_PATTERNS:
        match = pattern.search(text)
        if match:
            return " ".join(match.group(0).split())
    return ""


def _validate_logs(ledger: Ledger, run_dir: Path) -> str:
    log_path = run_dir / "exact.log"
    if not log_path.is_file() or not log_path.stat().st_size:
        return ""
    texts: list[tuple[Path, str]] = [
        (log_path, log_path.read_text(encoding="utf-8", errors="replace"))
    ]
    for pattern in ("slurm_*.out", "slurm_*.err", "slurm-*.out", "slurm-*.err"):
        for path in sorted(run_dir.glob(pattern)):
            if path.is_file() and path.stat().st_size:
                texts.append((path, path.read_text(encoding="utf-8", errors="replace")))
    markers = [
        f"{path.name}: {_find_fatal_marker(text)}"
        for path, text in texts
        if _find_fatal_marker(text)
    ]
    ledger.add(
        "WP6",
        "run",
        "logs:no_fatal_markers",
        "pass" if not markers else "fail",
        "no recognized fatal marker" if not markers else "; ".join(markers),
        log_path,
    )
    return texts[0][1]


def _read_metrics(path: Path) -> dict[str, float]:
    header, rows = _read_csv(path)
    normalized = {column.strip().lower(): column for column in header}
    if "metric" not in normalized or "value" not in normalized:
        raise ValueError(f"expected Metric,Value columns; found {header!r}")
    metric_column = normalized["metric"]
    value_column = normalized["value"]
    values: dict[str, float] = {}
    for row in rows:
        metric = row[metric_column].strip()
        if metric:
            values[metric] = _float(row[value_column])
    missing = [metric for metric in METRICS if metric not in values]
    if missing:
        raise ValueError("missing required metrics: " + ", ".join(missing))
    if any(not 0.0 <= values[metric] <= 1.0 for metric in METRICS):
        raise ValueError("one or more required metrics are outside [0,1]")
    return {metric: values[metric] for metric in METRICS}


def _validate_metrics(ledger: Ledger, path: Path) -> dict[str, float] | None:
    if not path.is_file() or not path.stat().st_size:
        return None
    try:
        metrics = _read_metrics(path)
        ok = True
        detail = json.dumps(metrics, sort_keys=True)
    except Exception as exc:
        metrics = None
        ok = False
        detail = f"{type(exc).__name__}: {exc}"
    ledger.add(
        "WP6",
        "run",
        "evaluation:metrics",
        "pass" if ok else "fail",
        detail,
        path,
    )
    return metrics


def _parse_wall_clock(times_path: Path, log_text: str) -> tuple[float | None, str]:
    if times_path.is_file() and times_path.stat().st_size:
        text = times_path.read_text(encoding="utf-8", errors="replace")
        matches = re.findall(
            r"(?im)^\s*Total:\s*([0-9]+(?:\.[0-9]+)?)\s+minutes?\s*$",
            text,
        )
        if matches:
            return float(matches[-1]), "times.txt:Total"
    patterns = (
        r"Alignment completed in\s+([0-9]+(?:\.[0-9]+)?)\s+minutes?",
        r"Completed alignment in\s+([0-9]+(?:\.[0-9]+)?)\s+minutes?",
    )
    for pattern in patterns:
        matches = re.findall(pattern, log_text, flags=re.IGNORECASE)
        if matches:
            return float(matches[-1]), "exact.log:completion"
    return None, ""


def _validate_timing(ledger: Ledger, times_path: Path, log_text: str) -> float | None:
    minutes, source = _parse_wall_clock(times_path, log_text)
    ok = minutes is not None and math.isfinite(minutes) and minutes > 0
    ledger.add(
        "WP6",
        "run",
        "timing:wall_clock",
        "pass" if ok else "missing",
        (
            f"{minutes:.10g} minutes from {source}"
            if ok and minutes is not None
            else "no completed Total timing in times.txt or exact.log"
        ),
        times_path,
    )
    return minutes if ok else None


def _read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            return next(reader)
        except StopIteration as exc:
            raise ValueError("CSV is empty") from exc


def _inspect_dump(
    ledger: Ledger,
    *,
    root: Path,
    dump_path: Path,
    log_text: str,
) -> DumpStats | None:
    if not dump_path.is_file() or not dump_path.stat().st_size:
        return None
    e1_dump_path = root / "exp/test/review_response/e1_dump/channel_dump.csv"
    try:
        e1_header = _read_header(e1_dump_path)
        baseline_ok = tuple(e1_header) == CANONICAL_DUMP_COLUMNS
        baseline_detail = (
            f"E1 has the canonical {len(e1_header)} columns"
            if baseline_ok
            else f"E1 header differs from canonical contract: {e1_header!r}"
        )
    except Exception as exc:
        e1_header = list(CANONICAL_DUMP_COLUMNS)
        baseline_ok = False
        baseline_detail = f"{type(exc).__name__}: {exc}"
    ledger.add(
        "WP6",
        "dump",
        "e1_header:canonical_35",
        "pass" if baseline_ok else "fail",
        baseline_detail,
        e1_dump_path,
    )

    try:
        handle = dump_path.open("r", encoding="utf-8", newline="")
    except OSError as exc:
        ledger.add("WP6", "dump", "csv:read", "fail", str(exc), dump_path)
        return None
    with handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            ledger.add("WP6", "dump", "csv:read", "fail", "CSV is empty", dump_path)
            return None
        exact_header = (
            header == e1_header
            and len(header) == 35
            and len(set(header)) == 35
            and tuple(header) == CANONICAL_DUMP_COLUMNS
        )
        ledger.add(
            "WP6",
            "dump",
            "columns:exact_same_35_as_e1",
            "pass" if exact_header else "fail",
            (
                "header is byte-for-byte column-identical to E1 and canonical WP1 order"
                if exact_header
                else f"e6={header!r}; e1={e1_header!r}"
            ),
            dump_path,
        )
        if not set(CANONICAL_DUMP_COLUMNS).issubset(header):
            return None
        indexes = {column: header.index(column) for column in CANONICAL_DUMP_COLUMNS}
        rows = 0
        sources: set[str] = set()
        sources_with_correct: set[str] = set()
        active_counts = {channel: 0 for channel in CHANNELS}
        sigma_inert_counts = {channel: 0 for channel in STRUCTURAL_CHANNELS}
        omega_inert_counts = {channel: 0 for channel in STRUCTURAL_CHANNELS}
        q_first: dict[str, float | None] = {channel: None for channel in CHANNELS}
        q_constant = {channel: True for channel in CHANNELS}
        invalid_rows = 0
        invalid_numeric = {
            column: 0 for column in sorted(DUMP_NUMERIC_COLUMNS | DUMP_BOOLEAN_COLUMNS)
        }
        omega_max_error = 0.0
        omega_mismatches = 0
        pair_max_error = 0.0
        pair_mismatches = 0
        pair_checked = 0

        for row in reader:
            rows += 1
            if len(row) != len(header):
                invalid_rows += 1
                continue
            src = row[indexes["src"]].strip()
            tgt = row[indexes["tgt"]].strip()
            if not src or not tgt:
                invalid_rows += 1
            if src:
                sources.add(src)
            values: dict[str, float] = {}
            row_valid = True
            for column in DUMP_NUMERIC_COLUMNS:
                try:
                    values[column] = _float(row[indexes[column]])
                except (TypeError, ValueError):
                    invalid_numeric[column] += 1
                    row_valid = False
            for column in DUMP_BOOLEAN_COLUMNS:
                try:
                    values[column] = float(_binary(row[indexes[column]]))
                except (TypeError, ValueError):
                    invalid_numeric[column] += 1
                    row_valid = False
            if not row_valid:
                continue
            if values["is_correct"] not in {0.0, 1.0}:
                invalid_numeric["is_correct"] += 1
                continue
            if values["is_correct"] == 1.0 and src:
                sources_with_correct.add(src)
            for channel in CHANNELS:
                active = values[f"sigma_{channel}"] > ACTIVE_SIGMA_THRESHOLD
                if active:
                    active_counts[channel] += 1
                    q_value = values[f"q_{channel}"]
                    if q_first[channel] is None:
                        q_first[channel] = q_value
                    elif q_value != q_first[channel]:
                        q_constant[channel] = False
            for channel in STRUCTURAL_CHANNELS:
                sigma_inert_counts[channel] += int(
                    values[f"sigma_{channel}"] < ACTIVE_SIGMA_THRESHOLD
                )
                omega_inert_counts[channel] += int(values[f"omega_{channel}"] < 0.01)

            omega_sum = sum(values[f"omega_{channel}"] for channel in STRUCTURAL_CHANNELS)
            expected_omega = 1.0 if values["sigma_struct"] > STRUCT_ACTIVE_THRESHOLD else 0.0
            omega_error = abs(omega_sum - expected_omega)
            omega_max_error = max(omega_max_error, omega_error)
            omega_mismatches += int(omega_error > IDENTITY_TOLERANCE)

            pair_checked += 1
            expected_pair = (1.0 - values["w_c"]) * values["s_lex"] + values["w_c"] * values[
                "S_struct"
            ]
            pair_error = abs(values["S_pair"] - expected_pair)
            pair_max_error = max(pair_max_error, pair_error)
            pair_mismatches += int(pair_error > IDENTITY_TOLERANCE)

    invalid_numeric = {column: count for column, count in invalid_numeric.items() if count}
    population_ok = rows > 0 and invalid_rows == 0 and not invalid_numeric
    ledger.add(
        "WP6",
        "dump",
        "finite_numeric_population",
        "pass" if population_ok else "fail",
        (
            f"rows={rows}; malformed_or_blank_id_rows={invalid_rows}; "
            f"invalid_numeric={json.dumps(invalid_numeric, sort_keys=True)}"
        ),
        dump_path,
    )
    ledger.add(
        "WP6",
        "dump",
        "source_count",
        "pass" if len(sources) == EXPECTED_FMA_SOURCES else "fail",
        f"distinct_src={len(sources)}; expected={EXPECTED_FMA_SOURCES}",
        dump_path,
    )
    identity_ok = (
        population_ok
        and omega_mismatches == 0
        and pair_mismatches == 0
        and pair_checked > 0
        and omega_max_error <= IDENTITY_TOLERANCE
        and pair_max_error <= IDENTITY_TOLERANCE
    )
    ledger.add(
        "WP6",
        "dump",
        "omega_and_S_pair_identities",
        "pass" if identity_ok else "fail",
        (
            f"omega_max_abs_error={omega_max_error:.12g}; "
            f"omega_mismatches={omega_mismatches}/{rows}; "
            f"S_pair_max_abs_error={pair_max_error:.12g}; "
            f"S_pair_mismatches={pair_mismatches}/{pair_checked}; "
            f"tolerance={IDENTITY_TOLERANCE:g}"
        ),
        dump_path,
    )

    removed_matches = re.findall(
        r"#Exact prefilter:\s*mappings=\d+,\s*removed_candidate_rows=(\d+)",
        log_text,
        flags=re.IGNORECASE,
    )
    exact_matches = re.findall(r"#Exact matches found:\s*(\d+)", log_text, flags=re.IGNORECASE)
    inference_sets = re.findall(r"#Inference Set:\s*(\d+)\s+samples", log_text, flags=re.IGNORECASE)
    removed: int | None
    if removed_matches:
        removed = int(removed_matches[-1])
    elif exact_matches and int(exact_matches[-1]) == 0:
        removed = 0
    else:
        removed = None
    inference_rows = int(inference_sets[-1]) if inference_sets else None
    accounting_ok = (
        removed is not None and rows + removed == EXPECTED_FMA_RAW_PAIRS and inference_rows == rows
    )
    ledger.add(
        "WP6",
        "dump",
        "raw_pair_accounting",
        "pass" if accounting_ok else "fail",
        (
            f"raw_candidate_pairs={EXPECTED_FMA_RAW_PAIRS}; scored_dump_rows={rows}; "
            f"removed_by_exact_prefilter={removed!r}; "
            f"log_inference_rows={inference_rows!r}; "
            f"accounted={rows + removed if removed is not None else None!r}"
        ),
        dump_path,
    )
    return DumpStats(
        rows=rows,
        sources=sources,
        sources_with_correct=sources_with_correct,
        active_counts=active_counts,
        sigma_inert_counts=sigma_inert_counts,
        omega_inert_counts=omega_inert_counts,
        q_first=q_first,
        q_constant=q_constant,
        invalid_rows=invalid_rows,
        invalid_numeric=invalid_numeric,
        omega_max_error=omega_max_error,
        omega_mismatches=omega_mismatches,
        pair_max_error=pair_max_error,
        pair_mismatches=pair_mismatches,
        pair_checked=pair_checked,
    )


def _normalized_header(header: Sequence[str]) -> dict[str, str]:
    return {
        re.sub(r"[^a-z0-9]+", "_", column.strip().lower()).strip("_"): column for column in header
    }


def _find_column(
    normalized: Mapping[str, str],
    *aliases: str,
    contains: Sequence[str] | None = None,
) -> str | None:
    for alias in aliases:
        key = re.sub(r"[^a-z0-9]+", "_", alias.lower()).strip("_")
        if key in normalized:
            return normalized[key]
    if contains:
        for normalized_name, original in normalized.items():
            if all(token in normalized_name for token in contains):
                return original
    return None


def _validate_activity_csv(
    ledger: Ledger,
    path: Path,
    dump_stats: DumpStats | None,
) -> dict[str, float] | None:
    if not ledger.artifact("WP6", "activity", path):
        return None
    try:
        header, rows = _read_csv(path)
        normalized = _normalized_header(header)
        required_exact = {
            "channel",
            "omim_sigma_inert",
            "fma_sigma_inert",
            "omim_omega_inert",
            "fma_omega_inert",
        }
        missing = sorted(required_exact - set(normalized))
        if missing:
            raise ValueError("missing activity columns: " + ", ".join(missing))
        channel_col = normalized["channel"]
        structural_rows = {
            row[channel_col].strip().lower(): row
            for row in rows
            if row.get(channel_col, "").strip().lower() in STRUCTURAL_CHANNELS
        }
        if set(structural_rows) != set(STRUCTURAL_CHANNELS):
            raise ValueError(
                f"expected four structural channel rows; found {sorted(structural_rows)}"
            )
        omim_expected = {
            "hier": 0.8274,
            "sim": 1.0,
            "diff": 1.0,
            "attr": 0.0,
        }
        values_ok = True
        for channel, row in structural_rows.items():
            for key in (
                "omim_sigma_inert",
                "fma_sigma_inert",
                "omim_omega_inert",
                "fma_omega_inert",
            ):
                value = _float(row[normalized[key]])
                if not 0.0 <= value <= 1.0:
                    values_ok = False
            for key in ("omim_sigma_inert", "omim_omega_inert"):
                if not math.isclose(
                    _float(row[normalized[key]]),
                    omim_expected[channel],
                    rel_tol=0.0,
                    abs_tol=5e-4,
                ):
                    values_ok = False
            if dump_stats is not None:
                expected_fma_sigma = (
                    dump_stats.rows - dump_stats.active_counts[channel]
                ) / dump_stats.rows
                if not math.isclose(
                    _float(row[normalized["fma_sigma_inert"]]),
                    expected_fma_sigma,
                    rel_tol=0.0,
                    abs_tol=1e-8,
                ):
                    values_ok = False

        stat_aliases = {
            "n": ("q_n", "fma_q_n"),
            "mean": ("q_mean", "fma_q_mean"),
            "sd": ("q_sd", "q_std", "fma_q_sd", "fma_q_std"),
            "min": ("q_min", "fma_q_min"),
            "p25": ("q_p25", "q_q25", "fma_q_p25"),
            "median": ("q_median", "q_p50", "fma_q_median"),
            "p75": ("q_p75", "q_q75", "fma_q_p75"),
            "max": ("q_max", "fma_q_max"),
        }
        distribution_columns: list[str] = []
        for score_prefix in ("q", "s"):
            for statistic, q_aliases in stat_aliases.items():
                aliases = tuple(alias.replace("q_", f"{score_prefix}_") for alias in q_aliases)
                column = _find_column(normalized, *aliases)
                if column is None:
                    values_ok = False
                else:
                    distribution_columns.append(column)
        s_tau_column = _find_column(
            normalized,
            "s_at_tau",
            "s_tau_count",
            "s_count_at_tau",
            "fma_s_at_tau",
            "fma_s_at_tau_count",
            contains=("s", "tau", "count"),
        )
        if s_tau_column is None:
            values_ok = False
        else:
            distribution_columns.append(s_tau_column)
        for row in structural_rows.values():
            for column in distribution_columns:
                value = _float(row[column])
                if column.endswith("_n") or "count" in column.lower() or "at_tau" in column.lower():
                    if value < 0 or value != int(value):
                        values_ok = False
            if dump_stats is not None:
                for column in distribution_columns:
                    normalized_column = re.sub(r"[^a-z0-9]+", "_", column.lower()).strip("_")
                    if normalized_column in {"q_n", "s_n", "fma_q_n", "fma_s_n"}:
                        if _int(row[column]) != dump_stats.rows:
                            values_ok = False

        diff_row = structural_rows["diff"]
        pivot_count_column = _find_column(
            normalized,
            "diff_q_positive_pivot_count",
            "q_diff_positive_at_pivot_count",
            "q_positive_and_diff_pivot_count",
            "fma_q_gt_0_and_deviation_lt_1e-6_count",
            contains=("pivot", "count"),
        )
        pivot_fraction_column = _find_column(
            normalized,
            "diff_q_positive_pivot_fraction",
            "q_diff_positive_at_pivot_fraction",
            "q_positive_and_diff_pivot_fraction",
            "fma_q_gt_0_and_deviation_lt_1e-6_fraction",
            contains=("pivot", "fraction"),
        )
        if pivot_count_column is None or pivot_fraction_column is None:
            values_ok = False
            pivot_values = None
        else:
            pivot_count = _int(diff_row[pivot_count_column])
            pivot_fraction = _float(diff_row[pivot_fraction_column])
            pivot_values = {
                "count": float(pivot_count),
                "fraction": pivot_fraction,
            }
            denominator = dump_stats.rows if dump_stats is not None else None
            if not 0.0 <= pivot_fraction <= 1.0 or pivot_count < 0:
                values_ok = False
            if denominator is not None and (
                pivot_count > denominator
                or not math.isclose(
                    pivot_fraction,
                    pivot_count / denominator,
                    rel_tol=0.0,
                    abs_tol=1e-8,
                )
            ):
                values_ok = False
        detail = (
            f"four structural rows; distribution_columns={distribution_columns}; "
            f"pivot_count_column={pivot_count_column!r}; "
            f"pivot_fraction_column={pivot_fraction_column!r}"
        )
    except Exception as exc:
        values_ok = False
        pivot_values = None
        detail = f"{type(exc).__name__}: {exc}"
    ledger.add(
        "WP6",
        "activity",
        "activity_schema_values_and_pivot",
        "pass" if values_ok else "fail",
        detail,
        path,
    )
    return pivot_values


def _extract_pivot_summary(
    header: Sequence[str], rows: Sequence[Mapping[str, str]]
) -> dict[str, float] | None:
    normalized = _normalized_header(header)
    count_column = _find_column(
        normalized,
        "q_positive_and_pivot_count",
        "diff_q_positive_pivot_count",
        "q_diff_positive_at_pivot_count",
        "q_gt_0_and_deviation_lt_1e-6_count",
        contains=("pivot", "count"),
    )
    fraction_column = _find_column(
        normalized,
        "q_positive_and_pivot_fraction",
        "diff_q_positive_pivot_fraction",
        "q_diff_positive_at_pivot_fraction",
        "q_gt_0_and_deviation_lt_1e-6_fraction",
        contains=("pivot", "fraction"),
    )
    if count_column and fraction_column and rows:
        summary_rows = [
            row
            for row in rows
            if str(row.get(_find_column(normalized, "record_type") or "", "")).strip().lower()
            in {"summary", "overall"}
        ]
        dataset_column = _find_column(normalized, "dataset")
        fma_rows = (
            [item for item in rows if item.get(dataset_column, "").strip().lower() == "fma"]
            if dataset_column
            else []
        )
        row = summary_rows[0] if summary_rows else (fma_rows[0] if fma_rows else rows[0])
        return {
            "count": float(_int(row[count_column])),
            "fraction": _float(row[fraction_column]),
        }

    metric_column = _find_column(normalized, "metric", "name", "statistic")
    value_column = _find_column(normalized, "value", "result")
    if metric_column and value_column:
        metrics = {row[metric_column].strip().lower(): row[value_column] for row in rows}
        count_key = next(
            (
                key
                for key in metrics
                if "pivot" in key and "count" in key and ("q" in key or "diff" in key)
            ),
            None,
        )
        fraction_key = next(
            (
                key
                for key in metrics
                if "pivot" in key
                and ("fraction" in key or "rate" in key)
                and ("q" in key or "diff" in key)
            ),
            None,
        )
        if count_key and fraction_key:
            return {
                "count": float(_int(metrics[count_key])),
                "fraction": _float(metrics[fraction_key]),
            }
    return None


def _validate_diff_pivot_csv(
    ledger: Ledger,
    path: Path,
    activity_pivot: Mapping[str, float] | None,
    dump_stats: DumpStats | None,
) -> None:
    if not ledger.artifact("WP6", "activity", path):
        return
    try:
        header, rows = _read_csv(path)
        if not rows:
            raise ValueError("CSV has no data rows")
        normalized = _normalized_header(header)
        textual_contract = " ".join(normalized)
        has_q = "q_diff" in textual_contract or any(
            name.startswith("q_bin") or name.startswith("q_gt_0") for name in normalized
        )
        has_deviation = any(
            ("abs" in name and "diff" in name and ("tau" in name or "pivot" in name))
            or ("distance" in name and "pivot" in name)
            or name.startswith("deviation_")
            for name in normalized
        )
        pivot = _extract_pivot_summary(header, rows)
        ok = has_q and has_deviation and pivot is not None
        if pivot is not None:
            count = _int(pivot["count"])
            fraction = _float(pivot["fraction"])
            ok = ok and count >= 0 and 0.0 <= fraction <= 1.0
            if dump_stats is not None:
                ok = (
                    ok
                    and count <= dump_stats.rows
                    and math.isclose(
                        fraction,
                        count / dump_stats.rows,
                        rel_tol=0.0,
                        abs_tol=1e-8,
                    )
                )
            if activity_pivot is not None:
                ok = (
                    ok
                    and count == _int(activity_pivot["count"])
                    and math.isclose(
                        fraction,
                        _float(activity_pivot["fraction"]),
                        rel_tol=0.0,
                        abs_tol=1e-10,
                    )
                )
        detail = (
            f"columns={header}; pivot={pivot}; activity_pivot={activity_pivot}; "
            f"has_q_diff={has_q}; has_abs_diff_minus_tau={has_deviation}"
        )
    except Exception as exc:
        ok = False
        detail = f"{type(exc).__name__}: {exc}"
    ledger.add(
        "WP6",
        "activity",
        "diff_pivot_detail_and_crosscheck",
        "pass" if ok else "fail",
        detail,
        path,
    )


def _flatten_json(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    flattened: list[tuple[str, Any]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.extend(_flatten_json(child, child_prefix))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            flattened.extend(_flatten_json(child, f"{prefix}[{index}]"))
    else:
        flattened.append((prefix, value))
    return flattened


def _validate_e6_validation_json(
    ledger: Ledger,
    path: Path,
    dump_stats: DumpStats | None,
) -> None:
    if not ledger.artifact("WP6", "activity", path):
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, (dict, list)):
            raise ValueError("top-level JSON must be an object or array")
        flat = _flatten_json(payload)
        if not flat:
            raise ValueError("validation JSON has no scalar values")
        false_checks = [
            key
            for key, value in flat
            if (
                key.lower().endswith((".passed", ".ok", ".valid"))
                or any(token in key.lower() for token in ("checks.", "validations."))
            )
            and value is False
        ]
        bad_statuses = [
            f"{key}={value}"
            for key, value in flat
            if key.lower().endswith("status")
            and isinstance(value, str)
            and value.strip().lower() in {"fail", "failed", "missing", "error", "invalid"}
        ]
        searchable = " ".join(key.lower() for key, _value in flat)
        required_topics = {
            "rows": ("row",),
            "sources": ("source",),
            "numeric": ("numeric",),
            "omega": ("omega",),
            "S_pair": ("s_pair",),
        }
        missing_topics = [
            topic
            for topic, tokens in required_topics.items()
            if not all(token in searchable for token in tokens)
        ]
        value_ok = not false_checks and not bad_statuses and not missing_topics
        if dump_stats is not None:
            scalar_values = [value for _key, value in flat]
            value_ok = (
                value_ok
                and dump_stats.rows in scalar_values
                and len(dump_stats.sources) in scalar_values
            )
            error_values = [
                _float(value)
                for key, value in flat
                if isinstance(value, (int, float, str))
                and ("max" in key.lower() and "error" in key.lower())
                and _is_finite_number(value)
            ]
            if error_values and max(error_values) > IDENTITY_TOLERANCE:
                value_ok = False
        detail = (
            f"false_checks={false_checks}; bad_statuses={bad_statuses}; "
            f"missing_topics={missing_topics}"
        )
    except Exception as exc:
        value_ok = False
        detail = f"{type(exc).__name__}: {exc}"
    ledger.add(
        "WP6",
        "activity",
        "validation_json:all_checks",
        "pass" if value_ok else "fail",
        detail,
        path,
    )


def _is_finite_number(value: Any) -> bool:
    try:
        _float(value)
        return True
    except (TypeError, ValueError):
        return False


def _validate_activity_outputs(
    ledger: Ledger,
    analysis_dir: Path,
    dump_stats: DumpStats | None,
) -> None:
    activity_path = analysis_dir / "e6_activity.csv"
    pivot_path = analysis_dir / "e6_diff_pivot.csv"
    validation_path = analysis_dir / "e6_validation.json"
    activity_pivot = _validate_activity_csv(ledger, activity_path, dump_stats)
    _validate_diff_pivot_csv(
        ledger,
        pivot_path,
        activity_pivot,
        dump_stats,
    )
    _validate_e6_validation_json(ledger, validation_path, dump_stats)
    for basename in ("e6_activity", "e6_diff_pivot", "e6_q_s_distributions"):
        for suffix in (".png", ".pdf"):
            _validate_plot(ledger, "WP6", analysis_dir / f"{basename}{suffix}")


def _validate_wp6_results(
    ledger: Ledger,
    path: Path,
    *,
    dump_stats: DumpStats | None,
    metrics: Mapping[str, float] | None,
    wall_clock_minutes: float | None,
    reliability_statuses: Mapping[str, str] | None = None,
) -> None:
    if not path.is_file() or not path.stat().st_size:
        return
    text = path.read_text(encoding="utf-8", errors="replace")
    lower = text.lower()
    accounting_ok = "32,724" in text or "32724" in text
    if dump_stats is not None:
        accounting_ok = accounting_ok and (
            f"{dump_stats.rows:,}" in text or str(dump_stats.rows) in text
        )
    ledger.add(
        "WP6",
        "results",
        "raw_pair_accounting_reported",
        "pass" if accounting_ok else "fail",
        "RESULTS.md reports raw and scored pair coverage",
        path,
    )
    context_ok = all(metric.lower() in lower for metric in METRICS)
    if metrics is not None:
        context_ok = context_ok and all(_number_appears(text, value) for value in metrics.values())
    ledger.add(
        "WP6",
        "results",
        "context_metrics_reported",
        "pass" if context_ok else "fail",
        "RESULTS.md labels MRR/Hits context metrics",
        path,
    )
    timing_ok = (
        "wall" in lower
        and re.search(r"\b(?:minute|minutes|hour|hours|h)\b", lower) is not None
        and ("estimate" in lower or "4.5" in lower or "5 h" in lower)
    )
    if wall_clock_minutes is not None:
        timing_ok = timing_ok and (
            _number_appears(text, wall_clock_minutes)
            or _number_appears(text, wall_clock_minutes / 60.0)
        )
    ledger.add(
        "WP6",
        "results",
        "actual_wall_clock_vs_estimate",
        "pass" if timing_ok else "fail",
        "RESULTS.md records actual wall clock against the 4.5–5 h estimate",
        path,
    )
    deviation_ok = (
        "tau_llm" in lower
        and "1.0" in lower
        and "0.5" in lower
        and ("deviation" in lower or "not comparable" in lower)
    )
    ledger.add(
        "WP6",
        "results",
        "tau_llm_deviation",
        "pass" if deviation_ok else "fail",
        "RESULTS.md explains WP6 tau_LLM=1.0 versus published 0.5",
        path,
    )
    pivot_ok = (
        "pivot" in lower
        and re.search(r"s[_ `-]*diff", lower) is not None
        and ("q_diff" in lower or "q diff" in lower or "difference quality" in lower)
    )
    ledger.add(
        "WP6",
        "results",
        "diff_pivot_sentence",
        "pass" if pivot_ok else "fail",
        "RESULTS.md explicitly answers the s_diff pivot question with q_diff context",
        path,
    )
    if reliability_statuses is not None:
        channel_terms = {
            "lex": ("lex", "lexical"),
            "hier": ("hier", "hierarchy"),
            "sim": ("sim", "similarity"),
            "diff": ("diff", "difference"),
            "attr": ("attr", "attribute"),
        }
        statuses_ok = all(
            any(term in lower for term in channel_terms[channel])
            and (
                status == "ok"
                or (status == "constant_quality" and "constant" in lower)
                or (status == "fully_inactive" and "inactive" in lower)
            )
            for channel, status in reliability_statuses.items()
        )
        ledger.add(
            "WP6",
            "results",
            "reliability_statuses_reported",
            "pass" if statuses_ok else "fail",
            json.dumps(reliability_statuses, sort_keys=True),
            path,
        )


def _number_appears(text: str, number: float) -> bool:
    candidates = {
        f"{number:g}",
        f"{number:.3f}",
        f"{number:.4f}",
        f"{number:.1f}",
    }
    return any(candidate in text for candidate in candidates)


def _validate_wp6(
    ledger: Ledger,
    *,
    root: Path,
    analysis_dir: Path,
) -> None:
    _validate_subset(ledger, root)
    run_dir = root / "exp/test/review_response/e6_fma_dump"
    _validate_e6_configs(ledger, root, run_dir)

    required_artifacts = (
        "evaluation_results.csv",
        "exact.log",
        "times.txt",
        "channel_dump.csv",
        "RESULTS.md",
    )
    artifact_presence: dict[str, bool] = {}
    for name in required_artifacts:
        artifact_presence[name] = ledger.artifact("WP6", "run", run_dir / name)

    log_text = _validate_logs(ledger, run_dir) if artifact_presence["exact.log"] else ""
    metrics = (
        _validate_metrics(ledger, run_dir / "evaluation_results.csv")
        if artifact_presence["evaluation_results.csv"]
        else None
    )
    wall_clock = (
        _validate_timing(ledger, run_dir / "times.txt", log_text)
        if artifact_presence["times.txt"]
        else None
    )
    dump_stats = (
        _inspect_dump(
            ledger,
            root=root,
            dump_path=run_dir / "channel_dump.csv",
            log_text=log_text,
        )
        if artifact_presence["channel_dump.csv"]
        else None
    )

    _validate_activity_outputs(ledger, analysis_dir, dump_stats)

    expected_statuses = dump_stats.expected_statuses if dump_stats else None
    expected_activity = dump_stats.activity if dump_stats else None
    expected_eligible = len(dump_stats.sources_with_correct) if dump_stats else None
    _validate_reliability_bundle(
        ledger,
        wp="WP6",
        prefix="e6",
        analysis_dir=analysis_dir,
        expected_statuses=expected_statuses,
        expected_activity=expected_activity,
        expected_sources=EXPECTED_FMA_SOURCES,
        expected_eligible_sources=expected_eligible,
    )

    _validate_reliability_manifests(ledger, analysis_dir, wp="WP6", prefix="e6")
    # Recover the common status map for the write-up check without making the write-up validator
    # depend on the internal return values of the bundle validator.
    reliability_statuses: dict[str, str] | None = None
    calibration_path = analysis_dir / "e6_calibration.csv"
    if calibration_path.is_file() and calibration_path.stat().st_size:
        try:
            _header, rows = _read_csv(calibration_path)
            summaries = [row for row in rows if row.get("record_type", "").strip() == "summary"]
            candidate = {row["channel"].strip(): row["status"].strip() for row in summaries}
            if set(candidate) == set(CHANNELS):
                reliability_statuses = candidate
        except Exception:
            reliability_statuses = None

    if artifact_presence["RESULTS.md"]:
        _validate_wp6_results(
            ledger,
            run_dir / "RESULTS.md",
            dump_stats=dump_stats,
            metrics=metrics,
            wall_clock_minutes=wall_clock,
            reliability_statuses=reliability_statuses,
        )


def _run_self_test() -> None:
    assert _float("1.25") == 1.25
    assert _int("42.0") == 42
    assert _binary("False") == 0
    assert _binary("1") == 1
    assert _pipe_floats("0|0.5|1") == [0.0, 0.5, 1.0]
    try:
        _float("nan")
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("_float accepted NaN")

    with tempfile.TemporaryDirectory(prefix="validate-review-addendum-") as temporary:
        root = Path(temporary)
        artifact = root / "artifact.csv"
        artifact.write_bytes(b"a,b\n1,2\n")
        digest = _sha256(artifact)
        assert len(digest) == 64
        manifest = root / "e5_hashes.sha256"
        manifest.write_text(f"{digest}  artifact.csv\n", encoding="utf-8")
        assert digest in manifest.read_text(encoding="utf-8")
        ledger = Ledger(root)
        assert ledger.artifact("SELF", "artifact", artifact)
        assert not ledger.blocking
        output = root / "ledger.csv"
        _write_ledger(output, ledger.checks)
        header, rows = _read_csv(output)
        assert header == ["wp", "area", "check", "status", "detail", "path"]
        assert rows and rows[0]["status"] == "pass"
    print("validate_review_addendum.py self-test passed.")


def main() -> None:
    args = parse_args()
    if args.self_test:
        _run_self_test()
        return

    root = args.repo_root.resolve()
    analysis_dir = root / "exp/test/review_response/analysis"
    output_path = (
        args.output if args.output is not None else analysis_dir / "addendum_validation.csv"
    )
    if not output_path.is_absolute():
        output_path = root / output_path

    ledger = Ledger(root)
    _validate_wp5(ledger, analysis_dir)
    _validate_wp6(ledger, root=root, analysis_dir=analysis_dir)

    blocking_before_summary = ledger.blocking
    if blocking_before_summary:
        summary_status = (
            "fail"
            if any(check.status == "fail" for check in blocking_before_summary)
            else "missing"
        )
        detail = (
            f"blocking_checks={len(blocking_before_summary)}; "
            f"failed={sum(check.status == 'fail' for check in blocking_before_summary)}; "
            f"missing={sum(check.status == 'missing' for check in blocking_before_summary)}"
        )
    else:
        summary_status = "pass"
        detail = "all WP5/WP6 addendum acceptance checks passed"
    ledger.add("ADDENDUM", "summary", "overall", summary_status, detail, output_path)
    _write_ledger(output_path, ledger.checks)

    counts: dict[str, int] = {}
    for check in ledger.checks:
        counts[check.status] = counts.get(check.status, 0) + 1
    print(
        f"Wrote {_display_path(root, output_path)} "
        f"({len(ledger.checks)} checks; {json.dumps(counts, sort_keys=True)})."
    )
    if args.strict and ledger.blocking:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
