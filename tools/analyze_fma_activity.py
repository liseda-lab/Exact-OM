#!/usr/bin/env python3
"""Compare structural-channel activity in the OMIM and FMA channel dumps.

The default inputs and outputs implement WP6 D3.  Before analysis, the tool
checks the shared 35-column dump contract, all numeric values, FMA source
coverage, and the structural-weight and pair-score identities.  It then writes
the side-by-side activity table, the difference-channel pivot diagnostic, a
machine-readable validation report, and PNG/PDF figures.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_OMIM_DUMP = Path("exp/test/review_response/e1_dump/channel_dump.csv")
DEFAULT_FMA_DUMP = Path("exp/test/review_response/e6_fma_dump/channel_dump.csv")
DEFAULT_OUTPUT_DIR = Path("exp/test/review_response/analysis")

CHANNELS = ("lex", "hier", "sim", "diff", "attr")
STRUCTURAL_CHANNELS = ("hier", "sim", "diff", "attr")
EXPECTED_FMA_SOURCES = 300
EXPECTED_OMIM_CANDIDATE_PAIRS = 30_502
EXPECTED_FMA_CANDIDATE_PAIRS = 32_724
SIGMA_INERT_THRESHOLD = 1e-6
OMEGA_INERT_THRESHOLD = 0.01
STRUCT_ACTIVE_THRESHOLD = 1e-8
PIVOT_DEVIATION_THRESHOLD = 1e-6
DEFAULT_IDENTITY_TOLERANCE = 1e-5

EXPECTED_SCHEMA = (
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

TEXT_COLUMNS = {
    "src",
    "tgt",
    "src_label_text",
    "tgt_label_text",
}
BINARY_COLUMNS = {
    "is_correct",
    "llm_invoked",
    "llm_gated",
}
NUMERIC_COLUMNS = tuple(
    column
    for column in EXPECTED_SCHEMA
    if column not in TEXT_COLUMNS and column not in {"llm_invoked", "llm_gated"}
)


class AnalysisValidationError(ValueError):
    """Raised when a dump does not satisfy the preregistered integrity checks."""


@dataclass(frozen=True)
class LoadedDump:
    key: str
    label: str
    path: Path
    header: tuple[str, ...]
    frame: pd.DataFrame


@dataclass(frozen=True)
class BinSpec:
    label: str
    lower: float | None
    upper: float | None
    lower_closed: bool
    upper_closed: bool


QUALITY_BINS = (
    BinSpec("q <= 0", None, 0.0, False, True),
    BinSpec("0 < q <= 0.01", 0.0, 0.01, False, True),
    BinSpec("0.01 < q <= 0.1", 0.01, 0.1, False, True),
    BinSpec("0.1 < q <= 0.25", 0.1, 0.25, False, True),
    BinSpec("0.25 < q <= 0.5", 0.25, 0.5, False, True),
    BinSpec("0.5 < q <= 0.75", 0.5, 0.75, False, True),
    BinSpec("q > 0.75", 0.75, None, False, False),
)

DEVIATION_BINS = (
    BinSpec("0 <= d < 1e-6", 0.0, 1e-6, True, False),
    BinSpec("1e-6 <= d < 1e-4", 1e-6, 1e-4, True, False),
    BinSpec("1e-4 <= d < 1e-3", 1e-4, 1e-3, True, False),
    BinSpec("1e-3 <= d < 0.01", 1e-3, 0.01, True, False),
    BinSpec("0.01 <= d < 0.1", 0.01, 0.1, True, False),
    BinSpec("0.1 <= d < 0.25", 0.1, 0.25, True, False),
    BinSpec("d >= 0.25", 0.25, None, True, False),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--omim-dump",
        type=Path,
        default=DEFAULT_OMIM_DUMP,
        help=f"Completed OMIM channel dump (default: {DEFAULT_OMIM_DUMP}).",
    )
    parser.add_argument(
        "--fma-dump",
        type=Path,
        default=DEFAULT_FMA_DUMP,
        help=f"Completed FMA channel dump (default: {DEFAULT_FMA_DUMP}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Artifact directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.5,
        help="Neutral score pivot used by both runs (default: 0.5).",
    )
    parser.add_argument(
        "--identity-tolerance",
        type=float,
        default=DEFAULT_IDENTITY_TOLERANCE,
        help="Maximum identity error; may be tightened but not exceed 1e-5.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run an isolated synthetic end-to-end test instead of reading experiment dumps.",
    )
    return parser.parse_args(argv)


def _read_header(path: Path) -> tuple[str, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"channel dump does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = tuple(next(reader))
        except StopIteration as exc:
            raise AnalysisValidationError(f"channel dump is empty: {path}") from exc
    if len(set(header)) != len(header):
        duplicates = sorted({column for column in header if header.count(column) > 1})
        raise AnalysisValidationError(f"{path}: duplicate header columns: {duplicates}")
    return header


def _as_binary(series: pd.Series, column: str, path: Path) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)
    numeric = pd.to_numeric(series, errors="coerce")
    mapped = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": 1.0,
                "false": 0.0,
                "yes": 1.0,
                "no": 0.0,
                "y": 1.0,
                "n": 0.0,
            }
        )
    )
    parsed = numeric.where(numeric.notna(), mapped)
    bad = parsed.isna() | ~np.isfinite(parsed) | ~parsed.isin([0.0, 1.0])
    if bad.any():
        examples = series.loc[bad].drop_duplicates().head(5).tolist()
        raise AnalysisValidationError(
            f"{path}: {column} has {int(bad.sum())} non-binary values; examples={examples}"
        )
    return parsed.astype(int)


def _load_dump(key: str, label: str, path: Path, header: tuple[str, ...]) -> LoadedDump:
    frame = pd.read_csv(path, low_memory=False)
    if tuple(frame.columns) != header:
        raise AnalysisValidationError(f"{path}: parsed columns differ from the raw CSV header")
    if frame.empty:
        raise AnalysisValidationError(f"{path}: channel dump has no data rows")

    for column in ("src", "tgt"):
        missing = frame[column].isna()
        blank = frame[column].astype(str).str.strip().eq("")
        if (missing | blank).any():
            raise AnalysisValidationError(
                f"{path}: {column} has {int((missing | blank).sum())} missing/blank values"
            )
        frame[column] = frame[column].astype(str)

    for column in sorted(BINARY_COLUMNS):
        frame[column] = _as_binary(frame[column], column, path)

    for column in sorted(set(NUMERIC_COLUMNS) - {"is_correct"}):
        numeric = pd.to_numeric(frame[column], errors="coerce")
        bad = numeric.isna() | ~np.isfinite(numeric)
        if bad.any():
            examples = frame.loc[bad, column].drop_duplicates().head(5).tolist()
            raise AnalysisValidationError(
                f"{path}: {column} has {int(bad.sum())} missing/non-finite values; "
                f"examples={examples}"
            )
        frame[column] = numeric.astype(float)

    return LoadedDump(
        key=key,
        label=label,
        path=path,
        header=header,
        frame=frame,
    )


def load_inputs(omim_path: Path, fma_path: Path) -> tuple[LoadedDump, LoadedDump]:
    omim_header = _read_header(omim_path)
    fma_header = _read_header(fma_path)
    if omim_header != fma_header:
        first_difference = next(
            (
                index
                for index, (omim_column, fma_column) in enumerate(zip(omim_header, fma_header))
                if omim_column != fma_column
            ),
            min(len(omim_header), len(fma_header)),
        )
        raise AnalysisValidationError(
            "OMIM/FMA schemas differ: "
            f"{len(omim_header)} vs {len(fma_header)} columns; "
            f"first differing position={first_difference + 1}"
        )
    if len(omim_header) != 35:
        raise AnalysisValidationError(
            f"shared channel-dump schema has {len(omim_header)} columns; expected 35"
        )
    if omim_header != EXPECTED_SCHEMA:
        missing = sorted(set(EXPECTED_SCHEMA) - set(omim_header))
        extra = sorted(set(omim_header) - set(EXPECTED_SCHEMA))
        order_only = not missing and not extra
        raise AnalysisValidationError(
            "shared 35-column schema does not match the WP1 contract: "
            f"missing={missing}, extra={extra}, order_only_mismatch={order_only}"
        )
    return (
        _load_dump("omim", "OMIM–ORDO", omim_path, omim_header),
        _load_dump("fma", "SNOMED–FMA Body", fma_path, fma_header),
    )


def _maximum(values: pd.Series) -> float:
    return float(values.max()) if not values.empty else 0.0


def _identity_summary(frame: pd.DataFrame, tolerance: float) -> dict[str, object]:
    omega_columns = [f"omega_{channel}" for channel in STRUCTURAL_CHANNELS]
    omega_sum = frame[omega_columns].sum(axis=1)
    active = frame["sigma_struct"].gt(STRUCT_ACTIVE_THRESHOLD)
    expected_omega_sum = pd.Series(
        np.where(active.to_numpy(), 1.0, 0.0),
        index=frame.index,
    )
    omega_error = (omega_sum - expected_omega_sum).abs()

    expected_pair = (1.0 - frame["w_c"]) * frame["s_lex"] + frame["w_c"] * frame["S_struct"]
    pair_error = (frame["S_pair"] - expected_pair).abs()

    omega_max = _maximum(omega_error)
    pair_max = _maximum(pair_error)
    return {
        "omega": {
            "definition": (
                "sum(omega_hier,omega_sim,omega_diff,omega_attr) = "
                "1 if sigma_struct > 1e-8 else 0"
            ),
            "active_rows": int(active.sum()),
            "inactive_rows": int((~active).sum()),
            "active_max_abs_error": _maximum(omega_error.loc[active]),
            "inactive_max_abs_error": _maximum(omega_error.loc[~active]),
            "max_abs_error": omega_max,
            "tolerance": tolerance,
            "passed": omega_max <= tolerance,
        },
        "S_pair": {
            "definition": "S_pair = (1 - w_c) * s_lex + w_c * S_struct",
            "rows": len(frame),
            "max_abs_error": pair_max,
            "mean_abs_error": float(pair_error.mean()),
            "tolerance": tolerance,
            "passed": pair_max <= tolerance,
        },
    }


def _check(
    name: str,
    passed: bool,
    observed: object,
    expected: object,
) -> dict[str, object]:
    return {
        "name": name,
        "passed": bool(passed),
        "observed": observed,
        "expected": expected,
    }


def build_validation_report(
    omim: LoadedDump,
    fma: LoadedDump,
    tau: float,
    tolerance: float,
) -> dict[str, object]:
    schema_hash = hashlib.sha256(",".join(omim.header).encode("utf-8")).hexdigest()
    omim_identity = _identity_summary(omim.frame, tolerance)
    fma_identity = _identity_summary(fma.frame, tolerance)

    checks = [
        _check("identical_schema", omim.header == fma.header, True, True),
        _check("schema_column_count", len(omim.header) == 35, len(omim.header), 35),
        _check(
            "schema_matches_wp1_contract",
            omim.header == EXPECTED_SCHEMA,
            schema_hash,
            hashlib.sha256(",".join(EXPECTED_SCHEMA).encode("utf-8")).hexdigest(),
        ),
        _check(
            "omim_finite_numeric_data",
            bool(np.isfinite(omim.frame[list(NUMERIC_COLUMNS)].to_numpy()).all()),
            len(NUMERIC_COLUMNS),
            f"all values finite in {len(NUMERIC_COLUMNS)} numeric columns",
        ),
        _check(
            "fma_finite_numeric_data",
            bool(np.isfinite(fma.frame[list(NUMERIC_COLUMNS)].to_numpy()).all()),
            len(NUMERIC_COLUMNS),
            f"all values finite in {len(NUMERIC_COLUMNS)} numeric columns",
        ),
        _check(
            "fma_unique_sources",
            fma.frame["src"].nunique() == EXPECTED_FMA_SOURCES,
            int(fma.frame["src"].nunique()),
            EXPECTED_FMA_SOURCES,
        ),
        _check(
            "omim_scored_rows_within_candidate_expectation",
            0 < len(omim.frame) <= EXPECTED_OMIM_CANDIDATE_PAIRS,
            len(omim.frame),
            f"1..{EXPECTED_OMIM_CANDIDATE_PAIRS}",
        ),
        _check(
            "fma_scored_rows_within_candidate_expectation",
            0 < len(fma.frame) <= EXPECTED_FMA_CANDIDATE_PAIRS,
            len(fma.frame),
            f"1..{EXPECTED_FMA_CANDIDATE_PAIRS}",
        ),
        _check(
            "omim_omega_identity",
            bool(omim_identity["omega"]["passed"]),
            omim_identity["omega"]["max_abs_error"],
            f"<= {tolerance:g}",
        ),
        _check(
            "fma_omega_identity",
            bool(fma_identity["omega"]["passed"]),
            fma_identity["omega"]["max_abs_error"],
            f"<= {tolerance:g}",
        ),
        _check(
            "omim_S_pair_identity",
            bool(omim_identity["S_pair"]["passed"]),
            omim_identity["S_pair"]["max_abs_error"],
            f"<= {tolerance:g}",
        ),
        _check(
            "fma_S_pair_identity",
            bool(fma_identity["S_pair"]["passed"]),
            fma_identity["S_pair"]["max_abs_error"],
            f"<= {tolerance:g}",
        ),
    ]
    status = "passed" if all(bool(check["passed"]) for check in checks) else "failed"
    return {
        "status": status,
        "parameters": {
            "tau": tau,
            "identity_tolerance": tolerance,
            "structurally_active_when": f"sigma_struct > {STRUCT_ACTIVE_THRESHOLD:g}",
            "sigma_inert_when": f"sigma_k < {SIGMA_INERT_THRESHOLD:g}",
            "omega_inert_when": f"omega_k < {OMEGA_INERT_THRESHOLD:g}",
            "diff_pivot_when": (
                f"q_diff > 0 and abs(s_diff - tau) < {PIVOT_DEVIATION_THRESHOLD:g}"
            ),
        },
        "schema": {
            "column_count": len(omim.header),
            "columns": list(omim.header),
            "sha256": schema_hash,
            "numeric_columns_checked": list(NUMERIC_COLUMNS),
        },
        "inputs": {
            "omim": {
                "label": omim.label,
                "path": str(omim.path.resolve()),
                "scored_rows": len(omim.frame),
                "candidate_pair_expectation": EXPECTED_OMIM_CANDIDATE_PAIRS,
                "exact_prefilter_reduction": (EXPECTED_OMIM_CANDIDATE_PAIRS - len(omim.frame)),
                "unique_sources": int(omim.frame["src"].nunique()),
            },
            "fma": {
                "label": fma.label,
                "path": str(fma.path.resolve()),
                "scored_rows": len(fma.frame),
                "candidate_pair_expectation": EXPECTED_FMA_CANDIDATE_PAIRS,
                "exact_prefilter_reduction": EXPECTED_FMA_CANDIDATE_PAIRS - len(fma.frame),
                "unique_sources": int(fma.frame["src"].nunique()),
            },
        },
        "identities": {
            "omim": omim_identity,
            "fma": fma_identity,
        },
        "checks": checks,
    }


def _describe(values: pd.Series, tau: float) -> dict[str, int | float]:
    n = len(values)
    quantiles = values.quantile([0.25, 0.5, 0.75])
    at_tau = int(values.eq(tau).sum())
    return {
        "n": n,
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)) if n > 1 else 0.0,
        "min": float(values.min()),
        "p25": float(quantiles.loc[0.25]),
        "median": float(quantiles.loc[0.5]),
        "p75": float(quantiles.loc[0.75]),
        "max": float(values.max()),
        "at_tau_count": at_tau,
        "at_tau_fraction": at_tau / n,
    }


def activity_table(
    dumps: Sequence[LoadedDump],
    tau: float,
) -> pd.DataFrame:
    by_key = {dump.key: dump for dump in dumps}
    rows: list[dict[str, object]] = []
    for channel in STRUCTURAL_CHANNELS:
        summaries: dict[str, dict[str, object]] = {}
        for key in ("omim", "fma"):
            frame = by_key[key].frame
            sigma = frame[f"sigma_{channel}"]
            omega = frame[f"omega_{channel}"]
            sigma_inert = sigma.lt(SIGMA_INERT_THRESHOLD)
            omega_inert = omega.lt(OMEGA_INERT_THRESHOLD)
            summaries[key] = {
                "pairs": len(frame),
                "sigma_inert_count": int(sigma_inert.sum()),
                "sigma_inert": float(sigma_inert.mean()),
                "omega_inert_count": int(omega_inert.sum()),
                "omega_inert": float(omega_inert.mean()),
                "q": _describe(frame[f"q_{channel}"], tau),
                "s": _describe(frame[f"s_{channel}"], tau),
            }

        row: dict[str, object] = {
            "channel": channel,
            "omim_sigma_inert": summaries["omim"]["sigma_inert"],
            "fma_sigma_inert": summaries["fma"]["sigma_inert"],
            "omim_omega_inert": summaries["omim"]["omega_inert"],
            "fma_omega_inert": summaries["fma"]["omega_inert"],
            "sigma_inert_threshold": SIGMA_INERT_THRESHOLD,
            "omega_inert_threshold": OMEGA_INERT_THRESHOLD,
            "tau": tau,
        }
        for key in ("omim", "fma"):
            summary = summaries[key]
            row[f"{key}_pairs"] = summary["pairs"]
            row[f"{key}_sigma_inert_count"] = summary["sigma_inert_count"]
            row[f"{key}_omega_inert_count"] = summary["omega_inert_count"]
            for quantity in ("q", "s"):
                distribution = summary[quantity]
                assert isinstance(distribution, dict)
                for statistic, value in distribution.items():
                    row[f"{key}_{quantity}_{statistic}"] = value
            if channel == "diff":
                frame = by_key[key].frame
                q_positive = frame["q_diff"].gt(0.0)
                deviation = frame["s_diff"].sub(tau).abs()
                near_pivot = deviation.lt(PIVOT_DEVIATION_THRESHOLD)
                joint = q_positive & near_pivot
                q_positive_count = int(q_positive.sum())
                near_pivot_count = int(near_pivot.sum())
                joint_count = int(joint.sum())
                row[f"{key}_q_gt_0_count"] = q_positive_count
                row[f"{key}_q_gt_0_fraction"] = float(q_positive.mean())
                row[f"{key}_deviation_lt_1e-6_count"] = near_pivot_count
                row[f"{key}_deviation_lt_1e-6_fraction"] = float(near_pivot.mean())
                row[f"{key}_q_gt_0_and_deviation_lt_1e-6_count"] = joint_count
                row[f"{key}_q_gt_0_and_deviation_lt_1e-6_fraction"] = float(joint.mean())
                row[f"{key}_q_gt_0_and_deviation_lt_1e-6_fraction_of_q_gt_0"] = (
                    joint_count / q_positive_count if q_positive_count else 0.0
                )
        rows.append(row)
    return pd.DataFrame(rows)


def _bin_codes(values: np.ndarray, bins: Sequence[BinSpec]) -> np.ndarray:
    codes = np.full(len(values), -1, dtype=int)
    for index, bin_spec in enumerate(bins):
        selected = np.ones(len(values), dtype=bool)
        if bin_spec.lower is not None:
            if bin_spec.lower_closed:
                selected &= values >= bin_spec.lower
            else:
                selected &= values > bin_spec.lower
        if bin_spec.upper is not None:
            if bin_spec.upper_closed:
                selected &= values <= bin_spec.upper
            else:
                selected &= values < bin_spec.upper
        overlap = (
            selected & codes.ge(0) if isinstance(codes, pd.Series) else selected & (codes >= 0)
        )
        if overlap.any():
            raise RuntimeError(f"overlapping bin definition at {bin_spec.label}")
        codes[selected] = index
    if (codes < 0).any():
        examples = values[codes < 0][:5].tolist()
        raise RuntimeError(f"bin definitions do not cover values: {examples}")
    return codes


def diff_pivot_table(
    dumps: Sequence[LoadedDump],
    tau: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dump in dumps:
        q = dump.frame["q_diff"].to_numpy(dtype=float)
        deviation = np.abs(dump.frame["s_diff"].to_numpy(dtype=float) - tau)
        q_codes = _bin_codes(q, QUALITY_BINS)
        deviation_codes = _bin_codes(deviation, DEVIATION_BINS)
        total = len(q)

        q_positive = q > 0.0
        near_pivot = deviation < PIVOT_DEVIATION_THRESHOLD
        joint_diagnostic = q_positive & near_pivot
        q_positive_count = int(q_positive.sum())
        near_pivot_count = int(near_pivot.sum())
        joint_count = int(joint_diagnostic.sum())

        for q_index, q_bin in enumerate(QUALITY_BINS):
            q_selected = q_codes == q_index
            q_count = int(q_selected.sum())
            for deviation_index, deviation_bin in enumerate(DEVIATION_BINS):
                deviation_selected = deviation_codes == deviation_index
                deviation_count = int(deviation_selected.sum())
                count = int((q_selected & deviation_selected).sum())
                rows.append(
                    {
                        "dataset": dump.key,
                        "dataset_label": dump.label,
                        "total_rows": total,
                        "q_bin_index": q_index + 1,
                        "q_bin": q_bin.label,
                        "q_lower": q_bin.lower,
                        "q_upper": q_bin.upper,
                        "deviation_bin_index": deviation_index + 1,
                        "deviation_bin": deviation_bin.label,
                        "deviation_lower": deviation_bin.lower,
                        "deviation_upper": deviation_bin.upper,
                        "count": count,
                        "fraction_of_dataset": count / total,
                        "q_bin_count": q_count,
                        "q_bin_fraction": q_count / total,
                        "deviation_bin_count": deviation_count,
                        "deviation_bin_fraction": deviation_count / total,
                        "fraction_within_q_bin": count / q_count if q_count else 0.0,
                        "q_gt_0_count": q_positive_count,
                        "q_gt_0_fraction": q_positive_count / total,
                        "deviation_lt_1e-6_count": near_pivot_count,
                        "deviation_lt_1e-6_fraction": near_pivot_count / total,
                        "q_gt_0_and_deviation_lt_1e-6_count": joint_count,
                        "q_gt_0_and_deviation_lt_1e-6_fraction": joint_count / total,
                    }
                )
    return pd.DataFrame(rows)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(
        temporary,
        index=False,
        float_format="%.10g",
        lineterminator="\n",
    )
    temporary.replace(path)


def _write_json(value: object, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def _save_figure(fig: plt.Figure, stem: Path) -> None:
    for suffix in ("png", "pdf"):
        destination = stem.with_suffix(f".{suffix}")
        temporary = destination.with_name(f".{destination.name}.tmp")
        fig.savefig(
            temporary,
            format=suffix,
            dpi=220,
            bbox_inches="tight",
        )
        temporary.replace(destination)
    plt.close(fig)


def plot_activity(table: pd.DataFrame, output_dir: Path) -> None:
    channels = table["channel"].tolist()
    x = np.arange(len(channels))
    width = 0.36
    colors = ("#3B6FB6", "#D97706")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), sharey=True)
    panels = (
        ("sigma", r"$\sigma_k < 10^{-6}$"),
        ("omega", r"$\omega_k < 0.01$"),
    )
    for axis, (metric, title) in zip(axes, panels):
        for offset, (key, label, color) in enumerate(
            (
                ("omim", "OMIM–ORDO", colors[0]),
                ("fma", "SNOMED–FMA Body", colors[1]),
            )
        ):
            positions = x + (offset - 0.5) * width
            values = table[f"{key}_{metric}_inert"].to_numpy(dtype=float)
            bars = axis.bar(positions, values, width, label=label, color=color, alpha=0.9)
            for bar, value in zip(bars, values):
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    min(value + 0.025, 1.055),
                    f"{value:.1%}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=90 if value > 0.92 else 0,
                )
        axis.set_title(title)
        axis.set_xticks(x, channels)
        axis.set_ylim(0.0, 1.1)
        axis.set_ylabel("Fraction of scored pairs")
        axis.grid(axis="y", alpha=0.25)
    axes[1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Structural-channel inertness by ontology pair")
    fig.tight_layout()
    _save_figure(fig, output_dir / "e6_activity")


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(values)
    probability = np.arange(1, len(ordered) + 1, dtype=float) / len(ordered)
    return ordered, probability


def plot_q_s_distributions(
    dumps: Sequence[LoadedDump],
    tau: float,
    output_dir: Path,
) -> None:
    colors = {"omim": "#3B6FB6", "fma": "#D97706"}
    fig, axes = plt.subplots(
        2,
        len(STRUCTURAL_CHANNELS),
        figsize=(15.5, 7.0),
        sharey=True,
    )
    for column_index, channel in enumerate(STRUCTURAL_CHANNELS):
        for row_index, quantity in enumerate(("q", "s")):
            axis = axes[row_index, column_index]
            for dump in dumps:
                x, y = _ecdf(dump.frame[f"{quantity}_{channel}"].to_numpy(dtype=float))
                axis.step(
                    x,
                    y,
                    where="post",
                    linewidth=1.5,
                    color=colors[dump.key],
                    label=dump.label,
                )
            axis.axvline(tau, color="#555555", linestyle=":", linewidth=1.0)
            axis.set_title(f"{quantity}_{channel}")
            axis.set_xlabel(quantity)
            axis.set_ylim(0.0, 1.01)
            axis.grid(alpha=0.22)
            if column_index == 0:
                axis.set_ylabel("Empirical CDF")
    axes[0, -1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Structural-channel quality and score distributions (dotted line: tau)")
    fig.tight_layout()
    _save_figure(fig, output_dir / "e6_q_s_distributions")


def plot_diff_pivot(table: pd.DataFrame, output_dir: Path) -> None:
    dataset_order = ("omim", "fma")
    q_labels = [bin_spec.label for bin_spec in QUALITY_BINS]
    deviation_labels = [bin_spec.label.replace("d", "|s−tau|") for bin_spec in DEVIATION_BINS]

    matrices: dict[str, np.ndarray] = {}
    for key in dataset_order:
        part = table.loc[table["dataset"].eq(key)]
        matrix = (
            part.pivot(
                index="q_bin_index",
                columns="deviation_bin_index",
                values="fraction_of_dataset",
            )
            .reindex(
                index=range(1, len(QUALITY_BINS) + 1),
                columns=range(1, len(DEVIATION_BINS) + 1),
                fill_value=0.0,
            )
            .to_numpy(dtype=float)
        )
        matrices[key] = matrix
    shared_max = max(float(matrix.max()) for matrix in matrices.values())
    if shared_max <= 0.0:
        shared_max = 1.0

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16.0, 6.2),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    image = None
    for axis, key in zip(axes, dataset_order):
        part = table.loc[table["dataset"].eq(key)]
        matrix = matrices[key]
        image = axis.imshow(
            matrix * 100.0,
            aspect="auto",
            origin="lower",
            cmap="YlGnBu",
            vmin=0.0,
            vmax=shared_max * 100.0,
        )
        total = int(part["total_rows"].iloc[0])
        joint_count = int(part["q_gt_0_and_deviation_lt_1e-6_count"].iloc[0])
        joint_fraction = float(part["q_gt_0_and_deviation_lt_1e-6_fraction"].iloc[0])
        label = str(part["dataset_label"].iloc[0])
        axis.set_title(
            f"{label}\nq>0 & |s−tau|<1e-6: " f"{joint_count:,}/{total:,} ({joint_fraction:.2%})"
        )
        axis.set_xticks(range(len(deviation_labels)), deviation_labels, rotation=35, ha="right")
        axis.set_yticks(range(len(q_labels)), q_labels)
        axis.set_xlabel("Absolute difference-score deviation bin")
        axis.set_ylabel("Difference-quality bin")
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                percentage = matrix[row_index, column_index] * 100.0
                if percentage <= 0.0:
                    continue
                text_color = "white" if percentage > shared_max * 50.0 else "black"
                axis.text(
                    column_index,
                    row_index,
                    f"{percentage:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color=text_color,
                )
    assert image is not None
    fig.colorbar(image, ax=list(axes), label="Percent of dataset rows", shrink=0.85)
    fig.suptitle("Joint distribution of q_diff and |s_diff − tau|")
    _save_figure(fig, output_dir / "e6_diff_pivot")


def analyze(
    omim_path: Path,
    fma_path: Path,
    output_dir: Path,
    tau: float = 0.5,
    tolerance: float = DEFAULT_IDENTITY_TOLERANCE,
) -> dict[str, object]:
    if not math.isfinite(tau):
        raise ValueError("--tau must be finite")
    if not math.isfinite(tolerance) or not (0.0 < tolerance <= 1e-5):
        raise ValueError("--identity-tolerance must be finite, positive, and <= 1e-5")

    omim, fma = load_inputs(omim_path, fma_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    validation = build_validation_report(omim, fma, tau, tolerance)
    _write_json(validation, output_dir / "e6_validation.json")
    if validation["status"] != "passed":
        failed = [str(check["name"]) for check in validation["checks"] if not bool(check["passed"])]
        raise AnalysisValidationError(
            "integrity validation failed; see e6_validation.json: " + ", ".join(failed)
        )

    dumps = (omim, fma)
    activity = activity_table(dumps, tau)
    pivot = diff_pivot_table(dumps, tau)
    _write_csv(activity, output_dir / "e6_activity.csv")
    _write_csv(pivot, output_dir / "e6_diff_pivot.csv")

    plot_activity(activity, output_dir)
    plot_q_s_distributions(dumps, tau, output_dir)
    plot_diff_pivot(pivot, output_dir)

    fma_pivot = pivot.loc[pivot["dataset"].eq("fma")].iloc[0]
    return {
        "omim_rows": len(omim.frame),
        "fma_rows": len(fma.frame),
        "fma_sources": int(fma.frame["src"].nunique()),
        "fma_q_gt_0_and_deviation_lt_1e-6_count": int(
            fma_pivot["q_gt_0_and_deviation_lt_1e-6_count"]
        ),
        "fma_q_gt_0_and_deviation_lt_1e-6_fraction": float(
            fma_pivot["q_gt_0_and_deviation_lt_1e-6_fraction"]
        ),
        "output_dir": str(output_dir.resolve()),
    }


def _synthetic_dump(
    source_prefix: str,
    target_prefix: str,
    source_count: int,
    tau: float,
    phase: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for source_index in range(source_count):
        for candidate_index in range(2):
            token = source_index * 2 + candidate_index + phase
            scores = {
                "hier": tau if token % 5 == 0 else (0.68 if token % 2 else 0.32),
                "sim": tau if token % 4 else 0.62,
                "diff": tau if token % 3 else 0.78,
                "attr": 0.72 if token % 2 else 0.64,
            }
            qualities = {
                "hier": (token % 6) / 5.0,
                "sim": (token % 5) / 4.0,
                "diff": 0.0 if token % 7 == 0 else (token % 9 + 1) / 10.0,
                "attr": 0.75,
            }
            sigmas = {
                channel: qualities[channel] * abs(scores[channel] - tau) ** 2
                for channel in STRUCTURAL_CHANNELS
            }
            sigma_struct = sum(sigmas.values())
            if sigma_struct > STRUCT_ACTIVE_THRESHOLD:
                omegas = {
                    channel: sigmas[channel] / sigma_struct for channel in STRUCTURAL_CHANNELS
                }
            else:
                omegas = {channel: 0.0 for channel in STRUCTURAL_CHANNELS}
            s_struct = (
                sum(omegas[channel] * scores[channel] for channel in STRUCTURAL_CHANNELS)
                if sigma_struct > STRUCT_ACTIVE_THRESHOLD
                else tau
            )
            q_struct = (
                sum(omegas[channel] * qualities[channel] for channel in STRUCTURAL_CHANNELS)
                if sigma_struct > STRUCT_ACTIVE_THRESHOLD
                else 0.0
            )
            s_lex = 0.7 if token % 2 else 0.3
            q_lex = 0.85
            sigma_lex = q_lex * abs(s_lex - tau) ** 2
            denominator = sigma_lex + sigma_struct
            w_c = sigma_struct / denominator if denominator > STRUCT_ACTIVE_THRESHOLD else 0.0
            s_pair = (1.0 - w_c) * s_lex + w_c * s_struct
            row: dict[str, object] = {
                "src": f"{source_prefix}{source_index:04d}",
                "tgt": f"{target_prefix}{source_index:04d}_{candidate_index}",
                "is_correct": int(candidate_index == 0),
                "s_lex": s_lex,
                "s_hier": scores["hier"],
                "s_sim": scores["sim"],
                "s_diff": scores["diff"],
                "s_attr": scores["attr"],
                "q_lex": q_lex,
                "q_hier": qualities["hier"],
                "q_sim": qualities["sim"],
                "q_diff": qualities["diff"],
                "q_attr": qualities["attr"],
                "sigma_lex": sigma_lex,
                "sigma_hier": sigmas["hier"],
                "sigma_sim": sigmas["sim"],
                "sigma_diff": sigmas["diff"],
                "sigma_attr": sigmas["attr"],
                "sigma_struct": sigma_struct,
                "omega_hier": omegas["hier"],
                "omega_sim": omegas["sim"],
                "omega_diff": omegas["diff"],
                "omega_attr": omegas["attr"],
                "w_c": w_c,
                "S_struct": s_struct,
                "Q_struct": q_struct,
                "S_lctx": s_pair,
                "U": abs(s_pair - tau),
                "p_llm": 0.0,
                "llm_invoked": False,
                "llm_gated": False,
                "S_pair": s_pair,
                "S_final": s_pair,
                "src_label_text": f"source {source_index}",
                "tgt_label_text": f"target {source_index} {candidate_index}",
            }
            rows.append(row)
    return pd.DataFrame(rows, columns=EXPECTED_SCHEMA)


def run_self_test() -> None:
    tau = 0.5
    with tempfile.TemporaryDirectory(prefix="exact-e6-activity-") as temporary_name:
        root = Path(temporary_name)
        omim_path = root / "omim.csv"
        fma_path = root / "fma.csv"
        output_dir = root / "analysis"

        omim_frame = _synthetic_dump(
            "http://omim.test/",
            "http://ordo.test/",
            source_count=12,
            tau=tau,
            phase=0,
        )
        fma_frame = _synthetic_dump(
            "http://snomed.test/",
            "http://fma.test/",
            source_count=EXPECTED_FMA_SOURCES,
            tau=tau,
            phase=1,
        )
        omim_frame.to_csv(omim_path, index=False, lineterminator="\n")
        fma_frame.to_csv(fma_path, index=False, lineterminator="\n")

        summary = analyze(omim_path, fma_path, output_dir, tau=tau)
        expected_artifacts = (
            "e6_activity.csv",
            "e6_diff_pivot.csv",
            "e6_validation.json",
            "e6_activity.png",
            "e6_activity.pdf",
            "e6_q_s_distributions.png",
            "e6_q_s_distributions.pdf",
            "e6_diff_pivot.png",
            "e6_diff_pivot.pdf",
        )
        missing = [name for name in expected_artifacts if not (output_dir / name).is_file()]
        if missing:
            raise AssertionError(f"self-test artifacts missing: {missing}")

        activity = pd.read_csv(output_dir / "e6_activity.csv")
        pivot = pd.read_csv(output_dir / "e6_diff_pivot.csv")
        validation = json.loads((output_dir / "e6_validation.json").read_text(encoding="utf-8"))
        if activity["channel"].tolist() != list(STRUCTURAL_CHANNELS):
            raise AssertionError("self-test activity table has incorrect channel rows")
        if validation["status"] != "passed":
            raise AssertionError("self-test validation report did not pass")
        if int(summary["fma_sources"]) != EXPECTED_FMA_SOURCES:
            raise AssertionError("self-test did not preserve 300 FMA sources")
        pivot_summary = pivot.loc[pivot["dataset"].eq("fma")].iloc[0]
        if int(pivot_summary["q_gt_0_and_deviation_lt_1e-6_count"]) <= 0:
            raise AssertionError("self-test did not exercise the positive-q neutral-pivot case")

        activity_diff = activity.loc[activity["channel"].eq("diff")].iloc[0]
        for key in ("omim", "fma"):
            detailed = pivot.loc[pivot["dataset"].eq(key)].iloc[0]
            count_columns = (
                "q_gt_0_count",
                "deviation_lt_1e-6_count",
                "q_gt_0_and_deviation_lt_1e-6_count",
            )
            fraction_columns = (
                "q_gt_0_fraction",
                "deviation_lt_1e-6_fraction",
                "q_gt_0_and_deviation_lt_1e-6_fraction",
            )
            for column in count_columns:
                if int(activity_diff[f"{key}_{column}"]) != int(detailed[column]):
                    raise AssertionError(f"activity/pivot count mismatch for {key}_{column}")
            for column in fraction_columns:
                if not math.isclose(
                    float(activity_diff[f"{key}_{column}"]),
                    float(detailed[column]),
                    rel_tol=1e-9,
                    abs_tol=1e-12,
                ):
                    raise AssertionError(f"activity/pivot rate mismatch for {key}_{column}")
        bad_fma_path = root / "bad_fma.csv"
        last_source = fma_frame["src"].iloc[-1]
        fma_frame.loc[fma_frame["src"].ne(last_source)].to_csv(
            bad_fma_path,
            index=False,
            lineterminator="\n",
        )
        try:
            analyze(omim_path, bad_fma_path, root / "bad_analysis", tau=tau)
        except AnalysisValidationError as exc:
            if "fma_unique_sources" not in str(exc):
                raise AssertionError("self-test failed for an unexpected reason") from exc
        else:
            raise AssertionError("self-test accepted an FMA dump with only 299 sources")

    print("Self-test passed: schema, finite data, identities, outputs, plots, and failures.")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        run_self_test()
        return 0
    try:
        summary = analyze(
            args.omim_dump,
            args.fma_dump,
            args.output_dir,
            tau=args.tau,
            tolerance=args.identity_tolerance,
        )
    except (AnalysisValidationError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    count = int(summary["fma_q_gt_0_and_deviation_lt_1e-6_count"])
    fraction = float(summary["fma_q_gt_0_and_deviation_lt_1e-6_fraction"])
    print(
        f"Validated and analysed {int(summary['omim_rows']):,} OMIM rows and "
        f"{int(summary['fma_rows']):,} FMA rows from "
        f"{int(summary['fma_sources']):,} FMA sources."
    )
    print(
        "FMA q_diff > 0 and abs(s_diff - tau) < 1e-6: "
        f"{count:,}/{int(summary['fma_rows']):,} ({fraction:.4%})."
    )
    print(f"Artifacts written to {summary['output_dir']}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
