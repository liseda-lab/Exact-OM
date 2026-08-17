#!/usr/bin/env python3
"""Summarize and validate the completed review-response experiments.

This tool never launches or changes a run. It reads the fixed e1/e2/e4 run directories and writes
two results tables plus a machine-readable validation ledger. Missing, failed, and invalid runs
remain in every output with an explicit status and reason.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml


METRICS = ("MRR", "Hits@1", "Hits@5", "Hits@10")
EXPECTED_CANDIDATE_PAIRS = 30_502
EXPECTED_SOURCE_COUNT = 300
FLOAT_ATOL = 1e-6

REFERENCE_CONFIG = Path("exp/test/Full_local_bioml/omim-ordo-val/config.yaml")
DEFAULT_CONFIG = Path("exact/default_config.yaml")
DEFAULT_RUNS_ROOT = Path("exp/test/review_response")
DEFAULT_ANALYSIS_DIR = DEFAULT_RUNS_ROOT / "analysis"

E2_ARMS = (
    ("full", "e1_dump"),
    ("uniform_weights", "e2_uniform_weights"),
    ("no_hier", "e2_no_hier"),
    ("no_sim", "e2_no_sim"),
    ("no_diff", "e2_no_diff"),
    ("no_attr", "e2_no_attr"),
    ("lex_only", "e2_lex_only"),
)

E4_SETTINGS = (
    ("gamma", 1.0, "e4_gamma_1", False),
    ("gamma", 2.0, "e4_gamma_2", False),
    ("gamma", 3.0, "e4_gamma_3", False),
    ("gamma", 4.0, "e4_gamma_4", False),
    ("tau", 0.4, "e4_tau_0.4", False),
    ("tau", 0.5, "e4_gamma_2", True),
    ("tau", 0.6, "e4_tau_0.6", False),
)

E2_DELTAS: dict[str, dict[tuple[str, ...], Any]] = {
    "e1_dump": {("inference_params", "channel_dump"): True},
    "e2_uniform_weights": {("model", "params", "uniform_weights"): True},
    "e2_no_hier": {("model", "params", "ablate_channels"): ["hier"]},
    "e2_no_sim": {("model", "params", "ablate_channels"): ["sim"]},
    "e2_no_diff": {("model", "params", "ablate_channels"): ["diff"]},
    "e2_no_attr": {("model", "params", "ablate_channels"): ["attr"]},
    "e2_lex_only": {("model", "params", "ablate_channels"): ["hier", "sim", "diff", "attr"]},
}

E4_DELTAS: dict[str, dict[tuple[str, ...], Any]] = {
    "e4_gamma_1": {
        ("model", "params", "gamma"): 1.0,
        ("model", "params", "tau"): 0.5,
    },
    "e4_gamma_2": {
        ("model", "params", "gamma"): 2.0,
        ("model", "params", "tau"): 0.5,
    },
    "e4_gamma_3": {
        ("model", "params", "gamma"): 3.0,
        ("model", "params", "tau"): 0.5,
    },
    "e4_gamma_4": {
        ("model", "params", "gamma"): 4.0,
        ("model", "params", "tau"): 0.5,
    },
    "e4_tau_0.4": {
        ("model", "params", "gamma"): 2.0,
        ("model", "params", "tau"): 0.4,
    },
    "e4_tau_0.6": {
        ("model", "params", "gamma"): 2.0,
        ("model", "params", "tau"): 0.6,
    },
}

E1_E2_RUNS = tuple(run_name for _arm, run_name in E2_ARMS)
E4_RUNS = tuple(dict.fromkeys(run_name for _series, _value, run_name, _reuse in E4_SETTINGS))
ALL_RUNS = E1_E2_RUNS + E4_RUNS

STRUCTURAL_CHANNELS = ("hier", "sim", "diff", "attr")
DUMP_REQUIRED_COLUMNS = {
    "src",
    "tgt",
    "is_correct",
    "src_label_text",
    "tgt_label_text",
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
}
DUMP_NUMERIC_COLUMNS = DUMP_REQUIRED_COLUMNS - {
    "src",
    "tgt",
    "src_label_text",
    "tgt_label_text",
    "llm_invoked",
    "llm_gated",
}
PHYSICALLY_CONSTANT_COLUMNS = {"llm_invoked", "llm_gated"}

FATAL_LOG_PATTERNS = (
    re.compile(r"Traceback \(most recent call last\)", re.IGNORECASE),
    re.compile(r"\bCUDA out of memory\b", re.IGNORECASE),
    re.compile(r"\bAlignment failed\b", re.IGNORECASE),
    re.compile(r"\brun failed\b", re.IGNORECASE),
    re.compile(r"slurmstepd: error:.*\bKilled\b", re.IGNORECASE),
)


@dataclass(frozen=True)
class RunSpec:
    name: str
    wp: str
    requires_dump: bool
    baseline: str
    expected_delta: dict[tuple[str, ...], Any]


@dataclass
class Validation:
    run_name: str
    wp: str
    check: str
    status: str
    detail: str = ""
    path: str = ""

    def as_row(self) -> dict[str, str]:
        return {
            "run_name": self.run_name,
            "wp": self.wp,
            "check": self.check,
            "status": self.status,
            "detail": self.detail,
            "path": self.path,
        }


@dataclass
class RunSummary:
    run_name: str
    wp: str
    run_dir: Path
    checks: list[Validation] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    wall_clock_minutes: float | None = None
    timing_source: str = ""
    fatal_log_marker: str = ""
    status: str = "incomplete"
    failure_reason: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=DEFAULT_RUNS_ROOT,
        help="Directory containing the fixed e1/e2/e4 run directories.",
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=DEFAULT_ANALYSIS_DIR,
        help="Destination for e2_ablation.csv, e4_sensitivity.csv, and review_validation.csv.",
    )
    parser.add_argument(
        "--reference-config",
        type=Path,
        default=REFERENCE_CONFIG,
        help="Published local OMIM--ORDO config used as the exact sparse-config baseline.",
    )
    parser.add_argument(
        "--default-config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Default config used to verify seed 42 and the uniform arm's hierarchy-family count.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Write all tables, then exit nonzero if any actual run is not complete and valid.",
    )
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("top-level YAML value is not a mapping")
    return payload


def _recursive_changes(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    prefix: tuple[str, ...] = (),
) -> dict[tuple[str, ...], Any]:
    changes: dict[tuple[str, ...], Any] = {}
    for key in set(baseline) | set(candidate):
        path = prefix + (str(key),)
        if key not in baseline:
            changes[path] = candidate[key]
        elif key not in candidate:
            changes[path] = {"__removed__": baseline[key]}
        elif isinstance(baseline[key], dict) and isinstance(candidate[key], dict):
            changes.update(_recursive_changes(baseline[key], candidate[key], path))
        elif baseline[key] != candidate[key]:
            changes[path] = candidate[key]
    return changes


def _format_delta(delta: dict[tuple[str, ...], Any]) -> str:
    serializable = {".".join(path): value for path, value in sorted(delta.items())}
    return json.dumps(serializable, sort_keys=True, separators=(",", ":"))


def _expected_run_yaml(run_name: str) -> dict[str, Any]:
    return {
        "dataset": {
            "data_dir": "data/omim-ordo",
            "source": "omim.owl",
            "target": "ordo.owl",
            "full_reference": "test.tsv",
            "candidates": "review300.test.cands.tsv",
        },
        "job": {
            "name": run_name,
            "output_dir": f"exp/test/review_response/{run_name}/",
            "config_file": f"exp/test/review_response/{run_name}/config.yaml",
            "memory": "60G",
            "device": 0,
            "save_logs": True,
            "run_eval": True,
        },
    }


def _artifact_check(spec: RunSpec, path: Path, required: bool = True) -> Validation:
    if not path.exists():
        return Validation(
            spec.name,
            spec.wp,
            f"artifact:{path.name}",
            "missing" if required else "warn",
            "required artifact is absent" if required else "optional artifact is absent",
            str(path),
        )
    if not path.is_file():
        return Validation(
            spec.name,
            spec.wp,
            f"artifact:{path.name}",
            "fail",
            "path exists but is not a regular file",
            str(path),
        )
    if path.stat().st_size == 0:
        return Validation(
            spec.name,
            spec.wp,
            f"artifact:{path.name}",
            "fail",
            "artifact is empty",
            str(path),
        )
    return Validation(
        spec.name,
        spec.wp,
        f"artifact:{path.name}",
        "pass",
        f"{path.stat().st_size} bytes",
        str(path),
    )


def _read_metrics(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV has no header")
        normalized = {name.strip().lower(): name for name in reader.fieldnames}
        if "metric" not in normalized or "value" not in normalized:
            raise ValueError(f"expected Metric,Value columns; found {reader.fieldnames!r}")
        metric_column = normalized["metric"]
        value_column = normalized["value"]
        for row in reader:
            metric = str(row.get(metric_column, "")).strip()
            if not metric:
                continue
            raw_value = row.get(value_column)
            try:
                value = float(raw_value) if raw_value is not None else math.nan
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{metric} has non-numeric value {raw_value!r}") from exc
            if not math.isfinite(value):
                raise ValueError(f"{metric} is not finite: {raw_value!r}")
            values[metric] = value
    missing = [metric for metric in METRICS if metric not in values]
    if missing:
        raise ValueError(f"missing required metrics: {', '.join(missing)}")
    out_of_range = [metric for metric in METRICS if not 0.0 <= values[metric] <= 1.0]
    if out_of_range:
        raise ValueError(f"metrics outside [0,1]: {', '.join(out_of_range)}")
    return {metric: values[metric] for metric in METRICS}


def _read_wall_clock(times_path: Path, log_text: str) -> tuple[float | None, str]:
    if times_path.is_file() and times_path.stat().st_size:
        matches = re.findall(
            r"(?im)^\s*Total:\s*([0-9]+(?:\.[0-9]+)?)\s+minutes?\s*$",
            times_path.read_text(encoding="utf-8", errors="replace"),
        )
        if matches:
            return float(matches[-1]), "times.txt:Total"
    matches = re.findall(
        r"Alignment completed in\s+([0-9]+(?:\.[0-9]+)?)\s+minutes?",
        log_text,
        flags=re.IGNORECASE,
    )
    if matches:
        return float(matches[-1]), "exact.log:Alignment completed"
    return None, ""


def _find_fatal_log_marker(log_text: str) -> str:
    for pattern in FATAL_LOG_PATTERNS:
        match = pattern.search(log_text)
        if match:
            return " ".join(match.group(0).split())
    return ""


def _binary_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    numeric = pd.to_numeric(series, errors="coerce")
    text = series.astype(str).str.strip().str.lower()
    mapped = text.map({"true": 1.0, "false": 0.0, "yes": 1.0, "no": 0.0})
    return numeric.fillna(mapped)


def _all_close(values: pd.Series | np.ndarray, target: Any, atol: float = FLOAT_ATOL) -> bool:
    array = np.asarray(values, dtype=float)
    return bool(
        array.size and np.isfinite(array).all() and np.allclose(array, target, rtol=0.0, atol=atol)
    )


def _dump_checks(
    spec: RunSpec,
    path: Path,
    hierarchy_family_count: int | None,
) -> list[Validation]:
    checks: list[Validation] = []
    try:
        frame = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        return [
            Validation(
                spec.name,
                spec.wp,
                "channel_dump:read",
                "fail",
                f"{type(exc).__name__}: {exc}",
                str(path),
            )
        ]

    missing_columns = sorted(DUMP_REQUIRED_COLUMNS - set(frame.columns))
    if missing_columns:
        checks.append(
            Validation(
                spec.name,
                spec.wp,
                "channel_dump:required_columns",
                "fail",
                f"missing columns: {', '.join(missing_columns)}",
                str(path),
            )
        )
        return checks
    checks.append(
        Validation(
            spec.name,
            spec.wp,
            "channel_dump:required_columns",
            "pass",
            f"all {len(DUMP_REQUIRED_COLUMNS)} required columns present",
            str(path),
        )
    )

    row_count = len(frame)
    if 0 < row_count <= EXPECTED_CANDIDATE_PAIRS:
        row_status = "pass"
    else:
        row_status = "fail"
    checks.append(
        Validation(
            spec.name,
            spec.wp,
            "channel_dump:row_count",
            row_status,
            (
                f"rows={row_count}; input_candidate_pairs={EXPECTED_CANDIDATE_PAIRS}; "
                f"reduction={EXPECTED_CANDIDATE_PAIRS - row_count}"
            ),
            str(path),
        )
    )

    source_count = int(frame["src"].nunique(dropna=True))
    checks.append(
        Validation(
            spec.name,
            spec.wp,
            "channel_dump:source_count",
            "pass" if source_count == EXPECTED_SOURCE_COUNT else "fail",
            (
                f"distinct_src={source_count}; expected={EXPECTED_SOURCE_COUNT}. "
                "SPEC §3 has 300 unique sources and 302 reference rows; WP1's phrase "
                "'302 distinct src values' is internally inconsistent."
            ),
            str(path),
        )
    )

    pair_duplicates = int(frame.duplicated(subset=["src", "tgt"]).sum())
    checks.append(
        Validation(
            spec.name,
            spec.wp,
            "channel_dump:pair_uniqueness",
            "pass" if pair_duplicates == 0 else "warn",
            (
                f"duplicate_(src,tgt)_rows={pair_duplicates}; scored input occurrences are "
                "retained, including repeated candidate pairs"
            ),
            str(path),
        )
    )

    numeric = frame[list(sorted(DUMP_NUMERIC_COLUMNS))].apply(pd.to_numeric, errors="coerce")
    binary = pd.DataFrame(
        {
            "llm_invoked": _binary_series(frame["llm_invoked"]),
            "llm_gated": _binary_series(frame["llm_gated"]),
        }
    )
    invalid_numeric = {
        column: int(series.isna().sum() + (~np.isfinite(series.fillna(0.0))).sum())
        for column, series in numeric.items()
        if series.isna().any() or not np.isfinite(series.fillna(0.0)).all()
    }
    invalid_binary = {}
    for column, series in binary.items():
        invalid_mask = series.isna() | ~series.isin([0.0, 1.0])
        if invalid_mask.any():
            invalid_binary[column] = int(invalid_mask.sum())
    invalid_values = {**invalid_numeric, **invalid_binary}
    checks.append(
        Validation(
            spec.name,
            spec.wp,
            "channel_dump:numeric_population",
            "pass" if not invalid_values else "fail",
            (
                "all required numeric/binary values are finite and populated"
                if not invalid_values
                else f"invalid value counts: {json.dumps(invalid_values, sort_keys=True)}"
            ),
            str(path),
        )
    )
    if invalid_values or frame.empty:
        return checks

    omega_sum = sum(numeric[f"omega_{channel}"] for channel in STRUCTURAL_CHANNELS)
    struct_active = numeric["sigma_struct"] > 1e-8
    expected_omega_sum = (
        np.ones(row_count)
        if spec.name == "e2_uniform_weights"
        else np.where(struct_active, 1.0, 0.0)
    )
    omega_mismatches = int((~np.isclose(omega_sum, expected_omega_sum, rtol=0.0, atol=1e-5)).sum())

    both_active = (numeric["sigma_lex"] > 1e-8) & struct_active
    expected_pair = (1.0 - numeric.loc[both_active, "w_c"]) * numeric.loc[
        both_active, "s_lex"
    ] + numeric.loc[both_active, "w_c"] * numeric.loc[both_active, "S_struct"]
    pair_mismatches = int(
        (
            ~np.isclose(
                numeric.loc[both_active, "S_pair"],
                expected_pair,
                rtol=0.0,
                atol=1e-5,
            )
        ).sum()
    )
    checks.append(
        Validation(
            spec.name,
            spec.wp,
            "channel_dump:mixture_identities",
            "pass" if not omega_mismatches and not pair_mismatches else "fail",
            (
                f"omega_sum_mismatches={omega_mismatches}/{row_count}; "
                f"S_pair_mismatches={pair_mismatches}/{int(both_active.sum())}"
            ),
            str(path),
        )
    )

    if spec.name == "e1_dump":
        constant_columns = sorted(
            column
            for column in DUMP_REQUIRED_COLUMNS
            if frame[column].nunique(dropna=False) <= 1
            and column not in PHYSICALLY_CONSTANT_COLUMNS
        )
        checks.append(
            Validation(
                spec.name,
                spec.wp,
                "channel_dump:unexpected_constant_columns",
                "warn" if constant_columns else "pass",
                (
                    "manual physical-constancy review required for: " + ", ".join(constant_columns)
                    if constant_columns
                    else "no unexpected constant required columns"
                ),
                str(path),
            )
        )

    if spec.name.startswith("e2_no_"):
        channel = spec.name.removeprefix("e2_no_")
        omega = numeric[f"omega_{channel}"]
        identity_ok = _all_close(omega, 0.0)
        checks.append(
            Validation(
                spec.name,
                spec.wp,
                f"ablation_identity:no_{channel}",
                "pass" if identity_ok else "fail",
                (
                    f"omega_{channel}_nonzero_rows="
                    f"{int((omega.abs() > FLOAT_ATOL).sum())}; "
                    f"s_{channel} and q_{channel} populated on all {row_count} rows"
                ),
                str(path),
            )
        )
    elif spec.name == "e2_uniform_weights":
        constant_omegas = all(
            numeric[f"omega_{channel}"].nunique(dropna=False) == 1
            for channel in STRUCTURAL_CHANNELS
        )
        sum_ok = _all_close(omega_sum, 1.0)
        wc_ok = _all_close(numeric["w_c"], 0.5)
        pair_ok = _all_close(
            numeric["S_pair"],
            0.5 * numeric["s_lex"] + 0.5 * numeric["S_struct"],
        )
        expected_detail = ""
        expected_grouped_ok = True
        if hierarchy_family_count is not None and hierarchy_family_count > 0:
            channel_count = hierarchy_family_count + 3
            expected_values = {
                "hier": hierarchy_family_count / channel_count,
                "sim": 1.0 / channel_count,
                "diff": 1.0 / channel_count,
                "attr": 1.0 / channel_count,
            }
            expected_grouped_ok = all(
                _all_close(numeric[f"omega_{channel}"], value)
                for channel, value in expected_values.items()
            )
            expected_detail = (
                f"; |K|={channel_count}, expected_grouped_omegas="
                f"{json.dumps(expected_values, sort_keys=True)}"
            )
        identity_ok = constant_omegas and sum_ok and wc_ok and pair_ok and expected_grouped_ok
        checks.append(
            Validation(
                spec.name,
                spec.wp,
                "ablation_identity:uniform_weights",
                "pass" if identity_ok else "fail",
                (
                    f"constant_omegas={constant_omegas}; omega_sum_one={sum_ok}; "
                    f"w_c_half={wc_ok}; S_pair_plain_mean={pair_ok}; "
                    f"grouped_values_match={expected_grouped_ok}"
                    f"{expected_detail}"
                ),
                str(path),
            )
        )
    elif spec.name == "e2_lex_only":
        sigma_ok = _all_close(numeric["sigma_struct"], 0.0)
        pair_ok = _all_close(numeric["S_pair"], numeric["s_lex"])
        checks.append(
            Validation(
                spec.name,
                spec.wp,
                "ablation_identity:lex_only",
                "pass" if sigma_ok and pair_ok else "fail",
                f"sigma_struct_zero={sigma_ok}; S_pair_equals_s_lex={pair_ok}",
                str(path),
            )
        )

    return checks


def _specs() -> dict[str, RunSpec]:
    specs: dict[str, RunSpec] = {}
    for run_name in E1_E2_RUNS:
        specs[run_name] = RunSpec(
            name=run_name,
            wp="WP1/WP2" if run_name == "e1_dump" else "WP2",
            requires_dump=True,
            baseline="reference" if run_name == "e1_dump" else "e1_dump",
            expected_delta=E2_DELTAS[run_name],
        )
    for run_name in E4_RUNS:
        specs[run_name] = RunSpec(
            name=run_name,
            wp="WP4",
            requires_dump=False,
            baseline="reference",
            expected_delta=E4_DELTAS[run_name],
        )
    return specs


def _inspect_run(
    spec: RunSpec,
    runs_root: Path,
    baselines: dict[str, dict[str, Any] | None],
    hierarchy_family_count: int | None,
) -> RunSummary:
    run_dir = runs_root / spec.name
    summary = RunSummary(spec.name, spec.wp, run_dir)

    config_path = run_dir / "config.yaml"
    run_yaml_path = run_dir / "omim-ordo.yaml"
    evaluation_path = run_dir / "evaluation_results.csv"
    log_path = run_dir / "exact.log"
    times_path = run_dir / "times.txt"
    dump_path = run_dir / "channel_dump.csv"

    required_paths = [config_path, run_yaml_path, evaluation_path, log_path]
    if spec.requires_dump:
        required_paths.append(dump_path)
    summary.checks.extend(_artifact_check(spec, path) for path in required_paths)

    config: dict[str, Any] | None = None
    if config_path.is_file() and config_path.stat().st_size:
        try:
            config = _load_yaml(config_path)
            baseline = baselines.get(spec.baseline)
            if baseline is None:
                summary.checks.append(
                    Validation(
                        spec.name,
                        spec.wp,
                        "config:exact_delta",
                        "fail",
                        f"baseline {spec.baseline!r} could not be loaded",
                        str(config_path),
                    )
                )
            else:
                actual_delta = _recursive_changes(baseline, config)
                exact = actual_delta == spec.expected_delta
                summary.checks.append(
                    Validation(
                        spec.name,
                        spec.wp,
                        "config:exact_delta",
                        "pass" if exact else "fail",
                        (
                            f"expected={_format_delta(spec.expected_delta)}; "
                            f"actual={_format_delta(actual_delta)}"
                        ),
                        str(config_path),
                    )
                )
        except Exception as exc:
            summary.checks.append(
                Validation(
                    spec.name,
                    spec.wp,
                    "config:parse",
                    "fail",
                    f"{type(exc).__name__}: {exc}",
                    str(config_path),
                )
            )

    if run_yaml_path.is_file() and run_yaml_path.stat().st_size:
        try:
            run_yaml = _load_yaml(run_yaml_path)
            expected_run_yaml = _expected_run_yaml(spec.name)
            exact = run_yaml == expected_run_yaml
            summary.checks.append(
                Validation(
                    spec.name,
                    spec.wp,
                    "run_yaml:identity",
                    "pass" if exact else "fail",
                    (
                        "dataset/job paths and flags match the specification"
                        if exact
                        else (
                            f"expected={json.dumps(expected_run_yaml, sort_keys=True)}; "
                            f"actual={json.dumps(run_yaml, sort_keys=True)}"
                        )
                    ),
                    str(run_yaml_path),
                )
            )
        except Exception as exc:
            summary.checks.append(
                Validation(
                    spec.name,
                    spec.wp,
                    "run_yaml:parse",
                    "fail",
                    f"{type(exc).__name__}: {exc}",
                    str(run_yaml_path),
                )
            )

    if evaluation_path.is_file() and evaluation_path.stat().st_size:
        try:
            summary.metrics = _read_metrics(evaluation_path)
            summary.checks.append(
                Validation(
                    spec.name,
                    spec.wp,
                    "evaluation:required_metrics",
                    "pass",
                    json.dumps(summary.metrics, sort_keys=True),
                    str(evaluation_path),
                )
            )
        except Exception as exc:
            summary.checks.append(
                Validation(
                    spec.name,
                    spec.wp,
                    "evaluation:required_metrics",
                    "fail",
                    f"{type(exc).__name__}: {exc}",
                    str(evaluation_path),
                )
            )

    log_text = ""
    if log_path.is_file() and log_path.stat().st_size:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        summary.fatal_log_marker = _find_fatal_log_marker(log_text)
        if summary.fatal_log_marker:
            marker_status = "warn" if summary.metrics else "fail"
            summary.checks.append(
                Validation(
                    spec.name,
                    spec.wp,
                    "log:fatal_marker",
                    marker_status,
                    (
                        f"found {summary.fatal_log_marker!r}; completed metrics also exist"
                        if summary.metrics
                        else f"found {summary.fatal_log_marker!r} and no complete metrics"
                    ),
                    str(log_path),
                )
            )
        else:
            summary.checks.append(
                Validation(
                    spec.name,
                    spec.wp,
                    "log:fatal_marker",
                    "pass",
                    "no recognized fatal marker",
                    str(log_path),
                )
            )

    summary.wall_clock_minutes, summary.timing_source = _read_wall_clock(times_path, log_text)
    summary.checks.append(
        Validation(
            spec.name,
            spec.wp,
            "timing:wall_clock",
            "pass" if summary.wall_clock_minutes is not None else "missing",
            (
                f"{summary.wall_clock_minutes:.10g} minutes from {summary.timing_source}"
                if summary.wall_clock_minutes is not None
                else "no parseable Total in times.txt or completion time in exact.log"
            ),
            str(times_path if times_path.exists() else log_path),
        )
    )

    if spec.requires_dump and dump_path.is_file() and dump_path.stat().st_size:
        summary.checks.extend(_dump_checks(spec, dump_path, hierarchy_family_count))

    blocking = [check for check in summary.checks if check.status in {"missing", "fail"}]
    if summary.fatal_log_marker and not summary.metrics:
        summary.status = "failed"
    elif any(check.status == "missing" for check in blocking):
        summary.status = "incomplete"
    elif blocking:
        summary.status = "invalid"
    else:
        summary.status = "complete"
    summary.failure_reason = "; ".join(f"{check.check}: {check.detail}" for check in blocking)
    summary.checks.append(
        Validation(
            spec.name,
            spec.wp,
            "run:summary",
            summary.status,
            summary.failure_reason or "all required artifacts and identities passed",
            str(run_dir),
        )
    )
    return summary


def _float_or_blank(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return ""
    return f"{value:.10g}"


def _metric_or_blank(summary: RunSummary, metric: str) -> str:
    value = summary.metrics.get(metric)
    return _float_or_blank(value)


def _e2_rows(summaries: dict[str, RunSummary]) -> list[dict[str, str]]:
    full_mrr = summaries["e1_dump"].metrics.get("MRR")
    rows: list[dict[str, str]] = []
    for arm, run_name in E2_ARMS:
        summary = summaries[run_name]
        mrr = summary.metrics.get("MRR")
        delta = mrr - full_mrr if mrr is not None and full_mrr is not None else None
        rows.append(
            {
                "arm": arm,
                "run_name": run_name,
                "status": summary.status,
                "failure_reason": summary.failure_reason,
                "MRR": _metric_or_blank(summary, "MRR"),
                "Hits@1": _metric_or_blank(summary, "Hits@1"),
                "Hits@5": _metric_or_blank(summary, "Hits@5"),
                "Hits@10": _metric_or_blank(summary, "Hits@10"),
                "delta_mrr_vs_full": _float_or_blank(delta),
                "wall_clock_minutes": _float_or_blank(summary.wall_clock_minutes),
                "timing_source": summary.timing_source,
                "run_dir": str(summary.run_dir),
            }
        )
    return rows


def _e4_rows(summaries: dict[str, RunSummary]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for series, setting, run_name, reused in E4_SETTINGS:
        summary = summaries[run_name]
        gamma = setting if series == "gamma" else 2.0
        tau = setting if series == "tau" else 0.5
        rows.append(
            {
                "series": series,
                "setting": _float_or_blank(setting),
                "run_name": run_name,
                "reused_run": "true" if reused else "false",
                "status": summary.status,
                "failure_reason": summary.failure_reason,
                "gamma": _float_or_blank(gamma),
                "tau": _float_or_blank(tau),
                "MRR": _metric_or_blank(summary, "MRR"),
                "Hits@1": _metric_or_blank(summary, "Hits@1"),
                "Hits@5": _metric_or_blank(summary, "Hits@5"),
                "Hits@10": _metric_or_blank(summary, "Hits@10"),
                "wall_clock_minutes": _float_or_blank(summary.wall_clock_minutes),
                "timing_source": summary.timing_source,
                "run_dir": str(summary.run_dir),
            }
        )
    return rows


def _write_csv(path: Path, fieldnames: Iterable[str], rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _shared_validations(
    reference: dict[str, Any] | None,
    default: dict[str, Any] | None,
    runs_root: Path,
    reference_path: Path,
    default_path: Path,
) -> tuple[list[Validation], int | None]:
    checks: list[Validation] = []
    checks.append(
        Validation(
            "_shared",
            "WP1/WP2/WP4",
            "baseline:reference_config",
            "pass" if reference is not None else "fail",
            "reference config loaded" if reference is not None else "reference config unavailable",
            str(reference_path),
        )
    )

    hierarchy_family_count: int | None = None
    if default is not None:
        seed = default.get("seed")
        families = (
            default.get("dataset_params", {}).get("hierarchical_relation_families", {})
            if isinstance(default.get("dataset_params"), dict)
            else {}
        )
        if isinstance(families, dict):
            hierarchy_family_count = len(families)
        checks.append(
            Validation(
                "_shared",
                "WP1/WP2/WP4",
                "default:seed",
                "pass" if seed == 42 else "fail",
                f"seed={seed!r}; expected=42",
                str(default_path),
            )
        )
        checks.append(
            Validation(
                "_shared",
                "WP2",
                "default:hierarchy_family_count",
                (
                    "pass"
                    if hierarchy_family_count is not None and hierarchy_family_count > 0
                    else "fail"
                ),
                f"hierarchy_families={hierarchy_family_count!r}",
                str(default_path),
            )
        )
    else:
        checks.append(
            Validation(
                "_shared",
                "WP1/WP2/WP4",
                "default:config",
                "fail",
                "default config unavailable",
                str(default_path),
            )
        )

    checks.extend(
        [
            Validation(
                "e1_dump",
                "WP2",
                "reuse:full_arm",
                "pass",
                "WP2 full is the e1_dump run; no duplicate e2_full run is required",
                str(runs_root / "e1_dump"),
            ),
            Validation(
                "e4_gamma_2",
                "WP4",
                "reuse:gamma_2_tau_0.5",
                "pass",
                (
                    "gamma=2,tau=0.5 is one run represented in both sensitivity series; "
                    "e4_tau_0.5 is not an independent result"
                ),
                str(runs_root / "e4_gamma_2"),
            ),
        ]
    )
    duplicate_tau_dir = runs_root / "e4_tau_0.5"
    checks.append(
        Validation(
            "e4_gamma_2",
            "WP4",
            "reuse:no_duplicate_tau_run",
            "warn" if duplicate_tau_dir.exists() else "pass",
            (
                "unexpected e4_tau_0.5 path exists; do not count it as an independent run"
                if duplicate_tau_dir.exists()
                else "no duplicate e4_tau_0.5 run directory"
            ),
            str(duplicate_tau_dir),
        )
    )
    return checks, hierarchy_family_count


def main() -> None:
    args = parse_args()
    runs_root = args.runs_root
    analysis_dir = args.analysis_dir

    try:
        reference = _load_yaml(args.reference_config)
    except Exception as exc:
        print(f"[WARN] Could not load reference config {args.reference_config}: {exc}")
        reference = None
    try:
        default = _load_yaml(args.default_config)
    except Exception as exc:
        print(f"[WARN] Could not load default config {args.default_config}: {exc}")
        default = None

    shared_checks, hierarchy_family_count = _shared_validations(
        reference,
        default,
        runs_root,
        args.reference_config,
        args.default_config,
    )

    e1_config_path = runs_root / "e1_dump" / "config.yaml"
    try:
        e1_config = _load_yaml(e1_config_path)
    except Exception:
        e1_config = None
    baselines = {"reference": reference, "e1_dump": e1_config}

    specs = _specs()
    summaries = {
        run_name: _inspect_run(
            specs[run_name],
            runs_root,
            baselines,
            hierarchy_family_count,
        )
        for run_name in ALL_RUNS
    }

    e2_path = analysis_dir / "e2_ablation.csv"
    e4_path = analysis_dir / "e4_sensitivity.csv"
    validation_path = analysis_dir / "review_validation.csv"

    e2_rows = _e2_rows(summaries)
    e4_rows = _e4_rows(summaries)
    validations = shared_checks + [
        check for run_name in ALL_RUNS for check in summaries[run_name].checks
    ]

    _write_csv(
        e2_path,
        (
            "arm",
            "run_name",
            "status",
            "failure_reason",
            "MRR",
            "Hits@1",
            "Hits@5",
            "Hits@10",
            "delta_mrr_vs_full",
            "wall_clock_minutes",
            "timing_source",
            "run_dir",
        ),
        e2_rows,
    )
    _write_csv(
        e4_path,
        (
            "series",
            "setting",
            "run_name",
            "reused_run",
            "status",
            "failure_reason",
            "gamma",
            "tau",
            "MRR",
            "Hits@1",
            "Hits@5",
            "Hits@10",
            "wall_clock_minutes",
            "timing_source",
            "run_dir",
        ),
        e4_rows,
    )
    _write_csv(
        validation_path,
        ("run_name", "wp", "check", "status", "detail", "path"),
        (validation.as_row() for validation in validations),
    )

    status_counts: dict[str, int] = {}
    for summary in summaries.values():
        status_counts[summary.status] = status_counts.get(summary.status, 0) + 1
    print(
        f"Wrote {e2_path} ({len(e2_rows)} arms), "
        f"{e4_path} ({len(e4_rows)} settings), and "
        f"{validation_path} ({len(validations)} checks)."
    )
    print(f"Actual-run status counts: {json.dumps(status_counts, sort_keys=True)}")

    shared_failure = any(check.status in {"fail", "missing"} for check in shared_checks)
    if args.strict and (
        shared_failure or any(summary.status != "complete" for summary in summaries.values())
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
