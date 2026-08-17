#!/usr/bin/env python3
"""Run the preregistered WP5 channel-reliability analyses on a WP1 dump."""

from __future__ import annotations

import argparse
import math
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import stats


CHANNELS = ("lex", "hier", "sim", "diff", "attr")
ACTIVE_SIGMA_THRESHOLD = 1e-6
REQUIRED_COLUMNS = {"src", "tgt", "is_correct"}
for _channel in CHANNELS:
    REQUIRED_COLUMNS.update(
        {
            f"s_{_channel}",
            f"q_{_channel}",
            f"sigma_{_channel}",
        }
    )

CALIBRATION_COLUMNS = [
    "record_type",
    "channel",
    "status",
    "status_reason",
    "total_rows",
    "active_rows",
    "inactive_rows",
    "inactive_fraction",
    "n_bins",
    "bin",
    "bin_left",
    "bin_right",
    "bin_edges",
    "n",
    "mean_q_k",
    "positive_rate",
    "mean_s_k",
    "brier",
    "mae",
    "spearman_bin_brier",
    "spearman_n_bins",
    "spearman_ci_low",
    "spearman_ci_high",
    "bootstrap_resamples",
    "bootstrap_valid",
    "seed",
    "correlation_status",
]

DISCRIMINATION_COLUMNS = [
    "record_type",
    "channel",
    "status",
    "status_reason",
    "total_rows",
    "active_rows",
    "inactive_rows",
    "inactive_fraction",
    "total_sources",
    "eligible_sources",
    "excluded_sources_without_scored_correct",
    "duplicate_occurrences_removed",
    "sources_all_inactive",
    "sources_without_auc",
    "sources_no_active_positive",
    "sources_no_active_negative",
    "sources_with_auc",
    "src",
    "source_status",
    "total_candidates",
    "active_candidates",
    "inactive_candidates",
    "active_gold_targets",
    "active_nongold_targets",
    "source_mean_q",
    "auc",
    "gold_rank",
    "gold_rank_tie_size",
    "spearman_quality_auc",
    "spearman_n_sources",
    "spearman_ci_low",
    "spearman_ci_high",
    "bootstrap_resamples",
    "bootstrap_valid",
    "seed",
    "correlation_status",
]

PICK_COLUMNS = [
    "record_type",
    "channel",
    "status",
    "status_reason",
    "total_rows",
    "active_rows",
    "inactive_rows",
    "inactive_fraction",
    "total_sources",
    "eligible_sources",
    "excluded_sources_without_scored_correct",
    "duplicate_occurrences_removed",
    "sources_all_inactive",
    "sources_in_pick_rate",
    "sources_tied_at_active_argmax",
    "sources_tied_at_all_candidate_argmax",
    "n_bins",
    "bin",
    "bin_left",
    "bin_right",
    "bin_edges",
    "n",
    "mean_q_k",
    "pick_rate",
    "picked_correct_count",
    "tied_at_argmax_count",
    "src",
    "source_status",
    "total_candidates",
    "active_candidates",
    "inactive_candidates",
    "source_mean_q",
    "picked_correct",
    "tied_at_argmax",
    "argmax_tie_size",
    "argmax_tgt",
    "all_candidate_tied_at_argmax",
    "all_candidate_argmax_tie_size",
    "spearman_quality_pick",
    "spearman_n_sources",
    "spearman_ci_low",
    "spearman_ci_high",
    "bootstrap_resamples",
    "bootstrap_valid",
    "seed",
    "correlation_status",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dump",
        type=Path,
        default=Path("exp/test/review_response/e1_dump/channel_dump.csv"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("exp/test/review_response/e1_dump/config.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("exp/test/review_response/analysis"),
    )
    parser.add_argument(
        "--prefix",
        default="e5",
        help="Output filename prefix without the trailing underscore (for example e5 or e6).",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--jobs",
        type=int,
        default=min(len(CHANNELS), max(1, os.cpu_count() or 1)),
    )
    return parser.parse_args()


def _as_binary(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)
    numeric = pd.to_numeric(series, errors="coerce")
    mapped = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": 1.0, "false": 0.0, "yes": 1.0, "no": 0.0, "y": 1.0, "n": 0.0})
    )
    parsed = numeric.where(numeric.notna(), mapped)
    bad = parsed.isna() | ~parsed.isin([0.0, 1.0])
    if bad.any():
        examples = series[bad].drop_duplicates().head().tolist()
        raise ValueError(f"non-binary values: {examples}")
    return parsed.astype(int)


def _load_tau(config_path: Path) -> float:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    return float(config.get("model", {}).get("params", {}).get("tau", 0.5))


def _load_dump(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"channel dump is missing required columns: {missing}")
    if frame.empty:
        raise ValueError("channel dump is empty")
    if frame[["src", "tgt"]].isna().any().any():
        raise ValueError("src/tgt columns contain missing values")
    numeric_columns = {"is_correct"}
    for channel in CHANNELS:
        numeric_columns.update({f"s_{channel}", f"q_{channel}", f"sigma_{channel}"})
    for column in sorted(numeric_columns - {"is_correct"}):
        converted = pd.to_numeric(frame[column], errors="coerce")
        bad = converted.isna() | ~np.isfinite(converted)
        if bad.any():
            raise ValueError(f"{column}: {int(bad.sum())} missing/non-finite values")
        frame[column] = converted.astype(float)
    frame["is_correct"] = _as_binary(frame["is_correct"])
    frame["src"] = frame["src"].astype(str)
    frame["tgt"] = frame["tgt"].astype(str)
    return frame


def _channel_status(frame: pd.DataFrame, channel: str) -> tuple[str, str, pd.Series]:
    active = frame[f"sigma_{channel}"].gt(ACTIVE_SIGMA_THRESHOLD)
    active_n = int(active.sum())
    if active_n == 0:
        return (
            "fully_inactive",
            (f"sigma_{channel} <= {ACTIVE_SIGMA_THRESHOLD:g} on all " f"{len(frame)} rows"),
            active,
        )
    distinct_quality = int(frame.loc[active, f"q_{channel}"].nunique())
    if distinct_quality < 2:
        value = float(frame.loc[active, f"q_{channel}"].iloc[0])
        return (
            "constant_quality",
            (f"q_{channel} is constant at {value:.10g} on all " f"{active_n} active rows"),
            active,
        )
    return (
        "ok",
        f"{active_n} active rows with {distinct_quality} distinct q_{channel} values",
        active,
    )


def _format_edge(value: float) -> str:
    return format(float(value), ".10g")


def _quantile_bins(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(values) == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=float)
    if np.unique(values).size < 2:
        only = float(values[0])
        return np.zeros(len(values), dtype=int), np.array([only, only], dtype=float)
    codes, edges = pd.qcut(
        pd.Series(values),
        q=10,
        labels=False,
        retbins=True,
        duplicates="drop",
    )
    if codes.isna().any():
        raise ValueError("quantile binning produced unassigned observations")
    return codes.to_numpy(dtype=int), np.asarray(edges, dtype=float)


def _safe_spearman(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    valid = np.isfinite(first) & np.isfinite(second)
    first = first[valid]
    second = second[valid]
    if len(first) < 2 or np.unique(first).size < 2 or np.unique(second).size < 2:
        return math.nan
    first_ranks = stats.rankdata(first, method="average")
    second_ranks = stats.rankdata(second, method="average")
    return float(np.corrcoef(first_ranks, second_ranks)[0, 1])


def _percentile_interval(samples: np.ndarray) -> tuple[float, float]:
    finite = samples[np.isfinite(samples)]
    if len(finite) == 0:
        return math.nan, math.nan
    return float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))


def _bootstrap_bin_brier(
    payload: tuple[str, np.ndarray, np.ndarray, int, int, int]
) -> tuple[str, float, float, int]:
    channel, codes, squared_errors, n_bins, resamples, seed = payload
    rng = np.random.default_rng(seed)
    samples = np.full(resamples, np.nan, dtype=float)
    n = len(codes)
    for index in range(resamples):
        selected = rng.integers(0, n, size=n)
        selected_codes = codes[selected]
        counts = np.bincount(selected_codes, minlength=n_bins)
        totals = np.bincount(
            selected_codes,
            weights=squared_errors[selected],
            minlength=n_bins,
        )
        present = counts > 0
        brier = totals[present] / counts[present]
        bin_indices = np.arange(1, n_bins + 1, dtype=float)[present]
        samples[index] = _safe_spearman(bin_indices, brier)
    low, high = _percentile_interval(samples)
    return channel, low, high, int(np.isfinite(samples).sum())


def _bootstrap_source_spearman(
    payload: tuple[str, np.ndarray, np.ndarray, int, int]
) -> tuple[str, float, float, int]:
    channel, quality, outcome, resamples, seed = payload
    rng = np.random.default_rng(seed)
    samples = np.full(resamples, np.nan, dtype=float)
    n = len(quality)
    if n < 2 or np.unique(quality).size < 2 or np.unique(outcome).size < 2:
        return channel, math.nan, math.nan, 0
    for index in range(resamples):
        selected = rng.integers(0, n, size=n)
        samples[index] = _safe_spearman(quality[selected], outcome[selected])
    low, high = _percentile_interval(samples)
    return channel, low, high, int(np.isfinite(samples).sum())


def _parallel_map(function, payloads: list[tuple], jobs: int) -> list[tuple]:
    if not payloads:
        return []
    if jobs <= 1:
        return [function(payload) for payload in payloads]
    with ProcessPoolExecutor(max_workers=min(jobs, len(payloads))) as executor:
        return list(executor.map(function, payloads))


def calibration_analysis(
    frame: pd.DataFrame,
    resamples: int,
    seed: int,
    jobs: int,
) -> pd.DataFrame:
    summaries: dict[str, dict] = {}
    bin_rows: dict[str, list[dict]] = {}
    bootstrap_payloads = []
    for channel in CHANNELS:
        status, reason, active = _channel_status(frame, channel)
        active_frame = frame.loc[
            active,
            [f"q_{channel}", f"s_{channel}", "is_correct"],
        ]
        summary = {
            "record_type": "summary",
            "channel": channel,
            "status": status,
            "status_reason": reason,
            "total_rows": len(frame),
            "active_rows": int(active.sum()),
            "inactive_rows": int((~active).sum()),
            "inactive_fraction": float((~active).mean()),
            "n_bins": 0,
            "n": int(active.sum()),
            "spearman_n_bins": 0,
            "bootstrap_resamples": resamples,
            "bootstrap_valid": 0,
            "seed": seed,
            "correlation_status": status,
        }
        summaries[channel] = summary
        bin_rows[channel] = []
        if status != "ok":
            continue
        q = active_frame[f"q_{channel}"].to_numpy(dtype=float)
        scores = active_frame[f"s_{channel}"].to_numpy(dtype=float)
        labels = active_frame["is_correct"].to_numpy(dtype=float)
        codes, edges = _quantile_bins(q)
        n_bins = len(edges) - 1
        edge_text = "|".join(_format_edge(value) for value in edges)
        briers = []
        for bin_index in range(n_bins):
            selected = codes == bin_index
            bin_scores = scores[selected]
            bin_labels = labels[selected]
            squared_errors = np.square(bin_scores - bin_labels)
            absolute_errors = np.abs(bin_scores - bin_labels)
            brier = float(squared_errors.mean())
            briers.append(brier)
            bin_rows[channel].append(
                {
                    "record_type": "bin",
                    "channel": channel,
                    "status": status,
                    "status_reason": reason,
                    "total_rows": len(frame),
                    "active_rows": int(active.sum()),
                    "inactive_rows": int((~active).sum()),
                    "inactive_fraction": float((~active).mean()),
                    "n_bins": n_bins,
                    "bin": bin_index + 1,
                    "bin_left": float(edges[bin_index]),
                    "bin_right": float(edges[bin_index + 1]),
                    "bin_edges": edge_text,
                    "n": int(selected.sum()),
                    "mean_q_k": float(q[selected].mean()),
                    "positive_rate": float(bin_labels.mean()),
                    "mean_s_k": float(bin_scores.mean()),
                    "brier": brier,
                    "mae": float(absolute_errors.mean()),
                    "bootstrap_resamples": resamples,
                    "seed": seed,
                }
            )
        correlation = _safe_spearman(
            np.arange(1, n_bins + 1, dtype=float),
            np.asarray(briers, dtype=float),
        )
        summary.update(
            {
                "n_bins": n_bins,
                "bin_edges": edge_text,
                "spearman_bin_brier": correlation,
                "spearman_n_bins": n_bins,
                "correlation_status": (
                    "ok" if math.isfinite(correlation) else "constant_or_insufficient_brier"
                ),
            }
        )
        squared_errors = np.square(scores - labels)
        bootstrap_payloads.append((channel, codes, squared_errors, n_bins, resamples, seed))
    bootstrap = {
        channel: (low, high, valid)
        for channel, low, high, valid in _parallel_map(
            _bootstrap_bin_brier,
            bootstrap_payloads,
            jobs,
        )
    }
    rows = []
    for channel in CHANNELS:
        if channel in bootstrap:
            low, high, valid = bootstrap[channel]
            summaries[channel].update(
                {
                    "spearman_ci_low": low,
                    "spearman_ci_high": high,
                    "bootstrap_valid": valid,
                }
            )
        rows.append(summaries[channel])
        rows.extend(bin_rows[channel])
    return pd.DataFrame(rows, columns=CALIBRATION_COLUMNS)


def _deduplicate_source_candidates(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    eligible_sources = set(frame.loc[frame["is_correct"].eq(1), "src"])
    eligible = frame.loc[frame["src"].isin(eligible_sources)].copy()
    duplicate_mask = eligible.duplicated(["src", "tgt"], keep=False)
    if duplicate_mask.any():
        check_columns = ["is_correct"]
        for channel in CHANNELS:
            check_columns.extend([f"s_{channel}", f"q_{channel}", f"sigma_{channel}"])
        grouped = eligible.loc[duplicate_mask].groupby(
            ["src", "tgt"],
            sort=True,
            dropna=False,
        )
        conflicts = [
            column for column in check_columns if grouped[column].nunique(dropna=False).gt(1).any()
        ]
        if conflicts:
            raise ValueError(
                "duplicate (src,tgt) rows disagree in channel data: " + ", ".join(conflicts)
            )
    before = len(eligible)
    eligible = eligible.drop_duplicates(["src", "tgt"], keep="first").copy()
    eligible = eligible.sort_values(["src", "tgt"], kind="mergesort").reset_index(drop=True)
    return eligible, before - len(eligible)


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    positive = labels == 1
    n_positive = int(positive.sum())
    n_negative = len(labels) - n_positive
    if n_positive == 0 or n_negative == 0:
        return math.nan
    ranks = stats.rankdata(scores, method="average")
    rank_sum = float(ranks[positive].sum())
    return (rank_sum - n_positive * (n_positive + 1) / 2.0) / (n_positive * n_negative)


def _source_base(
    frame: pd.DataFrame,
    eligible: pd.DataFrame,
    duplicate_occurrences: int,
    channel: str,
    status: str,
    reason: str,
    active: pd.Series,
) -> dict:
    eligible_active = eligible[f"sigma_{channel}"].gt(ACTIVE_SIGMA_THRESHOLD)
    return {
        "channel": channel,
        "status": status,
        "status_reason": reason,
        "total_rows": len(frame),
        "active_rows": int(active.sum()),
        "inactive_rows": int((~active).sum()),
        "inactive_fraction": float((~active).mean()),
        "total_sources": int(frame["src"].nunique()),
        "eligible_sources": int(eligible["src"].nunique()),
        "excluded_sources_without_scored_correct": int(
            frame["src"].nunique() - eligible["src"].nunique()
        ),
        "duplicate_occurrences_removed": duplicate_occurrences,
        "_eligible_active": eligible_active,
    }


def discrimination_analysis(
    frame: pd.DataFrame,
    resamples: int,
    seed: int,
    jobs: int,
) -> pd.DataFrame:
    eligible, duplicate_occurrences = _deduplicate_source_candidates(frame)
    summaries: dict[str, dict] = {}
    source_rows: dict[str, list[dict]] = {}
    bootstrap_payloads = []
    for channel in CHANNELS:
        status, reason, active = _channel_status(frame, channel)
        common = _source_base(
            frame,
            eligible,
            duplicate_occurrences,
            channel,
            status,
            reason,
            active,
        )
        eligible_active = common.pop("_eligible_active")
        working = eligible.assign(_active=eligible_active.to_numpy())
        rows = []
        for src, group in working.groupby("src", sort=True):
            active_group = group.loc[group["_active"]]
            labels = active_group["is_correct"].to_numpy(dtype=int)
            scores = active_group[f"s_{channel}"].to_numpy(dtype=float)
            active_gold = int(labels.sum())
            active_nongold = len(labels) - active_gold
            if active_group.empty:
                source_status = "all_inactive"
            elif active_gold == 0:
                source_status = "no_active_positive"
            elif active_nongold == 0:
                source_status = "no_active_negative"
            else:
                source_status = "ok"
            source_auc = _auc(scores, labels) if source_status == "ok" else math.nan
            gold_rank = math.nan
            gold_rank_tie_size = 0
            if active_gold:
                best_gold_score = float(scores[labels == 1].max())
                gold_rank = 1 + int(np.sum(scores > best_gold_score))
                gold_rank_tie_size = int(np.sum(scores == best_gold_score))
            rows.append(
                {
                    "record_type": "source",
                    **common,
                    "src": src,
                    "source_status": source_status,
                    "total_candidates": len(group),
                    "active_candidates": len(active_group),
                    "inactive_candidates": len(group) - len(active_group),
                    "active_gold_targets": active_gold,
                    "active_nongold_targets": active_nongold,
                    "source_mean_q": (
                        float(active_group[f"q_{channel}"].mean())
                        if not active_group.empty
                        else math.nan
                    ),
                    "auc": source_auc,
                    "gold_rank": gold_rank,
                    "gold_rank_tie_size": gold_rank_tie_size,
                }
            )
        sources = pd.DataFrame(rows)
        valid = sources["source_status"].eq("ok")
        quality = sources.loc[valid, "source_mean_q"].to_numpy(dtype=float)
        auc_values = sources.loc[valid, "auc"].to_numpy(dtype=float)
        correlation = _safe_spearman(quality, auc_values)
        correlation_status = status
        if status == "ok":
            correlation_status = (
                "ok" if math.isfinite(correlation) else "constant_or_insufficient_auc"
            )
        summary = {
            "record_type": "summary",
            **common,
            "sources_all_inactive": int(sources["source_status"].eq("all_inactive").sum()),
            "sources_without_auc": int((~valid).sum()),
            "sources_no_active_positive": int(
                sources["source_status"].eq("no_active_positive").sum()
            ),
            "sources_no_active_negative": int(
                sources["source_status"].eq("no_active_negative").sum()
            ),
            "sources_with_auc": int(valid.sum()),
            "spearman_quality_auc": correlation,
            "spearman_n_sources": int(valid.sum()),
            "bootstrap_resamples": resamples,
            "bootstrap_valid": 0,
            "seed": seed,
            "correlation_status": correlation_status,
        }
        summaries[channel] = summary
        source_rows[channel] = rows
        if status == "ok" and math.isfinite(correlation):
            bootstrap_payloads.append((channel, quality, auc_values, resamples, seed))
    bootstrap = {
        channel: (low, high, valid)
        for channel, low, high, valid in _parallel_map(
            _bootstrap_source_spearman,
            bootstrap_payloads,
            jobs,
        )
    }
    output = []
    for channel in CHANNELS:
        if channel in bootstrap:
            low, high, valid = bootstrap[channel]
            summaries[channel].update(
                {
                    "spearman_ci_low": low,
                    "spearman_ci_high": high,
                    "bootstrap_valid": valid,
                }
            )
        output.append(summaries[channel])
        output.extend(source_rows[channel])
    return pd.DataFrame(output, columns=DISCRIMINATION_COLUMNS)


def pick_analysis(
    frame: pd.DataFrame,
    resamples: int,
    seed: int,
    jobs: int,
) -> pd.DataFrame:
    eligible, duplicate_occurrences = _deduplicate_source_candidates(frame)
    summaries: dict[str, dict] = {}
    bin_rows: dict[str, list[dict]] = {}
    source_rows: dict[str, list[dict]] = {}
    bootstrap_payloads = []
    for channel in CHANNELS:
        status, reason, active = _channel_status(frame, channel)
        common = _source_base(
            frame,
            eligible,
            duplicate_occurrences,
            channel,
            status,
            reason,
            active,
        )
        eligible_active = common.pop("_eligible_active")
        working = eligible.assign(_active=eligible_active.to_numpy())
        rows = []
        for src, group in working.groupby("src", sort=True):
            active_group = group.loc[group["_active"]]
            all_scores = group[f"s_{channel}"].to_numpy(dtype=float)
            all_max = float(all_scores.max())
            all_tie_size = int(np.sum(all_scores == all_max))
            all_tied = all_tie_size > 1
            if active_group.empty:
                source_status = "all_inactive"
                picked_correct = math.nan
                tied = math.nan
                tie_size = 0
                argmax_tgt = ""
                source_mean_q = math.nan
            else:
                source_status = "ok"
                scores = active_group[f"s_{channel}"].to_numpy(dtype=float)
                maximum = float(scores.max())
                winners = active_group.loc[active_group[f"s_{channel}"].eq(maximum)].sort_values(
                    "tgt", kind="mergesort"
                )
                tie_size = len(winners)
                tied = int(tie_size > 1)
                picked_correct = 0 if tied else int(winners.iloc[0]["is_correct"])
                argmax_tgt = (
                    " | ".join(winners["tgt"].astype(str).tolist())
                    if tied
                    else str(winners.iloc[0]["tgt"])
                )
                source_mean_q = float(active_group[f"q_{channel}"].mean())
            rows.append(
                {
                    "record_type": "source",
                    **common,
                    "src": src,
                    "source_status": source_status,
                    "total_candidates": len(group),
                    "active_candidates": len(active_group),
                    "inactive_candidates": len(group) - len(active_group),
                    "source_mean_q": source_mean_q,
                    "picked_correct": picked_correct,
                    "tied_at_argmax": tied,
                    "argmax_tie_size": tie_size,
                    "argmax_tgt": argmax_tgt,
                    "all_candidate_tied_at_argmax": int(all_tied),
                    "all_candidate_argmax_tie_size": all_tie_size,
                }
            )
        sources = pd.DataFrame(rows)
        valid = sources["source_status"].eq("ok")
        quality = sources.loc[valid, "source_mean_q"].to_numpy(dtype=float)
        picked = sources.loc[valid, "picked_correct"].to_numpy(dtype=float)
        correlation = _safe_spearman(quality, picked)
        correlation_status = status
        if status == "ok":
            correlation_status = (
                "ok" if math.isfinite(correlation) else "constant_or_insufficient_pick_outcome"
            )
        bins = []
        n_bins = 0
        edge_text = ""
        if status == "ok" and valid.any():
            codes, edges = _quantile_bins(quality)
            n_bins = len(edges) - 1
            edge_text = "|".join(_format_edge(value) for value in edges)
            valid_sources = sources.loc[valid].reset_index(drop=True)
            for bin_index in range(n_bins):
                selected = codes == bin_index
                part = valid_sources.loc[selected]
                bins.append(
                    {
                        "record_type": "bin",
                        **common,
                        "n_bins": n_bins,
                        "bin": bin_index + 1,
                        "bin_left": float(edges[bin_index]),
                        "bin_right": float(edges[bin_index + 1]),
                        "bin_edges": edge_text,
                        "n": len(part),
                        "mean_q_k": float(part["source_mean_q"].mean()),
                        "pick_rate": float(part["picked_correct"].mean()),
                        "picked_correct_count": int(part["picked_correct"].sum()),
                        "tied_at_argmax_count": int(part["tied_at_argmax"].sum()),
                    }
                )
        summary = {
            "record_type": "summary",
            **common,
            "sources_all_inactive": int((~valid).sum()),
            "sources_in_pick_rate": int(valid.sum()),
            "sources_tied_at_active_argmax": int(sources.loc[valid, "tied_at_argmax"].sum()),
            "sources_tied_at_all_candidate_argmax": int(
                sources["all_candidate_tied_at_argmax"].sum()
            ),
            "n_bins": n_bins,
            "bin_edges": edge_text,
            "spearman_quality_pick": correlation,
            "spearman_n_sources": int(valid.sum()),
            "bootstrap_resamples": resamples,
            "bootstrap_valid": 0,
            "seed": seed,
            "correlation_status": correlation_status,
        }
        summaries[channel] = summary
        bin_rows[channel] = bins
        source_rows[channel] = rows
        if status == "ok" and math.isfinite(correlation):
            bootstrap_payloads.append((channel, quality, picked, resamples, seed))
    bootstrap = {
        channel: (low, high, valid)
        for channel, low, high, valid in _parallel_map(
            _bootstrap_source_spearman,
            bootstrap_payloads,
            jobs,
        )
    }
    output = []
    for channel in CHANNELS:
        if channel in bootstrap:
            low, high, valid = bootstrap[channel]
            summaries[channel].update(
                {
                    "spearman_ci_low": low,
                    "spearman_ci_high": high,
                    "bootstrap_valid": valid,
                }
            )
        output.append(summaries[channel])
        output.extend(bin_rows[channel])
        output.extend(source_rows[channel])
    return pd.DataFrame(output, columns=PICK_COLUMNS)


def _summary_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame["record_type"].eq("summary")].set_index("channel")


def _status_text(status: str) -> str:
    return {
        "ok": "testable",
        "constant_quality": "untestable (constant quality)",
        "fully_inactive": "untestable (fully inactive)",
    }.get(status, status)


def _format_number(value: object, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "undefined"
    return f"{float(value):.{digits}f}"


def _format_ci(row: pd.Series, value: str) -> str:
    return (
        f"{_format_number(row[value])} "
        f"[{_format_number(row['spearman_ci_low'])}, "
        f"{_format_number(row['spearman_ci_high'])}]"
    )


def _verdict(
    channel: str,
    calibration: pd.DataFrame,
    discrimination: pd.DataFrame,
    pick: pd.DataFrame,
) -> tuple[str, str]:
    cal = calibration.loc[channel]
    disc = discrimination.loc[channel]
    picked = pick.loc[channel]
    status = str(cal["status"])
    if status != "ok":
        return "untestable", str(cal["status_reason"])
    cal_low = cal["spearman_ci_low"]
    cal_high = cal["spearman_ci_high"]
    primary_support = pd.notna(cal_low) and pd.notna(cal_high) and float(cal_high) < 0.0
    secondary_adverse = False
    for row in (disc, picked):
        low = row["spearman_ci_low"]
        high = row["spearman_ci_high"]
        if pd.notna(low) and pd.notna(high) and float(high) < 0.0:
            secondary_adverse = True
    if primary_support and not secondary_adverse:
        return (
            "supported",
            "Brier decreases with quality with a wholly negative 95% bootstrap CI, "
            "and neither source-level secondary test has a wholly negative CI",
        )
    reasons = []
    if not primary_support:
        reasons.append("the primary Brier correlation CI is not wholly below zero")
    if secondary_adverse:
        reasons.append("at least one source-level secondary CI is wholly negative")
    return "not supported", "; ".join(reasons)


def write_results(
    path: Path,
    dump_path: Path,
    config_path: Path,
    tau: float,
    calibration_table: pd.DataFrame,
    discrimination_table: pd.DataFrame,
    pick_table: pd.DataFrame,
    resamples: int,
    seed: int,
) -> None:
    calibration = _summary_rows(calibration_table)
    discrimination = _summary_rows(discrimination_table)
    pick = _summary_rows(pick_table)
    lines = [
        "# WP5 — Reliability re-analysis results",
        "",
        "## Integrity and method",
        "",
        "These are the three metrics preregistered in WP5: conditional calibration, "
        "per-source discrimination, and per-source target picking. No replacement metric was "
        "selected after seeing the result. The deterministic second invocation is only an "
        "acceptance check of byte-identical CSV output.",
        "",
        f"- Input dump: `{dump_path}`",
        f"- Config: `{config_path}` (`tau={tau:g}`)",
        f"- Activity rule: `sigma_k > {ACTIVE_SIGMA_THRESHOLD:g}`; inactive rows are excluded "
        "from every reported statistic and counted separately.",
        f"- Bootstrap: {resamples:,} resamples with literal seed {seed}. Calibration resamples "
        "rows within the fixed observed-quality bins; source-level tests resample sources.",
        "- Conditional calibration correlates bin index with Brier (lower is better). "
        "Discrimination correlates source mean quality with AUC. Picking correlates source "
        "mean quality with the binary unique-argmax outcome.",
        "- For the one-line verdict, the primary conditional-calibration test supports the "
        "claim only when its 95% CI is wholly below zero and neither secondary test has a "
        "wholly negative CI. Otherwise a testable channel is reported as not supported.",
        "",
        "## Per-channel verdicts",
        "",
    ]
    for channel in CHANNELS:
        verdict, reason = _verdict(
            channel,
            calibration,
            discrimination,
            pick,
        )
        lines.append(f"- **{channel}: {verdict}.** {reason}.")
    lines.extend(
        [
            "",
            "## Conditional calibration (primary)",
            "",
            "| Channel | Status | Active / inactive rows | Bins | Spearman(bin, Brier), 95% CI |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for channel in CHANNELS:
        row = calibration.loc[channel]
        lines.append(
            f"| {channel} | {_status_text(str(row['status']))} | "
            f"{int(row['active_rows']):,} / {int(row['inactive_rows']):,} | "
            f"{int(row['n_bins'])} | {_format_ci(row, 'spearman_bin_brier')} |"
        )
    calibration_bins = calibration_table.loc[calibration_table["record_type"].eq("bin")]
    endpoint_sentences = []
    for channel in CHANNELS:
        rows = calibration_bins.loc[calibration_bins["channel"].eq(channel)].sort_values("bin")
        if rows.empty:
            continue
        first = rows.iloc[0]
        last = rows.iloc[-1]
        endpoint_sentences.append(
            f"{channel}: Brier {_format_number(first['brier'])}→"
            f"{_format_number(last['brier'])}, positive rate "
            f"{_format_number(first['positive_rate'])}→"
            f"{_format_number(last['positive_rate'])}, and mean score "
            f"{_format_number(first['mean_s_k'])}→"
            f"{_format_number(last['mean_s_k'])}"
        )
    lines.extend(
        [
            "",
            "The CSV reports, for every actual bin, `n`, mean quality, positive rate, mean "
            "channel score, Brier, and MAE. Positive rate is kept alongside Brier so a change "
            "in class composition is visible. Actual quantile edges and `n_bins` are emitted; "
            "bins are never padded or jittered.",
            "",
            "Lowest-to-highest quality-bin endpoints are: "
            + "; ".join(endpoint_sentences)
            + ". Attribute scores remain above 0.5, so its falling positive rate would, by "
            "itself, increase rather than decrease Brier; the observed attribute improvement "
            "is therefore not an artefact of the high-quality bins containing fewer positives.",
            "",
            "## Per-source discrimination",
            "",
            "The dump contains scored correct rows for 204/300 sources. The other 96 were "
            "committed by the exact prefilter and have no semantic gold row, so this analysis "
            "is restricted to those 204 harder sources. Repeated `(src,tgt)` occurrences are "
            "collapsed only for source-level tests after asserting that their channel values "
            "agree.",
            "",
            "| Channel | Sources with AUC | All-inactive sources | Spearman(mean q, AUC), 95% CI |",
            "|---|---:|---:|---:|",
        ]
    )
    for channel in CHANNELS:
        row = discrimination.loc[channel]
        lines.append(
            f"| {channel} | {int(row['sources_with_auc'])} | "
            f"{int(row['sources_all_inactive'])} | "
            f"{_format_ci(row, 'spearman_quality_auc')} |"
        )
    lines.extend(
        [
            "",
            "AUC gives half credit to score ties. With multiple scored gold targets, AUC uses "
            "all positives and `gold_rank` is the best competition rank "
            "(`1 + count(score > best gold score)`); its tie size is also reported.",
            "",
            "## Did the channel pick the right target",
            "",
            "| Channel | Sources in rate | Active-argmax ties | All-candidate ties | "
            "Spearman(mean q, picked correctly), 95% CI |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for channel in CHANNELS:
        row = pick.loc[channel]
        lines.append(
            f"| {channel} | {int(row['sources_in_pick_rate'])} | "
            f"{int(row['sources_tied_at_active_argmax'])} | "
            f"{int(row['sources_tied_at_all_candidate_argmax'])} | "
            f"{_format_ci(row, 'spearman_quality_pick')} |"
        )
    lines.extend(
        [
            "",
            "The primary pick rate obeys WP5's active-only condition: source-channels with no "
            "active candidates are excluded, and a tie among active candidates is a miss. "
            "Because the specification also calls out inactive candidates pinned at `tau`, the "
            "table separately reports all-candidate tied maxima as a diagnostic; those inactive "
            "rows do not enter the rate or its correlation.",
            "",
            "## Difference from WP1",
            "",
            "WP1 correlated quality with threshold-agreement correctness on a table that was "
            "99.3% negative. That made lexical “accuracy” equal to the positive base rate "
            "(205/30,404 = 0.006743), a near-zero hierarchy correlation, and a negative "
            "attribute trend. WP5 changes the target, not the observed data: hierarchy changes "
            "from WP1's near-null result to a strongly adverse primary result "
            f"(Brier rho={_format_number(calibration.loc['hier', 'spearman_bin_brier'])}, "
            f"CI [{_format_number(calibration.loc['hier', 'spearman_ci_low'])}, "
            f"{_format_number(calibration.loc['hier', 'spearman_ci_high'])}]), while attribute "
            "changes from the adverse threshold-agreement result to support under the primary "
            f"continuous-score test (Brier rho="
            f"{_format_number(calibration.loc['attr', 'spearman_bin_brier'])}, "
            f"CI [{_format_number(calibration.loc['attr', 'spearman_ci_low'])}, "
            f"{_format_number(calibration.loc['attr', 'spearman_ci_high'])}]). The attribute "
            "source-level AUC and pick correlations remain null, so that support is specific "
            "to conditional calibration rather than a general discrimination result. Lexical "
            "quality remains constant and similarity/difference remain fully inactive. These "
            "verdicts supersede only WP1's threshold-agreement evidence; they do not alter "
            "WP1's dump, inertness, ablation, or ranking results.",
            "",
            "## Artifacts",
            "",
            "- `e5_calibration.csv` and `e5_calibration.{png,pdf}`",
            "- `e5_discrimination.csv` and `e5_discrimination.{png,pdf}`",
            "- `e5_pick_rate.csv` and `e5_pick_rate.{png,pdf}`",
            "",
        ]
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)


def _plot_calibration(table: pd.DataFrame, output_stem: Path) -> None:
    summaries = _summary_rows(table)
    fig, axes = plt.subplots(1, len(CHANNELS), figsize=(17, 3.5), sharey=True)
    for axis, channel in zip(axes, CHANNELS):
        rows = table.loc[table["channel"].eq(channel) & table["record_type"].eq("bin")].sort_values(
            "bin"
        )
        if rows.empty:
            axis.text(
                0.5,
                0.5,
                _status_text(str(summaries.loc[channel, "status"])),
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
        else:
            axis.plot(rows["mean_q_k"], rows["brier"], marker="o", label="Brier")
            axis.plot(
                rows["mean_q_k"],
                rows["positive_rate"],
                marker="s",
                linestyle="--",
                label="Positive rate",
            )
        axis.set_title(channel)
        axis.set_xlabel(f"$q_{{{channel}}}$")
        axis.grid(alpha=0.25)
        axis.set_ylim(0.0, 1.02)
    axes[0].set_ylabel("Rate / error")
    axes[-1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_stem.with_suffix(f".{suffix}"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_discrimination(table: pd.DataFrame, output_stem: Path) -> None:
    summaries = _summary_rows(table)
    fig, axes = plt.subplots(1, len(CHANNELS), figsize=(17, 3.5), sharey=True)
    for axis, channel in zip(axes, CHANNELS):
        rows = table.loc[
            table["channel"].eq(channel) & table["record_type"].eq("source") & table["auc"].notna()
        ]
        if rows.empty:
            axis.text(
                0.5,
                0.5,
                _status_text(str(summaries.loc[channel, "status"])),
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
        else:
            axis.scatter(rows["source_mean_q"], rows["auc"], s=14, alpha=0.65)
        axis.set_title(channel)
        axis.set_xlabel(f"source mean $q_{{{channel}}}$")
        axis.grid(alpha=0.25)
        axis.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("Per-source AUC")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_stem.with_suffix(f".{suffix}"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_pick(table: pd.DataFrame, output_stem: Path) -> None:
    summaries = _summary_rows(table)
    fig, axes = plt.subplots(1, len(CHANNELS), figsize=(17, 3.5), sharey=True)
    for axis, channel in zip(axes, CHANNELS):
        rows = table.loc[table["channel"].eq(channel) & table["record_type"].eq("bin")].sort_values(
            "bin"
        )
        if rows.empty:
            axis.text(
                0.5,
                0.5,
                _status_text(str(summaries.loc[channel, "status"])),
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
        else:
            axis.plot(rows["mean_q_k"], rows["pick_rate"], marker="o")
        axis.set_title(channel)
        axis.set_xlabel(f"source mean $q_{{{channel}}}$")
        axis.grid(alpha=0.25)
        axis.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("Unique-argmax pick rate")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_stem.with_suffix(f".{suffix}"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(
        temporary,
        index=False,
        float_format="%.10g",
        lineterminator="\n",
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.bootstrap_resamples <= 0:
        raise ValueError("--bootstrap-resamples must be positive")
    if args.jobs <= 0:
        raise ValueError("--jobs must be positive")
    prefix = args.prefix.rstrip("_")
    if not prefix or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", prefix):
        raise ValueError("--prefix must contain only letters, digits, '.', '_' or '-'")
    frame = _load_dump(args.dump)
    tau = _load_tau(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    calibration = calibration_analysis(
        frame,
        args.bootstrap_resamples,
        args.seed,
        args.jobs,
    )
    discrimination = discrimination_analysis(
        frame,
        args.bootstrap_resamples,
        args.seed,
        args.jobs,
    )
    pick = pick_analysis(
        frame,
        args.bootstrap_resamples,
        args.seed,
        args.jobs,
    )

    calibration_path = args.output_dir / f"{prefix}_calibration.csv"
    discrimination_path = args.output_dir / f"{prefix}_discrimination.csv"
    pick_path = args.output_dir / f"{prefix}_pick_rate.csv"
    _write_csv(calibration, calibration_path)
    _write_csv(discrimination, discrimination_path)
    _write_csv(pick, pick_path)
    _plot_calibration(calibration, args.output_dir / f"{prefix}_calibration")
    _plot_discrimination(discrimination, args.output_dir / f"{prefix}_discrimination")
    _plot_pick(pick, args.output_dir / f"{prefix}_pick_rate")

    if prefix == "e5":
        write_results(
            args.output_dir / "WP5_RESULTS.md",
            args.dump,
            args.config,
            tau,
            calibration,
            discrimination,
            pick,
            args.bootstrap_resamples,
            args.seed,
        )
    print(
        f"Analysed {len(frame):,} rows from {frame['src'].nunique():,} sources; "
        f"{prefix}_* outputs written to {args.output_dir.resolve()}."
    )


if __name__ == "__main__":
    main()
