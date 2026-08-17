#!/usr/bin/env python3
"""Reproduce the WP1 channel-dump analyses from a completed ranking run."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import stats


CHANNELS = ("lex", "hier", "sim", "diff", "attr")
STRUCTURAL_CHANNELS = ("hier", "sim", "diff", "attr")
STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "as",
        "at",
        "by",
        "disease",
        "disorder",
        "due",
        "for",
        "from",
        "in",
        "of",
        "on",
        "or",
        "syndrome",
        "the",
        "to",
        "type",
        "with",
        "without",
    }
)
REQUIRED_COLUMNS = {
    "src",
    "tgt",
    "is_correct",
    "src_label_text",
    "tgt_label_text",
    "S_pair",
    "S_final",
    "S_struct",
    "sigma_struct",
    "Q_struct",
    "S_lctx",
    "U",
    "p_llm",
    "w_c",
    "llm_invoked",
    "llm_gated",
}
for _channel in CHANNELS:
    REQUIRED_COLUMNS.update(
        {
            f"s_{_channel}",
            f"q_{_channel}",
            f"sigma_{_channel}",
        }
    )
for _channel in STRUCTURAL_CHANNELS:
    REQUIRED_COLUMNS.add(f"omega_{_channel}")


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


def _wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return math.nan, math.nan
    p = successes / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    return centre - radius, centre + radius


def reliability_tables(df: pd.DataFrame, tau: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    inactive_rows: list[dict] = []
    labels = df["is_correct"].astype(int)
    for channel in CHANNELS:
        q = pd.to_numeric(df[f"q_{channel}"], errors="coerce")
        sigma = pd.to_numeric(df[f"sigma_{channel}"], errors="coerce")
        score = pd.to_numeric(df[f"s_{channel}"], errors="coerce")
        valid = q.notna() & sigma.notna() & score.notna()
        channel_frame = pd.DataFrame(
            {
                "q": q[valid],
                "sigma": sigma[valid],
                "score": score[valid],
                "label": labels[valid],
            }
        )
        channel_frame["inactive"] = channel_frame["q"].le(0.0) | channel_frame["sigma"].le(1e-8)
        channel_frame["standalone_correct"] = (
            channel_frame["score"].gt(tau).astype(int).eq(channel_frame["label"]).astype(int)
        )
        active = ~channel_frame["inactive"]
        inactive_rows.append(
            {
                "channel": channel,
                "total_pairs": len(channel_frame),
                "active_pairs": int(active.sum()),
                "inactive_pairs": int(channel_frame["inactive"].sum()),
                "q_zero_pairs": int(channel_frame["q"].eq(0.0).sum()),
                "sigma_le_1e-8_pairs": int(channel_frame["sigma"].le(1e-8).sum()),
                "inactive_fraction": (
                    float(channel_frame["inactive"].mean()) if len(channel_frame) else math.nan
                ),
            }
        )
        if channel_frame.empty:
            continue
        if channel_frame["q"].nunique() == 1:
            channel_frame["bin"] = 0
            only = float(channel_frame["q"].iloc[0])
            edges = np.array([only, only])
        else:
            try:
                channel_frame["bin"], edges = pd.qcut(
                    channel_frame["q"],
                    q=10,
                    labels=False,
                    retbins=True,
                    duplicates="drop",
                )
            except ValueError:
                channel_frame["bin"] = 0
                edges = np.array([channel_frame["q"].min(), channel_frame["q"].max()])
        if channel_frame["bin"].isna().all():
            channel_frame["bin"] = 0
            only = float(channel_frame["q"].iloc[0])
            edges = np.array([only, only])
        n_bins = int(channel_frame["bin"].max()) + 1
        for bin_index in range(n_bins):
            part = channel_frame.loc[channel_frame["bin"].eq(bin_index)]
            active_part = part.loc[~part["inactive"]]
            active_n = len(active_part)
            successes = int(active_part["standalone_correct"].sum())
            ci_low, ci_high = _wilson_interval(successes, active_n)
            rows.append(
                {
                    "channel": channel,
                    "bin": bin_index + 1,
                    "bin_left": float(edges[min(bin_index, len(edges) - 1)]),
                    "bin_right": float(edges[min(bin_index + 1, len(edges) - 1)]),
                    "q_min": float(part["q"].min()),
                    "q_max": float(part["q"].max()),
                    "q_median": float(part["q"].median()),
                    "bin_total_n": len(part),
                    "active_n": active_n,
                    "inactive_n": int(part["inactive"].sum()),
                    "q_zero_n": int(part["q"].eq(0.0).sum()),
                    "sigma_le_1e-8_n": int(part["sigma"].le(1e-8).sum()),
                    "n": active_n,
                    "standalone_correct": successes,
                    "standalone_accuracy": successes / active_n if active_n else math.nan,
                    "wilson_low": ci_low,
                    "wilson_high": ci_high,
                    "status": "active_accuracy" if active_n else "all_inactive",
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(inactive_rows)


def plot_reliability(table: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, len(CHANNELS), figsize=(17, 3.5), sharey=True)
    for axis, channel in zip(axes, CHANNELS):
        part = table.loc[
            table["channel"].eq(channel) & table["standalone_accuracy"].notna()
        ].sort_values("bin")
        if part.empty:
            axis.text(0.5, 0.5, "No active pairs", ha="center", va="center")
        else:
            y = part["standalone_accuracy"].to_numpy()
            low = part["wilson_low"].to_numpy()
            high = part["wilson_high"].to_numpy()
            axis.errorbar(
                part["q_median"],
                y,
                yerr=np.vstack([y - low, high - y]),
                marker="o",
                linewidth=1.5,
                capsize=2,
            )
        axis.set_title(channel)
        axis.set_xlabel(f"$q_{{{channel}}}$")
        axis.grid(alpha=0.25)
        axis.set_ylim(0.0, 1.02)
    axes[0].set_ylabel("Standalone accuracy (95% Wilson CI)")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"e1_reliability.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _bootstrap_correlations(payload: tuple[str, np.ndarray, np.ndarray, int, int]) -> dict:
    channel, q, correct, n_resamples, seed = payload
    n = len(q)
    base = {
        "channel": channel,
        "n": n,
        "bootstrap_resamples": n_resamples,
        "bootstrap_valid_spearman": 0,
        "bootstrap_valid_kendall": 0,
    }
    if n < 2 or np.unique(q).size < 2 or np.unique(correct).size < 2:
        return {
            **base,
            "spearman": math.nan,
            "spearman_ci_low": math.nan,
            "spearman_ci_high": math.nan,
            "kendall": math.nan,
            "kendall_ci_low": math.nan,
            "kendall_ci_high": math.nan,
        }
    spearman = float(stats.spearmanr(q, correct).statistic)
    kendall = float(stats.kendalltau(q, correct).statistic)
    rng = np.random.default_rng(seed)
    spearman_samples = np.empty(n_resamples, dtype=float)
    kendall_samples = np.empty(n_resamples, dtype=float)
    for index in range(n_resamples):
        selected = rng.integers(0, n, size=n)
        sampled_q = q[selected]
        sampled_correct = correct[selected]
        spearman_samples[index] = stats.spearmanr(sampled_q, sampled_correct).statistic
        kendall_samples[index] = stats.kendalltau(sampled_q, sampled_correct).statistic
    spearman_samples = spearman_samples[np.isfinite(spearman_samples)]
    kendall_samples = kendall_samples[np.isfinite(kendall_samples)]
    return {
        "channel": channel,
        "n": n,
        "spearman": spearman,
        "spearman_ci_low": (
            float(np.quantile(spearman_samples, 0.025)) if len(spearman_samples) else math.nan
        ),
        "spearman_ci_high": (
            float(np.quantile(spearman_samples, 0.975)) if len(spearman_samples) else math.nan
        ),
        "kendall": kendall,
        "kendall_ci_low": (
            float(np.quantile(kendall_samples, 0.025)) if len(kendall_samples) else math.nan
        ),
        "kendall_ci_high": (
            float(np.quantile(kendall_samples, 0.975)) if len(kendall_samples) else math.nan
        ),
        "bootstrap_resamples": n_resamples,
        "bootstrap_valid_spearman": len(spearman_samples),
        "bootstrap_valid_kendall": len(kendall_samples),
    }


def correlation_table(
    df: pd.DataFrame,
    tau: float,
    resamples: int,
    seed: int,
    jobs: int,
) -> pd.DataFrame:
    payloads = []
    labels = df["is_correct"].astype(int)
    for channel in CHANNELS:
        q = pd.to_numeric(df[f"q_{channel}"], errors="coerce")
        sigma = pd.to_numeric(df[f"sigma_{channel}"], errors="coerce")
        score = pd.to_numeric(df[f"s_{channel}"], errors="coerce")
        active = q.notna() & sigma.notna() & score.notna() & q.gt(0.0) & sigma.gt(1e-8)
        correctness = score.gt(tau).astype(int).eq(labels).astype(int)
        payloads.append(
            (
                channel,
                q[active].to_numpy(dtype=float),
                correctness[active].to_numpy(dtype=int),
                resamples,
                seed,
            )
        )
    if jobs <= 1:
        results = [_bootstrap_correlations(payload) for payload in payloads]
    else:
        with ProcessPoolExecutor(max_workers=min(jobs, len(payloads))) as executor:
            results = list(executor.map(_bootstrap_correlations, payloads))
    return pd.DataFrame(results)


def _tokens(value: object) -> set[str]:
    words = re.findall(r"[a-z0-9]+", str(value).lower())
    if value is None or (not isinstance(value, (list, tuple, set, dict)) and pd.isna(value)):
        return set()
    return {word for word in words if len(word) > 1 and word not in STOPWORDS}


def _distribution_overlap(first: np.ndarray, second: np.ndarray, bins: int = 50) -> float:
    if len(first) == 0 or len(second) == 0:
        return math.nan
    lower = min(float(np.min(first)), float(np.min(second)))
    upper = max(float(np.max(first)), float(np.max(second)))
    if lower == upper:
        return 1.0
    edges = np.linspace(lower, upper, bins + 1)
    hist_a, _ = np.histogram(first, bins=edges)
    hist_b, _ = np.histogram(second, bins=edges)
    prob_a = hist_a / hist_a.sum()
    prob_b = hist_b / hist_b.sum()
    return float(np.minimum(prob_a, prob_b).sum())


def confounder_analysis(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    correct_rows = df.loc[df["is_correct"].eq(1)].copy()
    reference_tokens = (
        correct_rows.groupby("src")["tgt_label_text"]
        .apply(lambda values: set().union(*(_tokens(value) for value in values)))
        .to_dict()
    )
    all_sources = set(df["src"])
    reference_sources = set(reference_tokens)
    missing_reference_sources = all_sources - reference_sources
    incorrect = df.loc[df["is_correct"].eq(0)].copy()
    incorrect["reference_tokens"] = incorrect["src"].map(reference_tokens)
    incorrect["candidate_tokens"] = incorrect["tgt_label_text"].map(_tokens)
    incorrect["shared_tokens"] = incorrect.apply(
        lambda row: sorted(
            (row["reference_tokens"] if isinstance(row["reference_tokens"], set) else set())
            & row["candidate_tokens"]
        ),
        axis=1,
    )
    confounders = incorrect.loc[incorrect["shared_tokens"].map(bool)].copy()
    confounders["shared_tokens"] = confounders["shared_tokens"].map(
        lambda values: " | ".join(values)
    )
    q_confounders = pd.to_numeric(confounders["q_lex"], errors="coerce").dropna().to_numpy()
    q_correct = pd.to_numeric(correct_rows["q_lex"], errors="coerce").dropna().to_numpy()
    if len(q_confounders) and len(q_correct):
        test = stats.mannwhitneyu(q_confounders, q_correct, alternative="two-sided")
        greater_test = stats.mannwhitneyu(
            q_confounders,
            q_correct,
            alternative="greater",
        )
        common_language = float(greater_test.statistic / (len(q_confounders) * len(q_correct)))
        u_stat = float(test.statistic)
        p_value = float(test.pvalue)
        greater_p_value = float(greater_test.pvalue)
    else:
        u_stat = p_value = greater_p_value = common_language = math.nan
    summary = pd.DataFrame(
        [
            {
                "all_sources": len(all_sources),
                "sources_with_scored_correct_row": len(reference_sources),
                "sources_without_scored_correct_row": len(missing_reference_sources),
                "omitted_reference_pairs": 302 - len(correct_rows),
                "coverage_status": (
                    "complete"
                    if not missing_reference_sources
                    else "restricted_to_sources_with_scored_correct_row"
                ),
                "confounder_pairs": len(q_confounders),
                "confounder_sources": confounders["src"].nunique(),
                "correct_pairs": len(q_correct),
                "correct_sources": correct_rows["src"].nunique(),
                "confounder_q_lex_median": (
                    float(np.median(q_confounders)) if len(q_confounders) else math.nan
                ),
                "correct_q_lex_median": (
                    float(np.median(q_correct)) if len(q_correct) else math.nan
                ),
                "median_difference_confounder_minus_correct": (
                    float(np.median(q_confounders) - np.median(q_correct))
                    if len(q_confounders) and len(q_correct)
                    else math.nan
                ),
                "mann_whitney_u": u_stat,
                "mann_whitney_two_sided_p": p_value,
                "mann_whitney_greater_p": greater_p_value,
                "common_language_p_confounder_gt_correct": common_language,
                "histogram_overlap_coefficient": _distribution_overlap(q_confounders, q_correct),
                "overlap_estimator": "histogram intersection on 50 common-width bins",
                "overlap_bins": 50,
                "normalization": (
                    "lowercase; non-alphanumeric stripped; one-character tokens and committed "
                    f"stopwords removed ({', '.join(sorted(STOPWORDS))})"
                ),
            }
        ]
    )
    keep_columns = [
        "src",
        "tgt",
        "src_label_text",
        "tgt_label_text",
        "shared_tokens",
        "q_lex",
        "s_lex",
    ]
    return summary, confounders[keep_columns]


def plot_confounders(df: pd.DataFrame, confounders: pd.DataFrame, output_dir: Path) -> None:
    correct = pd.to_numeric(df.loc[df["is_correct"].eq(1), "q_lex"], errors="coerce").dropna()
    confounder_values = pd.to_numeric(confounders["q_lex"], errors="coerce").dropna()
    fig, axis = plt.subplots(figsize=(6.4, 4.2))
    edges = np.linspace(0.0, 1.0, 41)
    axis.hist(correct, bins=edges, density=True, histtype="step", linewidth=2, label="Correct")
    axis.hist(
        confounder_values,
        bins=edges,
        density=True,
        histtype="step",
        linewidth=2,
        label="Same-family confounder",
    )
    axis.axvline(correct.median(), linestyle="--", alpha=0.7)
    if not confounder_values.empty:
        axis.axvline(confounder_values.median(), linestyle=":", alpha=0.7)
    axis.set_xlabel("$q_{lex}$")
    axis.set_ylabel("Density")
    axis.legend()
    axis.grid(alpha=0.2)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"e1_confounders.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def inertness_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for channel in STRUCTURAL_CHANNELS:
        sigma = pd.to_numeric(df[f"sigma_{channel}"], errors="coerce")
        omega = pd.to_numeric(df[f"omega_{channel}"], errors="coerce")
        rows.append(
            {
                "channel": channel,
                "pairs": int((sigma.notna() & omega.notna()).sum()),
                "sigma_lt_1e-6_count": int(sigma.lt(1e-6).sum()),
                "sigma_lt_1e-6_fraction": float(sigma.lt(1e-6).mean()),
                "omega_lt_0.01_count": int(omega.lt(0.01).sum()),
                "omega_lt_0.01_fraction": float(omega.lt(0.01).mean()),
            }
        )
    return pd.DataFrame(rows)


def llm_analysis(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    invoked = _as_binary(df["llm_invoked"])
    gated = _as_binary(df["llm_gated"])
    invoked_bool = invoked.astype(bool)
    gated_bool = gated.astype(bool)
    working = df.copy()
    working["_invoked"] = invoked
    top1_rows: list[dict] = []
    for src, group in working.groupby("src", sort=True):
        if not group["_invoked"].any():
            continue
        pair_scores = pd.to_numeric(group["S_pair"], errors="coerce")
        final_scores = pd.to_numeric(group["S_final"], errors="coerce")
        pair_index = pair_scores.idxmax()
        final_index = final_scores.idxmax()
        pair_row = group.loc[pair_index]
        final_row = group.loc[final_index]
        changed = str(pair_row["tgt"]) != str(final_row["tgt"])
        pair_correct = int(pair_row["is_correct"])
        final_correct = int(final_row["is_correct"])
        outcome = "unchanged"
        if changed and final_correct > pair_correct:
            outcome = "helped"
        elif changed and final_correct < pair_correct:
            outcome = "hurt"
        elif changed:
            outcome = "changed_neutral"
        top1_rows.append(
            {
                "src": src,
                "invoked_pairs": int(group["_invoked"].sum()),
                "pair_top_tgt": pair_row["tgt"],
                "final_top_tgt": final_row["tgt"],
                "pair_top_correct": pair_correct,
                "final_top_correct": final_correct,
                "pair_top_ties": int(pair_scores.eq(pair_scores.max()).sum()),
                "final_top_ties": int(final_scores.eq(final_scores.max()).sum()),
                "top1_changed": int(changed),
                "outcome": outcome,
            }
        )
    top1_columns = [
        "src",
        "invoked_pairs",
        "pair_top_tgt",
        "final_top_tgt",
        "pair_top_correct",
        "final_top_correct",
        "pair_top_ties",
        "final_top_ties",
        "top1_changed",
        "outcome",
    ]
    top1 = pd.DataFrame(top1_rows, columns=top1_columns)
    changed = int(top1["top1_changed"].sum()) if not top1.empty else 0
    summary = pd.DataFrame(
        [
            {
                "pairs": len(df),
                "llm_invoked_pairs": int(invoked.sum()),
                "llm_invoked_fraction": float(invoked.mean()),
                "llm_gated_pairs": int(gated.sum()),
                "llm_gated_fraction": float(gated.mean()),
                "gating_invocation_disagreement_pairs": int(invoked.ne(gated).sum()),
                "gated_without_invocation_pairs": int((gated_bool & ~invoked_bool).sum()),
                "invoked_without_gate_pairs": int((invoked_bool & ~gated_bool).sum()),
                "sources": df["src"].nunique(),
                "sources_with_invocation": int(top1["src"].nunique()) if not top1.empty else 0,
                "sources_top1_changed": changed,
                "top1_change_fraction_among_invoked_sources": (
                    changed / len(top1) if len(top1) else 0.0
                ),
                "top1_changes_helped": (
                    int(top1["outcome"].eq("helped").sum()) if not top1.empty else 0
                ),
                "top1_changes_hurt": int(top1["outcome"].eq("hurt").sum()) if not top1.empty else 0,
                "top1_changes_neutral": (
                    int(top1["outcome"].eq("changed_neutral").sum()) if not top1.empty else 0
                ),
            }
        ]
    )
    return summary, top1


def validation_table(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    omega_sum = sum(
        (pd.to_numeric(df[f"omega_{channel}"], errors="coerce") for channel in STRUCTURAL_CHANNELS)
    )
    sigma_struct = pd.to_numeric(df["sigma_struct"], errors="coerce")
    active_struct = sigma_struct.gt(1e-8)
    inactive_struct = ~active_struct
    sigma_lex = pd.to_numeric(df["sigma_lex"], errors="coerce")
    both_active = sigma_lex.gt(1e-8) & active_struct
    expected_pair = (1.0 - pd.to_numeric(df["w_c"], errors="coerce")) * pd.to_numeric(
        df["s_lex"], errors="coerce"
    ) + pd.to_numeric(df["w_c"], errors="coerce") * pd.to_numeric(df["S_struct"], errors="coerce")
    pair_error = (
        pd.to_numeric(df.loc[both_active, "S_pair"], errors="coerce")
        - expected_pair.loc[both_active]
    ).abs()
    _ = seed
    constant_columns = []
    for column in sorted(REQUIRED_COLUMNS - {"src", "tgt", "src_label_text", "tgt_label_text"}):
        if df[column].nunique(dropna=False) <= 1:
            constant_columns.append(column)
    rows = [
        {
            "check": "scored_rows",
            "value": len(df),
            "expected": "0 < rows <= 30502",
            "passed": 0 < len(df) <= 30502,
        },
        {
            "check": "rows_omitted_by_exact_prefilter",
            "value": 30502 - len(df),
            "expected": "reported (98 for this run)",
            "passed": len(df) <= 30502,
        },
        {
            "check": "scored_correct_rows",
            "value": int(df["is_correct"].sum()),
            "expected": "1..302; omitted references reported",
            "passed": 0 < int(df["is_correct"].sum()) <= 302,
        },
        {
            "check": "reference_rows_omitted_from_scoring",
            "value": 302 - int(df["is_correct"].sum()),
            "expected": "reported (97 for this run)",
            "passed": int(df["is_correct"].sum()) <= 302,
        },
        {
            "check": "sources_with_scored_correct_row",
            "value": int(df.loc[df["is_correct"].eq(1), "src"].nunique()),
            "expected": "coverage reported (204/300 for this run)",
            "passed": True,
        },
        {
            "check": "unique_sources",
            "value": df["src"].nunique(),
            "expected": 300,
            "passed": df["src"].nunique() == 300,
        },
        {
            "check": "duplicate_src_tgt_rows",
            "value": int(df.duplicated(["src", "tgt"]).sum()),
            "expected": "reported (input may repeat candidate pairs)",
            "passed": True,
        },
        {
            "check": "omega_sum_active_max_abs_error",
            "value": (
                float((omega_sum[active_struct] - 1.0).abs().max()) if active_struct.any() else 0.0
            ),
            "expected": "<=1e-5",
            "passed": bool(
                not active_struct.any() or (omega_sum[active_struct] - 1.0).abs().max() <= 1e-5
            ),
        },
        {
            "check": "omega_sum_inactive_max_abs_error",
            "value": (
                float(omega_sum[inactive_struct].abs().max()) if inactive_struct.any() else 0.0
            ),
            "expected": "<=1e-5",
            "passed": bool(
                not inactive_struct.any() or omega_sum[inactive_struct].abs().max() <= 1e-5
            ),
        },
        {
            "check": "S_pair_mixture_both_active_max_abs_error",
            "value": float(pair_error.max()) if not pair_error.empty else 0.0,
            "expected": "<=1e-5",
            "passed": bool(pair_error.empty or pair_error.max() <= 1e-5),
        },
        {
            "check": "constant_required_columns_full_dump",
            "value": " | ".join(constant_columns),
            "expected": "reported; physically constant columns are allowed",
            "passed": True,
        },
    ]
    return pd.DataFrame(rows)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, float_format="%.10g")


def _records(frame: pd.DataFrame) -> Iterable[dict]:
    return frame.astype(object).where(pd.notna(frame), None).to_dict(orient="records")


def main() -> None:
    args = parse_args()
    if args.bootstrap_resamples <= 0:
        raise ValueError("--bootstrap-resamples must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.dump)
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"channel dump is missing required columns: {missing}")
    numeric_columns = REQUIRED_COLUMNS - {
        "src",
        "tgt",
        "src_label_text",
        "tgt_label_text",
        "is_correct",
        "llm_invoked",
        "llm_gated",
    }
    for column in sorted(numeric_columns):
        converted = pd.to_numeric(frame[column], errors="coerce")
        bad = converted.isna() | ~np.isfinite(converted)
        if bad.any():
            raise ValueError(f"{column}: {int(bad.sum())} missing/non-finite values")
        frame[column] = converted
    frame["is_correct"] = _as_binary(frame["is_correct"])
    frame["llm_invoked"] = _as_binary(frame["llm_invoked"])
    frame["llm_gated"] = _as_binary(frame["llm_gated"])
    tau = _load_tau(args.config)

    reliability, inactive = reliability_tables(frame, tau)
    _write_csv(reliability, args.output_dir / "e1_reliability.csv")
    _write_csv(inactive, args.output_dir / "e1_reliability_inactive.csv")
    plot_reliability(reliability, args.output_dir)

    correlations = correlation_table(
        frame,
        tau,
        args.bootstrap_resamples,
        args.seed,
        args.jobs,
    )
    _write_csv(correlations, args.output_dir / "e1_rank_correlations.csv")

    confounder_summary, confounder_pairs = confounder_analysis(frame)
    _write_csv(confounder_summary, args.output_dir / "e1_confounders.csv")
    _write_csv(confounder_pairs, args.output_dir / "e1_confounder_pairs.csv")
    plot_confounders(frame, confounder_pairs, args.output_dir)

    inertness = inertness_table(frame)
    _write_csv(inertness, args.output_dir / "e1_inertness.csv")

    llm_summary, llm_top1 = llm_analysis(frame)
    _write_csv(llm_summary, args.output_dir / "e1_llm.csv")
    _write_csv(llm_top1, args.output_dir / "e1_llm_top1.csv")

    validation = validation_table(frame, args.seed)
    _write_csv(validation, args.output_dir / "e1_validation.csv")

    summary = {
        "dump": str(args.dump.resolve()),
        "config": str(args.config.resolve()),
        "tau": tau,
        "rows": len(frame),
        "bootstrap_resamples": args.bootstrap_resamples,
        "seed": args.seed,
        "jobs": args.jobs,
        "unique_sources": int(frame["src"].nunique()),
        "reliability": list(_records(reliability)),
        "inactive": list(_records(inactive)),
        "correlations": list(_records(correlations)),
        "confounders": list(_records(confounder_summary)),
        "inertness": list(_records(inertness)),
        "llm": list(_records(llm_summary)),
        "validation": list(_records(validation)),
    }

    def json_safe(value: object) -> object:
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [json_safe(item) for item in value]
        if isinstance(value, np.generic):
            return json_safe(value.item())
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    summary = json_safe(summary)
    summary_path = args.output_dir / "e1_analysis_summary.json"
    temporary_path = summary_path.with_suffix(".json.tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")
    temporary_path.replace(summary_path)
    print(
        f"Analysed {len(frame):,} pairs from {frame['src'].nunique():,} sources; "
        f"outputs written to {args.output_dir.resolve()}."
    )


if __name__ == "__main__":
    main()
