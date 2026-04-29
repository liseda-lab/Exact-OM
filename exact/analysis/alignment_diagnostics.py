from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import pandas as pd


Pair = Tuple[str, str]


def analyze_alignment_run(
    run_dir: Path,
    reference_path: Path,
    summary_path: Optional[Path] = None,
    alignment_path: Optional[Path] = None,
) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    reference_path = Path(reference_path)
    summary_path = summary_path or _find_summary_path(run_dir)
    alignment_path = alignment_path or _find_alignment_path(run_dir)

    reference_df = _read_table(reference_path)
    summary_df = _read_table(summary_path) if summary_path is not None else pd.DataFrame()
    alignment_df = _read_table(alignment_path) if alignment_path is not None else pd.DataFrame()

    reference_pairs = _pairs_from_columns(reference_df, reference_df.columns[:2])
    candidate_pairs = _pairs_from_columns(summary_df, ("src_iri", "tgt_iri"))
    if not alignment_df.empty:
        prediction_pairs = _pairs_from_columns(alignment_df, alignment_df.columns[:2])
    else:
        prediction_pairs = _saved_pairs_from_summary(summary_df)

    selected_pairs = _saved_pairs_from_summary(summary_df) or prediction_pairs
    tp = reference_pairs.intersection(prediction_pairs)
    fp = prediction_pairs.difference(reference_pairs)
    fn = reference_pairs.difference(prediction_pairs)
    present_missed = fn.intersection(candidate_pairs)
    absent_missed = fn.difference(candidate_pairs)
    oracle_recoverable_pairs = reference_pairs.intersection(candidate_pairs.union(prediction_pairs))

    source_predictions: Dict[str, set[str]] = {}
    for src, tgt in prediction_pairs:
        source_predictions.setdefault(src, set()).add(tgt)

    present_wrong_selected = {
        pair
        for pair in present_missed
        if source_predictions.get(pair[0]) and pair[1] not in source_predictions[pair[0]]
    }
    present_no_prediction = present_missed.difference(present_wrong_selected)
    present_abstained = _present_abstained_misses(summary_df, present_no_prediction)
    present_rejected = present_no_prediction.difference(present_abstained)

    diagnostics: Dict[str, Any] = {
        "paths": {
            "run_dir": str(run_dir),
            "reference": str(reference_path),
            "summary": str(summary_path) if summary_path else None,
            "alignment": str(alignment_path) if alignment_path else None,
        },
        "counts": {
            "reference_pairs": len(reference_pairs),
            "prediction_pairs": len(prediction_pairs),
            "candidate_pairs": len(candidate_pairs),
            "true_positive_pairs": len(tp),
            "false_positive_pairs": len(fp),
            "false_negative_pairs": len(fn),
        },
        "metrics": _precision_recall_f1(len(tp), len(fp), len(fn)),
        "oracle": {
            "candidate_oracle_recall": _safe_div(len(oracle_recoverable_pairs), len(reference_pairs)),
            "selected_recall": _safe_div(len(reference_pairs.intersection(selected_pairs)), len(reference_pairs)),
            "missed_present_in_candidates": len(present_missed),
            "missed_absent_from_candidates": len(absent_missed),
        },
        "miss_buckets": {
            "candidate_absent": len(absent_missed),
            "present_wrong_selected": len(present_wrong_selected),
            "present_abstained": len(present_abstained),
            "present_rejected_or_filtered": len(present_rejected),
        },
        "gold_rank": _gold_rank_summary(summary_df, fn),
        "llm": _llm_summary(summary_df, reference_pairs),
    }
    return diagnostics


def format_diagnostics_report(diagnostics: Mapping[str, Any]) -> str:
    counts = diagnostics.get("counts", {})
    metrics = diagnostics.get("metrics", {})
    oracle = diagnostics.get("oracle", {})
    buckets = diagnostics.get("miss_buckets", {})
    rank = diagnostics.get("gold_rank", {})
    llm = diagnostics.get("llm", {})

    lines = [
        "Alignment diagnostics",
        f"  Reference pairs: {counts.get('reference_pairs', 0)}",
        f"  Prediction pairs: {counts.get('prediction_pairs', 0)}",
        (
            "  Direct P/R/F1: "
            f"{metrics.get('precision', 0.0):.3f} / "
            f"{metrics.get('recall', 0.0):.3f} / "
            f"{metrics.get('f1', 0.0):.3f}"
        ),
        (
            "  Candidate oracle recall: "
            f"{oracle.get('candidate_oracle_recall', 0.0):.3f} "
            f"({oracle.get('missed_present_in_candidates', 0)} missed present, "
            f"{oracle.get('missed_absent_from_candidates', 0)} missed absent)"
        ),
        "Miss buckets",
        f"  candidate_absent: {buckets.get('candidate_absent', 0)}",
        f"  present_wrong_selected: {buckets.get('present_wrong_selected', 0)}",
        f"  present_abstained: {buckets.get('present_abstained', 0)}",
        f"  present_rejected_or_filtered: {buckets.get('present_rejected_or_filtered', 0)}",
    ]
    if rank:
        lines.extend(
            [
                "Gold rank among present misses",
                f"  pair_rank_median: {rank.get('pair_rank_median')}",
                f"  pair_rank_p90: {rank.get('pair_rank_p90')}",
                f"  utility_rank_median: {rank.get('utility_rank_median')}",
                f"  utility_rank_p90: {rank.get('utility_rank_p90')}",
            ]
        )
    if llm:
        lines.extend(
            [
                "LLM selector impact",
                f"  rows: {llm.get('rows', 0)}",
                f"  selected_tp: {llm.get('selected_tp', 0)}",
                f"  selected_fp: {llm.get('selected_fp', 0)}",
                f"  reasons: {llm.get('reasons', {})}",
            ]
        )
    return "\n".join(lines)


def _read_table(path: Optional[Path]) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    return pd.read_csv(path, sep=None, engine="python")


def _find_summary_path(run_dir: Path) -> Optional[Path]:
    candidates = sorted(run_dir.glob("model/alignment/*/summary_metrics.csv"))
    return candidates[0] if candidates else None


def _find_alignment_path(run_dir: Path) -> Optional[Path]:
    candidates = sorted(run_dir.glob("model/alignment/*.maps_global.tsv"))
    if not candidates:
        candidates = sorted(run_dir.glob("model/alignment/*.tsv"))
    return candidates[0] if candidates else None


def _pairs_from_columns(df: pd.DataFrame, columns: Sequence[str]) -> set[Pair]:
    if df.empty or len(columns) < 2:
        return set()
    src_col, tgt_col = columns[:2]
    if src_col not in df.columns or tgt_col not in df.columns:
        return set()
    return {
        (str(src), str(tgt))
        for src, tgt in zip(df[src_col], df[tgt_col])
        if str(src) and str(tgt)
    }


def _saved_pairs_from_summary(df: pd.DataFrame) -> set[Pair]:
    if df.empty or "saved_alignment_member" not in df.columns:
        return set()
    saved = _bool_series(df["saved_alignment_member"])
    return _pairs_from_columns(df.loc[saved], ("src_iri", "tgt_iri"))


def _present_abstained_misses(df: pd.DataFrame, present_misses: Iterable[Pair]) -> set[Pair]:
    if df.empty or "selector_abstained" not in df.columns:
        return set()
    miss_set = set(present_misses)
    if not miss_set:
        return set()
    abstained = _bool_series(df["selector_abstained"])
    rows = df.loc[abstained]
    return {
        (str(src), str(tgt))
        for src, tgt in zip(rows["src_iri"], rows["tgt_iri"])
        if (str(src), str(tgt)) in miss_set
    }


def _gold_rank_summary(df: pd.DataFrame, missed_pairs: set[Pair]) -> Dict[str, Any]:
    if df.empty or not missed_pairs or "src_iri" not in df.columns or "tgt_iri" not in df.columns:
        return {}

    pair_ranks = _rank_lookup(df, "S_pair_final")
    utility_ranks = _rank_lookup(df, "selection_utility")
    present_pair_ranks = [
        pair_ranks[pair]
        for pair in missed_pairs
        if pair in pair_ranks
    ]
    present_utility_ranks = [
        utility_ranks[pair]
        for pair in missed_pairs
        if pair in utility_ranks
    ]
    return {
        "present_missed_with_pair_rank": len(present_pair_ranks),
        "pair_rank_median": _quantile(present_pair_ranks, 0.5),
        "pair_rank_p90": _quantile(present_pair_ranks, 0.9),
        "present_missed_with_utility_rank": len(present_utility_ranks),
        "utility_rank_median": _quantile(present_utility_ranks, 0.5),
        "utility_rank_p90": _quantile(present_utility_ranks, 0.9),
    }


def _rank_lookup(df: pd.DataFrame, score_col: str) -> Dict[Pair, int]:
    if score_col not in df.columns:
        return {}
    lookup: Dict[Pair, int] = {}
    for _, group in df.groupby("src_iri", sort=False):
        ranked = group.sort_values(score_col, ascending=False, kind="mergesort")
        for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
            lookup[(str(row["src_iri"]), str(row["tgt_iri"]))] = rank
    return lookup


def _llm_summary(df: pd.DataFrame, reference_pairs: set[Pair]) -> Dict[str, Any]:
    if df.empty or "selector_llm_used" not in df.columns:
        return {}
    llm_rows = df.loc[_bool_series(df["selector_llm_used"])]
    if llm_rows.empty:
        return {"rows": 0, "selected_tp": 0, "selected_fp": 0, "reasons": {}}
    saved = _bool_series(llm_rows.get("saved_alignment_member", pd.Series(False, index=llm_rows.index)))
    selected_rows = llm_rows.loc[saved]
    selected_pairs = _pairs_from_columns(selected_rows, ("src_iri", "tgt_iri"))
    reasons = (
        llm_rows["selector_reason"].astype(str).value_counts().sort_index().to_dict()
        if "selector_reason" in llm_rows.columns
        else {}
    )
    return {
        "rows": int(len(llm_rows)),
        "selected_tp": len(selected_pairs.intersection(reference_pairs)),
        "selected_fp": len(selected_pairs.difference(reference_pairs)),
        "reasons": reasons,
    }


def _precision_recall_f1(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def _safe_div(num: float, denom: float) -> float:
    return float(num) / float(denom) if denom else 0.0


def _bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"1", "true", "yes", "y"})


def _quantile(values: Sequence[int], q: float) -> Optional[float]:
    if not values:
        return None
    return float(pd.Series(values).quantile(q))
