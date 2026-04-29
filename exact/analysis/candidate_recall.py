from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd


Pair = Tuple[str, str]


def analyze_candidate_recall(
    candidates: pd.DataFrame | Iterable[Pair],
    reference_pairs: Iterable[Pair],
    train_pairs: Optional[Iterable[Pair]] = None,
    exact_pairs: Optional[Iterable[Pair]] = None,
    src_col: str = "Src",
    tgt_col: str = "Tgt",
    score_col: str = "cand_sim",
) -> Dict[str, Any]:
    reference_set = _normalize_pairs(reference_pairs)
    train_set = _normalize_pairs(train_pairs or [])
    effective_reference = reference_set.difference(train_set)
    exact_set = _normalize_pairs(exact_pairs or []).difference(train_set)

    candidate_df = candidates if isinstance(candidates, pd.DataFrame) else _pairs_to_frame(candidates, src_col, tgt_col)
    candidate_pairs = _pairs_from_frame(candidate_df, src_col, tgt_col).difference(train_set)
    generated_hits = effective_reference.intersection(candidate_pairs)
    oracle_pairs = candidate_pairs.union(exact_set)
    oracle_hits = effective_reference.intersection(oracle_pairs)
    absent_generated = effective_reference.difference(candidate_pairs)
    absent_oracle = effective_reference.difference(oracle_pairs)

    return {
        "counts": {
            "reference_pairs": len(effective_reference),
            "raw_reference_pairs": len(reference_set),
            "train_reference_pairs": len(train_set),
            "candidate_pairs": len(candidate_pairs),
            "exact_pairs": len(exact_set),
            "generated_hits": len(generated_hits),
            "oracle_hits": len(oracle_hits),
            "absent_gold_pairs": len(absent_generated),
            "absent_gold_pairs_after_exact": len(absent_oracle),
        },
        "metrics": {
            "generated_candidate_recall": _safe_div(len(generated_hits), len(effective_reference)),
            "exact_prefilter_oracle_recall": _safe_div(len(oracle_hits), len(effective_reference)),
        },
        "gold_rank": _gold_rank_summary(candidate_df, effective_reference, src_col, tgt_col, score_col),
        "absent_gold_pairs": _pair_records(absent_generated),
        "absent_gold_pairs_after_exact": _pair_records(absent_oracle),
    }


def absent_gold_dataframe(analysis: Mapping[str, Any], after_exact: bool = False) -> pd.DataFrame:
    key = "absent_gold_pairs_after_exact" if after_exact else "absent_gold_pairs"
    return pd.DataFrame(list(analysis.get(key, [])), columns=["Src", "Tgt"])


def write_absent_gold_tsv(path: Path, analysis: Mapping[str, Any], after_exact: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    absent_gold_dataframe(analysis, after_exact=after_exact).to_csv(path, sep="\t", index=False)


def flatten_candidate_recall(top_k: int, analysis: Mapping[str, Any]) -> Dict[str, Any]:
    counts = dict(analysis.get("counts", {}))
    metrics = dict(analysis.get("metrics", {}))
    ranks = dict(analysis.get("gold_rank", {}))
    row: Dict[str, Any] = {"top_k": int(top_k)}
    row.update(counts)
    row.update(metrics)
    row.update({f"gold_rank_{key}": value for key, value in ranks.items()})
    return row


def pairs_from_table(path: Path) -> set[Pair]:
    df = pd.read_csv(path, sep=None, engine="python")
    return _pairs_from_frame(df, df.columns[0], df.columns[1])


def _pairs_to_frame(pairs: Iterable[Pair], src_col: str, tgt_col: str) -> pd.DataFrame:
    return pd.DataFrame(
        [(src, tgt) for src, tgt in sorted(_normalize_pairs(pairs))],
        columns=[src_col, tgt_col],
    )


def _pairs_from_frame(df: pd.DataFrame, src_col: str, tgt_col: str) -> set[Pair]:
    if df.empty or src_col not in df.columns or tgt_col not in df.columns:
        return set()
    return _normalize_pairs(zip(df[src_col], df[tgt_col]))


def _normalize_pairs(pairs: Iterable[Pair]) -> set[Pair]:
    return {
        (str(src), str(tgt))
        for src, tgt in pairs
        if str(src) and str(tgt)
    }


def _pair_records(pairs: Iterable[Pair]) -> List[Dict[str, str]]:
    return [
        {"Src": src, "Tgt": tgt}
        for src, tgt in sorted(_normalize_pairs(pairs))
    ]


def _gold_rank_summary(
    df: pd.DataFrame,
    reference_pairs: set[Pair],
    src_col: str,
    tgt_col: str,
    score_col: str,
) -> Dict[str, Any]:
    if df.empty or not reference_pairs or score_col not in df.columns:
        return {
            "present_pairs": 0,
            "rank_median": None,
            "rank_p90": None,
        }
    ranks = _rank_lookup(df, src_col, tgt_col, score_col)
    present_ranks = [
        ranks[pair]
        for pair in sorted(reference_pairs)
        if pair in ranks
    ]
    return {
        "present_pairs": len(present_ranks),
        "rank_median": _quantile(present_ranks, 0.5),
        "rank_p90": _quantile(present_ranks, 0.9),
    }


def _rank_lookup(df: pd.DataFrame, src_col: str, tgt_col: str, score_col: str) -> Dict[Pair, int]:
    lookup: Dict[Pair, int] = {}
    if src_col not in df.columns or tgt_col not in df.columns:
        return lookup
    working = df[[src_col, tgt_col, score_col]].copy()
    working[src_col] = working[src_col].astype(str)
    working[tgt_col] = working[tgt_col].astype(str)
    working[score_col] = pd.to_numeric(working[score_col], errors="coerce").fillna(0.0)
    for _, group in working.groupby(src_col, sort=False):
        ranked = group.sort_values([score_col, tgt_col], ascending=[False, True], kind="mergesort")
        for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
            lookup[(str(row[src_col]), str(row[tgt_col]))] = rank
    return lookup


def _quantile(values: Sequence[int], q: float) -> Optional[float]:
    if not values:
        return None
    series = pd.Series(list(values), dtype=float)
    return float(series.quantile(float(q)))


def _safe_div(num: int, denom: int) -> float:
    return float(num) / float(denom) if denom else 0.0
