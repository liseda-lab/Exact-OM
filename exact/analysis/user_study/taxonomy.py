from __future__ import annotations

import ast  # noqa: F401
import copy  # noqa: F401
import json  # noqa: F401
import logging  # noqa: F401
import time  # noqa: F401
import warnings  # noqa: F401
from dataclasses import dataclass  # noqa: F401
from pathlib import Path  # noqa: F401
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple  # noqa: F401

import pandas as pd  # noqa: F401
import torch  # noqa: F401

from exact.utils.data import read_yaml  # noqa: F401
from exact.utils.formatting import format_duration as _format_duration  # noqa: F401

from .selection import _safe_float


def _failure_taxonomy(source_df: pd.DataFrame, pair_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    nonlex_median = float(pair_df["nonlex_total"].median()) if not pair_df.empty else 0.0
    rows: List[Dict[str, Any]] = []
    gold_below_label = "gold_below_top5" if top_k == 5 else "gold_below_topk"
    for row in source_df.to_dict(orient="records"):
        gold_score = _safe_float(row.get("gold_score"), 0.0)
        top1_score = _safe_float(row.get("top1_score"), 0.0)
        delta_s_struct = _safe_float(row.get("delta_S_struct"), float("nan"))
        delta_s_label = _safe_float(row.get("delta_s_label"), float("nan"))
        top1_i_struct = _safe_float(row.get("top1_I_struct"), float("nan"))
        gold_i_struct = _safe_float(row.get("gold_I_struct"), float("nan"))
        top1_nonlex = (
            _safe_float(row.get("top1_hierarchy_count"), 0.0)
            + _safe_float(row.get("top1_similarity_count"), 0.0)
            + _safe_float(row.get("top1_difference_count"), 0.0)
            + _safe_float(row.get("top1_attribute_count"), 0.0)
        )
        gold_nonlex = (
            _safe_float(row.get("gold_hierarchy_count"), 0.0)
            + _safe_float(row.get("gold_similarity_count"), 0.0)
            + _safe_float(row.get("gold_difference_count"), 0.0)
            + _safe_float(row.get("gold_attribute_count"), 0.0)
        )
        flags = {
            "llm_active_top1": bool(row.get("top1_llm_active")),
            "llm_active_gold": bool(row.get("gold_llm_active")),
            "nonlex_sparse_top1": bool(top1_i_struct < 0.30 or top1_nonlex < nonlex_median),
            "nonlex_sparse_gold": bool(gold_i_struct < 0.30 or gold_nonlex < nonlex_median),
            "similarity_present_in_panel": bool(row.get("panel_has_similarity")),
            "high_disagreement_gold": bool(_safe_float(row.get("gold_U_dis"), 0.0) >= 0.10),
            "high_disagreement_top1": bool(_safe_float(row.get("top1_U_dis"), 0.0) >= 0.10),
        }
        primary = "evidence_sparse_other"
        if int(row.get("missing_panel_record_count", 0) or 0) > 0:
            primary = "missing_panel_record"
        elif int(row.get("gold_rank", -1) or -1) > top_k:
            primary = gold_below_label
        elif abs(top1_score - gold_score) <= 0.02:
            primary = "near_tie"
        elif not pd.isna(delta_s_struct) and delta_s_struct >= 0.02 and gold_score < top1_score:
            primary = "gold_structurally_better_but_loses"
        elif not pd.isna(delta_s_struct) and delta_s_struct <= -0.02:
            primary = "top1_structurally_better"
        elif not pd.isna(delta_s_label) and delta_s_label <= -0.03 and abs(delta_s_struct) < 0.02:
            primary = "lexical_overweight"

        rows.append(
            {
                "src_iri": row.get("src_iri"),
                "source_label": row.get("source_label"),
                "gold_tgt_iri": row.get("gold_tgt_iri"),
                "gold_tgt_label": row.get("gold_tgt_label"),
                "top1_tgt_iri": row.get("top1_tgt_iri"),
                "top1_tgt_label": row.get("top1_tgt_label"),
                "gold_rank": int(row.get("gold_rank", -1)),
                "top1_score": top1_score,
                "gold_score": gold_score,
                "score_gap": _safe_float(row.get("score_gap"), 0.0),
                "delta_s_label": _safe_float(row.get("delta_s_label"), float("nan")),
                "delta_S_struct": _safe_float(row.get("delta_S_struct"), float("nan")),
                "delta_I_label": _safe_float(row.get("delta_I_label"), float("nan")),
                "delta_I_struct": _safe_float(row.get("delta_I_struct"), float("nan")),
                "delta_U": _safe_float(row.get("delta_U"), float("nan")),
                "delta_U_dis": _safe_float(row.get("delta_U_dis"), float("nan")),
                "delta_hierarchy_count": _safe_float(
                    row.get("delta_hierarchy_count"), float("nan")
                ),
                "delta_difference_count": _safe_float(
                    row.get("delta_difference_count"), float("nan")
                ),
                "delta_similarity_count": _safe_float(
                    row.get("delta_similarity_count"), float("nan")
                ),
                "delta_attribute_count": _safe_float(
                    row.get("delta_attribute_count"), float("nan")
                ),
                "delta_bridge_total_count": _safe_float(
                    row.get("delta_bridge_total_count"), float("nan")
                ),
                "delta_bridge_support_count": _safe_float(
                    row.get("delta_bridge_support_count"), float("nan")
                ),
                "delta_bridge_contrast_count": _safe_float(
                    row.get("delta_bridge_contrast_count"), float("nan")
                ),
                "primary_failure_category": primary,
                "llm_active_top1": flags["llm_active_top1"],
                "llm_active_gold": flags["llm_active_gold"],
                "nonlex_sparse_top1": flags["nonlex_sparse_top1"],
                "nonlex_sparse_gold": flags["nonlex_sparse_gold"],
                "similarity_present_in_panel": flags["similarity_present_in_panel"],
                "high_disagreement_gold": flags["high_disagreement_gold"],
                "high_disagreement_top1": flags["high_disagreement_top1"],
                "top1_bridge_total_count": _safe_float(
                    row.get("top1_bridge_total_count"), float("nan")
                ),
                "gold_bridge_total_count": _safe_float(
                    row.get("gold_bridge_total_count"), float("nan")
                ),
                "top1_bridge_support_count": _safe_float(
                    row.get("top1_bridge_support_count"), float("nan")
                ),
                "gold_bridge_support_count": _safe_float(
                    row.get("gold_bridge_support_count"), float("nan")
                ),
                "top1_bridge_contrast_count": _safe_float(
                    row.get("top1_bridge_contrast_count"), float("nan")
                ),
                "gold_bridge_contrast_count": _safe_float(
                    row.get("gold_bridge_contrast_count"), float("nan")
                ),
                "top1_has_nonlexical_bridge": bool(row.get("top1_has_nonlexical_bridge")),
                "gold_has_nonlexical_bridge": bool(row.get("gold_has_nonlexical_bridge")),
                "panel_complete": bool(row.get("panel_complete")),
                "missing_panel_record_count": int(row.get("missing_panel_record_count", 0) or 0),
                "top1_record_present": bool(row.get("top1_record_present")),
                "gold_record_present": bool(row.get("gold_record_present")),
            }
        )
    return pd.DataFrame(rows)
