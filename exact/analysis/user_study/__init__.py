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

from . import export as _export
from .export import (
    _attribute_node_display,
    _build_explanation_backfill_model,
    _build_rationale_model,
    _build_study_mapping,
    _categorize_bridge_relevance,
    _edge_display_level,
    _load_configs_for_rationale,
    _mean_nonempty,
    _resolve_run_dataset_paths,
    _selected_records,
    _write_json,
)
from .notebook import _write_notebook
from .selection import (
    DEFAULT_PER_RANK,
    DEFAULT_SHORTLIST_PER_RANK,
    DEFAULT_TOP_K,
    RunAnalysisArtifacts,
    _attribute_items,
    _channel_items,
    _default_output_dir,
    _eligible_panels,
    _final_selection,
    _hierarchy_items,
    _merge_review_sheet,
    _ordered_path_nodes,
    _safe_float,
    _safe_text,
    _setup_logger,
    _shortlist_panels,
    load_run_analysis,
)
from .taxonomy import _failure_taxonomy


def _backfill_explanation_fields(*args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
    """Compatibility facade preserving package-level monkeypatch behavior."""
    _export._build_explanation_backfill_model = _build_explanation_backfill_model
    return _export._backfill_explanation_fields(*args, **kwargs)


def _backfill_rationales(*args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
    """Compatibility facade preserving package-level monkeypatch behavior."""
    _export._build_rationale_model = _build_rationale_model
    return _export._backfill_rationales(*args, **kwargs)


def run_user_study_analysis(
    run_dir: Path,
    output_dir: Optional[Path] = None,
    top_k: int = DEFAULT_TOP_K,
    per_rank: int = DEFAULT_PER_RANK,
    shortlist_per_rank: int = DEFAULT_SHORTLIST_PER_RANK,
    seed: int = 0,
    backfill_explanations: bool = True,
    generate_rationales: bool = True,
    config_path: Optional[Path] = None,
    device: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
    jvm_heap_size: Optional[str] = None,
) -> Dict[str, Path]:
    del seed
    if jvm_heap_size is not None:
        warnings.warn(
            "jvm_heap_size is deprecated and ignored; Exact-OM no longer needs Java.",
            DeprecationWarning,
            stacklevel=2,
        )
    logger = logger or _setup_logger()
    logger.info(
        "Starting user-study analysis: run_dir=%s, output_dir=%s, top_k=%d, per_rank=%d, shortlist_per_rank=%d, backfill_explanations=%s, generate_rationales=%s",
        run_dir,
        output_dir or _default_output_dir(run_dir),
        top_k,
        per_rank,
        shortlist_per_rank,
        backfill_explanations,
        generate_rationales,
    )
    artifacts = load_run_analysis(
        run_dir=run_dir,
        output_dir=output_dir,
        top_k=top_k,
        config_path=config_path,
        logger=logger,
    )
    output_dir = artifacts.output_dir
    output_paths: Dict[str, Path] = {}

    pair_metrics_path = output_dir / "pair_metrics.csv"
    artifacts.pair_df.to_csv(pair_metrics_path, index=False)
    output_paths["pair_metrics_csv"] = pair_metrics_path

    source_panels_path = output_dir / "source_panels.csv"
    artifacts.source_df.to_csv(source_panels_path, index=False)
    output_paths["source_panels_csv"] = source_panels_path

    eligible_df = _eligible_panels(artifacts.source_df, top_k=top_k)
    logger.info("Eligible panels: %d", len(eligible_df))
    for rank in range(1, top_k + 1):
        count = int((eligible_df["gold_rank"] == rank).sum())
        if count < per_rank:
            raise ValueError(
                f"Not enough eligible source panels for rank {rank}: need {per_rank}, found {count}."
            )
    eligible_path = output_dir / "eligible_panels.csv"
    eligible_df.to_csv(eligible_path, index=False)
    output_paths["eligible_panels_csv"] = eligible_path

    shortlist_df = _shortlist_panels(
        eligible_df, shortlist_per_rank=shortlist_per_rank, per_rank=per_rank
    )
    logger.info("Shortlisted panels: %d", len(shortlist_df))
    shortlist_path = output_dir / "study_shortlist.csv"
    shortlist_df.to_csv(shortlist_path, index=False)
    output_paths["study_shortlist_csv"] = shortlist_path

    review_path = output_dir / "study_selection_review.csv"
    review_df = _merge_review_sheet(shortlist_df, review_path, per_rank=per_rank)
    review_df.to_csv(review_path, index=False)
    output_paths["study_selection_review_csv"] = review_path

    selected_df = _final_selection(review_df, per_rank=per_rank)
    logger.info("Final selected panels: %d", len(selected_df))
    selected_records = _selected_records(
        selected_df=selected_df,
        ranked_candidates_by_source=artifacts.ranked_candidates_by_source,
        record_index=artifacts.record_index,
        top_k=top_k,
    )
    logger.info("Selected panel records: %d", len(selected_records))
    selected_records_path = output_dir / "study_selected_records.json"
    _write_json(selected_records_path, selected_records)
    output_paths["study_selected_records_json"] = selected_records_path

    selected_records = _backfill_explanation_fields(
        selected_records,
        run_dir=run_dir,
        output_dir=output_dir,
        logger=logger,
        configs=artifacts.configs,
        config_path=artifacts.config_path,
        device=device,
        backfill_explanations=backfill_explanations,
    )
    selected_records_with_rationales = _backfill_rationales(
        selected_records,
        output_dir=output_dir,
        logger=logger,
        configs=artifacts.configs,
        config_path=artifacts.config_path,
        device=device,
        generate_rationales=generate_rationales,
    )
    selected_records_with_rationales_path = (
        output_dir / "study_selected_records_with_rationales.json"
    )
    _write_json(selected_records_with_rationales_path, selected_records_with_rationales)
    output_paths["study_selected_records_with_rationales_json"] = (
        selected_records_with_rationales_path
    )

    selected_record_index = {
        (_safe_text(record.get("src_iri")), _safe_text(record.get("tgt_iri"))): record
        for record in selected_records_with_rationales
    }
    study_mapping = _build_study_mapping(
        selected_df=selected_df,
        ranked_candidates_by_source=artifacts.ranked_candidates_by_source,
        record_index=selected_record_index,
        top_k=top_k,
        logger=logger,
    )
    study_mapping_path = output_dir / "study_mapping.json"
    _write_json(study_mapping_path, study_mapping)
    output_paths["study_mapping_json"] = study_mapping_path

    failure_df = _failure_taxonomy(artifacts.source_df, artifacts.pair_df, top_k=top_k)
    failure_path = output_dir / "failure_taxonomy.csv"
    failure_df.to_csv(failure_path, index=False)
    output_paths["failure_taxonomy_csv"] = failure_path

    notebook_path = _write_notebook(output_dir)
    output_paths["notebook"] = notebook_path

    logger.info("User-study analysis artifacts written to %s", output_dir)
    return output_paths


__all__ = [
    "RunAnalysisArtifacts",
    "load_run_analysis",
    "run_user_study_analysis",
]
