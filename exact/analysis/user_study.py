from __future__ import annotations

import ast
import copy
import json
import logging
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import torch

from exact.utils.data import read_yaml
from exact.utils.formatting import format_duration as _format_duration

TRUTHY_STRINGS = {"1", "true", "yes", "y", "keep", "selected"}
DEFAULT_TOP_K = 5
DEFAULT_PER_RANK = 4
DEFAULT_SHORTLIST_PER_RANK = 8
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class RunAnalysisArtifacts:
    run_dir: Path
    output_dir: Path
    config_path: Path
    ranking_path: Path
    explanations_path: Path
    summary_metrics_path: Optional[Path]
    configs: Optional[Any]
    record_index: Dict[Tuple[str, str], Dict[str, Any]]
    pair_df: pd.DataFrame
    source_df: pd.DataFrame
    ranked_candidates_by_source: Dict[str, List[Tuple[str, float]]]


def _resolve_run_paths(
    run_dir: Path, config_path: Optional[Path] = None
) -> Tuple[Path, Path, Path, Optional[Path]]:
    ranking_path = run_dir / "model" / "alignment" / "src2tgt.maps_local.tsv"
    explanations_path = run_dir / "model" / "alignment" / "default" / "full_explanations.json"
    summary_metrics_path = run_dir / "model" / "alignment" / "default" / "summary_metrics.csv"
    resolved_config = config_path or (run_dir / "config.yaml")
    if not ranking_path.exists():
        raise FileNotFoundError(f"Local ranking file not found: {ranking_path}")
    if not explanations_path.exists():
        raise FileNotFoundError(f"Full explanations JSON not found: {explanations_path}")
    if not resolved_config.exists():
        raise FileNotFoundError(f"Config file not found: {resolved_config}")
    return (
        ranking_path,
        explanations_path,
        resolved_config,
        summary_metrics_path if summary_metrics_path.exists() else None,
    )


def _default_output_dir(run_dir: Path) -> Path:
    return run_dir / "analysis" / "user_study"


def _setup_logger(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("exact.user_study")
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
    return logger


def _read_summary_metrics(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None or not path.exists():
        return None
    return pd.read_csv(path, sep="\t")


def _summary_index(summary_df: Optional[pd.DataFrame]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    if summary_df is None or summary_df.empty:
        return {}
    rows: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in summary_df.to_dict(orient="records"):
        src = str(row.get("src_iri") or "")
        tgt = str(row.get("tgt_iri") or "")
        if src and tgt:
            rows[(src, tgt)] = row
    return rows


def _normalize_candidate_list(raw_candidates: Any) -> List[Tuple[str, float]]:
    values = (
        ast.literal_eval(raw_candidates)
        if isinstance(raw_candidates, str)
        else list(raw_candidates or [])
    )
    normalized: List[Tuple[str, float]] = []
    total = max(1, len(values))
    for idx, item in enumerate(values):
        if isinstance(item, str):
            normalized.append((item, float((total - idx) / total)))
            continue
        if isinstance(item, (tuple, list)) and item:
            tgt = str(item[0])
            score = float(item[1]) if len(item) > 1 else float((total - idx) / total)
            normalized.append((tgt, score))
            continue
        raise ValueError(f"Unsupported candidate entry: {item!r}")
    normalized.sort(key=lambda pair: pair[1], reverse=True)
    return normalized


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value in ("", None):
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _selected_label_list(record: Mapping[str, Any], side: str) -> List[str]:
    labels = dict(record.get("selected_labels") or {})
    value = _safe_text(labels.get(side))
    return [value] if value else []


def _coalesce(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def _hierarchy_items(record: Mapping[str, Any], side: str) -> List[Dict[str, Any]]:
    triple_attributions = (record.get("triple_attributions") or {}).get("hierarchy") or {}
    items: List[Dict[str, Any]] = []
    for family, payload in triple_attributions.items():
        for item in payload.get(side, []) or []:
            row = dict(item or {})
            row["family"] = family
            items.append(row)
    if items:
        return items
    context_triples = (record.get("context_triples") or {}).get(f"hierarchy_{side}") or {}
    for family, triples in context_triples.items():
        for triple in triples or []:
            items.append({"triple": list(triple), "family": family})
    return items


def _channel_items(record: Mapping[str, Any], channel: str, side: str) -> List[Dict[str, Any]]:
    triple_attributions = (record.get("triple_attributions") or {}).get(channel) or {}
    items = [dict(item or {}) for item in (triple_attributions.get(side) or [])]
    if items:
        return items
    context_triples = (record.get("context_triples") or {}).get(f"{channel}_{side}") or []
    return [{"triple": list(triple)} for triple in context_triples]


def _attribute_items(record: Mapping[str, Any], side: str) -> List[Dict[str, Any]]:
    return [dict(item or {}) for item in ((record.get("attributes") or {}).get(side) or [])]


def _structural_evidence_counts(record: Mapping[str, Any]) -> Dict[str, int]:
    hierarchy_count = len(_hierarchy_items(record, "source")) + len(
        _hierarchy_items(record, "target")
    )
    similarity_count = len(_channel_items(record, "similarity", "source")) + len(
        _channel_items(record, "similarity", "target")
    )
    difference_count = len(_channel_items(record, "difference", "source")) + len(
        _channel_items(record, "difference", "target")
    )
    attribute_count = len(_attribute_items(record, "source")) + len(
        _attribute_items(record, "target")
    )
    present_channels = (
        int(hierarchy_count > 0)
        + int(similarity_count > 0)
        + int(difference_count > 0)
        + int(attribute_count > 0)
    )
    return {
        "hierarchy_count": hierarchy_count,
        "similarity_count": similarity_count,
        "difference_count": difference_count,
        "attribute_count": attribute_count,
        "present_channels": present_channels,
        "total_evidence": hierarchy_count + similarity_count + difference_count + attribute_count,
    }


def _bridge_metrics(record: Mapping[str, Any]) -> Dict[str, int]:
    provenance = record.get("cross_side_provenance") or {}
    lexical_count = len(list(provenance.get("lexical") or []))
    hierarchy_count = sum(
        len(list(links or [])) for links in dict(provenance.get("hierarchy") or {}).values()
    )
    similarity_count = len(list(provenance.get("similarity") or []))
    attribute_payload = dict(provenance.get("attributes") or {})
    attribute_count = len(list(attribute_payload.get("source") or [])) + len(
        list(attribute_payload.get("target") or [])
    )
    difference_payload = dict(provenance.get("difference") or {})
    difference_count = len(list(difference_payload.get("source") or [])) + len(
        list(difference_payload.get("target") or [])
    )
    support_count = lexical_count + hierarchy_count + similarity_count + attribute_count
    contrast_count = difference_count
    total = support_count + contrast_count
    return {
        "bridge_total_count": int(total),
        "bridge_lexical_count": int(lexical_count),
        "bridge_hierarchy_count": int(hierarchy_count),
        "bridge_similarity_count": int(similarity_count),
        "bridge_attribute_count": int(attribute_count),
        "bridge_difference_count": int(difference_count),
        "bridge_support_count": int(support_count),
        "bridge_contrast_count": int(contrast_count),
        "has_nonlexical_bridge": int(
            (hierarchy_count + similarity_count + attribute_count + difference_count) > 0
        ),
    }


def _categorize_decision_basis(i_label: float, i_struct: float, i_llm: float) -> Dict[str, Any]:
    if i_llm >= 0.20:
        return {
            "label": "LLM-assisted",
            "description": "The final decision needed language-model arbitration after the earlier signals remained ambiguous.",
        }
    if i_label >= 0.65 and (i_label - i_struct) >= 0.15:
        return {
            "label": "Lexical-led",
            "description": "The decision relied mostly on label similarity rather than ontology context.",
        }
    if i_struct >= 0.50 and (i_struct - i_label) >= 0.10:
        return {
            "label": "Context-led",
            "description": "The decision relied mostly on ontology structure and contextual evidence.",
        }
    return {
        "label": "Mixed",
        "description": "Labels and ontology context both contributed meaningfully to the final decision.",
    }


def _categorize_evidence_strength(
    i_struct: float, q_struct: float, total_evidence: int
) -> Dict[str, Any]:
    volume_ratio = min(1.0, total_evidence / 10.0)
    strength_value = 0.45 * i_struct + 0.35 * q_struct + 0.20 * volume_ratio
    if strength_value >= 0.60:
        label = "Strong"
        description = (
            "The candidate is supported by substantial and reasonably reliable contextual evidence."
        )
    elif strength_value >= 0.32:
        label = "Moderate"
        description = "There is some useful contextual support, but it is not overwhelming."
    else:
        label = "Weak"
        description = "The candidate relies on sparse or low-impact contextual support."
    return {
        "label": label,
        "value": float(strength_value),
        "description": description,
    }


def _categorize_evidence_agreement(u_dis: float, total_evidence: int) -> Dict[str, Any]:
    if total_evidence <= 0:
        return {
            "label": "Limited",
            "value": float(u_dis),
            "description": "Very little contextual evidence is present, so cross-signal agreement is hard to assess.",
        }
    if u_dis >= 0.18:
        label = "Conflicting"
        description = "Different evidence sources point in noticeably different directions."
    elif u_dis >= 0.08:
        label = "Mixed"
        description = "The evidence is not fully aligned; some signals support the match more strongly than others."
    else:
        label = "Consistent"
        description = "The available evidence sources broadly support the same overall reading of the candidate."
    return {
        "label": label,
        "value": float(u_dis),
        "description": description,
    }


def _categorize_explanation_coverage(
    present_channels: int, counts: Mapping[str, int]
) -> Dict[str, Any]:
    if present_channels >= 3:
        label = "Rich"
        description = "The explanation draws from several different kinds of contextual evidence."
    elif present_channels == 2:
        label = "Balanced"
        description = (
            "The explanation includes more than one evidence type, but it is not especially broad."
        )
    else:
        label = "Narrow"
        description = (
            "The explanation relies on only one evidence family or very limited contextual support."
        )
    return {
        "label": label,
        "channels_present": int(present_channels),
        "description": description,
        "details": {
            "hierarchy": int(counts.get("hierarchy_count", 0)),
            "similarity": int(counts.get("similarity_count", 0)),
            "difference": int(counts.get("difference_count", 0)),
            "attributes": int(counts.get("attribute_count", 0)),
        },
    }


def _categorize_lead_over_next(score: float, next_score: Optional[float]) -> Dict[str, Any]:
    if next_score is None:
        return {
            "label": "Last shown",
            "margin": None,
            "description": "This is the last displayed candidate, so there is no lower-ranked candidate shown beneath it.",
        }
    margin = float(score - next_score)
    if margin <= 0.02:
        label = "Near tie"
        description = "This candidate is only marginally ahead of the next one."
    elif margin <= 0.08:
        label = "Close call"
        description = (
            "This candidate is ahead, but the separation from the next one is still fairly small."
        )
    else:
        label = "Clear lead"
        description = "This candidate is comfortably ahead of the next displayed alternative."
    return {
        "label": label,
        "margin": margin,
        "description": description,
    }


def _ordered_path_nodes(nodes: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    source_nodes = [dict(node) for node in nodes if node.get("type") == "Source"]
    target_nodes = [dict(node) for node in nodes if node.get("type") == "Target"]
    middle_nodes = [dict(node) for node in nodes if node.get("type") not in {"Source", "Target"}]
    return source_nodes + middle_nodes + target_nodes


def _normalize_pair_row(
    record: Mapping[str, Any], summary_row: Optional[Mapping[str, Any]] = None
) -> Dict[str, Any]:
    summary_row = summary_row or {}
    confidences = record.get("confidences") or {}
    importances = record.get("importances") or {}
    weights = record.get("weights") or {}
    prediction = record.get("prediction") or {}
    labels = record.get("selected_labels") or {}

    hierarchy_source = _hierarchy_items(record, "source")
    hierarchy_target = _hierarchy_items(record, "target")
    similarity_source = _channel_items(record, "similarity", "source")
    similarity_target = _channel_items(record, "similarity", "target")
    difference_source = _channel_items(record, "difference", "source")
    difference_target = _channel_items(record, "difference", "target")
    attributes_source = _attribute_items(record, "source")
    attributes_target = _attribute_items(record, "target")

    hierarchy_count = len(hierarchy_source) + len(hierarchy_target)
    similarity_count = len(similarity_source) + len(similarity_target)
    difference_count = len(difference_source) + len(difference_target)
    attribute_count = len(attributes_source) + len(attributes_target)
    bridge_counts = _bridge_metrics(record)

    llm_decision = _safe_text(prediction.get("llm_decision"))
    i_llm = _safe_float(importances.get("I_llm", summary_row.get("I_llm", 0.0)))

    return {
        "src_iri": _safe_text(record.get("src_iri")),
        "tgt_iri": _safe_text(record.get("tgt_iri")),
        "source_label": _safe_text(labels.get("source")),
        "target_label": _safe_text(labels.get("target")),
        "ground_truth": int(
            _safe_float(prediction.get("ground_truth", summary_row.get("ground_truth", 0)))
        ),
        "threshold_positive": bool(
            prediction.get("threshold_positive", summary_row.get("threshold_positive", False))
        ),
        "saved_alignment_member": bool(
            prediction.get(
                "saved_alignment_member", summary_row.get("saved_alignment_member", False)
            )
        ),
        "rationale_positive": bool(
            prediction.get("rationale_positive", summary_row.get("rationale_positive", False))
        ),
        "llm_decision": llm_decision,
        "llm_rationale_present": bool(_safe_text(prediction.get("llm_rationale"))),
        "pair_brief_present": bool(_safe_text(record.get("llm_pair_brief"))),
        "S_final": _safe_float(confidences.get("S_final", summary_row.get("S_final", 0.0))),
        "s_label": _safe_float(confidences.get("s_label", summary_row.get("s_label", 0.0))),
        "S_struct": _safe_float(confidences.get("S_struct", summary_row.get("S_struct", 0.0))),
        "p_llm": _safe_float(confidences.get("p_llm", summary_row.get("p_llm", 0.0))),
        "I_label": _safe_float(importances.get("I_label", summary_row.get("I_label", 0.0))),
        "I_struct": _safe_float(importances.get("I_struct", summary_row.get("I_struct", 0.0))),
        "I_llm": i_llm,
        "I_hier": _safe_float(importances.get("I_hier", summary_row.get("I_hier", 0.0))),
        "I_sim": _safe_float(importances.get("I_sim", summary_row.get("I_sim", 0.0))),
        "I_diff": _safe_float(importances.get("I_diff", summary_row.get("I_diff", 0.0))),
        "I_attr": _safe_float(importances.get("I_attr", summary_row.get("I_attr", 0.0))),
        "U": _safe_float(weights.get("U", summary_row.get("U", 0.0))),
        "U_dis": _safe_float(weights.get("U_dis", summary_row.get("U_dis", 0.0))),
        "hierarchy_count": hierarchy_count,
        "similarity_count": similarity_count,
        "difference_count": difference_count,
        "attribute_count": attribute_count,
        "nonlex_total": hierarchy_count + similarity_count + difference_count + attribute_count,
        **bridge_counts,
        "llm_active": bool(i_llm > 0.0 or llm_decision),
        "full_record_present": True,
    }


def _build_pair_dataframe(
    records: Sequence[Mapping[str, Any]],
    summary_lookup: Mapping[Tuple[str, str], Mapping[str, Any]],
) -> Tuple[pd.DataFrame, Dict[Tuple[str, str], Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    record_index: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for record in records:
        key = (_safe_text(record.get("src_iri")), _safe_text(record.get("tgt_iri")))
        row = _normalize_pair_row(record, summary_lookup.get(key))
        rows.append(row)
        record_index[key] = dict(record)
    return pd.DataFrame(rows), record_index


def _label_indexes(pair_df: pd.DataFrame) -> Tuple[Dict[str, str], Dict[str, str]]:
    source_labels: Dict[str, str] = {}
    target_labels: Dict[str, str] = {}
    for row in pair_df.to_dict(orient="records"):
        src = _safe_text(row.get("src_iri"))
        tgt = _safe_text(row.get("tgt_iri"))
        if src and _safe_text(row.get("source_label")) and src not in source_labels:
            source_labels[src] = _safe_text(row.get("source_label"))
        if tgt and _safe_text(row.get("target_label")) and tgt not in target_labels:
            target_labels[tgt] = _safe_text(row.get("target_label"))
    return source_labels, target_labels


def _build_source_dataframe(
    ranking_path: Path,
    pair_lookup: Mapping[Tuple[str, str], Mapping[str, Any]],
    top_k: int,
) -> Tuple[pd.DataFrame, Dict[str, List[Tuple[str, float]]]]:
    source_rows: List[Dict[str, Any]] = []
    rankings = pd.read_csv(ranking_path, sep="\t")
    source_labels, target_labels = _label_indexes(pd.DataFrame(pair_lookup.values()))
    ranked_candidates_by_source: Dict[str, List[Tuple[str, float]]] = {}

    for row in rankings.to_dict(orient="records"):
        src = _safe_text(row.get("SrcEntity"))
        gold = _safe_text(row.get("TgtEntity"))
        ranked = _normalize_candidate_list(row.get("TgtCandidates"))
        ranked_candidates_by_source[src] = ranked
        gold_rank = next(
            (idx for idx, (tgt, _score) in enumerate(ranked, start=1) if tgt == gold), None
        )
        top_candidates = list(ranked[:top_k])
        missing_topk_records = 0
        pair_rows: List[Mapping[str, Any]] = []
        for tgt, _score in top_candidates:
            pair_row = pair_lookup.get((src, tgt))
            if pair_row is None:
                missing_topk_records += 1
                continue
            pair_rows.append(pair_row)
        if len(top_candidates) < top_k:
            missing_topk_records += top_k - len(top_candidates)

        top1_tgt = top_candidates[0][0] if top_candidates else ""
        top1_candidate_score = float(top_candidates[0][1]) if top_candidates else 0.0
        gold_candidate_score = next((float(score) for tgt, score in ranked if tgt == gold), 0.0)

        top1_row = pair_lookup.get((src, top1_tgt))
        gold_row = pair_lookup.get((src, gold))

        source_label = _coalesce(
            source_labels.get(src),
            top1_row.get("source_label") if top1_row else None,
            gold_row.get("source_label") if gold_row else None,
            src,
        )
        top1_label = _coalesce(
            target_labels.get(top1_tgt),
            top1_row.get("target_label") if top1_row else None,
            top1_tgt,
        )
        gold_label = _coalesce(
            target_labels.get(gold),
            gold_row.get("target_label") if gold_row else None,
            gold,
        )

        pair_brief_count = sum(int(bool(item.get("pair_brief_present"))) for item in pair_rows)
        mean_i_struct = _mean([_safe_float(item.get("I_struct")) for item in pair_rows])
        mean_evidence_volume = _mean(
            [
                _safe_float(item.get("hierarchy_count"))
                + _safe_float(item.get("difference_count"))
                + _safe_float(item.get("attribute_count"))
                for item in pair_rows
            ]
        )
        mean_ambiguity = _mean([_safe_float(item.get("U")) for item in pair_rows])
        mean_bridge_total_count = _mean(
            [_safe_float(item.get("bridge_total_count")) for item in pair_rows]
        )
        mean_bridge_support_count = _mean(
            [_safe_float(item.get("bridge_support_count")) for item in pair_rows]
        )
        mean_bridge_contrast_count = _mean(
            [_safe_float(item.get("bridge_contrast_count")) for item in pair_rows]
        )
        panel_has_hierarchy = any(
            _safe_float(item.get("hierarchy_count")) > 0 for item in pair_rows
        )
        panel_has_difference = any(
            _safe_float(item.get("difference_count")) > 0 for item in pair_rows
        )
        panel_has_attributes = any(
            _safe_float(item.get("attribute_count")) > 0 for item in pair_rows
        )
        panel_has_similarity = any(
            _safe_float(item.get("similarity_count")) > 0 for item in pair_rows
        )
        panel_has_nonlexical_bridge = any(
            _safe_float(item.get("has_nonlexical_bridge")) > 0 for item in pair_rows
        )

        gold_score = _safe_float(
            gold_row.get("S_final") if gold_row else gold_candidate_score, gold_candidate_score
        )
        top1_score = _safe_float(
            top1_row.get("S_final") if top1_row else top1_candidate_score, top1_candidate_score
        )

        source_rows.append(
            {
                "src_iri": src,
                "source_label": _safe_text(source_label),
                "gold_tgt_iri": gold,
                "gold_tgt_label": _safe_text(gold_label),
                "top1_tgt_iri": top1_tgt,
                "top1_tgt_label": _safe_text(top1_label),
                "gold_rank": int(gold_rank) if gold_rank is not None else -1,
                "top1_score": top1_score,
                "gold_score": gold_score,
                "score_gap": top1_score - gold_score,
                "gold_s_label": _safe_float(
                    gold_row.get("s_label") if gold_row else None, float("nan")
                ),
                "top1_s_label": _safe_float(
                    top1_row.get("s_label") if top1_row else None, float("nan")
                ),
                "gold_S_struct": _safe_float(
                    gold_row.get("S_struct") if gold_row else None, float("nan")
                ),
                "top1_S_struct": _safe_float(
                    top1_row.get("S_struct") if top1_row else None, float("nan")
                ),
                "gold_I_label": _safe_float(
                    gold_row.get("I_label") if gold_row else None, float("nan")
                ),
                "top1_I_label": _safe_float(
                    top1_row.get("I_label") if top1_row else None, float("nan")
                ),
                "gold_I_struct": _safe_float(
                    gold_row.get("I_struct") if gold_row else None, float("nan")
                ),
                "top1_I_struct": _safe_float(
                    top1_row.get("I_struct") if top1_row else None, float("nan")
                ),
                "gold_U": _safe_float(gold_row.get("U") if gold_row else None, float("nan")),
                "top1_U": _safe_float(top1_row.get("U") if top1_row else None, float("nan")),
                "gold_U_dis": _safe_float(
                    gold_row.get("U_dis") if gold_row else None, float("nan")
                ),
                "top1_U_dis": _safe_float(
                    top1_row.get("U_dis") if top1_row else None, float("nan")
                ),
                "gold_llm_active": bool(gold_row.get("llm_active")) if gold_row else False,
                "top1_llm_active": bool(top1_row.get("llm_active")) if top1_row else False,
                "gold_hierarchy_count": _safe_float(
                    gold_row.get("hierarchy_count") if gold_row else None, float("nan")
                ),
                "top1_hierarchy_count": _safe_float(
                    top1_row.get("hierarchy_count") if top1_row else None, float("nan")
                ),
                "gold_similarity_count": _safe_float(
                    gold_row.get("similarity_count") if gold_row else None, float("nan")
                ),
                "top1_similarity_count": _safe_float(
                    top1_row.get("similarity_count") if top1_row else None, float("nan")
                ),
                "gold_difference_count": _safe_float(
                    gold_row.get("difference_count") if gold_row else None, float("nan")
                ),
                "top1_difference_count": _safe_float(
                    top1_row.get("difference_count") if top1_row else None, float("nan")
                ),
                "gold_attribute_count": _safe_float(
                    gold_row.get("attribute_count") if gold_row else None, float("nan")
                ),
                "top1_attribute_count": _safe_float(
                    top1_row.get("attribute_count") if top1_row else None, float("nan")
                ),
                "gold_bridge_total_count": _safe_float(
                    gold_row.get("bridge_total_count") if gold_row else None, float("nan")
                ),
                "top1_bridge_total_count": _safe_float(
                    top1_row.get("bridge_total_count") if top1_row else None, float("nan")
                ),
                "gold_bridge_support_count": _safe_float(
                    gold_row.get("bridge_support_count") if gold_row else None, float("nan")
                ),
                "top1_bridge_support_count": _safe_float(
                    top1_row.get("bridge_support_count") if top1_row else None, float("nan")
                ),
                "gold_bridge_contrast_count": _safe_float(
                    gold_row.get("bridge_contrast_count") if gold_row else None, float("nan")
                ),
                "top1_bridge_contrast_count": _safe_float(
                    top1_row.get("bridge_contrast_count") if top1_row else None, float("nan")
                ),
                "gold_has_nonlexical_bridge": (
                    bool(gold_row.get("has_nonlexical_bridge")) if gold_row else False
                ),
                "top1_has_nonlexical_bridge": (
                    bool(top1_row.get("has_nonlexical_bridge")) if top1_row else False
                ),
                "delta_s_label": _safe_float(
                    gold_row.get("s_label") if gold_row else None, float("nan")
                )
                - _safe_float(top1_row.get("s_label") if top1_row else None, float("nan")),
                "delta_S_struct": _safe_float(
                    gold_row.get("S_struct") if gold_row else None, float("nan")
                )
                - _safe_float(top1_row.get("S_struct") if top1_row else None, float("nan")),
                "delta_I_label": _safe_float(
                    gold_row.get("I_label") if gold_row else None, float("nan")
                )
                - _safe_float(top1_row.get("I_label") if top1_row else None, float("nan")),
                "delta_I_struct": _safe_float(
                    gold_row.get("I_struct") if gold_row else None, float("nan")
                )
                - _safe_float(top1_row.get("I_struct") if top1_row else None, float("nan")),
                "delta_U": _safe_float(gold_row.get("U") if gold_row else None, float("nan"))
                - _safe_float(top1_row.get("U") if top1_row else None, float("nan")),
                "delta_U_dis": _safe_float(
                    gold_row.get("U_dis") if gold_row else None, float("nan")
                )
                - _safe_float(top1_row.get("U_dis") if top1_row else None, float("nan")),
                "delta_hierarchy_count": _safe_float(
                    gold_row.get("hierarchy_count") if gold_row else None, float("nan")
                )
                - _safe_float(top1_row.get("hierarchy_count") if top1_row else None, float("nan")),
                "delta_similarity_count": _safe_float(
                    gold_row.get("similarity_count") if gold_row else None, float("nan")
                )
                - _safe_float(top1_row.get("similarity_count") if top1_row else None, float("nan")),
                "delta_difference_count": _safe_float(
                    gold_row.get("difference_count") if gold_row else None, float("nan")
                )
                - _safe_float(top1_row.get("difference_count") if top1_row else None, float("nan")),
                "delta_attribute_count": _safe_float(
                    gold_row.get("attribute_count") if gold_row else None, float("nan")
                )
                - _safe_float(top1_row.get("attribute_count") if top1_row else None, float("nan")),
                "delta_bridge_total_count": _safe_float(
                    gold_row.get("bridge_total_count") if gold_row else None, float("nan")
                )
                - _safe_float(
                    top1_row.get("bridge_total_count") if top1_row else None, float("nan")
                ),
                "delta_bridge_support_count": _safe_float(
                    gold_row.get("bridge_support_count") if gold_row else None, float("nan")
                )
                - _safe_float(
                    top1_row.get("bridge_support_count") if top1_row else None, float("nan")
                ),
                "delta_bridge_contrast_count": _safe_float(
                    gold_row.get("bridge_contrast_count") if gold_row else None, float("nan")
                )
                - _safe_float(
                    top1_row.get("bridge_contrast_count") if top1_row else None, float("nan")
                ),
                "panel_complete": bool(missing_topk_records == 0 and len(top_candidates) == top_k),
                "missing_panel_record_count": int(missing_topk_records),
                "top1_record_present": bool(top1_row),
                "gold_record_present": bool(gold_row),
                "pair_brief_count": int(pair_brief_count),
                "pair_brief_complete": bool(
                    pair_brief_count == top_k and len(top_candidates) == top_k
                ),
                "panel_has_hierarchy": bool(panel_has_hierarchy),
                "panel_has_difference": bool(panel_has_difference),
                "panel_has_attributes": bool(panel_has_attributes),
                "panel_has_similarity": bool(panel_has_similarity),
                "panel_has_nonlexical_bridge": bool(panel_has_nonlexical_bridge),
                "panel_structural_coverage": int(panel_has_hierarchy)
                + int(panel_has_difference)
                + int(panel_has_attributes),
                "mean_I_struct": mean_i_struct,
                "mean_evidence_volume": mean_evidence_volume,
                "mean_ambiguity": mean_ambiguity,
                "mean_bridge_total_count": mean_bridge_total_count,
                "mean_bridge_support_count": mean_bridge_support_count,
                "mean_bridge_contrast_count": mean_bridge_contrast_count,
                "ambiguity_distance_to_mid": abs(mean_ambiguity - 0.5),
                "topk_targets": json.dumps(
                    [tgt for tgt, _score in top_candidates], ensure_ascii=False
                ),
            }
        )
    return pd.DataFrame(source_rows), ranked_candidates_by_source


def load_run_analysis(
    run_dir: Path,
    output_dir: Optional[Path] = None,
    top_k: int = DEFAULT_TOP_K,
    config_path: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
) -> RunAnalysisArtifacts:
    logger = logger or _setup_logger()
    run_dir = run_dir.resolve()
    ranking_path, explanations_path, resolved_config, summary_metrics_path = _resolve_run_paths(
        run_dir, config_path
    )
    output_dir = (output_dir or _default_output_dir(run_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading run analysis artifacts from %s", run_dir)
    with explanations_path.open("r", encoding="utf-8") as fh:
        records = json.load(fh)
    summary_df = _read_summary_metrics(summary_metrics_path)
    pair_df, record_index = _build_pair_dataframe(records, _summary_index(summary_df))
    pair_lookup = {
        (str(row["src_iri"]), str(row["tgt_iri"])): row for row in pair_df.to_dict(orient="records")
    }
    source_df, ranked_candidates_by_source = _build_source_dataframe(
        ranking_path, pair_lookup, top_k=top_k
    )
    logger.info(
        "Loaded analysis artifacts: explanation_records=%d, pair_rows=%d, source_panels=%d, ranking_sources=%d",
        len(records),
        len(pair_df),
        len(source_df),
        len(ranked_candidates_by_source),
    )
    return RunAnalysisArtifacts(
        run_dir=run_dir,
        output_dir=output_dir,
        config_path=resolved_config,
        ranking_path=ranking_path,
        explanations_path=explanations_path,
        summary_metrics_path=summary_metrics_path,
        configs=None,
        record_index=record_index,
        pair_df=pair_df,
        source_df=source_df,
        ranked_candidates_by_source=ranked_candidates_by_source,
    )


def _eligible_panels(source_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    eligible = source_df[
        source_df["panel_complete"] & source_df["gold_rank"].between(1, top_k)
    ].copy()
    eligible["similarity_bonus"] = eligible["panel_has_similarity"].astype(int)
    eligible = eligible.sort_values(
        by=[
            "gold_rank",
            "pair_brief_count",
            "panel_structural_coverage",
            "mean_I_struct",
            "mean_evidence_volume",
            "ambiguity_distance_to_mid",
            "similarity_bonus",
            "src_iri",
        ],
        ascending=[True, False, False, False, False, True, False, True],
    ).reset_index(drop=True)
    eligible["auto_rank_within_bucket"] = eligible.groupby("gold_rank").cumcount() + 1
    return eligible


def _shortlist_panels(
    eligible_df: pd.DataFrame, shortlist_per_rank: int, per_rank: int
) -> pd.DataFrame:
    shortlist = eligible_df.groupby("gold_rank", group_keys=False).head(shortlist_per_rank).copy()
    shortlist["recommended_keep"] = shortlist["auto_rank_within_bucket"] <= per_rank
    return shortlist.reset_index(drop=True)


def _merge_review_sheet(
    shortlist_df: pd.DataFrame, review_path: Path, per_rank: int
) -> pd.DataFrame:
    review_df = shortlist_df.copy()
    review_df["recommended_keep"] = review_df["auto_rank_within_bucket"] <= per_rank
    review_df["keep"] = ""
    review_df["drop_reason"] = ""
    review_df["review_note"] = ""
    if review_path.exists():
        existing = pd.read_csv(review_path)
        merge_cols = ["src_iri", "gold_rank"]
        for col in ["keep", "drop_reason", "review_note"]:
            if col not in existing.columns:
                existing[col] = ""
        existing = existing[merge_cols + ["keep", "drop_reason", "review_note"]]
        review_df = review_df.drop(columns=["keep", "drop_reason", "review_note"]).merge(
            existing,
            on=merge_cols,
            how="left",
        )
        review_df["keep"] = review_df["keep"].fillna("")
        review_df["drop_reason"] = review_df["drop_reason"].fillna("")
        review_df["review_note"] = review_df["review_note"].fillna("")
    return review_df


def _is_truthy_keep(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in TRUTHY_STRINGS


def _review_is_edited(review_df: pd.DataFrame) -> bool:
    for col in ["keep", "drop_reason", "review_note"]:
        if col not in review_df.columns:
            continue
        if review_df[col].fillna("").astype(str).str.strip().ne("").any():
            return True
    return False


def _final_selection(review_df: pd.DataFrame, per_rank: int) -> pd.DataFrame:
    if _review_is_edited(review_df):
        selected = review_df[review_df["keep"].map(_is_truthy_keep)].copy()
    else:
        selected = review_df[review_df["recommended_keep"]].copy()
    counts = selected.groupby("gold_rank").size().to_dict()
    required_ranks = sorted(review_df["gold_rank"].unique().tolist())
    for rank in required_ranks:
        if counts.get(rank, 0) != per_rank:
            raise ValueError(
                f"Selection must contain exactly {per_rank} sources for rank {rank}, found {counts.get(rank, 0)}."
            )
    return selected.sort_values(by=["gold_rank", "auto_rank_within_bucket", "src_iri"]).reset_index(
        drop=True
    )


def _bind_model_logger(model: Any, logger: logging.Logger) -> Any:
    def _model_log(message: Any, level: str = "info", *args: Any, **kwargs: Any) -> None:
        log_fn = getattr(logger, str(level).lower(), logger.info)
        log_fn(str(message))

    model.log = _model_log
    return model


def _load_configs_for_rationale(config_path: Path) -> Any:
    from exact.impl import bootstrap_components

    bootstrap_components()
    from exact.core.entities.configs.config import ConfigModel

    configs = ConfigModel.load_config(config_path)
    configs.resolve_dependencies()
    return configs


def _resolve_run_dataset_paths(
    run_dir: Path, logger: logging.Logger
) -> Tuple[Optional[Path], Optional[Path]]:
    candidate_specs = sorted(run_dir.glob("*.yaml")) + sorted(run_dir.glob("*.yml"))
    for spec_path in candidate_specs:
        try:
            payload = read_yaml(spec_path)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Skipping unreadable run spec %s: %s", spec_path, exc)
            continue
        dataset_cfg = dict(payload.get("dataset") or {})
        data_dir = dataset_cfg.get("data_dir")
        source_name = dataset_cfg.get("source")
        target_name = dataset_cfg.get("target")
        if not data_dir or not source_name or not target_name:
            continue
        data_dir_path = Path(str(data_dir))
        if not data_dir_path.is_absolute():
            candidate_roots = [
                (spec_path.parent / data_dir_path).resolve(),
                (PROJECT_ROOT / data_dir_path).resolve(),
            ]
            existing_root = next((root for root in candidate_roots if root.exists()), None)
            data_dir_path = existing_root or candidate_roots[0]
        source_path = Path(str(source_name))
        target_path = Path(str(target_name))
        if not source_path.is_absolute():
            source_path = (data_dir_path / source_path).resolve()
        if not target_path.is_absolute():
            target_path = (data_dir_path / target_path).resolve()
        if source_path.exists() and target_path.exists():
            return source_path, target_path
    return None, None


def _build_rationale_model(
    configs: Any, cache_dir: Path, device: torch.device, logger: logging.Logger
) -> Any:
    model_spec = configs.get_model_sequence()[0]
    model_cls = model_spec.name
    params = {
        **dict(model_spec.params or {}),
        "llm_profiles": {k: v.model_dump() for k, v in configs.llm_profiles.items()},
        "llm_routing": configs.llm_routing.model_dump(),
        "request_seed": getattr(configs, "seed", None),
        "cache_dir": cache_dir,
        "use_lexical": False,
        "use_context": False,
        "use_llm": True,
        "return_explanations": False,
        "generate_llm_rationales": True,
    }
    params.update(configs.alignment_params.model_dump(exclude_none=True))
    model = _bind_model_logger(model_cls(device=device, **params), logger)
    return model


def _build_explanation_backfill_model(
    configs: Any,
    run_dir: Path,
    output_dir: Path,
    device: torch.device,
    logger: logging.Logger,
) -> Optional[Any]:
    model_spec = configs.get_model_sequence()[0]
    model_cls = model_spec.name
    if getattr(model_cls, "__name__", "") != "PairAdaptiveSemanticScorer":
        logger.info(
            "Explanation-field backfill is only supported for PairAdaptiveSemanticScorer; skipping targeted pair rehydrate."
        )
        return None
    params = {
        **dict(model_spec.params or {}),
        "llm_profiles": {k: v.model_dump() for k, v in configs.llm_profiles.items()},
        "llm_routing": configs.llm_routing.model_dump(),
        "request_seed": getattr(configs, "seed", None),
        "cache_dir": output_dir / "cache",
        "use_llm": False,
        "return_explanations": True,
        "generate_llm_rationales": False,
    }
    params.update(configs.alignment_params.model_dump(exclude_none=True))
    model = _bind_model_logger(model_cls(device=device, **params), logger)
    source_path, target_path = _resolve_run_dataset_paths(run_dir, logger)
    if source_path is None or target_path is None:
        logger.warning(
            "Could not resolve source/target ontology paths from %s; targeted explanation rehydrate will be unavailable, but saved-record reconstruction can still run.",
            run_dir,
        )
        return model
    dataset_cls = configs.dataset
    if dataset_cls is None:
        logger.warning(
            "Resolved config does not provide a dataset class; targeted explanation rehydrate unavailable, but saved-record reconstruction can still run."
        )
        return model
    dataset = dataset_cls(
        output_path=output_dir,
        logger=logger,
        cache_ok=True,
        device=device,
        llm_profiles={k: v.model_dump() for k, v in configs.llm_profiles.items()},
        llm_routing=configs.llm_routing.model_dump(),
        request_seed=getattr(configs, "seed", None),
        **configs.dataset_params.model_dump(),
    )
    dataset.load_ontologies(source_path, target_path)
    if hasattr(model, "attach_dataset"):
        model.attach_dataset(dataset)
    return model


def _triple_attributions_missing_item_ids(record: Mapping[str, Any]) -> bool:
    triple_attributions = record.get("triple_attributions") or {}
    hierarchy = triple_attributions.get("hierarchy") or {}
    for family_payload in hierarchy.values():
        payload = dict(family_payload or {})
        for side in ["source", "target"]:
            items = list(payload.get(side) or [])
            if any(not _safe_text(item.get("item_id")) for item in items):
                return True
    for channel in ["similarity", "difference"]:
        payload = dict(triple_attributions.get(channel) or {})
        for side in ["source", "target"]:
            items = list(payload.get(side) or [])
            if any(not _safe_text(item.get("item_id")) for item in items):
                return True
    return False


def _attributes_missing_item_ids(record: Mapping[str, Any]) -> bool:
    attributes = record.get("attributes") or {}
    for side in ["source", "target"]:
        items = list((attributes.get(side) or []))
        if any(not _safe_text(item.get("item_id")) for item in items):
            return True
    return False


def _record_missing_entity_provenance(record: Mapping[str, Any]) -> bool:
    triple_attributions = record.get("triple_attributions") or {}
    hierarchy = triple_attributions.get("hierarchy") or {}
    for family_payload in hierarchy.values():
        payload = dict(family_payload or {})
        for side in ["source", "target"]:
            for item in list(payload.get(side) or []):
                triple = list(item.get("triple") or [])
                if len(triple) < 3:
                    continue
                if not _safe_text(item.get("subject_iri")) or not _safe_text(
                    item.get("object_iri")
                ):
                    return True
    for channel in ["similarity", "difference"]:
        payload = dict(triple_attributions.get(channel) or {})
        for side in ["source", "target"]:
            for item in list(payload.get(side) or []):
                triple = list(item.get("triple") or [])
                if len(triple) < 3:
                    continue
                if not _safe_text(item.get("subject_iri")) or not _safe_text(
                    item.get("object_iri")
                ):
                    return True
    return False


def _record_has_missing_item_ids(record: Mapping[str, Any]) -> bool:
    return _triple_attributions_missing_item_ids(record) or _attributes_missing_item_ids(record)


def _record_needs_explanation_backfill(record: Mapping[str, Any]) -> bool:
    if int(record.get("explanation_schema_version", 0) or 0) < 3:
        return True
    if _record_has_missing_item_ids(record):
        return True
    if _record_missing_entity_provenance(record):
        return True
    provenance = record.get("cross_side_provenance") or {}
    return not bool(provenance)


def _merge_missing_explanation_fields(
    original: Dict[str, Any],
    repaired: Mapping[str, Any],
) -> Dict[str, Any]:
    merged = copy.deepcopy(original)
    if int(merged.get("explanation_schema_version", 0) or 0) < 3:
        merged["explanation_schema_version"] = int(
            repaired.get("explanation_schema_version", 3) or 3
        )
    if _triple_attributions_missing_item_ids(merged) or not merged.get("triple_attributions"):
        triple_attributions = repaired.get("triple_attributions")
        if triple_attributions:
            merged["triple_attributions"] = copy.deepcopy(triple_attributions)
    elif _record_missing_entity_provenance(merged):
        triple_attributions = repaired.get("triple_attributions")
        if triple_attributions:
            merged["triple_attributions"] = copy.deepcopy(triple_attributions)
    if _attributes_missing_item_ids(merged) or not merged.get("attributes"):
        attributes = repaired.get("attributes")
        if attributes:
            merged["attributes"] = copy.deepcopy(attributes)
    if not merged.get("cross_side_provenance"):
        provenance = repaired.get("cross_side_provenance")
        if provenance:
            merged["cross_side_provenance"] = copy.deepcopy(provenance)
    for key in ["context_sentences", "context_triples", "selected_labels"]:
        if not merged.get(key) and repaired.get(key):
            merged[key] = copy.deepcopy(repaired.get(key))
    return merged


def _backfill_explanation_fields(
    records: List[Dict[str, Any]],
    run_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
    configs: Optional[Any] = None,
    config_path: Optional[Path] = None,
    device: Optional[int] = None,
    backfill_explanations: bool = True,
    log_every: int = 10,
) -> List[Dict[str, Any]]:
    hydrated = [copy.deepcopy(record) for record in records]
    if not backfill_explanations:
        logger.info("Explanation-field backfill disabled; keeping selected records as-is.")
        return hydrated
    pending_indices = [
        idx for idx, record in enumerate(hydrated) if _record_needs_explanation_backfill(record)
    ]
    if not pending_indices:
        logger.info(
            "Selected records already contain provenance-backed explanation fields; skipping explanation backfill."
        )
        return hydrated
    if configs is None:
        if config_path is None:
            raise ValueError(
                "config_path is required to backfill explanation fields when configs is not provided."
            )
        logger.info("Loading configs for explanation backfill from %s", config_path)
        configs = _load_configs_for_rationale(config_path=config_path)

    device_obj = (
        torch.device(device)
        if device is not None and torch.cuda.is_available()
        else torch.device("cpu")
    )
    model = _build_explanation_backfill_model(
        configs, run_dir=run_dir, output_dir=output_dir, device=device_obj, logger=logger
    )
    processed = 0
    from_saved_record = 0
    from_pair_rehydrate = 0
    failed = 0
    started = time.perf_counter()
    logger.info(
        "Explanation backfill started: records=%d, pending=%d, model=%s, device=%s",
        len(hydrated),
        len(pending_indices),
        (
            getattr(getattr(configs.get_model_sequence()[0], "name", None), "__name__", "unknown")
            if configs
            else "unknown"
        ),
        device_obj,
    )
    for idx in pending_indices:
        record = hydrated[idx]
        repaired = None
        try:
            if model is not None and hasattr(model, "reconstruct_explanation_fields_from_record"):
                repaired = model.reconstruct_explanation_fields_from_record(record)
                from_saved_record += 1
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "Saved-record explanation reconstruction failed for (%s, %s): %s",
                record.get("src_iri"),
                record.get("tgt_iri"),
                exc,
            )
            repaired = None
        if (
            repaired is None
            and model is not None
            and hasattr(model, "reconstruct_explanation_fields_for_pair")
        ):
            try:
                repaired = model.reconstruct_explanation_fields_for_pair(
                    _safe_text(record.get("src_iri")),
                    _safe_text(record.get("tgt_iri")),
                    src_labels=_selected_label_list(record, "source"),
                    tgt_labels=_selected_label_list(record, "target"),
                )
                from_pair_rehydrate += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Targeted explanation rehydrate failed for (%s, %s): %s",
                    record.get("src_iri"),
                    record.get("tgt_iri"),
                    exc,
                )
                failed += 1
                repaired = None
        if repaired is not None:
            hydrated[idx] = _merge_missing_explanation_fields(record, repaired)
        processed += 1
        if processed == len(pending_indices) or processed % max(1, int(log_every)) == 0:
            elapsed = max(1e-8, time.perf_counter() - started)
            rate = processed / elapsed
            remaining = max(0, len(pending_indices) - processed)
            eta = _format_duration(remaining / rate) if rate > 0 else _format_duration(0.0)
            logger.info(
                "Explanation backfill progress: records=%d/%d, from_saved_record=%d, targeted_pair_rehydrate=%d, failed=%d, avg=%.2fs/record, ETA %s",
                processed,
                len(pending_indices),
                from_saved_record,
                from_pair_rehydrate,
                failed,
                elapsed / max(1, processed),
                eta,
            )
    elapsed = max(0.0, time.perf_counter() - started)
    logger.info(
        "Explanation backfill completed: pending=%d, from_saved_record=%d, targeted_pair_rehydrate=%d, failed=%d, duration=%s",
        len(pending_indices),
        from_saved_record,
        from_pair_rehydrate,
        failed,
        _format_duration(elapsed),
    )
    return hydrated


def _sync_record_model_usage(record: Dict[str, Any]) -> None:
    models = dict(record.get("models") or {})
    backend_usage = record.get("backend_usage") or {}
    summary_model = (backend_usage.get("summary") or {}).get("model")
    decision_model = (backend_usage.get("decision") or {}).get("model")
    rationale_model = (backend_usage.get("rationale") or {}).get("model")
    models["llm_summary_model"] = summary_model
    models["llm_decision_model"] = decision_model
    models["llm_rationale_model"] = rationale_model
    unique_models: List[str] = []
    for name in (summary_model, decision_model, rationale_model):
        if name and name not in unique_models:
            unique_models.append(name)
    if len(unique_models) == 1:
        models["llm_model"] = unique_models[0]
    elif len(unique_models) > 1:
        models["llm_model"] = "multiple"
    record["models"] = models


def _backfill_rationales(
    records: List[Dict[str, Any]],
    output_dir: Path,
    logger: logging.Logger,
    configs: Optional[Any] = None,
    config_path: Optional[Path] = None,
    device: Optional[int] = None,
    generate_rationales: bool = True,
    log_every: int = 5,
) -> List[Dict[str, Any]]:
    hydrated = [copy.deepcopy(record) for record in records]
    if not generate_rationales:
        logger.info("Rationale backfill disabled; keeping selected records as-is.")
        return hydrated
    pending = [
        record
        for record in hydrated
        if not _safe_text((record.get("prediction") or {}).get("llm_rationale"))
    ]
    if not pending:
        logger.info("Selected records already contain rationales; skipping rationale generation.")
        return hydrated
    if configs is None:
        if config_path is None:
            raise ValueError(
                "config_path is required to generate rationales when configs is not provided."
            )
        logger.info("Loading configs for rationale backfill from %s", config_path)
        configs = _load_configs_for_rationale(config_path=config_path)
    device_obj = (
        torch.device(device)
        if device is not None and torch.cuda.is_available()
        else torch.device("cpu")
    )
    logger.info("Initializing rationale model on device=%s", device_obj)
    model = _build_rationale_model(configs, output_dir / "cache", device_obj, logger)
    logger.info("Generating missing rationales for %d selected records", len(pending))

    progress_state: Dict[str, Any] = {
        "started": False,
        "start_time": None,
        "last_logged_uncached": 0,
        "interval_uncached_records": 0,
        "cached_records": 0,
        "uncached_records": 0,
        "uncached_unique_prompts": 0,
        "backend": None,
        "model": None,
        "concurrency": None,
    }

    def _progress_callback(event: Dict[str, Any]) -> None:
        if not event or event.get("stage") != "rationale":
            return
        event_type = str(event.get("event", ""))
        if event_type == "start":
            start_time = time.perf_counter()
            progress_state["started"] = True
            progress_state["start_time"] = start_time
            progress_state["cached_records"] = int(event.get("cached_records", 0) or 0)
            progress_state["uncached_records"] = int(event.get("uncached_records", 0) or 0)
            progress_state["uncached_unique_prompts"] = int(
                event.get("uncached_unique_prompts", 0) or 0
            )
            progress_state["backend"] = event.get("backend")
            progress_state["model"] = event.get("model")
            progress_state["concurrency"] = int(event.get("concurrency", 0) or 0)
            concurrency = max(1, progress_state["concurrency"] or 1)
            progress_state["interval_uncached_records"] = max(1, int(log_every)) * concurrency
            logger.info(
                "Rationale stage started: records=%d, uncached_records=%d, uncached_unique_prompts=%d, "
                "cached_records=%d, backend=%s, model=%s, concurrency=%d.",
                int(event.get("total_records", 0) or 0),
                progress_state["uncached_records"],
                progress_state["uncached_unique_prompts"],
                progress_state["cached_records"],
                progress_state["backend"],
                progress_state["model"],
                progress_state["concurrency"] or 1,
            )
            return
        if event_type != "progress" or not progress_state["started"]:
            return

        total_uncached = int(event.get("total_uncached_records", 0) or 0)
        completed_uncached = int(event.get("completed_uncached_records", 0) or 0)
        if total_uncached <= 0:
            return
        last_logged = int(progress_state["last_logged_uncached"] or 0)
        interval = int(progress_state["interval_uncached_records"] or 1)
        if completed_uncached < total_uncached and (completed_uncached - last_logged) < interval:
            return

        progress_state["last_logged_uncached"] = completed_uncached
        elapsed = max(1e-8, time.perf_counter() - float(progress_state["start_time"]))
        rate = completed_uncached / elapsed
        remaining = max(0, total_uncached - completed_uncached)
        eta = _format_duration(remaining / rate) if rate > 0 else _format_duration(0.0)
        avg_seconds = elapsed / max(1, completed_uncached)
        logger.info(
            "Rationale progress: uncached_records=%d/%d, unique_prompts=%d/%d, cached_records=%d, avg=%.2fs/record, ETA %s",
            completed_uncached,
            total_uncached,
            int(event.get("completed_unique_prompts", 0) or 0),
            int(event.get("total_unique_prompts", 0) or 0),
            int(event.get("cached_records", 0) or 0),
            avg_seconds,
            eta,
        )

    rationales = model.generate_final_rationales_for_records(
        pending, progress_callback=_progress_callback
    )
    rationale_meta = dict(getattr(model, "_last_rationale_backend_meta", {}) or {})
    for record, rationale in zip(pending, rationales):
        prediction = dict(record.get("prediction") or {})
        prediction["llm_rationale"] = rationale
        record["prediction"] = prediction
        backend_usage = dict(record.get("backend_usage") or {})
        backend_usage["rationale"] = dict(rationale_meta)
        record["backend_usage"] = backend_usage
        _sync_record_model_usage(record)
    if progress_state["started"]:
        elapsed = max(0.0, time.perf_counter() - float(progress_state["start_time"]))
        duration = _format_duration(elapsed)
        uncached_records = int(progress_state["uncached_records"] or 0)
        throughput = (
            (uncached_records / elapsed) if elapsed > 1e-8 and uncached_records > 0 else 0.0
        )
        avg_seconds = (elapsed / uncached_records) if uncached_records > 0 else 0.0
        logger.info(
            "Rationale stage completed: records=%d, uncached_records=%d, cached_records=%d, duration=%s, "
            "throughput=%.2f uncached records/s, avg=%.2fs/uncached record",
            len(hydrated),
            uncached_records,
            int(progress_state["cached_records"] or 0),
            duration,
            throughput,
            avg_seconds,
        )
    return hydrated


def _format_triple(triple: Sequence[Any]) -> str:
    subj = _safe_text(triple[0]) if len(triple) > 0 else ""
    rel = _safe_text(triple[1]) if len(triple) > 1 else ""
    obj = _safe_text(triple[2]) if len(triple) > 2 else ""
    return f"{subj} --{rel}--> {obj}"


def _node_id(
    display: str, node_type: str, used: Dict[Tuple[str, str], str], nodes: List[Dict[str, Any]]
) -> str:
    key = (node_type, display)
    if key in used:
        return used[key]
    candidate = display or node_type
    existing = {node["id"] for node in nodes}
    if candidate in existing:
        suffix = 2
        base = candidate
        while f"{base} [{suffix}]" in existing:
            suffix += 1
        candidate = f"{base} [{suffix}]"
    used[key] = candidate
    nodes.append({"id": candidate, "type": node_type})
    return candidate


def _append_edge(
    edges: List[Dict[str, Any]],
    seen_edges: set[Tuple[str, str, str, str]],
    source: str,
    target: str,
    label: str,
    score: Any,
    edge_type: str,
) -> None:
    key = (source, target, label, edge_type)
    if key in seen_edges:
        return
    seen_edges.add(key)
    edges.append(
        {
            "source": source,
            "target": target,
            "label": label,
            "score": score,
            "type": edge_type,
        }
    )


def _bridge_score_rank(score: Any) -> int:
    if isinstance(score, str):
        normalized = score.strip().lower()
        if normalized == "strong":
            return 3
        if normalized == "moderate":
            return 2
        if normalized == "weak":
            return 1
    return 0


def _append_bridge_edge(
    edges: List[Dict[str, Any]],
    bridge_index: Dict[Tuple[str, str, str], int],
    source: str,
    target: str,
    label: str,
    score: Any,
    edge_type: str,
) -> None:
    key = (source, target, edge_type)
    edge = {
        "source": source,
        "target": target,
        "label": label,
        "score": score,
        "type": edge_type,
    }
    current_idx = bridge_index.get(key)
    if current_idx is None:
        bridge_index[key] = len(edges)
        edges.append(edge)
        return
    current = edges[current_idx]
    if _bridge_score_rank(score) > _bridge_score_rank(current.get("score")):
        edges[current_idx] = edge


def _edge_identity(edge: Mapping[str, Any]) -> Tuple[str, str, str, str]:
    return (
        _safe_text(edge.get("source")),
        _safe_text(edge.get("target")),
        _safe_text(edge.get("label")),
        _safe_text(edge.get("type")),
    )


def _edge_display_level(edge: Mapping[str, Any]) -> Tuple[bool, int, str]:
    edge_type = _safe_text(edge.get("type"))
    if not edge_type.startswith("bridge-"):
        return False, 1, "Context edge"
    if edge_type == "bridge-contrast" or _safe_text(edge.get("label")).lower() == "label match":
        return True, 2, "Core bridge"
    if _safe_text(edge.get("score")).lower() == "weak":
        return True, 4, "Optional bridge"
    return True, 3, "Supporting bridge"


def _annotate_edge_metadata(paths: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not paths:
        return []
    updated_paths: List[Dict[str, Any]] = []
    for path in paths:
        new_path = dict(path)
        new_edges = []
        for edge in path.get("edges") or []:
            new_edge = dict(edge)
            is_bridge, level, level_label = _edge_display_level(edge)
            new_edge["bridge"] = is_bridge
            new_edge["level"] = level
            new_edge["level_label"] = level_label
            new_edges.append(new_edge)
        new_path["edges"] = new_edges
        updated_paths.append(new_path)
    return updated_paths


def _hierarchy_export_items(record: Mapping[str, Any], side: str) -> List[Dict[str, Any]]:
    return _hierarchy_items(record, side)


def _mean_nonempty(values: Sequence[float], default: float = 0.0) -> float:
    cleaned = [float(value) for value in values if value is not None]
    if not cleaned:
        return float(default)
    return float(sum(cleaned) / len(cleaned))


def _categorize_bridge_relevance(value: float) -> str:
    score = _safe_float(value, 0.0)
    if score >= 0.20:
        return "strong"
    if score >= 0.08:
        return "moderate"
    return "weak"


def _attribute_node_display(item: Mapping[str, Any]) -> str:
    value = _safe_text(item.get("value"))
    if value:
        return value
    text = _safe_text(item.get("text"))
    if ":" in text:
        suffix = text.split(":", 1)[1].strip()
        if suffix:
            return suffix
    if text:
        return text
    prop = _safe_text(item.get("property"))
    return prop or "attribute"


def _make_path_payload(
    record: Mapping[str, Any],
    rank: int,
    ground_truth: int,
    next_score: Optional[float] = None,
) -> Dict[str, Any]:
    labels = record.get("selected_labels") or {}
    prediction = record.get("prediction") or {}
    confidences = record.get("confidences") or {}
    importances = record.get("importances") or {}
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    used_nodes: Dict[Tuple[str, str], str] = {}
    seen_edges: set[Tuple[str, str, str, str]] = set()
    bridge_index: Dict[Tuple[str, str, str], int] = {}
    item_node_lookup: Dict[str, str] = {}
    item_importance_lookup: Dict[str, float] = {}

    source_text = _safe_text(labels.get("source")) or _safe_text(record.get("src_iri"))
    target_text = _safe_text(labels.get("target")) or _safe_text(record.get("tgt_iri"))
    src_iri = _safe_text(record.get("src_iri"))
    tgt_iri = _safe_text(record.get("tgt_iri"))

    source_node = _node_id(source_text, "Source", used_nodes, nodes)
    target_node = _node_id(target_text, "Target", used_nodes, nodes)
    item_node_lookup["__source__"] = source_node
    item_node_lookup["__target__"] = target_node

    def _context_node(display: Any, side: str) -> str:
        text = _safe_text(display)
        node_type = "source-context" if side == "source" else "target-context"
        return _node_id(text, node_type, used_nodes, nodes)

    def _triple_endpoint_node(value: Any, side: str) -> str:
        text = _safe_text(value)
        if side == "source" and text in {source_text, src_iri}:
            return source_node
        if side == "target" and text in {target_text, tgt_iri}:
            return target_node
        return _context_node(text, side)

    def _triple_item_node(subj: str, obj: str, side: str) -> str:
        endpoint = source_node if side == "source" else target_node
        if subj == endpoint and obj != endpoint:
            return obj
        if obj == endpoint and subj != endpoint:
            return subj
        return obj

    family_importances = dict(importances.get("family_importances") or {})
    i_label = _safe_float(importances.get("I_label"), 0.0)
    i_hier = _safe_float(importances.get("I_hier"), 0.0)
    i_sim = _safe_float(importances.get("I_sim"), 0.0)
    i_diff = _safe_float(importances.get("I_diff"), 0.0)
    i_attr = _safe_float(importances.get("I_attr"), 0.0)

    def _bridge_strength(channel_importance: float, local_mass: float) -> str:
        relevance = _safe_float(channel_importance, 0.0) * (
            0.5 + 0.5 * _safe_float(local_mass, 0.0)
        )
        return _categorize_bridge_relevance(relevance)

    for item in _hierarchy_export_items(record, "source"):
        triple = item.get("triple") or []
        if len(triple) < 3:
            continue
        subj = _triple_endpoint_node(triple[0], "source")
        obj = _triple_endpoint_node(triple[2], "source")
        relation = _safe_text(triple[1] if len(triple) > 1 else item.get("family")) or "hierarchy"
        score = _safe_float(item.get("importance", item.get("support", 1.0)), 1.0)
        _append_edge(edges, seen_edges, subj, obj, relation, score, edge_type="hierarchy")
        item_id = _safe_text(item.get("item_id"))
        if item_id:
            item_node_lookup[item_id] = _triple_item_node(subj, obj, "source")
            item_importance_lookup[item_id] = _safe_float(item.get("importance"), 0.0)
    for item in _hierarchy_export_items(record, "target"):
        triple = item.get("triple") or []
        if len(triple) < 3:
            continue
        subj = _triple_endpoint_node(triple[0], "target")
        obj = _triple_endpoint_node(triple[2], "target")
        relation = _safe_text(triple[1] if len(triple) > 1 else item.get("family")) or "hierarchy"
        score = _safe_float(item.get("importance", item.get("support", 1.0)), 1.0)
        _append_edge(edges, seen_edges, subj, obj, relation, score, edge_type="hierarchy")
        item_id = _safe_text(item.get("item_id"))
        if item_id:
            item_node_lookup[item_id] = _triple_item_node(subj, obj, "target")
            item_importance_lookup[item_id] = _safe_float(item.get("importance"), 0.0)

    for item in _channel_items(record, "similarity", "source"):
        triple = item.get("triple") or []
        if len(triple) < 3:
            continue
        subj = _triple_endpoint_node(triple[0], "source")
        obj = _triple_endpoint_node(triple[2], "source")
        relation = _safe_text(triple[1] if len(triple) > 1 else "similarity")
        score = _safe_float(item.get("importance", item.get("support", 1.0)), 1.0)
        _append_edge(edges, seen_edges, subj, obj, relation, score, edge_type="similarity")
        item_id = _safe_text(item.get("item_id"))
        if item_id:
            item_node_lookup[item_id] = _triple_item_node(subj, obj, "source")
            item_importance_lookup[item_id] = _safe_float(item.get("importance"), 0.0)
    for item in _channel_items(record, "similarity", "target"):
        triple = item.get("triple") or []
        if len(triple) < 3:
            continue
        subj = _triple_endpoint_node(triple[0], "target")
        obj = _triple_endpoint_node(triple[2], "target")
        relation = _safe_text(triple[1] if len(triple) > 1 else "similarity")
        score = _safe_float(item.get("importance", item.get("support", 1.0)), 1.0)
        _append_edge(edges, seen_edges, subj, obj, relation, score, edge_type="similarity")
        item_id = _safe_text(item.get("item_id"))
        if item_id:
            item_node_lookup[item_id] = _triple_item_node(subj, obj, "target")
            item_importance_lookup[item_id] = _safe_float(item.get("importance"), 0.0)

    for item in _channel_items(record, "difference", "source"):
        triple = item.get("triple") or []
        if len(triple) < 3:
            continue
        subj = _triple_endpoint_node(triple[0], "source")
        obj = _triple_endpoint_node(triple[2], "source")
        relation = _safe_text(triple[1] if len(triple) > 1 else "difference")
        score = _safe_float(item.get("importance", item.get("support", 1.0)), 1.0)
        _append_edge(edges, seen_edges, subj, obj, relation, score, edge_type="difference")
        item_id = _safe_text(item.get("item_id"))
        if item_id:
            item_node_lookup[item_id] = _triple_item_node(subj, obj, "source")
            item_importance_lookup[item_id] = _safe_float(item.get("importance"), 0.0)
    for item in _channel_items(record, "difference", "target"):
        triple = item.get("triple") or []
        if len(triple) < 3:
            continue
        subj = _triple_endpoint_node(triple[0], "target")
        obj = _triple_endpoint_node(triple[2], "target")
        relation = _safe_text(triple[1] if len(triple) > 1 else "difference")
        score = _safe_float(item.get("importance", item.get("support", 1.0)), 1.0)
        _append_edge(edges, seen_edges, subj, obj, relation, score, edge_type="difference")
        item_id = _safe_text(item.get("item_id"))
        if item_id:
            item_node_lookup[item_id] = _triple_item_node(subj, obj, "target")
            item_importance_lookup[item_id] = _safe_float(item.get("importance"), 0.0)

    for item in _attribute_items(record, "source"):
        display = _attribute_node_display(item)
        node = _context_node(display, "source")
        relation = _safe_text(item.get("property")) or "attribute"
        score = _safe_float(item.get("importance", item.get("support", 1.0)), 1.0)
        _append_edge(edges, seen_edges, source_node, node, relation, score, edge_type="attribute")
        item_id = _safe_text(item.get("item_id"))
        if item_id:
            item_node_lookup[item_id] = node
            item_importance_lookup[item_id] = _safe_float(item.get("importance"), 0.0)
    for item in _attribute_items(record, "target"):
        display = _attribute_node_display(item)
        node = _context_node(display, "target")
        relation = _safe_text(item.get("property")) or "attribute"
        score = _safe_float(item.get("importance", item.get("support", 1.0)), 1.0)
        _append_edge(edges, seen_edges, target_node, node, relation, score, edge_type="attribute")
        item_id = _safe_text(item.get("item_id"))
        if item_id:
            item_node_lookup[item_id] = node
            item_importance_lookup[item_id] = _safe_float(item.get("importance"), 0.0)

    provenance = dict(record.get("cross_side_provenance") or {})
    for link in list(provenance.get("lexical") or []):
        _append_bridge_edge(
            edges,
            bridge_index,
            source_node,
            target_node,
            "label match",
            _bridge_strength(i_label, 1.0),
            edge_type="bridge-support",
        )
    for family, links in dict(provenance.get("hierarchy") or {}).items():
        for link in list(links or []):
            src_item = item_node_lookup.get(_safe_text(link.get("source_item_id")))
            tgt_item = item_node_lookup.get(_safe_text(link.get("target_item_id")))
            if not src_item or not tgt_item:
                continue
            source_item_id = _safe_text(link.get("source_item_id"))
            target_item_id = _safe_text(link.get("target_item_id"))
            family_importance = _safe_float(family_importances.get(family), i_hier)
            local_mass = _mean_nonempty(
                [
                    item_importance_lookup.get(source_item_id),
                    item_importance_lookup.get(target_item_id),
                ],
                default=0.0,
            )
            _append_bridge_edge(
                edges,
                bridge_index,
                src_item,
                tgt_item,
                "shared hierarchy",
                _bridge_strength(family_importance, local_mass),
                edge_type="bridge-support",
            )
    for link in list(provenance.get("similarity") or []):
        src_item = item_node_lookup.get(_safe_text(link.get("source_item_id")))
        tgt_item = item_node_lookup.get(_safe_text(link.get("target_item_id")))
        if not src_item or not tgt_item:
            continue
        source_item_id = _safe_text(link.get("source_item_id"))
        target_item_id = _safe_text(link.get("target_item_id"))
        local_mass = _mean_nonempty(
            [
                item_importance_lookup.get(source_item_id),
                item_importance_lookup.get(target_item_id),
            ],
            default=0.0,
        )
        _append_bridge_edge(
            edges,
            bridge_index,
            src_item,
            tgt_item,
            "similar evidence",
            _bridge_strength(i_sim, local_mass),
            edge_type="bridge-support",
        )
    for side in ["source", "target"]:
        for link in list((dict(provenance.get("attributes") or {})).get(side) or []):
            item_node = item_node_lookup.get(_safe_text(link.get("item_id")))
            anchor_node = item_node_lookup.get(_safe_text(link.get("anchor_ref")))
            if not item_node or not anchor_node:
                continue
            item_id = _safe_text(link.get("item_id"))
            anchor_ref = _safe_text(link.get("anchor_ref"))
            local_mass = _mean_nonempty(
                [
                    item_importance_lookup.get(item_id),
                    item_importance_lookup.get(anchor_ref),
                ],
                default=_safe_float(item_importance_lookup.get(item_id), 0.0),
            )
            _append_bridge_edge(
                edges,
                bridge_index,
                item_node,
                anchor_node,
                "attribute evidence",
                _bridge_strength(i_attr, local_mass),
                edge_type="bridge-support",
            )
    for link in list((dict(provenance.get("difference") or {})).get("source") or []):
        item_node = item_node_lookup.get(_safe_text(link.get("item_id")))
        if not item_node:
            continue
        item_id = _safe_text(link.get("item_id"))
        _append_bridge_edge(
            edges,
            bridge_index,
            item_node,
            target_node,
            "distinctive evidence",
            _bridge_strength(i_diff, _safe_float(item_importance_lookup.get(item_id), 0.0)),
            edge_type="bridge-contrast",
        )
    for link in list((dict(provenance.get("difference") or {})).get("target") or []):
        item_node = item_node_lookup.get(_safe_text(link.get("item_id")))
        if not item_node:
            continue
        item_id = _safe_text(link.get("item_id"))
        _append_bridge_edge(
            edges,
            bridge_index,
            item_node,
            source_node,
            "distinctive evidence",
            _bridge_strength(i_diff, _safe_float(item_importance_lookup.get(item_id), 0.0)),
            edge_type="bridge-contrast",
        )

    counts = _structural_evidence_counts(record)
    i_label = _safe_float(importances.get("I_label"), 0.0)
    i_struct = _safe_float(importances.get("I_struct"), 0.0)
    i_llm = _safe_float(importances.get("I_llm"), 0.0)
    q_struct = _safe_float(confidences.get("Q_struct"), 0.0)
    u_dis = _safe_float((record.get("weights") or {}).get("U_dis"), 0.0)
    score_value = _safe_float(confidences.get("S_final"), 0.0)

    return {
        "id": _safe_text(record.get("tgt_iri")) or f"path_{rank}",
        "rank": int(rank),
        "ground_truth": int(ground_truth),
        "score": score_value,
        "metrics": {
            "decision_basis": _categorize_decision_basis(i_label, i_struct, i_llm),
            "evidence_strength": _categorize_evidence_strength(
                i_struct, q_struct, counts["total_evidence"]
            ),
            "evidence_agreement": _categorize_evidence_agreement(u_dis, counts["total_evidence"]),
            "explanation_coverage": _categorize_explanation_coverage(
                counts["present_channels"], counts
            ),
            "lead_over_next_candidate": _categorize_lead_over_next(score_value, next_score),
        },
        "llm": {
            "source": _safe_text(labels.get("source")),
            "target": _safe_text(labels.get("target")),
            "p_llm": _safe_float(confidences.get("p_llm"), 0.0),
            "llm_decision": _safe_text(prediction.get("llm_decision")),
            "llm_pair_brief": _safe_text(record.get("llm_pair_brief")),
            "llm_rationale": _safe_text(prediction.get("llm_rationale")),
        },
        "nodes": _ordered_path_nodes(nodes),
        "edges": edges,
    }


def _build_study_mapping(
    selected_df: pd.DataFrame,
    ranked_candidates_by_source: Mapping[str, List[Tuple[str, float]]],
    record_index: Mapping[Tuple[str, str], Mapping[str, Any]],
    top_k: int,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    pairs: List[Dict[str, Any]] = []
    warned_missing_provenance: set[Tuple[str, str]] = set()
    for row in selected_df.to_dict(orient="records"):
        src = _safe_text(row.get("src_iri"))
        gold = _safe_text(row.get("gold_tgt_iri"))
        candidates = list(ranked_candidates_by_source.get(src, []))[:top_k]
        paths: List[Dict[str, Any]] = []
        for rank, (tgt, _score) in enumerate(candidates, start=1):
            record = record_index.get((src, tgt))
            if record is None:
                raise ValueError(f"Missing selected record for source={src} target={tgt}")
            if logger is not None and not (record.get("cross_side_provenance") or {}):
                key = (src, tgt)
                if key not in warned_missing_provenance:
                    warned_missing_provenance.add(key)
                    logger.warning(
                        "Mapping export is using an unbridged path for (%s, %s) because cross_side_provenance is still unavailable.",
                        src,
                        tgt,
                    )
            next_score = None
            if rank < len(candidates):
                next_tgt, next_fallback_score = candidates[rank]
                next_record = record_index.get((src, next_tgt))
                if next_record is not None:
                    next_score = _safe_float(
                        (next_record.get("confidences") or {}).get("S_final"), next_fallback_score
                    )
                else:
                    next_score = float(next_fallback_score)
            paths.append(
                _make_path_payload(
                    record, rank=rank, ground_truth=int(tgt == gold), next_score=next_score
                )
            )
        pairs.append({"id": src, "paths": _annotate_edge_metadata(paths)})
    return {"pairs": pairs}


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


def _selected_records(
    selected_df: pd.DataFrame,
    ranked_candidates_by_source: Mapping[str, List[Tuple[str, float]]],
    record_index: Mapping[Tuple[str, str], Mapping[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for row in selected_df.to_dict(orient="records"):
        src = _safe_text(row.get("src_iri"))
        gold = _safe_text(row.get("gold_tgt_iri"))
        for rank, (tgt, score) in enumerate(
            ranked_candidates_by_source.get(src, [])[:top_k], start=1
        ):
            record = record_index.get((src, tgt))
            if record is None:
                raise ValueError(f"Missing selected record for source={src} target={tgt}")
            item = copy.deepcopy(record)
            item["study_metadata"] = {
                "source_iri": src,
                "rank": rank,
                "ground_truth": int(tgt == gold),
                "candidate_score": float(score),
                "gold_rank": int(row.get("gold_rank", -1)),
            }
            selected.append(item)
    return selected


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_notebook(output_dir: Path) -> Path:
    notebook_path = output_dir / "user_study_analysis.ipynb"
    analysis_dir = str(output_dir.resolve())
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# EXACT User Study Review and Failure Analysis\n",
                    "\n",
                    "This notebook is generated from the artifacts produced by `exact-user-study`.\n",
                    "It is designed for two separate review tasks:\n",
                    "\n",
                    "1. **User-study review**: inspect the shortlist and final selected 20 sources that will be shown to users.\n",
                    "2. **Failure analysis**: inspect where the ranking loses the gold target and which signals appear to drive those failures.\n",
                    "\n",
                    "The notebook keeps these tasks separate so case selection and model-error analysis remain easy to reason about.\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## What This Notebook Loads\n",
                    "\n",
                    "The notebook works only from the saved analysis artifacts. It does **not** rerun alignment or rationale generation.\n",
                    "\n",
                    "Key files:\n",
                    "- `eligible_panels.csv`: source panels that passed the hard completeness checks.\n",
                    "- `study_shortlist.csv`: auto-ranked shortlist before manual review.\n",
                    "- `study_selection_review.csv`: editable review sheet used to freeze the final 20 panels.\n",
                    "- `study_selected_records_with_rationales.json`: the final 20 x 5 selected pair records, including rationales when available.\n",
                    "- `study_mapping.json`: the exported user-study payload.\n",
                    "- `failure_taxonomy.csv`: one row per source for ranking-failure analysis.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from pathlib import Path\n",
                    "import json\n",
                    "from collections import Counter\n",
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "\n",
                    "try:\n",
                    "    from IPython.display import display\n",
                    "except ImportError:\n",
                    "    display = print\n",
                    "\n",
                    f'ANALYSIS_DIR = Path(r"{analysis_dir}")\n',
                    "PAIR_METRICS_PATH = ANALYSIS_DIR / 'pair_metrics.csv'\n",
                    "eligible = pd.read_csv(ANALYSIS_DIR / 'eligible_panels.csv')\n",
                    "source_panels = pd.read_csv(ANALYSIS_DIR / 'source_panels.csv')\n",
                    "shortlist = pd.read_csv(ANALYSIS_DIR / 'study_shortlist.csv')\n",
                    "review = pd.read_csv(ANALYSIS_DIR / 'study_selection_review.csv')\n",
                    "failure = pd.read_csv(ANALYSIS_DIR / 'failure_taxonomy.csv')\n",
                    "study_mapping = json.loads((ANALYSIS_DIR / 'study_mapping.json').read_text(encoding='utf-8'))\n",
                    "selected_records_path = ANALYSIS_DIR / 'study_selected_records_with_rationales.json'\n",
                    "selected_records = json.loads(selected_records_path.read_text(encoding='utf-8')) if selected_records_path.exists() else []\n",
                    "\n",
                    "def _bridge_counts(row):\n",
                    "    provenance = row.get('cross_side_provenance') or {}\n",
                    "    lexical = len(list(provenance.get('lexical') or []))\n",
                    "    hierarchy = sum(len(list(links or [])) for links in dict(provenance.get('hierarchy') or {}).values())\n",
                    "    similarity = len(list(provenance.get('similarity') or []))\n",
                    "    attributes = dict(provenance.get('attributes') or {})\n",
                    "    attribute = len(list(attributes.get('source') or [])) + len(list(attributes.get('target') or []))\n",
                    "    difference = dict(provenance.get('difference') or {})\n",
                    "    contrast = len(list(difference.get('source') or [])) + len(list(difference.get('target') or []))\n",
                    "    return {\n",
                    "        'bridge_total_count': lexical + hierarchy + similarity + attribute + contrast,\n",
                    "        'bridge_support_count': lexical + hierarchy + similarity + attribute,\n",
                    "        'bridge_contrast_count': contrast,\n",
                    "        'has_nonlexical_bridge': int((hierarchy + similarity + attribute + contrast) > 0),\n",
                    "    }\n",
                    "\n",
                    "selected_panel_rows = []\n",
                    "for row in selected_records:\n",
                    "    meta = row.get('study_metadata') or {}\n",
                    "    labels = row.get('selected_labels') or {}\n",
                    "    prediction = row.get('prediction') or {}\n",
                    "    importances = row.get('importances') or {}\n",
                    "    confidences = row.get('confidences') or {}\n",
                    "    bridges = _bridge_counts(row)\n",
                    "    selected_panel_rows.append({\n",
                    "        'source_iri': meta.get('source_iri'),\n",
                    "        'rank': meta.get('rank'),\n",
                    "        'gold_rank': meta.get('gold_rank'),\n",
                    "        'ground_truth': meta.get('ground_truth'),\n",
                    "        'source_label': labels.get('source'),\n",
                    "        'target_label': labels.get('target'),\n",
                    "        'target_iri': row.get('tgt_iri'),\n",
                    "        'score': confidences.get('S_final'),\n",
                    "        'I_label': importances.get('I_label'),\n",
                    "        'I_struct': importances.get('I_struct'),\n",
                    "        'I_llm': importances.get('I_llm'),\n",
                    "        'has_pair_brief': bool(row.get('llm_pair_brief')),\n",
                    "        'has_rationale': bool(prediction.get('llm_rationale')),\n",
                    "        'bridge_total_count': bridges['bridge_total_count'],\n",
                    "        'bridge_support_count': bridges['bridge_support_count'],\n",
                    "        'bridge_contrast_count': bridges['bridge_contrast_count'],\n",
                    "        'has_nonlexical_bridge': bridges['has_nonlexical_bridge'],\n",
                    "    })\n",
                    "selected_panels = pd.DataFrame(selected_panel_rows)\n",
                    "\n",
                    "selected_sources = (\n",
                    "    selected_panels[['source_iri', 'source_label', 'gold_rank']]\n",
                    "    .drop_duplicates()\n",
                    "    .sort_values(['gold_rank', 'source_iri'])\n",
                    "    .reset_index(drop=True)\n",
                    ")\n",
                    "\n",
                    "artifact_summary = pd.DataFrame([\n",
                    "    {'artifact': 'eligible_panels.csv', 'exists': (ANALYSIS_DIR / 'eligible_panels.csv').exists(), 'rows': len(eligible)},\n",
                    "    {'artifact': 'study_shortlist.csv', 'exists': (ANALYSIS_DIR / 'study_shortlist.csv').exists(), 'rows': len(shortlist)},\n",
                    "    {'artifact': 'study_selection_review.csv', 'exists': (ANALYSIS_DIR / 'study_selection_review.csv').exists(), 'rows': len(review)},\n",
                    "    {'artifact': 'study_selected_records_with_rationales.json', 'exists': selected_records_path.exists(), 'rows': len(selected_records)},\n",
                    "    {'artifact': 'study_mapping.json', 'exists': (ANALYSIS_DIR / 'study_mapping.json').exists(), 'rows': len(study_mapping.get('pairs', []))},\n",
                    "    {'artifact': 'failure_taxonomy.csv', 'exists': (ANALYSIS_DIR / 'failure_taxonomy.csv').exists(), 'rows': len(failure)},\n",
                    "])\n",
                    "display(artifact_summary)\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## User Study\n",
                    "\n",
                    "This section focuses on whether the selected cases are suitable for human review.\n",
                    "The goal here is not to analyze ranking failures yet, but to verify that the user-study set is balanced, complete, and explanation-rich.\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Selection Pipeline\n",
                    "\n",
                    "At a high level, the selection pipeline first removes panels that would be confusing or unusable for a study, and only then tries to rank the remaining panels by quality.\n",
                    "\n",
                    "In other words, the pipeline uses two layers:\n",
                    "- **Hard filters**: conditions that must hold for a panel to even be considered.\n",
                    "- **Soft ranking heuristics**: signals used to sort the surviving panels inside each gold-rank bucket.\n",
                    "\n",
                    "The analysis script applies the following steps:\n",
                    "\n",
                    "1. Read the score-sorted local ranking file and keep the top `k` candidates per source.\n",
                    "2. Exclude source panels with missing full explanation records among the displayed candidates. This is a usability filter: if a user will see five candidates, all five must have a proper explanation payload.\n",
                    "3. Keep only sources whose gold target rank is within the displayed review band. For a top-5 user study, the gold target must actually appear in the five candidates shown to the participant.\n",
                    "4. Auto-rank eligible panels inside each gold-rank bucket using pair-brief completeness, structural coverage, mean structural importance, evidence volume, and ambiguity. This does **not** guarantee semantic quality, but it is a pragmatic way to favor cases that are more reviewable and better supported.\n",
                    "5. Build a shortlist per bucket and then freeze a final set with exactly `per_rank` sources per bucket so the study is balanced across ranks 1 to 5.\n",
                    "\n",
                    "The next cells let you inspect those stages directly.\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### How to Read the Eligibility Metrics\n",
                    "\n",
                    "The first summary table is meant to answer a simple question: *do we have enough usable source panels in each gold-rank bucket, and are those panels rich enough to support a meaningful study?*\n",
                    "\n",
                    "The columns mean:\n",
                    "- `eligible_count`: how many source panels survived the hard filters in that gold-rank bucket.\n",
                    "- `mean_I_struct`: average structural importance across candidate panels in the bucket. Higher values mean the non-lexical evidence is actually influencing the score rather than being merely present.\n",
                    "- `mean_evidence_volume`: average number of evidence items carried by the panel, using hierarchy, difference, and attribute evidence. Higher values usually mean more material for explanation, although very large values can also mean the panel is noisy.\n",
                    "- `pair_brief_complete_rate`: fraction of panels in the bucket where all displayed candidates already have a pair brief. A high value is good because it means the panel is closer to being directly reviewable.\n",
                    "- `structural_coverage_mean`: average number of structural channel families represented in the panel. In practice this summarizes whether the panel relies on only one kind of non-lexical evidence or several complementary kinds.\n",
                    "- `mean_bridge_total_count`: average number of cross-side links present in the saved explanations. Higher values usually mean the panel has more explicit grounding between source-side and target-side evidence, rather than two disconnected context islands.\n",
                    "- `nonlexical_bridge_rate`: fraction of panels in the bucket that contain at least one non-lexical bridge. This is a quick proxy for whether the explanation graph can visually connect both sides using ontology evidence instead of only the lexical source-to-target edge.\n",
                    "\n",
                    "There is no single 'perfect' number here. The main use of this table is comparative: one bucket is more comfortable to sample from when it has enough eligible panels and those panels are reasonably rich in evidence.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "eligibility_overview = (\n",
                    "    eligible.groupby('gold_rank')\n",
                    "    .agg(\n",
                    "        eligible_count=('src_iri', 'size'),\n",
                    "        mean_I_struct=('mean_I_struct', 'mean'),\n",
                    "        mean_evidence_volume=('mean_evidence_volume', 'mean'),\n",
                    "        pair_brief_complete_rate=('pair_brief_complete', 'mean'),\n",
                    "        structural_coverage_mean=('panel_structural_coverage', 'mean'),\n",
                    "        mean_bridge_total_count=('mean_bridge_total_count', 'mean'),\n",
                    "        nonlexical_bridge_rate=('panel_has_nonlexical_bridge', 'mean'),\n",
                    "    )\n",
                    "    .reset_index()\n",
                    ")\n",
                    "display(eligibility_overview)\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "fig, ax = plt.subplots(figsize=(8, 4))\n",
                    "eligibility_overview.plot.bar(x='gold_rank', y='eligible_count', ax=ax, legend=False, color='#4C78A8')\n",
                    "ax.set_title('Eligible source panels by gold rank bucket')\n",
                    "ax.set_xlabel('Gold rank bucket')\n",
                    "ax.set_ylabel('Eligible panels')\n",
                    "plt.tight_layout()\n",
                    "plt.show()\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Shortlist Review\n",
                    "\n",
                    "The shortlist is the automatic pool from which the final balanced 20 are chosen.\n",
                    "This is the right place to inspect whether the heuristic ranking is surfacing evidence-rich and explanation-rich cases.\n",
                    "\n",
                    "Use the summary first, then inspect per-rank buckets with the helper table below.\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Heuristic Used to Rank the Shortlist\n",
                    "\n",
                    "The shortlist ranking is deliberately heuristic rather than learned. The goal is not to predict correctness again, but to prefer panels that are easier for a human to validate.\n",
                    "\n",
                    "In plain terms, the ranking prefers panels that:\n",
                    "- already have all pair briefs available,\n",
                    "- show several kinds of structural evidence rather than only one,\n",
                    "- give structural evidence non-trivial weight in the final score,\n",
                    "- contain enough evidence to be informative,\n",
                    "- are not completely trivial but also not so ambiguous that they become confusing,\n",
                    "- and, as a weak bonus, include similarity evidence when it exists.\n",
                    "\n",
                    "This is why the shortlist should be read as a **review aid**, not as a statement that the top auto-ranked panel is objectively the 'best' scientific example. It is the best example under the current operational goal: explainability and study usability.\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### How to Read the Shortlist Metrics\n",
                    "\n",
                    "The shortlist tables expose the signals that drive the automatic ranking. The most important columns are:\n",
                    "- `pair_brief_count`: how many of the displayed candidates already have pair briefs. Higher is better for reviewability.\n",
                    "- `panel_structural_coverage`: how many structural channel groups are represented in the panel. Higher means the user is less likely to see a one-dimensional explanation.\n",
                    "- `mean_I_struct`: average importance of structural evidence across the panel. Higher means the explanation is not purely lexical.\n",
                    "- `mean_evidence_volume`: average amount of structural evidence in the panel. Higher generally means richer context, but if this is high while `mean_I_struct` stays low, the evidence may be present but not actually influential.\n",
                    "- `mean_ambiguity`: average ambiguity over the panel. The heuristic prefers moderate ambiguity because extremely easy cases may not be informative for a study, while extremely ambiguous cases may be frustrating for participants.\n",
                    "- `mean_bridge_total_count`: average number of bridge edges in the panel. Higher values usually mean the graph has more explicit cross-side grounding to show users.\n",
                    "- `panel_has_nonlexical_bridge`: whether at least one candidate in the panel has a non-lexical bridge. This matters because it tells you whether the graph can connect both sides through more than the label-match edge.\n",
                    "- `recommended_keep`: whether the automatic heuristic would keep this source if no manual review were applied.\n",
                    "\n",
                    "A useful practical rule is: prefer panels where the evidence is both **present** and **doing work**. That usually means looking for a sensible combination of `panel_structural_coverage`, `mean_evidence_volume`, and `mean_I_struct`, rather than optimizing a single column in isolation.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "shortlist_overview = (\n",
                    "    shortlist.groupby('gold_rank')\n",
                    "    .agg(\n",
                    "        shortlist_count=('src_iri', 'size'),\n",
                    "        recommended_keeps=('recommended_keep', 'sum'),\n",
                    "        mean_I_struct=('mean_I_struct', 'mean'),\n",
                    "        mean_evidence_volume=('mean_evidence_volume', 'mean'),\n",
                    "        mean_ambiguity=('mean_ambiguity', 'mean'),\n",
                    "        structural_coverage_mean=('panel_structural_coverage', 'mean'),\n",
                    "        mean_bridge_total_count=('mean_bridge_total_count', 'mean'),\n",
                    "        nonlexical_bridge_rate=('panel_has_nonlexical_bridge', 'mean'),\n",
                    "    )\n",
                    "    .reset_index()\n",
                    ")\n",
                    "display(shortlist_overview)\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "review_columns = [\n",
                    "    'gold_rank', 'auto_rank_within_bucket', 'src_iri', 'source_label',\n",
                    "    'gold_tgt_label', 'pair_brief_count', 'panel_structural_coverage',\n",
                    "    'mean_I_struct', 'mean_evidence_volume', 'mean_ambiguity',\n",
                    "    'mean_bridge_total_count', 'panel_has_nonlexical_bridge',\n",
                    "    'recommended_keep', 'keep', 'drop_reason', 'review_note'\n",
                    "]\n",
                    "display(review[review_columns].sort_values(['gold_rank', 'auto_rank_within_bucket']).head(20))\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "def review_bucket(rank: int):\n",
                    "    cols = [\n",
                    "        'auto_rank_within_bucket', 'src_iri', 'source_label', 'gold_tgt_label',\n",
                    "        'pair_brief_count', 'panel_structural_coverage', 'mean_I_struct',\n",
                    "        'mean_evidence_volume', 'mean_ambiguity', 'mean_bridge_total_count',\n",
                    "        'panel_has_nonlexical_bridge', 'recommended_keep',\n",
                    "        'keep', 'drop_reason', 'review_note'\n",
                    "    ]\n",
                    "    bucket = review.loc[review['gold_rank'] == rank, cols].sort_values('auto_rank_within_bucket')\n",
                    "    return bucket.reset_index(drop=True)\n",
                    "\n",
                    "review_bucket(1)\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Final Selected Set\n",
                    "\n",
                    "The final study set should contain exactly 20 sources and exactly 5 displayed candidates per source.\n",
                    "These cells verify that the final selection is balanced by gold rank and that the selected panels contain the expected explanation material.\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### How to Read the Final-Set Metrics\n",
                    "\n",
                    "The final-set summary is less about ranking quality and more about *study readiness*.\n",
                    "\n",
                    "The main columns are:\n",
                    "- `selected_sources`: number of unique sources selected in the bucket. This should match the intended balance, typically 4 per rank.\n",
                    "- `candidate_rows`: number of candidate records represented in that bucket. With top-5 panels, this should be `selected_sources * 5`.\n",
                    "- `rationale_rate`: fraction of displayed candidates that already have a rationale. A high value matters because the user-facing material is more complete.\n",
                    "- `pair_brief_rate`: fraction of displayed candidates that have pair briefs. This is a direct proxy for whether the explanatory package is complete.\n",
                    "- `mean_I_label`, `mean_I_struct`, `mean_I_llm`: average top-level contributions in the final selected panels. These are useful for checking whether the study set is overly dominated by one signal family.\n",
                    "- `mean_bridge_total_count`: average number of bridges per displayed candidate. Higher values generally mean more explicit cross-side grounding.\n",
                    "- `nonlexical_bridge_rate`: fraction of displayed candidates that contain at least one non-lexical bridge. This is a good sanity check for whether the final study set really showcases the pair-adaptive evidence instead of only lexical links.\n",
                    "- `mean_bridge_support_count` vs `mean_bridge_contrast_count`: rough split between bridges that connect supporting evidence and bridges that express one-sided contrast. A useful study set usually contains some of both, but not only contrast.\n",
                    "\n",
                    "A balanced user study usually benefits from variation. If the final set is almost entirely lexical or almost entirely structural, participants may get a narrow picture of how the system behaves.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "selected_sources.groupby('gold_rank').size().rename('selected_source_count')\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "selected_panel_summary = (\n",
                    "    selected_panels.groupby('gold_rank')\n",
                    "    .agg(\n",
                    "        selected_sources=('source_iri', 'nunique'),\n",
                    "        candidate_rows=('target_iri', 'size'),\n",
                    "        rationale_rate=('has_rationale', 'mean'),\n",
                    "        pair_brief_rate=('has_pair_brief', 'mean'),\n",
                    "        mean_I_label=('I_label', 'mean'),\n",
                    "        mean_I_struct=('I_struct', 'mean'),\n",
                    "        mean_I_llm=('I_llm', 'mean'),\n",
                    "        mean_bridge_total_count=('bridge_total_count', 'mean'),\n",
                    "        mean_bridge_support_count=('bridge_support_count', 'mean'),\n",
                    "        mean_bridge_contrast_count=('bridge_contrast_count', 'mean'),\n",
                    "        nonlexical_bridge_rate=('has_nonlexical_bridge', 'mean'),\n",
                    "    )\n",
                    "    .reset_index()\n",
                    ")\n",
                    "display(selected_panel_summary)\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "display(selected_sources)\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Selected Case Drilldown\n",
                    "\n",
                    "Use the helper below to inspect any selected source panel in detail.\n",
                    "It prints the five ranked candidates, their top-level scores and importances, and whether pair briefs and rationales are present.\n",
                    "\n",
                    "When reading the printed output, a useful mental model is:\n",
                    "- `score`, `s_label`, and `S_struct` tell you how the system arrived at the ranking.\n",
                    "- `I_label`, `I_struct`, and `I_llm` tell you which top-level signal families actually mattered.\n",
                    "- the evidence counts tell you whether the explanation is rich because there is real contextual support, or thin because the case relies mostly on labels.\n",
                    "- the bridge counts tell you whether the source and target sides are explicitly connected in the displayed graph, and whether those links are mostly support links or contrast links.\n",
                    "\n",
                    "This is the best place in the notebook to decide whether a source panel 'feels right' for a participant-facing study.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "def _record_channel_counts(row):\n",
                    "    ctx = row.get('context_triples') or {}\n",
                    "    attrs = row.get('attributes') or {}\n",
                    "    hier_count = 0\n",
                    "    for side in ('source', 'target'):\n",
                    "        hier_payload = ctx.get(f'hierarchy_{side}') or {}\n",
                    "        for triples in hier_payload.values():\n",
                    "            hier_count += len(triples or [])\n",
                    "    sim_count = len(ctx.get('similarity_source') or []) + len(ctx.get('similarity_target') or [])\n",
                    "    diff_count = len(ctx.get('difference_source') or []) + len(ctx.get('difference_target') or [])\n",
                    "    attr_count = len(attrs.get('source') or []) + len(attrs.get('target') or [])\n",
                    "    return hier_count, sim_count, diff_count, attr_count\n",
                    "\n",
                    "def _record_bridge_counts(row):\n",
                    "    bridges = _bridge_counts(row)\n",
                    "    return bridges['bridge_total_count'], bridges['bridge_support_count'], bridges['bridge_contrast_count'], bridges['has_nonlexical_bridge']\n",
                    "\n",
                    "def show_selected_source(source_iri: str):\n",
                    "    rows = [row for row in selected_records if (row.get('study_metadata') or {}).get('source_iri') == source_iri]\n",
                    "    rows = sorted(rows, key=lambda row: (row.get('study_metadata') or {}).get('rank', 999))\n",
                    "    if not rows:\n",
                    "        print(f'No selected rows found for {source_iri}')\n",
                    "        return\n",
                    "    for row in rows:\n",
                    "        labels = row.get('selected_labels') or {}\n",
                    "        meta = row.get('study_metadata') or {}\n",
                    "        prediction = row.get('prediction') or {}\n",
                    "        importances = row.get('importances') or {}\n",
                    "        confidences = row.get('confidences') or {}\n",
                    "        hier_count, sim_count, diff_count, attr_count = _record_channel_counts(row)\n",
                    "        bridge_total, bridge_support, bridge_contrast, has_nonlex_bridge = _record_bridge_counts(row)\n",
                    "        print(f\"Rank {meta.get('rank')} | GT={meta.get('ground_truth')} | target={labels.get('target')}\")\n",
                    "        print(f\"  score={confidences.get('S_final')} s_label={confidences.get('s_label')} S_struct={confidences.get('S_struct')} p_llm={confidences.get('p_llm')}\")\n",
                    "        print(f\"  I_label={importances.get('I_label')} I_struct={importances.get('I_struct')} I_llm={importances.get('I_llm')}\")\n",
                    '        print(f"  evidence counts: hierarchy={hier_count} similarity={sim_count} difference={diff_count} attributes={attr_count}")\n',
                    '        print(f"  bridge counts: total={bridge_total} support={bridge_support} contrast={bridge_contrast} nonlexical_bridge={bool(has_nonlex_bridge)}")\n',
                    "        print(f\"  brief={bool(row.get('llm_pair_brief'))} rationale={bool(prediction.get('llm_rationale'))}\")\n",
                    "        print('  Pair brief:')\n",
                    "        print(row.get('llm_pair_brief', ''))\n",
                    "        print('  Rationale:')\n",
                    "        print(prediction.get('llm_rationale', ''))\n",
                    "        print('-' * 100)\n",
                    "\n",
                    "show_selected_source(selected_sources.iloc[0]['source_iri']) if not selected_sources.empty else None\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Failure Analysis\n",
                    "\n",
                    "This section focuses on ranking failures.\n",
                    "Each row in `failure_taxonomy.csv` compares the gold target with the top-1 distractor for one source and assigns a primary failure category.\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Failure Taxonomy\n",
                    "\n",
                    "The failure taxonomy is a structured attempt to turn a raw ranking miss into a more interpretable explanation of *why* the gold target lost.\n",
                    "It is intentionally simple and rule-based. The goal is not to prove causality, but to sort errors into categories that are useful for diagnosis.\n",
                    "\n",
                    "Interpret the primary categories as follows:\n",
                    "\n",
                    "- `missing_panel_record`: the displayed panel was incomplete because at least one top-k candidate lacked a full explanation record.\n",
                    "- `gold_below_top5`: the gold target fell below the review band.\n",
                    "- `near_tie`: gold and top-1 are very close in final score.\n",
                    "- `gold_structurally_better_but_loses`: structural evidence supports the gold target, but the distractor still wins the final score.\n",
                    "- `top1_structurally_better`: the distractor has a clear structural advantage.\n",
                    "- `lexical_overweight`: lexical advantage appears to dominate while structural evidence is not clearly against the distractor.\n",
                    "- `evidence_sparse_other`: fallback bucket when none of the more specific rules fire.\n",
                    "\n",
                    "The ordering matters. For example, `near_tie` is checked before several other categories, because a tiny score difference is often the most important fact about the case even if other signals also differ.\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### How to Read the Failure Metrics\n",
                    "\n",
                    "Most failure-analysis columns are written as **gold minus top-1 distractor**.\n",
                    "That sign convention is important:\n",
                    "- a **positive** delta means the gold target has more of that property,\n",
                    "- a **negative** delta means the top-1 distractor has more of it.\n",
                    "\n",
                    "The most useful fields are:\n",
                    "- `score_gap`: `top1_score - gold_score`. A small positive value means a near miss; a large value means the distractor won comfortably.\n",
                    "- `delta_s_label`: lexical advantage of gold over top-1. Negative values mean the distractor looked lexically better.\n",
                    "- `delta_S_struct`: structural advantage of gold over top-1. Positive values mean the context supported gold more strongly.\n",
                    "- `delta_I_label` and `delta_I_struct`: differences in how much lexical vs structural evidence actually mattered in the final decision.\n",
                    "- `delta_bridge_total_count`: difference in explicit cross-side grounding between gold and top-1. Positive values mean the gold explanation exposes more bridge structure.\n",
                    "- `delta_bridge_support_count` and `delta_bridge_contrast_count`: split the bridge comparison into supporting links versus contrast links.\n",
                    "- `delta_U` and `delta_U_dis`: differences in uncertainty and disagreement. These are mainly useful as contextual cues, not as standalone failure explanations.\n",
                    "- `nonlex_sparse_*`: whether the case had weak non-lexical support relative to the run. These flags help distinguish genuine structural conflict from simple lack of evidence.\n",
                    "- `gold_has_nonlexical_bridge` and `top1_has_nonlexical_bridge`: whether each side of the comparison was backed by at least one non-lexical cross-side link. These help separate 'ungrounded' misses from grounded-but-conflicting ones.\n",
                    "\n",
                    "A simple reading strategy is:\n",
                    "1. Look at `score_gap` to judge whether the miss is mild or severe.\n",
                    "2. Compare `delta_s_label` and `delta_S_struct` to see whether lexical and structural evidence point in the same direction.\n",
                    "3. Use the bridge deltas and sparsity flags to decide whether the case is a real disagreement, a weakly grounded case, or simply an evidence-poor example.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "failure['primary_failure_category'].value_counts().rename_axis('category').reset_index(name='count')\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "failure['primary_failure_category'].value_counts().plot(kind='bar', figsize=(10, 4), title='Failure categories')\n",
                    "plt.tight_layout()\n",
                    "plt.show()\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "failure_overview = pd.DataFrame([\n",
                    "    {'metric': 'sources analysed', 'value': len(failure)},\n",
                    "    {'metric': 'panel complete rate', 'value': failure['panel_complete'].mean()},\n",
                    "    {'metric': 'llm active on top1 rate', 'value': failure['llm_active_top1'].mean()},\n",
                    "    {'metric': 'llm active on gold rate', 'value': failure['llm_active_gold'].mean()},\n",
                    "    {'metric': 'nonlex sparse top1 rate', 'value': failure['nonlex_sparse_top1'].mean()},\n",
                    "    {'metric': 'nonlex sparse gold rate', 'value': failure['nonlex_sparse_gold'].mean()},\n",
                    "    {'metric': 'top1 nonlexical bridge rate', 'value': failure['top1_has_nonlexical_bridge'].mean()},\n",
                    "    {'metric': 'gold nonlexical bridge rate', 'value': failure['gold_has_nonlexical_bridge'].mean()},\n",
                    "])\n",
                    "display(failure_overview)\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "bridge_failure_summary = (\n",
                    "    failure.groupby('primary_failure_category')\n",
                    "    .agg(\n",
                    "        count=('src_iri', 'size'),\n",
                    "        mean_score_gap=('score_gap', 'mean'),\n",
                    "        mean_delta_bridge_total=('delta_bridge_total_count', 'mean'),\n",
                    "        mean_delta_bridge_support=('delta_bridge_support_count', 'mean'),\n",
                    "        mean_delta_bridge_contrast=('delta_bridge_contrast_count', 'mean'),\n",
                    "        gold_nonlexical_bridge_rate=('gold_has_nonlexical_bridge', 'mean'),\n",
                    "        top1_nonlexical_bridge_rate=('top1_has_nonlexical_bridge', 'mean'),\n",
                    "    )\n",
                    "    .sort_values('count', ascending=False)\n",
                    "    .reset_index()\n",
                    ")\n",
                    "display(bridge_failure_summary)\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "ax = failure.plot.scatter(\n",
                    "    x='delta_s_label',\n",
                    "    y='delta_S_struct',\n",
                    "    c=failure['primary_failure_category'].astype('category').cat.codes,\n",
                    "    colormap='tab10',\n",
                    "    figsize=(7, 5),\n",
                    "    title='Gold minus top-1: lexical vs structural deltas',\n",
                    ")\n",
                    "ax.axhline(0.0, color='black', linewidth=1)\n",
                    "ax.axvline(0.0, color='black', linewidth=1)\n",
                    "ax.set_xlabel('delta_s_label (gold - top1)')\n",
                    "ax.set_ylabel('delta_S_struct (gold - top1)')\n",
                    "plt.tight_layout()\n",
                    "plt.show()\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### How to Read the Scatter Plot\n",
                    "\n",
                    "The scatter plot compares lexical and structural deltas at the same time.\n",
                    "\n",
                    "A rough quadrant interpretation is:\n",
                    "- **Upper right**: gold is stronger both lexically and structurally. If such cases still fail, something unusual is happening downstream.\n",
                    "- **Lower left**: the distractor is stronger on both signals. These are usually straightforward hard negatives from the system's perspective.\n",
                    "- **Upper left**: gold is structurally stronger but lexically weaker. These are often the most interesting cases because they expose tension between lexical and contextual evidence.\n",
                    "- **Lower right**: gold is lexically stronger but structurally weaker. These are also informative because they show where structure pushes against the lexical signal.\n",
                    "\n",
                    "After reading the quadrants, it is useful to compare them with the bridge summary above. That tells you whether a disagreement case is at least well grounded by explicit cross-side links, or whether the failure is happening in a graph that never became meaningfully connected in the first place.\n",
                    "\n",
                    "In practice, the most diagnostically useful cases are often the ones near the axes or in the disagreement quadrants, not necessarily the ones with the largest absolute values.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "top5_errors = failure[failure['gold_rank'].between(2, 5)]\n",
                    "below_top5 = failure[failure['primary_failure_category'] == 'gold_below_top5']\n",
                    "plt.figure(figsize=(8, 4))\n",
                    "plt.hist(top5_errors['score_gap'].dropna(), bins=20, alpha=0.6, label='rank 2-5 misses')\n",
                    "if not below_top5.empty:\n",
                    "    plt.hist(below_top5['score_gap'].dropna(), bins=20, alpha=0.6, label='gold below top 5')\n",
                    "plt.legend()\n",
                    "plt.title('Score-gap histograms')\n",
                    "plt.xlabel('top1 score - gold score')\n",
                    "plt.ylabel('number of sources')\n",
                    "plt.tight_layout()\n",
                    "plt.show()\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "worst_losses = failure.sort_values('score_gap', ascending=False)[[\n",
                    "    'src_iri', 'source_label', 'gold_rank', 'primary_failure_category',\n",
                    "    'score_gap', 'delta_s_label', 'delta_S_struct', 'delta_U_dis'\n",
                    "]].head(20)\n",
                    "display(worst_losses)\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Reading the Worst-Loss Table\n",
                    "\n",
                    "The worst-loss table is a prioritization tool.\n",
                    "It surfaces the sources where the gold target lost by the largest final-score margin, together with the key lexical and structural deltas.\n",
                    "\n",
                    "This is useful when you want to inspect only a handful of failures manually. A practical way to use it is to pick:\n",
                    "- one or two large-margin failures from each major category,\n",
                    "- one or two near ties,\n",
                    "- and one or two disagreement cases where lexical and structural evidence pull in opposite directions.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "def show_failure_case(source_iri: str):\n",
                    "    row = failure.loc[failure['src_iri'] == source_iri]\n",
                    "    if row.empty:\n",
                    "        print(f'No failure row found for {source_iri}')\n",
                    "        return\n",
                    "    row = row.iloc[0]\n",
                    "    print('Source:', row['source_label'])\n",
                    "    print('Primary category:', row['primary_failure_category'])\n",
                    "    print('Gold target:', row['gold_tgt_label'])\n",
                    "    print('Top-1 target:', row['top1_tgt_label'])\n",
                    "    print('Gold rank:', row['gold_rank'])\n",
                    "    print('Score gap:', row['score_gap'])\n",
                    "    print('delta_s_label:', row['delta_s_label'])\n",
                    "    print('delta_S_struct:', row['delta_S_struct'])\n",
                    "    print('delta_I_label:', row['delta_I_label'])\n",
                    "    print('delta_I_struct:', row['delta_I_struct'])\n",
                    "    print('delta_bridge_total_count:', row['delta_bridge_total_count'])\n",
                    "    print('delta_bridge_support_count:', row['delta_bridge_support_count'])\n",
                    "    print('delta_bridge_contrast_count:', row['delta_bridge_contrast_count'])\n",
                    "    print('delta_U:', row['delta_U'])\n",
                    "    print('delta_U_dis:', row['delta_U_dis'])\n",
                    "    print('Flags:')\n",
                    "    for col in [\n",
                    "        'llm_active_top1', 'llm_active_gold', 'nonlex_sparse_top1',\n",
                    "        'nonlex_sparse_gold', 'similarity_present_in_panel',\n",
                    "        'top1_has_nonlexical_bridge', 'gold_has_nonlexical_bridge',\n",
                    "        'high_disagreement_gold', 'high_disagreement_top1'\n",
                    "    ]:\n",
                    "        print(f'  {col}: {row[col]}')\n",
                    "    print('\\nSelected-panel drilldown, if this source is part of the study set:')\n",
                    "    show_selected_source(source_iri)\n",
                    "\n",
                    "show_failure_case(failure.iloc[0]['src_iri']) if not failure.empty else None\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Recommended Review Workflow\n",
                    "\n",
                    "A practical workflow is:\n",
                    "\n",
                    "1. Inspect the **eligible panel counts** by gold rank to make sure each bucket is sampleable.\n",
                    "2. Review the **shortlist** with `review_bucket(rank)` and decide whether the recommended picks are acceptable from a human-study perspective.\n",
                    "3. Verify the **final selected set** is balanced and explanation-rich, rather than dominated by one signal family.\n",
                    "4. Use `show_selected_source(...)` to inspect any source that might enter the user study. This is the qualitative sanity-check step.\n",
                    "5. Move to the **failure analysis** section and use `show_failure_case(...)` on representative failures from each category.\n",
                    "6. Only after both reviews are complete should the study set be treated as final. The point is not just to get 20 examples, but to get 20 examples that are both analytically informative and participant-friendly.\n",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    _write_json(notebook_path, notebook)
    return notebook_path


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
