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

from exact.runs import RunReader  # noqa: F401
from exact.utils.data import read_yaml  # noqa: F401
from exact.utils.formatting import format_duration as _format_duration  # noqa: F401

TRUTHY_STRINGS = {"1", "true", "yes", "y", "keep", "selected"}
DEFAULT_TOP_K = 5
DEFAULT_PER_RANK = 4
DEFAULT_SHORTLIST_PER_RANK = 8
PROJECT_ROOT = Path(__file__).resolve().parents[3]


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
    reader = RunReader.open(run_dir)
    layout = reader.layout
    ranking_path = layout.mapping_path("local")
    explanations_path = (
        layout.explanation_index_path
        if layout.explanation_index_path.is_file()
        else layout.full_explanations_path
    )
    summary_metrics_path = layout.summary_metrics_path
    resolved_config = config_path or layout.config_path
    if not ranking_path.exists():
        raise FileNotFoundError(f"Local ranking file not found: {ranking_path}")
    if not explanations_path.exists():
        raise FileNotFoundError(f"Explanation artifact not found: {explanations_path}")
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


def _summary_index(
    summary_df: Optional[pd.DataFrame],
) -> Dict[Tuple[str, str], Dict[str, Any]]:
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
    triple_attributions = (record.get("triple_attributions") or {}).get(
        "hierarchy"
    ) or {}
    items: List[Dict[str, Any]] = []
    for family, payload in triple_attributions.items():
        for item in payload.get(side, []) or []:
            row = dict(item or {})
            row["family"] = family
            items.append(row)
    if items:
        return items
    context_triples = (record.get("context_triples") or {}).get(
        f"hierarchy_{side}"
    ) or {}
    for family, triples in context_triples.items():
        for triple in triples or []:
            items.append({"triple": list(triple), "family": family})
    return items


def _channel_items(
    record: Mapping[str, Any], channel: str, side: str
) -> List[Dict[str, Any]]:
    triple_attributions = (record.get("triple_attributions") or {}).get(channel) or {}
    items = [dict(item or {}) for item in (triple_attributions.get(side) or [])]
    if items:
        return items
    context_triples = (record.get("context_triples") or {}).get(
        f"{channel}_{side}"
    ) or []
    return [{"triple": list(triple)} for triple in context_triples]


def _attribute_items(record: Mapping[str, Any], side: str) -> List[Dict[str, Any]]:
    return [
        dict(item or {}) for item in ((record.get("attributes") or {}).get(side) or [])
    ]


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
        "total_evidence": hierarchy_count
        + similarity_count
        + difference_count
        + attribute_count,
    }


def _bridge_metrics(record: Mapping[str, Any]) -> Dict[str, int]:
    provenance = record.get("cross_side_provenance") or {}
    lexical_count = len(list(provenance.get("lexical") or []))
    hierarchy_count = sum(
        len(list(links or []))
        for links in dict(provenance.get("hierarchy") or {}).values()
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
            (hierarchy_count + similarity_count + attribute_count + difference_count)
            > 0
        ),
    }


def _categorize_decision_basis(
    i_label: float, i_struct: float, i_llm: float
) -> Dict[str, Any]:
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
        description = "The candidate is supported by substantial and reasonably reliable contextual evidence."
    elif strength_value >= 0.32:
        label = "Moderate"
        description = (
            "There is some useful contextual support, but it is not overwhelming."
        )
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
        description = (
            "Different evidence sources point in noticeably different directions."
        )
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
        description = (
            "The explanation draws from several different kinds of contextual evidence."
        )
    elif present_channels == 2:
        label = "Balanced"
        description = "The explanation includes more than one evidence type, but it is not especially broad."
    else:
        label = "Narrow"
        description = "The explanation relies on only one evidence family or very limited contextual support."
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


def _categorize_lead_over_next(
    score: float, next_score: Optional[float]
) -> Dict[str, Any]:
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
        description = "This candidate is ahead, but the separation from the next one is still fairly small."
    else:
        label = "Clear lead"
        description = (
            "This candidate is comfortably ahead of the next displayed alternative."
        )
    return {
        "label": label,
        "margin": margin,
        "description": description,
    }


def _ordered_path_nodes(nodes: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    source_nodes = [dict(node) for node in nodes if node.get("type") == "Source"]
    target_nodes = [dict(node) for node in nodes if node.get("type") == "Target"]
    middle_nodes = [
        dict(node) for node in nodes if node.get("type") not in {"Source", "Target"}
    ]
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
            _safe_float(
                prediction.get("ground_truth", summary_row.get("ground_truth", 0))
            )
        ),
        "threshold_positive": bool(
            prediction.get(
                "threshold_positive", summary_row.get("threshold_positive", False)
            )
        ),
        "saved_alignment_member": bool(
            prediction.get(
                "saved_alignment_member",
                summary_row.get("saved_alignment_member", False),
            )
        ),
        "rationale_positive": bool(
            prediction.get(
                "rationale_positive", summary_row.get("rationale_positive", False)
            )
        ),
        "llm_decision": llm_decision,
        "llm_rationale_present": bool(_safe_text(prediction.get("llm_rationale"))),
        "pair_brief_present": bool(_safe_text(record.get("llm_pair_brief"))),
        "S_final": _safe_float(
            confidences.get("S_final", summary_row.get("S_final", 0.0))
        ),
        "s_label": _safe_float(
            confidences.get("s_label", summary_row.get("s_label", 0.0))
        ),
        "S_struct": _safe_float(
            confidences.get("S_struct", summary_row.get("S_struct", 0.0))
        ),
        "p_llm": _safe_float(confidences.get("p_llm", summary_row.get("p_llm", 0.0))),
        "I_label": _safe_float(
            importances.get("I_label", summary_row.get("I_label", 0.0))
        ),
        "I_struct": _safe_float(
            importances.get("I_struct", summary_row.get("I_struct", 0.0))
        ),
        "I_llm": i_llm,
        "I_hier": _safe_float(
            importances.get("I_hier", summary_row.get("I_hier", 0.0))
        ),
        "I_sim": _safe_float(importances.get("I_sim", summary_row.get("I_sim", 0.0))),
        "I_diff": _safe_float(
            importances.get("I_diff", summary_row.get("I_diff", 0.0))
        ),
        "I_attr": _safe_float(
            importances.get("I_attr", summary_row.get("I_attr", 0.0))
        ),
        "U": _safe_float(weights.get("U", summary_row.get("U", 0.0))),
        "U_dis": _safe_float(weights.get("U_dis", summary_row.get("U_dis", 0.0))),
        "hierarchy_count": hierarchy_count,
        "similarity_count": similarity_count,
        "difference_count": difference_count,
        "attribute_count": attribute_count,
        "nonlex_total": hierarchy_count
        + similarity_count
        + difference_count
        + attribute_count,
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
            (idx for idx, (tgt, _score) in enumerate(ranked, start=1) if tgt == gold),
            None,
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
        gold_candidate_score = next(
            (float(score) for tgt, score in ranked if tgt == gold), 0.0
        )

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

        pair_brief_count = sum(
            int(bool(item.get("pair_brief_present"))) for item in pair_rows
        )
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
            gold_row.get("S_final") if gold_row else gold_candidate_score,
            gold_candidate_score,
        )
        top1_score = _safe_float(
            top1_row.get("S_final") if top1_row else top1_candidate_score,
            top1_candidate_score,
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
                "gold_U": _safe_float(
                    gold_row.get("U") if gold_row else None, float("nan")
                ),
                "top1_U": _safe_float(
                    top1_row.get("U") if top1_row else None, float("nan")
                ),
                "gold_U_dis": _safe_float(
                    gold_row.get("U_dis") if gold_row else None, float("nan")
                ),
                "top1_U_dis": _safe_float(
                    top1_row.get("U_dis") if top1_row else None, float("nan")
                ),
                "gold_llm_active": bool(gold_row.get("llm_active"))
                if gold_row
                else False,
                "top1_llm_active": bool(top1_row.get("llm_active"))
                if top1_row
                else False,
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
                    gold_row.get("bridge_total_count") if gold_row else None,
                    float("nan"),
                ),
                "top1_bridge_total_count": _safe_float(
                    top1_row.get("bridge_total_count") if top1_row else None,
                    float("nan"),
                ),
                "gold_bridge_support_count": _safe_float(
                    gold_row.get("bridge_support_count") if gold_row else None,
                    float("nan"),
                ),
                "top1_bridge_support_count": _safe_float(
                    top1_row.get("bridge_support_count") if top1_row else None,
                    float("nan"),
                ),
                "gold_bridge_contrast_count": _safe_float(
                    gold_row.get("bridge_contrast_count") if gold_row else None,
                    float("nan"),
                ),
                "top1_bridge_contrast_count": _safe_float(
                    top1_row.get("bridge_contrast_count") if top1_row else None,
                    float("nan"),
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
                - _safe_float(
                    top1_row.get("s_label") if top1_row else None, float("nan")
                ),
                "delta_S_struct": _safe_float(
                    gold_row.get("S_struct") if gold_row else None, float("nan")
                )
                - _safe_float(
                    top1_row.get("S_struct") if top1_row else None, float("nan")
                ),
                "delta_I_label": _safe_float(
                    gold_row.get("I_label") if gold_row else None, float("nan")
                )
                - _safe_float(
                    top1_row.get("I_label") if top1_row else None, float("nan")
                ),
                "delta_I_struct": _safe_float(
                    gold_row.get("I_struct") if gold_row else None, float("nan")
                )
                - _safe_float(
                    top1_row.get("I_struct") if top1_row else None, float("nan")
                ),
                "delta_U": _safe_float(
                    gold_row.get("U") if gold_row else None, float("nan")
                )
                - _safe_float(top1_row.get("U") if top1_row else None, float("nan")),
                "delta_U_dis": _safe_float(
                    gold_row.get("U_dis") if gold_row else None, float("nan")
                )
                - _safe_float(
                    top1_row.get("U_dis") if top1_row else None, float("nan")
                ),
                "delta_hierarchy_count": _safe_float(
                    gold_row.get("hierarchy_count") if gold_row else None, float("nan")
                )
                - _safe_float(
                    top1_row.get("hierarchy_count") if top1_row else None, float("nan")
                ),
                "delta_similarity_count": _safe_float(
                    gold_row.get("similarity_count") if gold_row else None, float("nan")
                )
                - _safe_float(
                    top1_row.get("similarity_count") if top1_row else None, float("nan")
                ),
                "delta_difference_count": _safe_float(
                    gold_row.get("difference_count") if gold_row else None, float("nan")
                )
                - _safe_float(
                    top1_row.get("difference_count") if top1_row else None, float("nan")
                ),
                "delta_attribute_count": _safe_float(
                    gold_row.get("attribute_count") if gold_row else None, float("nan")
                )
                - _safe_float(
                    top1_row.get("attribute_count") if top1_row else None, float("nan")
                ),
                "delta_bridge_total_count": _safe_float(
                    gold_row.get("bridge_total_count") if gold_row else None,
                    float("nan"),
                )
                - _safe_float(
                    top1_row.get("bridge_total_count") if top1_row else None,
                    float("nan"),
                ),
                "delta_bridge_support_count": _safe_float(
                    gold_row.get("bridge_support_count") if gold_row else None,
                    float("nan"),
                )
                - _safe_float(
                    top1_row.get("bridge_support_count") if top1_row else None,
                    float("nan"),
                ),
                "delta_bridge_contrast_count": _safe_float(
                    gold_row.get("bridge_contrast_count") if gold_row else None,
                    float("nan"),
                )
                - _safe_float(
                    top1_row.get("bridge_contrast_count") if top1_row else None,
                    float("nan"),
                ),
                "panel_complete": bool(
                    missing_topk_records == 0 and len(top_candidates) == top_k
                ),
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
    ranking_path, explanations_path, resolved_config, summary_metrics_path = (
        _resolve_run_paths(run_dir, config_path)
    )
    output_dir = (output_dir or _default_output_dir(run_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading run analysis artifacts from %s", run_dir)
    records = list(RunReader.open(run_dir).iter_explanations())
    summary_df = _read_summary_metrics(summary_metrics_path)
    pair_df, record_index = _build_pair_dataframe(records, _summary_index(summary_df))
    pair_lookup = {
        (str(row["src_iri"]), str(row["tgt_iri"])): row
        for row in pair_df.to_dict(orient="records")
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
    shortlist = (
        eligible_df.groupby("gold_rank", group_keys=False)
        .head(shortlist_per_rank)
        .copy()
    )
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
        review_df = review_df.drop(
            columns=["keep", "drop_reason", "review_note"]
        ).merge(
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
    return selected.sort_values(
        by=["gold_rank", "auto_rank_within_bucket", "src_iri"]
    ).reset_index(drop=True)
