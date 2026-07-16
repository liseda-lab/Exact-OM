from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple


def safe_float(value: Any, default: float = 0.0) -> float:
    if value in ("", None):
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def safe_text(value: Any) -> str:
    return str(value or "").strip()


def mean_nonempty(values: Sequence[float], default: float = 0.0) -> float:
    cleaned = [float(value) for value in values if value is not None]
    if not cleaned:
        return float(default)
    return float(sum(cleaned) / len(cleaned))


def hierarchy_items(record: Mapping[str, Any], side: str) -> List[Dict[str, Any]]:
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


def channel_items(record: Mapping[str, Any], channel: str, side: str) -> List[Dict[str, Any]]:
    triple_attributions = (record.get("triple_attributions") or {}).get(channel) or {}
    items = [dict(item or {}) for item in (triple_attributions.get(side) or [])]
    if items:
        return items
    context_triples = (record.get("context_triples") or {}).get(f"{channel}_{side}") or []
    return [{"triple": list(triple)} for triple in context_triples]


def attribute_items(record: Mapping[str, Any], side: str) -> List[Dict[str, Any]]:
    return [dict(item or {}) for item in ((record.get("attributes") or {}).get(side) or [])]


def ordered_path_nodes(nodes: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    source_nodes = [dict(node) for node in nodes if node.get("type") == "Source"]
    target_nodes = [dict(node) for node in nodes if node.get("type") == "Target"]
    middle_nodes = [dict(node) for node in nodes if node.get("type") not in {"Source", "Target"}]
    return source_nodes + middle_nodes + target_nodes


def edge_display_level(edge: Mapping[str, Any]) -> Tuple[bool, int, str]:
    edge_type = safe_text(edge.get("type"))
    if not edge_type.startswith("bridge-"):
        return False, 1, "Context edge"
    if edge_type == "bridge-contrast" or safe_text(edge.get("label")).lower() == "label match":
        return True, 2, "Core bridge"
    if safe_text(edge.get("score")).lower() == "weak":
        return True, 4, "Optional bridge"
    return True, 3, "Supporting bridge"


def categorize_bridge_relevance(value: float) -> str:
    score = safe_float(value, 0.0)
    if score >= 0.20:
        return "strong"
    if score >= 0.08:
        return "moderate"
    return "weak"


def attribute_node_display(item: Mapping[str, Any]) -> str:
    value = safe_text(item.get("value"))
    if value:
        return value
    text = safe_text(item.get("text"))
    if ":" in text:
        suffix = text.split(":", 1)[1].strip()
        if suffix:
            return suffix
    if text:
        return text
    prop = safe_text(item.get("property"))
    return prop or "attribute"
