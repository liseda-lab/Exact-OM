"""Descriptor-backed literal evidence; no semantics are inferred from xref spelling."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Mapping

_CATEGORIES = {"definition", "synonym", "identifier", "comment", "other"}
_NORMALIZATIONS = {"identity", "strip", "casefold", "strip_casefold"}
_FIELDS = {
    "category",
    "fact_group",
    "evidence_id",
    "identifier_namespace",
    "normalization",
    "value_prefix",
    "exclusive_values",
}


def validate_annotation_semantics(rules: Mapping[str, Mapping[str, Any]]) -> dict:
    """Validate a pinned property descriptor without inventing its semantic authority."""
    result = deepcopy(dict(rules))
    for prop, rule in result.items():
        if not isinstance(prop, str) or not prop or not isinstance(rule, dict):
            raise ValueError("annotation semantics require property IRIs and mapping rules")
        if set(rule) - _FIELDS:
            raise ValueError(
                f"unknown annotation semantics fields for {prop}: {set(rule) - _FIELDS}"
            )
        if rule.get("category", "other") not in _CATEGORIES:
            raise ValueError(f"unsupported annotation category for {prop}")
        if rule.get("normalization", "identity") not in _NORMALIZATIONS:
            raise ValueError(f"unsupported identifier normalization for {prop}")
        for field in ("fact_group", "evidence_id", "identifier_namespace", "value_prefix"):
            if field in rule and (not isinstance(rule[field], str) or not rule[field].strip()):
                raise ValueError(f"annotation {field} must be a nonempty string")
        if (rule.get("fact_group") or rule.get("identifier_namespace")) and not rule.get(
            "evidence_id"
        ):
            raise ValueError(
                "fact equivalence and identifier namespaces require descriptor evidence_id"
            )
        if "exclusive_values" in rule and not isinstance(rule["exclusive_values"], bool):
            raise ValueError("exclusive_values must be an explicit boolean")
        if rule.get("exclusive_values") and not rule.get("identifier_namespace"):
            raise ValueError("exclusive_values requires a pinned identifier namespace")
        if rule.get("identifier_namespace") and rule.get("category") != "identifier":
            raise ValueError("identifier namespaces require identifier category")
    categories: dict[str, str] = {}
    for rule in result.values():
        if rule.get("fact_group"):
            category = rule.get("category", "other")
            previous = categories.setdefault(rule["fact_group"], category)
            if category != previous:
                raise ValueError(
                    "a fact_group cannot merge definitions, synonyms, or other evidence categories"
                )
    return result


def annotate_literal(item: Mapping[str, Any], rules: Mapping[str, Mapping[str, Any]]) -> dict:
    result = dict(item)
    rule = rules.get(str(item.get("prop_iri", "")))
    if not rule:
        return result
    result["annotation_category"] = rule.get("category", "other")
    if rule.get("fact_group"):
        result["annotation_fact_group"] = rule["fact_group"]
    if rule.get("evidence_id"):
        result["annotation_semantics_evidence_id"] = rule["evidence_id"]
    namespace = rule.get("identifier_namespace")
    value = str(item.get("literal_lexical_form", item.get("value", "")))
    if namespace:
        normalizer = rule.get("normalization", "identity")
        if "strip" in normalizer:
            value = value.strip()
        if "casefold" in normalizer:
            value = value.casefold()
        prefix = rule.get("value_prefix", "")
        if "casefold" in normalizer:
            prefix = prefix.casefold()
        if prefix and not value.startswith(prefix):
            # The property may carry several xref namespaces. An unmatched prefix
            # is unknown, never an identifier in the configured namespace.
            return result
        value = value[len(prefix) :]
        if value:
            result.update(
                identifier_namespace=namespace,
                identifier_normalized=value,
                identifier_exclusive=bool(rule.get("exclusive_values", False)),
            )
    return result


def annotation_fact_key(item: Mapping[str, Any]) -> str:
    """Only explicitly equivalent predicates may share one fact's evidence mass."""
    predicate = item.get("annotation_fact_group") or item.get("prop_iri") or item.get("prop")
    identity = [
        predicate,
        item.get("value", item.get("text")),
        item.get("datatype"),
        item.get("language"),
    ]
    if item.get("identifier_namespace") and item.get("identifier_normalized"):
        identity[:2] = ["identifier:" + item["identifier_namespace"], item["identifier_normalized"]]
    return json.dumps(identity, ensure_ascii=False, sort_keys=True)


def deduplicate_annotations(items) -> list[dict]:
    """Collapse facts before truncation while retaining every original property."""
    groups: dict[str, list[dict]] = {}
    for item in items:
        groups.setdefault(annotation_fact_key(item), []).append(dict(item))
    result = []
    for key, rows in sorted(groups.items()):
        rows.sort(key=lambda row: (str(row.get("prop_iri", "")), str(row.get("text", ""))))
        merged = dict(rows[0])
        origins = []
        ids = set()
        for row in rows:
            origins.extend(row.get("grouped_semantic_features", []))
            origins.append(
                {
                    name: value
                    for name, value in row.items()
                    if name not in {"grouped_semantic_features", "provenance_items", "item_id"}
                }
            )
            ids.update(row.get("provenance_items", []))
            if row.get("item_id"):
                ids.add(row["item_id"])
        unique = {json.dumps(row, sort_keys=True, ensure_ascii=False): row for row in origins}
        merged["grouped_semantic_features"] = [unique[value] for value in sorted(unique)]
        if ids:
            merged["provenance_items"] = sorted(ids)
        # The category remains explicit; a definition never becomes a synonym.
        if merged.get("annotation_fact_group"):
            merged["text"] = (
                f"{merged.get('annotation_category', 'other')}: {merged.get('value', '')}"
            )
        result.append(merged)
    return result
