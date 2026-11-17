"""Deterministic, score-free evidence packets for pair decisions."""
from __future__ import annotations
import json
from typing import Any, Mapping
from exact.utils.fitted_artifacts import fingerprint

FACT_FIELDS = ('item_id', 'triple', 'text', 'subject_iri', 'object_iri', 'rel_iri', 'property_iri', 'state', 'contradiction_semantics', 'axiom_id', 'datatype', 'language')

def _digest(value: Any) -> str:
    return str(fingerprint(value))


def structured_facts(
    source_label: str,
    target_label: str,
    groups: Mapping[str, Mapping[str, Any]],
    *,
    max_facts_per_group_side: int = 8,
    max_fact_bytes: int = 2048,
    max_packet_bytes: int = 20 * 1024,
) -> str:
    """Render native selected facts, omitting whole oversized facts instead of altering them.

    Scores and reference labels are excluded by an allowlist. The deterministic
    bounds and omission counts are part of the packet and therefore its identity.
    Original score-bearing briefs belong only to the named historical controls.
    """
    if any(
        type(value) is not int or value < 1
        for value in (max_facts_per_group_side, max_fact_bytes, max_packet_bytes)
    ):
        raise ValueError("Evidence bounds must be positive integers")
    if not isinstance(source_label, str) or not isinstance(target_label, str):
        raise ValueError("Entity labels must be strings")
    packet: dict[str, Any] = {
        "source_label": source_label,
        "target_label": target_label,
        "facts": [],
        "omitted_facts": 0,
        "bounds": {
            "per_group_side": max_facts_per_group_side,
            "fact_bytes": max_fact_bytes,
            "packet_bytes": max_packet_bytes,
        },
        "unknowns": "An absent counterpart is unobserved, not a contradiction.",
    }

    def encoded() -> str:
        return json.dumps(
            packet, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        )

    if len(encoded().encode()) > max_packet_bytes:
        raise ValueError("Entity labels alone exceed the evidence packet bound")
    seen = set()
    for group, payload in sorted(groups.items()):
        for side, field in (("source", "src_selected"), ("target", "tgt_selected")):
            accepted = 0
            for item in payload.get(field, []):
                if not isinstance(item, Mapping):
                    raise ValueError("Native selected facts must be mappings")
                fact = {key: item[key] for key in FACT_FIELDS if key in item}
                # Nested arbitrary metadata is not an allowed route for score/label leakage.
                if any(
                    isinstance(value, Mapping)
                    or (
                        isinstance(value, (list, tuple))
                        and any(isinstance(part, Mapping) for part in value)
                    )
                    for value in fact.values()
                ):
                    raise ValueError("Selected fact contains unsupported nested metadata")
                identity = _digest(
                    [side, {key: value for key, value in fact.items() if key != "item_id"}]
                )
                if not fact or identity in seen:
                    continue
                seen.add(identity)
                record = {"side": side, "group": group, **fact}
                if (
                    accepted >= max_facts_per_group_side
                    or len(json.dumps(record, ensure_ascii=False).encode()) > max_fact_bytes
                ):
                    packet["omitted_facts"] += 1
                    continue
                packet["facts"].append(record)
                if len(encoded().encode()) + 32 > max_packet_bytes:
                    packet["facts"].pop()
                    packet["omitted_facts"] += 1
                    continue
                accepted += 1
    result = encoded()
    if len(result.encode()) > max_packet_bytes:
        raise ValueError("Evidence omission accounting exceeds the packet bound")
    return result
