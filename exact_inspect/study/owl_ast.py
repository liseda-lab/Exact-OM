"""Validate the frozen public OWL structural shape without loading an OWL parser."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_SCHEMA = json.loads(Path(__file__).with_name("owl_ast_schema.json").read_text())
CONSTRUCTORS = _SCHEMA["constructors"]


def _matches(value: Any, descriptor: Any) -> bool:
    if descriptor == "string":
        return isinstance(value, str)
    if descriptor == "integer":
        return type(value) is int and value >= 0
    if descriptor == "null":
        return value is None
    if descriptor == "bytes":
        return (
            isinstance(value, dict)
            and set(value) == {"encoding", "value"}
            and value["encoding"] == "hex"
            and isinstance(value["value"], str)
            and re.fullmatch(r"(?:[a-f0-9]{2})*", value["value"]) is not None
        )
    if "one_of" in descriptor:
        return any(_matches(value, option) for option in descriptor["one_of"])
    if "enum" in descriptor:
        return isinstance(value, str) and value in descriptor["enum"]
    if "nodes" in descriptor:
        return isinstance(value, dict) and value.get("type") in descriptor["nodes"]
    return isinstance(value, list)


def validate_axiom(ast: dict[str, Any]) -> None:
    """Validate a complete original axiom without parsing ontology source syntax."""
    _validate(ast, _SCHEMA["axioms"])


def validate_annotation(annotation: dict[str, Any]) -> None:
    """Validate an original qualifier, including its recursively nested annotations."""
    _validate(annotation, ["Annotation"])


def _validate(ast: dict[str, Any], roots: list[str]) -> None:
    # These are admission limits; semantic loading remains the preparation engine's job.
    pending: list[tuple[Any, Any, int]] = [(ast, {"nodes": roots}, 0)]
    seen = 0
    while pending:
        value, descriptor, depth = pending.pop()
        seen += 1
        if depth > 128 or seen > 100_000:
            raise ValueError("Prepared OWL structure exceeds admission limits")
        if not _matches(value, descriptor):
            raise ValueError("Prepared OWL field differs from its structural type")
        if isinstance(descriptor, str):
            continue
        if "one_of" in descriptor:
            descriptor = next(option for option in descriptor["one_of"] if _matches(value, option))
            pending.append((value, descriptor, depth))
        elif "items" in descriptor:
            pending.extend((item, descriptor["items"], depth + 1) for item in value)
        elif "nodes" in descriptor:
            kind = value["type"]
            fields = CONSTRUCTORS[kind]
            if set(value) != {"type", *fields}:
                raise ValueError("Prepared OWL constructor has missing or extra fields")
            expected_kind = _SCHEMA["entity_kinds"].get(kind)
            if expected_kind and value["kind"] != expected_kind:
                raise ValueError("Prepared OWL entity kind differs from its constructor")
            pending.extend(
                (value[field], field_type, depth + 1) for field, field_type in fields.items()
            )
