"""Static XR-2 checks, using only the JSON Schema vocabulary present in schema.json.

This is deliberately not a general-purpose JSON Schema implementation. It does
not run OWL verification, compile circuits, train models or measure experiments.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys


def require(condition, message):
    if not condition:
        raise ValueError(message)


def reject_constant(value):
    raise ValueError(f"non-JSON numeric constant: {value}")


def unique_object(pairs):
    result = {}
    for k, v in pairs:
        require(k not in result, f"duplicate JSON field: {k}")
        result[k] = v
    return result


def read_json(path):
    return json.loads(Path(path).read_text(), parse_constant=reject_constant,
                      object_pairs_hook=unique_object)


def merge(base, override):
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = value
    return result


def resolve(path: Path, seen=frozenset()):
    path = Path(path).resolve()
    require(path not in seen, "cyclic protocol inheritance")
    value = read_json(path)
    require(isinstance(value, dict), "protocol must be an object")
    parent = value.pop("extends", None)
    if parent is not None:
        require(isinstance(parent, str) and bool(parent), "extends must be a nonempty path")
        return merge(resolve(path.parent / parent, seen | {path}), value)
    return value


def equal_json(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal_json(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(equal_json(x, y) for x, y in zip(a, b))
    return a == b


def schema_check(value, schema, path="$"):
    known = {"$schema", "title", "type", "const", "enum", "properties", "required",
             "additionalProperties", "minimum", "exclusiveMinimum", "maximum",
             "items", "minItems", "uniqueItems", "minLength"}
    require(set(schema) <= known, f"unsupported schema vocabulary at {path}")
    if "const" in schema:
        require(equal_json(value, schema["const"]), f"{path}: unexpected value")
    if "enum" in schema:
        require(any(equal_json(value, v) for v in schema["enum"]), f"{path}: invalid enum")
    kind = schema.get("type")
    predicates = {
        "object": lambda v: type(v) is dict, "array": lambda v: type(v) is list,
        "integer": lambda v: type(v) is int,
        "number": lambda v: type(v) in (int, float) and math.isfinite(v),
        "string": lambda v: type(v) is str,
    }
    if kind is not None:
        require(kind in predicates, f"unsupported schema type: {kind}")
        require(predicates[kind](value), f"{path}: expected {kind}")
    if kind == "object":
        props = schema["properties"]
        require(set(schema["required"]) <= set(value), f"{path}: missing fields")
        require(schema["additionalProperties"] is False, "schema must reject unknown fields")
        require(set(value) <= set(props), f"{path}: unknown fields {sorted(set(value)-set(props))}")
        for key, item in value.items():
            schema_check(item, props[key], f"{path}.{key}")
    elif kind == "array":
        require(len(value) >= schema.get("minItems", 0), f"{path}: too few items")
        if schema.get("uniqueItems"):
            require(all(not equal_json(v, w) for i, v in enumerate(value) for w in value[:i]),
                    f"{path}: duplicate items")
        for i, item in enumerate(value):
            schema_check(item, schema["items"], f"{path}[{i}]")
    elif kind == "string":
        require(len(value) >= schema.get("minLength", 0), f"{path}: empty string")
    elif kind in ("number", "integer"):
        require(math.isfinite(value), f"{path}: non-finite number")
        if "minimum" in schema:
            require(value >= schema["minimum"], f"{path}: below minimum")
        if "exclusiveMinimum" in schema:
            require(value > schema["exclusiveMinimum"], f"{path}: below exclusive minimum")
        if "maximum" in schema:
            require(value <= schema["maximum"], f"{path}: above maximum")


def validate(c):
    schema = read_json(Path(__file__).resolve().parents[1] / "protocol/schema.json")
    schema_check(c, schema)
    g, r, t = c["graph"], c["resources"], c["training"]
    require(g["hidden_width"] % g["attention_heads"] == 0,
            "hidden width must be divisible by attention heads")
    require(g["dropout"] < 1, "graph dropout must be less than one")
    require(t["patience"] <= t["max_epochs"], "patience exceeds training epochs")
    require(c["grammar"]["max_depth"] <= c["grammar"]["max_constructors"],
            "depth cannot exceed the constructor budget")
    require(c["preferences"]["cost_weights"]["human_ontology_edit"] > 0,
            "human ontology edits need a positive additional default cost")
    require(c["preferences"]["cost_weights"]["ontology_edit"] > 0,
            "ontology edits need a positive default cost")
    require(c["analysis"]["alpha"] < 1 and c["analysis"]["interval"] < 1,
            "statistical levels must be between zero and one")
    for deadline in (r["initial_diagnosis_seconds"], r["verification_call_seconds"],
                     c["circuit"]["compile_seconds"]):
        require(deadline <= r["run_deadline_seconds"], "call deadline exceeds run deadline")
    require(sum(r["stage_deadline_seconds"].values()) <= r["campaign_deadline_seconds"],
            "stage budgets exceed campaign deadline")
    require(r["cleanup_grace_seconds"] <= r["run_deadline_seconds"],
            "cleanup grace exceeds run deadline")
    require(c["teacher"]["case_deadline_seconds"] <= r["stage_deadline_seconds"]["label"],
            "teacher case deadline exceeds labelling stage deadline")
    for cap in (*c["actions"]["max_states"].values(), *c["experiments"]["candidate_caps"]):
        require(cap >= 4, "candidate caps must accommodate elementary equivalence alternatives")
    # Exact per-object representative capacity depends on available families and is checked at runtime.
    generated = c["data"]
    per_group = generated["corruptions_per_group"] + generated["clean_controls_per_group"]
    counts = {split: len(generated["families"]) * n * per_group
              for split, n in generated["groups_per_family"].items()}
    canonical = json.dumps(c, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {"campaign_id": c["campaign_id"], "resolved_sha256": hashlib.sha256(
                canonical.encode()).hexdigest(), "requested_generated_cases": counts,
            "requested_generated_total": sum(counts.values()),
            "scope": "static configuration checks only"}


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    paths = [Path(p) for p in sys.argv[1:]] or [root / "protocol/pilot.json",
                                              root / "protocol/smoke.json"]
    for path in paths:
        print(json.dumps(validate(resolve(path)), indent=2))
