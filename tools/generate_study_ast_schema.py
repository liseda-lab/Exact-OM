"""Freeze the public pyowl structural contract for parser-free study admission.

Run after intentionally upgrading pyowl-core; review and commit the resulting JSON.
This uses only public constructor classes and Python type annotations.
"""

from __future__ import annotations

import dataclasses
import enum
import json
import types
import typing
from importlib.metadata import version
from pathlib import Path

import pyowl_core as core


def schema() -> dict:
    classes = set(core.MODEL_CONSTRUCTORS) | {
        core.Class,
        core.ObjectProperty,
        core.DataProperty,
        core.AnnotationProperty,
        core.NamedIndividual,
        core.Datatype,
    }

    def describe(annotation):
        if annotation in (str, int, bytes, type(None)):
            return {str: "string", int: "integer", bytes: "bytes", type(None): "null"}[annotation]
        origin, arguments = typing.get_origin(annotation), typing.get_args(annotation)
        if origin in (types.UnionType, typing.Union):
            variants = [describe(argument) for argument in arguments]
            if all(isinstance(item, dict) and "nodes" in item for item in variants):
                return {"nodes": sorted({node for item in variants for node in item["nodes"]})}
            return {"one_of": variants}
        if origin in (tuple, core.CanonicalSet):
            assert origin is core.CanonicalSet or arguments[-1] is Ellipsis
            return {"items": describe(arguments[0])}
        if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
            return {"enum": [item.value for item in annotation]}
        if annotation in classes:
            return {"nodes": sorted(cls.__name__ for cls in classes if issubclass(cls, annotation))}
        raise TypeError(f"Unsupported public structural field: {annotation}")

    constructors = {}
    for cls in sorted(classes, key=lambda value: value.__name__):
        hints = typing.get_type_hints(cls)
        constructors[cls.__name__] = {
            field.name: describe(hints[field.name]) for field in dataclasses.fields(cls)
        }
    return {
        "schema": "exact-study-owl-ast/1",
        "source": "pyowl-core public MODEL_CONSTRUCTORS and typed entity dataclasses",
        "package_version": version("pyowl-core"),
        "model_schema_version": core.MODEL_SCHEMA_VERSION,
        "axioms": sorted(cls.__name__ for cls in core.AXIOM_TYPES),
        "entity_kinds": {
            "Class": "class",
            "ObjectProperty": "object_property",
            "DataProperty": "data_property",
            "AnnotationProperty": "annotation_property",
            "NamedIndividual": "named_individual",
            "Datatype": "datatype",
        },
        "constructors": constructors,
    }


if __name__ == "__main__":
    destination = Path(__file__).resolve().parents[1] / "exact_inspect/study/owl_ast_schema.json"
    destination.write_text(json.dumps(schema(), indent=2, sort_keys=True) + "\n")
