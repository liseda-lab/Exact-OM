"""YAML loading, round-trip migration, and generated-default rendering."""

from __future__ import annotations

from copy import copy, deepcopy
from enum import Enum
from io import StringIO
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional, Tuple

from pydantic import BaseModel
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq

from exact.core.entities.configs.config import CONFIG_VERSION, ConfigModel, DEFAULT_CONFIG_PATH
from exact.core.entities.configs.migration import (
    MigrationReport,
    V1_TO_V2,
    migrate_v1_mapping,
)

GENERATED_HEADER = "# GENERATED — edit the pydantic models, then run `exact config default`.\n"


def _yaml(*, safe: bool = False) -> YAML:
    instance = YAML(typ="safe" if safe else "rt")
    instance.version = (1, 2)
    instance.default_flow_style = False
    instance.allow_unicode = True
    instance.width = 100
    instance.indent(mapping=2, sequence=4, offset=2)
    return instance


def load_yaml_mapping(path: Path) -> Mapping[str, Any]:
    """Load a YAML mapping without relying on the committed default artifact."""

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = _yaml(safe=True).load(handle)
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ValueError("configuration root must be a mapping")
    return payload


def load_round_trip_mapping(path: Path) -> CommentedMap:
    """Load a YAML mapping with comments, order, and styles retained."""

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = _yaml().load(handle)
    if payload is None:
        return CommentedMap()
    if not isinstance(payload, CommentedMap):
        raise ValueError("configuration root must be a mapping")
    return payload


def _commented_value(value: Any, *, indent: int = 0) -> Any:
    if isinstance(value, BaseModel):
        return _model_to_commented(value, indent=indent)
    if isinstance(value, CommentedMap):
        return deepcopy(value)
    if isinstance(value, Mapping):
        result = CommentedMap()
        for key, item in value.items():
            result[str(key)] = _commented_value(item, indent=indent + 2)
        return result
    if isinstance(value, (list, tuple, CommentedSeq)):
        return CommentedSeq(_commented_value(item, indent=indent + 2) for item in value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    return deepcopy(value)


def _model_to_commented(model: BaseModel, *, indent: int = 0) -> CommentedMap:
    result = CommentedMap()
    for name, info in model.__class__.model_fields.items():
        key = info.alias if isinstance(info.alias, str) else name
        result[key] = _commented_value(getattr(model, name), indent=indent + 2)
        if info.description:
            result.yaml_set_comment_before_after_key(
                key, before=info.description, indent=indent
            )
    return result


def render_default_yaml() -> str:
    """Render the complete commented default v2 configuration."""

    stream = StringIO()
    _yaml().dump(_model_to_commented(ConfigModel(config_version=CONFIG_VERSION)), stream)
    rendered = stream.getvalue()
    if rendered.startswith("%YAML 1.2\n---\n"):
        rendered = rendered[len("%YAML 1.2\n---\n") :]
    return GENERATED_HEADER + rendered


def write_default_config(path: Path = DEFAULT_CONFIG_PATH) -> Path:
    """Regenerate the committed default configuration artifact."""

    output = Path(path)
    output.write_text(render_default_yaml(), encoding="utf-8")
    return output


def default_config_is_in_sync(path: Path = DEFAULT_CONFIG_PATH) -> bool:
    """Return whether the committed generated YAML matches the models exactly."""

    candidate = Path(path)
    return candidate.is_file() and candidate.read_text(encoding="utf-8") == render_default_yaml()


def _parent_and_key(mapping: Mapping[str, Any], path: str) -> Tuple[Optional[Any], Optional[str]]:
    parts = path.split(".")
    current: Any = mapping
    for part in parts[:-1]:
        if not isinstance(current, Mapping) or part not in current:
            return None, None
        current = current[part]
    if not isinstance(current, Mapping) or parts[-1] not in current:
        return None, None
    return current, parts[-1]


def _copy_key_comment(
    source: Mapping[str, Any], source_path: str, destination: Mapping[str, Any], destination_path: str
) -> None:
    src_parent, src_key = _parent_and_key(source, source_path)
    dst_parent, dst_key = _parent_and_key(destination, destination_path)
    if (
        src_parent is None
        or dst_parent is None
        or src_key is None
        or dst_key is None
        or not hasattr(src_parent, "ca")
        or not hasattr(dst_parent, "ca")
    ):
        return
    comment = src_parent.ca.items.get(src_key)
    if comment is not None:
        dst_parent.ca.items[dst_key] = copy(comment)


def migrate_round_trip_document(
    document: CommentedMap,
) -> tuple[CommentedMap, MigrationReport]:
    """Migrate a comment-preserving v1 document and validate the resulting v2 schema."""

    version = document.get("config_version")
    if version is not None:
        ConfigModel.from_mapping(document, warn_v1=False)
        return deepcopy(document), MigrationReport()

    migrated, report = migrate_v1_mapping(document)
    ConfigModel.from_mapping(migrated, warn_v1=False)
    result = _commented_value(migrated)
    assert isinstance(result, CommentedMap)

    if getattr(document, "ca", None) is not None and document.ca.comment is not None:
        result.ca.comment = copy(document.ca.comment)

    for source, target in V1_TO_V2.items():
        if isinstance(target, str) and "*" not in source:
            _copy_key_comment(document, source, result, target)

    root_moves = {
        "alignment_params": "matching",
        "dataset_params": "dataset",
        "candidates_params": "candidates",
        "inference_params": "inference",
        "llm_profiles": "llm",
        "plot_params": "output",
        "sanity_check_params": "output",
        "model": "pipeline",
        "model_chain": "pipeline",
        "second_model": "pipeline",
    }
    for source, target in root_moves.items():
        _copy_key_comment(document, source, result, target)
    return result, report


def dump_yaml_document(document: Mapping[str, Any]) -> str:
    """Serialize one round-trip YAML document without a generated-file header."""

    stream = StringIO()
    _yaml().dump(document, stream)
    rendered = stream.getvalue()
    if rendered.startswith("%YAML 1.2\n---\n"):
        rendered = rendered[len("%YAML 1.2\n---\n") :]
    return rendered


__all__ = [
    "GENERATED_HEADER",
    "default_config_is_in_sync",
    "dump_yaml_document",
    "load_round_trip_mapping",
    "load_yaml_mapping",
    "migrate_round_trip_document",
    "render_default_yaml",
    "write_default_config",
]
