"""Validation for YAML-driven track descriptors."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import yaml

from .provider import DescriptorError

DESCRIPTOR_VERSION = 1


def _mapping(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DescriptorError(f"{location} must be a mapping")
    return {str(key): item for key, item in value.items()}


def _keys(value: Mapping[str, Any], allowed: set[str], location: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise DescriptorError(f"Unknown key(s) at {location}: {', '.join(unknown)}")


def safe_relative_path(value: str, location: str) -> str:
    """Validate and normalize a descriptor path without touching the filesystem."""

    path = PurePosixPath(str(value).replace("\\", "/"))
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise DescriptorError(f"{location} must be a safe relative path, got {value!r}")
    return path.as_posix()


def safe_glob(value: str, location: str) -> str:
    """Validate a glob while allowing wildcard path components."""

    normalized = str(value).replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise DescriptorError(f"{location} must be a safe relative glob, got {value!r}")
    return normalized


@dataclass(frozen=True)
class ResourceSpec:
    """One upstream path/glob and its optional materialization transform."""

    path: str | None = None
    glob: str | None = None
    output: str | None = None
    transform: str | None = None
    optional: bool = False

    @classmethod
    def parse(cls, value: Any, location: str) -> "ResourceSpec":
        if isinstance(value, str):
            return cls(path=safe_relative_path(value, location))
        raw = _mapping(value, location)
        _keys(raw, {"path", "glob", "output", "transform", "optional"}, location)
        path = raw.get("path")
        glob = raw.get("glob")
        if (path is None) == (glob is None):
            raise DescriptorError(f"{location} requires exactly one of 'path' or 'glob'")
        output = raw.get("output")
        transform = raw.get("transform")
        if transform is not None and not str(transform).strip():
            raise DescriptorError(f"{location}.transform cannot be empty")
        return cls(
            path=safe_relative_path(str(path), f"{location}.path") if path is not None else None,
            glob=safe_glob(str(glob), f"{location}.glob") if glob is not None else None,
            output=(
                safe_relative_path(str(output), f"{location}.output")
                if output is not None
                else None
            ),
            transform=str(transform) if transform is not None else None,
            optional=bool(raw.get("optional", False)),
        )


@dataclass(frozen=True)
class UserSuppliedSpec:
    """A file the provider cannot redistribute."""

    path: str
    destination: str
    help: str
    sha256: str | None = None
    sha256_from: Mapping[str, Any] | None = None

    @classmethod
    def parse(cls, value: Any, location: str) -> "UserSuppliedSpec":
        raw = _mapping(value, location)
        _keys(raw, {"path", "destination", "help", "sha256", "sha256_from"}, location)
        if "path" not in raw:
            raise DescriptorError(f"{location}.path is required")
        path = safe_relative_path(str(raw["path"]), f"{location}.path")
        destination = safe_relative_path(
            str(raw.get("destination", path)), f"{location}.destination"
        )
        digest = raw.get("sha256")
        if digest is not None:
            digest = str(digest).lower()
            if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                raise DescriptorError(f"{location}.sha256 must be a 64-character hex digest")
        sha256_from = raw.get("sha256_from")
        if sha256_from is not None:
            sha256_from = _mapping(sha256_from, f"{location}.sha256_from")
            _keys(sha256_from, {"path", "key", "filename"}, f"{location}.sha256_from")
            if "path" not in sha256_from:
                raise DescriptorError(f"{location}.sha256_from.path is required")
            sha256_from = {
                **sha256_from,
                "path": safe_relative_path(
                    str(sha256_from["path"]), f"{location}.sha256_from.path"
                ),
            }
        return cls(
            path=path,
            destination=destination,
            help=str(raw.get("help", "Supply this file according to its upstream licence.")),
            sha256=digest,
            sha256_from=sha256_from,
        )


@dataclass(frozen=True)
class TaskSpec:
    """Declarative mapping from an upstream snapshot into :class:`TaskLayout`."""

    source: ResourceSpec
    target: ResourceSpec
    refs: Mapping[str, ResourceSpec] = field(default_factory=dict)
    candidates: ResourceSpec | None = None
    extras: Mapping[str, Any] = field(default_factory=dict)
    user_supplied: tuple[str, ...] = ()

    @classmethod
    def parse(cls, value: Any, location: str) -> "TaskSpec":
        raw = _mapping(value, location)
        _keys(raw, {"source", "target", "refs", "candidates", "extras", "user_supplied"}, location)
        if "source" not in raw or "target" not in raw:
            raise DescriptorError(f"{location} requires both source and target")
        refs_raw = _mapping(raw.get("refs", {}), f"{location}.refs")
        refs = {
            str(split): ResourceSpec.parse(spec, f"{location}.refs.{split}")
            for split, spec in refs_raw.items()
        }
        extras = _mapping(raw.get("extras", {}), f"{location}.extras")
        users = raw.get("user_supplied", [])
        if not isinstance(users, list) or not all(isinstance(item, str) for item in users):
            raise DescriptorError(f"{location}.user_supplied must be a list of names")
        source = ResourceSpec.parse(raw["source"], f"{location}.source")
        target = ResourceSpec.parse(raw["target"], f"{location}.target")
        if source.optional or target.optional:
            raise DescriptorError(f"{location}.source and target cannot be optional")
        return cls(
            source=source,
            target=target,
            refs=refs,
            candidates=(
                ResourceSpec.parse(raw["candidates"], f"{location}.candidates")
                if raw.get("candidates") is not None
                else None
            ),
            extras=extras,
            user_supplied=tuple(users),
        )


@dataclass(frozen=True)
class TrackDescriptor:
    """Validated representation of a built-in or user-provided YAML descriptor."""

    name: str
    provider: str
    provider_version: str
    upstream: Mapping[str, Any]
    tasks: Mapping[str, TaskSpec]
    user_supplied: Mapping[str, UserSuppliedSpec] = field(default_factory=dict)
    description: str = ""
    aliases: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, value: Any, *, source: str = "descriptor") -> "TrackDescriptor":
        raw = _mapping(value, source)
        _keys(
            raw,
            {
                "descriptor_version",
                "name",
                "provider",
                "provider_version",
                "description",
                "aliases",
                "upstream",
                "user_supplied",
                "tasks",
            },
            source,
        )
        version = raw.get("descriptor_version", DESCRIPTOR_VERSION)
        if version != DESCRIPTOR_VERSION:
            raise DescriptorError(
                f"{source} uses descriptor_version {version!r}; supported version is {DESCRIPTOR_VERSION}"
            )
        name = str(raw.get("name", "")).strip()
        if not name or "/" in name or "\\" in name or name in {".", ".."}:
            raise DescriptorError(f"{source}.name must be a non-empty path-safe identifier")
        provider = str(raw.get("provider", "")).lower()
        if provider not in {"http", "hf", "stub"}:
            raise DescriptorError(f"{source}.provider must be one of: http, hf, stub")
        upstream = _mapping(raw.get("upstream", {}), f"{source}.upstream")
        tasks_raw = _mapping(raw.get("tasks", {}), f"{source}.tasks")
        tasks: dict[str, TaskSpec] = {}
        for task_name, task in tasks_raw.items():
            normalized_task = str(task_name).strip()
            if (
                not normalized_task
                or normalized_task in {".", ".."}
                or "/" in normalized_task
                or "\\" in normalized_task
            ):
                raise DescriptorError(f"{source}.tasks contains an unsafe task name {task_name!r}")
            tasks[normalized_task] = TaskSpec.parse(task, f"{source}.tasks.{normalized_task}")
        if provider != "stub" and not tasks:
            raise DescriptorError(f"{source}.tasks cannot be empty for provider {provider}")
        supplied_raw = _mapping(raw.get("user_supplied", {}), f"{source}.user_supplied")
        supplied = {
            str(name): UserSuppliedSpec.parse(item, f"{source}.user_supplied.{name}")
            for name, item in supplied_raw.items()
        }
        for task_name, task in tasks.items():
            missing = sorted(set(task.user_supplied) - set(supplied))
            if missing:
                raise DescriptorError(
                    f"{source}.tasks.{task_name} references undefined user-supplied file(s): "
                    + ", ".join(missing)
                )
        aliases = raw.get("aliases", [])
        if not isinstance(aliases, list) or not all(isinstance(item, str) for item in aliases):
            raise DescriptorError(f"{source}.aliases must be a list of strings")
        return cls(
            name=name,
            provider=provider,
            provider_version=str(raw.get("provider_version", "1")),
            upstream=upstream,
            tasks=tasks,
            user_supplied=supplied,
            description=str(raw.get("description", "")),
            aliases=tuple(aliases),
        )

    @classmethod
    def load(cls, path: Path) -> "TrackDescriptor":
        """Load and validate a YAML descriptor from ``path``."""

        try:
            with Path(path).open("r", encoding="utf-8") as stream:
                value = yaml.safe_load(stream)
        except OSError as exc:
            raise DescriptorError(f"Could not read track descriptor {path}: {exc}") from exc
        except yaml.YAMLError as exc:
            raise DescriptorError(f"Invalid YAML in track descriptor {path}: {exc}") from exc
        return cls.from_mapping(value, source=str(path))
