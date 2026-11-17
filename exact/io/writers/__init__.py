"""Alignment writer registry and built-in dispatch."""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module, metadata
from pathlib import Path, PurePosixPath
from typing import Any, cast

from exact.io.writers.base import (
    AlignmentWriter,
    WriterDependencyError,
    WriterError,
    WriterOptionsError,
    WriterPluginError,
)

ENTRY_POINT_GROUP = "exact.writers"

_BUILTINS = {
    "json": "exact.io.writers.json:WRITER",
    "oaei-rdf": "exact.io.writers.oaei_rdf:WRITER",
    "tsv-global": "exact.io.writers.tsv:GLOBAL_WRITER",
    "tsv-local": "exact.io.writers.tsv:LOCAL_WRITER",
    "typed-tsv": "exact.io.writers.typed_tsv:WRITER",
}


def _load_object(specification: str) -> Any:
    module_name, attribute = specification.split(":", 1)
    return getattr(import_module(module_name), attribute)


class WriterRegistry:
    """Resolve built-in and entry-point-backed alignment writers."""

    def __init__(self, *, discover_plugins: bool = True) -> None:
        self._writers: dict[str, AlignmentWriter | Any | str] = dict(_BUILTINS)
        self._plugins_discovered = not discover_plugins

    def register(
        self, writer: AlignmentWriter, *, name: str | None = None, replace: bool = False
    ) -> None:
        """Register an in-process alignment writer."""

        if not isinstance(writer, AlignmentWriter):
            raise TypeError("Writer must implement the AlignmentWriter protocol")
        normalized = str(name or writer.name).strip().lower()
        if not normalized:
            raise ValueError("Writer name cannot be empty")
        if normalized in self._writers and not replace:
            raise ValueError(f"Writer {normalized!r} is already registered")
        self._writers[normalized] = writer

    def discover_plugins(self) -> None:
        """Discover writer entry points without importing their packages."""

        if self._plugins_discovered:
            return
        self._plugins_discovered = True
        discovered = metadata.entry_points()
        if hasattr(discovered, "select"):
            entries = discovered.select(group=ENTRY_POINT_GROUP)
        else:  # pragma: no cover - compatibility with old metadata backports.
            entries = cast(Any, discovered).get(ENTRY_POINT_GROUP, ())
        for entry in entries:
            normalized = str(entry.name).strip().lower()
            if normalized and normalized not in self._writers:
                self._writers[normalized] = entry

    def names(self) -> list[str]:
        """List built-in and discovered writer names."""

        self.discover_plugins()
        return sorted(self._writers)

    def get(self, name: str) -> AlignmentWriter:
        """Load and validate a named writer."""

        self.discover_plugins()
        normalized = str(name).strip().lower()
        try:
            registered = self._writers[normalized]
        except KeyError as exc:
            available = ", ".join(self.names()) or "none"
            raise WriterError(
                f"Unknown output format {normalized!r}; available writers: {available}"
            ) from exc
        try:
            if isinstance(registered, str):
                loaded = _load_object(registered)
            elif hasattr(registered, "load") and not isinstance(registered, AlignmentWriter):
                loaded = registered.load()
            else:
                loaded = registered
        except ImportError as exc:
            raise WriterDependencyError(
                f"Could not load writer {normalized!r}: {exc}. Install the extra "
                "documented by the writer provider."
            ) from exc
        except Exception as exc:
            raise WriterPluginError(f"Could not load writer plugin {normalized!r}: {exc}") from exc

        if isinstance(loaded, type):
            loaded = loaded()
        elif not isinstance(loaded, AlignmentWriter) and callable(loaded):
            loaded = loaded()
        if not isinstance(loaded, AlignmentWriter):
            raise TypeError(f"Writer {normalized!r} does not implement AlignmentWriter")
        self._writers[normalized] = loaded
        return loaded

    def write(
        self,
        name: str,
        mappings: Any,
        output_dir: Path,
        *,
        options: Mapping[str, Any] | None = None,
        filename: str | None = None,
    ) -> Path:
        """Write mappings into ``output_dir`` with the selected writer."""

        writer = self.get(name)
        relative = filename or writer.default_filename
        relative_path = PurePosixPath(str(relative).replace("\\", "/"))
        if (
            relative_path.is_absolute()
            or len(relative_path.parts) != 1
            or relative_path.name in {"", ".", ".."}
        ):
            raise WriterOptionsError("Writer filename must be a single relative filename")
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        try:
            return writer.write(mappings, directory / relative_path.name, options=options)
        except WriterError:
            raise
        except Exception as exc:
            if str(name).strip().lower() in _BUILTINS:
                raise
            raise WriterPluginError(f"Writer plugin {name!r} failed: {exc}") from exc


_REGISTRY: WriterRegistry | None = None


def get_registry() -> WriterRegistry:
    """Return the lazily initialized process-wide writer registry."""

    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = WriterRegistry()
    return _REGISTRY


def write(
    name: str,
    mappings: Any,
    output_dir: Path,
    *,
    options: Mapping[str, Any] | None = None,
    filename: str | None = None,
) -> Path:
    """Write a mapping table using a registered output format."""

    return get_registry().write(name, mappings, output_dir, options=options, filename=filename)


def get_writer(name: str) -> AlignmentWriter:
    """Return a registered alignment writer."""

    return get_registry().get(name)


def list_writers() -> list[str]:
    """List registered output format names."""

    return get_registry().names()


def register_writer(
    writer: AlignmentWriter, *, name: str | None = None, replace: bool = False
) -> None:
    """Register a writer in the process-wide registry."""

    get_registry().register(writer, name=name, replace=replace)


__all__ = [
    "AlignmentWriter",
    "ENTRY_POINT_GROUP",
    "WriterDependencyError",
    "WriterError",
    "WriterOptionsError",
    "WriterPluginError",
    "WriterRegistry",
    "get_registry",
    "get_writer",
    "list_writers",
    "register_writer",
    "write",
]
