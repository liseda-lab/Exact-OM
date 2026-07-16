"""Knowledge-source registry and format dispatch."""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module, metadata
from pathlib import Path
from typing import Any, Protocol, cast

from exact.core.contracts.knowledge import KnowledgeSource

ENTRY_POINT_GROUP = "exact.sources"


class SourceError(RuntimeError):
    """Base exception for knowledge-source resolution failures."""


class SourceOptionsError(SourceError, ValueError):
    """Raised when source options or a source descriptor are invalid."""


class SourceDependencyError(SourceError, ImportError):
    """Raised when a selected source backend is missing a dependency."""


class SourcePluginError(SourceError):
    """Raised when a third-party source plugin cannot be loaded or invoked."""


class SourceFactory(Protocol):
    """Factory contract exposed by the ``exact.sources`` entry-point group."""

    def __call__(self, path: Path, *, options: Mapping[str, Any]) -> KnowledgeSource: ...


_BUILTINS = {
    "csv-kg": "exact.io.sources.csv_kg:create_source",
    "owl": "exact.io.sources.owl:create_source",
    "rdf": "exact.io.sources.rdf:create_source",
}

_AUTO_FORMATS = {
    ".csv": "csv-kg",
    ".n3": "rdf",
    ".nt": "rdf",
    ".ofn": "owl",
    ".owl": "owl",
    ".owx": "owl",
    ".rdf": "owl",
    ".tsv": "csv-kg",
    ".ttl": "rdf",
    ".xml": "rdf",
}


def _load_object(specification: str) -> Any:
    module_name, attribute = specification.split(":", 1)
    return getattr(import_module(module_name), attribute)


class SourceRegistry:
    """Resolve built-in and entry-point-backed knowledge-source factories."""

    def __init__(self, *, discover_plugins: bool = True) -> None:
        self._factories: dict[str, Any] = dict(_BUILTINS)
        self._plugins_discovered = not discover_plugins

    def register(self, name: str, factory: SourceFactory, *, replace: bool = False) -> None:
        """Register an in-process source factory."""

        normalized = str(name).strip().lower()
        if not normalized:
            raise ValueError("Source format name cannot be empty")
        if normalized in self._factories and not replace:
            raise ValueError(f"Source format {normalized!r} is already registered")
        if not callable(factory):
            raise TypeError("Source factory must be callable")
        self._factories[normalized] = factory

    def discover_plugins(self) -> None:
        """Discover factories in the ``exact.sources`` entry-point group lazily."""

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
            if normalized and normalized not in self._factories:
                self._factories[normalized] = entry

    def names(self) -> list[str]:
        """List built-in and discovered source format names."""

        self.discover_plugins()
        return sorted(self._factories)

    def _factory(self, name: str) -> SourceFactory:
        self.discover_plugins()
        try:
            registered = self._factories[name]
        except KeyError as exc:
            available = ", ".join(self.names()) or "none"
            raise SourceError(
                f"Unknown input format {name!r}; available formats: {available}"
            ) from exc

        if isinstance(registered, str):
            try:
                loaded = _load_object(registered)
            except ImportError as exc:
                raise SourceDependencyError(
                    f"Could not load the {name!r} source backend: {exc}. "
                    'Reinstall Exact-OM with `pip install "exact-om"`.'
                ) from exc
            self._factories[name] = loaded
            return cast(SourceFactory, loaded)

        if hasattr(registered, "load") and not callable(registered):
            try:
                loaded = registered.load()
            except Exception as exc:
                raise SourcePluginError(
                    f"Could not load source plugin {name!r} from {ENTRY_POINT_GROUP}: {exc}. "
                    "Install or repair the package that provides this plugin."
                ) from exc
            self._factories[name] = cast(SourceFactory, loaded)
            return cast(SourceFactory, loaded)
        return cast(SourceFactory, registered)

    def resolve(
        self,
        path: Path,
        *,
        source_format: str = "auto",
        options: Mapping[str, Any] | None = None,
    ) -> KnowledgeSource:
        """Resolve ``path`` into a :class:`KnowledgeSource`."""

        source_path = Path(path).expanduser()
        if not source_path.exists():
            raise FileNotFoundError(f"Knowledge source does not exist: {source_path}")
        normalized_options = dict(options or {})
        normalized_format = str(source_format).strip().lower()
        if normalized_format == "auto":
            normalized_format = infer_format(source_path, normalized_options)
        factory = self._factory(normalized_format)
        creator = getattr(factory, "create", factory)
        if not callable(creator):
            raise SourcePluginError(
                f"Source plugin {normalized_format!r} must be callable or expose create()"
            )
        try:
            source = creator(source_path, options=normalized_options)
        except (SourceError, FileNotFoundError):
            raise
        except ImportError as exc:
            raise SourceDependencyError(
                f"Source backend {normalized_format!r} is missing a dependency: {exc}. "
                "Install the extra documented by the plugin provider."
            ) from exc
        except Exception as exc:
            if normalized_format in _BUILTINS:
                raise
            raise SourcePluginError(
                f"Source plugin {normalized_format!r} failed for {source_path}: {exc}"
            ) from exc
        if not isinstance(source, KnowledgeSource):
            raise TypeError(
                f"Source factory {normalized_format!r} returned an object that does not "
                "implement KnowledgeSource"
            )
        return source


def infer_format(path: Path, options: Mapping[str, Any] | None = None) -> str:
    """Infer a built-in source format from a path and optional descriptor values."""

    source_path = Path(path)
    if source_path.is_dir():
        if (source_path / "kg.yaml").is_file() or "triples_files" in (options or {}):
            return "csv-kg"
        raise SourceOptionsError(
            f"Cannot infer an input format for directory {source_path}. Add kg.yaml, "
            "provide source_options.triples_files, or select format='csv-kg'."
        )
    try:
        return _AUTO_FORMATS[source_path.suffix.lower()]
    except KeyError as exc:
        supported = ", ".join(sorted(_AUTO_FORMATS))
        raise SourceOptionsError(
            f"Cannot infer an input format from {source_path.name!r}; supported extensions: "
            f"{supported}. Select the format explicitly for a plugin source."
        ) from exc


_REGISTRY: SourceRegistry | None = None


def get_registry() -> SourceRegistry:
    """Return the lazily initialized process-wide source registry."""

    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = SourceRegistry()
    return _REGISTRY


def resolve(
    path: Path,
    format: str = "auto",
    options: Mapping[str, Any] | None = None,
) -> KnowledgeSource:
    """Resolve a file or directory using a built-in or plugin source backend."""

    return get_registry().resolve(path, source_format=format, options=options)


def list_sources() -> list[str]:
    """List registered source names."""

    return get_registry().names()


def register_source(name: str, factory: SourceFactory, *, replace: bool = False) -> None:
    """Register a source factory in the process-wide registry."""

    get_registry().register(name, factory, replace=replace)


__all__ = [
    "ENTRY_POINT_GROUP",
    "SourceDependencyError",
    "SourceError",
    "SourceFactory",
    "SourceOptionsError",
    "SourcePluginError",
    "SourceRegistry",
    "get_registry",
    "infer_format",
    "list_sources",
    "register_source",
    "resolve",
]
