"""Dataset-track registry, built-ins, and public retrieval contracts."""

from __future__ import annotations

import warnings
from importlib import metadata
from pathlib import Path
from typing import Any, cast

from .base import UnavailableDescriptorProvider
from .descriptor import TrackDescriptor
from .hf import HfProvider
from .http import DeclarativeHttpProvider
from .provider import (
    DescriptorError,
    IntegrityError,
    LocalDriftError,
    OptionalDependencyError,
    TaskLayout,
    TrackError,
    TrackProvider,
    TrackStatus,
    TrackUnavailableError,
    UserSuppliedFilesError,
    VerificationReport,
)

ENTRY_POINT_GROUP = "exact.tracks"


def provider_from_descriptor(path_or_descriptor: Path | TrackDescriptor) -> TrackProvider:
    """Construct a generic provider from a validated descriptor or YAML path."""

    descriptor = (
        path_or_descriptor
        if isinstance(path_or_descriptor, TrackDescriptor)
        else TrackDescriptor.load(Path(path_or_descriptor))
    )
    if descriptor.provider == "http":
        return DeclarativeHttpProvider(descriptor)
    if descriptor.provider == "hf":
        return HfProvider(descriptor)
    if descriptor.provider == "stub":
        return UnavailableDescriptorProvider(descriptor)
    raise DescriptorError(f"Unsupported track provider backend {descriptor.provider!r}")


class TrackRegistry:
    """Registry of built-in descriptors and third-party provider entry points."""

    def __init__(self, *, load_builtins: bool = True, discover_plugins: bool = True):
        self._providers: dict[str, TrackProvider] = {}
        self._aliases: dict[str, str] = {}
        self._plugins_discovered = not discover_plugins
        if load_builtins:
            self.load_builtins()

    def register(
        self,
        provider: TrackProvider,
        *,
        aliases: tuple[str, ...] = (),
        replace: bool = False,
    ) -> None:
        """Register a provider and optional aliases."""

        if not isinstance(provider, TrackProvider):
            raise TypeError("Track provider must implement the TrackProvider protocol")
        name = str(provider.name).strip()
        if not name:
            raise ValueError("Track provider name cannot be empty")
        if not replace and (name in self._providers or name in self._aliases):
            raise ValueError(f"Track provider {name!r} is already registered")
        self._providers[name] = provider
        for alias in aliases:
            alias = str(alias).strip()
            if not alias:
                raise ValueError("Track alias cannot be empty")
            if not replace and (alias in self._providers or alias in self._aliases):
                raise ValueError(f"Track alias {alias!r} is already registered")
            self._aliases[alias] = name

    def load_descriptor(self, path: Path, *, replace: bool = False) -> TrackProvider:
        """Validate and register a user-provided YAML descriptor."""

        descriptor = TrackDescriptor.load(Path(path))
        provider = provider_from_descriptor(descriptor)
        self.register(provider, aliases=descriptor.aliases, replace=replace)
        return provider

    def load_builtins(self) -> None:
        """Load every descriptor shipped in ``exact.tracks.builtin``."""

        directory = Path(__file__).with_name("builtin")
        for path in sorted(directory.glob("*.yaml")):
            descriptor = TrackDescriptor.load(path)
            provider = provider_from_descriptor(descriptor)
            self.register(provider, aliases=descriptor.aliases)

    def discover_plugins(self) -> None:
        """Load providers registered through the ``exact.tracks`` entry-point group."""

        if self._plugins_discovered:
            return
        self._plugins_discovered = True
        discovered = metadata.entry_points()
        if hasattr(discovered, "select"):
            entries = discovered.select(group=ENTRY_POINT_GROUP)
        else:  # pragma: no cover - compatibility with Python 3.10 backports
            entries = cast(Any, discovered).get(ENTRY_POINT_GROUP, [])
        for entry in entries:
            try:
                loaded: Any = entry.load()
                if isinstance(loaded, (str, Path)):
                    provider = provider_from_descriptor(Path(loaded))
                elif isinstance(loaded, TrackProvider):
                    provider = loaded
                elif callable(loaded):
                    provider = loaded()
                else:
                    provider = loaded
                self.register(provider)
            except Exception as exc:
                warnings.warn(
                    f"Could not load track plugin {entry.name!r}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )

    def get(self, name: str) -> TrackProvider:
        """Resolve a canonical provider name or alias."""

        self.discover_plugins()
        canonical = self._aliases.get(name, name)
        try:
            return self._providers[canonical]
        except KeyError as exc:
            available = ", ".join(self.names()) or "none"
            raise KeyError(f"Unknown track {name!r}; available: {available}") from exc

    def names(self, *, include_aliases: bool = False) -> list[str]:
        """List registered canonical names, optionally including aliases."""

        self.discover_plugins()
        names = set(self._providers)
        if include_aliases:
            names.update(self._aliases)
        return sorted(names)


_REGISTRY: TrackRegistry | None = None


def get_registry() -> TrackRegistry:
    """Return the process-wide track registry, initializing it lazily."""

    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = TrackRegistry()
    return _REGISTRY


def get_track(name: str) -> TrackProvider:
    """Resolve a built-in or plugin track provider."""

    return get_registry().get(name)


def list_tracks() -> list[str]:
    """List canonical built-in and plugin track names."""

    return get_registry().names()


def register_track(provider: TrackProvider, *, aliases: tuple[str, ...] = ()) -> None:
    """Register a provider in the process-wide registry."""

    get_registry().register(provider, aliases=aliases)


__all__ = [
    "DescriptorError",
    "IntegrityError",
    "LocalDriftError",
    "OptionalDependencyError",
    "TaskLayout",
    "TrackError",
    "TrackProvider",
    "TrackRegistry",
    "TrackStatus",
    "TrackUnavailableError",
    "UserSuppliedFilesError",
    "VerificationReport",
    "get_registry",
    "get_track",
    "list_tracks",
    "provider_from_descriptor",
    "register_track",
]
