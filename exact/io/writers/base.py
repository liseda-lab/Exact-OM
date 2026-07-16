"""Shared contracts and exceptions for alignment writers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


class WriterError(RuntimeError):
    """Base exception for alignment writer failures."""


class WriterOptionsError(WriterError, ValueError):
    """Raised when a writer receives invalid data or options."""


class WriterDependencyError(WriterError, ImportError):
    """Raised when a selected writer is missing a dependency."""


class WriterPluginError(WriterError):
    """Raised when a third-party writer plugin fails."""


@runtime_checkable
class AlignmentWriter(Protocol):
    """Contract exposed through the ``exact.writers`` entry-point group."""

    name: str
    default_filename: str

    def write(
        self,
        mappings: Any,
        path: Path,
        *,
        options: Mapping[str, Any] | None = None,
    ) -> Path: ...


__all__ = [
    "AlignmentWriter",
    "WriterDependencyError",
    "WriterError",
    "WriterOptionsError",
    "WriterPluginError",
]
