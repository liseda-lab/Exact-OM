"""Public contracts for materializing versioned dataset tracks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Protocol, runtime_checkable

TrackStatus = Literal["ok", "local-drift", "upstream-moved", "not-materialized"]


@dataclass(frozen=True)
class TaskLayout:
    """Canonical files consumed by an Exact alignment task.

    Attributes:
        source: Source ontology file or directory-backed knowledge graph.
        target: Target ontology file or directory-backed knowledge graph.
        refs: Reference alignments keyed by split, for example ``train`` and ``test``.
        candidates: Optional local-ranking candidate table.
        extras: Provider-specific materialized paths and declarative flags.
        provenance: Immutable upstream and checksum metadata from the dataset lock.
    """

    source: Path
    target: Path
    refs: dict[str, Path] = field(default_factory=dict)
    candidates: Path | None = None
    extras: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VerificationReport:
    """Result of checking a materialized task against local and upstream pins."""

    provider: str
    task: str
    status: TrackStatus
    checked_files: int = 0
    issues: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    lock_entry: Mapping[str, Any] | None = None

    @property
    def ok(self) -> bool:
        """Return whether every mandatory integrity check passed."""

        return self.status == "ok"


@runtime_checkable
class TrackProvider(Protocol):
    """Dataset retrieval seam used by the CLI and configuration resolver."""

    name: str

    def tasks(self) -> list[str]:
        """List task identifiers exposed by this track."""

    def materialize(
        self,
        task: str,
        data_root: Path,
        *,
        revision: str | None = None,
        update: bool = False,
    ) -> TaskLayout:
        """Materialize one task, pinning it unless ``update`` is explicit."""

    def verify(self, task: str, data_root: Path) -> VerificationReport:
        """Verify local hashes and detect movement of a mutable upstream ref."""

    def status(self, task: str, data_root: Path) -> TrackStatus:
        """Return the concise status for one task."""


class TrackError(RuntimeError):
    """Base class for actionable dataset-track failures."""


class DescriptorError(TrackError, ValueError):
    """Raised when a declarative track descriptor is malformed."""


class IntegrityError(TrackError):
    """Raised when downloaded or materialized data fails an integrity check."""


class LocalDriftError(IntegrityError):
    """Raised when a pinned task was modified locally."""


class OptionalDependencyError(TrackError, ImportError):
    """Raised when a selected provider requires an uninstalled extra."""


class UserSuppliedFilesError(TrackError, FileNotFoundError):
    """Raised when licensed or otherwise non-redistributable inputs are absent."""


class TrackUnavailableError(TrackError):
    """Raised for a declared track whose upstream has not been published yet."""
