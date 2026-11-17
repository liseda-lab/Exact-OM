"""Optional Hugging Face snapshot provider for YAML-described tracks."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from .base import BaseDescriptorProvider, FetchResult
from .descriptor import TrackDescriptor
from .provider import DescriptorError, IntegrityError, OptionalDependencyError


class HfClient(Protocol):
    """Small injectable seam around ``huggingface_hub``."""

    def resolve_revision(self, repo_id: str, revision: str) -> str:
        """Resolve a tag, branch, or commit reference to an immutable commit SHA."""

    def snapshot_download(
        self,
        repo_id: str,
        revision: str,
        destination: Path,
        *,
        allow_patterns: Sequence[str] | None = None,
        ignore_patterns: Sequence[str] | None = None,
    ) -> Path:
        """Download a dataset snapshot and return its local root."""


class HuggingFaceHubClient:
    """Lazily imported ``huggingface_hub`` implementation."""

    @staticmethod
    def _api() -> Any:
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise OptionalDependencyError(
                "Hugging Face tracks require the optional dependency. Install `exact-om[hf]` "
                "or `pip install huggingface_hub`."
            ) from exc
        return HfApi()

    def resolve_revision(self, repo_id: str, revision: str) -> str:
        """Resolve a dataset revision through the Hugging Face API."""

        info = self._api().dataset_info(repo_id=repo_id, revision=revision)
        if not info.sha:
            raise IntegrityError(f"Hugging Face returned no commit SHA for {repo_id}@{revision}")
        return str(info.sha)

    def snapshot_download(
        self,
        repo_id: str,
        revision: str,
        destination: Path,
        *,
        allow_patterns: Sequence[str] | None = None,
        ignore_patterns: Sequence[str] | None = None,
    ) -> Path:
        """Download an immutable dataset snapshot into ``destination``."""

        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise OptionalDependencyError(
                "Hugging Face tracks require the optional dependency. Install `exact-om[hf]` "
                "or `pip install huggingface_hub`."
            ) from exc
        result = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            local_dir=str(destination),
            allow_patterns=list(allow_patterns) if allow_patterns else None,
            ignore_patterns=list(ignore_patterns) if ignore_patterns else None,
        )
        return Path(result)


class HfProvider(BaseDescriptorProvider):
    """Materialize an immutable Hugging Face dataset snapshot."""

    backend = "hf"

    def __init__(self, descriptor: TrackDescriptor, *, client: HfClient | None = None) -> None:
        if descriptor.provider != self.backend:
            raise DescriptorError(
                f"Descriptor {descriptor.name!r} selects {descriptor.provider!r}, not Hugging Face"
            )
        super().__init__(descriptor)
        allowed = {"repo_id", "revision", "allow_patterns", "ignore_patterns", "checksum_manifest"}
        unknown = set(descriptor.upstream) - allowed
        if unknown:
            raise DescriptorError(
                f"Unknown Hugging Face upstream key(s): {', '.join(sorted(unknown))}"
            )
        repo_id = str(descriptor.upstream.get("repo_id", "")).strip()
        if not repo_id or "/" not in repo_id:
            raise DescriptorError("Hugging Face upstream.repo_id must be an 'owner/repository' id")
        revision = str(descriptor.upstream.get("revision", "")).strip()
        if not revision:
            raise DescriptorError("Hugging Face upstream.revision is required")
        self.repo_id = repo_id
        self.default_revision = revision
        self.allow_patterns = self._patterns(
            descriptor.upstream.get("allow_patterns"), "allow_patterns"
        )
        self.ignore_patterns = self._patterns(
            descriptor.upstream.get("ignore_patterns"), "ignore_patterns"
        )
        self.client = client or HuggingFaceHubClient()

    @staticmethod
    def _patterns(value: Any, name: str) -> tuple[str, ...] | None:
        if value is None:
            return None
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise DescriptorError(f"Hugging Face upstream.{name} must be a list of strings")
        return tuple(value)

    def _fetch(
        self, destination: Path, revision: str | None, task: str, update: bool
    ) -> FetchResult:
        requested = revision or self.default_revision
        try:
            resolved = self.client.resolve_revision(self.repo_id, requested)
            snapshot = self.client.snapshot_download(
                self.repo_id,
                resolved,
                destination,
                allow_patterns=self.allow_patterns,
                ignore_patterns=self.ignore_patterns,
            )
        except OptionalDependencyError:
            raise
        except Exception as exc:
            raise IntegrityError(
                f"Could not materialize Hugging Face track {self.repo_id}@{requested}: {exc}"
            ) from exc
        snapshot = Path(snapshot).resolve()
        destination = destination.resolve()
        if snapshot != destination:
            self._copy_snapshot(snapshot, destination)
        return FetchResult(
            root=destination,
            upstream_id=self.repo_id,
            revision=resolved,
            revision_ref=requested,
        )

    @staticmethod
    def _copy_snapshot(source: Path, destination: Path) -> None:
        if not source.is_dir():
            raise IntegrityError(f"Hugging Face snapshot root is not a directory: {source}")
        for child in source.rglob("*"):
            if child.is_symlink():
                raise IntegrityError(f"Hugging Face snapshot contains a symbolic link: {child}")
        shutil.copytree(source, destination, dirs_exist_ok=True)

    def _check_upstream(self, entry: Mapping[str, Any]) -> tuple[bool, tuple[str, ...]]:
        revision_ref = entry.get("revision_ref") or entry.get("revision")
        try:
            current = self.client.resolve_revision(self.repo_id, str(revision_ref))
        except OptionalDependencyError:
            raise
        except Exception as exc:
            return False, (
                f"Could not check whether Hugging Face upstream moved for "
                f"{self.repo_id}@{revision_ref}: {exc}",
            )
        return current != entry.get("revision"), ()
