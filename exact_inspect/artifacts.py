"""Inert portable bundles, bounded local import and atomic artifact publication."""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import tempfile
import threading
import zipfile
from collections import OrderedDict
from contextlib import contextmanager, nullcontext
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Literal

from pydantic import Field

from .contracts import (
    DomainError,
    VisibilityPolicy,
    WireModel,
    canonical_hash,
    canonical_json,
    file_hash,
)


def atomic_json(path: Path, value: Any) -> None:
    """Publish durable JSON by same-filesystem rename; readers never see partial bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".publish-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(canonical_json(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        Path(temporary).unlink(missing_ok=True)


def read_metadata(path: Path) -> dict[str, Any]:
    """Read an inert metadata object with the same 8 MiB bound at every level."""
    if path.is_symlink():
        raise DomainError("invalid_manifest", "Linked manifests are not supported", 422)
    with path.open("rb") as stream:
        raw = stream.read(8 * 1024**2 + 1)
    if len(raw) > 8 * 1024**2:
        raise DomainError("invalid_manifest", "Manifest exceeds its allowed bound", 413)
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise DomainError("invalid_manifest", "Manifest must be a JSON object", 422)
    return value


def relative_path(root: Path, locator: str) -> Path:
    """Resolve an inert relative locator, refusing traversal and symlink components."""
    pure = PurePosixPath(locator)
    if (
        not locator
        or "\\" in locator
        or pure.is_absolute()
        or ":" in locator
        or any(part in {".", ".."} for part in locator.split("/"))
    ):
        raise DomainError("unsafe_path", "Artifact locator must be a safe relative path")
    result = root
    for part in pure.parts:
        result = result / part
        if result.is_symlink():
            raise DomainError("unsafe_path", "Symbolic links are not bundle artifacts")
    if not result.resolve().is_relative_to(root.resolve()):
        raise DomainError("unsafe_path", "Artifact escapes package root")
    return result


class Artifact(WireModel):
    """A content-bound inert file in a relocatable package."""

    path: str
    sha256: str = Field(pattern=r"^sha256:[a-f0-9]{64}$")
    size: int = Field(ge=0)
    media_type: str = "application/json"


class InspectionBundle(WireModel):
    """The immutable root manifest; filesystem locators never determine identities."""

    schema_version: Literal["exact-inspection-bundle/1"] = "exact-inspection-bundle/1"
    contract_version: Literal["exact-explain/1.0"] = "exact-explain/1.0"
    package_id: str
    audience: Literal["local", "development_demo"] = "local"
    policy: VisibilityPolicy = Field(default_factory=VisibilityPolicy)
    artifacts: list[Artifact]
    ontologies: dict[str, str] = Field(default_factory=dict)
    runs: dict[str, str] = Field(default_factory=dict)
    explanations: dict[str, str] = Field(default_factory=dict)
    jobs: dict[str, str] = Field(default_factory=dict)
    capabilities: dict[str, str] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    license_treatment: dict[str, str] = Field(default_factory=dict)
    portable: bool = True


def validate_bundle(path: Path, *, verify_hashes: bool = True) -> InspectionBundle:
    """Validate all references and hashes before making a package visible to readers."""
    manifest = InspectionBundle.model_validate(read_metadata(path))
    root = path.parent
    seen: set[str] = set()
    for artifact in manifest.artifacts:
        if artifact.path in seen:
            raise DomainError("duplicate_artifact", "Manifest contains duplicate locators")
        seen.add(artifact.path)
        resolved = relative_path(root, artifact.path)
        if not resolved.is_file() or resolved.stat().st_size != artifact.size:
            raise DomainError(
                "corrupt_artifact", "A package artifact is missing or incomplete", 409
            )
        if verify_hashes and file_hash(resolved) != artifact.sha256:
            raise DomainError(
                "corrupt_artifact", "A package artifact failed checksum verification", 409
            )
    for mapping in (manifest.ontologies, manifest.runs, manifest.explanations, manifest.jobs):
        for locator in mapping.values():
            relative_path(root, locator)
            if locator not in seen and not any(
                p.startswith(locator.rstrip("/") + "/") for p in seen
            ):
                raise DomainError("unbound_artifact", "Resource is not bound to package contents")
    expected = canonical_hash(manifest.model_dump(mode="json", exclude={"package_id"}))
    if manifest.package_id != expected:
        raise DomainError("invalid_identity", "Package identity does not match manifest", 409)
    return manifest


def publish_bundle(root: Path, **metadata: Any) -> Path:
    """Freeze prepared files into a validated manifest without parsing or generating data."""
    artifacts = [
        Artifact(
            path=str(path.relative_to(root)),
            sha256=file_hash(path),
            size=path.stat().st_size,
            media_type=(
                "application/vnd.sqlite3" if path.suffix == ".sqlite" else "application/json"
            ),
        )
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "package.json" and not path.name.startswith(".")
    ]
    manifest = InspectionBundle(package_id="pending", artifacts=artifacts, **metadata)
    manifest = manifest.model_copy(
        update={
            "package_id": canonical_hash(manifest.model_dump(mode="json", exclude={"package_id"}))
        }
    )
    atomic_json(root / "package.json", manifest)
    validate_bundle(root / "package.json")
    return root / "package.json"


def export_archive(manifest_path: Path, destination: Path) -> None:
    """Export only checksum-bound data; a ZIP contains no implicit executable payloads."""
    manifest = validate_bundle(manifest_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".archive-", dir=destination.parent)
    os.close(fd)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(manifest_path, "package.json")
            for artifact in manifest.artifacts:
                archive.write(relative_path(manifest_path.parent, artifact.path), artifact.path)
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


class BundleLibrary:
    """Bounded, atomic local imports; failures leave the selected package unchanged."""

    def __init__(
        self,
        root: Path,
        *,
        max_bytes: int = 32 * 1024**3,
        max_files: int = 10000,
        max_ratio: int = 200,
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.root = self.root.resolve()
        self.max_bytes, self.max_files, self.max_ratio = max_bytes, max_files, max_ratio

    @contextmanager
    def operation(self):
        """Serialize publication, selection and removal across local serving processes."""
        import fcntl

        fd = os.open(self.root / ".library.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise DomainError(
                    "library_busy", "A library import or change is in progress", 409
                ) from exc
            yield
        finally:
            os.close(fd)

    def package_path(self, package_id: str) -> Path:
        """Resolve only content-addressed library copies, never caller-supplied paths."""
        if re.fullmatch(r"sha256:[a-f0-9]{64}", package_id) is None:
            raise DomainError("unknown_package", "Unknown package", 404)
        path = relative_path(self.root, package_id[7:] + "/package.json")
        if not path.is_file():
            raise DomainError("unknown_package", "Unknown package", 404)
        return path

    def lease(self, package_id: str):
        """Hold a shared process-safe lease while an immutable service can read a copy."""
        import fcntl

        self.package_path(package_id)
        path = self.root / (".use-" + package_id[7:])
        fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BaseException:
            os.close(fd)
            raise
        return fd

    def metadata(self, package_id: str) -> dict[str, Any]:
        """Return bounded identifying metadata without reading any ontology or run data."""
        path = self.package_path(package_id)
        manifest = InspectionBundle.model_validate(read_metadata(path))
        if (
            manifest.package_id != package_id
            or canonical_hash(manifest.model_dump(mode="json", exclude={"package_id"}))
            != package_id
        ):
            raise DomainError("invalid_identity", "Package identity does not match manifest", 409)
        marker = path.parent / ".library-copy.json"
        owned = marker.is_file() and read_metadata(marker).get("package_id") == package_id
        value = {
            "package_id": package_id,
            "contract_version": manifest.contract_version,
            "audience": manifest.audience,
            "portable": manifest.portable,
            "capabilities": manifest.capabilities,
            "policy_id": manifest.policy.policy_id,
            "policy_hash": manifest.policy.policy_hash,
            "license_treatment": manifest.license_treatment,
            "counts": {
                name: len(getattr(manifest, name))
                for name in ("ontologies", "runs", "explanations", "jobs", "artifacts")
            },
            "artifact_bytes": sum(item.size for item in manifest.artifacts),
            "owned_library_copy": owned,
        }
        if len(canonical_json(value)) > 64 * 1024:
            raise DomainError(
                "metadata_too_large", "Package metadata exceeds its allowed bound", 413
            )
        return value

    def remove(self, package_id: str) -> dict[str, Any]:
        """Remove an idle owned copy; never follow links or remove a mounted original."""
        import fcntl

        with self.operation():
            path = self.package_path(package_id)
            if not self.metadata(package_id)["owned_library_copy"]:
                raise DomainError(
                    "unowned_package",
                    "Only copies created by this library importer can be removed",
                    409,
                )
            selection = self.root / "selection.json"
            if selection.is_file() and read_metadata(selection).get("package_id") == package_id:
                raise DomainError(
                    "package_in_use", "Select another bundle before removing this copy", 409
                )
            fd = os.open(
                self.root / (".use-" + package_id[7:]),
                os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW,
                0o600,
            )
            try:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise DomainError(
                        "package_in_use", "This library copy is still in use", 409
                    ) from exc
                # Reject links and mounted subtrees before deletion. fd-based rmtree
                # also refuses a directory swapped for a symlink during traversal.
                device = self.root.stat().st_dev
                for root, directories, files in os.walk(path.parent, followlinks=False):
                    for entry in [Path(root), *(Path(root) / name for name in directories + files)]:
                        if (
                            entry.is_symlink()
                            or entry.stat().st_dev != device
                            or os.path.ismount(entry)
                        ):
                            raise DomainError(
                                "unsafe_path",
                                "Linked or mounted files are not removable library copies",
                                409,
                            )
                if not shutil.rmtree.avoids_symlink_attacks:
                    raise DomainError(
                        "deletion_unsupported",
                        "Safe library deletion is unavailable on this platform",
                        409,
                    )
                shutil.rmtree(path.parent)
            finally:
                os.close(fd)
            return {"package_id": package_id, "status": "removed"}

    def import_archive(
        self,
        archive_path: Path,
        *,
        cancelled: Callable[[], bool] = lambda: False,
        operation_locked: bool = False,
    ) -> InspectionBundle:
        """Reject unsafe ZIP entries and stream bounded contents into isolated staging."""
        if archive_path.stat().st_size > self.max_bytes:
            raise DomainError("payload_too_large", "Upload exceeds the local import limit", 413)
        with nullcontext() if operation_locked else self.operation(), tempfile.TemporaryDirectory(
            prefix=".import-", dir=self.root
        ) as directory:
            staging = Path(directory)
            try:
                with zipfile.ZipFile(archive_path) as archive:
                    entries = archive.infolist()
                    if (
                        len(entries) > self.max_files
                        or sum(x.file_size for x in entries) > self.max_bytes
                    ):
                        raise DomainError(
                            "payload_too_large", "Expanded archive exceeds import limits", 413
                        )
                    seen: set[str] = set()
                    actual = 0
                    for entry in entries:
                        target = relative_path(staging, entry.filename.rstrip("/"))
                        mode = entry.external_attr >> 16
                        if (
                            entry.filename in seen
                            or stat.S_ISLNK(mode)
                            or (stat.S_IFMT(mode) not in {0, stat.S_IFREG, stat.S_IFDIR})
                            or entry.flag_bits & 1
                        ):
                            raise DomainError(
                                "unsafe_archive",
                                "Duplicate, linked, encrypted or special archive entry",
                            )
                        seen.add(entry.filename)
                        if entry.file_size > max(1, entry.compress_size) * self.max_ratio:
                            raise DomainError(
                                "payload_too_large",
                                "Archive decompression ratio exceeds limit",
                                413,
                            )
                        if entry.is_dir():
                            target.mkdir(parents=True, exist_ok=True)
                            continue
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with archive.open(entry) as source, target.open("xb") as output:
                            while block := source.read(1024 * 1024):
                                if cancelled():
                                    raise DomainError(
                                        "import_cancelled", "Import was cancelled", 409
                                    )
                                actual += len(block)
                                if actual > self.max_bytes:
                                    raise DomainError(
                                        "payload_too_large",
                                        "Expanded archive exceeds import limit",
                                        413,
                                    )
                                output.write(block)
                manifest = validate_bundle(staging / "package.json")
                allowed = {a.path for a in manifest.artifacts} | {"package.json"}
                if {p for p in seen if not p.endswith("/")} != allowed:
                    raise DomainError("unbound_artifact", "Archive contains unmanifested files")
                destination = self.root / manifest.package_id.removeprefix("sha256:")
                relative_path(self.root, manifest.package_id[7:])
                if destination.exists():
                    validate_bundle(destination / "package.json")
                else:
                    os.replace(staging, destination)
                    atomic_json(
                        destination / ".library-copy.json", {"package_id": manifest.package_id}
                    )
                return manifest
            except (zipfile.BadZipFile, OSError) as exc:
                raise DomainError(
                    "corrupt_archive", "Archive is corrupt or incomplete", 409
                ) from exc

    def list(self, *, limit: int = 100, after: str = "") -> list[dict[str, Any]]:
        """List published manifests only; incomplete staging is never a library entry."""
        if limit < 1 or limit > 101:
            raise DomainError("invalid_limit", "Library limit must be between 1 and 100")
        identities = sorted(
            "sha256:" + p.name for p in self.root.iterdir() if re.fullmatch(r"[a-f0-9]{64}", p.name)
        )
        return [
            self.metadata(identity)
            for identity in [item for item in identities if item > after][:limit]
        ]

    def select(self, package_id: str, *, operation_locked: bool = False) -> Path:
        """Persist local selection only after validating the requested library package."""
        with nullcontext() if operation_locked else self.operation():
            path = self.package_path(package_id)
            validate_bundle(path)
            atomic_json(self.root / "selection.json", {"package_id": package_id})
            return path


class BoundedCache:
    """Thread-safe LRU of immutable JSON bytes, bounded by both bytes and entries."""

    def __init__(self, max_entries: int = 128, max_bytes: int = 32 * 1024**2):
        self.max_entries, self.max_bytes = max_entries, max_bytes
        self.bytes = 0
        self._values: OrderedDict[str, bytes] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        """Return a fresh decoded value to prevent mutation between concurrent readers."""
        with self._lock:
            data = self._values.get(key)
            if data is None:
                return None
            self._values.move_to_end(key)
            return json.loads(data)

    def put(self, key: str, value: Any) -> None:
        """Evict least recently used items; oversized entries are never retained."""
        data = canonical_json(value)
        with self._lock:
            old = self._values.pop(key, None)
            self.bytes -= len(old) if old is not None else 0
            if len(data) > self.max_bytes:
                return
            self._values[key] = data
            self.bytes += len(data)
            while len(self._values) > self.max_entries or self.bytes > self.max_bytes:
                _, evicted = self._values.popitem(last=False)
                self.bytes -= len(evicted)
