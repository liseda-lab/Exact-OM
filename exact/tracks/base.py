"""Shared transaction, mapping, and verification logic for descriptor providers."""

from __future__ import annotations

import copy
import json
import os
import shutil
import tempfile
import uuid
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .descriptor import ResourceSpec, TaskSpec, TrackDescriptor, UserSuppliedSpec
from .lockfile import (
    hash_tree,
    lock_operation,
    path_within,
    read_lock,
    relative_to_root,
    sha256_file,
    verify_hashes,
    write_lock,
)
from .provider import (
    IntegrityError,
    LocalDriftError,
    TaskLayout,
    TrackStatus,
    TrackUnavailableError,
    UserSuppliedFilesError,
    VerificationReport,
)
from .transforms import TransformResult, apply_transform


@dataclass(frozen=True)
class FetchResult:
    """A fetched upstream snapshot and the immutable metadata used to pin it."""

    root: Path
    upstream_id: str
    revision: str
    revision_ref: str | None = None
    validators: tuple[Mapping[str, Any], ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class _PlacedResource:
    result: TransformResult | None
    copied_files: Mapping[str, str]


class BaseDescriptorProvider:
    """Base implementation for declarative HTTP and Hugging Face providers."""

    backend: str

    def __init__(self, descriptor: TrackDescriptor):
        self.descriptor = descriptor
        self.name = descriptor.name

    def tasks(self) -> list[str]:
        """List descriptor tasks in deterministic order."""

        return sorted(self.descriptor.tasks)

    def _task(self, task: str) -> TaskSpec:
        try:
            return self.descriptor.tasks[task]
        except KeyError as exc:
            available = ", ".join(self.tasks()) or "none"
            raise KeyError(
                f"Unknown task {task!r} for track {self.name!r}; available: {available}"
            ) from exc

    def _fetch(
        self, destination: Path, revision: str | None, task: str, update: bool
    ) -> FetchResult:
        raise NotImplementedError

    def _check_upstream(self, entry: Mapping[str, Any]) -> tuple[bool, tuple[str, ...]]:
        raise NotImplementedError

    def _entry_key(self, task: str) -> str:
        return f"{self.name}/{task}"

    def _task_root(self, data_root: Path, task: str) -> Path:
        if task in {"", ".", ".."} or "/" in task or "\\" in task:
            raise KeyError(f"Unsafe task identifier: {task!r}")
        return Path(data_root) / self.name / task

    def materialize(
        self,
        task: str,
        data_root: Path,
        *,
        revision: str | None = None,
        update: bool = False,
    ) -> TaskLayout:
        """Materialize one task transactionally and write its immutable lock entry."""

        task_spec = self._task(task)
        data_root = Path(data_root).expanduser().resolve()
        data_root.mkdir(parents=True, exist_ok=True)
        key = self._entry_key(task)
        with lock_operation(data_root):
            lock = read_lock(data_root)
            previous = lock["tasks"].get(key)
            if previous is not None and not update:
                locked_ref = previous.get("revision_ref") or previous.get("revision")
                if revision is not None and revision != locked_ref:
                    raise IntegrityError(
                        f"{key} is pinned to {locked_ref!r}; use update=True to repin to {revision!r}"
                    )
                issues, _ = verify_hashes(previous.get("files", {}), data_root)
                if issues:
                    raise LocalDriftError(
                        f"{key} has local drift; repair the files or repin explicitly with update=True: "
                        + "; ".join(issues)
                    )
                return self._layout_from_entry(data_root, previous)

            final_root = self._task_root(data_root, task)
            if previous is None and final_root.exists() and not update:
                raise IntegrityError(
                    f"Refusing to overwrite untracked task directory {final_root}; move it away or use update=True"
                )

            with tempfile.TemporaryDirectory(
                prefix=f".exact-{self.name}-{task}-", dir=data_root
            ) as temporary:
                staging_root = Path(temporary)
                upstream_root = staging_root / "upstream"
                upstream_root.mkdir()
                fetched = self._fetch(upstream_root, revision, task, update)
                try:
                    fetched.root.resolve().relative_to(staging_root.resolve())
                except ValueError as exc:
                    raise IntegrityError(
                        f"Provider returned a snapshot outside its staging directory: {fetched.root}"
                    ) from exc
                self._reject_snapshot_symlinks(fetched.root)
                supplied_expected, supplied_warnings = self._inject_user_supplied(
                    task_spec, fetched.root, data_root
                )
                vendor_expected = self._validate_checksum_manifests(fetched.root)
                staged_task = staging_root / "task"
                staged_task.mkdir()
                layout, copied_files = self._build_layout(task_spec, fetched.root, staged_task)

                declared_by_output: dict[str, dict[str, Any]] = {}
                all_expected: dict[str, tuple[str, bool, str]] = {
                    relative: (digest, True, "upstream checksum manifest")
                    for relative, digest in vendor_expected.items()
                }
                all_expected.update(supplied_expected)
                for source_relative, output_relative in copied_files.items():
                    expected = all_expected.get(source_relative)
                    if expected is None:
                        continue
                    digest, enforce, origin = expected
                    declared_by_output[output_relative] = {
                        "sha256": digest,
                        "enforce": enforce,
                        "origin": origin,
                    }

                serialized_layout = self._serialize_layout(layout, staged_task)
                task_relative = relative_to_root(final_root, data_root)
                staged_files = hash_tree(staged_task, staged_task)
                files = {
                    f"{task_relative}/{relative}": record
                    for relative, record in staged_files.items()
                }
                declared_hashes = {
                    f"{task_relative}/{relative}": record
                    for relative, record in declared_by_output.items()
                }
                entry: dict[str, Any] = {
                    "provider": self.name,
                    "backend": self.backend,
                    "provider_version": self.descriptor.provider_version,
                    "upstream_id": fetched.upstream_id,
                    "revision": fetched.revision,
                    "revision_ref": fetched.revision_ref,
                    "validators": [dict(item) for item in fetched.validators],
                    "retrieved_at": datetime.now(timezone.utc).isoformat(),
                    "task_root": task_relative,
                    "layout": serialized_layout,
                    "files": files,
                    "declared_hashes": declared_hashes,
                    "upstream_manifest_hashes": vendor_expected,
                    "warnings": list(fetched.warnings) + supplied_warnings,
                }
                self._publish_directory(staged_task, final_root)
                lock["tasks"][key] = entry
                try:
                    write_lock(data_root, lock)
                except Exception:
                    # The task directory is intentionally retained. A subsequent status
                    # reports it as untracked instead of pretending the old pin applies.
                    lock["tasks"].pop(key, None)
                    raise
                return self._layout_from_entry(data_root, entry)

    def verify(self, task: str, data_root: Path) -> VerificationReport:
        """Check local lock hashes, upstream manifests, and mutable upstream refs."""

        self._task(task)
        data_root = Path(data_root).expanduser().resolve()
        lock = read_lock(data_root)
        entry = lock["tasks"].get(self._entry_key(task))
        if entry is None:
            return VerificationReport(self.name, task, "not-materialized")
        if entry.get("provider") != self.name:
            return VerificationReport(
                self.name,
                task,
                "local-drift",
                issues=(f"lock entry belongs to provider {entry.get('provider')!r}",),
                lock_entry=copy.deepcopy(entry),
            )

        issues, checked = verify_hashes(entry.get("files", {}), data_root)
        report_warnings = list(entry.get("warnings", []))
        for relative, record in sorted(entry.get("declared_hashes", {}).items()):
            path = path_within(data_root, relative)
            if not path.is_file():
                continue
            actual = sha256_file(path)
            expected = record.get("sha256")
            if expected and actual != expected:
                message = f"declared checksum mismatch for {relative}: expected {expected}, found {actual}"
                if record.get("enforce", True):
                    issues.append(message)
                else:
                    report_warnings.append(message)

        moved, upstream_warnings = self._check_upstream(entry)
        report_warnings.extend(upstream_warnings)
        status: TrackStatus
        if issues:
            status = "local-drift"
        elif moved:
            status = "upstream-moved"
        else:
            status = "ok"
        return VerificationReport(
            provider=self.name,
            task=task,
            status=status,
            checked_files=checked,
            issues=tuple(issues),
            warnings=tuple(dict.fromkeys(report_warnings)),
            lock_entry=copy.deepcopy(entry),
        )

    def status(self, task: str, data_root: Path) -> TrackStatus:
        """Return the four-state status of a descriptor task."""

        return self.verify(task, data_root).status

    def _inject_user_supplied(
        self, task: TaskSpec, upstream_root: Path, data_root: Path
    ) -> tuple[dict[str, tuple[str, bool, str]], list[str]]:
        missing: list[tuple[Path, UserSuppliedSpec]] = []
        supplied_root = data_root / "user_supplied" / self.name
        for name in task.user_supplied:
            spec = self.descriptor.user_supplied[name]
            path = supplied_root / spec.path
            if not path.is_file() or path.is_symlink():
                missing.append((path, spec))
        if missing:
            details = "\n".join(f"- place {path}: {spec.help}" for path, spec in missing)
            raise UserSuppliedFilesError(
                f"Track {self.name!r} requires user-supplied ontology files:\n{details}"
            )

        expected: dict[str, tuple[str, bool, str]] = {}
        emitted_warnings: list[str] = []
        for name in task.user_supplied:
            spec = self.descriptor.user_supplied[name]
            source = supplied_root / spec.path
            destination = upstream_root / spec.destination
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            expected_digest = self._user_supplied_digest(spec, upstream_root)
            if expected_digest is None:
                continue
            actual = sha256_file(destination)
            origin = f"user-supplied pin {name}"
            expected[spec.destination] = (expected_digest, False, origin)
            if actual != expected_digest:
                message = (
                    f"Licensed file {source} differs from its published pin: expected "
                    f"{expected_digest}, found {actual}. The file is accepted because licensing "
                    "manifests can lag upstream releases."
                )
                warnings.warn(message, UserWarning, stacklevel=3)
                emitted_warnings.append(message)
        return expected, emitted_warnings

    def _user_supplied_digest(self, spec: UserSuppliedSpec, upstream_root: Path) -> str | None:
        if spec.sha256:
            return spec.sha256
        if not spec.sha256_from:
            return None
        manifest = upstream_root / str(spec.sha256_from["path"])
        if not manifest.is_file():
            raise IntegrityError(
                f"Cannot resolve licensed-file checksum: manifest is missing at {manifest}"
            )
        try:
            value = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise IntegrityError(f"Could not read checksum manifest {manifest}: {exc}") from exc
        key = spec.sha256_from.get("key")
        if key:
            current: Any = value
            for component in str(key).split("."):
                if not isinstance(current, Mapping) or component not in current:
                    raise IntegrityError(f"Checksum key {key!r} is absent from {manifest}")
                current = current[component]
            digest = self._extract_digest(current)
        else:
            filename = str(spec.sha256_from.get("filename", Path(spec.path).name))
            digest = self._find_digest(value, filename)
        if digest is None:
            raise IntegrityError(
                f"Could not find a SHA-256 for {spec.path!r} in manifest {manifest}"
            )
        return digest

    @staticmethod
    def _extract_digest(value: Any) -> str | None:
        if isinstance(value, str):
            candidate = value.lower()
            if len(candidate) == 64 and all(ch in "0123456789abcdef" for ch in candidate):
                return candidate
        if isinstance(value, Mapping):
            for key in ("sha256", "checksum", "digest"):
                digest = BaseDescriptorProvider._extract_digest(value.get(key))
                if digest:
                    return digest
        return None

    @staticmethod
    def _find_digest(value: Any, filename: str) -> str | None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if str(key) == filename or (
                    isinstance(item, Mapping)
                    and str(item.get("filename", item.get("path", ""))).endswith(filename)
                ):
                    digest = BaseDescriptorProvider._extract_digest(item)
                    if digest:
                        return digest
            for item in value.values():
                digest = BaseDescriptorProvider._find_digest(item, filename)
                if digest:
                    return digest
        elif isinstance(value, list):
            for item in value:
                digest = BaseDescriptorProvider._find_digest(item, filename)
                if digest:
                    return digest
        return None

    def _validate_checksum_manifests(self, upstream_root: Path) -> dict[str, str]:
        configured = self.descriptor.upstream.get("checksum_manifest", [])
        patterns = [configured] if isinstance(configured, str) else configured
        if not isinstance(patterns, list) or not all(isinstance(item, str) for item in patterns):
            raise IntegrityError("upstream.checksum_manifest must be a string or list of globs")
        expected: dict[str, str] = {}
        for pattern in patterns:
            matches = sorted(upstream_root.glob(pattern))
            if not matches:
                raise IntegrityError(f"Checksum manifest glob {pattern!r} matched no files")
            for manifest in matches:
                if not manifest.is_file() or manifest.is_symlink():
                    raise IntegrityError(f"Invalid checksum manifest: {manifest}")
                for line_number, line in enumerate(
                    manifest.read_text(encoding="utf-8").splitlines(), start=1
                ):
                    stripped = line.strip()
                    if not stripped or stripped.startswith("#"):
                        continue
                    parts = stripped.split(maxsplit=1)
                    if len(parts) != 2:
                        raise IntegrityError(f"Malformed checksum at {manifest}:{line_number}")
                    digest, filename = parts
                    filename = filename.lstrip("* ")
                    if len(digest) != 64 or any(
                        ch not in "0123456789abcdefABCDEF" for ch in digest
                    ):
                        raise IntegrityError(f"Malformed SHA-256 at {manifest}:{line_number}")
                    candidates = (manifest.parent / filename, upstream_root / filename)
                    target = next((path for path in candidates if path.is_file()), None)
                    if target is None:
                        raise IntegrityError(
                            f"Checksum manifest {manifest} references missing file {filename!r}"
                        )
                    try:
                        relative = target.resolve().relative_to(upstream_root.resolve()).as_posix()
                    except ValueError as exc:
                        raise IntegrityError(
                            f"Checksum manifest {manifest} references a path outside the snapshot"
                        ) from exc
                    actual = sha256_file(target)
                    digest = digest.lower()
                    if actual != digest:
                        raise IntegrityError(
                            f"Upstream checksum mismatch for {relative}: expected {digest}, found {actual}"
                        )
                    expected[relative] = digest
        return expected

    @staticmethod
    def _reject_snapshot_symlinks(upstream_root: Path) -> None:
        for path in upstream_root.rglob("*"):
            if path.is_symlink():
                raise IntegrityError(f"Upstream snapshot contains a symbolic link: {path}")

    def _build_layout(
        self, task: TaskSpec, upstream_root: Path, task_root: Path
    ) -> tuple[TaskLayout, dict[str, str]]:
        copied: dict[str, str] = {}
        emitted_extras: dict[str, Any] = {}

        source = self._place_resource(task.source, upstream_root, task_root, "source")
        target = self._place_resource(task.target, upstream_root, task_root, "target")
        if source.result is None or target.result is None:
            raise IntegrityError("Task source and target are mandatory")
        copied.update(source.copied_files)
        copied.update(target.copied_files)

        refs: dict[str, Path] = {}
        for split, spec in task.refs.items():
            placed = self._place_resource(spec, upstream_root, task_root, f"refs/{split}")
            copied.update(placed.copied_files)
            if placed.result is None:
                continue
            refs[split] = placed.result.path
            emitted_extras.update(placed.result.extras)

        candidate_path: Path | None = None
        if task.candidates is not None:
            placed = self._place_resource(
                task.candidates, upstream_root, task_root, "candidates/candidates"
            )
            copied.update(placed.copied_files)
            if placed.result is not None:
                candidate_path = placed.result.path
                emitted_extras.update(placed.result.extras)

        extras = self._materialize_extras(task.extras, upstream_root, task_root, copied)
        extras.update(emitted_extras)
        return (
            TaskLayout(
                source=source.result.path,
                target=target.result.path,
                refs=refs,
                candidates=candidate_path,
                extras=extras,
            ),
            copied,
        )

    def _place_resource(
        self,
        spec: ResourceSpec,
        upstream_root: Path,
        task_root: Path,
        default_output: str,
    ) -> _PlacedResource:
        source = self._resolve_resource(spec, upstream_root)
        if source is None:
            return _PlacedResource(None, {})
        suffix = source.suffix
        if spec.output:
            relative_output = spec.output
        elif spec.transform == "alignment_rdf_to_tsv":
            relative_output = f"{default_output}.tsv"
        elif spec.transform == "pools_jsonl_to_cands_tsv":
            relative_output = f"{default_output}.tsv"
        elif spec.transform == "flatten_refs_equiv":
            relative_output = default_output
        elif source.is_dir():
            relative_output = default_output
        else:
            relative_output = f"{default_output}{suffix}"
        destination = task_root / relative_output
        try:
            destination.resolve().relative_to(task_root.resolve())
        except ValueError as exc:
            raise IntegrityError(
                f"Resource output escapes task directory: {relative_output}"
            ) from exc

        copied: dict[str, str] = {}
        if spec.transform:
            result = apply_transform(spec.transform, source, destination)
        else:
            self._copy_resource(source, destination)
            result = TransformResult(destination)
            if source.is_file():
                copied[source.resolve().relative_to(upstream_root.resolve()).as_posix()] = (
                    destination.resolve().relative_to(task_root.resolve()).as_posix()
                )
            else:
                for source_file in sorted(source.rglob("*")):
                    if source_file.is_file():
                        destination_file = destination / source_file.relative_to(source)
                        copied[
                            source_file.resolve().relative_to(upstream_root.resolve()).as_posix()
                        ] = (destination_file.resolve().relative_to(task_root.resolve()).as_posix())
        return _PlacedResource(result, copied)

    @staticmethod
    def _resolve_resource(spec: ResourceSpec, upstream_root: Path) -> Path | None:
        if spec.path:
            candidate = upstream_root / spec.path
            matches = [candidate] if candidate.exists() else []
        else:
            matches = sorted(upstream_root.glob(spec.glob or ""))
        matches = [path for path in matches if not path.is_symlink()]
        if not matches and spec.optional:
            return None
        if not matches:
            selector = spec.path or spec.glob
            raise IntegrityError(f"Descriptor resource {selector!r} matched no files")
        if len(matches) != 1:
            selector = spec.path or spec.glob
            listed = ", ".join(str(path.relative_to(upstream_root)) for path in matches[:5])
            raise IntegrityError(
                f"Descriptor resource {selector!r} must match exactly one path; found {len(matches)}: {listed}"
            )
        candidate = matches[0]
        try:
            candidate.resolve().relative_to(upstream_root.resolve())
        except ValueError as exc:
            raise IntegrityError(f"Resource escapes upstream snapshot: {candidate}") from exc
        return candidate

    @staticmethod
    def _copy_resource(source: Path, destination: Path) -> None:
        if source.is_symlink():
            raise IntegrityError(f"Refusing to materialize symbolic link {source}")
        if source.is_dir():
            for child in source.rglob("*"):
                if child.is_symlink():
                    raise IntegrityError(f"Refusing to materialize symbolic link {child}")
            shutil.copytree(source, destination)
        elif source.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        else:
            raise IntegrityError(f"Resource is not a regular file or directory: {source}")

    def _materialize_extras(
        self,
        value: Mapping[str, Any],
        upstream_root: Path,
        task_root: Path,
        copied: dict[str, str],
        prefix: str = "extras",
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(item, Mapping) and ("path" in item or "glob" in item):
                spec = ResourceSpec.parse(item, f"tasks.extras.{key}")
                placed = self._place_resource(spec, upstream_root, task_root, f"{prefix}/{key}")
                copied.update(placed.copied_files)
                if placed.result is None:
                    continue
                result[key] = placed.result.path
                result.update(placed.result.extras)
            elif isinstance(item, Mapping):
                result[key] = self._materialize_extras(
                    item, upstream_root, task_root, copied, f"{prefix}/{key}"
                )
            elif isinstance(item, list):
                result[key] = copy.deepcopy(item)
            else:
                result[key] = item
        return result

    @staticmethod
    def _publish_directory(staged: Path, final: Path) -> None:
        final.parent.mkdir(parents=True, exist_ok=True)
        if not final.exists():
            os.replace(staged, final)
            return
        backup = final.with_name(f".{final.name}.backup-{uuid.uuid4().hex}")
        os.replace(final, backup)
        try:
            os.replace(staged, final)
        except Exception:
            os.replace(backup, final)
            raise
        shutil.rmtree(backup)

    def _serialize_layout(self, layout: TaskLayout, staged_root: Path) -> dict[str, Any]:
        def relative(path: Path) -> str:
            return path.resolve().relative_to(staged_root.resolve()).as_posix()

        def encode(value: Any) -> Any:
            if isinstance(value, Path):
                return {"__path__": relative(value)}
            if isinstance(value, Mapping):
                return {str(key): encode(item) for key, item in value.items()}
            if isinstance(value, tuple):
                return [encode(item) for item in value]
            if isinstance(value, list):
                return [encode(item) for item in value]
            return value

        return {
            "source": relative(layout.source),
            "target": relative(layout.target),
            "refs": {name: relative(path) for name, path in layout.refs.items()},
            "candidates": relative(layout.candidates) if layout.candidates else None,
            "extras": encode(layout.extras),
        }

    @staticmethod
    def _layout_from_entry(data_root: Path, entry: Mapping[str, Any]) -> TaskLayout:
        task_root = path_within(data_root, str(entry["task_root"]))
        layout = entry.get("layout", {})

        def path(relative: str) -> Path:
            candidate = (task_root / relative).resolve()
            try:
                candidate.relative_to(task_root.resolve())
            except ValueError as exc:
                raise IntegrityError(f"Unsafe layout path in dataset lock: {relative!r}") from exc
            return candidate

        def decode(value: Any) -> Any:
            if isinstance(value, Mapping) and set(value) == {"__path__"}:
                return path(str(value["__path__"]))
            if isinstance(value, Mapping):
                return {str(key): decode(item) for key, item in value.items()}
            if isinstance(value, list):
                return [decode(item) for item in value]
            return value

        files = {
            relative: record.get("sha256")
            for relative, record in entry.get("files", {}).items()
            if isinstance(record, Mapping)
        }
        provenance = {
            "provider": entry.get("provider"),
            "backend": entry.get("backend"),
            "provider_version": entry.get("provider_version"),
            "upstream_id": entry.get("upstream_id"),
            "revision": entry.get("revision"),
            "revision_ref": entry.get("revision_ref"),
            "retrieved_at": entry.get("retrieved_at"),
            "hashes": files,
        }
        return TaskLayout(
            source=path(str(layout["source"])),
            target=path(str(layout["target"])),
            refs={name: path(str(item)) for name, item in layout.get("refs", {}).items()},
            candidates=(path(str(layout["candidates"])) if layout.get("candidates") else None),
            extras=decode(layout.get("extras", {})),
            provenance=provenance,
        )


class UnavailableDescriptorProvider:
    """Provider for a published descriptor whose remote dataset is not available yet."""

    def __init__(self, descriptor: TrackDescriptor):
        self.descriptor = descriptor
        self.name = descriptor.name

    def tasks(self) -> list[str]:
        """Return any task names already declared by the future track."""

        return sorted(self.descriptor.tasks)

    def materialize(
        self,
        task: str,
        data_root: Path,
        *,
        revision: str | None = None,
        update: bool = False,
    ) -> TaskLayout:
        """Fail actionably until the descriptor receives a real upstream."""

        raise TrackUnavailableError(
            f"Track {self.name!r} has not been published yet. Update its YAML descriptor "
            "with the repository id and revision once the upstream becomes available."
        )

    def verify(self, task: str, data_root: Path) -> VerificationReport:
        """Report an unpublished task as not materialized."""

        return VerificationReport(self.name, task, "not-materialized")

    def status(self, task: str, data_root: Path) -> TrackStatus:
        """Report an unpublished task as not materialized."""

        return "not-materialized"
