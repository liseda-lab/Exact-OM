"""Immutable experiment artifacts, relocatable checkpoints, and selective repair."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import socket
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from exact.utils.provenance import sha256_file

SCHEMA_VERSION = 2
STAGE_PARENTS = {
    "inputs": (),
    "ontology": ("inputs",),
    "embeddings": ("inputs", "ontology"),
    "candidates": ("inputs", "ontology", "embeddings"),
    "evidence": ("ontology", "embeddings", "candidates"),
    "fitted_heads": ("inputs", "candidates", "evidence"),
    "pair_scores": ("evidence", "fitted_heads"),
    "llm_requests": ("evidence", "candidates", "pair_scores"),
    "llm_responses": ("llm_requests",),
    "decisions": ("pair_scores", "llm_responses", "fitted_heads"),
    "extraction": ("decisions", "ontology"),
    "typing": ("extraction", "ontology"),
    "evaluation": ("inputs", "decisions", "extraction", "typing"),
    "reports": ("evaluation",),
}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _valid_id(value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"Invalid artifact/content ID: {value!r}")
    return value


def _relative(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"Unsafe artifact path: {value!r}")
    return path


def _publish_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish once after fsync; a failed write never replaces a valid boundary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical(payload) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


def implementation_identity(root: Path, entrypoints: Iterable[str | Path]) -> dict[str, Any]:
    """Hash explicit files and transitively imported local Python modules.

    Dynamic imports and non-Python resources must be supplied as entrypoints.
    Full source bytes are hashed; documentation outside this graph is irrelevant.
    """
    root = Path(root).resolve()
    pending = [root / Path(name) for name in entrypoints]
    files: dict[str, str] = {}
    while pending:
        path = pending.pop().resolve()
        name = path.relative_to(root).as_posix()
        if name in files:
            continue
        files[name] = sha256_file(path)
        if path.suffix != ".py":
            continue
        package = list(path.relative_to(root).with_suffix("").parts)[:-1]
        if path.name == "__init__.py":
            package = list(path.parent.relative_to(root).parts)
        imports: set[str] = set()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                prefix = package[: len(package) - node.level + 1] if node.level else []
                target = ".".join([*prefix, *([node.module] if node.module else [])])
                if target:
                    imports.add(target)
                    imports.update(f"{target}.{alias.name}" for alias in node.names)
        for imported in imports:
            parts = imported.split(".")
            for end in range(1, len(parts) + 1):
                base = root.joinpath(*parts[:end])
                for candidate in (base.with_suffix(".py"), base / "__init__.py"):
                    if candidate.is_file():
                        pending.append(candidate)
    return {"sha256": _digest(files), "files": dict(sorted(files.items()))}


def stage_identity(
    stage: str,
    *,
    parameters: Mapping[str, Any],
    inputs: Mapping[str, str],
    role: str,
    entity_kind: str,
    implementation: Mapping[str, Any],
    dependencies: Mapping[str, Any],
    parents: Sequence[str] = (),
    seed: int | None = None,
) -> dict[str, Any]:
    """Build a numerical key from stage-specific semantic, content and backend locks."""
    if stage not in STAGE_PARENTS or not role or not entity_kind or not implementation:
        raise ValueError("Stage identity requires a known stage, role, kind, and implementation")
    for digest in [*inputs.values(), *parents]:
        _valid_id(digest)
    identity = {
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "parameters": dict(parameters),
        "inputs": dict(inputs),
        "parents": sorted(set(parents)),
        "role": role,
        "entity_kind": entity_kind,
        "implementation": dict(implementation),
        "dependencies": dict(dependencies),
        "seed": seed,
    }
    return {**identity, "artifact_id": _digest(identity)}


def invalidation_closure(stages: Iterable[str]) -> list[str]:
    """Conservatively expand repair scope through the semantic dependency graph."""
    affected = set(stages)
    if affected - STAGE_PARENTS.keys():
        raise ValueError(f"Unknown repair stages: {sorted(affected - STAGE_PARENTS.keys())}")
    while True:
        expanded = affected | {
            stage for stage, parents in STAGE_PARENTS.items() if affected.intersection(parents)
        }
        if expanded == affected:
            return sorted(affected)
        affected = expanded


class ArtifactStore:
    """Immutable manifests and deduplicated bytes relative to a campaign root."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.directory = self.root / "artifacts"

    def _manifest_path(self, artifact_id: str) -> Path:
        return self.directory / "stages" / f"{_valid_id(artifact_id)}.json"

    def _blob(self, digest: str) -> Path:
        return self.directory / "blobs" / _valid_id(digest)

    @contextmanager
    def lock(self, artifact_id: str) -> Iterator[None]:
        """Acquire a single-writer lease; stale leases require explicit recovery."""
        path = self.directory / "locks" / f"{_valid_id(artifact_id)}.json"
        owner = {"pid": os.getpid(), "host": socket.gethostname(), "nonce": uuid.uuid4().hex}
        try:
            _publish_json(path, owner)
        except FileExistsError as exc:
            raise RuntimeError(
                f"Artifact writer already holds {path}; inspect before recovery"
            ) from exc
        try:
            yield
        finally:
            if path.exists() and json.loads(path.read_text()) == owner:
                path.unlink()

    def recover_stale_lock(self, artifact_id: str) -> None:
        """Remove a lease only when its local owner process is confirmed absent."""
        path = self.directory / "locks" / f"{_valid_id(artifact_id)}.json"
        owner = json.loads(path.read_text())
        if owner.get("host") != socket.gethostname():
            raise ValueError("Cannot establish that a writer on another host is gone")
        try:
            os.kill(int(owner["pid"]), 0)
        except ProcessLookupError:
            path.unlink()
        else:
            raise ValueError("Writer is still alive; refusing stale-lock recovery")

    def _put(self, value: Path | bytes) -> dict[str, Any]:
        directory = self.directory / "blobs"
        directory.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".pending-", dir=directory)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                if isinstance(value, bytes):
                    stream.write(value)
                else:
                    with Path(value).open("rb") as source:
                        shutil.copyfileobj(source, stream, 1024 * 1024)
                stream.flush()
                os.fsync(stream.fileno())
            digest = sha256_file(Path(temporary))
            destination = self._blob(digest)
            try:
                os.link(temporary, destination)
            except FileExistsError:
                if sha256_file(destination) != digest:
                    raise ValueError(f"Corrupted content-addressed blob {digest}")
            return {
                "path": destination.relative_to(self.root).as_posix(),
                "sha256": digest,
                "bytes": Path(temporary).stat().st_size,
            }
        finally:
            Path(temporary).unlink(missing_ok=True)

    @staticmethod
    def _identity(identity: Mapping[str, Any]) -> str:
        payload = dict(identity)
        artifact_id = payload.pop("artifact_id")
        if payload.get("schema_version") != SCHEMA_VERSION or _digest(payload) != artifact_id:
            raise ValueError("Invalid stage identity or unsupported schema")
        return _valid_id(artifact_id)

    def _verify_outputs(self, payload: Mapping[str, Any]) -> None:
        for name, item in payload["outputs"].items():
            _relative(name)
            path = self._blob(item["sha256"])
            if item["path"] != path.relative_to(self.root).as_posix():
                raise ValueError("Artifact reference is not canonical and relocatable")
            if path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]:
                raise ValueError(f"Corrupt artifact output: {name}")

    def verify(self, artifact_id: str, *, parents: bool = True) -> dict[str, Any]:
        """Verify the numerical key and every consumed byte, including parents."""
        active: set[str] = set()
        verified: dict[str, dict[str, Any]] = {}

        def visit(current: str) -> dict[str, Any]:
            if current in active:
                raise ValueError("Cyclic artifact dependency graph")
            if current in verified:
                return verified[current]
            active.add(current)
            payload = json.loads(self._manifest_path(current).read_text())
            if (
                self._identity(payload["identity"]) != current
                or payload.get("status") != "complete"
            ):
                raise ValueError("Incomplete or incompatible artifact manifest")
            self._verify_outputs(payload)
            if parents:
                for parent in payload["identity"]["parents"]:
                    visit(parent)
            active.remove(current)
            verified[current] = dict(payload)
            return dict(payload)

        return visit(artifact_id)

    def publish(
        self, identity: Mapping[str, Any], outputs: Mapping[str, Path | bytes]
    ) -> dict[str, Any]:
        """Commit a stage; changed bytes require a new numerical identity."""
        artifact_id = self._identity(identity)
        if not outputs:
            raise ValueError("A completed artifact must have durable outputs")
        for name in outputs:
            _relative(name)
        for parent in identity["parents"]:
            self.verify(parent)
        with self.lock(artifact_id):
            payload = {
                "identity": dict(identity),
                "status": "complete",
                "outputs": {name: self._put(value) for name, value in sorted(outputs.items())},
            }
            self._verify_outputs(payload)
            path = self._manifest_path(artifact_id)
            if path.exists():
                if self.verify(artifact_id) != payload:
                    raise ValueError("An immutable artifact already exists with different outputs")
            else:
                _publish_json(path, payload)
        return payload

    def restore(self, artifact_id: str, destination: Path) -> dict[str, Path]:
        """Copy verified bytes for consumers; shared blobs remain separate from outputs."""
        payload = self.verify(artifact_id)
        return self.restore_checkpoint(payload, destination)

    def restore_checkpoint(self, payload: Mapping[str, Any], destination: Path) -> dict[str, Path]:
        """Materialize verified checkpoint outputs without sharing writable inodes."""
        self._verify_outputs(payload)
        result = {}
        root = Path(destination).resolve()
        for name, output in payload["outputs"].items():
            path = root / _relative(name)
            if not path.resolve().is_relative_to(root):
                raise ValueError("Artifact destination escapes through a symlink")
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self._blob(output["sha256"]), path)
            result[name] = path
        return result

    def checkpoint(
        self,
        identity: Mapping[str, Any],
        *,
        completed_ids: Sequence[str],
        cursor: Any,
        outputs: Mapping[str, Path | bytes],
        state: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Persist exact completed items and opaque sampler/RNG/training state."""
        artifact_id = self._identity(identity)
        if len(set(completed_ids)) != len(completed_ids):
            raise ValueError("Checkpoint completed IDs must be unique")
        for name in outputs:
            _relative(name)
        with self.lock(artifact_id):
            directory = self.directory / "checkpoints" / artifact_id
            versions = sorted(directory.glob("[0-9]*.json"))
            sequence = int(versions[-1].stem) + 1 if versions else 1
            payload = {
                "identity": dict(identity),
                "sequence": sequence,
                "completed_ids": list(completed_ids),
                "cursor": cursor,
                "state": dict(state or {}),
                "outputs": {name: self._put(value) for name, value in sorted(outputs.items())},
            }
            self._verify_outputs(payload)
            _publish_json(directory / f"{sequence:08d}.json", payload)
        return payload

    def latest_checkpoint(self, artifact_id: str) -> dict[str, Any] | None:
        """Find the latest valid boundary, retaining predecessors after corruption."""
        directory = self.directory / "checkpoints" / _valid_id(artifact_id)
        for path in sorted(directory.glob("[0-9]*.json"), reverse=True):
            try:
                payload = json.loads(path.read_text())
                if self._identity(payload["identity"]) != artifact_id:
                    continue
                self._verify_outputs(payload)
                for parent in payload["identity"]["parents"]:
                    self.verify(parent)
                return dict(payload)
            except (OSError, ValueError, KeyError, TypeError):
                continue
        return None

    def import_artifact(self, source_root: Path, artifact_id: str) -> dict[str, Any]:
        """Import verified parents and bytes while leaving the old root untouched."""
        source = ArtifactStore(source_root)
        payload = source.verify(artifact_id)
        for parent in payload["identity"]["parents"]:
            self.import_artifact(source_root, parent)
        return self.publish(
            payload["identity"],
            {name: source._blob(output["sha256"]) for name, output in payload["outputs"].items()},
        )

    def import_checkpoint(self, source_root: Path, artifact_id: str) -> dict[str, Any] | None:
        """Import the latest verified boundary and parents into a fresh attempt root."""
        source = ArtifactStore(source_root)
        payload = source.latest_checkpoint(artifact_id)
        if payload is None:
            return None
        for parent in payload["identity"]["parents"]:
            self.import_artifact(source_root, parent)
        return self.checkpoint(
            payload["identity"],
            completed_ids=payload["completed_ids"],
            cursor=payload["cursor"],
            state=payload["state"],
            outputs={
                name: source._blob(output["sha256"]) for name, output in payload["outputs"].items()
            },
        )


def build_reuse_plan(
    store: ArtifactStore,
    expected: Mapping[str, Mapping[str, Any]],
    previous: Mapping[str, str],
    *,
    changed_stages: Iterable[str] = (),
    estimates: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Freeze checksum-backed reuse and dependency-derived recomputation decisions."""
    affected = set(invalidation_closure(changed_stages))
    entries = []
    for stage, identity in expected.items():
        artifact_id = ArtifactStore._identity(identity)
        reason = "verified numerical identity and all output/parent checksums"
        action = "reuse"
        if stage in affected:
            action, reason = "recompute", "declared repair or invalidated semantic descendant"
        else:
            try:
                store.verify(artifact_id)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                action, reason = "recompute", f"unverified artifact: {exc}"
        entries.append(
            {"stage": stage, "artifact_id": artifact_id, "action": action, "reason": reason}
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "stages": entries,
        "invalidated_descendants": sorted(affected),
        "estimated_incremental_cost": sum(
            float((estimates or {}).get(item["stage"], 0))
            for item in entries
            if item["action"] == "recompute"
        ),
        "estimate_available": estimates is not None,
        "plan_id": _digest(entries),
    }


def create_attempt(
    root: Path,
    *,
    design_id: str,
    provenance: Mapping[str, Any],
    parent_attempt: str | None = None,
    repair_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Record immutable attempt provenance independently from numerical identity."""
    attempt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "attempt_id": uuid.uuid4().hex,
        "design_id": design_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "root": str(Path(root).resolve()),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "parent_attempt": parent_attempt,
        "repair_record": dict(repair_record) if repair_record is not None else None,
        "provenance": dict(provenance),
    }
    _publish_json(Path(root) / "attempts" / attempt["attempt_id"] / "attempt.json", attempt)
    return attempt


def finish_attempt(
    root: Path,
    attempt_id: str,
    *,
    status: str,
    continuation: str,
    summary: Mapping[str, Any],
) -> None:
    """Append a final outcome with the exact continuation command; never overwrite it."""
    if status not in {"complete", "interrupted", "failed"}:
        raise ValueError("Unknown attempt status")
    if len(_relative(attempt_id).parts) != 1:
        raise ValueError("Attempt ID must be one path component")
    if not (Path(root) / "attempts" / attempt_id / "attempt.json").is_file():
        raise ValueError("Cannot finish an unknown attempt")
    _publish_json(
        Path(root) / "attempts" / attempt_id / "outcome.json",
        {"status": status, "continuation": continuation, "summary": dict(summary)},
    )


def validate_current_result_set(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_cells: Iterable[str],
    design_id: str,
    selection_id: str | None = None,
    invalidated_artifacts: Iterable[str] = (),
    store: ArtifactStore | None = None,
) -> None:
    """Reject duplicate/missing cells, stale selections and mixed repair revisions.

    Callers first select approved repair lineage. Metrics never choose a revision.
    Each row supplies cell_id, artifact_id, design_id, semantics_id and selection_id.
    """
    ids = [row["cell_id"] for row in records]
    if len(ids) != len(set(ids)) or set(ids) != set(expected_cells):
        raise ValueError(
            "Current result set has duplicate cells or missing/extra cells and controls"
        )
    semantics = {row.get("semantics_id") for row in records}
    if len(semantics) != 1 or None in semantics:
        raise ValueError(
            "Current result set mixes incompatible or unidentified execution revisions"
        )
    invalidated = set(invalidated_artifacts)
    for row in records:
        if row.get("design_id") != design_id or row.get("selection_id") != selection_id:
            raise ValueError("Current result set has a changed design or stale selection record")
        if row["artifact_id"] in invalidated:
            raise ValueError("Current result set includes an invalidated artifact")
        if store is not None:
            store.verify(row["artifact_id"])


def legacy_import_record(source_root: Path) -> dict[str, Any]:
    """Inventory legacy provenance without assigning invented current identities."""
    root = Path(source_root).resolve()
    paths = sorted(
        path
        for path in root.rglob("*.json")
        if path.name in {"experiment_manifest.json", "manifest.json", "provenance.json"}
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "source": str(root),
        "source_hashes": {path.relative_to(root).as_posix(): sha256_file(path) for path in paths},
        "original_manifests": {
            path.relative_to(root).as_posix(): json.loads(path.read_text()) for path in paths
        },
        "reused": [],
        "reason": "Legacy stage semantics, role/model locks and output checksums require verification",
    }
