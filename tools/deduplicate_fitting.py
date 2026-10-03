#!/usr/bin/env python3
"""Deduplicate completed immutable fitting JSON while preserving every artifact path."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import stat
import sys
import uuid
from contextlib import nullcontext
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exact.experiments.recovery import ArtifactStore  # noqa: E402


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _read_binding(binding):
    path = Path(binding["path"])
    if _sha(path) != binding["sha256"]:
        raise ValueError("Completed artifact binding changed: " + str(path))
    return path, json.loads(path.read_text())


def _signature(path):
    value = path.stat()
    return {
        "device": value.st_dev,
        "inode": value.st_ino,
        "bytes": value.st_size,
        "mtime_ns": value.st_mtime_ns,
        "blocks_bytes": value.st_blocks * 512,
    }


def _plan(root):
    completion_path = root / "completion.json"
    result = {
        "schema_version": 1,
        "operation": "deduplicate_completed_fitting",
        "root": str(root),
        "status": "skipped",
        "files": [],
    }
    if not completion_path.is_file():
        return {**result, "reason": "No completed comparison receipt"}
    completed = json.loads(completion_path.read_text())
    if completed.get("status") != "complete" or completed.get("exit_code") != 0:
        return {**result, "reason": "Incomplete comparison retains all working files"}
    selection = completed.get("selection") or {}
    if not selection.get("path"):
        return {**result, "reason": "No completed selection binding"}
    selection_path = Path(selection["path"]).resolve()
    if not selection_path.is_relative_to(root):
        return {**result, "reason": "Finalizer references another comparison's runtime"}
    _read_binding(selection)
    if selection_path.parent.name != "screen":
        return {**result, "reason": "Only completed screen runtimes are eligible"}
    runtime = selection_path.parent.parent
    cells = []
    for binding in completed.get("manifests", []):
        path = Path(binding["path"]).resolve()
        if not path.is_relative_to(runtime / "screen/runs"):
            return {**result, "reason": "Comparison owns no independent materialized outputs"}
        _, cell = _read_binding(binding)
        if cell.get("status") != "complete" or cell.get("extraction_complete") is not True:
            return {**result, "reason": "Every cell must be complete before deduplication"}
        cells.append((path.parent, cell))
    if not cells:
        return {**result, "reason": "No completed cell manifests"}
    store = ArtifactStore(runtime)
    checked = {}
    files = []
    for directory, cell in cells:
        identifier = cell["recovery"]["artifacts"]["extraction"]
        stage_path = store._manifest_path(identifier)
        stage = json.loads(stage_path.read_text())
        if stage.get("status") != "complete" or store._identity(stage["identity"]) != identifier:
            raise ValueError("Extraction artifact is not complete and immutable")
        for name, output in stage["outputs"].items():
            relative = Path(name)
            if relative.parts[0] != "fitting" or relative.suffix != ".json":
                continue
            materialized = directory / relative
            blob = store._blob(output["sha256"])
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or materialized.is_symlink()
                or blob.is_symlink()
                or not materialized.resolve().is_relative_to(directory)
                or not blob.resolve().is_relative_to(runtime / "artifacts/blobs")
                or output["path"] != blob.relative_to(runtime).as_posix()
            ):
                raise ValueError("Fitting output escapes the completed runtime")
            before, canonical = _signature(materialized), _signature(blob)
            if before["bytes"] != output["bytes"] or canonical["bytes"] != output["bytes"]:
                raise ValueError("Fitting output size changed: " + str(materialized))
            if blob not in checked:
                if _sha(blob) != output["sha256"] or _signature(blob) != canonical:
                    raise ValueError("Canonical fitting blob changed: " + str(blob))
                checked[blob] = canonical
            elif checked[blob] != canonical:
                raise ValueError("Canonical fitting blob changed during inspection")
            linked = (before["device"], before["inode"]) == (
                canonical["device"],
                canonical["inode"],
            )
            if not linked and (
                _sha(materialized) != output["sha256"] or _signature(materialized) != before
            ):
                raise ValueError("Materialized fitting bytes changed: " + str(materialized))
            if before["device"] != canonical["device"]:
                action = "skip_other_filesystem"
            elif linked:
                action = "already_linked"
            elif materialized.stat().st_nlink != 1:
                action = "skip_shared_materialized_inode"
            elif blob.stat().st_nlink != 1 and stat.S_IMODE(blob.stat().st_mode) & 0o222:
                action = "skip_shared_writable_blob"
            else:
                action = "link"
            files.append(
                {
                    "path": str(materialized),
                    "blob": str(blob),
                    "sha256": output["sha256"],
                    "before": before,
                    "canonical": canonical,
                    "action": action,
                }
            )
    return {
        **result,
        "status": "planned",
        "runtime": str(runtime),
        "completion_sha256": _sha(completion_path),
        "files": files,
        "planned_reclaim_bytes": sum(
            row["before"]["blocks_bytes"] for row in files if row["action"] == "link"
        ),
    }


def deduplicate_completed(root: Path, *, apply=False):
    """Only completed comparisons: share verified read-only fitting JSON, never live files.

    Fitting writers use immutable freeze_json/atomic replacement. Completed workers
    reject relaunch, and fresh recovery destinations receive independent copies.
    """
    root = Path(root).resolve()
    if "experiments-v2" not in root.parts:
        raise ValueError("Deduplication requires an experiments-v2 comparison directory")
    lock = (root / "worker.lock").open("a") if apply else nullcontext()
    with lock as handle:
        if apply:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = _plan(root)
        result.update(applied=apply, reclaimed_bytes=0)
        if result["status"] != "planned" or not apply:
            return result
        for row in result["files"]:
            if row["action"] != "link":
                continue
            path, blob = Path(row["path"]), Path(row["blob"])
            temporary = path.with_name(".dedup-" + uuid.uuid4().hex)
            try:
                if _signature(path) != row["before"] or _signature(blob) != row["canonical"]:
                    raise ValueError("Fitting file changed after validation: " + str(path))
                # Remove write permissions without granting any new readers.
                blob.chmod(stat.S_IMODE(blob.stat().st_mode) & ~0o222)
                os.link(blob, temporary)
                os.replace(temporary, path)
                descriptor = os.open(path.parent, os.O_DIRECTORY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
                row["after"] = _signature(path)
                row["action"] = "linked"
                result["reclaimed_bytes"] += row["before"]["blocks_bytes"]
            except Exception as error:
                result.update(status="failed", error=f"{type(error).__name__}: {error}")
                row["action"] = "failed"
                break
            finally:
                temporary.unlink(missing_ok=True)
        else:
            result["status"] = "complete"
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    receipt = args.receipt.resolve()
    if receipt.is_relative_to(args.root.resolve() / "runtime"):
        parser.error("Deduplication receipts must remain outside scientific runtime artifacts")
    if receipt.exists():
        previous = json.loads(receipt.read_text())
        if previous.get("operation") != "deduplicate_completed_fitting":
            parser.error("Refusing to overwrite a non-deduplication receipt")
    try:
        result = deduplicate_completed(args.root, apply=args.apply)
    except Exception as error:
        result = {
            "operation": "deduplicate_completed_fitting",
            "status": "failed",
            "root": str(args.root),
            "error": f"{type(error).__name__}: {error}",
        }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.receipt.with_suffix(args.receipt.suffix + ".pending")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.receipt)
    print(json.dumps({key: value for key, value in result.items() if key != "files"}))
    raise SystemExit(1 if result["status"] == "failed" else 0)


if __name__ == "__main__":
    main()
