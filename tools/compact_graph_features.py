"""Compact canonical raw training shards without decoding their repeated graph manifests.

Run only with the shard's writer stopped. Original files are replaced only with --apply;
otherwise verified compact candidates and receipts are left for inspection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from exact.impl.models.graph_head import compact_graph_fingerprints
from exact.utils.fitted_artifacts import fingerprint, freeze_json

_CHUNK = 4 * 1024 * 1024
_MAX_MANIFEST = 256 * 1024 * 1024
_MAX_ROW = 16 * 1024 * 1024
_MANIFEST_KEY = b'\n        "graph_fingerprints": '


class _Reader:
    """Hash original bytes once while consuming bounded canonical JSON regions."""

    def __init__(self, stream):
        self.stream, self.buffer = stream, b""
        self.digest = hashlib.sha256()

    def _fill(self):
        block = self.stream.read(_CHUNK)
        self.digest.update(block)
        self.buffer += block
        return bool(block)

    def take(self, count):
        while len(self.buffer) < count:
            if not self._fill():
                raise ValueError("Truncated raw training shard")
        value, self.buffer = self.buffer[:count], self.buffer[count:]
        return value

    def expect(self, value):
        if self.take(len(value)) != value:
            raise ValueError("Expected canonical freeze_json training shard layout")

    def until(self, marker, limit):
        parts, size = [], 0
        while True:
            index = self.buffer.find(marker)
            if index >= 0:
                part = self.take(index + len(marker))
                if size + len(part) > limit:
                    raise ValueError("Raw training shard region exceeds safe size")
                return b"".join([*parts, part])
            count = max(0, len(self.buffer) - len(marker) + 1)
            if count:
                part = self.take(count)
                parts.append(part)
                size += len(part)
                if size > limit:
                    raise ValueError("Raw training shard region exceeds safe size")
            if not self._fill():
                raise ValueError("Truncated or noncanonical raw training shard")

    def rest(self, limit):
        while self._fill():
            if len(self.buffer) > limit:
                raise ValueError("Training shard metadata exceeds safe size")
        return self.take(len(self.buffer))


def _signature(path):
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _sync_directory(path):
    descriptor = os.open(path, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _receipt(path, payload):
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _sync_directory(path.parent)


def _preserved(row):
    """All source/target IDs, labels, scores and feature numbers remain identical."""
    retained = dict(row)
    retained["graph_features"] = dict(row["graph_features"])
    del retained["graph_features"]["graph_fingerprints"]
    return fingerprint(retained)


class GraphManifest:
    """The full graph provenance is decoded and persisted once for all shards."""

    def __init__(self, raw, directory):
        full = json.loads(raw)
        self.compact = compact_graph_fingerprints(full)
        if self.compact == full:
            raise ValueError("Shard has no repeated hierarchy-removal manifest to compact")
        self.raw = raw
        self.bindings = []
        for side, manifest in full.items():
            identity = self.compact[side].get("manifest_sha256")
            if identity is None:
                continue
            if fingerprint(manifest) != identity:
                raise ValueError("Compact graph manifest identity mismatch")
            path = directory / (identity + ".json")
            freeze_json(path, manifest)
            self.bindings.append(
                {
                    "side": side,
                    "path": str(path.resolve()),
                    "manifest_sha256": identity,
                    "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        self.encoded = json.dumps(
            self.compact, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()


def compact_shard(path, *, manifest_directory, receipt_path, apply=False, manifest=None):
    """One original-file read; candidate validation precedes atomic replacement.

    Return (receipt, manifest), so subsequent shards reuse the validated full manifest.
    A crash after replacement leaves a prepared receipt containing both content hashes.
    """
    path, receipt_path = Path(path), Path(receipt_path).resolve()
    directory = Path(manifest_directory).resolve()
    if path.is_symlink() or path.suffix != ".json":
        raise ValueError("Only completed JSON shards can be compacted")
    path = path.resolve()
    if receipt_path.exists():
        raise ValueError("Compaction receipt already exists; inspect it before retrying")
    signature = _signature(path)
    candidate = path.with_suffix(".json.compact.partial")
    row_count, source_groups = 0, set()
    preserved = hashlib.sha256()
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation never overwrites an interrupted or independently prepared candidate.
    with candidate.open("xb") as output:
        try:
            with path.open("rb") as stream:
                reader = _Reader(stream)
                reader.expect(b'{\n  "rows": [\n')
                output.write(b'{"rows":[')
                while True:
                    prefix = reader.until(_MANIFEST_KEY, _MAX_ROW)
                    if manifest is None:
                        raw = reader.until(b"\n        }", _MAX_MANIFEST)
                        manifest = GraphManifest(raw, directory)
                    elif reader.take(len(manifest.raw)) != manifest.raw:
                        raise ValueError("Training rows mix graph manifests or noncanonical bytes")
                    suffix = reader.until(b"\n    }", _MAX_ROW)
                    row = json.loads(prefix + manifest.encoded + suffix)
                    if row["graph_features"]["graph_fingerprints"] != manifest.compact:
                        raise ValueError("Unexpected graph fingerprint location")
                    preserved.update(_preserved(row).encode())
                    source_groups.add(row["Src"])
                    if row_count:
                        output.write(b",")
                    output.write(
                        json.dumps(
                            row, sort_keys=True, separators=(",", ":"), allow_nan=False
                        ).encode()
                    )
                    row_count += 1
                    delimiter = reader.take(2)
                    if delimiter == b",\n":
                        continue
                    if delimiter != b"\n ":
                        raise ValueError("Unexpected training row delimiter")
                    reader.expect(b' ],\n  "source_ids": ')
                    tail = reader.rest(_MAX_ROW)
                    metadata = json.loads(b'{"source_ids":' + tail)
                    source_ids = metadata["source_ids"]
                    if set(metadata) != {"source_ids"} or set(source_ids) != source_groups:
                        raise ValueError("Training shard source membership mismatch")
                    if path.stem != fingerprint(source_ids):
                        raise ValueError("Training shard filename does not match its source groups")
                    output.write(b'],"source_ids":')
                    output.write(json.dumps(source_ids, separators=(",", ":")).encode())
                    output.write(b"}\n")
                    break
            output.flush()
            os.fsync(output.fileno())
        except BaseException:
            candidate.unlink(missing_ok=True)
            raise
    # This file is small after deduplication; verify every retained scientific value.
    candidate_bytes = candidate.read_bytes()
    verified = json.loads(candidate_bytes)
    verification = hashlib.sha256()
    for row in verified["rows"]:
        if row["graph_features"]["graph_fingerprints"] != manifest.compact:
            raise ValueError("Candidate graph provenance mismatch")
        verification.update(_preserved(row).encode())
    if (
        len(verified["rows"]) != row_count
        or verified["source_ids"] != source_ids
        or verification.hexdigest() != preserved.hexdigest()
    ):
        raise ValueError("Candidate changed training membership or scientific values")
    if _signature(path) != signature:
        raise ValueError("Original shard changed during compaction; writer must be stopped")
    receipt = {
        "schema_version": 1,
        "status": "prepared",
        "path": str(path),
        "candidate": str(candidate),
        "original_sha256": reader.digest.hexdigest(),
        "compact_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
        "original_bytes": signature[2],
        "compact_bytes": len(candidate_bytes),
        "row_count": row_count,
        "source_ids": source_ids,
        "preserved_rows_sha256": preserved.hexdigest(),
        "graph_manifests": manifest.bindings,
    }
    _receipt(receipt_path, receipt)
    if apply:
        if _signature(path) != signature:
            raise ValueError("Original shard changed before replacement")
        os.replace(candidate, path)
        _sync_directory(path.parent)
        receipt["status"] = "compacted"
        _receipt(receipt_path, receipt)
    return receipt, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shards", nargs="+", type=Path)
    parser.add_argument("--manifest-directory", required=True, type=Path)
    parser.add_argument("--receipt-directory", required=True, type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    manifest = None
    for shard in args.shards:
        receipt, manifest = compact_shard(
            shard,
            manifest_directory=args.manifest_directory,
            receipt_path=args.receipt_directory / shard.name,
            apply=args.apply,
            manifest=manifest,
        )
        print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
