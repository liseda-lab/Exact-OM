"""Crash-safe, indexed explanation storage for layout-v2 runs."""

from __future__ import annotations

import copy
import csv
import hashlib
import heapq
import itertools
import json
import os
import re
import shutil
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional, TextIO, cast

import zstandard as zstd

from .layout import LAYOUT_VERSION, RunLayout


EXPLANATION_SCHEMA_VERSION = 1
INDEX_SCHEMA_VERSION = 1
DEFAULT_SHARD_MB = 32.0
DEFAULT_HASH_BUCKETS = 64
_SEQUENCE_FIELD = "_exact_store_sequence"
_STORE_SHARD_PATTERN = re.compile(r"^(?:g[0-9a-f]+-)?\d{5,}\.jsonl(?:\.zst)?$")
_OVERLAY_PATTERN = re.compile(r"^overlay-[0-9a-f-]+\.jsonl(?:\.zst)?$")


def _json_default(value: Any) -> str:
    return str(value)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _source_iri(record: Mapping[str, Any]) -> str:
    for key in ("src_iri", "Src", "source_iri", "source"):
        value = record.get(key)
        if value is not None and str(value):
            return str(value)
    raise ValueError("Explanation record is missing a source IRI")


def _target_iri(record: Mapping[str, Any]) -> Optional[str]:
    for key in ("tgt_iri", "Tgt", "target_iri", "target"):
        value = record.get(key)
        if value is not None and str(value):
            return str(value)
    return None


def _pair(record: Mapping[str, Any]) -> Optional[tuple[str, str]]:
    try:
        source = _source_iri(record)
    except ValueError:
        return None
    target = _target_iri(record)
    return (source, target) if target is not None else None


def _without_internal_fields(record: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key != _SEQUENCE_FIELD}


class ExplanationStore:
    """Append explanation records and retrieve one source from one shard.

    Every append writes complete zstd frames before atomically advancing
    ``index.json``. On reopen, uncommitted tails are truncated to the byte
    offsets recorded in the index. A source is permanently assigned to one
    hash-bucket shard, so :meth:`get` performs exactly one shard read.
    """

    def __init__(
        self,
        directory: Path,
        *,
        run_id: Optional[str] = None,
        shard_mb: float = DEFAULT_SHARD_MB,
        hash_buckets: int = DEFAULT_HASH_BUCKETS,
        compression: str = "zstd",
    ) -> None:
        self.directory = Path(directory).expanduser().resolve()
        self.shards_dir = self.directory / "shards"
        self.index_path = self.directory / "index.json"
        self.overlays_dir = self.directory / "overlays"
        self.run_id = run_id
        self.shard_reads = 0
        self._overlay_cache: Optional[dict[tuple[str, str], dict[str, Any]]] = None
        self.directory.mkdir(parents=True, exist_ok=True)
        self.shards_dir.mkdir(parents=True, exist_ok=True)
        if self.index_path.is_file():
            self._index = self._load_index()
        else:
            if compression not in {"zstd", "none"}:
                raise ValueError(
                    f"Unsupported explanation compression: {compression!r}"
                )
            self._index = {
                "schema_version": INDEX_SCHEMA_VERSION,
                "explanation_schema_version": EXPLANATION_SCHEMA_VERSION,
                "compression": compression,
                "shard_bytes": max(1, int(float(shard_mb) * 1024 * 1024)),
                "hash_buckets": max(1, int(hash_buckets)),
                "next_sequence": 0,
                "next_shard": 0,
                "total_records": 0,
                "shards": {},
                "sources": {},
                "overlays": [],
            }
            self._write_index(self._index)
        self._recover_uncommitted_files()

    @classmethod
    def open(
        cls,
        run_dir: Path,
        *,
        run_id: Optional[str] = None,
        shard_mb: float = DEFAULT_SHARD_MB,
    ) -> "ExplanationStore":
        """Open or create the explanation store for a v2 run."""

        layout = RunLayout.open(run_dir)
        if layout.version != LAYOUT_VERSION:
            if layout.manifest_path.exists() or Path(run_dir).exists():
                raise ValueError(
                    "ExplanationStore is only available for layout-v2 runs"
                )
            layout = RunLayout.create(run_dir)
        layout.ensure_directories()
        return cls(layout.explanations_dir, run_id=run_id, shard_mb=shard_mb)

    @classmethod
    def create(
        cls,
        run_dir: Path,
        *,
        run_id: Optional[str] = None,
        shard_mb: float = DEFAULT_SHARD_MB,
    ) -> "ExplanationStore":
        layout = RunLayout.create(run_dir)
        return cls(layout.explanations_dir, run_id=run_id, shard_mb=shard_mb)

    @property
    def record_count(self) -> int:
        return int(self._index.get("total_records", 0))

    @property
    def source_count(self) -> int:
        return len(self._index.get("sources") or {})

    @property
    def overlay_count(self) -> int:
        return sum(
            int(entry.get("records", 0)) for entry in self._index.get("overlays") or []
        )

    @property
    def stored_bytes(self) -> int:
        return self._stored_bytes()

    def _load_index(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Invalid explanation index at {self.index_path}: {exc}"
            ) from exc
        if int(payload.get("schema_version", -1)) != INDEX_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported explanation index schema: {payload.get('schema_version')!r}"
            )
        if payload.get("compression") not in {"zstd", "none"}:
            raise ValueError("Explanation index declares an unsupported compression")
        if not isinstance(payload.get("shards"), dict) or not isinstance(
            payload.get("sources"), dict
        ):
            raise ValueError("Explanation index must contain shard and source mappings")
        return cast(dict[str, Any], payload)

    def _write_index(self, payload: Mapping[str, Any]) -> None:
        _atomic_json(self.index_path, payload)

    def _resolve_store_path(self, relative: str) -> Path:
        path = (self.directory / relative).resolve()
        try:
            path.relative_to(self.directory)
        except ValueError as exc:
            raise ValueError(
                f"Explanation index path escapes its store: {relative!r}"
            ) from exc
        return path

    def _recover_uncommitted_files(self) -> None:
        referenced: set[Path] = set()
        for shard in (self._index.get("shards") or {}).values():
            path = self._resolve_store_path(str(shard["path"]))
            referenced.add(path)
            committed_bytes = int(shard.get("bytes", 0))
            try:
                actual_bytes = path.stat().st_size
            except FileNotFoundError as exc:
                if committed_bytes == 0:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.touch()
                    continue
                raise ValueError(f"Explanation shard is missing: {path}") from exc
            if actual_bytes < committed_bytes:
                raise ValueError(
                    f"Explanation shard {path} is shorter than its committed index offset"
                )
            if actual_bytes > committed_bytes:
                with path.open("r+b") as stream:
                    stream.truncate(committed_bytes)
        for path in self.shards_dir.iterdir():
            if (
                path.is_file()
                and path not in referenced
                and _STORE_SHARD_PATTERN.match(path.name)
            ):
                path.unlink()

        overlay_paths = {
            self._resolve_store_path(str(entry["path"]))
            for entry in self._index.get("overlays") or []
        }
        if self.overlays_dir.is_dir():
            for path in self.overlays_dir.iterdir():
                if (
                    path.is_file()
                    and path not in overlay_paths
                    and _OVERLAY_PATTERN.match(path.name)
                ):
                    path.unlink()

    def _encode_lines(self, records: Iterable[Mapping[str, Any]]) -> bytes:
        raw = b"".join(
            (
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    default=_json_default,
                )
                + "\n"
            ).encode("utf-8")
            for record in records
        )
        if self._index.get("compression") == "zstd":
            return zstd.ZstdCompressor(level=3).compress(raw)
        return raw

    def _open_text(self, path: Path, compression: Optional[str] = None) -> TextIO:
        resolved = compression or str(self._index.get("compression", "zstd"))
        if resolved == "zstd":
            return cast(TextIO, zstd.open(path, mode="rt", encoding="utf-8"))
        return path.open("r", encoding="utf-8")

    @staticmethod
    def _bucket(source: str, count: int) -> int:
        digest = hashlib.sha256(source.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") % count

    def _new_shard(
        self, index: dict[str, Any], bucket: int
    ) -> tuple[str, dict[str, Any]]:
        number = int(index.get("next_shard", 0))
        shard_id = f"{number:05d}"
        suffix = ".jsonl.zst" if index.get("compression") == "zstd" else ".jsonl"
        metadata = {
            "path": f"shards/{shard_id}{suffix}",
            "bucket": int(bucket),
            "bytes": 0,
            "records": 0,
            "sources": 0,
        }
        index["next_shard"] = number + 1
        index.setdefault("shards", {})[shard_id] = metadata
        return shard_id, metadata

    def _select_shard(
        self,
        index: dict[str, Any],
        source: str,
        encoded_bytes: int,
    ) -> tuple[str, dict[str, Any]]:
        source_entry = (index.get("sources") or {}).get(source)
        if source_entry:
            shard_id = str(source_entry["shard"])
            return shard_id, index["shards"][shard_id]
        bucket = self._bucket(
            source, int(index.get("hash_buckets", DEFAULT_HASH_BUCKETS))
        )
        bucket_shards = [
            (shard_id, shard)
            for shard_id, shard in index.get("shards", {}).items()
            if int(shard.get("bucket", -1)) == bucket
        ]
        if bucket_shards:
            shard_id, shard = max(bucket_shards, key=lambda item: int(item[0]))
            if int(shard.get("bytes", 0)) + encoded_bytes <= int(index["shard_bytes"]):
                return shard_id, shard
        return self._new_shard(index, bucket)

    @staticmethod
    def _append_bytes(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("ab") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())

    def append(
        self,
        records: Iterable[Mapping[str, Any]],
        *,
        run_id: Optional[str] = None,
    ) -> int:
        """Append records and commit their index offsets as one transaction."""

        materialized = [dict(record) for record in records]
        if not materialized:
            return 0
        working = copy.deepcopy(self._index)
        next_sequence = int(working.get("next_sequence", 0))
        resolved_run_id = run_id or self.run_id
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in materialized:
            source = _source_iri(record)
            if resolved_run_id is not None:
                record.setdefault("run_id", str(resolved_run_id))
            record.setdefault("explanation_schema_version", EXPLANATION_SCHEMA_VERSION)
            record[_SEQUENCE_FIELD] = next_sequence
            next_sequence += 1
            grouped[source].append(record)

        baselines: dict[Path, int] = {}
        created: set[Path] = set()
        try:
            for source, source_records in grouped.items():
                encoded = self._encode_lines(source_records)
                shard_id, shard = self._select_shard(working, source, len(encoded))
                path = self._resolve_store_path(str(shard["path"]))
                if path not in baselines:
                    baselines[path] = path.stat().st_size if path.exists() else 0
                    if not path.exists():
                        created.add(path)
                self._append_bytes(path, encoded)
                shard["bytes"] = int(shard.get("bytes", 0)) + len(encoded)
                shard["records"] = int(shard.get("records", 0)) + len(source_records)
                source_entry = working.setdefault("sources", {}).get(source)
                if source_entry is None:
                    working["sources"][source] = {
                        "shard": shard_id,
                        "records": len(source_records),
                    }
                    shard["sources"] = int(shard.get("sources", 0)) + 1
                else:
                    source_entry["records"] = int(source_entry.get("records", 0)) + len(
                        source_records
                    )
            working["next_sequence"] = next_sequence
            working["total_records"] = int(working.get("total_records", 0)) + len(
                materialized
            )
            self._write_index(working)
        except BaseException:
            for path, offset in baselines.items():
                if path.exists():
                    with path.open("r+b") as stream:
                        stream.truncate(offset)
                if path in created and offset == 0 and path.exists():
                    path.unlink()
            raise
        self._index = working
        return len(materialized)

    def _iter_shard_records(self, shard: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
        path = self._resolve_store_path(str(shard["path"]))
        self.shard_reads += 1
        with self._open_text(path) as stream:
            for line in stream:
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Corrupt explanation record in {path}: {exc}"
                    ) from exc
                if not isinstance(payload, dict):
                    raise ValueError(f"Non-object explanation record in {path}")
                yield payload

    def _iter_overlay_records(self) -> Iterator[dict[str, Any]]:
        for entry in self._index.get("overlays") or []:
            path = self._resolve_store_path(str(entry["path"]))
            with self._open_text(
                path, compression=str(entry.get("compression", "zstd"))
            ) as stream:
                for line in stream:
                    if line.strip():
                        payload = json.loads(line)
                        if isinstance(payload, dict):
                            yield payload

    def _overlay_lookup(self) -> dict[tuple[str, str], dict[str, Any]]:
        if self._overlay_cache is None:
            lookup: dict[tuple[str, str], dict[str, Any]] = {}
            for overlay in self._iter_overlay_records():
                key = _pair(overlay)
                if key is not None:
                    current = lookup.get(key, {})
                    lookup[key] = self._merge_overlay(current, overlay)
            self._overlay_cache = lookup
        return self._overlay_cache

    @staticmethod
    def _merge_overlay(
        record: Mapping[str, Any], overlay: Mapping[str, Any]
    ) -> dict[str, Any]:
        merged = dict(record)
        identifier_keys = {"Src", "Tgt", "src_iri", "tgt_iri", "source", "target"}
        for key, value in overlay.items():
            if key in identifier_keys:
                continue
            if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
                section = dict(merged[key])
                section.update(value)
                merged[key] = section
            else:
                merged[key] = value
        return merged

    def get(self, src_iri: str) -> list[dict[str, Any]]:
        """Read all records for ``src_iri`` by decompressing one shard."""

        entry = (self._index.get("sources") or {}).get(str(src_iri))
        if entry is None:
            return []
        shard = self._index["shards"][str(entry["shard"])]
        overlays = self._overlay_lookup()
        records: list[dict[str, Any]] = []
        for record in self._iter_shard_records(shard):
            if _source_iri(record) != str(src_iri):
                continue
            key = _pair(record)
            if key is not None and key in overlays:
                record = self._merge_overlay(record, overlays[key])
            records.append(record)
        records.sort(key=lambda record: int(record.get(_SEQUENCE_FIELD, 0)))
        return [_without_internal_fields(record) for record in records]

    def iter_all(self) -> Iterator[dict[str, Any]]:
        """Stream all records in append order using a shard-wise merge."""

        overlays = self._overlay_lookup()
        heap: list[tuple[int, int, dict[str, Any], Iterator[dict[str, Any]]]] = []
        counter = 0
        for shard_id in sorted(self._index.get("shards") or {}):
            iterator = self._iter_shard_records(self._index["shards"][shard_id])
            try:
                record = next(iterator)
            except StopIteration:
                continue
            heapq.heappush(
                heap,
                (int(record.get(_SEQUENCE_FIELD, 0)), counter, record, iterator),
            )
            counter += 1
        while heap:
            _, _, record, iterator = heapq.heappop(heap)
            key = _pair(record)
            if key is not None and key in overlays:
                record = self._merge_overlay(record, overlays[key])
            yield _without_internal_fields(record)
            try:
                following = next(iterator)
            except StopIteration:
                continue
            heapq.heappush(
                heap,
                (int(following.get(_SEQUENCE_FIELD, 0)), counter, following, iterator),
            )
            counter += 1

    def append_overlay(self, records: Iterable[Mapping[str, Any]]) -> int:
        """Write crash-safe transient corrections for existing pairs."""

        materialized = [dict(record) for record in records]
        if not materialized:
            return 0
        for record in materialized:
            if _pair(record) is None:
                raise ValueError("Overlay record must identify both source and target")
        self.overlays_dir.mkdir(parents=True, exist_ok=True)
        identifier = uuid.uuid4().hex
        suffix = ".jsonl.zst" if self._index.get("compression") == "zstd" else ".jsonl"
        relative = f"overlays/overlay-{identifier}{suffix}"
        path = self._resolve_store_path(relative)
        temporary = path.with_name(f".{path.name}.tmp")
        encoded = self._encode_lines(materialized)
        with temporary.open("wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        working = copy.deepcopy(self._index)
        working.setdefault("overlays", []).append(
            {
                "path": relative,
                "records": len(materialized),
                "bytes": len(encoded),
                "compression": self._index.get("compression", "zstd"),
            }
        )
        try:
            self._write_index(working)
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        self._index = working
        self._overlay_cache = None
        return len(materialized)

    def compact(self) -> dict[str, int]:
        """Merge overlays into fresh shards and atomically switch the index."""

        return self._replace_records(list(self.iter_all()))

    def truncate(self, record_count: int) -> dict[str, int]:
        """Roll the store back to a checkpoint's committed record boundary.

        The explanation index is committed independently from the checkpoint
        manifest.  A process can therefore stop after appending a batch but
        before advancing its checkpoint.  Resume calls this method to discard
        that uncheckpointed suffix before inference continues.
        """

        keep = int(record_count)
        if keep < 0 or keep > self.record_count:
            raise ValueError(
                f"Cannot truncate {self.record_count} explanation records to {keep}"
            )
        if keep == self.record_count:
            return {
                "before_bytes": self._stored_bytes(),
                "after_bytes": self._stored_bytes(),
                "records": keep,
            }
        records = list(itertools.islice(self.iter_all(), keep))
        return self._replace_records(records)

    def clear(self) -> dict[str, int]:
        """Atomically replace all store-owned records with an empty index."""

        return self._replace_records([])

    def _stored_bytes(self) -> int:
        return sum(
            self._resolve_store_path(str(shard["path"])).stat().st_size
            for shard in (self._index.get("shards") or {}).values()
        ) + sum(
            self._resolve_store_path(str(entry["path"])).stat().st_size
            for entry in self._index.get("overlays") or []
        )

    def _replace_records(self, records: list[dict[str, Any]]) -> dict[str, int]:
        """Atomically install ``records`` as the complete store contents."""

        before = self._stored_bytes()
        token = uuid.uuid4().hex[:12]
        temporary_dir = self.directory / f".compact-{token}"
        temporary_store = ExplanationStore(
            temporary_dir,
            shard_mb=float(self._index.get("shard_bytes", 1)) / (1024 * 1024),
            hash_buckets=int(self._index.get("hash_buckets", DEFAULT_HASH_BUCKETS)),
            compression=str(self._index.get("compression", "zstd")),
        )
        temporary_store.append(records)
        replacement = copy.deepcopy(temporary_store._index)
        replacement["overlays"] = []
        moved: list[Path] = []
        old_paths = [
            self._resolve_store_path(str(shard["path"]))
            for shard in (self._index.get("shards") or {}).values()
        ]
        old_overlay_paths = [
            self._resolve_store_path(str(entry["path"]))
            for entry in self._index.get("overlays") or []
        ]
        try:
            for shard in replacement.get("shards", {}).values():
                source = temporary_store._resolve_store_path(str(shard["path"]))
                destination = self.shards_dir / f"g{token}-{source.name}"
                os.replace(source, destination)
                moved.append(destination)
                shard["path"] = f"shards/{destination.name}"
            self._write_index(replacement)
        except BaseException:
            for path in moved:
                path.unlink(missing_ok=True)
            shutil.rmtree(temporary_dir, ignore_errors=True)
            raise
        self._index = replacement
        self._overlay_cache = None
        for path in [*old_paths, *old_overlay_paths]:
            path.unlink(missing_ok=True)
        shutil.rmtree(temporary_dir, ignore_errors=True)
        if self.overlays_dir.is_dir() and not any(self.overlays_dir.iterdir()):
            self.overlays_dir.rmdir()
        after = sum(path.stat().st_size for path in moved)
        return {"before_bytes": before, "after_bytes": after, "records": len(records)}

    def export(
        self,
        path: Path,
        *,
        format: str = "json",
        src_iri: Optional[str] = None,
    ) -> Path:
        """Generate a derived JSON, JSONL, or CSV explanation view."""

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        records: Iterable[dict[str, Any]] = (
            self.get(src_iri) if src_iri is not None else self.iter_all()
        )
        normalized = format.lower()
        if normalized == "json":
            with output.open("w", encoding="utf-8") as stream:
                stream.write("[")
                first = True
                for record in records:
                    if not first:
                        stream.write(",")
                    stream.write(
                        json.dumps(
                            record,
                            ensure_ascii=False,
                            separators=(",", ":"),
                            default=_json_default,
                        )
                    )
                    first = False
                stream.write("]")
        elif normalized == "jsonl":
            with output.open("w", encoding="utf-8") as stream:
                for record in records:
                    stream.write(
                        json.dumps(
                            record,
                            ensure_ascii=False,
                            separators=(",", ":"),
                            default=_json_default,
                        )
                    )
                    stream.write("\n")
        elif normalized == "csv":
            materialized = list(records)
            fieldnames = sorted({key for record in materialized for key in record})
            with output.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                for record in materialized:
                    writer.writerow(
                        {
                            key: (
                                json.dumps(
                                    value, ensure_ascii=False, separators=(",", ":")
                                )
                                if isinstance(value, (dict, list))
                                else value
                            )
                            for key, value in record.items()
                        }
                    )
        else:
            raise ValueError(f"Unsupported explanation export format: {format!r}")
        return output


__all__ = [
    "DEFAULT_HASH_BUCKETS",
    "DEFAULT_SHARD_MB",
    "EXPLANATION_SCHEMA_VERSION",
    "ExplanationStore",
    "INDEX_SCHEMA_VERSION",
]
