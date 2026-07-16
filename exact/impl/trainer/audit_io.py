from __future__ import annotations

import hashlib  # noqa: F401
import inspect  # noqa: F401
import json  # noqa: F401
import os
import time  # noqa: F401
from pathlib import Path  # noqa: F401
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple  # noqa: F401

import pandas as pd  # noqa: F401
import torch  # noqa: F401
from torch.utils.data import DataLoader, Subset  # noqa: F401
from torch.utils.data._utils.collate import default_collate  # noqa: F401

from exact.core.entities.configs.dataset import DatasetMask  # noqa: F401
from exact.core.entities.mappings import EntityMapping  # noqa: F401
from exact.runs.store import ExplanationStore
from exact.utils.formatting import format_duration as _format_duration  # noqa: F401
from exact.utils.timing import CacheStatus, StageRecord  # noqa: F401

try:
    import zstandard as zstd  # noqa: F401
except (
    ImportError
):  # pragma: no cover - exercised only when optional dependency is absent
    zstd = None


def _semantic_collate_fn(batch):
    """
    Keep variable-length fields (labels/contexts) as plain Python lists while default-collating the rest.
    This avoids torch's default_collate trying to stack ragged sequences of strings.
    """
    if not batch:
        return {}

    ragged_keys = {
        "src_labels",
        "tgt_labels",
        "src_ctx_triples",
        "tgt_ctx_triples",
        "src_ctx_raw_triples",
        "tgt_ctx_raw_triples",
        "src_ctx_bridge_triples",
        "tgt_ctx_bridge_triples",
    }
    list_only_keys = {"label"}
    collated = {}
    for key in batch[0].keys():
        values = [sample[key] for sample in batch]
        if key in ragged_keys or values[0] is None:
            collated[key] = values
        elif key in list_only_keys:
            collated[key] = values
        else:
            collated[key] = default_collate(values)

    if "src_ctx_triples" in collated:
        collated["src_contexts"] = collated["src_ctx_triples"]
    if "tgt_ctx_triples" in collated:
        collated["tgt_contexts"] = collated["tgt_ctx_triples"]
    return collated


def _json_default(value: Any) -> str:
    return str(value)


class AuditIOMixin:
    @staticmethod
    def _jsonl_suffix(compression: str) -> str:
        return ".jsonl.zst" if compression == "zstd" else ".jsonl"

    def _open_jsonl_writer(self, path: Path, compression: str):
        if compression == "zstd":
            return zstd.open(path, mode="wt", encoding="utf-8")  # type: ignore[union-attr]
        return open(path, "w", encoding="utf-8")

    def _open_jsonl_reader(self, path: Path, compression: str):
        if compression == "zstd":
            return zstd.open(path, mode="rt", encoding="utf-8")  # type: ignore[union-attr]
        return open(path, "r", encoding="utf-8")

    def _write_jsonl_records_atomic(
        self,
        path: Path,
        records: Iterable[Dict[str, Any]],
        compression: Optional[str] = None,
    ) -> int:
        resolved = self._resolve_text_compression(
            compression or self._audit_shard_compression
        )
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        count = 0
        with self._open_jsonl_writer(tmp_path, resolved) as f:
            for record in records:
                f.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        default=_json_default,
                    )
                )
                f.write("\n")
                count += 1
        tmp_path.replace(path)
        return count

    def _read_jsonl_records(
        self, path: Path, compression: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        resolved = self._resolve_text_compression(
            compression or ("zstd" if path.suffix == ".zst" else "none")
        )
        records: List[Dict[str, Any]] = []
        with self._open_jsonl_reader(path, resolved) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

    def _iter_jsonl_records(self, path: Path, compression: Optional[str] = None):
        resolved = self._resolve_text_compression(
            compression or ("zstd" if path.suffix == ".zst" else "none")
        )
        with self._open_jsonl_reader(path, resolved) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def _iter_records_from_manifest(
        self,
        manifest_path: Path,
        label: str,
        progress_every: int = 50000,
    ):
        manifest = self._load_audit_manifest(manifest_path)
        if not manifest:
            return
        compression = self._resolve_text_compression(
            manifest.get("compression", "none")
        )
        total_records = int(manifest.get("total_records", 0) or 0)
        shards = list(manifest.get("shards") or [])
        start = time.perf_counter()
        seen = 0
        self.log(
            (
                f"{label}: streaming {total_records or 'unknown'} records "
                f"from {len(shards)} shard(s) at {manifest_path}"
            ),
            "debug",
        )
        for shard_index, shard in enumerate(shards, start=1):
            rel_path = shard.get("path")
            if not rel_path:
                continue
            shard_path = (manifest_path.parent / str(rel_path)).resolve()
            try:
                for record in self._iter_jsonl_records(
                    shard_path, compression=compression
                ):
                    seen += 1
                    if progress_every > 0 and seen % progress_every == 0:
                        elapsed = max(1.0e-8, time.perf_counter() - start)
                        rate = seen / elapsed
                        remaining = max(0, total_records - seen) if total_records else 0
                        eta = (
                            _format_duration(remaining / rate)
                            if total_records and rate > 0
                            else "unknown"
                        )
                        self.log(
                            (
                                f"{label} progress: records={seen}/{total_records or '?'}, "
                                f"shard={shard_index}/{len(shards)}, avg={rate:.1f} records/s, ETA {eta}"
                            ),
                            "debug",
                        )
                    yield record
            except (OSError, json.JSONDecodeError) as exc:
                self.log(
                    f"{label}: failed to read shard {shard_path}: {exc}", "warning"
                )
                return
        elapsed = max(0.0, time.perf_counter() - start)
        rate = seen / elapsed if elapsed > 1.0e-8 else 0.0
        self.log(
            (
                f"{label}: streamed {seen} records in {_format_duration(elapsed)} "
                f"({rate:.1f} records/s)"
            ),
            "debug",
        )

    def _candidate_manifest_for_checkpoint(self, checkpoint_path: Path) -> Path:
        return (
            checkpoint_path.parent
            / f"{checkpoint_path.stem}_candidates"
            / "manifest.json"
        )

    def _overlay_manifest_for_checkpoint(self, checkpoint_path: Path) -> Path:
        return (
            checkpoint_path.parent / f"{checkpoint_path.stem}_overlay" / "manifest.json"
        )

    def _close_candidate_writer(self) -> None:
        writer = getattr(self, "_candidate_current_writer", None)
        if writer is None:
            return
        try:
            writer.close()
        finally:
            self._candidate_current_writer = None
            self._candidate_current_shard = None

    def _prepare_candidate_shards(
        self,
        checkpoint_path: Optional[Path],
        enabled: bool,
        compression: str,
        records_per_shard: int,
        append_existing: bool = True,
    ) -> None:
        self._close_candidate_writer()
        if hasattr(self, "run_layout"):
            # Layout v2 stores the union record once in ExplanationStore.
            self._candidate_records_enabled = False
            self._candidate_manifest_path = None
            self._candidate_shard_dir = None
            self._candidate_shards = []
            self._candidate_total_records = 0
            return
        self._candidate_records_enabled = bool(enabled and checkpoint_path is not None)
        self._candidate_manifest_path = None
        self._candidate_shard_dir = None
        self._candidate_shards = []
        self._candidate_total_records = 0
        if not self._candidate_records_enabled or checkpoint_path is None:
            return
        self._candidate_manifest_path = self._candidate_manifest_for_checkpoint(
            checkpoint_path
        )
        self._candidate_shard_dir = self._candidate_manifest_path.parent
        self._candidate_shard_dir.mkdir(parents=True, exist_ok=True)
        if not append_existing:
            for path in self._candidate_shard_dir.glob("shard-*.jsonl*"):
                try:
                    path.unlink()
                except OSError as exc:
                    self.log(
                        f"Failed to remove stale candidate shard {path}: {exc}",
                        "warning",
                    )
            if self._candidate_manifest_path.exists():
                try:
                    self._candidate_manifest_path.unlink()
                except OSError as exc:
                    self.log(
                        f"Failed to remove stale candidate manifest {self._candidate_manifest_path}: {exc}",
                        "warning",
                    )
        manifest = (
            self._load_audit_manifest(self._candidate_manifest_path)
            if append_existing
            else {}
        )
        self._candidate_shards = list(manifest.get("shards") or [])
        self._candidate_total_records = int(manifest.get("total_records", 0) or 0)
        manifest_compression = manifest.get("compression")
        if manifest_compression:
            self._audit_shard_compression = self._resolve_text_compression(
                manifest_compression
            )
        else:
            self._audit_shard_compression = self._resolve_text_compression(compression)
        self._audit_shard_records = max(1, int(records_per_shard or 50000))

    def _start_new_candidate_shard(self) -> None:
        if not self._candidate_records_enabled or self._candidate_shard_dir is None:
            return
        self._close_candidate_writer()
        idx = len(self._candidate_shards)
        suffix = self._jsonl_suffix(self._audit_shard_compression)
        shard_name = f"shard-{idx:06d}{suffix}"
        shard_path = self._candidate_shard_dir / shard_name
        shard = {"path": shard_name, "records": 0}
        self._candidate_shards.append(shard)
        self._candidate_current_shard = shard
        self._candidate_current_writer = self._open_jsonl_writer(
            shard_path, self._audit_shard_compression
        )

    def _append_candidate_records(self, records: List[Dict[str, Any]]) -> None:
        if getattr(self, "_explanation_store", None) is not None:
            return
        if not records or not self._candidate_records_enabled:
            return
        for record in records:
            if (
                self._candidate_current_writer is None
                or self._candidate_current_shard is None
                or int(self._candidate_current_shard.get("records", 0))
                >= self._audit_shard_records
            ):
                self._start_new_candidate_shard()
            if (
                self._candidate_current_writer is None
                or self._candidate_current_shard is None
            ):
                return
            self._candidate_current_writer.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    default=_json_default,
                )
            )
            self._candidate_current_writer.write("\n")
            self._candidate_current_shard["records"] = (
                int(self._candidate_current_shard.get("records", 0)) + 1
            )
            self._candidate_total_records += 1

    def _write_candidate_manifest(self) -> Optional[Path]:
        if getattr(self, "_explanation_store", None) is not None:
            return None
        if not self._candidate_records_enabled or self._candidate_manifest_path is None:
            return None
        self._close_candidate_writer()
        payload = {
            "version": 1,
            "format": "jsonl",
            "compression": self._audit_shard_compression,
            "records_per_shard": self._audit_shard_records,
            "total_records": self._candidate_total_records,
            "shards": self._candidate_shards,
        }
        self._write_json_atomic(self._candidate_manifest_path, payload)
        return self._candidate_manifest_path

    def _read_candidate_records_from_manifest(
        self, manifest_path: Path
    ) -> List[Dict[str, Any]]:
        return list(
            self._iter_records_from_manifest(
                manifest_path, "Checkpoint candidate restore"
            )
        )

    def _selector_evidence_items_for_record(
        self, record: Dict[str, Any]
    ) -> List[Dict[str, float]]:
        for extra_model in getattr(self, "models", [])[1:]:
            extractor = getattr(extra_model, "_record_evidence_items", None)
            if not callable(extractor):
                continue
            try:
                return [
                    {"key": str(key), "strength": float(strength)}
                    for key, strength in extractor(record)
                    if key
                ]
            except Exception as exc:  # noqa: BLE001
                self.log(
                    f"Failed to compact selector evidence for audit record: {exc}",
                    "debug",
                )
                return []
        return []

    def _candidate_row_from_explanation_record(
        self, record: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        src = record.get("src_iri")
        tgt = record.get("tgt_iri")
        if src is None or tgt is None:
            return None
        row: Dict[str, Any] = {
            "Src": str(src),
            "Tgt": str(tgt),
            "ground_truth": (record.get("prediction") or {}).get("ground_truth"),
            "src_label_text": (record.get("selected_labels") or {}).get("source", ""),
            "tgt_label_text": (record.get("selected_labels") or {}).get("target", ""),
            "llm_pair_brief": record.get("llm_pair_brief", ""),
        }
        for payload_name in ["confidences", "qualities", "weights", "importances"]:
            payload = record.get(payload_name) or {}
            for key, value in payload.items():
                if isinstance(value, (dict, list, tuple)):
                    continue
                row[key] = value
        if "S_final" not in row:
            return None
        evidence_items = self._selector_evidence_items_for_record(record)
        if evidence_items:
            row["selector_evidence_items"] = evidence_items
        return row

    @staticmethod
    def _union_explanation_record(
        candidate: Dict[str, Any],
        explanation: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Merge the former candidate sidecar fields into one explanation record."""

        record = dict(explanation or {})
        record.setdefault("explanation_schema_version", 1)
        record.setdefault("src_iri", str(candidate.get("Src", "")))
        record.setdefault("tgt_iri", str(candidate.get("Tgt", "")))
        record.setdefault("llm_pair_brief", candidate.get("llm_pair_brief", ""))

        prediction = dict(record.get("prediction") or {})
        prediction.setdefault("ground_truth", candidate.get("ground_truth"))
        record["prediction"] = prediction
        selected_labels = dict(record.get("selected_labels") or {})
        selected_labels.setdefault("source", candidate.get("src_label_text", ""))
        selected_labels.setdefault("target", candidate.get("tgt_label_text", ""))
        record["selected_labels"] = selected_labels

        sections = {
            "confidences": dict(record.get("confidences") or {}),
            "qualities": dict(record.get("qualities") or {}),
            "weights": dict(record.get("weights") or {}),
            "importances": dict(record.get("importances") or {}),
        }
        ignored = {
            "Src",
            "Tgt",
            "ground_truth",
            "src_label_text",
            "tgt_label_text",
            "src_context_text",
            "tgt_context_text",
            "llm_pair_brief",
            "selector_evidence_items",
        }
        for key, value in candidate.items():
            if key in ignored or isinstance(value, (dict, list, tuple)):
                continue
            if key.startswith("q_") or key == "Q_struct":
                section = "qualities"
            elif key.startswith("I_"):
                section = "importances"
            elif key.startswith("w_") or key in {"U", "U_ind", "U_dis"}:
                section = "weights"
            else:
                section = "confidences"
            sections[section].setdefault(key, value)
        for name, payload in sections.items():
            if payload:
                record[name] = payload
        evidence_items = candidate.get("selector_evidence_items")
        if evidence_items and "selector_evidence_items" not in record:
            record["selector_evidence_items"] = evidence_items
        return record

    def _migrate_legacy_audit_checkpoint(
        self,
        checkpoint_path: Path,
        payload: Dict[str, Any],
        audit_manifest_path: Path,
    ) -> List[Dict[str, Any]]:
        self.log(
            (
                f"Checkpoint {checkpoint_path.name} has only full audit shards for resume. "
                "Streaming them once to build slim candidate sidecars."
            ),
            "info",
        )
        compression = self._resolve_text_compression(
            (self._load_audit_manifest(audit_manifest_path) or {}).get(
                "compression", self._audit_shard_compression
            )
        )
        records_per_shard = int(
            (self._load_audit_manifest(audit_manifest_path) or {}).get(
                "records_per_shard",
                self._audit_shard_records,
            )
            or 50000
        )
        candidate_dir = self._candidate_manifest_for_checkpoint(checkpoint_path).parent
        if candidate_dir.exists():
            for stale_path in candidate_dir.glob("shard-*.jsonl*"):
                try:
                    stale_path.unlink()
                except OSError as exc:
                    self.log(
                        f"Failed to remove stale candidate shard {stale_path}: {exc}",
                        "warning",
                    )
            stale_manifest = candidate_dir / "manifest.json"
            if stale_manifest.exists():
                try:
                    stale_manifest.unlink()
                except OSError as exc:
                    self.log(
                        f"Failed to remove stale candidate manifest {stale_manifest}: {exc}",
                        "warning",
                    )
        self._prepare_candidate_shards(
            checkpoint_path,
            enabled=True,
            compression=compression,
            records_per_shard=records_per_shard,
        )
        candidate_rows: List[Dict[str, Any]] = []
        batch: List[Dict[str, Any]] = []
        for record in self._iter_records_from_manifest(
            audit_manifest_path,
            "Legacy audit migration",
        ):
            row = self._candidate_row_from_explanation_record(record)
            if row is None:
                continue
            candidate_rows.append(row)
            batch.append(row)
            if len(batch) >= 1000:
                self._append_candidate_records(batch)
                batch = []
        if batch:
            self._append_candidate_records(batch)
        candidate_manifest_path = self._write_candidate_manifest()
        if candidate_manifest_path is not None:
            payload = dict(payload)
            payload["checkpoint_schema_version"] = 2
            payload["candidate_records_manifest_path"] = self._relative_to_checkpoint(
                candidate_manifest_path,
                checkpoint_path,
            )
            payload["candidate_records_count"] = len(candidate_rows)
            payload.setdefault("checkpoint_payload", self._checkpoint_payload_mode)
            self._write_json_atomic(checkpoint_path, payload)
            self.log(
                (
                    f"Legacy audit migration complete: wrote {len(candidate_rows)} slim candidate "
                    f"records to {candidate_manifest_path}"
                ),
                "info",
            )
        return candidate_rows

    def _relative_to_checkpoint(self, path: Path, checkpoint_path: Path) -> str:
        return os.path.relpath(path.resolve(), checkpoint_path.parent.resolve())

    def _audit_manifest_for_checkpoint(self, checkpoint_path: Path) -> Path:
        return (
            checkpoint_path.parent / f"{checkpoint_path.stem}_audit" / "manifest.json"
        )

    def _close_audit_writer(self) -> None:
        writer = getattr(self, "_audit_current_writer", None)
        if writer is None:
            return
        try:
            writer.close()
        finally:
            self._audit_current_writer = None
            self._audit_current_shard = None

    def _load_audit_manifest(self, manifest_path: Path) -> Dict[str, Any]:
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
        except (OSError, json.JSONDecodeError) as exc:
            self.log(f"Failed to load audit manifest {manifest_path}: {exc}", "warning")
            return {}

    def _prepare_audit_shards(
        self,
        checkpoint_path: Optional[Path],
        enabled: bool,
        compression: str,
        records_per_shard: int,
        append_existing: bool = True,
    ) -> None:
        self._close_audit_writer()
        if hasattr(self, "run_layout"):
            self._audit_shards_enabled = bool(enabled)
            self._audit_shard_compression = self._resolve_text_compression(compression)
            self._audit_shard_records = max(1, int(records_per_shard or 50000))
            self._audit_manifest_path = None
            self._audit_shard_dir = None
            self._audit_shards = []
            self._audit_total_records = 0
            if not self._audit_shards_enabled:
                return
            store = getattr(self, "_explanation_store", None)
            if store is None:
                store = ExplanationStore(
                    self.run_layout.explanations_dir,
                    run_id=getattr(self, "_run_id", None),
                    shard_mb=float(getattr(self, "_explanation_shard_mb", 32.0)),
                    compression=self._audit_shard_compression,
                )
                self._explanation_store = store
            if not append_existing and store.record_count:
                store.clear()
            self._audit_manifest_path = store.index_path
            self._audit_shard_dir = store.shards_dir
            self._audit_total_records = store.record_count
            return
        self._audit_shards_enabled = bool(enabled and checkpoint_path is not None)
        self._audit_shard_compression = self._resolve_text_compression(compression)
        self._audit_shard_records = max(1, int(records_per_shard or 50000))
        self._audit_manifest_path = None
        self._audit_shard_dir = None
        self._audit_shards = []
        self._audit_total_records = 0
        if not self._audit_shards_enabled or checkpoint_path is None:
            return
        self._audit_manifest_path = self._audit_manifest_for_checkpoint(checkpoint_path)
        self._audit_shard_dir = self._audit_manifest_path.parent
        self._audit_shard_dir.mkdir(parents=True, exist_ok=True)
        if not append_existing:
            for path in self._audit_shard_dir.glob("shard-*.jsonl*"):
                try:
                    path.unlink()
                except OSError as exc:
                    self.log(
                        f"Failed to remove stale audit shard {path}: {exc}", "warning"
                    )
            if self._audit_manifest_path.exists():
                try:
                    self._audit_manifest_path.unlink()
                except OSError as exc:
                    self.log(
                        f"Failed to remove stale audit manifest {self._audit_manifest_path}: {exc}",
                        "warning",
                    )
        manifest = (
            self._load_audit_manifest(self._audit_manifest_path)
            if append_existing
            else {}
        )
        self._audit_shards = list(manifest.get("shards") or [])
        self._audit_total_records = int(manifest.get("total_records", 0) or 0)
        manifest_compression = manifest.get("compression")
        if manifest_compression:
            self._audit_shard_compression = self._resolve_text_compression(
                manifest_compression
            )

    def _start_new_audit_shard(self) -> None:
        if not self._audit_shards_enabled or self._audit_shard_dir is None:
            return
        self._close_audit_writer()
        idx = len(self._audit_shards)
        suffix = self._jsonl_suffix(self._audit_shard_compression)
        shard_name = f"shard-{idx:06d}{suffix}"
        shard_path = self._audit_shard_dir / shard_name
        shard = {"path": shard_name, "records": 0}
        self._audit_shards.append(shard)
        self._audit_current_shard = shard
        self._audit_current_writer = self._open_jsonl_writer(
            shard_path, self._audit_shard_compression
        )

    def _append_audit_records(self, records: List[Dict[str, Any]]) -> None:
        store = getattr(self, "_explanation_store", None)
        if store is not None:
            store.append(
                records,
                run_id=getattr(self, "_run_id", None),
            )
            self._audit_total_records = store.record_count
            return
        if not records or not self._audit_shards_enabled:
            return
        for record in records:
            if (
                self._audit_current_writer is None
                or self._audit_current_shard is None
                or int(self._audit_current_shard.get("records", 0))
                >= self._audit_shard_records
            ):
                self._start_new_audit_shard()
            if self._audit_current_writer is None or self._audit_current_shard is None:
                return
            self._audit_current_writer.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    default=_json_default,
                )
            )
            self._audit_current_writer.write("\n")
            self._audit_current_shard["records"] = (
                int(self._audit_current_shard.get("records", 0)) + 1
            )
            self._audit_total_records += 1

    def _write_audit_manifest(self) -> Optional[Path]:
        store = getattr(self, "_explanation_store", None)
        if store is not None:
            self._audit_total_records = store.record_count
            self._audit_manifest_path = store.index_path
            return store.index_path
        if not self._audit_shards_enabled or self._audit_manifest_path is None:
            return None
        self._close_audit_writer()
        payload = {
            "version": 1,
            "format": "jsonl",
            "compression": self._audit_shard_compression,
            "records_per_shard": self._audit_shard_records,
            "total_records": self._audit_total_records,
            "shards": self._audit_shards,
        }
        self._write_json_atomic(self._audit_manifest_path, payload)
        return self._audit_manifest_path

    def _rewrite_audit_shards(self, records: List[Dict[str, Any]]) -> None:
        store = getattr(self, "_explanation_store", None)
        if store is not None:
            store.clear()
            store.append(records, run_id=getattr(self, "_run_id", None))
            self._audit_total_records = store.record_count
            return
        if not self._audit_shards_enabled or self._audit_shard_dir is None:
            return
        self._close_audit_writer()
        for path in self._audit_shard_dir.glob("shard-*.jsonl*"):
            try:
                path.unlink()
            except OSError as exc:
                self.log(f"Failed to remove stale audit shard {path}: {exc}", "warning")
        self._audit_shards = []
        self._audit_total_records = 0
        self._append_audit_records(records)
        self._write_audit_manifest()

    def _read_audit_records_from_manifest(
        self, manifest_path: Path
    ) -> List[Dict[str, Any]]:
        if manifest_path.name == "index.json":
            try:
                store = ExplanationStore(manifest_path.parent)
                self._explanation_store = store
                self._audit_manifest_path = store.index_path
                self._audit_total_records = store.record_count
                return list(store.iter_all())
            except ValueError as exc:
                self.log(
                    f"Failed to read explanation store {manifest_path}: {exc}",
                    "warning",
                )
                return []
        manifest = self._load_audit_manifest(manifest_path)
        if not manifest:
            return []
        compression = self._resolve_text_compression(
            manifest.get("compression", "none")
        )
        records: List[Dict[str, Any]] = []
        for shard in manifest.get("shards") or []:
            rel_path = shard.get("path")
            if not rel_path:
                continue
            shard_path = (manifest_path.parent / str(rel_path)).resolve()
            try:
                records.extend(
                    self._read_jsonl_records(shard_path, compression=compression)
                )
            except (OSError, json.JSONDecodeError) as exc:
                self.log(f"Failed to read audit shard {shard_path}: {exc}", "warning")
                return []
        return records

    def has_streamed_explanations(self) -> bool:
        store = getattr(self, "_explanation_store", None)
        if store is not None:
            return store.record_count > 0
        manifest_path = getattr(self, "_audit_manifest_path", None)
        return bool(manifest_path and Path(manifest_path).exists())

    def _load_overlay_lookup(self) -> Dict[Tuple[str, str], Dict[str, Any]]:
        manifest_path = getattr(self, "_overlay_manifest_path", None)
        if not manifest_path or not Path(manifest_path).exists():
            return {}
        lookup: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for record in self._iter_records_from_manifest(
            Path(manifest_path), "Final overlay load"
        ):
            src = record.get("Src")
            tgt = record.get("Tgt")
            if src is None or tgt is None:
                continue
            lookup[(str(src), str(tgt))] = record
        return lookup

    @staticmethod
    def _merge_overlay_record(
        record: Dict[str, Any], overlay: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if not overlay:
            return record
        merged = dict(record)
        for section in ["confidences", "prediction", "models", "backend_usage"]:
            payload = overlay.get(section)
            if not isinstance(payload, dict):
                continue
            current = dict(merged.get(section) or {})
            current.update(payload)
            merged[section] = current
        return merged

    def write_full_explanations_json(self, path: Path) -> None:
        store = getattr(self, "_explanation_store", None)
        if store is not None and store.record_count:
            store.export(path, format="json")
            return
        manifest_path = getattr(self, "_audit_manifest_path", None)
        manifest = self._load_audit_manifest(manifest_path) if manifest_path else {}
        if not manifest:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    self.results_json,
                    f,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    default=_json_default,
                )
            return

        compression = self._resolve_text_compression(
            manifest.get("compression", "none")
        )
        overlay_lookup = self._load_overlay_lookup()
        first = True
        written = 0
        start = time.perf_counter()
        self.log(f"Streaming full explanations JSON to {path}", "info")
        with open(path, "w", encoding="utf-8") as out:
            out.write("[")
            for shard in manifest.get("shards") or []:
                rel_path = shard.get("path")
                if not rel_path:
                    continue
                shard_path = (manifest_path.parent / str(rel_path)).resolve()
                with self._open_jsonl_reader(shard_path, compression) as source:
                    for line in source:
                        line = line.strip()
                        if not line:
                            continue
                        if not first:
                            out.write(",")
                        record = json.loads(line)
                        overlay = overlay_lookup.get(
                            (str(record.get("src_iri")), str(record.get("tgt_iri")))
                        )
                        out.write(
                            json.dumps(
                                self._merge_overlay_record(record, overlay),
                                ensure_ascii=False,
                                separators=(",", ":"),
                                default=_json_default,
                            )
                        )
                        first = False
                        written += 1
                        if written % 50000 == 0:
                            elapsed = max(1.0e-8, time.perf_counter() - start)
                            self.log(
                                (
                                    f"Full explanations export progress: records={written}/"
                                    f"{int(manifest.get('total_records', 0) or 0) or '?'}, "
                                    f"avg={written / elapsed:.1f} records/s"
                                ),
                                "debug",
                            )
            out.write("]")
        elapsed = max(0.0, time.perf_counter() - start)
        self.log(
            f"Finished full explanations JSON export: records={written}, duration={_format_duration(elapsed)}",
            "info",
        )
