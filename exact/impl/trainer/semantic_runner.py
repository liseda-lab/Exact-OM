import inspect
import json
import hashlib
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Set

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data._utils.collate import default_collate
from exact.core.contracts.trainer import ITrainer
from exact.core.entities.mappings import EntityMapping
from exact.core.entities.configs.dataset import DatasetMask

try:
    import zstandard as zstd
except ImportError:  # pragma: no cover - exercised only when optional dependency is absent
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


def _format_duration(total_seconds: float) -> str:
    """
    Convert seconds into a days:hours:minutes:seconds string for readable ETAs.
    """
    seconds = max(0, int(round(total_seconds)))
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{days}d:{hours:02d}:{minutes:02d}:{seconds:02d}"


def _json_default(value: Any) -> str:
    return str(value)


class SemanticAlignmentRunner(ITrainer):
    """
    External loop orchestrator for SemanticScorer inference.
    Collects all outputs (scores, weights, explanations) for saving & plotting.
    """

    def __init__(
        self,
        dataset,
        model=None,
        model_params: Optional[Dict[str, Any]] = None,
        models: Optional[List[Tuple[Any, Dict[str, Any]]]] = None,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        output_dir: Optional[Path] = None,
        logger: Optional[Any] = None,
        **kwargs,
    ):
        params = dict(model_params or {})
        cache_dir = params.get("cache_dir")
        if cache_dir is None and output_dir is not None:
            params["cache_dir"] = (output_dir / "cache").resolve()
        ds_signature = getattr(dataset, "dataset_signature", None)
        if ds_signature:
            params.setdefault("dataset_signature", ds_signature)
            params.setdefault("cache_namespace", ds_signature)

        model_specs = None
        if models is not None:
            model_specs = []
            for idx, (m_cls, m_params) in enumerate(models):
                spec_params = dict(m_params or {})
                if idx == 0:
                    cache_dir = spec_params.get("cache_dir")
                    if cache_dir is None and output_dir is not None:
                        spec_params["cache_dir"] = (output_dir / "cache").resolve()
                    if ds_signature:
                        spec_params.setdefault("dataset_signature", ds_signature)
                        spec_params.setdefault("cache_namespace", ds_signature)
                model_specs.append((m_cls, spec_params))

        super().__init__(
            dataset=dataset,
            model=model,
            model_params=params if model_specs is None else None,
            models=model_specs,
            device=device,
            output_dir=output_dir,
            logger=logger,
            **kwargs,
        )
        self._last_stage_timings: Dict[str, float] = {}
        self._checkpoint_fingerprint_payload: Dict[str, Any] = self._build_checkpoint_fingerprint_payload()
        self._checkpoint_fingerprint: str = self._hash_checkpoint_fingerprint_payload(
            self._checkpoint_fingerprint_payload
        )
        self._audit_shards_enabled: bool = True
        self._audit_shard_compression: str = "zstd"
        self._audit_shard_records: int = 50000
        self._audit_manifest_path: Optional[Path] = None
        self._audit_shard_dir: Optional[Path] = None
        self._audit_shards: List[Dict[str, Any]] = []
        self._audit_total_records: int = 0
        self._audit_current_writer: Optional[Any] = None
        self._audit_current_shard: Optional[Dict[str, Any]] = None
        self._candidate_records_enabled: bool = True
        self._candidate_manifest_path: Optional[Path] = None
        self._candidate_shard_dir: Optional[Path] = None
        self._candidate_shards: List[Dict[str, Any]] = []
        self._candidate_total_records: int = 0
        self._candidate_current_writer: Optional[Any] = None
        self._candidate_current_shard: Optional[Dict[str, Any]] = None
        self._overlay_manifest_path: Optional[Path] = None
        self._checkpoint_payload_mode: str = "compact"
        self._cache_persist_policy: str = "finalize"
        self._cache_persist_skip_logged: Set[str] = set()
        self._restored_candidate_rows: List[Dict[str, Any]] = []

    @staticmethod
    def _hash_checkpoint_fingerprint_payload(payload: Dict[str, Any]) -> str:
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()

    @staticmethod
    def _format_checkpoint_mismatch_value(value: Any, max_chars: int = 120) -> str:
        try:
            rendered = json.dumps(value, sort_keys=True, default=str)
        except TypeError:
            rendered = repr(value)
        if len(rendered) > max_chars:
            return f"{rendered[: max_chars - 3]}..."
        return rendered

    def _build_checkpoint_fingerprint_payload(
        self,
        generate_llm_rationales_override: Optional[bool] = None,
    ) -> Dict[str, Any]:
        dataset_signature = getattr(self.dataset, "dataset_signature", None)
        dataset_fingerprint = getattr(self.dataset, "cache_fingerprint", None)
        # Checkpoints contain primary inference outputs. Extra models in the
        # chain are post-inference transforms and are reapplied after restore,
        # so they must not invalidate an otherwise reusable checkpoint.
        models = [getattr(self, "model", None)]
        model_payloads: List[Dict[str, Any]] = []
        for model in models:
            if model is None:
                continue
            fingerprint_payload: Optional[Dict[str, Any]] = None
            if hasattr(model, "runtime_fingerprint"):
                if hasattr(model, "runtime_fingerprint_payload"):
                    fingerprint_payload = model.runtime_fingerprint_payload(
                        generate_llm_rationales_override=generate_llm_rationales_override
                    )
                    fingerprint = self._hash_checkpoint_fingerprint_payload(fingerprint_payload)
                else:
                    fingerprint = model.runtime_fingerprint()
            elif hasattr(model, "_cache_fingerprint"):
                fingerprint = getattr(model, "_cache_fingerprint")
            else:
                fingerprint = None
            model_entry: Dict[str, Any] = {
                "class": model.__class__.__name__,
                "fingerprint": fingerprint,
            }
            if fingerprint_payload is not None:
                model_entry["payload"] = fingerprint_payload
            model_payloads.append(model_entry)
        return {
            "dataset_signature": dataset_signature,
            "dataset_fingerprint": dataset_fingerprint,
            "models": model_payloads,
        }

    def _build_checkpoint_fingerprint(
        self,
        generate_llm_rationales_override: Optional[bool] = None,
    ) -> str:
        payload = self._build_checkpoint_fingerprint_payload(
            generate_llm_rationales_override=generate_llm_rationales_override
        )
        return self._hash_checkpoint_fingerprint_payload(payload)

    def _collect_checkpoint_fingerprint_diffs(
        self,
        checkpoint_value: Any,
        current_value: Any,
        path: str,
        diffs: List[str],
        max_diffs: int,
    ) -> None:
        if len(diffs) >= max_diffs:
            return
        if type(checkpoint_value) is not type(current_value):
            diffs.append(
                f"{path}: checkpoint={self._format_checkpoint_mismatch_value(checkpoint_value)}, "
                f"current={self._format_checkpoint_mismatch_value(current_value)}"
            )
            return
        if isinstance(checkpoint_value, dict):
            for key in sorted(set(checkpoint_value) | set(current_value)):
                if len(diffs) >= max_diffs:
                    return
                next_path = f"{path}.{key}" if path else str(key)
                if key not in checkpoint_value:
                    diffs.append(
                        f"{next_path}: checkpoint=<missing>, "
                        f"current={self._format_checkpoint_mismatch_value(current_value[key])}"
                    )
                    continue
                if key not in current_value:
                    diffs.append(
                        f"{next_path}: checkpoint={self._format_checkpoint_mismatch_value(checkpoint_value[key])}, "
                        "current=<missing>"
                    )
                    continue
                self._collect_checkpoint_fingerprint_diffs(
                    checkpoint_value[key],
                    current_value[key],
                    next_path,
                    diffs,
                    max_diffs,
                )
            return
        if isinstance(checkpoint_value, list):
            shared = min(len(checkpoint_value), len(current_value))
            for idx in range(shared):
                if len(diffs) >= max_diffs:
                    return
                self._collect_checkpoint_fingerprint_diffs(
                    checkpoint_value[idx],
                    current_value[idx],
                    f"{path}[{idx}]",
                    diffs,
                    max_diffs,
                )
            if len(diffs) >= max_diffs:
                return
            if len(checkpoint_value) != len(current_value):
                diffs.append(
                    f"{path}.length: checkpoint={len(checkpoint_value)}, current={len(current_value)}"
                )
            return
        if checkpoint_value != current_value:
            diffs.append(
                f"{path}: checkpoint={self._format_checkpoint_mismatch_value(checkpoint_value)}, "
                f"current={self._format_checkpoint_mismatch_value(current_value)}"
            )

    def _describe_checkpoint_fingerprint_mismatch(
        self,
        checkpoint_payload: Optional[Dict[str, Any]],
        current_payload: Optional[Dict[str, Any]],
        max_diffs: int = 8,
    ) -> Optional[str]:
        if not isinstance(checkpoint_payload, dict) or not isinstance(current_payload, dict):
            return None
        diffs: List[str] = []
        self._collect_checkpoint_fingerprint_diffs(
            checkpoint_payload,
            current_payload,
            "",
            diffs,
            max_diffs + 1,
        )
        if not diffs:
            return None
        extra = len(diffs) - max_diffs
        shown = diffs[:max_diffs]
        summary = "; ".join(shown)
        if extra > 0:
            summary = f"{summary}; ... (+{extra} more)"
        return summary

    def _checkpoint_matches_rationale_toggle_override(
        self,
        payload_fingerprint: Optional[str],
    ) -> bool:
        if not payload_fingerprint:
            return False
        model = getattr(self, "model", None)
        if model is None or not hasattr(model, "generate_llm_rationales"):
            return False
        current_value = getattr(model, "generate_llm_rationales", None)
        if not isinstance(current_value, bool):
            return False
        alternate = self._build_checkpoint_fingerprint(
            generate_llm_rationales_override=(not current_value)
        )
        return payload_fingerprint == alternate

    def _auto_checkpoint_filename(self, kind: DatasetMask) -> str:
        return f"{kind.name.lower()}_{int(time.time())}.json"

    def _ensure_checkpoint_path(
        self,
        kind: DatasetMask,
        preferred_file: Optional[str],
        existing_path: Optional[Path],
    ) -> Optional[Path]:
        if existing_path:
            return existing_path

        filename = preferred_file or self._auto_checkpoint_filename(kind)
        try:
            path = (self.checkpoint_dir / filename).resolve()
        except OSError as exc:
            self.log(f"Unable to prepare checkpoint file '{filename}': {exc}", level="warning")
            return None

        if path.exists():
            self.log(
                f"Checkpoint file {path} already exists and will be overwritten.",
                level="warning",
            )
        else:
            self.log(f"Writing checkpoints to {path}", level="debug")
        return path

    def _restore_from_available_checkpoints(
        self,
        kind: DatasetMask,
        preferred_file: Optional[str],
        allow_rationale_toggle_checkpoint_resume: bool = False,
    ) -> Tuple[Optional[Path], List[Tuple[str, str, float]], List[Dict[str, Any]], int]:
        candidates: List[Path] = []
        if preferred_file:
            candidates.append((self.checkpoint_dir / preferred_file).resolve())

        try:
            existing = sorted(
                (p for p in self.checkpoint_dir.glob("*.json") if p.is_file()),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        except OSError as exc:
            self.log(f"Unable to list checkpoints in {self.checkpoint_dir}: {exc}", level="warning")
            existing = []

        seen = set()
        ordered: List[Path] = []
        for path in candidates + existing:
            if path in seen:
                continue
            seen.add(path)
            ordered.append(path)

        if ordered:
            self.log(
                f"Scanning {len(ordered)} checkpoint candidate(s) for {kind.name} resume.",
                "debug",
            )
        for path in ordered:
            mappings, results_json, processed_examples = self._load_checkpoint_state(
                path,
                kind,
                allow_rationale_toggle_checkpoint_resume=allow_rationale_toggle_checkpoint_resume,
            )
            if processed_examples > 0:
                self.log(
                    (
                        f"Accepted checkpoint {path.name}: "
                        f"processed_examples={processed_examples}, mappings={len(mappings)}, "
                        f"full_records_in_memory={len(results_json)}."
                    ),
                    "info",
                )
                return path, mappings, results_json, processed_examples
        return None, [], [], 0

    def _load_checkpoint_state(
        self,
        checkpoint_path: Path,
        kind: DatasetMask,
        allow_rationale_toggle_checkpoint_resume: bool = False,
    ) -> Tuple[List[Tuple[str, str, float]], List[Dict[str, Any]], int]:
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except FileNotFoundError:
            return [], [], 0
        except json.JSONDecodeError as exc:
            self.log(f"Failed to parse checkpoint {checkpoint_path}: {exc}", "warning")
            return [], [], 0

        if payload.get("kind") and payload["kind"] != kind.name:
            self.log(
                (
                    f"Ignoring checkpoint '{checkpoint_path.name}' because it was created "
                    f"for dataset kind '{payload['kind']}' (current: '{kind.name}')."
                ),
                level="warning",
            )
            return [], [], 0
        payload_dataset_signature = payload.get("dataset_signature")
        current_dataset_signature = getattr(self.dataset, "dataset_signature", None)
        if payload_dataset_signature != current_dataset_signature:
            self.log(
                (
                    f"Ignoring checkpoint '{checkpoint_path.name}' because its dataset signature "
                    f"does not match the current dataset."
                ),
                level="warning",
            )
            return [], [], 0
        payload_fingerprint = payload.get("checkpoint_fingerprint")
        payload_fingerprint_payload = payload.get("checkpoint_fingerprint_payload")
        if not payload_fingerprint:
            self.log(
                (
                    f"Ignoring checkpoint '{checkpoint_path.name}' because it lacks a model/config "
                    f"fingerprint and may have been created by an older or different run."
                ),
                level="warning",
            )
            return [], [], 0
        if payload_fingerprint != self._checkpoint_fingerprint:
            if (
                allow_rationale_toggle_checkpoint_resume
                and self._checkpoint_matches_rationale_toggle_override(payload_fingerprint)
            ):
                self.log(
                    (
                        f"Accepting checkpoint '{checkpoint_path.name}' despite a fingerprint mismatch "
                        "because the only allowed change is `generate_llm_rationales`."
                    ),
                    level="warning",
                )
            else:
                mismatch_details = self._describe_checkpoint_fingerprint_mismatch(
                    payload_fingerprint_payload,
                    getattr(self, "_checkpoint_fingerprint_payload", None),
                )
                detail_suffix = f" Mismatch details: {mismatch_details}" if mismatch_details else ""
                self.log(
                    (
                        f"Ignoring checkpoint '{checkpoint_path.name}' because its model/config "
                        f"fingerprint does not match the current run.{detail_suffix}"
                    ),
                    level="warning",
                )
                return [], [], 0

        self._restored_candidate_rows = []
        results_json = payload.get("results_json") or []
        mappings: List[Tuple[str, str, float]] = []

        candidate_manifest_value = payload.get("candidate_records_manifest_path")
        if candidate_manifest_value:
            candidate_manifest_path = Path(str(candidate_manifest_value))
            if not candidate_manifest_path.is_absolute():
                candidate_manifest_path = (checkpoint_path.parent / candidate_manifest_path).resolve()
            self._candidate_manifest_path = candidate_manifest_path
            candidate_rows = self._read_candidate_records_from_manifest(candidate_manifest_path)
            self._restored_candidate_rows = candidate_rows
            for row in candidate_rows:
                src = row.get("Src")
                tgt = row.get("Tgt")
                score = row.get("S_final")
                if src is None or tgt is None or score is None:
                    continue
                try:
                    mappings.append((str(src), str(tgt), float(score)))
                except (TypeError, ValueError):
                    continue
            self.log(
                (
                    f"Loaded {len(candidate_rows)} slim candidate records from "
                    f"{candidate_manifest_path}"
                ),
                "debug",
            )

        overlay_value = payload.get("final_overlay_manifest_path")
        if overlay_value:
            overlay_path = Path(str(overlay_value))
            if not overlay_path.is_absolute():
                overlay_path = (checkpoint_path.parent / overlay_path).resolve()
            self._overlay_manifest_path = overlay_path

        for rec in payload.get("mappings", []):
            src = rec.get("src")
            tgt = rec.get("tgt")
            score = rec.get("score")
            if src is None or tgt is None:
                continue
            try:
                mappings.append((src, tgt, float(score)))
            except (TypeError, ValueError):
                continue

        if not self._restored_candidate_rows and results_json:
            candidate_rows = []
            for rec in results_json:
                row = self._candidate_row_from_explanation_record(rec)
                if row is None:
                    continue
                candidate_rows.append(row)
                try:
                    mappings.append((str(row["Src"]), str(row["Tgt"]), float(row["S_final"])))
                except (TypeError, ValueError):
                    continue
            self._restored_candidate_rows = candidate_rows

        manifest_value = payload.get("audit_manifest_path")
        if manifest_value:
            manifest_path = Path(str(manifest_value))
            if not manifest_path.is_absolute():
                manifest_path = (checkpoint_path.parent / manifest_path).resolve()
            self._audit_manifest_path = manifest_path
        else:
            manifest_path = None
        if manifest_path is not None and not results_json:
            results_json = self._read_audit_records_from_manifest(manifest_path)
            if results_json:
                self.log(
                    f"Loaded {len(results_json)} audit records from {manifest_path}",
                    "debug",
                )
        if not mappings and manifest_path is not None:
            migrated_rows = self._migrate_legacy_audit_checkpoint(
                checkpoint_path,
                payload,
                manifest_path,
            )
            self._restored_candidate_rows = migrated_rows
            for row in migrated_rows:
                try:
                    mappings.append((str(row["Src"]), str(row["Tgt"]), float(row["S_final"])))
                except (KeyError, TypeError, ValueError):
                    continue

        processed_examples = int(payload.get("processed_examples", len(mappings)))
        if processed_examples > 0 and not mappings and not results_json and not self._restored_candidate_rows:
            self.log(
                (
                    f"Ignoring checkpoint '{checkpoint_path.name}' because it records "
                    f"{processed_examples} processed examples but no restorable mappings or candidate rows."
                ),
                "warning",
            )
            return [], [], 0
        return mappings, results_json, processed_examples

    def _write_checkpoint_state(
        self,
        checkpoint_path: Path,
        kind: DatasetMask,
        total_examples: int,
        processed_examples: int,
        mappings: List[Tuple[str, str, float]],
        results_json: List[Dict[str, Any]],
    ) -> None:
        audit_manifest_path = self._write_audit_manifest()
        candidate_manifest_path = self._write_candidate_manifest()
        payload = {
            "checkpoint_schema_version": 2,
            "kind": kind.name,
            "dataset_signature": getattr(getattr(self, "_dataset", None), "dataset_signature", None),
            "dataset_fingerprint": getattr(getattr(self, "_dataset", None), "cache_fingerprint", None),
            "checkpoint_fingerprint": self._checkpoint_fingerprint,
            "checkpoint_fingerprint_payload": getattr(
                self,
                "_checkpoint_fingerprint_payload",
                self._build_checkpoint_fingerprint_payload(),
            ),
            "total_examples": total_examples,
            "processed_examples": processed_examples,
            "checkpoint_payload": self._checkpoint_payload_mode,
            "mappings_count": len(mappings),
            "results_json_count": len(results_json),
        }
        if audit_manifest_path is not None:
            payload["audit_manifest_path"] = self._relative_to_checkpoint(audit_manifest_path, checkpoint_path)
        if candidate_manifest_path is not None:
            payload["candidate_records_manifest_path"] = self._relative_to_checkpoint(
                candidate_manifest_path,
                checkpoint_path,
            )
            payload["candidate_records_count"] = int(getattr(self, "_candidate_total_records", 0) or 0)
        overlay_manifest_path = getattr(self, "_overlay_manifest_path", None)
        if overlay_manifest_path is not None and Path(overlay_manifest_path).exists():
            payload["final_overlay_manifest_path"] = self._relative_to_checkpoint(
                Path(overlay_manifest_path),
                checkpoint_path,
            )
        if self._checkpoint_payload_mode == "full" or (
            audit_manifest_path is None and candidate_manifest_path is None
        ):
            payload["mappings"] = [
                {"src": s, "tgt": t, "score": score} for s, t, score in mappings
            ]
            payload["results_json"] = results_json

        tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False, default=_json_default)
            tmp_path.replace(checkpoint_path)
            self.log(
                (
                    f"Wrote checkpoint ({processed_examples}/{total_examples} examples) "
                    f"to {checkpoint_path}"
                ),
                level="debug",
            )
        except OSError as exc:
            self.log(f"Failed to write checkpoint {checkpoint_path}: {exc}", "warning")

    def _model_fingerprint_entry(self, model: Any) -> Dict[str, Any]:
        fingerprint_payload: Optional[Dict[str, Any]] = None
        fingerprint = None
        if hasattr(model, "runtime_fingerprint_payload"):
            try:
                fingerprint_payload = model.runtime_fingerprint_payload()
            except TypeError:
                fingerprint_payload = model.runtime_fingerprint_payload(generate_llm_rationales_override=None)
            fingerprint = self._hash_checkpoint_fingerprint_payload(fingerprint_payload)
        elif hasattr(model, "runtime_fingerprint"):
            fingerprint = model.runtime_fingerprint()
        elif hasattr(model, "_cache_fingerprint"):
            fingerprint = getattr(model, "_cache_fingerprint")
        entry: Dict[str, Any] = {"class": model.__class__.__name__, "fingerprint": fingerprint}
        if fingerprint_payload is not None:
            entry["payload"] = fingerprint_payload
        return entry

    def _postprocess_fingerprint_payload(
        self,
        kind: DatasetMask,
        local_alignment: bool,
        threshold: Optional[float],
        cardinality: Optional[int],
    ) -> Dict[str, Any]:
        models = [model for model in (getattr(self, "models", None) or []) if model is not None]
        if not models:
            models = [getattr(self, "model", None)]
        return {
            "dataset_signature": getattr(getattr(self, "_dataset", None), "dataset_signature", None),
            "kind": kind.name,
            "local_alignment": bool(local_alignment),
            "threshold": threshold,
            "cardinality": cardinality,
            "models": [self._model_fingerprint_entry(model) for model in models if model is not None],
        }

    def _stage_checkpoint_path(
        self,
        kind: DatasetMask,
        stage: str,
        local_alignment: bool,
        threshold: Optional[float],
        cardinality: Optional[int],
    ) -> Path:
        payload = self._postprocess_fingerprint_payload(kind, local_alignment, threshold, cardinality)
        fingerprint = self._hash_checkpoint_fingerprint_payload(payload)
        return (self.checkpoint_dir / f"{kind.name.lower()}_{stage}_{fingerprint[:12]}.json").resolve()

    def _write_json_atomic(self, path: Path, payload: Dict[str, Any]) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=_json_default)
        tmp_path.replace(path)

    def _resolve_text_compression(self, compression: Optional[str]) -> str:
        resolved = str(compression or "none").lower()
        if resolved == "zstd" and zstd is None:
            skip_logged = getattr(self, "_cache_persist_skip_logged", set())
            if "zstd_missing" not in skip_logged:
                self.log(
                    "zstandard is not installed; writing audit artifacts as plain JSONL.",
                    "warning",
                )
                skip_logged.add("zstd_missing")
                self._cache_persist_skip_logged = skip_logged
            return "none"
        return resolved if resolved in {"zstd", "none"} else "none"

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
        resolved = self._resolve_text_compression(compression or self._audit_shard_compression)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        count = 0
        with self._open_jsonl_writer(tmp_path, resolved) as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":"), default=_json_default))
                f.write("\n")
                count += 1
        tmp_path.replace(path)
        return count

    def _read_jsonl_records(self, path: Path, compression: Optional[str] = None) -> List[Dict[str, Any]]:
        resolved = self._resolve_text_compression(compression or ("zstd" if path.suffix == ".zst" else "none"))
        records: List[Dict[str, Any]] = []
        with self._open_jsonl_reader(path, resolved) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

    def _iter_jsonl_records(self, path: Path, compression: Optional[str] = None):
        resolved = self._resolve_text_compression(compression or ("zstd" if path.suffix == ".zst" else "none"))
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
        compression = self._resolve_text_compression(manifest.get("compression", "none"))
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
                for record in self._iter_jsonl_records(shard_path, compression=compression):
                    seen += 1
                    if progress_every > 0 and seen % progress_every == 0:
                        elapsed = max(1.0e-8, time.perf_counter() - start)
                        rate = seen / elapsed
                        remaining = max(0, total_records - seen) if total_records else 0
                        eta = _format_duration(remaining / rate) if total_records and rate > 0 else "unknown"
                        self.log(
                            (
                                f"{label} progress: records={seen}/{total_records or '?'}, "
                                f"shard={shard_index}/{len(shards)}, avg={rate:.1f} records/s, ETA {eta}"
                            ),
                            "debug",
                        )
                    yield record
            except (OSError, json.JSONDecodeError) as exc:
                self.log(f"{label}: failed to read shard {shard_path}: {exc}", "warning")
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
        return checkpoint_path.parent / f"{checkpoint_path.stem}_candidates" / "manifest.json"

    def _overlay_manifest_for_checkpoint(self, checkpoint_path: Path) -> Path:
        return checkpoint_path.parent / f"{checkpoint_path.stem}_overlay" / "manifest.json"

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
        self._candidate_records_enabled = bool(enabled and checkpoint_path is not None)
        self._candidate_manifest_path = None
        self._candidate_shard_dir = None
        self._candidate_shards = []
        self._candidate_total_records = 0
        if not self._candidate_records_enabled or checkpoint_path is None:
            return
        self._candidate_manifest_path = self._candidate_manifest_for_checkpoint(checkpoint_path)
        self._candidate_shard_dir = self._candidate_manifest_path.parent
        self._candidate_shard_dir.mkdir(parents=True, exist_ok=True)
        if not append_existing:
            for path in self._candidate_shard_dir.glob("shard-*.jsonl*"):
                try:
                    path.unlink()
                except OSError as exc:
                    self.log(f"Failed to remove stale candidate shard {path}: {exc}", "warning")
            if self._candidate_manifest_path.exists():
                try:
                    self._candidate_manifest_path.unlink()
                except OSError as exc:
                    self.log(f"Failed to remove stale candidate manifest {self._candidate_manifest_path}: {exc}", "warning")
        manifest = self._load_audit_manifest(self._candidate_manifest_path) if append_existing else {}
        self._candidate_shards = list(manifest.get("shards") or [])
        self._candidate_total_records = int(manifest.get("total_records", 0) or 0)
        manifest_compression = manifest.get("compression")
        if manifest_compression:
            self._audit_shard_compression = self._resolve_text_compression(manifest_compression)
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
        self._candidate_current_writer = self._open_jsonl_writer(shard_path, self._audit_shard_compression)

    def _append_candidate_records(self, records: List[Dict[str, Any]]) -> None:
        if not records or not self._candidate_records_enabled:
            return
        for record in records:
            if (
                self._candidate_current_writer is None
                or self._candidate_current_shard is None
                or int(self._candidate_current_shard.get("records", 0)) >= self._audit_shard_records
            ):
                self._start_new_candidate_shard()
            if self._candidate_current_writer is None or self._candidate_current_shard is None:
                return
            self._candidate_current_writer.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":"), default=_json_default)
            )
            self._candidate_current_writer.write("\n")
            self._candidate_current_shard["records"] = int(self._candidate_current_shard.get("records", 0)) + 1
            self._candidate_total_records += 1

    def _write_candidate_manifest(self) -> Optional[Path]:
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

    def _read_candidate_records_from_manifest(self, manifest_path: Path) -> List[Dict[str, Any]]:
        return list(self._iter_records_from_manifest(manifest_path, "Checkpoint candidate restore"))

    def _selector_evidence_items_for_record(self, record: Dict[str, Any]) -> List[Dict[str, float]]:
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
                self.log(f"Failed to compact selector evidence for audit record: {exc}", "debug")
                return []
        return []

    def _candidate_row_from_explanation_record(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
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
            (self._load_audit_manifest(audit_manifest_path) or {}).get("compression", self._audit_shard_compression)
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
                    self.log(f"Failed to remove stale candidate shard {stale_path}: {exc}", "warning")
            stale_manifest = candidate_dir / "manifest.json"
            if stale_manifest.exists():
                try:
                    stale_manifest.unlink()
                except OSError as exc:
                    self.log(f"Failed to remove stale candidate manifest {stale_manifest}: {exc}", "warning")
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
        try:
            return str(path.relative_to(checkpoint_path.parent))
        except ValueError:
            return str(path)

    def _audit_manifest_for_checkpoint(self, checkpoint_path: Path) -> Path:
        return checkpoint_path.parent / f"{checkpoint_path.stem}_audit" / "manifest.json"

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
                    self.log(f"Failed to remove stale audit shard {path}: {exc}", "warning")
            if self._audit_manifest_path.exists():
                try:
                    self._audit_manifest_path.unlink()
                except OSError as exc:
                    self.log(f"Failed to remove stale audit manifest {self._audit_manifest_path}: {exc}", "warning")
        manifest = self._load_audit_manifest(self._audit_manifest_path) if append_existing else {}
        self._audit_shards = list(manifest.get("shards") or [])
        self._audit_total_records = int(manifest.get("total_records", 0) or 0)
        manifest_compression = manifest.get("compression")
        if manifest_compression:
            self._audit_shard_compression = self._resolve_text_compression(manifest_compression)

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
        self._audit_current_writer = self._open_jsonl_writer(shard_path, self._audit_shard_compression)

    def _append_audit_records(self, records: List[Dict[str, Any]]) -> None:
        if not records or not self._audit_shards_enabled:
            return
        for record in records:
            if (
                self._audit_current_writer is None
                or self._audit_current_shard is None
                or int(self._audit_current_shard.get("records", 0)) >= self._audit_shard_records
            ):
                self._start_new_audit_shard()
            if self._audit_current_writer is None or self._audit_current_shard is None:
                return
            self._audit_current_writer.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":"), default=_json_default)
            )
            self._audit_current_writer.write("\n")
            self._audit_current_shard["records"] = int(self._audit_current_shard.get("records", 0)) + 1
            self._audit_total_records += 1

    def _write_audit_manifest(self) -> Optional[Path]:
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

    def _read_audit_records_from_manifest(self, manifest_path: Path) -> List[Dict[str, Any]]:
        manifest = self._load_audit_manifest(manifest_path)
        if not manifest:
            return []
        compression = self._resolve_text_compression(manifest.get("compression", "none"))
        records: List[Dict[str, Any]] = []
        for shard in manifest.get("shards") or []:
            rel_path = shard.get("path")
            if not rel_path:
                continue
            shard_path = (manifest_path.parent / str(rel_path)).resolve()
            try:
                records.extend(self._read_jsonl_records(shard_path, compression=compression))
            except (OSError, json.JSONDecodeError) as exc:
                self.log(f"Failed to read audit shard {shard_path}: {exc}", "warning")
                return []
        return records

    def has_streamed_explanations(self) -> bool:
        manifest_path = getattr(self, "_audit_manifest_path", None)
        return bool(manifest_path and Path(manifest_path).exists())

    def _load_overlay_lookup(self) -> Dict[Tuple[str, str], Dict[str, Any]]:
        manifest_path = getattr(self, "_overlay_manifest_path", None)
        if not manifest_path or not Path(manifest_path).exists():
            return {}
        lookup: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for record in self._iter_records_from_manifest(Path(manifest_path), "Final overlay load"):
            src = record.get("Src")
            tgt = record.get("Tgt")
            if src is None or tgt is None:
                continue
            lookup[(str(src), str(tgt))] = record
        return lookup

    @staticmethod
    def _merge_overlay_record(record: Dict[str, Any], overlay: Optional[Dict[str, Any]]) -> Dict[str, Any]:
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

        compression = self._resolve_text_compression(manifest.get("compression", "none"))
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

    def _load_additional_models_checkpoint(
        self,
        kind: DatasetMask,
        local_alignment: bool,
        threshold: Optional[float],
        cardinality: Optional[int],
    ) -> Optional[pd.DataFrame]:
        if not bool(getattr(self, "_postprocess_checkpoints_enabled", False)):
            return None
        if not bool(getattr(self, "_additional_model_checkpoint_resume_enabled", True)):
            if not bool(getattr(self, "_additional_model_checkpoint_skip_logged", False)):
                self.log(
                    "Skipping additional-model checkpoint resume because "
                    "resume_additional_model_checkpoints=False.",
                    "debug",
                )
                self._additional_model_checkpoint_skip_logged = True
            return None
        path = self._stage_checkpoint_path(kind, "additional_models", local_alignment, threshold, cardinality)
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError) as exc:
            self.log(f"Failed to load additional-model checkpoint {path}: {exc}", "warning")
            return None
        expected = self._postprocess_fingerprint_payload(kind, local_alignment, threshold, cardinality)
        if payload.get("fingerprint_payload") != expected:
            self.log(f"Ignoring stale additional-model checkpoint {path}.", "warning")
            return None
        if payload.get("complete") is False:
            self.log(f"Ignoring incomplete additional-model checkpoint {path}.", "debug")
            return None
        records = payload.get("candidate_records") or []
        records_path_value = payload.get("candidate_records_path")
        if not records and records_path_value:
            records_path = Path(str(records_path_value))
            if not records_path.is_absolute():
                records_path = (path.parent / records_path).resolve()
            try:
                start = time.perf_counter()
                compression = payload.get("candidate_records_compression", "none")
                records = []
                for record in self._iter_jsonl_records(records_path, compression=compression):
                    records.append(record)
                    if len(records) % 50000 == 0:
                        elapsed = max(1.0e-8, time.perf_counter() - start)
                        self.log(
                            (
                                f"Additional-model checkpoint load progress: "
                                f"records={len(records)}/{payload.get('candidate_records_count', '?')}, "
                                f"avg={len(records) / elapsed:.1f} records/s"
                            ),
                            "debug",
                        )
            except (OSError, json.JSONDecodeError) as exc:
                self.log(f"Failed to load additional-model records {records_path}: {exc}", "warning")
                return None
        if not records:
            return None
        df = pd.DataFrame.from_records(records)
        self._sync_selector_fields_from_candidate_df(df)
        self.log(f"Loaded additional-model checkpoint from {path}", "info")
        return df

    def _write_additional_models_checkpoint(
        self,
        kind: DatasetMask,
        candidate_df: pd.DataFrame,
        local_alignment: bool,
        threshold: Optional[float],
        cardinality: Optional[int],
        log_level: str = "debug",
        complete: bool = True,
    ) -> None:
        if not bool(getattr(self, "_postprocess_checkpoints_enabled", False)):
            return
        if candidate_df.empty or len(getattr(self, "models", [])) <= 1:
            return
        stage = "additional_models" if complete else "additional_models_partial"
        path = self._stage_checkpoint_path(kind, stage, local_alignment, threshold, cardinality)
        compression = self._resolve_text_compression(getattr(self, "_audit_shard_compression", "zstd"))
        records_path = path.with_suffix(self._jsonl_suffix(compression))
        payload = {
            "stage": stage,
            "complete": bool(complete),
            "fingerprint_payload": self._postprocess_fingerprint_payload(kind, local_alignment, threshold, cardinality),
            "candidate_records_path": records_path.name,
            "candidate_records_format": "jsonl",
            "candidate_records_compression": compression,
            "candidate_records_count": int(len(candidate_df)),
        }
        try:
            self._write_jsonl_records_atomic(
                records_path,
                candidate_df.to_dict(orient="records"),
                compression=compression,
            )
            self._write_json_atomic(path, payload)
            self.log(f"Wrote additional-model checkpoint to {path}", log_level)
        except OSError as exc:
            self.log(f"Failed to write additional-model checkpoint {path}: {exc}", "warning")

    def _sync_selector_fields_from_candidate_df(self, df: pd.DataFrame) -> None:
        if not self.results_json or df.empty or "Src" not in df.columns or "Tgt" not in df.columns:
            return
        row_lookup = {(str(row["Src"]), str(row["Tgt"])): row for _, row in df.iterrows()}
        for record in self.results_json:
            row = row_lookup.get((str(record.get("src_iri")), str(record.get("tgt_iri"))))
            if row is None:
                continue
            conf = record.get("confidences") or {}
            for key in [
                "cand_sim",
                "cand_sim_semantic",
                "cand_sim_lexical",
                "S_pair_final",
                "S_select",
                "P_select",
                "selection_margin",
                "selection_entropy",
                "selection_no_match_prob",
                "selection_distinctive",
                "selection_utility",
                "P_rank",
                "P_match",
                "selection_accept_threshold",
                "S_final",
            ]:
                if key in row:
                    conf[key] = float(row.get(key, 0.0))
            record["confidences"] = conf
            pred = record.get("prediction") or {}
            if "selection_abstained" in row:
                pred["selector_abstained"] = bool(row.get("selection_abstained", False))
            if "selection_llm_used" in row:
                pred["selector_llm_used"] = bool(row.get("selection_llm_used", False))
            if "selection_reason" in row:
                pred["selector_reason"] = str(row.get("selection_reason", ""))
            if "selection_winner" in row:
                pred["selector_winner"] = bool(row.get("selection_winner", False))
            record["prediction"] = pred

    @staticmethod
    def _rationale_record_key(record: Dict[str, Any]) -> str:
        return f"{record.get('src_iri', '')}\u241F{record.get('tgt_iri', '')}"

    def _rationale_checkpoint_path(
        self,
        kind: DatasetMask,
        local_alignment: bool,
        threshold: Optional[float],
        cardinality: Optional[int],
    ) -> Path:
        payload = self._postprocess_fingerprint_payload(kind, local_alignment, threshold, cardinality)
        fingerprint = self._hash_checkpoint_fingerprint_payload(payload)
        return (self.checkpoint_dir / f"{kind.name.lower()}_rationales_{fingerprint[:12]}.json").resolve()

    def _load_rationale_checkpoint(
        self,
        kind: DatasetMask,
        local_alignment: bool,
        threshold: Optional[float],
        cardinality: Optional[int],
    ) -> Dict[str, str]:
        if not bool(getattr(self, "_postprocess_checkpoints_enabled", False)):
            return {}
        path = self._rationale_checkpoint_path(kind, local_alignment, threshold, cardinality)
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except FileNotFoundError:
            return {}
        except (OSError, json.JSONDecodeError) as exc:
            self.log(f"Failed to load rationale checkpoint {path}: {exc}", "warning")
            return {}
        expected = self._postprocess_fingerprint_payload(kind, local_alignment, threshold, cardinality)
        if payload.get("fingerprint_payload") != expected:
            self.log(f"Ignoring stale rationale checkpoint {path}.", "warning")
            return {}
        rationales = payload.get("rationales") or {}
        if not isinstance(rationales, dict):
            return {}
        restored = 0
        for record in self.results_json:
            key = self._rationale_record_key(record)
            rationale = rationales.get(key)
            if not rationale:
                continue
            pred = record.get("prediction") or {}
            if not pred.get("llm_rationale"):
                pred["llm_rationale"] = str(rationale)
                record["prediction"] = pred
                restored += 1
        if restored:
            self.log(f"Loaded {restored} rationales from checkpoint {path}", "debug")
        return {str(key): str(value) for key, value in rationales.items() if value}

    def _write_rationale_checkpoint(
        self,
        kind: DatasetMask,
        local_alignment: bool,
        threshold: Optional[float],
        cardinality: Optional[int],
        rationales: Dict[str, str],
    ) -> None:
        if not bool(getattr(self, "_postprocess_checkpoints_enabled", False)):
            return
        if not rationales:
            return
        path = self._rationale_checkpoint_path(kind, local_alignment, threshold, cardinality)
        payload = {
            "stage": "rationales",
            "fingerprint_payload": self._postprocess_fingerprint_payload(kind, local_alignment, threshold, cardinality),
            "rationales": rationales,
        }
        try:
            self._write_json_atomic(path, payload)
            self.log(f"Wrote rationale checkpoint ({len(rationales)} records) to {path}", "debug")
        except OSError as exc:
            self.log(f"Failed to write rationale checkpoint {path}: {exc}", "warning")

    def _maybe_persist_model_cache(self, reason: str, force: bool = False) -> None:
        policy = str(getattr(self, "_cache_persist_policy", "checkpoint") or "checkpoint").lower()
        if policy == "never":
            return
        if reason == "checkpoint" and policy != "checkpoint":
            if reason not in self._cache_persist_skip_logged:
                self.log(
                    f"Skipping model cache persistence during {reason}; cache_persist_policy={policy}.",
                    "debug",
                )
                self._cache_persist_skip_logged.add(reason)
            return
        if reason == "finalize" and policy not in {"checkpoint", "finalize"}:
            return
        models = getattr(self, "models", None) or [getattr(self, "model", None)]
        for model in models:
            if model is None or not hasattr(model, "persist_caches"):
                continue
            log_level = "debug"
            model_name = model.__class__.__name__
            start = time.perf_counter()
            self.log(
                f"Persisting model cache for {model_name} (reason={reason}, force={bool(force)}).",
                log_level,
            )
            try:
                model.persist_caches(force=force, reason=reason)
            except Exception as exc:  # noqa: BLE001
                self.log(f"Failed to persist model cache during {reason}: {exc}", "warning")
                continue
            elapsed = max(0.0, time.perf_counter() - start)
            self.log(
                (
                    f"Persisted model cache for {model_name} "
                    f"in {_format_duration(elapsed)} (reason={reason})."
                ),
                log_level,
            )

    def _record_llm_calibration(self, calib: Optional[Dict[str, Any]]) -> None:
        if not calib or self._llm_calibration_report is None:
            return
        messages = calib.get("messages") or []
        for msg in messages:
            if not msg:
                continue
            if msg not in self._llm_calibration_messages_logged:
                self._llm_calibration_messages_logged.add(msg)
                self._llm_calibration_report["messages"].append(msg)
                self.log(msg, level="debug")
        learned = calib.get("learned")
        if learned:
            self._llm_calibration_report["learned"] = learned
        pending = calib.get("pending_total_samples")
        if pending is not None:
            self._llm_calibration_report["pending_total_samples"] = pending

    def _annotate_final_prediction_records(
        self,
        preds: List[EntityMapping],
        threshold: Optional[float],
        local_alignment: bool,
    ) -> None:
        if not self.results_json:
            return
        kept_pairs = {(m.head, m.tail) for m in preds}
        threshold_value = float(threshold) if threshold is not None else None
        for rec in self.results_json:
            pred = rec.get("prediction") or {}
            conf = rec.get("confidences") or {}
            src = rec.get("src_iri")
            tgt = rec.get("tgt_iri")
            s_final = conf.get("S_final")
            threshold_positive = False
            if s_final is not None and threshold_value is not None:
                threshold_positive = float(s_final) >= threshold_value
            elif s_final is not None and threshold_value is None:
                threshold_positive = True
            saved_alignment_member = (src, tgt) in kept_pairs
            pred["threshold_positive"] = bool(threshold_positive)
            pred["saved_alignment_member"] = bool(saved_alignment_member)
            if local_alignment:
                rationale_positive = bool(threshold_positive)
            else:
                rationale_positive = bool(saved_alignment_member)
            pred["rationale_positive"] = rationale_positive
            pred["rationale_decision_label"] = "Match" if rationale_positive else "No match"
            rec["prediction"] = pred

    def _annotate_candidate_dataframe(
        self,
        candidate_df: pd.DataFrame,
        preds: List[EntityMapping],
        threshold: Optional[float],
        local_alignment: bool,
    ) -> None:
        if candidate_df.empty or "Src" not in candidate_df.columns or "Tgt" not in candidate_df.columns:
            return
        kept_pairs = {(m.head, m.tail) for m in preds}
        threshold_value = float(threshold) if threshold is not None else None
        saved_values = []
        threshold_values = []
        rationale_values = []
        labels = []
        for _, row in candidate_df.iterrows():
            src = str(row.get("Src"))
            tgt = str(row.get("Tgt"))
            s_final = row.get("S_final")
            threshold_positive = False
            if s_final is not None and threshold_value is not None:
                threshold_positive = float(s_final) >= threshold_value
            elif s_final is not None and threshold_value is None:
                threshold_positive = True
            saved = (src, tgt) in kept_pairs
            rationale_positive = bool(threshold_positive) if local_alignment else bool(saved)
            saved_values.append(bool(saved))
            threshold_values.append(bool(threshold_positive))
            rationale_values.append(bool(rationale_positive))
            labels.append("Match" if rationale_positive else "No match")
        candidate_df["saved_alignment_member"] = saved_values
        candidate_df["threshold_positive"] = threshold_values
        candidate_df["rationale_positive"] = rationale_values
        candidate_df["rationale_decision_label"] = labels

    @staticmethod
    def _row_scalar(row: pd.Series, key: str, default: Any = None) -> Any:
        if key not in row:
            return default
        value = row.get(key)
        if value is None:
            return default
        try:
            missing = pd.isna(value)
            if isinstance(missing, bool) and missing:
                return default
        except (TypeError, ValueError):
            pass
        return value

    def _should_generate_final_rationales(self) -> bool:
        model = getattr(self, "model", None)
        if model is None or not hasattr(model, "generate_final_rationales_for_records"):
            return False
        return bool(getattr(model, "generate_llm_rationales", True))

    def _ensure_compact_rationale_records(self, candidate_df: pd.DataFrame) -> bool:
        if self.results_json or candidate_df.empty or not self._should_generate_final_rationales():
            return False
        records: List[Dict[str, Any]] = []
        confidence_keys = [
            "S_pair_final",
            "S_select",
            "P_select",
            "selection_margin",
            "selection_entropy",
            "selection_no_match_prob",
            "selection_evidence_support",
            "selection_distinctive",
            "selection_utility",
            "P_rank",
            "P_match",
            "selection_accept_threshold",
            "selection_target_conflict_enabled",
            "selection_target_cardinality",
            "S_final",
            "s_label",
            "s_label_star",
            "s_ctx",
            "S_lctx",
            "S_base",
            "S_struct",
            "s_hier",
            "s_sim",
            "s_diff",
            "s_attr",
            "cand_sim",
            "cand_sim_semantic",
            "cand_sim_lexical",
            "p_llm",
        ]
        weight_keys = ["w_c", "w_struct", "w_i", "U", "U_ind", "U_dis"]
        importance_keys = [
            "I_label",
            "I_struct",
            "I_ctx",
            "I_hier",
            "I_sim",
            "I_diff",
            "I_attr",
            "I_llm",
        ]
        for _, row in candidate_df.iterrows():
            src = self._row_scalar(row, "Src")
            tgt = self._row_scalar(row, "Tgt")
            if src is None or tgt is None:
                continue
            confidences: Dict[str, Any] = {}
            for key in confidence_keys:
                value = self._row_scalar(row, key)
                if value is None:
                    continue
                if key == "selection_target_conflict_enabled":
                    confidences[key] = bool(value)
                elif key == "selection_target_cardinality":
                    confidences[key] = int(value or 0)
                else:
                    try:
                        confidences[key] = float(value)
                    except (TypeError, ValueError):
                        continue
            weights: Dict[str, float] = {}
            for key in weight_keys:
                value = self._row_scalar(row, key)
                if value is None:
                    continue
                try:
                    weights[key] = float(value)
                except (TypeError, ValueError):
                    continue
            importances: Dict[str, float] = {}
            for key in importance_keys:
                value = self._row_scalar(row, key)
                if value is None:
                    continue
                try:
                    importances[key] = float(value)
                except (TypeError, ValueError):
                    continue
            prediction: Dict[str, Any] = {}
            ground_truth = self._row_scalar(row, "ground_truth")
            if ground_truth is not None:
                prediction["ground_truth"] = ground_truth
            for key in [
                "threshold_positive",
                "saved_alignment_member",
                "rationale_positive",
                "rationale_decision_label",
                "selection_abstained",
                "selection_llm_used",
                "selection_reason",
                "selection_winner",
            ]:
                value = self._row_scalar(row, key)
                if value is None:
                    continue
                if key in {
                    "threshold_positive",
                    "saved_alignment_member",
                    "rationale_positive",
                    "selection_abstained",
                    "selection_llm_used",
                    "selection_winner",
                }:
                    prediction[key] = bool(value)
                else:
                    prediction[key] = str(value)
            if "selection_reason" in prediction:
                prediction["selector_reason"] = prediction.pop("selection_reason")
            if "selection_abstained" in prediction:
                prediction["selector_abstained"] = prediction.pop("selection_abstained")
            if "selection_llm_used" in prediction:
                prediction["selector_llm_used"] = prediction.pop("selection_llm_used")
            if "selection_winner" in prediction:
                prediction["selector_winner"] = prediction.pop("selection_winner")
            record = {
                "src_iri": str(src),
                "tgt_iri": str(tgt),
                "selected_labels": {
                    "source": str(self._row_scalar(row, "src_label_text", "") or ""),
                    "target": str(self._row_scalar(row, "tgt_label_text", "") or ""),
                },
                "llm_summaries": {
                    "source": str(self._row_scalar(row, "src_context_text", "") or ""),
                    "target": str(self._row_scalar(row, "tgt_context_text", "") or ""),
                },
                "llm_pair_brief": str(self._row_scalar(row, "llm_pair_brief", "") or ""),
                "confidences": confidences,
                "weights": weights,
                "importances": importances,
                "prediction": prediction,
            }
            records.append(record)
        if not records:
            return False
        self.results_json.extend(records)
        self.log(
            (
                f"Prepared {len(records)} compact records for final rationale generation "
                "without loading full audit JSON into memory."
            ),
            "debug",
        )
        return True

    @staticmethod
    def _result_record_lookup(records: List[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
        lookup: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for record in records or []:
            src = record.get("src_iri")
            tgt = record.get("tgt_iri")
            if src is None or tgt is None:
                continue
            lookup[(str(src), str(tgt))] = record
        return lookup

    def _overlay_record_from_candidate_row(
        self,
        row: pd.Series,
        result_record_lookup: Optional[Dict[Tuple[str, str], Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        src = str(row.get("Src"))
        tgt = str(row.get("Tgt"))
        confidence_keys = [
            "S_pair_final",
            "S_select",
            "P_select",
            "selection_margin",
            "selection_entropy",
            "selection_no_match_prob",
            "selection_evidence_support",
            "selection_distinctive",
            "selection_utility",
            "P_rank",
            "P_match",
            "selection_accept_threshold",
            "selection_target_conflict_enabled",
            "selection_target_cardinality",
            "S_final",
            "cand_sim",
            "cand_sim_semantic",
            "cand_sim_lexical",
        ]
        confidences: Dict[str, Any] = {}
        for key in confidence_keys:
            if key not in row or pd.isna(row.get(key)):
                continue
            value = row.get(key)
            if key in {"selection_target_conflict_enabled"}:
                confidences[key] = bool(value)
            elif key in {"selection_target_cardinality"}:
                confidences[key] = int(value or 0)
            else:
                confidences[key] = float(value)
        prediction_keys = [
            "threshold_positive",
            "saved_alignment_member",
            "rationale_positive",
            "rationale_decision_label",
            "selection_abstained",
            "selection_llm_used",
            "selection_reason",
            "selection_winner",
        ]
        prediction: Dict[str, Any] = {}
        for key in prediction_keys:
            if key not in row or pd.isna(row.get(key)):
                continue
            value = row.get(key)
            if key in {
                "threshold_positive",
                "saved_alignment_member",
                "rationale_positive",
                "selection_abstained",
                "selection_llm_used",
                "selection_winner",
            }:
                prediction[key] = bool(value)
            else:
                prediction[key] = str(value)
        if "selection_reason" in prediction:
            prediction["selector_reason"] = prediction.pop("selection_reason")
        if "selection_abstained" in prediction:
            prediction["selector_abstained"] = prediction.pop("selection_abstained")
        if "selection_llm_used" in prediction:
            prediction["selector_llm_used"] = prediction.pop("selection_llm_used")
        if "selection_winner" in prediction:
            prediction["selector_winner"] = prediction.pop("selection_winner")
        overlay: Dict[str, Any] = {
            "Src": src,
            "Tgt": tgt,
            "confidences": confidences,
            "prediction": prediction,
        }
        source_record = (result_record_lookup or {}).get((src, tgt))
        if source_record:
            source_prediction = source_record.get("prediction") or {}
            if source_prediction.get("llm_rationale"):
                prediction["llm_rationale"] = str(source_prediction.get("llm_rationale"))
            backend_usage = source_record.get("backend_usage")
            if isinstance(backend_usage, dict) and backend_usage:
                overlay["backend_usage"] = backend_usage
            models = source_record.get("models")
            if isinstance(models, dict) and models:
                overlay["models"] = models
        return overlay

    def _write_final_overlay(
        self,
        checkpoint_path: Path,
        candidate_df: pd.DataFrame,
        preds: List[EntityMapping],
        threshold: Optional[float],
        local_alignment: bool,
        compression: str,
        records_per_shard: int,
    ) -> Optional[Path]:
        if candidate_df.empty:
            return None
        self._annotate_candidate_dataframe(candidate_df, preds, threshold, local_alignment)
        overlay_manifest_path = self._overlay_manifest_for_checkpoint(checkpoint_path)
        overlay_dir = overlay_manifest_path.parent
        overlay_dir.mkdir(parents=True, exist_ok=True)
        for path in overlay_dir.glob("shard-*.jsonl*"):
            try:
                path.unlink()
            except OSError as exc:
                self.log(f"Failed to remove stale overlay shard {path}: {exc}", "warning")
        resolved = self._resolve_text_compression(compression)
        shard_limit = max(1, int(records_per_shard or 50000))
        shards: List[Dict[str, Any]] = []
        total = 0
        start = time.perf_counter()
        writer = None
        current_shard: Optional[Dict[str, Any]] = None
        result_record_lookup = self._result_record_lookup(self.results_json)
        try:
            for _, row in candidate_df.iterrows():
                if writer is None or current_shard is None or int(current_shard.get("records", 0)) >= shard_limit:
                    if writer is not None:
                        writer.close()
                    shard_name = f"shard-{len(shards):06d}{self._jsonl_suffix(resolved)}"
                    current_shard = {"path": shard_name, "records": 0}
                    shards.append(current_shard)
                    writer = self._open_jsonl_writer(overlay_dir / shard_name, resolved)
                writer.write(
                    json.dumps(
                        self._overlay_record_from_candidate_row(row, result_record_lookup),
                        ensure_ascii=False,
                        separators=(",", ":"),
                        default=_json_default,
                    )
                )
                writer.write("\n")
                current_shard["records"] = int(current_shard.get("records", 0)) + 1
                total += 1
                if total % 50000 == 0:
                    elapsed = max(1.0e-8, time.perf_counter() - start)
                    self.log(
                        (
                            f"Final overlay write progress: records={total}/{len(candidate_df)}, "
                            f"avg={total / elapsed:.1f} records/s"
                        ),
                        "debug",
                    )
        finally:
            if writer is not None:
                writer.close()
        payload = {
            "version": 1,
            "format": "jsonl",
            "compression": resolved,
            "records_per_shard": shard_limit,
            "total_records": total,
            "shards": shards,
        }
        self._write_json_atomic(overlay_manifest_path, payload)
        self._overlay_manifest_path = overlay_manifest_path
        elapsed = max(0.0, time.perf_counter() - start)
        self.log(
            (
                f"Wrote final overlay for {total} records to {overlay_manifest_path} "
                f"in {_format_duration(elapsed)}"
            ),
            "info",
        )
        return overlay_manifest_path

    def _generate_final_rationales(
        self,
        log_every: int = 10,
        kind: DatasetMask = DatasetMask.inference,
        local_alignment: bool = False,
        threshold: Optional[float] = None,
        cardinality: Optional[int] = None,
    ) -> None:
        if not self.results_json:
            return
        model = getattr(self, "model", None)
        if model is None or not hasattr(model, "generate_final_rationales_for_records"):
            return
        if not bool(getattr(model, "generate_llm_rationales", True)):
            return
        rationale_checkpoint = self._load_rationale_checkpoint(kind, local_alignment, threshold, cardinality)
        progress_state: Dict[str, Any] = {
            "started": False,
            "start_time": None,
            "last_logged_uncached": 0,
            "last_saved": len(rationale_checkpoint),
            "interval_uncached_records": 0,
            "cached_records": 0,
            "uncached_records": 0,
            "uncached_unique_prompts": 0,
            "backend": None,
            "model": None,
            "concurrency": None,
        }

        def _progress_callback(event: Dict[str, Any]) -> None:
            if not event or event.get("stage") != "rationale":
                return
            event_type = str(event.get("event", ""))
            if event_type == "start":
                start_time = time.perf_counter()
                progress_state["started"] = True
                progress_state["start_time"] = start_time
                progress_state["cached_records"] = int(event.get("cached_records", 0) or 0)
                progress_state["uncached_records"] = int(event.get("uncached_records", 0) or 0)
                progress_state["uncached_unique_prompts"] = int(event.get("uncached_unique_prompts", 0) or 0)
                progress_state["backend"] = event.get("backend")
                progress_state["model"] = event.get("model")
                progress_state["concurrency"] = int(event.get("concurrency", 0) or 0)
                concurrency = max(1, progress_state["concurrency"] or 1)
                progress_state["interval_uncached_records"] = max(1, int(log_every)) * concurrency
                self.log(
                    (
                        "Rationale stage started: "
                        f"records={int(event.get('total_records', 0) or 0)}, "
                        f"uncached_records={progress_state['uncached_records']}, "
                        f"uncached_unique_prompts={progress_state['uncached_unique_prompts']}, "
                        f"cached_records={progress_state['cached_records']}, "
                        f"backend={progress_state['backend']}, "
                        f"model={progress_state['model']}, "
                        f"concurrency={progress_state['concurrency'] or 1}."
                    ),
                    level="info",
                )
                return
            if event_type != "progress" or not progress_state["started"]:
                return

            total_uncached = int(event.get("total_uncached_records", 0) or 0)
            completed_uncached = int(event.get("completed_uncached_records", 0) or 0)
            if total_uncached <= 0:
                return
            last_logged = int(progress_state["last_logged_uncached"] or 0)
            interval = int(progress_state["interval_uncached_records"] or 1)
            if completed_uncached < total_uncached and (completed_uncached - last_logged) < interval:
                return

            progress_state["last_logged_uncached"] = completed_uncached
            elapsed = max(1e-8, time.perf_counter() - float(progress_state["start_time"]))
            rate = completed_uncached / elapsed
            remaining = max(0, total_uncached - completed_uncached)
            eta = _format_duration(remaining / rate) if rate > 0 else _format_duration(0.0)
            avg_seconds = elapsed / max(1, completed_uncached)
            self.log(
                (
                    "Rationale progress: "
                    f"uncached_records={completed_uncached}/{total_uncached}, "
                    f"unique_prompts={int(event.get('completed_unique_prompts', 0) or 0)}/"
                    f"{int(event.get('total_unique_prompts', 0) or 0)}, "
                    f"cached_records={int(event.get('cached_records', 0) or 0)}, "
                    f"avg={avg_seconds:.2f}s/record, ETA {eta}"
                ),
                level="info",
            )

        pending_indices: List[int] = []
        pending_records: List[Dict[str, Any]] = []
        for rec_idx, record in enumerate(self.results_json):
            pred = record.get("prediction") or {}
            if pred.get("llm_rationale"):
                continue
            pending_indices.append(rec_idx)
            pending_records.append(record)
        if not pending_records:
            if rationale_checkpoint:
                self._write_rationale_checkpoint(kind, local_alignment, threshold, cardinality, rationale_checkpoint)
            return

        def _apply_rationale(record_idx: int, rationale: str, rationale_meta: Optional[Dict[str, Any]] = None) -> None:
            rec = self.results_json[record_idx]
            pred = rec.get("prediction") or {}
            pred["llm_rationale"] = rationale
            rec["prediction"] = pred
            if rationale_meta is not None:
                backend_usage = rec.get("backend_usage") or {}
                backend_usage["rationale"] = dict(rationale_meta)
                rec["backend_usage"] = backend_usage
            self._sync_record_model_usage(rec)
            if rationale:
                rationale_checkpoint[self._rationale_record_key(rec)] = rationale

        def _completion_callback(event: Dict[str, Any]) -> None:
            if not event or event.get("stage") != "rationale" or event.get("event") != "completion":
                return
            rationale = str(event.get("rationale") or "")
            rationale_meta = getattr(model, "_last_rationale_backend_meta", {}) or {}
            for pending_idx in event.get("indices") or []:
                try:
                    original_idx = pending_indices[int(pending_idx)]
                except (TypeError, ValueError, IndexError):
                    continue
                _apply_rationale(original_idx, rationale, rationale_meta)
            completed = len(rationale_checkpoint)
            interval = max(1, int(progress_state.get("interval_uncached_records") or log_every or 1))
            last_saved = int(progress_state.get("last_saved") or 0)
            if completed - last_saved >= interval:
                self._write_rationale_checkpoint(kind, local_alignment, threshold, cardinality, rationale_checkpoint)
                if hasattr(model, "persist_caches"):
                    try:
                        model.persist_caches(force=True, reason="rationale_checkpoint")
                    except Exception as exc:  # noqa: BLE001
                        self.log(f"Failed to persist model cache during rationale checkpoint: {exc}", "warning")
                progress_state["last_saved"] = completed

        rationale_sig = inspect.signature(model.generate_final_rationales_for_records)
        accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in rationale_sig.parameters.values())
        rationale_kwargs: Dict[str, Any] = {"progress_callback": _progress_callback}
        if accepts_var_kw or "completion_callback" in rationale_sig.parameters:
            rationale_kwargs["completion_callback"] = _completion_callback
        rationales = model.generate_final_rationales_for_records(pending_records, **rationale_kwargs)
        rationale_meta = getattr(model, "_last_rationale_backend_meta", {}) or {}
        for original_idx, rationale in zip(pending_indices, rationales):
            _apply_rationale(original_idx, rationale, rationale_meta)
        self._write_rationale_checkpoint(kind, local_alignment, threshold, cardinality, rationale_checkpoint)
        if progress_state["started"]:
            elapsed = max(0.0, time.perf_counter() - float(progress_state["start_time"]))
            duration = _format_duration(elapsed)
            uncached_records = int(progress_state["uncached_records"] or 0)
            throughput = (uncached_records / elapsed) if elapsed > 1e-8 and uncached_records > 0 else 0.0
            avg_seconds = (elapsed / uncached_records) if uncached_records > 0 else 0.0
            self.log(
                (
                    "Rationale stage completed: "
                    f"records={len(pending_records)}, "
                    f"uncached_records={uncached_records}, "
                    f"cached_records={int(progress_state['cached_records'] or 0)}, "
                    f"duration={duration}, "
                    f"throughput={throughput:.2f} uncached records/s, "
                    f"avg={avg_seconds:.2f}s/uncached record"
                ),
                level="info",
            )

    @staticmethod
    def _sync_record_model_usage(rec: Dict[str, Any]) -> None:
        models = dict(rec.get("models") or {})
        backend_usage = rec.get("backend_usage") or {}
        summary_model = (backend_usage.get("summary") or {}).get("model")
        decision_model = (backend_usage.get("decision") or {}).get("model")
        rationale_model = (backend_usage.get("rationale") or {}).get("model")
        models["llm_summary_model"] = summary_model
        models["llm_decision_model"] = decision_model
        models["llm_rationale_model"] = rationale_model
        unique_models: List[str] = []
        for name in (summary_model, decision_model, rationale_model):
            if name and name not in unique_models:
                unique_models.append(name)
        if not unique_models:
            pass
        elif len(unique_models) == 1:
            models["llm_model"] = unique_models[0]
        else:
            models["llm_model"] = "multiple"
        rec["models"] = models

    def _finalize_llm_calibration(self) -> None:
        model = getattr(self, "model", None)
        if model is None or not hasattr(model, "finalize_llm_calibration"):
            return
        payload = model.finalize_llm_calibration()
        if payload:
            self._record_llm_calibration(payload)

    @staticmethod
    def _summarize_label(labels: Optional[List[str]]) -> str:
        if not labels:
            return ""
        return " | ".join(labels[:2])

    @staticmethod
    def _summarize_context(ctx: Optional[List[str]]) -> str:
        if not ctx:
            return ""
        snippet = " ".join(ctx[:2])
        return snippet[:512]

    @torch.no_grad()
    def predict(
        self,
        kind: DatasetMask = DatasetMask.inference,
        threshold: Optional[float] = 0.7,
        cardinality: Optional[int] = None,
        target_cardinality: Optional[int] = None,
        local_alignment: bool = False,
        batch_size: int = 8,
        num_workers: int = 0,
        log_every: int = 1,
        mixed_precision: bool = False,
        checkpoint_file: Optional[str] = None,
        checkpoint_every: int = 10,
        resume_from_checkpoint: bool = True,
        enable_checkpoints: bool = True,
        resume_additional_model_checkpoints: bool = True,
        allow_rationale_toggle_checkpoint_resume: bool = False,
        audit_shards_enabled: bool = True,
        audit_shard_compression: str = "zstd",
        audit_shard_records: int = 50000,
        checkpoint_payload: str = "compact",
        cache_persist_policy: str = "finalize",
        save_json: bool = False,
        run_progress: Optional[Any] = None,
        **kwargs,
    ) -> Tuple[List[EntityMapping], float]:
        self.dataset.default_kind = kind
        self.model.eval()
        for extra_model in getattr(self, "models", [])[1:]:
            try:
                extra_model.eval()
            except Exception:
                pass
        self.results_json.clear()
        self.results_df = None
        self._llm_summary_stats: Optional[Dict[str, Any]] = None
        self._llm_calibration_messages_logged: Set[str] = set()
        self._llm_calibration_report: Optional[Dict[str, Any]] = None
        self._candidate_rows: List[Dict[str, Any]] = []
        self._last_stage_timings = {}
        self._restored_candidate_rows = []
        self._overlay_manifest_path = None
        if hasattr(self.model, "reset_llm_calibration_tracking"):
            self.model.reset_llm_calibration_tracking()
        if hasattr(self.model, "reset_summary_stats"):
            self.model.reset_summary_stats()
        if hasattr(self.model, "use_llm_calibration"):
            self._llm_calibration_report = {
                "configured": {
                    "enabled": bool(getattr(self.model, "use_llm_calibration", False)),
                    "a": getattr(self.model, "llm_calibration_a", None),
                    "b": getattr(self.model, "llm_calibration_b", None),
                    "info": getattr(self.model, "llm_calibration_info", None),
                },
                "learned": None,
                "messages": [],
                "pending_total_samples": 0,
            }

        checkpoint_enabled = enable_checkpoints
        self._checkpoint_payload_mode = str(checkpoint_payload or "compact").lower()
        if self._checkpoint_payload_mode not in {"compact", "full"}:
            self._checkpoint_payload_mode = "compact"
        self._cache_persist_policy = str(cache_persist_policy or "finalize").lower()
        if self._cache_persist_policy not in {"checkpoint", "finalize", "never"}:
            self._cache_persist_policy = "finalize"
        self._cache_persist_skip_logged = set()
        self._prepare_audit_shards(
            None,
            enabled=False,
            compression=audit_shard_compression,
            records_per_shard=audit_shard_records,
        )
        self._prepare_candidate_shards(
            None,
            enabled=False,
            compression=audit_shard_compression,
            records_per_shard=audit_shard_records,
        )
        self._postprocess_checkpoints_enabled = bool(checkpoint_enabled)
        self._additional_model_checkpoint_resume_enabled = bool(
            checkpoint_enabled and resume_additional_model_checkpoints
        )
        self._additional_model_checkpoint_skip_logged = False
        checkpoint_every = max(1, int(checkpoint_every))

        cp_path: Optional[Path] = None
        restored_examples = 0
        all_mappings: List[Tuple[str, str, float]] = []
        restored_json: List[Dict[str, Any]] = []

        if checkpoint_enabled and resume_from_checkpoint:
            cp_path, restored_mappings, restored_json, restored_examples = self._restore_from_available_checkpoints(
                kind,
                checkpoint_file,
                allow_rationale_toggle_checkpoint_resume=allow_rationale_toggle_checkpoint_resume,
            )
            if restored_examples and cp_path:
                self.log(
                    (
                        f"Resuming from checkpoint {cp_path} with "
                        f"{restored_examples} / {len(self.dataset)} examples already processed."
                    ),
                    level="info",
                )
            all_mappings.extend(restored_mappings)
            if restored_json:
                self.results_json.extend(restored_json)
            restored_candidate_rows = getattr(self, "_restored_candidate_rows", []) or []
            if restored_candidate_rows:
                self._candidate_rows.extend(restored_candidate_rows)

        total_examples = len(self.dataset)
        remaining_examples = max(0, total_examples - restored_examples)

        if remaining_examples == 0:
            if checkpoint_enabled and cp_path is not None:
                self._prepare_audit_shards(
                    cp_path,
                    enabled=bool(audit_shards_enabled),
                    compression=audit_shard_compression,
                    records_per_shard=audit_shard_records,
                    append_existing=True,
                )
                self._prepare_candidate_shards(
                    cp_path,
                    enabled=True,
                    compression=audit_shard_compression,
                    records_per_shard=audit_shard_records,
                    append_existing=True,
                )
            self.log(
                (
                    "Checkpoint already contains predictions for all samples. "
                    "Skipping inference."
                ),
                level="info",
            )
            if run_progress is not None:
                run_progress.finish("Inference", "checkpoint already complete")
            candidate_df = self._build_candidate_dataframe()
            if candidate_df.empty and self.results_json:
                candidate_df = self._build_candidate_dataframe_from_records(self.results_json)
            if not candidate_df.empty:
                n_sources = int(candidate_df["Src"].nunique()) if "Src" in candidate_df.columns else 0
                post_progress_started = False
                if run_progress is not None and len(getattr(self, "models", [])) > 1:
                    run_progress.start("PostInference", f"rows={len(candidate_df)}, sources={n_sources}")
                    post_progress_started = True
                self.log(
                    (
                        "Post-inference processing restored candidate scores: "
                        f"rows={len(candidate_df)}, sources={n_sources}."
                    ),
                    "debug",
                )
                checkpointed_df = self._load_additional_models_checkpoint(
                    kind, local_alignment, threshold, cardinality
                )
                if checkpointed_df is not None:
                    candidate_df = checkpointed_df
                else:
                    candidate_df = self._apply_additional_models(
                        candidate_df,
                        kind=kind,
                        local_alignment=local_alignment,
                        threshold=threshold,
                        cardinality=cardinality,
                        target_cardinality=target_cardinality,
                        results_json=self.results_json,
                        log_every=log_every,
                        run_progress=run_progress,
                    )
                    self._write_additional_models_checkpoint(
                        kind, candidate_df, local_alignment, threshold, cardinality
                    )
                if post_progress_started:
                    run_progress.finish("PostInference", f"rows={len(candidate_df)}")
                all_mappings = list(zip(candidate_df["Src"], candidate_df["Tgt"], candidate_df["S_final"]))
                self._selector_target_conflict_enabled = self._selector_target_conflict_enabled_from_df(candidate_df)
                threshold = self._effective_alignment_threshold(candidate_df, threshold)
            df = pd.DataFrame(all_mappings, columns=["Src", "Tgt", "Scores"])
            preds = EntityMapping.read_table_mappings(df, threshold=threshold, cardinality=cardinality)
            compact_rationale_records = False
            if not candidate_df.empty:
                self._annotate_candidate_dataframe(candidate_df, preds, threshold, local_alignment)
                compact_rationale_records = self._ensure_compact_rationale_records(candidate_df)
            self._annotate_final_prediction_records(preds, threshold=threshold, local_alignment=local_alignment)
            rationale_start = time.perf_counter()
            self._generate_final_rationales(
                log_every=log_every,
                kind=kind,
                local_alignment=local_alignment,
                threshold=threshold,
                cardinality=cardinality,
            )
            rationale_elapsed = time.perf_counter() - rationale_start
            self.results_df = (
                self._make_summary_dataframe(self.results_json)
                if self.results_json and not compact_rationale_records
                else candidate_df
            )
            if not candidate_df.empty and cp_path is not None:
                self._write_final_overlay(
                    cp_path,
                    candidate_df,
                    preds,
                    threshold,
                    local_alignment,
                    compression=audit_shard_compression,
                    records_per_shard=audit_shard_records,
                )
            if checkpoint_enabled and cp_path is not None:
                self._write_checkpoint_state(
                    cp_path,
                    kind,
                    total_examples=total_examples,
                    processed_examples=restored_examples,
                    mappings=all_mappings,
                    results_json=self.results_json,
                )
            self._last_stage_timings = {
                "Alignment.Inference": 0.0,
                "Alignment.PostInference": 0.0,
                "Postprocess.Rationales": rationale_elapsed / 60.0,
            }
            return preds, 0.0

        if checkpoint_enabled:
            cp_path = self._ensure_checkpoint_path(kind, checkpoint_file, cp_path)
            if cp_path is None:
                checkpoint_enabled = False
            elif not resume_from_checkpoint and checkpoint_file:
                self.log(
                    f"Checkpoint file {cp_path} will be overwritten for this run.",
                    level="info",
                )
        if checkpoint_enabled and cp_path is not None:
            self._prepare_audit_shards(
                cp_path,
                enabled=bool(audit_shards_enabled),
                compression=audit_shard_compression,
                records_per_shard=audit_shard_records,
                append_existing=restored_examples > 0,
            )
            self._prepare_candidate_shards(
                cp_path,
                enabled=True,
                compression=audit_shard_compression,
                records_per_shard=audit_shard_records,
                append_existing=restored_examples > 0,
            )

        dataset_for_dl = self.dataset
        if restored_examples > 0:
            dataset_for_dl = Subset(self.dataset, range(restored_examples, total_examples))

        dl = DataLoader(
            dataset_for_dl,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else None,
            collate_fn=_semantic_collate_fn,
        )

        total_batches = len(dl)
        start_time = time.perf_counter()
        processed_examples = restored_examples
        batches_run = 0

        self.log(
            (
                f"Running Semantic Alignment on {remaining_examples} remaining pairs "
                f"({total_examples} total)"
            ),
            "info",
        )

        for step, batch in enumerate(dl, start=1):
            src_iri = batch["src_iri"]
            tgt_iri = batch["tgt_iri"]
            src_labels = batch["src_labels"]
            tgt_labels = batch["tgt_labels"]
            src_ctxs = batch.get("src_contexts", None)
            tgt_ctxs = batch.get("tgt_contexts", None)
            src_ctx_raw = batch.get("src_ctx_raw_triples")
            tgt_ctx_raw = batch.get("tgt_ctx_raw_triples")
            src_ctx_bridges = batch.get("src_ctx_bridge_triples")
            tgt_ctx_bridges = batch.get("tgt_ctx_bridge_triples")
            labels = batch.get("label")
            if isinstance(labels, torch.Tensor):
                labels = labels.tolist()

            with torch.amp.autocast("cuda", enabled=mixed_precision):
                out = self.model.forward(
                    src_iris=src_iri,
                    tgt_iris=tgt_iri,
                    src_label_lists=src_labels,
                    tgt_label_lists=tgt_labels,
                    src_contexts=src_ctxs,
                    tgt_contexts=tgt_ctxs,
                    src_ctx_raw=src_ctx_raw,
                    tgt_ctx_raw=tgt_ctx_raw,
                    src_ctx_bridges=src_ctx_bridges,
                    tgt_ctx_bridges=tgt_ctx_bridges,
                    label=labels,
                )
            self._record_llm_calibration(out.get("llm_calibration"))
            pair_batch_stats = out.get("batch_pair_adaptive_stats") or {}
            if step == 1 and pair_batch_stats:
                src_pool = pair_batch_stats.get("src_pool") or {}
                tgt_pool = pair_batch_stats.get("tgt_pool") or {}
                evidence = pair_batch_stats.get("pair_evidence") or {}
                self.log(
                    (
                        "Pair-adaptive pair context is assembled during inference batches. "
                        f"First batch: pairs={pair_batch_stats.get('pairs', len(src_iri))}, "
                        f"unique src/tgt={pair_batch_stats.get('unique_src', 0)}/{pair_batch_stats.get('unique_tgt', 0)}, "
                        f"entity-cache hits src={pair_batch_stats.get('src_cache_hits', 0)}/{pair_batch_stats.get('unique_src', 0)}, "
                        f"tgt={pair_batch_stats.get('tgt_cache_hits', 0)}/{pair_batch_stats.get('unique_tgt', 0)}; "
                        f"src pools h/o/a={src_pool.get('hier_nonempty', 0)}/{src_pool.get('entities', 0)},"
                        f"{src_pool.get('obj_nonempty', 0)}/{src_pool.get('entities', 0)},"
                        f"{src_pool.get('attr_nonempty', 0)}/{src_pool.get('entities', 0)}; "
                        f"tgt pools h/o/a={tgt_pool.get('hier_nonempty', 0)}/{tgt_pool.get('entities', 0)},"
                        f"{tgt_pool.get('obj_nonempty', 0)}/{tgt_pool.get('entities', 0)},"
                        f"{tgt_pool.get('attr_nonempty', 0)}/{tgt_pool.get('entities', 0)}; "
                        f"selected pair evidence hier/sim/diff/attr={evidence.get('hier_selected', 0)}/{pair_batch_stats.get('pairs', len(src_iri))},"
                        f"{evidence.get('sim_selected', 0)}/{pair_batch_stats.get('pairs', len(src_iri))},"
                        f"{evidence.get('diff_selected', 0)}/{pair_batch_stats.get('pairs', len(src_iri))},"
                        f"{evidence.get('attr_selected', 0)}/{pair_batch_stats.get('pairs', len(src_iri))}."
                    ),
                    "debug",
                )

            # Accumulate mappings
            def _tensor_list(name: str, default: float = 0.0):
                value = out.get(name)
                if isinstance(value, torch.Tensor):
                    return value.detach().cpu().tolist()
                return [default] * len(src_iri)

            s_label_vals = _tensor_list("s_label")
            s_label_star_vals = _tensor_list("s_label_star")
            s_ctx_vals = _tensor_list("s_ctx")
            s_lctx_vals = _tensor_list("S_lctx")
            s_base_vals = _tensor_list("S_base")
            s_struct_vals = _tensor_list("S_struct")
            s_hier_vals = _tensor_list("s_hier")
            s_sim_vals = _tensor_list("s_sim")
            s_diff_vals = _tensor_list("s_diff")
            s_attr_vals = _tensor_list("s_attr")
            q_label_vals = _tensor_list("q_label")
            q_struct_vals = _tensor_list("Q_struct")
            q_hier_vals = _tensor_list("q_hier")
            q_sim_vals = _tensor_list("q_sim")
            q_diff_vals = _tensor_list("q_diff")
            q_attr_vals = _tensor_list("q_attr")
            s_final = _tensor_list("S_final")
            w_c_vals = _tensor_list("w_c")
            w_struct_vals = _tensor_list("w_struct")
            w_i_vals = _tensor_list("w_i")
            u_vals = _tensor_list("U")
            u_ind_vals = _tensor_list("U_ind")
            u_dis_vals = _tensor_list("U_dis")
            p_llm_vals = _tensor_list("p_llm")
            i_label_vals = _tensor_list("I_label")
            i_struct_vals = _tensor_list("I_struct")
            i_ctx_vals = _tensor_list("I_ctx")
            i_hier_vals = _tensor_list("I_hier")
            i_sim_vals = _tensor_list("I_sim")
            i_diff_vals = _tensor_list("I_diff")
            i_attr_vals = _tensor_list("I_attr")
            i_llm_vals = _tensor_list("I_llm")
            llm_pair_briefs = list(out.get("llm_pair_briefs") or [""] * len(src_iri))
            ground_truth = labels or [None] * len(src_iri)
            candidate_start_idx = len(self._candidate_rows)

            for idx, (s, t, score) in enumerate(zip(src_iri, tgt_iri, s_final)):
                all_mappings.append((s, t, float(score)))
                self._candidate_rows.append({
                    "Src": s,
                    "Tgt": t,
                    "ground_truth": ground_truth[idx],
                    "s_label": float(s_label_vals[idx]),
                    "s_label_star": float(s_label_star_vals[idx]),
                    "s_ctx": float(s_ctx_vals[idx]),
                    "S_lctx": float(s_lctx_vals[idx]),
                    "S_base": float(s_base_vals[idx]),
                    "S_struct": float(s_struct_vals[idx]),
                    "s_hier": float(s_hier_vals[idx]),
                    "s_sim": float(s_sim_vals[idx]),
                    "s_diff": float(s_diff_vals[idx]),
                    "s_attr": float(s_attr_vals[idx]),
                    "q_label": float(q_label_vals[idx]),
                    "Q_struct": float(q_struct_vals[idx]),
                    "q_hier": float(q_hier_vals[idx]),
                    "q_sim": float(q_sim_vals[idx]),
                    "q_diff": float(q_diff_vals[idx]),
                    "q_attr": float(q_attr_vals[idx]),
                    "S_final": float(score),
                    "w_c": float(w_c_vals[idx]),
                    "w_struct": float(w_struct_vals[idx]),
                    "w_i": float(w_i_vals[idx]),
                    "U": float(u_vals[idx]),
                    "U_ind": float(u_ind_vals[idx]),
                    "U_dis": float(u_dis_vals[idx]),
                    "p_llm": float(p_llm_vals[idx]),
                    "I_label": float(i_label_vals[idx]),
                    "I_struct": float(i_struct_vals[idx]),
                    "I_ctx": float(i_ctx_vals[idx]),
                    "I_hier": float(i_hier_vals[idx]),
                    "I_sim": float(i_sim_vals[idx]),
                    "I_diff": float(i_diff_vals[idx]),
                    "I_attr": float(i_attr_vals[idx]),
                    "I_llm": float(i_llm_vals[idx]),
                    "llm_pair_brief": llm_pair_briefs[idx],
                    "src_label_text": self._summarize_label(src_labels[idx]),
                    "tgt_label_text": self._summarize_label(tgt_labels[idx]),
                    "src_context_text": self._summarize_context(src_ctxs[idx] if src_ctxs else []),
                    "tgt_context_text": self._summarize_context(tgt_ctxs[idx] if tgt_ctxs else []),
                })

            processed_examples += len(src_iri)
            batches_run += 1

            # Accumulate full JSONs 
            if "explanations" in out:
                explanations = list(out["explanations"])
                for local_idx, record in enumerate(explanations):
                    candidate_idx = candidate_start_idx + local_idx
                    if candidate_idx >= len(self._candidate_rows):
                        continue
                    evidence_items = self._selector_evidence_items_for_record(record)
                    if evidence_items:
                        self._candidate_rows[candidate_idx]["selector_evidence_items"] = evidence_items
                self.results_json.extend(explanations)
                self._append_audit_records(explanations)
            self._append_candidate_records(self._candidate_rows[candidate_start_idx:])

            if (
                checkpoint_enabled
                and cp_path
                and (step % checkpoint_every == 0 or step == total_batches)
            ):
                self._write_checkpoint_state(
                    cp_path,
                    kind,
                    total_examples=total_examples,
                    processed_examples=processed_examples,
                    mappings=all_mappings,
                    results_json=self.results_json,
                )
                self._maybe_persist_model_cache(reason="checkpoint")

            if step % log_every == 0 or step == total_batches:
                elapsed = time.perf_counter() - start_time
                avg_batch_time = elapsed / max(1, step)
                remaining_batches = max(0, total_batches - step)
                remaining_time = remaining_batches * avg_batch_time
                eta_str = _format_duration(remaining_time)
                pair_diag = ""
                if pair_batch_stats:
                    evidence = pair_batch_stats.get("pair_evidence") or {}
                    pair_total = max(1, int(pair_batch_stats.get("pairs", len(src_iri))))
                    pair_diag = (
                        " | pair-adaptive ctx: "
                        f"cache src={pair_batch_stats.get('src_cache_hits', 0)}/{pair_batch_stats.get('unique_src', 0)}, "
                        f"tgt={pair_batch_stats.get('tgt_cache_hits', 0)}/{pair_batch_stats.get('unique_tgt', 0)}; "
                        f"selected hier/sim/diff/attr={evidence.get('hier_selected', 0)}/{pair_total},"
                        f"{evidence.get('sim_selected', 0)}/{pair_total},"
                        f"{evidence.get('diff_selected', 0)}/{pair_total},"
                        f"{evidence.get('attr_selected', 0)}/{pair_total}; "
                        f"struct-active={pair_batch_stats.get('struct_active_pairs', 0)}/{pair_total}; "
                        f"llm-gated={pair_batch_stats.get('llm_gated_pairs', 0)}/{pair_total}"
                    )
                self.log(
                    (
                        f"Batch {step}/{total_batches} done "
                        f"(avg {avg_batch_time:.2f}s/batch, ETA {eta_str})"
                        f"{pair_diag}"
                    ),
                    "debug",
                )
                if run_progress is not None:
                    run_progress.update(
                        "Inference",
                        completed=processed_examples,
                        total=total_examples,
                        detail=f"pairs={processed_examples}/{total_examples}, batches={step}/{total_batches}",
                    )

        self._finalize_llm_calibration()

        inference_elapsed_seconds = time.perf_counter() - start_time
        total_time = inference_elapsed_seconds
        duration_str = _format_duration(total_time)
        new_examples = max(0, processed_examples - restored_examples)
        avg_t = total_time / max(1, new_examples)
        avg_bt = total_time / max(1, batches_run)
        self.log(
            (
                f"Completed {kind.name} in {duration_str} "
                f"(processed {new_examples} new pairs, ~{avg_t:.3f}s/example, ~{avg_bt:.2f}s/batch)"
            ),
            "info",
        )
        if run_progress is not None:
            run_progress.finish("Inference", f"processed={processed_examples}/{total_examples}")
        if hasattr(self.model, "llm_summary_stats"):
            self._llm_summary_stats = self.model.llm_summary_stats()
            stats = self._llm_summary_stats or {}
            requested = stats.get("requested", 0)
            usable = stats.get("usable", 0)
            empty = stats.get("empty", 0)
            if requested:
                empty_pct = (empty / requested) * 100.0
                self.log(
                    (
                        "LLM summary/brief quality: "
                        f"{requested} requested, {usable} usable, {empty} empty (~{empty_pct:.2f}% empty)"
                    ),
                    "info",
                )
            else:
                self.log("LLM summaries/briefs were not requested for this run.", "debug")

        # Normalize LLM importance: if the LLM did not run (p_llm == 0 or no decision),
        # clamp I_llm to 0 so downstream summaries/plots are consistent with behavior.
        for rec in self.results_json or []:
            imps = rec.get("importances") or {}
            if "I_llm" not in imps:
                continue
            conf = rec.get("confidences") or {}
            pred = rec.get("prediction") or {}
            p_llm = conf.get("p_llm")
            llm_decision = pred.get("llm_decision")
            if p_llm is None or float(p_llm) == 0.0 or llm_decision in ("", None):
                imps["I_llm"] = 0.0

        self._maybe_persist_model_cache(reason="finalize", force=True)

        post_inference_start = time.perf_counter()
        post_progress_started = False
        if run_progress is not None and len(getattr(self, "models", [])) > 1:
            run_progress.start("PostInference", "assembling candidate dataframe")
            post_progress_started = True
        self.log("Post-inference processing started: assembling candidate dataframe.", "info")
        candidate_df = self._build_candidate_dataframe()
        if not candidate_df.empty:
            n_sources = int(candidate_df["Src"].nunique()) if "Src" in candidate_df.columns else 0
            if post_progress_started:
                run_progress.update(
                    "PostInference",
                    fraction=0.15,
                    detail=f"rows={len(candidate_df)}, sources={n_sources}",
                    force=True,
                )
            self.log(
                (
                    "Candidate dataframe assembled for post-inference processing: "
                    f"rows={len(candidate_df)}, sources={n_sources}."
                ),
                "debug",
            )
            checkpointed_df = self._load_additional_models_checkpoint(
                kind, local_alignment, threshold, cardinality
            )
            if checkpointed_df is not None:
                candidate_df = checkpointed_df
            else:
                candidate_df = self._apply_additional_models(
                    candidate_df,
                    kind=kind,
                    local_alignment=local_alignment,
                    threshold=threshold,
                    cardinality=cardinality,
                    target_cardinality=target_cardinality,
                    results_json=self.results_json,
                    log_every=log_every,
                    run_progress=run_progress,
                )
                self._write_additional_models_checkpoint(
                    kind, candidate_df, local_alignment, threshold, cardinality
                )
            all_mappings = list(zip(candidate_df["Src"], candidate_df["Tgt"], candidate_df["S_final"]))
            self._selector_target_conflict_enabled = self._selector_target_conflict_enabled_from_df(candidate_df)
            threshold = self._effective_alignment_threshold(candidate_df, threshold)
            if self.results_json:
                # Prefer the richer JSON (includes importances and LLM info) for plotting/summary.
                # Additional models are expected to sync their public metrics back into records.
                self.results_df = self._make_summary_dataframe(self.results_json)
            else:
                self.results_df = candidate_df
        else:
            self.results_df = self._make_summary_dataframe(self.results_json)
        if post_progress_started:
            run_progress.finish("PostInference", f"rows={len(candidate_df)}")

        df_scores = pd.DataFrame(all_mappings, columns=["Src", "Tgt", "Scores"])
        preds = EntityMapping.read_table_mappings(df_scores, threshold=threshold, cardinality=cardinality)
        compact_rationale_records = False
        if not candidate_df.empty:
            self._annotate_candidate_dataframe(candidate_df, preds, threshold, local_alignment)
            compact_rationale_records = self._ensure_compact_rationale_records(candidate_df)
        self._annotate_final_prediction_records(preds, threshold=threshold, local_alignment=local_alignment)
        rationale_start = time.perf_counter()
        self._generate_final_rationales(
            log_every=log_every,
            kind=kind,
            local_alignment=local_alignment,
            threshold=threshold,
            cardinality=cardinality,
        )
        rationale_elapsed_seconds = time.perf_counter() - rationale_start
        post_inference_elapsed_seconds = time.perf_counter() - post_inference_start - rationale_elapsed_seconds
        if self.results_json and not compact_rationale_records:
            self.results_df = self._make_summary_dataframe(self.results_json)
            for rec in self.results_json:
                self._sync_record_model_usage(rec)
        elif not candidate_df.empty:
            self.results_df = candidate_df
        if not candidate_df.empty and checkpoint_enabled and cp_path is not None:
            self._write_final_overlay(
                cp_path,
                candidate_df,
                preds,
                threshold,
                local_alignment,
                compression=audit_shard_compression,
                records_per_shard=audit_shard_records,
            )
        if checkpoint_enabled and cp_path is not None:
            self._write_checkpoint_state(
                cp_path,
                kind,
                total_examples=total_examples,
                processed_examples=processed_examples,
                mappings=all_mappings,
                results_json=self.results_json,
            )
        self._last_stage_timings = {
            "Alignment.Inference": inference_elapsed_seconds / 60.0,
            "Alignment.PostInference": max(0.0, post_inference_elapsed_seconds) / 60.0,
            "Postprocess.Rationales": rationale_elapsed_seconds / 60.0,
        }
        return preds, avg_t

    @staticmethod
    def _effective_alignment_threshold(
        candidate_df: pd.DataFrame,
        threshold: Optional[float],
    ) -> Optional[float]:
        if threshold is None or candidate_df.empty:
            return threshold
        if "selection_accept_threshold" not in candidate_df.columns:
            return threshold
        selected = pd.to_numeric(candidate_df["selection_accept_threshold"], errors="coerce")
        selected = selected[selected.notna() & (selected > 0.0)]
        if selected.empty:
            return threshold
        return float(selected.median())

    @staticmethod
    def _selector_target_conflict_enabled_from_df(candidate_df: pd.DataFrame) -> Optional[bool]:
        if candidate_df.empty or "selection_target_conflict_enabled" not in candidate_df.columns:
            return None
        if "selection_accept_threshold" in candidate_df.columns:
            thresholds = pd.to_numeric(candidate_df["selection_accept_threshold"], errors="coerce")
            if thresholds[thresholds.notna() & (thresholds > 0.0)].empty:
                return None
        values = candidate_df["selection_target_conflict_enabled"]
        if values.empty:
            return None
        normalized = values.astype(str).str.lower().isin({"true", "1", "yes", "y"})
        return bool(normalized.any())

    @property
    def last_stage_timings(self) -> Dict[str, float]:
        return dict(self._last_stage_timings)

    def _build_candidate_dataframe(self) -> pd.DataFrame:
        if not self._candidate_rows:
            return pd.DataFrame()
        df = pd.DataFrame(self._candidate_rows)
        return self._merge_dataset_candidate_columns(df)

    def _build_candidate_dataframe_from_records(self, records: List[Dict[str, Any]]) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for record in records or []:
            src = record.get("src_iri")
            tgt = record.get("tgt_iri")
            if src is None or tgt is None:
                continue
            pred = record.get("prediction") or {}
            row: Dict[str, Any] = {
                "Src": src,
                "Tgt": tgt,
                "ground_truth": pred.get("ground_truth"),
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
            if "S_final" in row:
                rows.append(row)
        if not rows:
            return pd.DataFrame()
        return self._merge_dataset_candidate_columns(pd.DataFrame(rows))

    def _merge_dataset_candidate_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        dataset_df = getattr(self.dataset, "dataframe", None)
        if dataset_df is None or dataset_df.empty:
            return df
        prefixes = (
            "cand_",
            "src_hier",
            "tgt_hier",
            "src_obj",
            "tgt_obj",
            "src_attr",
            "tgt_attr",
            "src_lab",
            "tgt_lab",
        )
        extra_cols = [
            col
            for col in dataset_df.columns
            if col in {"Src", "Tgt"} or any(str(col).startswith(prefix) for prefix in prefixes)
        ]
        if len(extra_cols) <= 2:
            return df
        extra_df = dataset_df[extra_cols].drop_duplicates(subset=["Src", "Tgt"])
        merge_cols = [col for col in extra_df.columns if col not in df.columns or col in {"Src", "Tgt"}]
        if len(merge_cols) <= 2:
            return df
        return df.merge(extra_df[merge_cols], on=["Src", "Tgt"], how="left")

    def _apply_additional_models(
        self,
        df: pd.DataFrame,
        kind: DatasetMask = DatasetMask.inference,
        local_alignment: bool = False,
        threshold: Optional[float] = None,
        cardinality: Optional[int] = None,
        target_cardinality: Optional[int] = None,
        results_json: Optional[List[Dict[str, Any]]] = None,
        log_every: int = 10,
        run_progress: Optional[Any] = None,
    ) -> pd.DataFrame:
        """
        Run any additional models (beyond the first) sequentially.
        Each model is expected to accept the candidate dataframe and return
        either an updated dataframe or a dict containing 'candidate_df'.
        """
        current = df
        if len(getattr(self, "models", [])) <= 1:
            return current
        checkpoint_every_groups = max(50000, max(1, int(log_every)) * 5000)

        def _checkpoint_current(candidate_df: pd.DataFrame) -> None:
            self._write_additional_models_checkpoint(
                kind,
                candidate_df,
                local_alignment,
                threshold,
                cardinality,
                log_level="debug",
                complete=False,
            )

        checkpoint_callback = (
            _checkpoint_current
            if bool(getattr(self, "_postprocess_checkpoints_enabled", False))
            else None
        )
        for idx, extra_model in enumerate(self.models[1:], start=2):
            extra_model.eval()
            model_name = extra_model.__class__.__name__
            model_start = time.perf_counter()
            self.log(
                (
                    f"Running additional model #{idx} ({model_name}) on "
                    f"{len(current)} candidate rows."
                ),
                "info",
            )
            if run_progress is not None:
                run_progress.update(
                    "PostInference",
                    fraction=0.25,
                    detail=f"model={model_name}, rows={len(current)}",
                    force=True,
                )
            sig = inspect.signature(extra_model.forward)
            accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            call_kwargs = {"candidate_df": current}
            if accepts_var_kw or "dataset" in sig.parameters:
                call_kwargs["dataset"] = self.dataset
            if accepts_var_kw or "logger" in sig.parameters:
                call_kwargs["logger"] = getattr(self, "logger", None)
            if accepts_var_kw or "primary_model" in sig.parameters:
                call_kwargs["primary_model"] = self.model
            if accepts_var_kw or "results_json" in sig.parameters:
                call_kwargs["results_json"] = results_json
            if accepts_var_kw or "local_alignment" in sig.parameters:
                call_kwargs["local_alignment"] = local_alignment
            if accepts_var_kw or "threshold" in sig.parameters:
                call_kwargs["threshold"] = threshold
            if accepts_var_kw or "cardinality" in sig.parameters:
                call_kwargs["cardinality"] = cardinality
            if accepts_var_kw or "target_cardinality" in sig.parameters:
                call_kwargs["target_cardinality"] = target_cardinality
            if accepts_var_kw or "log_every" in sig.parameters:
                call_kwargs["log_every"] = log_every
            if accepts_var_kw or "run_progress" in sig.parameters:
                call_kwargs["run_progress"] = run_progress
            if checkpoint_callback is not None and (
                accepts_var_kw or "checkpoint_callback" in sig.parameters
            ):
                call_kwargs["checkpoint_callback"] = checkpoint_callback
            if checkpoint_callback is not None and (
                accepts_var_kw or "checkpoint_every_groups" in sig.parameters
            ):
                call_kwargs["checkpoint_every_groups"] = checkpoint_every_groups
            out = extra_model.forward(**call_kwargs)
            if isinstance(out, pd.DataFrame):
                current = out
            elif isinstance(out, dict) and "candidate_df" in out:
                current = out["candidate_df"]
                extra_json = out.get("results_json")
                if extra_json and extra_json is not self.results_json:
                    self.results_json.extend(extra_json)
                    results_json = self.results_json
            else:
                self.log(
                    f"Additional model #{idx} returned unsupported type {type(out)}; ignoring its output.",
                    level="warning",
                )
            elapsed = max(0.0, time.perf_counter() - model_start)
            self.log(
                (
                    f"Additional model #{idx} ({model_name}) finished in "
                    f"{_format_duration(elapsed)} with {len(current)} candidate rows."
                ),
                "info",
            )
            if run_progress is not None:
                run_progress.update(
                    "PostInference",
                    fraction=0.85,
                    detail=f"model={model_name} finished, rows={len(current)}",
                    force=True,
                )
        return current
