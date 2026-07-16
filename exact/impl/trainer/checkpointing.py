from __future__ import annotations

import hashlib  # noqa: F401
import inspect  # noqa: F401
import json  # noqa: F401
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

from .audit_io import _json_default


class CheckpointingMixin:
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
                    fingerprint = self._hash_checkpoint_fingerprint_payload(
                        fingerprint_payload
                    )
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
        if not isinstance(checkpoint_payload, dict) or not isinstance(
            current_payload, dict
        ):
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
            self.log(
                f"Unable to prepare checkpoint file '{filename}': {exc}",
                level="warning",
            )
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
            self.log(
                f"Unable to list checkpoints in {self.checkpoint_dir}: {exc}",
                level="warning",
            )
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
                and self._checkpoint_matches_rationale_toggle_override(
                    payload_fingerprint
                )
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
                detail_suffix = (
                    f" Mismatch details: {mismatch_details}" if mismatch_details else ""
                )
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

        explanation_index_value = payload.get("explanation_index_path")
        if explanation_index_value:
            explanation_index_path = Path(str(explanation_index_value))
            if not explanation_index_path.is_absolute():
                explanation_index_path = (
                    checkpoint_path.parent / explanation_index_path
                ).resolve()
            if explanation_index_path != self.run_layout.explanation_index_path:
                self.log(
                    f"Ignoring checkpoint '{checkpoint_path.name}': explanation index escapes the run layout.",
                    "warning",
                )
                return [], [], 0
            try:
                store = ExplanationStore(explanation_index_path.parent)
            except (OSError, ValueError) as exc:
                self.log(
                    f"Ignoring checkpoint '{checkpoint_path.name}': invalid explanation store: {exc}",
                    "warning",
                )
                return [], [], 0
            expected_records = int(payload.get("explanation_records_count", 0) or 0)
            if store.record_count < expected_records:
                self.log(
                    (
                        f"Ignoring checkpoint '{checkpoint_path.name}' because its explanation "
                        f"store has {store.record_count} records; expected {expected_records}."
                    ),
                    "warning",
                )
                return [], [], 0
            if store.record_count > expected_records:
                removed_records = store.record_count - expected_records
                store.truncate(expected_records)
                self.log(
                    (
                        "Discarded uncheckpointed explanation suffix: "
                        f"kept={expected_records}, removed={removed_records}."
                    ),
                    "info",
                )
            self._explanation_store = store
            self._audit_manifest_path = store.index_path
            self._audit_total_records = store.record_count
            results_json = list(store.iter_all())
            candidate_rows = []
            for record in results_json:
                row = self._candidate_row_from_explanation_record(record)
                if row is None:
                    continue
                candidate_rows.append(row)
                try:
                    mappings.append(
                        (str(row["Src"]), str(row["Tgt"]), float(row["S_final"]))
                    )
                except (KeyError, TypeError, ValueError):
                    continue
            self._restored_candidate_rows = candidate_rows

        candidate_manifest_value = (
            None
            if explanation_index_value
            else payload.get("candidate_records_manifest_path")
        )
        if candidate_manifest_value:
            candidate_manifest_path = Path(str(candidate_manifest_value))
            if not candidate_manifest_path.is_absolute():
                candidate_manifest_path = (
                    checkpoint_path.parent / candidate_manifest_path
                ).resolve()
            self._candidate_manifest_path = candidate_manifest_path
            candidate_rows = self._read_candidate_records_from_manifest(
                candidate_manifest_path
            )
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

        for rec in [] if explanation_index_value else payload.get("mappings", []):
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
                    mappings.append(
                        (str(row["Src"]), str(row["Tgt"]), float(row["S_final"]))
                    )
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
        if not mappings and results_json:
            candidate_rows = []
            for record in results_json:
                row = self._candidate_row_from_explanation_record(record)
                if row is None:
                    continue
                candidate_rows.append(row)
                try:
                    mappings.append(
                        (str(row["Src"]), str(row["Tgt"]), float(row["S_final"]))
                    )
                except (KeyError, TypeError, ValueError):
                    continue
            self._restored_candidate_rows = candidate_rows
        if not mappings and manifest_path is not None:
            migrated_rows = self._migrate_legacy_audit_checkpoint(
                checkpoint_path,
                payload,
                manifest_path,
            )
            self._restored_candidate_rows = migrated_rows
            for row in migrated_rows:
                try:
                    mappings.append(
                        (str(row["Src"]), str(row["Tgt"]), float(row["S_final"]))
                    )
                except (KeyError, TypeError, ValueError):
                    continue

        if not explanation_index_value and results_json:
            store = getattr(self, "_explanation_store", None)
            if store is not None:
                store.clear()
                store.append(results_json, run_id=getattr(self, "_run_id", None))
                self._audit_manifest_path = store.index_path
                self._audit_total_records = store.record_count

        processed_examples = int(payload.get("processed_examples", len(mappings)))
        if (
            processed_examples > 0
            and not mappings
            and not results_json
            and not self._restored_candidate_rows
        ):
            self.log(
                (
                    f"Ignoring checkpoint '{checkpoint_path.name}' because it records "
                    f"{processed_examples} processed examples but no restorable mappings or candidate rows."
                ),
                "warning",
            )
            return [], [], 0
        timing_payload = payload.get("timing") or {}
        try:
            restored_seconds = float(
                timing_payload.get("inference_seconds_cumulative", 0.0) or 0.0
            )
        except (TypeError, ValueError):
            restored_seconds = 0.0
        if not 0.0 <= restored_seconds < float("inf"):
            restored_seconds = 0.0
        try:
            restored_rate = float(timing_payload.get("examples_per_second_ema"))
        except (TypeError, ValueError):
            restored_rate = None
        if restored_rate is not None and not 0.0 < restored_rate < float("inf"):
            restored_rate = None
        self._restored_inference_seconds_cumulative = restored_seconds
        self._inference_seconds_cumulative = restored_seconds
        self._examples_per_second_ema = restored_rate
        return mappings, results_json, processed_examples

    def _checkpoint_timing_payload(self) -> Dict[str, Optional[float]]:
        cumulative_seconds = self._inference_seconds_cumulative
        if self._inference_session_started_at is not None:
            cumulative_seconds = self._restored_inference_seconds_cumulative + max(
                0.0,
                time.perf_counter() - self._inference_session_started_at,
            )
        return {
            "inference_seconds_cumulative": cumulative_seconds,
            "examples_per_second_ema": self._examples_per_second_ema,
        }

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
        explanation_store = getattr(self, "_explanation_store", None)
        payload = {
            "checkpoint_schema_version": 3 if explanation_store is not None else 2,
            "kind": kind.name,
            "dataset_signature": getattr(
                getattr(self, "_dataset", None), "dataset_signature", None
            ),
            "dataset_fingerprint": getattr(
                getattr(self, "_dataset", None), "cache_fingerprint", None
            ),
            "checkpoint_fingerprint": self._checkpoint_fingerprint,
            "checkpoint_fingerprint_payload": getattr(
                self,
                "_checkpoint_fingerprint_payload",
                self._build_checkpoint_fingerprint_payload(),
            ),
            "total_examples": total_examples,
            "processed_examples": processed_examples,
            "timing": self._checkpoint_timing_payload(),
            "checkpoint_payload": self._checkpoint_payload_mode,
            "mappings_count": len(mappings),
            "results_json_count": len(results_json),
        }
        if explanation_store is not None:
            payload["explanation_index_path"] = self._relative_to_checkpoint(
                explanation_store.index_path,
                checkpoint_path,
            )
            payload["explanation_records_count"] = explanation_store.record_count
        elif audit_manifest_path is not None:
            payload["audit_manifest_path"] = self._relative_to_checkpoint(
                audit_manifest_path, checkpoint_path
            )
        if candidate_manifest_path is not None:
            payload["candidate_records_manifest_path"] = self._relative_to_checkpoint(
                candidate_manifest_path,
                checkpoint_path,
            )
            payload["candidate_records_count"] = int(
                getattr(self, "_candidate_total_records", 0) or 0
            )
        overlay_manifest_path = getattr(self, "_overlay_manifest_path", None)
        if (
            explanation_store is None
            and overlay_manifest_path is not None
            and Path(overlay_manifest_path).exists()
        ):
            payload["final_overlay_manifest_path"] = self._relative_to_checkpoint(
                Path(overlay_manifest_path),
                checkpoint_path,
            )
        if self._checkpoint_payload_mode == "full" or (
            explanation_store is None
            and audit_manifest_path is None
            and candidate_manifest_path is None
        ):
            payload["mappings"] = [
                {"src": s, "tgt": t, "score": score} for s, t, score in mappings
            ]
            payload["results_json"] = results_json

        tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(
                    payload, f, indent=2, ensure_ascii=False, default=_json_default
                )
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
                fingerprint_payload = model.runtime_fingerprint_payload(
                    generate_llm_rationales_override=None
                )
            fingerprint = self._hash_checkpoint_fingerprint_payload(fingerprint_payload)
        elif hasattr(model, "runtime_fingerprint"):
            fingerprint = model.runtime_fingerprint()
        elif hasattr(model, "_cache_fingerprint"):
            fingerprint = getattr(model, "_cache_fingerprint")
        entry: Dict[str, Any] = {
            "class": model.__class__.__name__,
            "fingerprint": fingerprint,
        }
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
        models = [
            model
            for model in (getattr(self, "models", None) or [])
            if model is not None
        ]
        if not models:
            models = [getattr(self, "model", None)]
        return {
            "dataset_signature": getattr(
                getattr(self, "_dataset", None), "dataset_signature", None
            ),
            "kind": kind.name,
            "local_alignment": bool(local_alignment),
            "threshold": threshold,
            "cardinality": cardinality,
            "models": [
                self._model_fingerprint_entry(model)
                for model in models
                if model is not None
            ],
        }

    def _stage_checkpoint_path(
        self,
        kind: DatasetMask,
        stage: str,
        local_alignment: bool,
        threshold: Optional[float],
        cardinality: Optional[int],
    ) -> Path:
        payload = self._postprocess_fingerprint_payload(
            kind, local_alignment, threshold, cardinality
        )
        fingerprint = self._hash_checkpoint_fingerprint_payload(payload)
        return (
            self.checkpoint_dir / f"{kind.name.lower()}_{stage}_{fingerprint[:12]}.json"
        ).resolve()

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
            if not bool(
                getattr(self, "_additional_model_checkpoint_skip_logged", False)
            ):
                self.log(
                    "Skipping additional-model checkpoint resume because "
                    "resume_additional_model_checkpoints=False.",
                    "debug",
                )
                self._additional_model_checkpoint_skip_logged = True
            return None
        path = self._stage_checkpoint_path(
            kind, "additional_models", local_alignment, threshold, cardinality
        )
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError) as exc:
            self.log(
                f"Failed to load additional-model checkpoint {path}: {exc}", "warning"
            )
            return None
        expected = self._postprocess_fingerprint_payload(
            kind, local_alignment, threshold, cardinality
        )
        if payload.get("fingerprint_payload") != expected:
            self.log(f"Ignoring stale additional-model checkpoint {path}.", "warning")
            return None
        if payload.get("complete") is False:
            self.log(
                f"Ignoring incomplete additional-model checkpoint {path}.", "debug"
            )
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
                for record in self._iter_jsonl_records(
                    records_path, compression=compression
                ):
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
                self.log(
                    f"Failed to load additional-model records {records_path}: {exc}",
                    "warning",
                )
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
        path = self._stage_checkpoint_path(
            kind, stage, local_alignment, threshold, cardinality
        )
        compression = self._resolve_text_compression(
            getattr(self, "_audit_shard_compression", "zstd")
        )
        records_path = path.with_suffix(self._jsonl_suffix(compression))
        payload = {
            "stage": stage,
            "complete": bool(complete),
            "fingerprint_payload": self._postprocess_fingerprint_payload(
                kind, local_alignment, threshold, cardinality
            ),
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
            self.log(
                f"Failed to write additional-model checkpoint {path}: {exc}", "warning"
            )
