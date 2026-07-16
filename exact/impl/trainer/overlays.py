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
from exact.utils.formatting import format_duration as _format_duration  # noqa: F401
from exact.utils.timing import CacheStatus, StageRecord  # noqa: F401

try:
    import zstandard as zstd  # noqa: F401
except (
    ImportError
):  # pragma: no cover - exercised only when optional dependency is absent
    zstd = None

from .audit_io import _json_default


class OverlaysMixin:
    def _sync_selector_fields_from_candidate_df(self, df: pd.DataFrame) -> None:
        if (
            not self.results_json
            or df.empty
            or "Src" not in df.columns
            or "Tgt" not in df.columns
        ):
            return
        row_lookup = {
            (str(row["Src"]), str(row["Tgt"])): row for _, row in df.iterrows()
        }
        for record in self.results_json:
            row = row_lookup.get(
                (str(record.get("src_iri")), str(record.get("tgt_iri")))
            )
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

    def _maybe_persist_model_cache(self, reason: str, force: bool = False) -> None:
        policy = str(
            getattr(self, "_cache_persist_policy", "checkpoint") or "checkpoint"
        ).lower()
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
                self.log(
                    f"Failed to persist model cache during {reason}: {exc}", "warning"
                )
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
            pred["rationale_decision_label"] = (
                "Match" if rationale_positive else "No match"
            )
            rec["prediction"] = pred

    def _annotate_candidate_dataframe(
        self,
        candidate_df: pd.DataFrame,
        preds: List[EntityMapping],
        threshold: Optional[float],
        local_alignment: bool,
    ) -> None:
        if (
            candidate_df.empty
            or "Src" not in candidate_df.columns
            or "Tgt" not in candidate_df.columns
        ):
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
            rationale_positive = (
                bool(threshold_positive) if local_alignment else bool(saved)
            )
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

    @staticmethod
    def _result_record_lookup(
        records: List[Dict[str, Any]],
    ) -> Dict[Tuple[str, str], Dict[str, Any]]:
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
                prediction["llm_rationale"] = str(
                    source_prediction.get("llm_rationale")
                )
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
        self._annotate_candidate_dataframe(
            candidate_df, preds, threshold, local_alignment
        )
        store = getattr(self, "_explanation_store", None)
        if store is not None:
            start = time.perf_counter()
            total = 0
            chunk: List[Dict[str, Any]] = []
            result_record_lookup = self._result_record_lookup(self.results_json)
            for _, row in candidate_df.iterrows():
                chunk.append(
                    self._overlay_record_from_candidate_row(row, result_record_lookup)
                )
                if len(chunk) >= 5000:
                    total += store.append_overlay(chunk)
                    chunk = []
            if chunk:
                total += store.append_overlay(chunk)
            self._overlay_manifest_path = store.index_path
            self.log(
                (
                    f"Wrote transient final overlay for {total} records to "
                    f"{store.overlays_dir} in "
                    f"{_format_duration(max(0.0, time.perf_counter() - start))}"
                ),
                "info",
            )
            return store.index_path
        overlay_manifest_path = self._overlay_manifest_for_checkpoint(checkpoint_path)
        overlay_dir = overlay_manifest_path.parent
        overlay_dir.mkdir(parents=True, exist_ok=True)
        for path in overlay_dir.glob("shard-*.jsonl*"):
            try:
                path.unlink()
            except OSError as exc:
                self.log(
                    f"Failed to remove stale overlay shard {path}: {exc}", "warning"
                )
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
                if (
                    writer is None
                    or current_shard is None
                    or int(current_shard.get("records", 0)) >= shard_limit
                ):
                    if writer is not None:
                        writer.close()
                    shard_name = (
                        f"shard-{len(shards):06d}{self._jsonl_suffix(resolved)}"
                    )
                    current_shard = {"path": shard_name, "records": 0}
                    shards.append(current_shard)
                    writer = self._open_jsonl_writer(overlay_dir / shard_name, resolved)
                writer.write(
                    json.dumps(
                        self._overlay_record_from_candidate_row(
                            row, result_record_lookup
                        ),
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
