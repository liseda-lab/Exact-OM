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
except ImportError:  # pragma: no cover - exercised only when optional dependency is absent
    zstd = None


class RationalesMixin:
    @staticmethod
    def _rationale_record_key(record: Dict[str, Any]) -> str:
        return f"{record.get('src_iri', '')}\u241f{record.get('tgt_iri', '')}"

    def _rationale_checkpoint_path(
        self,
        kind: DatasetMask,
        local_alignment: bool,
        threshold: Optional[float],
        cardinality: Optional[int],
    ) -> Path:
        payload = self._postprocess_fingerprint_payload(
            kind, local_alignment, threshold, cardinality
        )
        fingerprint = self._hash_checkpoint_fingerprint_payload(payload)
        return (
            self.checkpoint_dir / f"{kind.name.lower()}_rationales_{fingerprint[:12]}.json"
        ).resolve()

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
        expected = self._postprocess_fingerprint_payload(
            kind, local_alignment, threshold, cardinality
        )
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
            "fingerprint_payload": self._postprocess_fingerprint_payload(
                kind, local_alignment, threshold, cardinality
            ),
            "rationales": rationales,
        }
        try:
            self._write_json_atomic(path, payload)
            self.log(f"Wrote rationale checkpoint ({len(rationales)} records) to {path}", "debug")
        except OSError as exc:
            self.log(f"Failed to write rationale checkpoint {path}: {exc}", "warning")

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
        rationale_checkpoint = self._load_rationale_checkpoint(
            kind, local_alignment, threshold, cardinality
        )
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
                progress_state["uncached_unique_prompts"] = int(
                    event.get("uncached_unique_prompts", 0) or 0
                )
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
            if (
                completed_uncached < total_uncached
                and (completed_uncached - last_logged) < interval
            ):
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
                self._write_rationale_checkpoint(
                    kind, local_alignment, threshold, cardinality, rationale_checkpoint
                )
            return

        def _apply_rationale(
            record_idx: int, rationale: str, rationale_meta: Optional[Dict[str, Any]] = None
        ) -> None:
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
            interval = max(
                1, int(progress_state.get("interval_uncached_records") or log_every or 1)
            )
            last_saved = int(progress_state.get("last_saved") or 0)
            if completed - last_saved >= interval:
                self._write_rationale_checkpoint(
                    kind, local_alignment, threshold, cardinality, rationale_checkpoint
                )
                if hasattr(model, "persist_caches"):
                    try:
                        model.persist_caches(force=True, reason="rationale_checkpoint")
                    except Exception as exc:  # noqa: BLE001
                        self.log(
                            f"Failed to persist model cache during rationale checkpoint: {exc}",
                            "warning",
                        )
                progress_state["last_saved"] = completed

        rationale_sig = inspect.signature(model.generate_final_rationales_for_records)
        accepts_var_kw = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in rationale_sig.parameters.values()
        )
        rationale_kwargs: Dict[str, Any] = {"progress_callback": _progress_callback}
        if accepts_var_kw or "completion_callback" in rationale_sig.parameters:
            rationale_kwargs["completion_callback"] = _completion_callback
        rationales = model.generate_final_rationales_for_records(
            pending_records, **rationale_kwargs
        )
        rationale_meta = getattr(model, "_last_rationale_backend_meta", {}) or {}
        for original_idx, rationale in zip(pending_indices, rationales):
            _apply_rationale(original_idx, rationale, rationale_meta)
        self._write_rationale_checkpoint(
            kind, local_alignment, threshold, cardinality, rationale_checkpoint
        )
        if progress_state["started"]:
            elapsed = max(0.0, time.perf_counter() - float(progress_state["start_time"]))
            duration = _format_duration(elapsed)
            uncached_records = int(progress_state["uncached_records"] or 0)
            throughput = (
                (uncached_records / elapsed) if elapsed > 1e-8 and uncached_records > 0 else 0.0
            )
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
