from __future__ import annotations

import hashlib  # noqa: F401
import inspect  # noqa: F401
import json  # noqa: F401
import time  # noqa: F401
import uuid
from dataclasses import dataclass
from pathlib import Path  # noqa: F401
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple  # noqa: F401

import pandas as pd  # noqa: F401
import torch  # noqa: F401
from torch.utils.data import DataLoader, Subset  # noqa: F401
from torch.utils.data._utils.collate import default_collate  # noqa: F401

from exact.core.entities.configs.dataset import DatasetMask  # noqa: F401
from exact.core.entities.mappings import EntityMapping  # noqa: F401
from exact.utils.formatting import format_duration as _format_duration  # noqa: F401
from exact.utils.run_context import current_run_session
from exact.utils.timing import CacheStatus, StageRecord  # noqa: F401

try:
    import zstandard as zstd  # noqa: F401
except (
    ImportError
):  # pragma: no cover - exercised only when optional dependency is absent
    zstd = None

from exact.core.contracts.trainer import ITrainer

from .audit_io import AuditIOMixin, _semantic_collate_fn
from .checkpointing import CheckpointingMixin
from .overlays import OverlaysMixin
from .rationales import RationalesMixin


@dataclass
class _FinalizationState:
    kind: DatasetMask
    threshold: Optional[float]
    cardinality: Optional[int]
    target_cardinality: Optional[int]
    local_alignment: bool
    all_mappings: List[Tuple[str, str, float]]
    total_examples: int
    processed_examples: int
    restored_examples: int
    checkpoint_enabled: bool
    checkpoint_path: Optional[Path]
    audit_shard_compression: str
    audit_shard_records: int
    log_every: int
    run_progress: Optional[Any]
    inference_seconds: float
    average_seconds_per_example: float
    inference_work_done: int
    inference_status: CacheStatus
    post_inference_status: CacheStatus
    completed_from_checkpoint: bool = False


class SemanticAlignmentRunner(
    CheckpointingMixin, AuditIOMixin, RationalesMixin, OverlaysMixin, ITrainer
):
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
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
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
        self._last_stage_timings: List[StageRecord] = []
        self._inference_seconds_cumulative: float = 0.0
        self._restored_inference_seconds_cumulative: float = 0.0
        self._inference_session_started_at: Optional[float] = None
        self._examples_per_second_ema: Optional[float] = None
        self._last_effective_threshold: Optional[float] = None
        self._last_effective_threshold_origin: str = "configured"
        self._checkpoint_fingerprint_payload: Dict[str, Any] = (
            self._build_checkpoint_fingerprint_payload()
        )
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
        run_session = current_run_session()
        self._run_id: str = (
            str(run_session.run_id)
            if getattr(run_session, "run_id", None)
            else str(uuid.uuid4())
        )
        self._explanation_store: Optional[Any] = None
        self._explanation_shard_mb: float = 32.0

    def _finalize_run(
        self, state: _FinalizationState
    ) -> Tuple[List[EntityMapping], float]:
        """Apply the shared post-inference path for fresh and completed checkpoints."""
        post_inference_start = time.perf_counter()
        candidate_df = self._build_candidate_dataframe()
        if state.completed_from_checkpoint and candidate_df.empty and self.results_json:
            candidate_df = self._build_candidate_dataframe_from_records(
                self.results_json
            )

        post_progress_started = False
        if state.run_progress is not None and len(getattr(self, "models", [])) > 1:
            n_sources = (
                int(candidate_df["Src"].nunique())
                if not candidate_df.empty and "Src" in candidate_df.columns
                else 0
            )
            detail = (
                f"rows={len(candidate_df)}, sources={n_sources}"
                if state.completed_from_checkpoint
                else "assembling candidate dataframe"
            )
            state.run_progress.start("PostInference", detail)
            post_progress_started = True

        if state.completed_from_checkpoint:
            if not candidate_df.empty:
                n_sources = (
                    int(candidate_df["Src"].nunique())
                    if "Src" in candidate_df.columns
                    else 0
                )
                self.log(
                    (
                        "Post-inference processing restored candidate scores: "
                        f"rows={len(candidate_df)}, sources={n_sources}."
                    ),
                    "debug",
                )
        else:
            self.log(
                "Post-inference processing started: assembling candidate dataframe.",
                "info",
            )

        if not candidate_df.empty:
            n_sources = (
                int(candidate_df["Src"].nunique())
                if "Src" in candidate_df.columns
                else 0
            )
            if post_progress_started and not state.completed_from_checkpoint:
                state.run_progress.update(
                    "PostInference",
                    fraction=0.15,
                    detail=f"rows={len(candidate_df)}, sources={n_sources}",
                    force=True,
                )
            if not state.completed_from_checkpoint:
                self.log(
                    (
                        "Candidate dataframe assembled for post-inference processing: "
                        f"rows={len(candidate_df)}, sources={n_sources}."
                    ),
                    "debug",
                )
            checkpointed_df = self._load_additional_models_checkpoint(
                state.kind,
                state.local_alignment,
                state.threshold,
                state.cardinality,
            )
            if checkpointed_df is not None:
                candidate_df = checkpointed_df
            else:
                candidate_df = self._apply_additional_models(
                    candidate_df,
                    kind=state.kind,
                    local_alignment=state.local_alignment,
                    threshold=state.threshold,
                    cardinality=state.cardinality,
                    target_cardinality=state.target_cardinality,
                    results_json=self.results_json,
                    log_every=state.log_every,
                    run_progress=state.run_progress,
                )
                self._write_additional_models_checkpoint(
                    state.kind,
                    candidate_df,
                    state.local_alignment,
                    state.threshold,
                    state.cardinality,
                )
            state.all_mappings = list(
                zip(candidate_df["Src"], candidate_df["Tgt"], candidate_df["S_final"])
            )
            self._selector_target_conflict_enabled = (
                self._selector_target_conflict_enabled_from_df(candidate_df)
            )
            self._last_effective_threshold_origin = (
                self._effective_alignment_threshold_origin(
                    candidate_df, state.threshold
                )
            )
            state.threshold = self._effective_alignment_threshold(
                candidate_df, state.threshold
            )
            self._last_effective_threshold = state.threshold

            if not state.completed_from_checkpoint:
                if self.results_json:
                    self.results_df = self._make_summary_dataframe(self.results_json)
                else:
                    self.results_df = candidate_df
        elif not state.completed_from_checkpoint:
            self.results_df = self._make_summary_dataframe(self.results_json)

        if post_progress_started:
            state.run_progress.finish("PostInference", f"rows={len(candidate_df)}")

        score_frame = pd.DataFrame(state.all_mappings, columns=["Src", "Tgt", "Scores"])
        predictions = EntityMapping.read_table_mappings(
            score_frame,
            threshold=state.threshold,
            cardinality=state.cardinality,
        )
        compact_rationale_records = False
        if not candidate_df.empty:
            self._annotate_candidate_dataframe(
                candidate_df,
                predictions,
                state.threshold,
                state.local_alignment,
            )
            compact_rationale_records = self._ensure_compact_rationale_records(
                candidate_df
            )
        self._annotate_final_prediction_records(
            predictions,
            threshold=state.threshold,
            local_alignment=state.local_alignment,
        )

        rationale_start = time.perf_counter()
        self._generate_final_rationales(
            log_every=state.log_every,
            kind=state.kind,
            local_alignment=state.local_alignment,
            threshold=state.threshold,
            cardinality=state.cardinality,
        )
        rationale_elapsed = time.perf_counter() - rationale_start
        post_inference_elapsed = max(
            0.0,
            time.perf_counter() - post_inference_start - rationale_elapsed,
        )

        if self.results_json and not compact_rationale_records:
            self.results_df = self._make_summary_dataframe(self.results_json)
            if not state.completed_from_checkpoint:
                for record in self.results_json:
                    self._sync_record_model_usage(record)
        elif not candidate_df.empty:
            self.results_df = candidate_df

        if (
            not candidate_df.empty
            and state.checkpoint_enabled
            and state.checkpoint_path is not None
        ):
            self._write_final_overlay(
                state.checkpoint_path,
                candidate_df,
                predictions,
                state.threshold,
                state.local_alignment,
                compression=state.audit_shard_compression,
                records_per_shard=state.audit_shard_records,
            )
        if state.checkpoint_enabled and state.checkpoint_path is not None:
            self._write_checkpoint_state(
                state.checkpoint_path,
                state.kind,
                total_examples=state.total_examples,
                processed_examples=state.processed_examples,
                mappings=state.all_mappings,
                results_json=self.results_json,
            )

        explanation_store = getattr(self, "_explanation_store", None)
        if explanation_store is not None and explanation_store.overlay_count:
            compaction = explanation_store.compact()
            self._overlay_manifest_path = None
            self.log(
                (
                    "Compacted transient explanation overlays: "
                    f"records={compaction['records']}, "
                    f"before={compaction['before_bytes']} bytes, "
                    f"after={compaction['after_bytes']} bytes."
                ),
                "info",
            )

        self._last_stage_timings = [
            StageRecord(
                stage="Alignment.Inference",
                seconds=state.inference_seconds,
                cache_status=state.inference_status,
                work_done=state.inference_work_done,
                work_total=state.total_examples,
                unit="examples",
            ),
            StageRecord(
                stage="Alignment.PostInference",
                seconds=post_inference_elapsed,
                cache_status=state.post_inference_status,
            ),
            StageRecord(
                stage="Postprocess.Rationales",
                seconds=rationale_elapsed,
                cache_status=CacheStatus.FRESH,
            ),
        ]
        return predictions, state.average_seconds_per_example

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
        explanation_shard_mb: float = 32.0,
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
        self._last_stage_timings = []
        self._inference_seconds_cumulative = 0.0
        self._restored_inference_seconds_cumulative = 0.0
        self._inference_session_started_at = None
        self._examples_per_second_ema = None
        self._last_effective_threshold = threshold
        self._last_effective_threshold_origin = "configured"
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
        self._explanation_shard_mb = max(0.001, float(explanation_shard_mb))
        self._prepare_audit_shards(
            None,
            enabled=True,
            compression=audit_shard_compression,
            records_per_shard=audit_shard_records,
            append_existing=True,
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
            cp_path, restored_mappings, restored_json, restored_examples = (
                self._restore_from_available_checkpoints(
                    kind,
                    checkpoint_file,
                    allow_rationale_toggle_checkpoint_resume=allow_rationale_toggle_checkpoint_resume,
                )
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
            restored_candidate_rows = (
                getattr(self, "_restored_candidate_rows", []) or []
            )
            if restored_candidate_rows:
                self._candidate_rows.extend(restored_candidate_rows)

        total_examples = len(self.dataset)
        remaining_examples = max(0, total_examples - restored_examples)

        if restored_examples == 0:
            self._prepare_audit_shards(
                cp_path,
                enabled=True,
                compression=audit_shard_compression,
                records_per_shard=audit_shard_records,
                append_existing=False,
            )

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
            return self._finalize_run(
                _FinalizationState(
                    kind=kind,
                    threshold=threshold,
                    cardinality=cardinality,
                    target_cardinality=target_cardinality,
                    local_alignment=local_alignment,
                    all_mappings=all_mappings,
                    total_examples=total_examples,
                    processed_examples=restored_examples,
                    restored_examples=restored_examples,
                    checkpoint_enabled=checkpoint_enabled,
                    checkpoint_path=cp_path,
                    audit_shard_compression=audit_shard_compression,
                    audit_shard_records=audit_shard_records,
                    log_every=log_every,
                    run_progress=run_progress,
                    inference_seconds=0.0,
                    average_seconds_per_example=0.0,
                    inference_work_done=0,
                    inference_status=CacheStatus.SKIPPED,
                    post_inference_status=CacheStatus.RESUMED,
                    completed_from_checkpoint=True,
                )
            )

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
            dataset_for_dl = Subset(
                self.dataset, range(restored_examples, total_examples)
            )

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
        self._inference_session_started_at = start_time
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
                self._candidate_rows.append(
                    {
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
                        "src_context_text": self._summarize_context(
                            src_ctxs[idx] if src_ctxs else []
                        ),
                        "tgt_context_text": self._summarize_context(
                            tgt_ctxs[idx] if tgt_ctxs else []
                        ),
                    }
                )

            processed_examples += len(src_iri)
            batches_run += 1
            elapsed_for_rate = max(1.0e-9, time.perf_counter() - start_time)
            current_rate = (processed_examples - restored_examples) / elapsed_for_rate
            if current_rate > 0.0:
                if self._examples_per_second_ema is None:
                    self._examples_per_second_ema = current_rate
                else:
                    self._examples_per_second_ema = (
                        0.2 * current_rate + 0.8 * self._examples_per_second_ema
                    )

            # Accumulate full JSONs
            explanations = list(out.get("explanations") or [])
            union_records: List[Dict[str, Any]] = []
            batch_candidate_rows = self._candidate_rows[candidate_start_idx:]
            for local_idx, candidate_row in enumerate(batch_candidate_rows):
                explanation = (
                    explanations[local_idx] if local_idx < len(explanations) else None
                )
                record = self._union_explanation_record(candidate_row, explanation)
                if explanation is not None:
                    candidate_idx = candidate_start_idx + local_idx
                    if candidate_idx >= len(self._candidate_rows):
                        continue
                    evidence_items = self._selector_evidence_items_for_record(record)
                    if evidence_items:
                        self._candidate_rows[candidate_idx][
                            "selector_evidence_items"
                        ] = evidence_items
                        record["selector_evidence_items"] = evidence_items
                union_records.append(record)
            self.results_json.extend(union_records)
            self._append_audit_records(union_records)

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
                    pair_total = max(
                        1, int(pair_batch_stats.get("pairs", len(src_iri)))
                    )
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
        self._inference_seconds_cumulative = (
            self._restored_inference_seconds_cumulative + inference_elapsed_seconds
        )
        self._inference_session_started_at = None
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
            run_progress.finish(
                "Inference", f"processed={processed_examples}/{total_examples}"
            )
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
                self.log(
                    "LLM summaries/briefs were not requested for this run.", "debug"
                )

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
        inference_status = (
            CacheStatus.RESUMED if restored_examples > 0 else CacheStatus.FRESH
        )
        return self._finalize_run(
            _FinalizationState(
                kind=kind,
                threshold=threshold,
                cardinality=cardinality,
                target_cardinality=target_cardinality,
                local_alignment=local_alignment,
                all_mappings=all_mappings,
                total_examples=total_examples,
                processed_examples=processed_examples,
                restored_examples=restored_examples,
                checkpoint_enabled=checkpoint_enabled,
                checkpoint_path=cp_path,
                audit_shard_compression=audit_shard_compression,
                audit_shard_records=audit_shard_records,
                log_every=log_every,
                run_progress=run_progress,
                inference_seconds=inference_elapsed_seconds,
                average_seconds_per_example=avg_t,
                inference_work_done=new_examples,
                inference_status=inference_status,
                post_inference_status=inference_status,
            )
        )

    @staticmethod
    def _effective_alignment_threshold(
        candidate_df: pd.DataFrame,
        threshold: Optional[float],
    ) -> Optional[float]:
        if threshold is None or candidate_df.empty:
            return threshold
        if "selection_accept_threshold" not in candidate_df.columns:
            return threshold
        selected = pd.to_numeric(
            candidate_df["selection_accept_threshold"], errors="coerce"
        )
        selected = selected[selected.notna() & (selected > 0.0)]
        if selected.empty:
            return threshold
        return float(selected.median())

    @staticmethod
    def _effective_alignment_threshold_origin(
        candidate_df: pd.DataFrame,
        threshold: Optional[float],
    ) -> str:
        if threshold is None or candidate_df.empty:
            return "configured"
        if "selection_accept_threshold" not in candidate_df.columns:
            return "configured"
        selected = pd.to_numeric(
            candidate_df["selection_accept_threshold"], errors="coerce"
        )
        selected = selected[selected.notna() & (selected > 0.0)]
        return "selector-median override" if not selected.empty else "configured"

    @staticmethod
    def _selector_target_conflict_enabled_from_df(
        candidate_df: pd.DataFrame,
    ) -> Optional[bool]:
        if (
            candidate_df.empty
            or "selection_target_conflict_enabled" not in candidate_df.columns
        ):
            return None
        if "selection_accept_threshold" in candidate_df.columns:
            thresholds = pd.to_numeric(
                candidate_df["selection_accept_threshold"], errors="coerce"
            )
            if thresholds[thresholds.notna() & (thresholds > 0.0)].empty:
                return None
        values = candidate_df["selection_target_conflict_enabled"]
        if values.empty:
            return None
        normalized = values.astype(str).str.lower().isin({"true", "1", "yes", "y"})
        return bool(normalized.any())

    @property
    def last_stage_timings(self) -> List[StageRecord]:
        return list(self._last_stage_timings)

    @property
    def inference_seconds_cumulative(self) -> float:
        return self._inference_seconds_cumulative

    @property
    def examples_per_second_ema(self) -> Optional[float]:
        return self._examples_per_second_ema

    @property
    def last_effective_threshold(self) -> Optional[float]:
        return self._last_effective_threshold

    @property
    def last_effective_threshold_origin(self) -> str:
        return self._last_effective_threshold_origin

    def _build_candidate_dataframe(self) -> pd.DataFrame:
        if not self._candidate_rows:
            return pd.DataFrame()
        df = pd.DataFrame(self._candidate_rows)
        return self._merge_dataset_candidate_columns(df)

    def _build_candidate_dataframe_from_records(
        self, records: List[Dict[str, Any]]
    ) -> pd.DataFrame:
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
                "src_label_text": (record.get("selected_labels") or {}).get(
                    "source", ""
                ),
                "tgt_label_text": (record.get("selected_labels") or {}).get(
                    "target", ""
                ),
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
            if col in {"Src", "Tgt"}
            or any(str(col).startswith(prefix) for prefix in prefixes)
        ]
        if len(extra_cols) <= 2:
            return df
        extra_df = dataset_df[extra_cols].drop_duplicates(subset=["Src", "Tgt"])
        merge_cols = [
            col
            for col in extra_df.columns
            if col not in df.columns or col in {"Src", "Tgt"}
        ]
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
            accepts_var_kw = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
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
