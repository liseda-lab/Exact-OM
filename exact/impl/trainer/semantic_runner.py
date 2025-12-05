import inspect
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data._utils.collate import default_collate
from exact.core.contracts.trainer import ITrainer
from exact.core.entities.mappings import EntityMapping
from exact.core.entities.configs.dataset import DatasetMask


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

        for path in ordered:
            mappings, results_json, processed_examples = self._load_checkpoint_state(path, kind)
            if processed_examples > 0:
                return path, mappings, results_json, processed_examples
        return None, [], [], 0

    def _load_checkpoint_state(
        self,
        checkpoint_path: Path,
        kind: DatasetMask,
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

        mappings: List[Tuple[str, str, float]] = []
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

        results_json = payload.get("results_json") or []
        processed_examples = int(payload.get("processed_examples", len(mappings)))
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
        payload = {
            "kind": kind.name,
            "total_examples": total_examples,
            "processed_examples": processed_examples,
            "mappings": [
                {"src": s, "tgt": t, "score": score} for s, t, score in mappings
            ],
            "results_json": results_json,
        }

        tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
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

    def _maybe_persist_model_cache(self, reason: str, force: bool = False) -> None:
        models = getattr(self, "models", None) or [getattr(self, "model", None)]
        for model in models:
            if model is None or not hasattr(model, "persist_caches"):
                continue
            try:
                model.persist_caches(force=force, reason=reason)
            except Exception as exc:  # noqa: BLE001
                self.log(f"Failed to persist model cache during {reason}: {exc}", "warning")

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
        batch_size: int = 8,
        num_workers: int = 0,
        log_every: int = 1,
        mixed_precision: bool = False,
        checkpoint_file: Optional[str] = None,
        checkpoint_every: int = 10,
        resume_from_checkpoint: bool = True,
        enable_checkpoints: bool = True,
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
        checkpoint_every = max(1, int(checkpoint_every))

        cp_path: Optional[Path] = None
        restored_examples = 0
        all_mappings: List[Tuple[str, str, float]] = []
        restored_json: List[Dict[str, Any]] = []

        if checkpoint_enabled and resume_from_checkpoint:
            cp_path, restored_mappings, restored_json, restored_examples = self._restore_from_available_checkpoints(
                kind, checkpoint_file
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
            self.results_json.extend(restored_json)

        total_examples = len(self.dataset)
        remaining_examples = max(0, total_examples - restored_examples)

        if remaining_examples == 0:
            self.log(
                (
                    "Checkpoint already contains predictions for all samples. "
                    "Skipping inference."
                ),
                level="info",
            )
            df = pd.DataFrame(all_mappings, columns=["Src", "Tgt", "Scores"])
            preds = EntityMapping.read_table_mappings(df, threshold=threshold, cardinality=cardinality)
            self.results_df = self._make_summary_dataframe(self.results_json)
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

            # Accumulate mappings
            s_label_vals = out["s_label"].detach().cpu().tolist()
            s_label_star_vals = out["s_label_star"].detach().cpu().tolist()
            s_ctx_vals = out["s_ctx"].detach().cpu().tolist()
            s_lctx_vals = out["S_lctx"].detach().cpu().tolist()
            s_final = out["S_final"].detach().cpu().tolist()
            w_c_vals = out["w_c"].detach().cpu().tolist()
            w_i_vals = out["w_i"].detach().cpu().tolist()
            p_llm_vals = out["p_llm"].detach().cpu().tolist()
            ground_truth = labels or [None] * len(src_iri)

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
                    "S_final": float(score),
                    "w_c": float(w_c_vals[idx]),
                    "w_i": float(w_i_vals[idx]),
                    "p_llm": float(p_llm_vals[idx]),
                    "src_label_text": self._summarize_label(src_labels[idx]),
                    "tgt_label_text": self._summarize_label(tgt_labels[idx]),
                    "src_context_text": self._summarize_context(src_ctxs[idx] if src_ctxs else []),
                    "tgt_context_text": self._summarize_context(tgt_ctxs[idx] if tgt_ctxs else []),
                })

            processed_examples += len(src_iri)
            batches_run += 1

            # Accumulate full JSONs 
            if "explanations" in out:
                self.results_json.extend(out["explanations"])

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
                self.log(
                    (
                        f"Batch {step}/{total_batches} done "
                        f"(avg {avg_batch_time:.2f}s/batch, ETA {eta_str})"
                    ),
                    "debug",
                )

        self._finalize_llm_calibration()

        total_time = time.perf_counter() - start_time
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
                        "LLM summary quality: "
                        f"{requested} requested, {usable} usable, {empty} empty (~{empty_pct:.2f}% empty)"
                    ),
                    "info",
                )
            else:
                self.log("LLM summaries were not requested for this run.", "info")

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

        candidate_df = self._build_candidate_dataframe()
        if self.results_json:
            # Prefer the richer JSON (includes importances and LLM info) for plotting/summary
            self.results_df = self._make_summary_dataframe(self.results_json)
        elif not candidate_df.empty:
            candidate_df = self._apply_additional_models(candidate_df)
            all_mappings = list(zip(candidate_df["Src"], candidate_df["Tgt"], candidate_df["S_final"]))
            self.results_df = candidate_df
        else:
            self.results_df = self._make_summary_dataframe(self.results_json)

        df_scores = pd.DataFrame(all_mappings, columns=["Src", "Tgt", "Scores"])
        preds = EntityMapping.read_table_mappings(df_scores, threshold=threshold, cardinality=cardinality)
        return preds, avg_t

    def _build_candidate_dataframe(self) -> pd.DataFrame:
        if not self._candidate_rows:
            return pd.DataFrame()
        return pd.DataFrame(self._candidate_rows)

    def _apply_additional_models(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run any additional models (beyond the first) sequentially.
        Each model is expected to accept the candidate dataframe and return
        either an updated dataframe or a dict containing 'candidate_df'.
        """
        current = df
        if len(getattr(self, "models", [])) <= 1:
            return current
        for idx, extra_model in enumerate(self.models[1:], start=2):
            extra_model.eval()
            sig = inspect.signature(extra_model.forward)
            accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            call_kwargs = {"candidate_df": current}
            if accepts_var_kw or "dataset" in sig.parameters:
                call_kwargs["dataset"] = self.dataset
            if accepts_var_kw or "logger" in sig.parameters:
                call_kwargs["logger"] = getattr(self, "logger", None)
            if accepts_var_kw or "primary_model" in sig.parameters:
                call_kwargs["primary_model"] = self.model
            out = extra_model.forward(**call_kwargs)
            if isinstance(out, pd.DataFrame):
                current = out
            elif isinstance(out, dict) and "candidate_df" in out:
                current = out["candidate_df"]
                extra_json = out.get("results_json")
                if extra_json:
                    self.results_json.extend(extra_json)
            else:
                self.log(
                    f"Additional model #{idx} returned unsupported type {type(out)}; ignoring its output.",
                    level="warning",
                )
        return current
