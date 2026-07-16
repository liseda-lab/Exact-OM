from __future__ import annotations

import ast  # noqa: F401
import copy  # noqa: F401
import json  # noqa: F401
import logging  # noqa: F401
import time  # noqa: F401
import warnings  # noqa: F401
from dataclasses import dataclass  # noqa: F401
from pathlib import Path  # noqa: F401
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple  # noqa: F401

import pandas as pd  # noqa: F401
import torch  # noqa: F401

from exact.utils.data import read_yaml  # noqa: F401
from exact.utils.formatting import format_duration as _format_duration  # noqa: F401

from .selection import (
    PROJECT_ROOT,
    _attribute_items,
    _categorize_decision_basis,
    _categorize_evidence_agreement,
    _categorize_evidence_strength,
    _categorize_explanation_coverage,
    _categorize_lead_over_next,
    _channel_items,
    _hierarchy_items,
    _ordered_path_nodes,
    _safe_float,
    _safe_text,
    _selected_label_list,
    _structural_evidence_counts,
)


def _bind_model_logger(model: Any, logger: logging.Logger) -> Any:
    def _model_log(message: Any, level: str = "info", *args: Any, **kwargs: Any) -> None:
        log_fn = getattr(logger, str(level).lower(), logger.info)
        log_fn(str(message))

    model.log = _model_log
    return model


def _load_configs_for_rationale(config_path: Path) -> Any:
    from exact.impl import bootstrap_components

    bootstrap_components()
    from exact.core.entities.configs.config import ConfigModel

    configs = ConfigModel.load_config(config_path)
    configs.resolve_dependencies()
    return configs


def _resolve_run_dataset_paths(
    run_dir: Path, logger: logging.Logger
) -> Tuple[Optional[Path], Optional[Path]]:
    candidate_specs = sorted(run_dir.glob("*.yaml")) + sorted(run_dir.glob("*.yml"))
    for spec_path in candidate_specs:
        try:
            payload = read_yaml(spec_path)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Skipping unreadable run spec %s: %s", spec_path, exc)
            continue
        dataset_cfg = dict(payload.get("dataset") or {})
        data_dir = dataset_cfg.get("data_dir")
        source_name = dataset_cfg.get("source")
        target_name = dataset_cfg.get("target")
        if not data_dir or not source_name or not target_name:
            continue
        data_dir_path = Path(str(data_dir))
        if not data_dir_path.is_absolute():
            candidate_roots = [
                (spec_path.parent / data_dir_path).resolve(),
                (PROJECT_ROOT / data_dir_path).resolve(),
            ]
            existing_root = next((root for root in candidate_roots if root.exists()), None)
            data_dir_path = existing_root or candidate_roots[0]
        source_path = Path(str(source_name))
        target_path = Path(str(target_name))
        if not source_path.is_absolute():
            source_path = (data_dir_path / source_path).resolve()
        if not target_path.is_absolute():
            target_path = (data_dir_path / target_path).resolve()
        if source_path.exists() and target_path.exists():
            return source_path, target_path
    return None, None


def _build_rationale_model(
    configs: Any, cache_dir: Path, device: torch.device, logger: logging.Logger
) -> Any:
    model_spec = configs.get_model_sequence()[0]
    model_cls = model_spec.name
    params = {
        **dict(model_spec.params or {}),
        "llm_profiles": {k: v.model_dump() for k, v in configs.llm_profiles.items()},
        "llm_routing": configs.llm_routing.model_dump(),
        "request_seed": getattr(configs, "seed", None),
        "cache_dir": cache_dir,
        "use_lexical": False,
        "use_context": False,
        "use_llm": True,
        "return_explanations": False,
        "generate_llm_rationales": True,
    }
    params.update(configs.alignment_params.model_dump(exclude_none=True))
    model = _bind_model_logger(model_cls(device=device, **params), logger)
    return model


def _build_explanation_backfill_model(
    configs: Any,
    run_dir: Path,
    output_dir: Path,
    device: torch.device,
    logger: logging.Logger,
) -> Optional[Any]:
    model_spec = configs.get_model_sequence()[0]
    model_cls = model_spec.name
    if getattr(model_cls, "__name__", "") != "PairAdaptiveSemanticScorer":
        logger.info(
            "Explanation-field backfill is only supported for PairAdaptiveSemanticScorer; skipping targeted pair rehydrate."
        )
        return None
    params = {
        **dict(model_spec.params or {}),
        "llm_profiles": {k: v.model_dump() for k, v in configs.llm_profiles.items()},
        "llm_routing": configs.llm_routing.model_dump(),
        "request_seed": getattr(configs, "seed", None),
        "cache_dir": output_dir / "cache",
        "use_llm": False,
        "return_explanations": True,
        "generate_llm_rationales": False,
    }
    params.update(configs.alignment_params.model_dump(exclude_none=True))
    model = _bind_model_logger(model_cls(device=device, **params), logger)
    source_path, target_path = _resolve_run_dataset_paths(run_dir, logger)
    if source_path is None or target_path is None:
        logger.warning(
            "Could not resolve source/target ontology paths from %s; targeted explanation rehydrate will be unavailable, but saved-record reconstruction can still run.",
            run_dir,
        )
        return model
    dataset_cls = configs.dataset
    if dataset_cls is None:
        logger.warning(
            "Resolved config does not provide a dataset class; targeted explanation rehydrate unavailable, but saved-record reconstruction can still run."
        )
        return model
    dataset = dataset_cls(
        output_path=output_dir,
        logger=logger,
        cache_ok=True,
        device=device,
        llm_profiles={k: v.model_dump() for k, v in configs.llm_profiles.items()},
        llm_routing=configs.llm_routing.model_dump(),
        request_seed=getattr(configs, "seed", None),
        **configs.dataset_params.model_dump(),
    )
    dataset.load_ontologies(source_path, target_path)
    if hasattr(model, "attach_dataset"):
        model.attach_dataset(dataset)
    return model


def _triple_attributions_missing_item_ids(record: Mapping[str, Any]) -> bool:
    triple_attributions = record.get("triple_attributions") or {}
    hierarchy = triple_attributions.get("hierarchy") or {}
    for family_payload in hierarchy.values():
        payload = dict(family_payload or {})
        for side in ["source", "target"]:
            items = list(payload.get(side) or [])
            if any(not _safe_text(item.get("item_id")) for item in items):
                return True
    for channel in ["similarity", "difference"]:
        payload = dict(triple_attributions.get(channel) or {})
        for side in ["source", "target"]:
            items = list(payload.get(side) or [])
            if any(not _safe_text(item.get("item_id")) for item in items):
                return True
    return False


def _attributes_missing_item_ids(record: Mapping[str, Any]) -> bool:
    attributes = record.get("attributes") or {}
    for side in ["source", "target"]:
        items = list((attributes.get(side) or []))
        if any(not _safe_text(item.get("item_id")) for item in items):
            return True
    return False


def _record_missing_entity_provenance(record: Mapping[str, Any]) -> bool:
    triple_attributions = record.get("triple_attributions") or {}
    hierarchy = triple_attributions.get("hierarchy") or {}
    for family_payload in hierarchy.values():
        payload = dict(family_payload or {})
        for side in ["source", "target"]:
            for item in list(payload.get(side) or []):
                triple = list(item.get("triple") or [])
                if len(triple) < 3:
                    continue
                if not _safe_text(item.get("subject_iri")) or not _safe_text(
                    item.get("object_iri")
                ):
                    return True
    for channel in ["similarity", "difference"]:
        payload = dict(triple_attributions.get(channel) or {})
        for side in ["source", "target"]:
            for item in list(payload.get(side) or []):
                triple = list(item.get("triple") or [])
                if len(triple) < 3:
                    continue
                if not _safe_text(item.get("subject_iri")) or not _safe_text(
                    item.get("object_iri")
                ):
                    return True
    return False


def _record_has_missing_item_ids(record: Mapping[str, Any]) -> bool:
    return _triple_attributions_missing_item_ids(record) or _attributes_missing_item_ids(record)


def _record_needs_explanation_backfill(record: Mapping[str, Any]) -> bool:
    if int(record.get("explanation_schema_version", 0) or 0) < 3:
        return True
    if _record_has_missing_item_ids(record):
        return True
    if _record_missing_entity_provenance(record):
        return True
    provenance = record.get("cross_side_provenance") or {}
    return not bool(provenance)


def _merge_missing_explanation_fields(
    original: Dict[str, Any],
    repaired: Mapping[str, Any],
) -> Dict[str, Any]:
    merged = copy.deepcopy(original)
    if int(merged.get("explanation_schema_version", 0) or 0) < 3:
        merged["explanation_schema_version"] = int(
            repaired.get("explanation_schema_version", 3) or 3
        )
    if _triple_attributions_missing_item_ids(merged) or not merged.get("triple_attributions"):
        triple_attributions = repaired.get("triple_attributions")
        if triple_attributions:
            merged["triple_attributions"] = copy.deepcopy(triple_attributions)
    elif _record_missing_entity_provenance(merged):
        triple_attributions = repaired.get("triple_attributions")
        if triple_attributions:
            merged["triple_attributions"] = copy.deepcopy(triple_attributions)
    if _attributes_missing_item_ids(merged) or not merged.get("attributes"):
        attributes = repaired.get("attributes")
        if attributes:
            merged["attributes"] = copy.deepcopy(attributes)
    if not merged.get("cross_side_provenance"):
        provenance = repaired.get("cross_side_provenance")
        if provenance:
            merged["cross_side_provenance"] = copy.deepcopy(provenance)
    for key in ["context_sentences", "context_triples", "selected_labels"]:
        if not merged.get(key) and repaired.get(key):
            merged[key] = copy.deepcopy(repaired.get(key))
    return merged


def _backfill_explanation_fields(
    records: List[Dict[str, Any]],
    run_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
    configs: Optional[Any] = None,
    config_path: Optional[Path] = None,
    device: Optional[int] = None,
    backfill_explanations: bool = True,
    log_every: int = 10,
) -> List[Dict[str, Any]]:
    hydrated = [copy.deepcopy(record) for record in records]
    if not backfill_explanations:
        logger.info("Explanation-field backfill disabled; keeping selected records as-is.")
        return hydrated
    pending_indices = [
        idx for idx, record in enumerate(hydrated) if _record_needs_explanation_backfill(record)
    ]
    if not pending_indices:
        logger.info(
            "Selected records already contain provenance-backed explanation fields; skipping explanation backfill."
        )
        return hydrated
    if configs is None:
        if config_path is None:
            raise ValueError(
                "config_path is required to backfill explanation fields when configs is not provided."
            )
        logger.info("Loading configs for explanation backfill from %s", config_path)
        configs = _load_configs_for_rationale(config_path=config_path)

    device_obj = (
        torch.device(device)
        if device is not None and torch.cuda.is_available()
        else torch.device("cpu")
    )
    model = _build_explanation_backfill_model(
        configs, run_dir=run_dir, output_dir=output_dir, device=device_obj, logger=logger
    )
    processed = 0
    from_saved_record = 0
    from_pair_rehydrate = 0
    failed = 0
    started = time.perf_counter()
    logger.info(
        "Explanation backfill started: records=%d, pending=%d, model=%s, device=%s",
        len(hydrated),
        len(pending_indices),
        (
            getattr(getattr(configs.get_model_sequence()[0], "name", None), "__name__", "unknown")
            if configs
            else "unknown"
        ),
        device_obj,
    )
    for idx in pending_indices:
        record = hydrated[idx]
        repaired = None
        try:
            if model is not None and hasattr(model, "reconstruct_explanation_fields_from_record"):
                repaired = model.reconstruct_explanation_fields_from_record(record)
                from_saved_record += 1
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "Saved-record explanation reconstruction failed for (%s, %s): %s",
                record.get("src_iri"),
                record.get("tgt_iri"),
                exc,
            )
            repaired = None
        if (
            repaired is None
            and model is not None
            and hasattr(model, "reconstruct_explanation_fields_for_pair")
        ):
            try:
                repaired = model.reconstruct_explanation_fields_for_pair(
                    _safe_text(record.get("src_iri")),
                    _safe_text(record.get("tgt_iri")),
                    src_labels=_selected_label_list(record, "source"),
                    tgt_labels=_selected_label_list(record, "target"),
                )
                from_pair_rehydrate += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Targeted explanation rehydrate failed for (%s, %s): %s",
                    record.get("src_iri"),
                    record.get("tgt_iri"),
                    exc,
                )
                failed += 1
                repaired = None
        if repaired is not None:
            hydrated[idx] = _merge_missing_explanation_fields(record, repaired)
        processed += 1
        if processed == len(pending_indices) or processed % max(1, int(log_every)) == 0:
            elapsed = max(1e-8, time.perf_counter() - started)
            rate = processed / elapsed
            remaining = max(0, len(pending_indices) - processed)
            eta = _format_duration(remaining / rate) if rate > 0 else _format_duration(0.0)
            logger.info(
                "Explanation backfill progress: records=%d/%d, from_saved_record=%d, targeted_pair_rehydrate=%d, failed=%d, avg=%.2fs/record, ETA %s",
                processed,
                len(pending_indices),
                from_saved_record,
                from_pair_rehydrate,
                failed,
                elapsed / max(1, processed),
                eta,
            )
    elapsed = max(0.0, time.perf_counter() - started)
    logger.info(
        "Explanation backfill completed: pending=%d, from_saved_record=%d, targeted_pair_rehydrate=%d, failed=%d, duration=%s",
        len(pending_indices),
        from_saved_record,
        from_pair_rehydrate,
        failed,
        _format_duration(elapsed),
    )
    return hydrated


def _sync_record_model_usage(record: Dict[str, Any]) -> None:
    models = dict(record.get("models") or {})
    backend_usage = record.get("backend_usage") or {}
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
    if len(unique_models) == 1:
        models["llm_model"] = unique_models[0]
    elif len(unique_models) > 1:
        models["llm_model"] = "multiple"
    record["models"] = models


def _backfill_rationales(
    records: List[Dict[str, Any]],
    output_dir: Path,
    logger: logging.Logger,
    configs: Optional[Any] = None,
    config_path: Optional[Path] = None,
    device: Optional[int] = None,
    generate_rationales: bool = True,
    log_every: int = 5,
) -> List[Dict[str, Any]]:
    hydrated = [copy.deepcopy(record) for record in records]
    if not generate_rationales:
        logger.info("Rationale backfill disabled; keeping selected records as-is.")
        return hydrated
    pending = [
        record
        for record in hydrated
        if not _safe_text((record.get("prediction") or {}).get("llm_rationale"))
    ]
    if not pending:
        logger.info("Selected records already contain rationales; skipping rationale generation.")
        return hydrated
    if configs is None:
        if config_path is None:
            raise ValueError(
                "config_path is required to generate rationales when configs is not provided."
            )
        logger.info("Loading configs for rationale backfill from %s", config_path)
        configs = _load_configs_for_rationale(config_path=config_path)
    device_obj = (
        torch.device(device)
        if device is not None and torch.cuda.is_available()
        else torch.device("cpu")
    )
    logger.info("Initializing rationale model on device=%s", device_obj)
    model = _build_rationale_model(configs, output_dir / "cache", device_obj, logger)
    logger.info("Generating missing rationales for %d selected records", len(pending))

    progress_state: Dict[str, Any] = {
        "started": False,
        "start_time": None,
        "last_logged_uncached": 0,
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
            logger.info(
                "Rationale stage started: records=%d, uncached_records=%d, uncached_unique_prompts=%d, "
                "cached_records=%d, backend=%s, model=%s, concurrency=%d.",
                int(event.get("total_records", 0) or 0),
                progress_state["uncached_records"],
                progress_state["uncached_unique_prompts"],
                progress_state["cached_records"],
                progress_state["backend"],
                progress_state["model"],
                progress_state["concurrency"] or 1,
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
        logger.info(
            "Rationale progress: uncached_records=%d/%d, unique_prompts=%d/%d, cached_records=%d, avg=%.2fs/record, ETA %s",
            completed_uncached,
            total_uncached,
            int(event.get("completed_unique_prompts", 0) or 0),
            int(event.get("total_unique_prompts", 0) or 0),
            int(event.get("cached_records", 0) or 0),
            avg_seconds,
            eta,
        )

    rationales = model.generate_final_rationales_for_records(
        pending, progress_callback=_progress_callback
    )
    rationale_meta = dict(getattr(model, "_last_rationale_backend_meta", {}) or {})
    for record, rationale in zip(pending, rationales):
        prediction = dict(record.get("prediction") or {})
        prediction["llm_rationale"] = rationale
        record["prediction"] = prediction
        backend_usage = dict(record.get("backend_usage") or {})
        backend_usage["rationale"] = dict(rationale_meta)
        record["backend_usage"] = backend_usage
        _sync_record_model_usage(record)
    if progress_state["started"]:
        elapsed = max(0.0, time.perf_counter() - float(progress_state["start_time"]))
        duration = _format_duration(elapsed)
        uncached_records = int(progress_state["uncached_records"] or 0)
        throughput = (
            (uncached_records / elapsed) if elapsed > 1e-8 and uncached_records > 0 else 0.0
        )
        avg_seconds = (elapsed / uncached_records) if uncached_records > 0 else 0.0
        logger.info(
            "Rationale stage completed: records=%d, uncached_records=%d, cached_records=%d, duration=%s, "
            "throughput=%.2f uncached records/s, avg=%.2fs/uncached record",
            len(hydrated),
            uncached_records,
            int(progress_state["cached_records"] or 0),
            duration,
            throughput,
            avg_seconds,
        )
    return hydrated


def _format_triple(triple: Sequence[Any]) -> str:
    subj = _safe_text(triple[0]) if len(triple) > 0 else ""
    rel = _safe_text(triple[1]) if len(triple) > 1 else ""
    obj = _safe_text(triple[2]) if len(triple) > 2 else ""
    return f"{subj} --{rel}--> {obj}"


def _node_id(
    display: str, node_type: str, used: Dict[Tuple[str, str], str], nodes: List[Dict[str, Any]]
) -> str:
    key = (node_type, display)
    if key in used:
        return used[key]
    candidate = display or node_type
    existing = {node["id"] for node in nodes}
    if candidate in existing:
        suffix = 2
        base = candidate
        while f"{base} [{suffix}]" in existing:
            suffix += 1
        candidate = f"{base} [{suffix}]"
    used[key] = candidate
    nodes.append({"id": candidate, "type": node_type})
    return candidate


def _append_edge(
    edges: List[Dict[str, Any]],
    seen_edges: set[Tuple[str, str, str, str]],
    source: str,
    target: str,
    label: str,
    score: Any,
    edge_type: str,
) -> None:
    key = (source, target, label, edge_type)
    if key in seen_edges:
        return
    seen_edges.add(key)
    edges.append(
        {
            "source": source,
            "target": target,
            "label": label,
            "score": score,
            "type": edge_type,
        }
    )


def _bridge_score_rank(score: Any) -> int:
    if isinstance(score, str):
        normalized = score.strip().lower()
        if normalized == "strong":
            return 3
        if normalized == "moderate":
            return 2
        if normalized == "weak":
            return 1
    return 0


def _append_bridge_edge(
    edges: List[Dict[str, Any]],
    bridge_index: Dict[Tuple[str, str, str], int],
    source: str,
    target: str,
    label: str,
    score: Any,
    edge_type: str,
) -> None:
    key = (source, target, edge_type)
    edge = {
        "source": source,
        "target": target,
        "label": label,
        "score": score,
        "type": edge_type,
    }
    current_idx = bridge_index.get(key)
    if current_idx is None:
        bridge_index[key] = len(edges)
        edges.append(edge)
        return
    current = edges[current_idx]
    if _bridge_score_rank(score) > _bridge_score_rank(current.get("score")):
        edges[current_idx] = edge


def _edge_identity(edge: Mapping[str, Any]) -> Tuple[str, str, str, str]:
    return (
        _safe_text(edge.get("source")),
        _safe_text(edge.get("target")),
        _safe_text(edge.get("label")),
        _safe_text(edge.get("type")),
    )


def _edge_display_level(edge: Mapping[str, Any]) -> Tuple[bool, int, str]:
    edge_type = _safe_text(edge.get("type"))
    if not edge_type.startswith("bridge-"):
        return False, 1, "Context edge"
    if edge_type == "bridge-contrast" or _safe_text(edge.get("label")).lower() == "label match":
        return True, 2, "Core bridge"
    if _safe_text(edge.get("score")).lower() == "weak":
        return True, 4, "Optional bridge"
    return True, 3, "Supporting bridge"


def _annotate_edge_metadata(paths: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not paths:
        return []
    updated_paths: List[Dict[str, Any]] = []
    for path in paths:
        new_path = dict(path)
        new_edges = []
        for edge in path.get("edges") or []:
            new_edge = dict(edge)
            is_bridge, level, level_label = _edge_display_level(edge)
            new_edge["bridge"] = is_bridge
            new_edge["level"] = level
            new_edge["level_label"] = level_label
            new_edges.append(new_edge)
        new_path["edges"] = new_edges
        updated_paths.append(new_path)
    return updated_paths


def _hierarchy_export_items(record: Mapping[str, Any], side: str) -> List[Dict[str, Any]]:
    return _hierarchy_items(record, side)


def _mean_nonempty(values: Sequence[float], default: float = 0.0) -> float:
    cleaned = [float(value) for value in values if value is not None]
    if not cleaned:
        return float(default)
    return float(sum(cleaned) / len(cleaned))


def _categorize_bridge_relevance(value: float) -> str:
    score = _safe_float(value, 0.0)
    if score >= 0.20:
        return "strong"
    if score >= 0.08:
        return "moderate"
    return "weak"


def _attribute_node_display(item: Mapping[str, Any]) -> str:
    value = _safe_text(item.get("value"))
    if value:
        return value
    text = _safe_text(item.get("text"))
    if ":" in text:
        suffix = text.split(":", 1)[1].strip()
        if suffix:
            return suffix
    if text:
        return text
    prop = _safe_text(item.get("property"))
    return prop or "attribute"


def _make_path_payload(
    record: Mapping[str, Any],
    rank: int,
    ground_truth: int,
    next_score: Optional[float] = None,
) -> Dict[str, Any]:
    labels = record.get("selected_labels") or {}
    prediction = record.get("prediction") or {}
    confidences = record.get("confidences") or {}
    importances = record.get("importances") or {}
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    used_nodes: Dict[Tuple[str, str], str] = {}
    seen_edges: set[Tuple[str, str, str, str]] = set()
    bridge_index: Dict[Tuple[str, str, str], int] = {}
    item_node_lookup: Dict[str, str] = {}
    item_importance_lookup: Dict[str, float] = {}

    source_text = _safe_text(labels.get("source")) or _safe_text(record.get("src_iri"))
    target_text = _safe_text(labels.get("target")) or _safe_text(record.get("tgt_iri"))
    src_iri = _safe_text(record.get("src_iri"))
    tgt_iri = _safe_text(record.get("tgt_iri"))

    source_node = _node_id(source_text, "Source", used_nodes, nodes)
    target_node = _node_id(target_text, "Target", used_nodes, nodes)
    item_node_lookup["__source__"] = source_node
    item_node_lookup["__target__"] = target_node

    def _context_node(display: Any, side: str) -> str:
        text = _safe_text(display)
        node_type = "source-context" if side == "source" else "target-context"
        return _node_id(text, node_type, used_nodes, nodes)

    def _triple_endpoint_node(value: Any, side: str) -> str:
        text = _safe_text(value)
        if side == "source" and text in {source_text, src_iri}:
            return source_node
        if side == "target" and text in {target_text, tgt_iri}:
            return target_node
        return _context_node(text, side)

    def _triple_item_node(subj: str, obj: str, side: str) -> str:
        endpoint = source_node if side == "source" else target_node
        if subj == endpoint and obj != endpoint:
            return obj
        if obj == endpoint and subj != endpoint:
            return subj
        return obj

    family_importances = dict(importances.get("family_importances") or {})
    i_label = _safe_float(importances.get("I_label"), 0.0)
    i_hier = _safe_float(importances.get("I_hier"), 0.0)
    i_sim = _safe_float(importances.get("I_sim"), 0.0)
    i_diff = _safe_float(importances.get("I_diff"), 0.0)
    i_attr = _safe_float(importances.get("I_attr"), 0.0)

    def _bridge_strength(channel_importance: float, local_mass: float) -> str:
        relevance = _safe_float(channel_importance, 0.0) * (
            0.5 + 0.5 * _safe_float(local_mass, 0.0)
        )
        return _categorize_bridge_relevance(relevance)

    for item in _hierarchy_export_items(record, "source"):
        triple = item.get("triple") or []
        if len(triple) < 3:
            continue
        subj = _triple_endpoint_node(triple[0], "source")
        obj = _triple_endpoint_node(triple[2], "source")
        relation = _safe_text(triple[1] if len(triple) > 1 else item.get("family")) or "hierarchy"
        score = _safe_float(item.get("importance", item.get("support", 1.0)), 1.0)
        _append_edge(edges, seen_edges, subj, obj, relation, score, edge_type="hierarchy")
        item_id = _safe_text(item.get("item_id"))
        if item_id:
            item_node_lookup[item_id] = _triple_item_node(subj, obj, "source")
            item_importance_lookup[item_id] = _safe_float(item.get("importance"), 0.0)
    for item in _hierarchy_export_items(record, "target"):
        triple = item.get("triple") or []
        if len(triple) < 3:
            continue
        subj = _triple_endpoint_node(triple[0], "target")
        obj = _triple_endpoint_node(triple[2], "target")
        relation = _safe_text(triple[1] if len(triple) > 1 else item.get("family")) or "hierarchy"
        score = _safe_float(item.get("importance", item.get("support", 1.0)), 1.0)
        _append_edge(edges, seen_edges, subj, obj, relation, score, edge_type="hierarchy")
        item_id = _safe_text(item.get("item_id"))
        if item_id:
            item_node_lookup[item_id] = _triple_item_node(subj, obj, "target")
            item_importance_lookup[item_id] = _safe_float(item.get("importance"), 0.0)

    for item in _channel_items(record, "similarity", "source"):
        triple = item.get("triple") or []
        if len(triple) < 3:
            continue
        subj = _triple_endpoint_node(triple[0], "source")
        obj = _triple_endpoint_node(triple[2], "source")
        relation = _safe_text(triple[1] if len(triple) > 1 else "similarity")
        score = _safe_float(item.get("importance", item.get("support", 1.0)), 1.0)
        _append_edge(edges, seen_edges, subj, obj, relation, score, edge_type="similarity")
        item_id = _safe_text(item.get("item_id"))
        if item_id:
            item_node_lookup[item_id] = _triple_item_node(subj, obj, "source")
            item_importance_lookup[item_id] = _safe_float(item.get("importance"), 0.0)
    for item in _channel_items(record, "similarity", "target"):
        triple = item.get("triple") or []
        if len(triple) < 3:
            continue
        subj = _triple_endpoint_node(triple[0], "target")
        obj = _triple_endpoint_node(triple[2], "target")
        relation = _safe_text(triple[1] if len(triple) > 1 else "similarity")
        score = _safe_float(item.get("importance", item.get("support", 1.0)), 1.0)
        _append_edge(edges, seen_edges, subj, obj, relation, score, edge_type="similarity")
        item_id = _safe_text(item.get("item_id"))
        if item_id:
            item_node_lookup[item_id] = _triple_item_node(subj, obj, "target")
            item_importance_lookup[item_id] = _safe_float(item.get("importance"), 0.0)

    for item in _channel_items(record, "difference", "source"):
        triple = item.get("triple") or []
        if len(triple) < 3:
            continue
        subj = _triple_endpoint_node(triple[0], "source")
        obj = _triple_endpoint_node(triple[2], "source")
        relation = _safe_text(triple[1] if len(triple) > 1 else "difference")
        score = _safe_float(item.get("importance", item.get("support", 1.0)), 1.0)
        _append_edge(edges, seen_edges, subj, obj, relation, score, edge_type="difference")
        item_id = _safe_text(item.get("item_id"))
        if item_id:
            item_node_lookup[item_id] = _triple_item_node(subj, obj, "source")
            item_importance_lookup[item_id] = _safe_float(item.get("importance"), 0.0)
    for item in _channel_items(record, "difference", "target"):
        triple = item.get("triple") or []
        if len(triple) < 3:
            continue
        subj = _triple_endpoint_node(triple[0], "target")
        obj = _triple_endpoint_node(triple[2], "target")
        relation = _safe_text(triple[1] if len(triple) > 1 else "difference")
        score = _safe_float(item.get("importance", item.get("support", 1.0)), 1.0)
        _append_edge(edges, seen_edges, subj, obj, relation, score, edge_type="difference")
        item_id = _safe_text(item.get("item_id"))
        if item_id:
            item_node_lookup[item_id] = _triple_item_node(subj, obj, "target")
            item_importance_lookup[item_id] = _safe_float(item.get("importance"), 0.0)

    for item in _attribute_items(record, "source"):
        display = _attribute_node_display(item)
        node = _context_node(display, "source")
        relation = _safe_text(item.get("property")) or "attribute"
        score = _safe_float(item.get("importance", item.get("support", 1.0)), 1.0)
        _append_edge(edges, seen_edges, source_node, node, relation, score, edge_type="attribute")
        item_id = _safe_text(item.get("item_id"))
        if item_id:
            item_node_lookup[item_id] = node
            item_importance_lookup[item_id] = _safe_float(item.get("importance"), 0.0)
    for item in _attribute_items(record, "target"):
        display = _attribute_node_display(item)
        node = _context_node(display, "target")
        relation = _safe_text(item.get("property")) or "attribute"
        score = _safe_float(item.get("importance", item.get("support", 1.0)), 1.0)
        _append_edge(edges, seen_edges, target_node, node, relation, score, edge_type="attribute")
        item_id = _safe_text(item.get("item_id"))
        if item_id:
            item_node_lookup[item_id] = node
            item_importance_lookup[item_id] = _safe_float(item.get("importance"), 0.0)

    provenance = dict(record.get("cross_side_provenance") or {})
    for link in list(provenance.get("lexical") or []):
        _append_bridge_edge(
            edges,
            bridge_index,
            source_node,
            target_node,
            "label match",
            _bridge_strength(i_label, 1.0),
            edge_type="bridge-support",
        )
    for family, links in dict(provenance.get("hierarchy") or {}).items():
        for link in list(links or []):
            src_item = item_node_lookup.get(_safe_text(link.get("source_item_id")))
            tgt_item = item_node_lookup.get(_safe_text(link.get("target_item_id")))
            if not src_item or not tgt_item:
                continue
            source_item_id = _safe_text(link.get("source_item_id"))
            target_item_id = _safe_text(link.get("target_item_id"))
            family_importance = _safe_float(family_importances.get(family), i_hier)
            local_mass = _mean_nonempty(
                [
                    item_importance_lookup.get(source_item_id),
                    item_importance_lookup.get(target_item_id),
                ],
                default=0.0,
            )
            _append_bridge_edge(
                edges,
                bridge_index,
                src_item,
                tgt_item,
                "shared hierarchy",
                _bridge_strength(family_importance, local_mass),
                edge_type="bridge-support",
            )
    for link in list(provenance.get("similarity") or []):
        src_item = item_node_lookup.get(_safe_text(link.get("source_item_id")))
        tgt_item = item_node_lookup.get(_safe_text(link.get("target_item_id")))
        if not src_item or not tgt_item:
            continue
        source_item_id = _safe_text(link.get("source_item_id"))
        target_item_id = _safe_text(link.get("target_item_id"))
        local_mass = _mean_nonempty(
            [
                item_importance_lookup.get(source_item_id),
                item_importance_lookup.get(target_item_id),
            ],
            default=0.0,
        )
        _append_bridge_edge(
            edges,
            bridge_index,
            src_item,
            tgt_item,
            "similar evidence",
            _bridge_strength(i_sim, local_mass),
            edge_type="bridge-support",
        )
    for side in ["source", "target"]:
        for link in list((dict(provenance.get("attributes") or {})).get(side) or []):
            item_node = item_node_lookup.get(_safe_text(link.get("item_id")))
            anchor_node = item_node_lookup.get(_safe_text(link.get("anchor_ref")))
            if not item_node or not anchor_node:
                continue
            item_id = _safe_text(link.get("item_id"))
            anchor_ref = _safe_text(link.get("anchor_ref"))
            local_mass = _mean_nonempty(
                [
                    item_importance_lookup.get(item_id),
                    item_importance_lookup.get(anchor_ref),
                ],
                default=_safe_float(item_importance_lookup.get(item_id), 0.0),
            )
            _append_bridge_edge(
                edges,
                bridge_index,
                item_node,
                anchor_node,
                "attribute evidence",
                _bridge_strength(i_attr, local_mass),
                edge_type="bridge-support",
            )
    for link in list((dict(provenance.get("difference") or {})).get("source") or []):
        item_node = item_node_lookup.get(_safe_text(link.get("item_id")))
        if not item_node:
            continue
        item_id = _safe_text(link.get("item_id"))
        _append_bridge_edge(
            edges,
            bridge_index,
            item_node,
            target_node,
            "distinctive evidence",
            _bridge_strength(i_diff, _safe_float(item_importance_lookup.get(item_id), 0.0)),
            edge_type="bridge-contrast",
        )
    for link in list((dict(provenance.get("difference") or {})).get("target") or []):
        item_node = item_node_lookup.get(_safe_text(link.get("item_id")))
        if not item_node:
            continue
        item_id = _safe_text(link.get("item_id"))
        _append_bridge_edge(
            edges,
            bridge_index,
            item_node,
            source_node,
            "distinctive evidence",
            _bridge_strength(i_diff, _safe_float(item_importance_lookup.get(item_id), 0.0)),
            edge_type="bridge-contrast",
        )

    counts = _structural_evidence_counts(record)
    i_label = _safe_float(importances.get("I_label"), 0.0)
    i_struct = _safe_float(importances.get("I_struct"), 0.0)
    i_llm = _safe_float(importances.get("I_llm"), 0.0)
    q_struct = _safe_float(confidences.get("Q_struct"), 0.0)
    u_dis = _safe_float((record.get("weights") or {}).get("U_dis"), 0.0)
    score_value = _safe_float(confidences.get("S_final"), 0.0)

    return {
        "id": _safe_text(record.get("tgt_iri")) or f"path_{rank}",
        "rank": int(rank),
        "ground_truth": int(ground_truth),
        "score": score_value,
        "metrics": {
            "decision_basis": _categorize_decision_basis(i_label, i_struct, i_llm),
            "evidence_strength": _categorize_evidence_strength(
                i_struct, q_struct, counts["total_evidence"]
            ),
            "evidence_agreement": _categorize_evidence_agreement(u_dis, counts["total_evidence"]),
            "explanation_coverage": _categorize_explanation_coverage(
                counts["present_channels"], counts
            ),
            "lead_over_next_candidate": _categorize_lead_over_next(score_value, next_score),
        },
        "llm": {
            "source": _safe_text(labels.get("source")),
            "target": _safe_text(labels.get("target")),
            "p_llm": _safe_float(confidences.get("p_llm"), 0.0),
            "llm_decision": _safe_text(prediction.get("llm_decision")),
            "llm_pair_brief": _safe_text(record.get("llm_pair_brief")),
            "llm_rationale": _safe_text(prediction.get("llm_rationale")),
        },
        "nodes": _ordered_path_nodes(nodes),
        "edges": edges,
    }


def _build_study_mapping(
    selected_df: pd.DataFrame,
    ranked_candidates_by_source: Mapping[str, List[Tuple[str, float]]],
    record_index: Mapping[Tuple[str, str], Mapping[str, Any]],
    top_k: int,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    pairs: List[Dict[str, Any]] = []
    warned_missing_provenance: set[Tuple[str, str]] = set()
    for row in selected_df.to_dict(orient="records"):
        src = _safe_text(row.get("src_iri"))
        gold = _safe_text(row.get("gold_tgt_iri"))
        candidates = list(ranked_candidates_by_source.get(src, []))[:top_k]
        paths: List[Dict[str, Any]] = []
        for rank, (tgt, _score) in enumerate(candidates, start=1):
            record = record_index.get((src, tgt))
            if record is None:
                raise ValueError(f"Missing selected record for source={src} target={tgt}")
            if logger is not None and not (record.get("cross_side_provenance") or {}):
                key = (src, tgt)
                if key not in warned_missing_provenance:
                    warned_missing_provenance.add(key)
                    logger.warning(
                        "Mapping export is using an unbridged path for (%s, %s) because cross_side_provenance is still unavailable.",
                        src,
                        tgt,
                    )
            next_score = None
            if rank < len(candidates):
                next_tgt, next_fallback_score = candidates[rank]
                next_record = record_index.get((src, next_tgt))
                if next_record is not None:
                    next_score = _safe_float(
                        (next_record.get("confidences") or {}).get("S_final"), next_fallback_score
                    )
                else:
                    next_score = float(next_fallback_score)
            paths.append(
                _make_path_payload(
                    record, rank=rank, ground_truth=int(tgt == gold), next_score=next_score
                )
            )
        pairs.append({"id": src, "paths": _annotate_edge_metadata(paths)})
    return {"pairs": pairs}


def _selected_records(
    selected_df: pd.DataFrame,
    ranked_candidates_by_source: Mapping[str, List[Tuple[str, float]]],
    record_index: Mapping[Tuple[str, str], Mapping[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for row in selected_df.to_dict(orient="records"):
        src = _safe_text(row.get("src_iri"))
        gold = _safe_text(row.get("gold_tgt_iri"))
        for rank, (tgt, score) in enumerate(
            ranked_candidates_by_source.get(src, [])[:top_k], start=1
        ):
            record = record_index.get((src, tgt))
            if record is None:
                raise ValueError(f"Missing selected record for source={src} target={tgt}")
            item = copy.deepcopy(record)
            item["study_metadata"] = {
                "source_iri": src,
                "rank": rank,
                "ground_truth": int(tgt == gold),
                "candidate_score": float(score),
                "gold_rank": int(row.get("gold_rank", -1)),
            }
            selected.append(item)
    return selected


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
