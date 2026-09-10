import csv
import functools
import json
import logging
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import torch

from exact.analysis.candidate_recall import analyze_candidate_recall
from exact.core.actions.evaluation import run_evaluation
from exact.core.entities.configs.config import ConfigModel
from exact.core.entities.registry import ComponentRegistry, ComponentType
from exact.runs import RunLayout, finalize_artifacts
from exact.tracks import get_track, provider_from_descriptor
from exact.utils.data import read_table
from exact.utils.logs import (
    ProgressTask,
    RunProgressLogger,
    configure_exact_logger,
    summarize_progress_estimates,
)
from exact.utils.provenance import file_provenance
from exact.utils.timing import CacheStatus, RunSession, TimingLedger, config_fingerprint


@dataclass(frozen=True)
class ResolvedAlignmentInputs:
    """Effective alignment inputs after CLI, config, and track precedence."""

    source: Path
    target: Path
    training_reference: Optional[Path]
    full_reference: Optional[Path]
    candidates: Optional[Path]
    task_name: Optional[str]
    track_provenance: Optional[dict[str, Any]]


def _resolved_path(path: Optional[Path]) -> Optional[Path]:
    return Path(path).expanduser().resolve() if path is not None else None


def _require_existing(path: Optional[Path], label: str, *, required: bool = False) -> None:
    if path is None:
        if required:
            raise ValueError(
                f"{label} is required. Pass an explicit path or configure data.track/data.task."
            )
        return
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def _merge_run_stats(path: Path, additions: Mapping[str, Any]) -> None:
    """Atomically merge nested run-stat metadata without discarding trainer output."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logging.getLogger("exact").warning(
                "Replacing unreadable run statistics at %s: %s", path, exc
            )
        else:
            if isinstance(loaded, dict):
                payload = loaded
    for key, value in additions.items():
        if isinstance(value, Mapping) and isinstance(payload.get(key), Mapping):
            payload[key] = {**dict(payload[key]), **dict(value)}
        else:
            payload[key] = value
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_table(frame: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, sep="\t", index=False)
    os.replace(temporary, path)


def _candidate_recall_run_stats(
    dataset: Any,
    *,
    training_reference_path: Optional[Path],
) -> dict[str, Any]:
    """Summarize the materialized ranked pool against this stage's reference.

    This is called only for explicitly audited experiment runs. It keeps gold
    information out of the candidate-pool manifest while making the declared
    candidate-recall endpoint available to screen/confirm reporting.
    """

    def pair_columns(frame: Any, label: str) -> tuple[str, str]:
        columns = {str(column).strip().lower(): str(column) for column in frame.columns}
        source = next(
            (
                columns[name]
                for name in ("src", "srcentity", "source", "source_id")
                if name in columns
            ),
            None,
        )
        target = next(
            (
                columns[name]
                for name in ("tgt", "tgtentity", "target", "target_id")
                if name in columns
            ),
            None,
        )
        if source is None or target is None:
            raise ValueError(f"{label} has no unambiguous source/target columns")
        return source, target

    def pairs(frame: Any, label: str) -> list[tuple[str, str]]:
        if frame is None or frame.empty:
            return []
        source, target = pair_columns(frame, label)
        return [
            (str(src), str(tgt))
            for src, tgt in frame[[source, target]].itertuples(index=False, name=None)
        ]

    reference = getattr(dataset, "reference", None)
    candidates = getattr(dataset, "candidates", None)
    if reference is None or reference.empty:
        return {
            "metric_applicability": {"candidate_recall": False},
            "candidate_recall_diagnostics": {
                "status": "not_applicable",
                "reason": "stage reference is absent or empty",
            },
        }

    candidate_input: Any = []
    if candidates is not None and not candidates.empty:
        source, target = pair_columns(candidates, "candidate pool")
        score = next(
            (
                str(column)
                for column in candidates.columns
                if str(column).strip().lower() in {"cand_sim", "score", "scores", "similarity"}
            ),
            None,
        )
        candidate_input = candidates[[source, target]].rename(
            columns={source: "Src", target: "Tgt"}
        )
        candidate_input["cand_sim"] = (
            candidates[score].to_numpy(copy=True) if score is not None else 0.0
        )

    training_pairs: list[tuple[str, str]] = []
    if training_reference_path is not None:
        training_frame = read_table(Path(training_reference_path))
        training_pairs = pairs(training_frame, "training reference")
    analysis = analyze_candidate_recall(
        candidate_input,
        pairs(reference, "stage reference"),
        train_pairs=training_pairs,
        exact_pairs=pairs(getattr(dataset, "exact_matches", None), "exact prefilter"),
    )
    counts = dict(analysis["counts"])
    metrics = dict(analysis["metrics"])
    gold_rank = dict(analysis["gold_rank"])
    pool_manifest = getattr(dataset, "candidate_pool_manifest", None)
    pool_summary = (
        pool_manifest.get("gold_free_summary") if isinstance(pool_manifest, Mapping) else None
    )
    mean_pool_size: Optional[float] = None
    if isinstance(pool_summary, Mapping) and pool_summary.get("mean_pool_size") is not None:
        mean_pool_size = float(pool_summary["mean_pool_size"])
    elif candidates is not None and not candidates.empty:
        source, _target = pair_columns(candidates, "candidate pool")
        mean_pool_size = float(candidates.groupby(source, sort=False).size().mean())
    if int(counts.get("reference_pairs") or 0) == 0:
        return {
            "metric_applicability": {"candidate_recall": False},
            "candidate_recall_diagnostics": {
                "status": "not_applicable",
                "reason": "no evaluation reference pairs remain after training-pair exclusion",
                "counts": counts,
                "metrics": metrics,
                "gold_rank": gold_rank,
            },
        }
    return {
        "metric_applicability": {"candidate_recall": True},
        "candidate_recall": float(metrics["generated_candidate_recall"]),
        "candidate_recall_after_exact": float(metrics["exact_prefilter_oracle_recall"]),
        "mean_pool_size": mean_pool_size,
        "gold_rank_p90": gold_rank.get("rank_p90"),
        "gold_rank_median": gold_rank.get("rank_median"),
        "candidate_recall_diagnostics": {
            "status": "available",
            "counts": counts,
            "metrics": metrics,
            "gold_rank": gold_rank,
        },
    }


def _table_columns(path: Path) -> list[str]:
    delimiter = "\t" if str(path).lower().endswith(".tsv") else ","
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        try:
            return [str(value) for value in next(csv.reader(stream, delimiter=delimiter))]
        except StopIteration as exc:
            raise ValueError(f"reference table is empty: {path}") from exc


def _materialize_evaluation_reference(
    frame: Any,
    *,
    parent_path: Path,
    output_path: Path,
) -> Path:
    """Persist the evaluated pair set with explicit relation and entity kinds."""

    materialized = frame.copy()
    materialized = materialized.rename(columns={"Src": "SrcEntity", "Tgt": "TgtEntity"})
    source_columns = {column.strip().lower() for column in _table_columns(parent_path)}
    if "relation" in source_columns:
        materialized = materialized.rename(columns={"Label": "Relation"})
    else:
        # Local candidate splits and two-column reference alignments encode
        # equivalence by protocol; their third column, when present, is a
        # candidate list rather than a relation label.
        materialized["Relation"] = "="
    required = [
        "SrcEntity",
        "TgtEntity",
        "Relation",
        "SrcKind",
        "TgtKind",
    ]
    missing = [column for column in required if column not in materialized.columns]
    if missing:
        raise ValueError(
            f"evaluation reference is missing enriched columns {missing}: {parent_path}"
        )
    _atomic_table(materialized[required], output_path)
    return output_path


def _write_resolved_config(configs: ConfigModel, layout: RunLayout) -> None:
    """Persist the canonical v2 settings used by this run."""

    from exact.core.entities.configs.yaml_io import dump_yaml_document

    rendered = dump_yaml_document(configs.model_dump(mode="json", by_alias=True))
    temporary = layout.config_path.with_name(f".{layout.config_path.name}.{os.getpid()}.tmp")
    temporary.write_text(rendered, encoding="utf-8")
    os.replace(temporary, layout.config_path)


def run_alignment(
    source_file_path: Optional[Path] = None,
    target_file_path: Optional[Path] = None,
    output_dir_path: Optional[Path] = None,
    configs_file_path: Optional[Union[Path, ConfigModel]] = None,
    configs_source: Optional[Path] = None,
    training_reference_file_path: Optional[Path] = None,
    full_reference_file_path: Optional[Path] = None,
    candidates_file_path: Optional[Path] = None,
    log_file_path: Optional[Path] = None,
    run_eval: bool = False,
    task_name: Optional[str] = None,
    device: Optional[int] = None,
) -> tuple[Optional[dict[str, Any]], dict[str, float]]:
    """Resolve inputs, run the alignment pipeline, and persist timing metadata."""

    if output_dir_path is None:
        raise ValueError("output_dir_path is required")
    output_dir_path = Path(output_dir_path).expanduser().resolve()
    output_dir_path.mkdir(parents=True, exist_ok=True)
    run_layout = RunLayout.create(output_dir_path)

    if configs_file_path is None:
        configs = ConfigModel()
    elif isinstance(configs_file_path, ConfigModel):
        configs = configs_file_path
    else:
        configs = ConfigModel.load_config(configs_file_path)
    configs.resolve_dependencies()
    _write_resolved_config(configs, run_layout)
    configure_exact_logger(
        logging.getLogger("exact"),
        configs.logging_level,
        log_file_path=log_file_path,
    )

    resolved = resolve_alignment_inputs(
        configs=configs,
        source_file_path=source_file_path,
        target_file_path=target_file_path,
        training_reference_file_path=training_reference_file_path,
        full_reference_file_path=full_reference_file_path,
        candidates_file_path=candidates_file_path,
        task_name=task_name,
    )

    fingerprint_source: Any = configs
    if resolved.track_provenance is not None:
        fingerprint_provenance = {
            key: value for key, value in resolved.track_provenance.items() if key != "retrieved_at"
        }
        fingerprint_source = {
            "config": configs.model_dump(mode="python"),
            "dataset_provenance": fingerprint_provenance,
        }
    fingerprint = config_fingerprint(fingerprint_source, run_dir=output_dir_path)
    ledger = TimingLedger.open(output_dir_path)
    with ledger.session(
        command="align",
        config_fingerprint=fingerprint,
    ) as timing_session:
        with timing_session.stage("Total") as total_span:
            session_runner = _run_alignment_session
            compatibility_runner = AlignmentAction._run_session
            if compatibility_runner is not _DEFAULT_RUN_ALIGNMENT_SESSION:
                session_runner = compatibility_runner
            results, run_stats_path = session_runner(
                source_file_path=resolved.source,
                target_file_path=resolved.target,
                output_dir_path=output_dir_path,
                configs=configs,
                configs_source=(
                    configs_source if configs_source is not None else configs_file_path
                ),
                training_reference_file_path=resolved.training_reference,
                full_reference_file_path=resolved.full_reference,
                candidates_file_path=resolved.candidates,
                log_file_path=log_file_path,
                run_eval=run_eval,
                task_name=resolved.task_name,
                device=device,
                timing_ledger=ledger,
                timing_session=timing_session,
            )

    if resolved.track_provenance is not None and run_stats_path is not None:
        _merge_run_stats(
            run_stats_path,
            {"provenance": {"dataset": resolved.track_provenance}},
        )

    totals = ledger.stage_totals(config_fingerprint=fingerprint)
    timings_result = {
        stage: (total.compute_seconds if total.compute_seconds > 0.0 else total.overhead_seconds)
        / 60.0
        for stage, total in totals.items()
    }
    timings_result["Total"] = totals["Total"].compute_seconds / 60.0

    session_record = next(
        session for session in ledger.sessions() if session.run_id == timing_session.run_id
    )
    this_session_seconds: dict[str, float] = {}
    for stage in session_record.stages:
        this_session_seconds[stage.stage] = (
            this_session_seconds.get(stage.stage, 0.0) + stage.seconds
        )
    cumulative_compute_seconds = {stage: total.compute_seconds for stage, total in totals.items()}
    timing_stats = {
        "run_id": timing_session.run_id,
        "config_fingerprint": fingerprint,
        "this_session_seconds": this_session_seconds,
        "cumulative_compute_seconds": cumulative_compute_seconds,
        "cumulative_overhead_seconds": {
            stage: total.overhead_seconds for stage, total in totals.items()
        },
    }
    if run_stats_path is not None and run_stats_path.exists():
        try:
            _merge_run_stats(run_stats_path, {"timing": timing_stats})
        except OSError as exc:
            logging.getLogger("exact").warning(
                "Could not add timing summary to %s: %s",
                run_stats_path,
                exc,
            )

    size_report = finalize_artifacts(
        run_layout,
        run_id=timing_session.run_id,
        save_full_explanations=configs.output.save.full_explanations_json,
        checkpoint_retention=configs.output.retention.checkpoints,
    )

    total_stage = totals["Total"]
    logger = logging.getLogger("exact")
    logger.info(
        "Total compute across %d sessions: %.1fm (this session: %.1fm)",
        total_stage.sessions,
        total_stage.compute_seconds / 60.0,
        (total_span.seconds or 0.0) / 60.0,
    )
    logger.info("Times updated at %s", ledger.times_path)
    logger.info(
        "Run artifacts finalized at %s (checkpoints removed: %d, bytes freed: %.2f MiB)",
        size_report["manifest"],
        size_report["checkpoints_removed"],
        size_report["checkpoint_bytes_removed"] / (1024 * 1024),
    )
    before_sizes = size_report["before_bytes"]
    after_sizes = size_report["after_bytes"]
    logger.info(
        "Artifact sizes before → after (MiB): %s",
        ", ".join(
            f"{name}={before_sizes.get(name, 0) / (1024 * 1024):.2f}→"
            f"{after_sizes.get(name, 0) / (1024 * 1024):.2f}"
            for name in sorted(set(before_sizes) | set(after_sizes))
        ),
    )
    return results, timings_result


def resolve_alignment_inputs(
    *,
    configs: ConfigModel,
    source_file_path: Optional[Path] = None,
    target_file_path: Optional[Path] = None,
    training_reference_file_path: Optional[Path] = None,
    full_reference_file_path: Optional[Path] = None,
    candidates_file_path: Optional[Path] = None,
    task_name: Optional[str] = None,
) -> ResolvedAlignmentInputs:
    """Resolve input precedence and lazily materialize a configured track."""

    data = configs.effective_data_config()
    layout = None
    track_provenance: Optional[dict[str, Any]] = None
    if data is not None and (data.track or data.descriptor):
        if not data.task:
            raise ValueError("data.task is required when selecting a dataset track")
        if data.descriptor is not None:
            descriptor_path = _resolved_path(data.descriptor)
            _require_existing(descriptor_path, "Track descriptor", required=True)
            provider = provider_from_descriptor(descriptor_path)
        else:
            provider = get_track(str(data.track))
        root = Path(data.root).expanduser().resolve()
        logging.getLogger("exact").info(
            "Materializing dataset track %s/%s under %s",
            provider.name,
            data.task,
            root,
        )
        layout = provider.materialize(
            data.task,
            root,
            revision=data.revision,
            update=False,
        )
        track_provenance = {
            **dict(layout.provenance),
            "track": provider.name,
            "task": data.task,
        }

    configured_refs = {
        str(split): _resolved_path(path)
        for split, path in ((data.refs if data is not None else {}) or {}).items()
    }
    layout_refs = dict(layout.refs) if layout is not None else {}
    refs = {**layout_refs, **configured_refs}

    configured_source = _resolved_path(data.source) if data is not None else None
    configured_target = _resolved_path(data.target) if data is not None else None
    configured_candidates = _resolved_path(data.candidates) if data is not None else None
    source = (
        _resolved_path(source_file_path)
        or configured_source
        or (layout.source if layout is not None else None)
    )
    target = (
        _resolved_path(target_file_path)
        or configured_target
        or (layout.target if layout is not None else None)
    )
    training_reference = _resolved_path(training_reference_file_path) or refs.get("train")
    selected_reference = None
    split_candidates = None
    if data is not None and data.reference_role:
        reference_role = str(data.reference_role)
        selected_reference = refs.get(reference_role)
        if layout is not None:
            candidate_value = layout.extras.get(f"{reference_role}_candidates")
            if candidate_value is None and reference_role in {"test", "full"}:
                candidate_value = layout.candidates
            if candidate_value is not None:
                split_candidates = _resolved_path(Path(candidate_value))
        # Bio-ML local candidate files carry the gold target in column two and
        # are therefore also the canonical validation/test reference table.
        if selected_reference is None:
            selected_reference = split_candidates
        if selected_reference is None:
            available = sorted(
                set(refs)
                | {
                    key[: -len("_candidates")]
                    for key in (layout.extras if layout is not None else {})
                    if key.endswith("_candidates")
                }
            )
            raise ValueError(
                f"Configured data.reference_role {data.reference_role!r} is unavailable; "
                f"available reference splits: {', '.join(available) or 'none'}"
            )
    full_reference = (
        _resolved_path(full_reference_file_path)
        or selected_reference
        or refs.get("full")
        or refs.get("test")
        or refs.get("valid")
    )
    explicit_candidates = _resolved_path(candidates_file_path) or configured_candidates
    candidate_source = data.candidate_source if data is not None else "track"
    if candidate_source == "generated" and configured_candidates is not None:
        raise ValueError("data.candidate_source=generated conflicts with data.candidates")
    if explicit_candidates is not None:
        candidates = explicit_candidates
    elif candidate_source == "generated":
        candidates = None
    elif candidate_source == "reference_split":
        if split_candidates is None:
            raise ValueError(
                "data.candidate_source=reference_split requires a materialized "
                "split-specific candidate pool"
            )
        candidates = split_candidates
    else:
        candidates = layout.candidates if layout is not None else None

    _require_existing(source, "Source ontology", required=True)
    _require_existing(target, "Target ontology", required=True)
    _require_existing(training_reference, "Training reference")
    _require_existing(full_reference, "Full reference")
    _require_existing(candidates, "Candidates file")
    assert source is not None
    assert target is not None
    return ResolvedAlignmentInputs(
        source=source,
        target=target,
        training_reference=training_reference,
        full_reference=full_reference,
        candidates=candidates,
        task_name=task_name or (data.task if data is not None else None),
        track_provenance=track_provenance,
    )


def _run_alignment_session(
    source_file_path: Path,
    target_file_path: Path,
    output_dir_path: Path,
    configs: ConfigModel,
    configs_source: Optional[Union[Path, ConfigModel]] = None,
    training_reference_file_path: Optional[Path] = None,
    full_reference_file_path: Optional[Path] = None,
    candidates_file_path: Optional[Path] = None,
    log_file_path: Optional[Path] = None,
    run_eval: bool = False,
    task_name: Optional[str] = None,
    device: Optional[int] = None,
    timing_ledger: TimingLedger = None,
    timing_session: RunSession = None,
):

    # Loading logging configuration from configs

    logger = configure_exact_logger(
        logging.getLogger("exact"),
        configs.logging_level,
        log_file_path=log_file_path,
    )

    logger.debug(f"Logging level set to {configs.logging_level}")

    # log configs state

    if configs_source is not None:
        logger.info(f"Using configuration from {configs_source}")
    else:
        logger.info("Using default configuration")

    # Resolve dependencies
    model_sequence = configs.get_model_sequence()
    if not model_sequence:
        raise ValueError("No models configured for alignment.")
    has_post_inference = any(
        extra.name is not None
        and not (isinstance(extra.params, dict) and extra.params.get("enabled") is False)
        for extra in model_sequence[1:]
    )
    progress_tasks = [
        ProgressTask("Setup", "Setup", estimate_seconds=12.0),
        ProgressTask("Dataset", "Dataset", estimate_seconds=300.0),
        ProgressTask("Trainer", "Trainer/model", estimate_seconds=30.0),
        ProgressTask("Inference", "Inference", estimate_seconds=600.0),
    ]
    if has_post_inference:
        progress_tasks.append(
            ProgressTask("PostInference", "Post-inference", estimate_seconds=60.0)
        )
    if configs.dataset_params.filter_exact_matches:
        progress_tasks.append(ProgressTask("Prefilter", "Exact prefilter", estimate_seconds=30.0))
    progress_tasks.extend(
        [
            ProgressTask("Outputs", "Outputs", estimate_seconds=60.0),
            ProgressTask("Plots", "Plots", estimate_seconds=60.0),
        ]
    )
    if run_eval:
        progress_tasks.append(ProgressTask("Evaluation", "Evaluation", estimate_seconds=60.0))
    progress = RunProgressLogger(
        logger,
        progress_tasks,
        estimates_minutes=summarize_progress_estimates(
            ledger=timing_ledger,
            config_fingerprint=timing_session.config_fingerprint,
        ),
    )
    progress.start("Setup", "configuration resolved")
    logger.info(
        "Dataset params: only_taxonomy=%s all_labels=%s filter_exact_matches=%s use_file_cache=%s",
        configs.dataset_params.only_taxonomy,
        configs.dataset_params.all_labels,
        configs.dataset_params.filter_exact_matches,
        configs.use_file_cache,
    )

    # set seed

    if configs.seed is not None:
        logger.info(f"Setting seed to {configs.seed}")
        seed_setter = ComponentRegistry.get(ComponentType.SEED_SETTER, "SeedSetter")
        seed_setter(configs.seed)

    if device is not None and not torch.cuda.is_available():
        logger.warning("CUDA device specified but not available. Using CPU instead.")

    device = (
        torch.device(device)
        if device is not None and torch.cuda.is_available()
        else torch.device("cpu")
    )
    progress.finish("Setup", f"device={device}")

    sampled_sources: set[str] = set()
    sampled_source_groups: set[tuple[str, str]] = set()
    sampled_full_reference_path: Optional[Path] = None
    sampled_training_reference_path: Optional[Path] = None
    effective_candidates_file_path = candidates_file_path
    declared_mode = configs.data.execution_mode
    execution_mode = declared_mode or (
        "local_ranking" if candidates_file_path is not None else "global_alignment"
    )
    local_ranking = execution_mode == "local_ranking"
    if local_ranking and candidates_file_path is None:
        raise ValueError("data.execution_mode=local_ranking requires a supplied candidate pool")
    if declared_mode is not None and configs.evaluation.dual_global_local:
        raise ValueError(
            "explicit execution_mode requires separate global and local runs; "
            "dual_global_local cannot evaluate one execution as both tasks"
        )
    evaluation_full_reference: Any = full_reference_file_path
    evaluation_training_reference: Any = training_reference_file_path
    materialized_full_reference_path: Optional[Path] = None
    evaluation_alignment_path: Optional[Path] = None

    # Create Dataset

    progress.start("Dataset", "building dataset inputs")
    logger.info("Building Dataset...")
    with timing_session.stage("Dataset") as dataset_span:
        dataset_factory = configs.dataset_runtime
        if dataset_factory is None:
            raise RuntimeError("Dataset dependency was not resolved")
        dataset = dataset_factory(
            output_path=output_dir_path,
            logger=logger,
            cache_ok=configs.use_file_cache,
            device=device,
            llm_profiles={k: v.model_dump() for k, v in configs.llm_profiles.items()},
            llm_routing=configs.llm_routing.model_dump(),
            request_seed=configs.seed,
            candidate_generation_params={
                **configs.candidates.model_dump(mode="python"),
                "candidates_file_path": (
                    str(candidates_file_path) if candidates_file_path else None
                ),
            },
            input_format=configs.io.input_format,
            source_options=configs.io.source_options,
            target_options=configs.io.target_options,
            entity_kinds=configs.matching.entity_kinds,
            **configs.dataset_params.model_dump(),
        )

        with timing_session.stage("Dataset.LoadOntologies"):
            dataset.load_ontologies(source_file_path, target_file_path)
        if configs.data.source_universe is not None:
            dataset.freeze_source_universe(
                Path(configs.data.source_universe).read_text(encoding="utf-8").splitlines(),
                cap=configs.run.source_cap,
                seed=configs.seed,
            )
        dataset_loaded_from_cache = dataset.has_cache()

        if dataset_loaded_from_cache:
            dataset_span.cache_status = CacheStatus.SKIPPED
            timing_session.record(
                "Dataset.LoadCandidates",
                seconds=0.0,
                cache_status=CacheStatus.SKIPPED,
            )
            with timing_session.stage(
                "Dataset.CacheLoad",
                cache_status=CacheStatus.CACHE_HIT,
            ):
                dataset.process()
            timing_session.record(
                "Dataset.Process",
                seconds=0.0,
                cache_status=CacheStatus.SKIPPED,
            )
            timing_session.record(
                "Dataset.Save",
                seconds=0.0,
                cache_status=CacheStatus.SKIPPED,
            )
            timing_session.record(
                "Dataset.Plotting",
                seconds=0.0,
                cache_status=CacheStatus.SKIPPED,
            )
        else:
            timing_session.record(
                "Dataset.CacheLoad",
                seconds=0.0,
                cache_status=CacheStatus.SKIPPED,
            )
            with timing_session.stage("Dataset.LoadCandidates"):
                if full_reference_file_path is not None:
                    dataset.load_reference(full_reference_file_path)
                dataset.load_candidates(
                    candidates_file_path,
                    device=device,
                    **configs.candidates.model_dump(mode="python"),
                )
            with timing_session.stage("Dataset.Process"):
                dataset.process()
            with timing_session.stage("Dataset.Save"):
                dataset.save()

            if getattr(dataset, "emit_feature_metrics_on_build", lambda: False)():
                dataset.save_feature_metrics()

            if configs.output.sanity_checks.enabled:
                dataset.log_sanity_examples(**configs.sanity_check_params.model_dump())
            with timing_session.stage("Dataset.Plotting"):
                dataset.plot_feature_distributions(
                    which=configs.dataset_params.which,
                    **configs.plot_params.model_dump(),
                )

        if configs.run.source_cap is not None or configs.data.source_universe is not None:
            sampled_sources = dataset.restrict_sources(
                cap=configs.run.source_cap or len(dataset.eligible_source_iris),
                seed=configs.seed,
            )
            logger.info(
                "Development source cap retained %d source groups (cap=%d, seed=%d).",
                len(sampled_sources),
                configs.run.source_cap,
                configs.seed,
            )
            sampled_manifest = dataset.candidate_pool_manifest
            sampled_manifest_path = (
                Path(dataset.output_path) / "candidate_pool_sample_manifest.json"
            )
            temporary_manifest = sampled_manifest_path.with_name(
                f".{sampled_manifest_path.name}.{os.getpid()}.tmp"
            )
            temporary_manifest.write_text(
                json.dumps(sampled_manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary_manifest, sampled_manifest_path)

            sampled_frame = dataset.dataframe
            if (
                sampled_frame is not None
                and not sampled_frame.empty
                and {"Src", "SrcKind"}.issubset(sampled_frame.columns)
            ):
                sampled_source_groups = {
                    (str(src), str(kind))
                    for src, kind in sampled_frame[["Src", "SrcKind"]]
                    .drop_duplicates()
                    .itertuples(index=False, name=None)
                }

            if getattr(dataset, "eligible_source_groups", None) is not None:
                sampled_source_groups = set(dataset.eligible_source_groups)

            def sampled_mask(frame: Any) -> Any:
                source_column = (
                    "Src"
                    if "Src" in frame.columns
                    else "SrcEntity" if "SrcEntity" in frame.columns else frame.columns[0]
                )
                kind_column = (
                    "SrcKind"
                    if "SrcKind" in frame.columns
                    else "Kind" if "Kind" in frame.columns else None
                )
                if kind_column is not None and sampled_source_groups:
                    return [
                        (str(src), str(kind)) in sampled_source_groups
                        for src, kind in frame[[source_column, kind_column]].itertuples(
                            index=False,
                            name=None,
                        )
                    ]
                return frame[source_column].astype(str).isin(sampled_sources)

            sampled_inputs_dir = Path(dataset.output_path) / "sampled_inputs"
            if candidates_file_path is not None:
                candidate_frame = read_table(candidates_file_path)
                candidate_frame = candidate_frame.loc[sampled_mask(candidate_frame)].reset_index(
                    drop=True
                )
                effective_candidates_file_path = sampled_inputs_dir / "reference_candidates.tsv"
                _atomic_table(candidate_frame, effective_candidates_file_path)
            if dataset.reference is not None and full_reference_file_path is not None:
                sampled_full_reference_path = _materialize_evaluation_reference(
                    dataset.reference,
                    parent_path=full_reference_file_path,
                    output_path=sampled_inputs_dir / "full_reference.tsv",
                )
                evaluation_full_reference = sampled_full_reference_path
            if training_reference_file_path is not None:
                training_frame = read_table(training_reference_file_path)
                training_frame = training_frame.loc[sampled_mask(training_frame)].reset_index(
                    drop=True
                )
                sampled_training_reference_path = sampled_inputs_dir / "training_reference.tsv"
                _atomic_table(training_frame, sampled_training_reference_path)
                evaluation_training_reference = sampled_training_reference_path

        if (
            run_eval
            and configs.run.experiment_audit
            and dataset.reference is not None
            and full_reference_file_path is not None
        ):
            materialized_full_reference_path = _materialize_evaluation_reference(
                dataset.reference,
                parent_path=full_reference_file_path,
                output_path=(Path(output_dir_path) / "evaluation_inputs" / "full_reference.tsv"),
            )
            evaluation_full_reference = materialized_full_reference_path

        timing_session.set_dataset_signature(getattr(dataset, "dataset_signature", None))

    dataset_elapsed = (dataset_span.seconds or 0.0) / 60.0
    logger.info(f"Dataset built in {dataset_elapsed:.1f} minutes")
    progress.finish("Dataset", f"loaded_from_cache={dataset_loaded_from_cache}")

    # Trainer module

    # Train model
    progress.start("Trainer", "constructing trainer and model chain")
    logger.info("Building Trainer and Model...")

    training_available = training_reference_file_path is not None
    llm_supervision, _ = configs.supervision.resolve_component(
        "llm",
        training_available=training_available,
        profile_binding={"dataset_signature": getattr(dataset, "dataset_signature", None)},
    )
    selector_supervision = {
        component: configs.supervision.resolve_component(
            component,
            training_available=training_available,
            profile_binding={"dataset_signature": getattr(dataset, "dataset_signature", None)},
        )[0]
        for component in ("rerank", "accept", "calibration")
    }

    model_specs = []
    primary = model_sequence[0]
    primary_params = {
        **primary.params,
        **configs.matching.channels.model_dump(mode="python"),
        "fusion_config": configs.matching.fusion.model_dump(mode="python"),
        "llm_experiment_config": configs.llm.experiment.model_dump(mode="python"),
        "llm_profiles": {k: v.model_dump() for k, v in configs.llm_profiles.items()},
        "llm_routing": configs.llm_routing.model_dump(),
        "request_seed": configs.seed,
        **configs.alignment_params.model_dump(exclude_none=True),
    }
    if training_reference_file_path is not None and llm_supervision == "supervised":
        primary_params.setdefault(
            "llm_calibration_reference_file_path",
            str(training_reference_file_path),
        )
    fitting_graph_config = None
    graph = primary_params.get("graph", {})
    if configs.data.train_candidates is not None and graph.get("mode") in {
        "inductive",
        "graph_only",
    }:
        if not graph.get("artifact") or not Path(graph["artifact"]).is_file():
            fitting_graph_config = dict(graph)
            primary_params["graph"] = {**graph, "mode": "features", "artifact": None}
            if graph["mode"] == "graph_only":
                primary_params.update(use_lexical=False, use_context=False, use_llm=False)
                primary_params["strsim"] = {**primary_params.get("strsim", {}), "enabled": False}
    fitting_llm_config = None
    experiment = primary_params["llm_experiment_config"]
    pending_learning = (
        (
            experiment.get("exemplars") == "knn"
            and (
                not experiment.get("exemplar_artifact")
                or not Path(experiment["exemplar_artifact"]).is_file()
            )
        )
        or (
            experiment.get("distill") == "student"
            and (
                not experiment.get("distill_artifact")
                or not Path(experiment["distill_artifact"]).is_file()
            )
        )
        or (
            experiment["gate"]["mode"] == "learned"
            and (
                not experiment["gate"].get("artifact")
                or not Path(experiment["gate"]["artifact"]).is_file()
            )
        )
    )
    if experiment.get("enabled") and configs.data.train_candidates is not None and pending_learning:
        fitting_llm_config = {**experiment, "gate": dict(experiment["gate"])}
        primary_params["llm_experiment_config"] = {
            **experiment,
            "exemplars": "off",
            "distill": "off",
        }
        if experiment["gate"]["mode"] == "learned":
            primary_params["llm_experiment_config"]["gate"] = {
                **experiment["gate"],
                "mode": "off",
                "artifact": None,
            }
    fitting_fusion_config = None
    if configs.data.train_candidates is not None and primary_params["fusion_config"].get(
        "mode"
    ) in {"analytic_fitted", "learned_global"}:
        artifact = primary_params["fusion_config"].get("artifact")
        if not artifact or not Path(artifact).is_file():
            fitting_fusion_config = dict(primary_params["fusion_config"])
            primary_params["fusion_config"] = {
                **fitting_fusion_config,
                "mode": "analytic_shipped",
                "artifact": None,
            }
    fitting_gate_config = None
    gate = primary_params["llm_experiment_config"]["gate"]
    if primary_params["llm_experiment_config"].get("enabled") and gate["mode"] in {
        "source_top_fraction",
        "pair_top_fraction",
    }:
        artifact = gate.get("artifact")
        if not artifact or not Path(artifact).is_file():
            fitting_gate_config = dict(gate)
            primary_params["llm_experiment_config"]["gate"] = {
                **gate,
                "mode": "off",
                "artifact": None,
            }
    model_specs.append((primary.name, primary_params))
    for extra in model_sequence[1:]:
        if extra.name is None:
            continue
        if isinstance(extra.params, dict) and extra.params.get("enabled") is False:
            continue
        extra_params = dict(extra.params or {})
        model_name = getattr(extra.name, "__name__", "")
        selector_uses_labels = any(mode == "supervised" for mode in selector_supervision.values())
        if (
            training_reference_file_path is not None
            and selector_uses_labels
            and model_name in {"CandidateSetSelector", "SecondPassReranker"}
        ):
            extra_params.setdefault(
                "training_reference_file_path",
                str(training_reference_file_path),
            )
        if model_name == "CandidateSetSelector":
            if configs.selector.runtime_enabled is not None:
                extra_params["enabled"] = configs.selector.runtime_enabled
            extra_params.setdefault("request_seed", configs.seed)
            extra_params.setdefault("experiment_config", configs.selector.model_dump(mode="python"))
            extra_params.setdefault(
                "matching_calibration",
                configs.matching.calibration.model_dump(mode="python"),
            )
            extra_params.setdefault("nil_config", configs.matching.nil.model_dump(mode="python"))
            if (
                configs.data.train_candidates is not None
                and extra_params["matching_calibration"]["mode"] != "none"
                and not extra_params["matching_calibration"].get("artifact")
            ):
                extra_params["matching_calibration"]["artifact"] = str(
                    Path(output_dir_path) / "fitting" / "score_calibrator.json"
                )
        model_specs.append((extra.name, extra_params))

    trainer_factory = configs.trainer_runtime
    if trainer_factory is None:
        raise RuntimeError("Trainer dependency was not resolved")
    trainer = trainer_factory(
        dataset=dataset,
        models=model_specs,
        device=device,
        output_dir=output_dir_path,
        logger=logger,
        extraction_config=configs.matching.extraction.model_dump(mode="python"),
        training_candidates_file_path=configs.data.train_candidates,
        fitting_fusion_config=fitting_fusion_config,
        fitting_graph_config=fitting_graph_config,
        fitting_llm_config=fitting_llm_config,
        training_reference_file_path=training_reference_file_path,
        fitting_gate_config=fitting_gate_config,
        supervision_config=configs.supervision.model_dump(mode="python"),
    )
    progress.finish("Trainer", f"models={len(model_specs)}")

    logger.info("Computing alignment...")
    inference_kwargs = configs.inference_params.model_dump()
    inference_kwargs["local_alignment"] = local_ranking
    inference_kwargs["explanation_shard_mb"] = configs.output.explanations.shard_mb

    with timing_session.stage("Alignment") as alignment_span:
        progress.start("Inference", f"pairs={len(dataset)}")
        inference_kwargs["run_progress"] = progress
        if not local_ranking:
            inference_kwargs.update(configs.alignment_params.model_dump())
        alignment, avg_t = trainer.predict(**inference_kwargs)
        if getattr(progress, "fractions", {}).get("Inference", 0.0) < 1.0:
            progress.finish("Inference", f"avg={avg_t:.4f}s/example")
        if (
            has_post_inference
            and getattr(progress, "fractions", {}).get("PostInference", 0.0) < 1.0
        ):
            progress.finish("PostInference", "post-inference completed")

        trainer_stage_records = list(getattr(trainer, "last_stage_timings", []) or [])
        for stage_record in trainer_stage_records:
            timing_session.record(stage_record)
        inference_record = next(
            (record for record in trainer_stage_records if record.stage == "Alignment.Inference"),
            None,
        )
        if inference_record is None:
            raise RuntimeError("Trainer did not report an Alignment.Inference timing record")
        if inference_record.cache_status in {
            CacheStatus.RESUMED,
            CacheStatus.SKIPPED,
        }:
            alignment_span.cache_status = CacheStatus.RESUMED

        effective_threshold = getattr(
            trainer,
            "last_effective_threshold",
            inference_kwargs.get("threshold"),
        )
        threshold_origin = getattr(
            trainer,
            "last_effective_threshold_origin",
            "configured",
        )
        logger.info(
            "Effective decision threshold: %s (origin: %s)",
            "disabled" if effective_threshold is None else f"{effective_threshold:.6g}",
            threshold_origin,
        )
        logger.info(f"Average inference time per example: {avg_t:.4f} seconds")

        if dataset.filter_exact_matches:
            progress.start("Prefilter", "applying exact matches")
            logger.info("Applying Exact Matches to alignment...")
            with timing_session.stage("Alignment.Prefilter"):
                if not local_ranking:
                    alignment = trainer.apply_prefilter(
                        alignment, **configs.alignment_params.model_dump()
                    )
                else:
                    alignment = trainer.apply_prefilter(alignment)
            progress.finish("Prefilter", f"mappings={len(alignment)}")
        else:
            timing_session.record(
                "Alignment.Prefilter",
                seconds=0.0,
                cache_status=CacheStatus.SKIPPED,
            )

    alignment_elapsed = (alignment_span.seconds or 0.0) / 60.0
    logger.info(f"Alignment computed in {alignment_elapsed:.1f} minutes")

    # Save Alignment

    progress.start("Outputs", "writing alignment artifacts")
    logger.info("Writing alignment...")
    with timing_session.stage("Postprocess.Outputs") as outputs_span:
        save_params = configs.alignment_params.model_dump()
        # The monolithic explanation JSON is a derived v2 export assembled after
        # overlays are compacted, rather than a second write-time source of truth.
        save_params["save_json"] = False
        output_paths = trainer.save_results(
            alignment,
            candidates_one2many_path=(effective_candidates_file_path if local_ranking else None),
            sub_dir=task_name,
            output_formats=configs.io.output_formats,
            relation_prediction=configs.matching.relation_prediction,
            relation_semantic_backend=configs.matching.relation_semantic_backend,
            relation_equivalence_anchor_threshold=(
                configs.matching.relation_equivalence_anchor_threshold
            ),
            relation_equivalence_anchor_margin=(
                configs.matching.relation_equivalence_anchor_margin
            ),
            relation_confidence_threshold=configs.matching.relation_confidence_threshold,
            relation_reasoning_timeout_seconds=(
                configs.matching.relation_reasoning_timeout_seconds
            ),
            source_uri=source_file_path.resolve().as_uri(),
            target_uri=target_file_path.resolve().as_uri(),
            paper_audit=configs.run.experiment_audit,
            **save_params,
        )
    alignment_file_path = output_paths["alignment_tsv"]
    evaluation_alignment_path = output_paths.get("alignment_global_audit", alignment_file_path)
    run_stats_path = output_paths.get("run_stats_json") or (
        Path(alignment_file_path).parent / "run_stats.json"
    )

    logger.info(f"Alignment written to {alignment_file_path}")
    progress.finish("Outputs", str(alignment_file_path))

    # Plot Distributions
    progress.start("Plots", "writing plots")
    with timing_session.stage("Postprocess.Plotting") as plotting_span:
        trainer.plot_distributions(
            which=configs.inference_params.which,
            **configs.plot_params.model_dump(),
        )
        trainer.plot_scores_vs_labels(
            which=configs.inference_params.which,
            figsize=configs.plot_params.figsize,
            alpha=configs.plot_params.alpha,
            dpi=configs.plot_params.dpi,
        )
    progress.finish("Plots", "plots written")

    # Evaluate Alignment

    results = None

    if run_eval:
        progress.start("Evaluation", "evaluating alignment")
        logger.info("Evaluating alignment...")
        with timing_session.stage("Postprocess.Evaluation") as evaluation_span:
            evaluation_dir = RunLayout.open(output_dir_path).evaluation_dir
            backend_options = {
                "builtin": {"entity_kinds": configs.matching.entity_kinds},
                "bioml": configs.evaluation.bioml,
            }
            if (
                configs.evaluation.dual_global_local
                and effective_candidates_file_path is not None
                and evaluation_full_reference is not None
            ):
                global_results = run_evaluation(
                    alignment=Path(evaluation_alignment_path),
                    output_dir_path=evaluation_dir,
                    error_on_fail=False,
                    K=configs.k,
                    source_file_path=dataset.source,
                    target_file_path=dataset.target,
                    train_reference_file_path=evaluation_training_reference,
                    full_reference_file_path=evaluation_full_reference,
                    reference_candidates=None,
                    logger=logger,
                    backends=configs.evaluation.backends,
                    backend_options=backend_options,
                    run_stats_path=run_stats_path,
                )
                local_results = run_evaluation(
                    alignment=Path(alignment_file_path),
                    output_dir_path=evaluation_dir / "local",
                    error_on_fail=False,
                    K=configs.k,
                    source_file_path=dataset.source,
                    target_file_path=dataset.target,
                    train_reference_file_path=None,
                    full_reference_file_path=None,
                    reference_candidates=(
                        effective_candidates_file_path if local_ranking else None
                    ),
                    logger=logger,
                    backends=configs.evaluation.backends,
                    backend_options=backend_options,
                    run_stats_path=None,
                )
                results = {
                    **{f"global.{key}": value for key, value in (global_results or {}).items()},
                    **{f"local.{key}": value for key, value in (local_results or {}).items()},
                }
            else:
                results = run_evaluation(
                    alignment=Path(
                        evaluation_alignment_path
                        if not local_ranking and evaluation_full_reference is not None
                        else alignment_file_path
                    ),
                    output_dir_path=evaluation_dir,
                    error_on_fail=False,
                    K=configs.k,
                    source_file_path=dataset.source,
                    target_file_path=dataset.target,
                    train_reference_file_path=evaluation_training_reference,
                    full_reference_file_path=(
                        evaluation_full_reference if not local_ranking else None
                    ),
                    reference_candidates=(
                        effective_candidates_file_path if local_ranking else None
                    ),
                    logger=logger,
                    backends=configs.evaluation.backends,
                    backend_options=backend_options,
                    run_stats_path=run_stats_path,
                )
        progress.finish("Evaluation", "evaluation completed")
    else:
        evaluation_span = None
        timing_session.record(
            "Postprocess.Evaluation",
            seconds=0.0,
            cache_status=CacheStatus.SKIPPED,
        )

    rationale_seconds = sum(
        record.seconds
        for record in trainer_stage_records
        if record.stage == "Postprocess.Rationales"
    )
    postprocess_seconds = (
        rationale_seconds
        + (outputs_span.seconds or 0.0)
        + (plotting_span.seconds or 0.0)
        + ((evaluation_span.seconds or 0.0) if evaluation_span is not None else 0.0)
    )
    timing_session.record(
        "Postprocess",
        seconds=postprocess_seconds,
        cache_status=CacheStatus.FRESH,
    )
    run_metadata: dict[str, Any] = {"ontology_stack": dataset.ontology_stack_provenance()}
    if configs.run.experiment_audit and evaluation_alignment_path is not None:
        run_metadata.update(
            _candidate_recall_run_stats(
                dataset,
                training_reference_path=(
                    Path(evaluation_training_reference)
                    if evaluation_training_reference is not None
                    else None
                ),
            )
        )
        run_metadata["evaluation_inputs"] = {
            "alignment_primary": file_provenance(Path(alignment_file_path)),
            "alignment_global_audit": file_provenance(Path(evaluation_alignment_path)),
            "full_reference_parent": (
                file_provenance(full_reference_file_path)
                if full_reference_file_path is not None
                else None
            ),
            "full_reference_materialized": (
                file_provenance(materialized_full_reference_path)
                if materialized_full_reference_path is not None
                else None
            ),
            "training_reference_parent": (
                file_provenance(training_reference_file_path)
                if training_reference_file_path is not None
                else None
            ),
            "training_reference_effective": (
                file_provenance(Path(evaluation_training_reference))
                if evaluation_training_reference is not None
                else None
            ),
        }
    if configs.run.source_cap is not None or configs.data.source_universe is not None:
        sample_config = (dataset.candidate_pool_manifest.get("retrieval_config") or {}).get(
            "source_sample"
        ) or {}
        reference_inputs = {
            "full_parent": (
                file_provenance(full_reference_file_path)
                if full_reference_file_path is not None
                else None
            ),
            "full_sample": (
                file_provenance(sampled_full_reference_path)
                if sampled_full_reference_path is not None
                else None
            ),
            "training_parent": (
                file_provenance(training_reference_file_path)
                if training_reference_file_path is not None
                else None
            ),
            "training_sample": (
                file_provenance(sampled_training_reference_path)
                if sampled_training_reference_path is not None
                else None
            ),
        }
        run_metadata["source_sampling"] = {
            "cap": configs.run.source_cap,
            "seed": configs.seed,
            "selected_sources": len(sampled_sources),
            "selected_source_kind_groups": len(sampled_source_groups),
            "sample_sha256": sample_config.get("sha256"),
            "candidate_pool_fingerprint": dataset.candidate_pool_fingerprint,
            "reference_inputs": reference_inputs,
        }
    run_metadata["execution"] = {
        "mode": execution_mode,
        "explicit": declared_mode is not None,
        "candidate_provenance": configs.data.candidate_provenance
        or ("benchmark_supplied" if effective_candidates_file_path else "generated"),
        "executed_stages": [record.stage for record in trainer_stage_records],
        "global_extraction": not local_ranking,
    }
    _merge_run_stats(run_stats_path, run_metadata)
    progress.complete("run stages completed")
    return results, run_stats_path


_DEFAULT_RUN_ALIGNMENT_SESSION = _run_alignment_session


class AlignmentAction:
    """Deprecated namespace compatibility for the functional action API."""

    resolve_inputs = staticmethod(resolve_alignment_inputs)
    _run_session = staticmethod(_run_alignment_session)

    @staticmethod
    @functools.wraps(run_alignment)
    def run(*args, **kwargs):
        warnings.warn(
            "AlignmentAction.run is deprecated; use run_alignment instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return run_alignment(*args, **kwargs)


__all__ = [
    "AlignmentAction",
    "ResolvedAlignmentInputs",
    "resolve_alignment_inputs",
    "run_alignment",
]
