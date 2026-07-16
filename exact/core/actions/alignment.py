import functools
import json
import logging
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import torch

from exact.core.actions.evaluation import run_evaluation
from exact.core.entities.configs.config import ConfigModel
from exact.core.entities.registry import ComponentRegistry, ComponentType
from exact.runs import RunLayout, finalize_artifacts
from exact.tracks import get_track, provider_from_descriptor
from exact.utils.logs import (
    ProgressTask,
    RunProgressLogger,
    configure_exact_logger,
    summarize_progress_estimates,
)
from exact.utils.timing import (
    CacheStatus,
    RunSession,
    TimingLedger,
    config_fingerprint,
)


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
    full_reference = (
        _resolved_path(full_reference_file_path)
        or refs.get("full")
        or refs.get("test")
        or refs.get("valid")
    )
    candidates = (
        _resolved_path(candidates_file_path)
        or configured_candidates
        or (layout.candidates if layout is not None else None)
    )

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

        timing_session.set_dataset_signature(getattr(dataset, "dataset_signature", None))

    dataset_elapsed = (dataset_span.seconds or 0.0) / 60.0
    logger.info(f"Dataset built in {dataset_elapsed:.1f} minutes")
    progress.finish("Dataset", f"loaded_from_cache={dataset_loaded_from_cache}")

    # Trainer module

    # Train model
    progress.start("Trainer", "constructing trainer and model chain")
    logger.info("Building Trainer and Model...")

    model_specs = []
    primary = model_sequence[0]
    primary_params = {
        **primary.params,
        **configs.matching.channels.model_dump(mode="python"),
        "llm_profiles": {k: v.model_dump() for k, v in configs.llm_profiles.items()},
        "llm_routing": configs.llm_routing.model_dump(),
        "request_seed": configs.seed,
        **configs.alignment_params.model_dump(exclude_none=True),
    }
    if training_reference_file_path is not None:
        primary_params.setdefault(
            "llm_calibration_reference_file_path",
            str(training_reference_file_path),
        )
    model_specs.append((primary.name, primary_params))
    for extra in model_sequence[1:]:
        if extra.name is None:
            continue
        if isinstance(extra.params, dict) and extra.params.get("enabled") is False:
            continue
        extra_params = dict(extra.params or {})
        if training_reference_file_path is not None and getattr(extra.name, "__name__", "") in {
            "CandidateSetSelector",
            "SecondPassReranker",
        }:
            extra_params.setdefault(
                "training_reference_file_path",
                str(training_reference_file_path),
            )
        if getattr(extra.name, "__name__", "") == "CandidateSetSelector":
            extra_params.setdefault("request_seed", configs.seed)
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
    )
    progress.finish("Trainer", f"models={len(model_specs)}")

    logger.info("Computing alignment...")
    inference_kwargs = configs.inference_params.model_dump()
    inference_kwargs["local_alignment"] = candidates_file_path is not None
    inference_kwargs["explanation_shard_mb"] = configs.output.explanations.shard_mb

    with timing_session.stage("Alignment") as alignment_span:
        progress.start("Inference", f"pairs={len(dataset)}")
        inference_kwargs["run_progress"] = progress
        if candidates_file_path is None:
            inference_kwargs.update(configs.alignment_params.model_dump())
            alignment, avg_t = trainer.predict(**inference_kwargs)
        else:
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
                if candidates_file_path is None:
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
            candidates_one2many_path=candidates_file_path,
            sub_dir=task_name,
            output_formats=configs.io.output_formats,
            relation_prediction=configs.matching.relation_prediction,
            source_uri=source_file_path.resolve().as_uri(),
            target_uri=target_file_path.resolve().as_uri(),
            **save_params,
        )
    alignment_file_path = output_paths["alignment_tsv"]
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
            results = run_evaluation(
                alignment=Path(alignment_file_path),
                output_dir_path=RunLayout.open(output_dir_path).evaluation_dir,
                error_on_fail=False,
                K=configs.k,
                source_file_path=dataset.source,
                target_file_path=dataset.target,
                train_reference_file_path=training_reference_file_path,
                full_reference_file_path=(
                    full_reference_file_path if candidates_file_path is None else None
                ),
                reference_candidates=candidates_file_path,
                logger=logger,
                backends=configs.evaluation.backends,
                backend_options={
                    "builtin": {"entity_kinds": configs.matching.entity_kinds},
                    "bioml": configs.evaluation.bioml,
                },
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
