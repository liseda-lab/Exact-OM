import json
import logging
from pathlib import Path
from typing import Optional, Protocol, Union

import torch

from exact.core.actions.evaluation import EvaluationAction
from exact.core.entities.configs.config import ConfigModel
from exact.impl import bootstrap_components
from exact.impl.seed import SeedSetter
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


class AlignmentAction(Protocol):
    @staticmethod
    def run(
        source_file_path: Path,
        target_file_path: Path,
        output_dir_path: Path,
        configs_file_path: Optional[Union[Path, ConfigModel]] = None,
        training_reference_file_path: Optional[Path] = None,
        full_reference_file_path: Optional[Path] = None,
        candidates_file_path: Optional[Path] = None,
        log_file_path: Optional[Path] = None,
        run_eval: bool = False,
        task_name: Optional[str] = None,
        device: Optional[int] = None,
    ) -> Optional[dict]:

        bootstrap_components()
        if configs_file_path is None:
            configs = ConfigModel()
        elif isinstance(configs_file_path, ConfigModel):
            configs = configs_file_path
        else:
            configs = ConfigModel.load_config(configs_file_path)
        configs.resolve_dependencies()

        fingerprint = config_fingerprint(configs, run_dir=output_dir_path)
        ledger = TimingLedger.open(output_dir_path)
        with ledger.session(
            command="align",
            config_fingerprint=fingerprint,
        ) as timing_session:
            with timing_session.stage("Total") as total_span:
                results, run_stats_path = AlignmentAction._run_session(
                    source_file_path=source_file_path,
                    target_file_path=target_file_path,
                    output_dir_path=output_dir_path,
                    configs=configs,
                    configs_source=configs_file_path,
                    training_reference_file_path=training_reference_file_path,
                    full_reference_file_path=full_reference_file_path,
                    candidates_file_path=candidates_file_path,
                    log_file_path=log_file_path,
                    run_eval=run_eval,
                    task_name=task_name,
                    device=device,
                    timing_ledger=ledger,
                    timing_session=timing_session,
                )

        totals = ledger.stage_totals(config_fingerprint=fingerprint)
        timings_result = {
            stage: (
                total.compute_seconds if total.compute_seconds > 0.0 else total.overhead_seconds
            )
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
        cumulative_compute_seconds = {
            stage: total.compute_seconds for stage, total in totals.items()
        }
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
                stats = json.loads(run_stats_path.read_text(encoding="utf-8"))
                stats["timing"] = timing_stats
                tmp_path = run_stats_path.with_suffix(run_stats_path.suffix + ".tmp")
                tmp_path.write_text(
                    json.dumps(stats, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                tmp_path.replace(run_stats_path)
            except (OSError, json.JSONDecodeError) as exc:
                logging.getLogger("exact").warning(
                    "Could not add timing summary to %s: %s",
                    run_stats_path,
                    exc,
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
        return results, timings_result

    @staticmethod
    def _run_session(
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
            progress_tasks.append(
                ProgressTask("Prefilter", "Exact prefilter", estimate_seconds=30.0)
            )
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
            SeedSetter(configs.seed)

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
            dataset = configs.dataset(
                output_path=output_dir_path,
                logger=logger,
                cache_ok=configs.use_file_cache,
                device=device,
                llm_profiles={k: v.model_dump() for k, v in configs.llm_profiles.items()},
                llm_routing=configs.llm_routing.model_dump(),
                request_seed=configs.seed,
                candidate_generation_params={
                    **configs.candidates_params.model_dump(),
                    "candidates_file_path": (
                        str(candidates_file_path) if candidates_file_path else None
                    ),
                },
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
                        **configs.candidates_params.model_dump(),
                    )
                with timing_session.stage("Dataset.Process"):
                    dataset.process()
                with timing_session.stage("Dataset.Save"):
                    dataset.save()

                if getattr(dataset, "emit_feature_metrics_on_build", lambda: False)():
                    dataset.save_feature_metrics()

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

        trainer = configs.trainer(
            dataset=dataset,
            models=model_specs,
            device=device,
            output_dir=output_dir_path / "model",
            logger=logger,
        )
        progress.finish("Trainer", f"models={len(model_specs)}")

        logger.info("Computing alignment...")
        inference_kwargs = configs.inference_params.model_dump()
        inference_kwargs["local_alignment"] = candidates_file_path is not None

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
                (
                    record
                    for record in trainer_stage_records
                    if record.stage == "Alignment.Inference"
                ),
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
            output_paths = trainer.save_results(
                alignment,
                candidates_one2many_path=candidates_file_path,
                sub_dir=task_name,
                **configs.alignment_params.model_dump(),
            )
        alignment_file_path = output_paths["alignment_tsv"]
        run_stats_path = output_paths.get("run_stats_json")

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
                results = EvaluationAction.run(
                    alignment=Path(alignment_file_path),
                    output_dir_path=(
                        output_dir_path / task_name if task_name is not None else output_dir_path
                    ),
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
                    backend_options={"bioml": configs.evaluation.bioml},
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
