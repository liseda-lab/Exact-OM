import logging
import time
from pathlib import Path
from typing import Optional, Protocol, List, Union
import pandas as pd

from exact.core.actions.evaluation import EvaluationAction
from exact.core.entities.configs.config import ConfigModel
from exact.impl.seed import SeedSetter
from exact.utils.logs import (
    ProgressTask,
    RunProgressLogger,
    configure_exact_logger,
    summarize_progress_estimates,
)
from exact.utils.timing import load_recorded_timings, update_recorded_timings, write_recorded_timings

import torch

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

        start_time = time.time()
        times_file_path = output_dir_path / "times.txt"
        stage_timings: dict[str, float] = {}

        # Load Configs

        if configs_file_path is not None:
            if isinstance(configs_file_path, ConfigModel):
                configs = configs_file_path
            else:
                configs = ConfigModel.load_config(configs_file_path)

        else:
            configs = ConfigModel()

        # Loading logging configuration from configs

        logger = configure_exact_logger(
            logging.getLogger("exact"),
            configs.logging_level,
            log_file_path=log_file_path,
        )

        logger.debug(f"Logging level set to {configs.logging_level}")

        # log configs state

        if configs_file_path is not None:
            logger.info(f"Using configuration from {configs_file_path}")
        else:
            logger.info(f"Using default configuration")

        # Resolve dependencies
        configs.resolve_dependencies()
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
            progress_tasks.append(ProgressTask("PostInference", "Post-inference", estimate_seconds=60.0))
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
            estimates_minutes=summarize_progress_estimates(load_recorded_timings(times_file_path)),
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
            logger.warning(f"CUDA device specified but not available. Using CPU instead.")

        device = torch.device(device) if device is not None and torch.cuda.is_available() else torch.device('cpu')
        progress.finish("Setup", f"device={device}")

        # Create Dataset

        progress.start("Dataset", "building dataset inputs")
        logger.info(f'Building Dataset...')
        dataset_start = time.time()
        dataset_stage_timings: dict[str, float] = {}

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
                "candidates_file_path": str(candidates_file_path) if candidates_file_path else None,
            },
            **configs.dataset_params.model_dump(),
        )

        dataset_step_start = time.time()
        dataset.load_ontologies(source_file_path, target_file_path)
        dataset_stage_timings["Dataset.LoadOntologies"] = (time.time() - dataset_step_start) / 60
        dataset_loaded_from_cache = dataset.has_cache()

        if not dataset_loaded_from_cache:
            dataset_step_start = time.time()

            if full_reference_file_path is not None:
                dataset.load_reference(full_reference_file_path)

            dataset.load_candidates(candidates_file_path, device=device, **configs.candidates_params.model_dump())
            dataset_stage_timings["Dataset.LoadCandidates"] = (time.time() - dataset_step_start) / 60

        dataset_step_start = time.time()
        dataset.process()
        dataset_stage_timings["Dataset.Process"] = (time.time() - dataset_step_start) / 60

        if not dataset_loaded_from_cache:

            dataset_step_start = time.time()
            dataset.save()
            dataset_stage_timings["Dataset.Save"] = (time.time() - dataset_step_start) / 60

            if getattr(dataset, "emit_feature_metrics_on_build", lambda: False)():
                dataset_step_start = time.time()
                dataset.save_feature_metrics()
                dataset_stage_timings["Dataset.FeatureMetrics"] = (time.time() - dataset_step_start) / 60

            dataset_step_start = time.time()
            dataset.log_sanity_examples(**configs.sanity_check_params.model_dump())
            dataset_stage_timings["Dataset.Sanity"] = (time.time() - dataset_step_start) / 60

            dataset_step_start = time.time()
            dataset.plot_feature_distributions(
                which=configs.dataset_params.which,
                **configs.plot_params.model_dump()
            )
            dataset_stage_timings["Dataset.Plotting"] = (time.time() - dataset_step_start) / 60

        dataset_end = time.time()
        dataset_elapsed = (dataset_end - dataset_start) / 60
        logger.info(f"Dataset built in {dataset_elapsed:.1f} minutes")
        if dataset_loaded_from_cache:
            stage_timings["Dataset.CacheLoad"] = dataset_elapsed
        else:
            stage_timings["Dataset"] = dataset_elapsed
            stage_timings.update(dataset_stage_timings)
        update_recorded_timings(times_file_path, stage_timings)
        logger.debug(f"Persisted dataset timings to {times_file_path}")
        progress.finish("Dataset", f"loaded_from_cache={dataset_loaded_from_cache}")


        # Trainer module

        ## Train Model
        progress.start("Trainer", "constructing trainer and model chain")
        logger.info(f"Building Trainer and Model...")

        model_specs = []
        primary = model_sequence[0]
        primary_params = {
            **primary.params,
            "llm_profiles": {k: v.model_dump() for k, v in configs.llm_profiles.items()},
            "llm_routing": configs.llm_routing.model_dump(),
            "request_seed": configs.seed,
            **configs.alignment_params.model_dump(exclude_none=True),
        }
        model_specs.append((primary.name, primary_params))
        for extra in model_sequence[1:]:
            if extra.name is None:
                continue
            if isinstance(extra.params, dict) and extra.params.get("enabled") is False:
                continue
            extra_params = dict(extra.params or {})
            if (
                training_reference_file_path is not None
                and getattr(extra.name, "__name__", "") in {"CandidateSetSelector", "SecondPassReranker"}
            ):
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
            output_dir= output_dir_path / "model",
            logger=logger,
        )
        progress.finish("Trainer", f"models={len(model_specs)}")

        logger.info(f"Computing alignment...")

        alignment_core_minutes = 0.0

        inference_kwargs = configs.inference_params.model_dump()
        inference_kwargs["local_alignment"] = candidates_file_path is not None

        predict_start = time.time()
        progress.start("Inference", f"pairs={len(dataset)}")
        inference_kwargs["run_progress"] = progress
        if candidates_file_path is None:
            inference_kwargs.update(configs.alignment_params.model_dump())
            alignment, avg_t = trainer.predict(**inference_kwargs)
        else:
            alignment, avg_t = trainer.predict(**inference_kwargs)
        if getattr(progress, "fractions", {}).get("Inference", 0.0) < 1.0:
            progress.finish("Inference", f"avg={avg_t:.4f}s/example")
        if has_post_inference and getattr(progress, "fractions", {}).get("PostInference", 0.0) < 1.0:
            progress.finish("PostInference", "post-inference completed")
        predict_elapsed = (time.time() - predict_start) / 60
        trainer_stage_timings = getattr(trainer, "last_stage_timings", {}) or {}
        inference_minutes = trainer_stage_timings.get("Alignment.Inference", predict_elapsed)
        post_inference_minutes = trainer_stage_timings.get("Alignment.PostInference", 0.0)
        rationale_minutes = trainer_stage_timings.get("Postprocess.Rationales", 0.0)
        stage_timings["Alignment.Inference"] = inference_minutes
        if post_inference_minutes > 0:
            stage_timings["Alignment.PostInference"] = post_inference_minutes
        if rationale_minutes > 0:
            stage_timings["Postprocess.Rationales"] = rationale_minutes
        alignment_core_minutes += inference_minutes + post_inference_minutes

        logger.info(f"Average inference time per example: {avg_t:.4f} seconds")

        if dataset.filter_exact_matches:
            progress.start("Prefilter", "applying exact matches")
            logger.info(f"Applying Exact Matches to alignment...")
            prefilter_start = time.time()

            if candidates_file_path is None:
                alignment = trainer.apply_prefilter(alignment, **configs.alignment_params.model_dump())

            else:
                alignment = trainer.apply_prefilter(alignment)
            prefilter_minutes = (time.time() - prefilter_start) / 60
            stage_timings["Alignment.Prefilter"] = prefilter_minutes
            alignment_core_minutes += prefilter_minutes
            progress.finish("Prefilter", f"mappings={len(alignment)}")

        alignment_elapsed = alignment_core_minutes
        stage_timings["Alignment"] = alignment_elapsed
        update_recorded_timings(times_file_path, stage_timings)
        logger.debug(f"Persisted alignment timings to {times_file_path}")
        logger.info(f"Alignment computed in {alignment_elapsed:.1f} minutes")

        # Save Alignment
        
        progress.start("Outputs", "writing alignment artifacts")
        logger.info(f"Writing alignment...")
        save_results_start = time.time()
        alignment_file_path = trainer.save_results(alignment, 
                                                     candidates_one2many_path=candidates_file_path, 
                                                     sub_dir=task_name,
                                                     **configs.alignment_params.model_dump()
                                                     )["alignment_tsv"]
        stage_timings["Postprocess.Outputs"] = (time.time() - save_results_start) / 60
        stage_timings["Postprocess"] = (
            stage_timings.get("Postprocess.Rationales", 0.0)
            + stage_timings.get("Postprocess.Outputs", 0.0)
        )
        update_recorded_timings(times_file_path, stage_timings)
        logger.debug(f"Persisted output timings to {times_file_path}")

        logger.info(f"Alignment written to {alignment_file_path}")
        progress.finish("Outputs", str(alignment_file_path))

        # Plot Distributions
        progress.start("Plots", "writing plots")
        plot_start = time.time()
        trainer.plot_distributions(
            which=configs.inference_params.which,
            **configs.plot_params.model_dump()
        )
        trainer.plot_scores_vs_labels(
            which=configs.inference_params.which,
            figsize=configs.plot_params.figsize,
            alpha=configs.plot_params.alpha,
            dpi=configs.plot_params.dpi,
        )
        stage_timings["Postprocess.Plotting"] = (time.time() - plot_start) / 60
        stage_timings["Postprocess"] = (
            stage_timings.get("Postprocess.Rationales", 0.0)
            + stage_timings.get("Postprocess.Outputs", 0.0)
            + stage_timings.get("Postprocess.Plotting", 0.0)
        )
        update_recorded_timings(times_file_path, stage_timings)
        logger.debug(f"Persisted plotting timings to {times_file_path}")
        progress.finish("Plots", "plots written")

        # Evaluate Alignment

        results = None

        if run_eval:
            progress.start("Evaluation", "evaluating alignment")
            logger.info(f"Evaluating alignment...")
            eval_start = time.time()

            results = EvaluationAction.run(
                alignment=Path(alignment_file_path),
                output_dir_path=output_dir_path / task_name if task_name is not None else output_dir_path,
                error_on_fail=False,
                K=configs.k,
                source_file_path=dataset.source,
                target_file_path=dataset.target,
                train_reference_file_path=training_reference_file_path,
                full_reference_file_path=full_reference_file_path if candidates_file_path is None else None,
                reference_candidates=candidates_file_path,
                logger=logger,
            )
            stage_timings["Postprocess.Evaluation"] = (time.time() - eval_start) / 60
            stage_timings["Postprocess"] = (
                stage_timings.get("Postprocess.Rationales", 0.0)
                + stage_timings.get("Postprocess.Outputs", 0.0)
                + stage_timings.get("Postprocess.Plotting", 0.0)
                + stage_timings.get("Postprocess.Evaluation", 0.0)
            )
            update_recorded_timings(times_file_path, stage_timings)
            logger.debug(f"Persisted evaluation timings to {times_file_path}")
            progress.finish("Evaluation", "evaluation completed")

        end_time = time.time()
        elapsed_time = (end_time - start_time)/ 60
        logger.info(f"Alignment completed in {elapsed_time:.1f} minutes")

        # Save Times
        timings = load_recorded_timings(times_file_path)
        timings["Total"] = elapsed_time
        timings["Alignment"] = alignment_elapsed
        if not dataset_loaded_from_cache:
            timings["Dataset"] = dataset_elapsed
        if "Postprocess" not in timings:
            timings["Postprocess"] = (
                stage_timings.get("Postprocess.Rationales", 0.0)
                + stage_timings.get("Postprocess.Outputs", 0.0)
                + stage_timings.get("Postprocess.Plotting", 0.0)
                + stage_timings.get("Postprocess.Evaluation", 0.0)
            )

        write_recorded_timings(times_file_path, timings)
        logger.info(f"Times updated at {times_file_path}")
        progress.complete(f"total={elapsed_time:.1f} minutes")

        timmings = {
            "Alignment": alignment_elapsed,
            "Total": elapsed_time,
        }
        if not dataset_loaded_from_cache:
            timmings["Dataset"] = dataset_elapsed
        elif "Dataset" in timings:
            timmings["Dataset"] = timings["Dataset"]
        if "Postprocess" in timings:
            timmings["Postprocess"] = timings["Postprocess"]
        for step in (
            "Alignment.Inference",
            "Alignment.PostInference",
            "Alignment.Prefilter",
            "Postprocess.Rationales",
            "Postprocess.Outputs",
            "Postprocess.Plotting",
            "Postprocess.Evaluation",
            "Dataset.LoadOntologies",
            "Dataset.LoadCandidates",
            "Dataset.Process",
            "Dataset.Save",
            "Dataset.Plotting",
            "Dataset.CacheLoad",
        ):
            if step in timings:
                timmings[step] = timings[step]

        return results, timmings
