import logging
import time
from pathlib import Path
from typing import Optional, Protocol, List, Union
import pandas as pd

from exact.core.actions.evaluation import EvaluationAction
from exact.core.entities.configs.config import ConfigModel
from exact.impl.seed import SeedSetter
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

        logger = logging.getLogger("exact")
        logger.setLevel(configs.logging_level)

        if log_file_path is not None:
            file_handler = logging.FileHandler(log_file_path)
            file_handler.setLevel(configs.logging_level)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.info(f"Logging to file {log_file_path}")

        logger.debug(f"Logging level set to {configs.logging_level}")

        # log configs state

        if configs_file_path is not None:
            logger.info(f"Using configuration from {configs_file_path}")
        else:
            logger.info(f"Using default configuration")

        # Resolve dependencies

        configs.resolve_dependencies()
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

        # Create Dataset

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


        # Trainer module

        ## Train Model
        logger.info(f"Building Trainer and Model...")

        model_sequence = configs.get_model_sequence()
        if not model_sequence:
            raise ValueError("No models configured for alignment.")
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
            model_specs.append((extra.name, extra.params))

        trainer = configs.trainer(
            dataset=dataset,
            models=model_specs,
            device=device,
            output_dir= output_dir_path / "model",
            logger=logger,
        )

        logger.info(f"Computing alignment...")

        alignment_core_minutes = 0.0

        inference_kwargs = configs.inference_params.model_dump()
        inference_kwargs["local_alignment"] = candidates_file_path is not None

        predict_start = time.time()
        if candidates_file_path is None:
            inference_kwargs.update(configs.alignment_params.model_dump())
            alignment, avg_t = trainer.predict(**inference_kwargs)
        else:
            alignment, avg_t = trainer.predict(**inference_kwargs)
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
            logger.info(f"Applying Exact Matches to alignment...")
            prefilter_start = time.time()

            if candidates_file_path is None:
                alignment = trainer.apply_prefilter(alignment, **configs.alignment_params.model_dump())

            else:
                alignment = trainer.apply_prefilter(alignment)
            prefilter_minutes = (time.time() - prefilter_start) / 60
            stage_timings["Alignment.Prefilter"] = prefilter_minutes
            alignment_core_minutes += prefilter_minutes

        alignment_elapsed = alignment_core_minutes
        stage_timings["Alignment"] = alignment_elapsed
        update_recorded_timings(times_file_path, stage_timings)
        logger.debug(f"Persisted alignment timings to {times_file_path}")
        logger.info(f"Alignment computed in {alignment_elapsed:.1f} minutes")

        # Save Alignment
        
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

        # Plot Distributions
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

        # Evaluate Alignment

        results = None

        if run_eval:
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
