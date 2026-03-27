import logging
import time
from pathlib import Path
from typing import Optional, Protocol, List, Union
import concurrent.futures
from multiprocessing import Manager
import pandas as pd

from exact.core.actions.evaluation import EvaluationAction
from exact.core.entities.configs.config import ConfigModel
from exact.impl.seed import SeedSetter

import shutil, yaml, os
from datetime import datetime
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

        dataset = configs.dataset(
            output_path=output_dir_path,
            logger=logger,
            cache_ok=configs.use_file_cache,
            device=device,
            **configs.dataset_params.model_dump(),
        )

        dataset.load_ontologies(source_file_path, target_file_path)

        if not dataset.has_cache():

            if full_reference_file_path is not None:
                dataset.load_reference(full_reference_file_path)
            
            dataset.load_candidates(candidates_file_path, device=device, **configs.candidates_params.model_dump())

        dataset.process()

        if not dataset.has_cache():

            dataset.save()

            dataset.log_sanity_examples(**configs.sanity_check_params.model_dump())

            dataset.plot_feature_distributions(
                which=configs.dataset_params.which,
                **configs.plot_params.model_dump()
            )

        dataset_end = time.time()
        dataset_elapsed = (dataset_end - dataset_start) / 60
        logger.info(f"Dataset built in {dataset_elapsed:.1f} minutes")


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

        alignment_start = time.time()

        inference_kwargs = configs.inference_params.model_dump()

        if candidates_file_path is None:
            inference_kwargs.update(configs.alignment_params.model_dump())
            alignment, avg_t = trainer.predict(**inference_kwargs)
        else:
            alignment, avg_t = trainer.predict(**inference_kwargs)

        logger.info(f"Average inference time per example: {avg_t:.4f} seconds")

        if dataset.filter_exact_matches:
            logger.info(f"Applying Exact Matches to alignment...")

            if candidates_file_path is None:
                alignment = trainer.apply_prefilter(alignment, **configs.alignment_params.model_dump())

            else:
                alignment = trainer.apply_prefilter(alignment)

        alignment_end = time.time()
        alignment_elapsed = (alignment_end - alignment_start) / 60
        logger.info(f"Alignment computed in {alignment_elapsed:.1f} minutes")

        # Save Alignment
        
        logger.info(f"Writing alignment...")

        alignment_file_path = trainer.save_results(alignment, 
                                                     candidates_one2many_path=candidates_file_path, 
                                                     sub_dir=task_name,
                                                     **configs.alignment_params.model_dump()
                                                     )["alignment_tsv"]

        logger.info(f"Alignment written to {alignment_file_path}")

        # Plot Distributions
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

        # Evaluate Alignment

        results = None

        if run_eval:
            logger.info(f"Evaluating alignment...")

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

        end_time = time.time()
        elapsed_time = (end_time - start_time)/ 60
        logger.info(f"Alignment completed in {elapsed_time:.1f} minutes")

        # Save Times
        times_file_path = output_dir_path / "times.txt"
        with open(times_file_path, "a") as f:
            f.write(f"Total: {elapsed_time:.1f} minutes\n")
            f.write(f"Dataset: {dataset_elapsed:.1f} minutes\n")
            f.write(f"Alignment: {alignment_elapsed:.1f} minutes\n")
        logger.info(f"Times written to {times_file_path}")

        timmings = {
            "Dataset": dataset_elapsed,
            "Alignment": alignment_elapsed,
            "Total": elapsed_time,
        }

        return results, timmings
