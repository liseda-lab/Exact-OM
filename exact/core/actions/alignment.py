import logging
import time
from pathlib import Path
from typing import Optional, Protocol, List, Union
import concurrent.futures
from multiprocessing import Manager
import pandas as pd

from exact.core.actions.evaluation import EvaluationAction
from exact.core.entities.configs.config import ConfigModel, ConfigTuner
from exact.core.entities.directories import OEAIDir
from exact.impl.matcha import Matcha
from exact.impl.seed import SeedSetter
from exact.utils.directories import OEAIDirSearcher

import shutil, yaml, os
from datetime import datetime

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

        logger = logging.getLogger("matcha-dl")
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

        # set seed

        if configs.seed is not None:
            logger.info(f"Setting seed to {configs.seed}")
            SeedSetter(configs.seed)

        # Matcha module

        logger.info(f"Matching {source_file_path} and {target_file_path}")

        # Create Dataset

        logger.info(f'Building Dataset...')
        dataset_start = time.time()

        dataset = configs.dataset(
            output_path=output_dir_path,
            logger=logger,
            cache_ok=configs.use_file_cache,
            plot_params=configs.plot_params.model_dump(),
            **configs.dataset_params.model_dump(),
        )

        dataset.load_ontologies(source_file_path, target_file_path)

        if not dataset.has_cache():
            dataset.load_candidates(candidates_file_path)

            if full_reference_file_path is not None:
                dataset.load_reference(full_reference_file_path)

        dataset.process()

        dataset_end = time.time()
        dataset_elapsed = (dataset_end - dataset_start) / 60
        logger.info(f"Dataset built in {dataset_elapsed:.1f} minutes")


        # Trainer module

        ## Train Model

        trainer = configs.trainer(
            dataset=dataset,
            model=configs.model.name,
            model_params=configs.model.params,
            device=device if device is not None else 'cpu',
            output_dir= output_dir_path / "model",
            logger=logger,
            plot_params=configs.plot_params.model_dump()
        )

        logger.info(f"Computing alignment...")

        alignment_start = time.time()

        if candidates_file_path is None:
            alignment, _ = trainer.predict(**configs.training_params.model_dump(), **configs.alignment_params.model_dump())
        else:
            alignment, _ = trainer.predict(**configs.training_params.model_dump())

        if dataset.pre_filtering:
            logger.info(f"Applying pre-filtering to alignment...")

            if candidates_file_path is None:
                pre_filtered_mappings = trainer.apply_prefilter(**configs.alignment_params.model_dump())
            
            else:
                pre_filtered_mappings = trainer.apply_prefilter()

            alignment += pre_filtered_mappings

        alignment_end = time.time()
        alignment_elapsed = (alignment_end - alignment_start) / 60
        logger.info(f"Alignment computed in {alignment_elapsed:.1f} minutes")

        # Save Alignment
        
        logger.info(f"Writing alignment...")

        alignment_file_path = trainer.save_alignment(alignment, 
                                                     candidates_one2many_path=candidates_file_path, 
                                                     sub_dir=task_name)

        logger.info(f"Alignment written to {alignment_file_path}")

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
