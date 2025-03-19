import logging
import time
from pathlib import Path
from typing import Optional, Protocol, List, Union
import concurrent.futures
from multiprocessing import Manager
import pandas as pd

from matcha_dl.core.actions.evaluation import EvaluationAction
from matcha_dl.core.entities.configs.config import ConfigModel, ConfigTuner
from matcha_dl.core.entities.directories import OEAIDir
from matcha_dl.core.values import N_CLASSES
from matcha_dl.impl.matcha import Matcha
from matcha_dl.impl.seed import SeedSetter
from matcha_dl.utils.directories import OEAIDirSearcher

class AlignmentAction(Protocol):
    @staticmethod
    def run(
        source_file_path: Path,
        target_file_path: Path,
        output_dir_path: Path,
        configs_file_path: Optional[Union[Path, ConfigModel]] = None,
        reference_file_path: Optional[Path] = None,
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

        matcha = Matcha(
            output_path=output_dir_path,
            logger=logger,
            **configs.matcha_params.model_dump(),
            cache_ok=configs.use_file_cache,
        )

        logger.info(f"Computing matcha scores...")
        logger.debug(f"Matcha error logs are being written to {matcha.log_file}")

        matcha.load_ontologies(source_file_path, target_file_path)

        if reference_file_path is not None:
            matcha.load_reference(reference_file_path)

        if candidates_file_path is not None:
            matcha.load_candidates(candidates_file_path)

        matcha.match()

        # Create Dataset

        logger.info(f'Building Dataset...')

        dataset = configs.dataset(
            output_path=output_dir_path,
            matchers=matcha.matchers,
            logger=logger,
            cache_ok=configs.use_file_cache,
            **configs.dataset_params.model_dump(),
        )

        if matcha.reference is not None:
            dataset.load_reference(matcha.reference)
            dataset.load_negatives(matcha.negatives)

        dataset.load_ontologies(source_file_path, target_file_path)
        
        dataset.load_candidates(matcha.candidates)
        dataset.load_data(matcha.matcha_features)

        dataset.process()

        logger.info(f"Dataset ready")

        dataset.save()

        if configs.plot_negatives_params.enabled:
            dataset.plot_negative_distributions(**configs.plot_negatives_params.model_dump())

        # Trainer module

        ## Parse model params

        if dataset.reference is not None:

            model_params = configs.model.params
            model_params["n"] = dataset.x().shape[1]
            model_params["n_classes"] = N_CLASSES

        else:
            model_params = configs.model.params

        ## Train Model

        trainer = configs.trainer(
            dataset=dataset,
            model=configs.model.name,
            loss=configs.loss.name,
            optimizer=configs.optimizer.name,
            loss_params=configs.loss.params,
            optimizer_params=configs.optimizer.params,
            model_params=model_params,
            stopping=configs.stopper.name,
            stopping_params=configs.stopper.params,
            device=device if device is not None else 'cpu',
            output_dir= output_dir_path / "model",
            use_last_checkpoint=configs.use_last_checkpoint,
            skip_training_if_checkpoint=configs.skip_training_if_checkpoint,
            logger=logger,
        )

        if dataset.reference is not None:
            logger.info(f"Training model with {reference_file_path}")
            trainer.train(**configs.training_params.model_dump())

        logger.info(f"Computing alignment...")

        if candidates_file_path is not None:
            alignment, _ = trainer.predict(**configs.alignment_params.model_dump())
        else:
            alignment, _ = trainer.predict(
                threshold=configs.alignment_params.threshold)

        # Save Alignment
        
        logger.info(f"Writing alignment...")

        alignment_file_path = trainer.save_alignment(alignment=alignment, 
                                                     candidates_one2many_path=candidates_file_path, 
                                                     sub_dir=task_name)

        logger.info(f"Alignment written to {alignment_file_path}")

        # Evaluate Alignment

        results = None

        if run_eval:
            logger.info(f"Evaluating alignment...")

            results = EvaluationAction.run(
                alignment=Path(alignment_file_path),
                output_dir_path=output_dir_path / task_name,
                error_on_fail=False,
                K=configs.k,
                source_file_path=dataset.source,
                target_file_path=dataset.target,
                train_reference_file_path=reference_file_path,
                full_reference_file_path=full_reference_file_path if candidates_file_path is None else None,
                reference_candidates=candidates_file_path,
                logger=logger,
            )

        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Alignment completed in {elapsed_time:.3f} seconds")

        return results

# TODO global and local candidates are different
class DirectoryAlignmentAction(Protocol):
    @staticmethod
    def run(
        data_dir: Path,
        output_dir: Path,
        configs_file_path: Optional[Path] = None,
        run_eval: bool = False,
        save_logs: bool = False,
        devices: Optional[List[int]] = None,
    ) -> None:
        
        start_time = time.time()

        # Load Configs

        if configs_file_path is not None:
            configs = ConfigModel.load_config(configs_file_path)

        else:
            configs = ConfigModel()

        # Loading logging configuration from configs

        logger = logging.getLogger("matcha-dl-dir")
        logger.setLevel(configs.logging_level)

        if save_logs:
            temp_log = output_dir / "matcha_dl-dir.log"
            file_handler = logging.FileHandler(temp_log)
            file_handler.setLevel(configs.logging_level)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.info(f"Logging to file {temp_log}")

        logger.debug(f"Logging level set to {configs.logging_level}")

        searcher = OEAIDirSearcher(data_dir)

        try:

            logger.info(f"Searching OEAI directories in {data_dir}")
            oeai_dirs = searcher.find_oeai_dirs()

        except Exception as e:
            logging.error(f"Error searching OEAI directories: {e}")
            return

        def process_oeai_dir(oeai_dir: OEAIDir, available_devices: List[int]) -> dict:
                    temp_output_dir = output_dir / oeai_dir.data_name
                    temp_output_dir.mkdir(parents=True, exist_ok=True)

                    tasks = [
                        ("global_supervised", None, oeai_dir.reference_file_path),
                        ("local_supervised", oeai_dir.candidates_file_path, oeai_dir.reference_file_path),
                        ("global_unsupervised", None, None),
                        ("local_unsupervised", oeai_dir.candidates_file_path, None),
                    ]

                    with available_devices.get_lock():
                        if available_devices:
                            device = available_devices.pop(0)
                        else:
                            device = None

                    task_results = {}

                    try:
                        for task_name, candidates_file_path, reference_file_path in tasks:
                            results = AlignmentAction.run(
                                source_file_path=oeai_dir.source_file_path,
                                target_file_path=oeai_dir.target_file_path,
                                output_dir_path=temp_output_dir,
                                configs_file_path=configs_file_path,
                                reference_file_path=reference_file_path,
                                full_reference_file_path=oeai_dir.full_reference_file_path,
                                candidates_file_path=candidates_file_path,
                                log_file_path=temp_output_dir / f"{task_name}.log" if save_logs else None,
                                run_eval=run_eval,
                                task_name=task_name,
                                device=device,
                            )
                            task_results[task_name] = results
                    finally:
                        with available_devices.get_lock():
                            if device is not None:
                                available_devices.append(device)

                    return {oeai_dir.data_name: task_results}

                    

        logger.info(f"Running Alignments on found OEAI directories...")

        all_results = {}

        with Manager() as manager:
            available_devices = manager.list(devices) if devices else manager.list()
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = [
                    executor.submit(process_oeai_dir, oeai_dir, available_devices)
                    for oeai_dir in oeai_dirs
                ]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        task_results = future.result()
                        all_results.update(task_results)
                    except Exception as e:
                        logging.error(f"Error processing OEAI directory: {e}")

        logger.info(f"All OEAI directories processed successfully")
        logger.info(f"Writing results...")

        results_list = []
        for oeai_dir_name, tasks in all_results.items():
            for task_name, result in tasks.items():
                flattened_result = {"OEAI Directory": oeai_dir_name, "Task Name": task_name}
                if isinstance(result, dict):
                    flattened_result.update(result)
                else:
                    flattened_result["Result"] = result
                results_list.append(flattened_result)

        df = pd.DataFrame(results_list)
        df.to_csv(output_dir / "alignment_full_results.csv", index=False)

        logger.info(f"Results written to {output_dir / 'alignment_full_results.csv'}")

        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Directory alignment completed in {elapsed_time:.3f} seconds")

class TuningAlignmentAction(Protocol):
    @staticmethod
    def run(
        source_file_path: Path,
        target_file_path: Path,
        reference_file_path: Path,
        candidates_file_path: Path,
        output_dir_path: Path,
        configs_file_path: Path,
        full_reference_file_path: Path,
        save_logs: bool = False,
        devices: Optional[List[int]] = None,
        max_workers: Optional[int] = None,
        max_combinations: Optional[int] = None,
    ) -> None:
        

        start_time = time.time()

        logger = logging.getLogger("matcha-dl-tuner")
        logger.setLevel(logging.DEBUG)

        if save_logs:
            temp_log = output_dir_path / "matcha_dl-tuner.log"
            file_handler = logging.FileHandler(temp_log)
            file_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.info(f"Logging to file {temp_log}")

        logger.debug(f"Logging level set to {logging.DEBUG}")

        logger.info(f"Loading Possible Configs...")

        config_tuner = ConfigTuner(configs_file_path)

        tune_configs = config_tuner.load_tuned_config(max_combinations)

        logger.info(f"Loaded {len(tune_configs)} {'random' if max_combinations is None else 'possible'} configurations")

        logger.info(f"Running Alignments on possible combinations...")

        from matcha_dl.utils.action import process_config

        all_tuning_results = []
        error_counts = 0

        if max_workers is not None and devices is not None:
                max_workers = min(max_workers, len(devices))

        with Manager() as manager:
            available_devices = manager.list(devices) if devices else None
            condition = manager.Condition()
             
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(process_config, 
                                    tag=str(i),
                                    config=config, 
                                    available_devices=available_devices,
                                    condition=condition,
                                    source_file_path=source_file_path, 
                                    target_file_path=target_file_path, 
                                    reference_file_path=reference_file_path, 
                                    candidates_file_path=candidates_file_path, 
                                    output_dir_path=output_dir_path, 
                                    full_reference_file_path=full_reference_file_path, 
                                    save_logs=save_logs)

                    for i, config in enumerate(tune_configs)
                ]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        tuning_result = future.result()
                        all_tuning_results.append(tuning_result)
                    except Exception as e:
                        logger.error(f"Error processing combination: {e}", exc_info=True)
                        error_counts += 1
                        continue

        if error_counts == len(tune_configs):
            logger.error(f"All combinations failed")
            return
        elif error_counts > 0:
            logger.warning(f"{error_counts} combinations failed")
        else:
            logger.info(f"All combinations run successfully")
            
        logger.info(f"Writing results...")

        tuning_results_list = []
        for tuning_result in all_tuning_results:
            config_tag = tuning_result["config_tag"]
            config = tuning_result["config"]
            results = tuning_result["results"]
            for task_name, result in results.items():
                flattened_result = {"Config Tag": config_tag, "Task Name": task_name}
                if isinstance(result, dict):
                    flattened_result.update(result)
                else:
                    flattened_result["Result"] = result
                flattened_result.update(config)
                tuning_results_list.append(flattened_result)

        tuning_df = pd.DataFrame(tuning_results_list)

        tuning_df.to_csv(output_dir_path / "tuning_alignment_full_results.csv", index=False)

        logger.info(f"Results written to {output_dir_path / 'tuning_alignment_full_results.csv'}")

        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Tuning alignment completed in {elapsed_time:.3f} seconds")
        
