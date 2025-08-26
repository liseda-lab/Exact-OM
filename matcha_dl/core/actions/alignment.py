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
from matcha_dl.impl.matcha import Matcha
from matcha_dl.impl.seed import SeedSetter
from matcha_dl.utils.directories import OEAIDirSearcher

import shutil, yaml, os
from datetime import datetime

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
        
        # TODO ensure support in all contracts for torch device format
        # Process device into torch device format
        # if device is not None:
        #     if isinstance(device, int):
        #         device = f"cuda:{device}" if device >= 0 else "cpu"
        #     elif isinstance(device, str):
        #         device = device.lower()
        #     else:
        #         raise ValueError(f"Invalid device type: {type(device)}. Expected int or str.")

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

        logger.info(f"Running Matcha...")
        matcha_start = time.time()

        matcha = Matcha(
            output_path=output_dir_path,
            logger=logger,
            **configs.matcha_params.model_dump(),
            cache_ok=configs.use_file_cache,
        )

        logger.debug(f"Matcha error logs are being written to {matcha.log_file}")

        if not matcha.has_cache:

            matcha.load_ontologies(source_file_path, target_file_path)

            if reference_file_path is not None:
                matcha.load_reference(reference_file_path)

            if candidates_file_path is not None:
                matcha.load_candidates(candidates_file_path)

        matcha.match()

        if matcha.generated_reference.exists() and reference_file_path is None:
            logger.info(f"Running on self-supervised setting, using generated reference file")
            logger.info(f"Generated reference file: {matcha.generated_reference}")

            reference_file_path = matcha.generated_reference


        matcha_end = time.time()
        matcha_elapsed = (matcha_end - matcha_start) / 60
        logger.info(f"Matcha completed in {matcha_elapsed:.1f} minutes")

        # Create Dataset

        logger.info(f'Building Dataset...')
        dataset_start = time.time()

        dataset = configs.dataset(
            output_path=output_dir_path,
            matchers=matcha.matchers,
            logger=logger,
            cache_ok=configs.use_file_cache,
            plot_params=configs.plot_params.model_dump(),
            **configs.dataset_params.model_dump(),
        )

        dataset.load_ontologies(source_file_path, target_file_path)

        if not dataset.has_cache():
            dataset.load_candidates(matcha.candidates)

            if full_reference_file_path is not None:
                dataset.load_full_reference(full_reference_file_path)

            if configs.matcha_params.calculate_scores:
                dataset.load_data(matcha.matcha_features)

            if matcha.reference is not None:
                dataset.load_reference(matcha.reference)
                dataset.load_negatives(matcha.negatives)

        dataset.process()

        if not dataset.has_cache():
            if configs.plot_params.enabled:
                dataset.plot_matcha_features() 
            dataset.save()

        dataset_end = time.time()
        dataset_elapsed = (dataset_end - dataset_start) / 60
        logger.info(f"Dataset built in {dataset_elapsed:.1f} minutes")


        # Trainer module

        ## Train Model

        trainer = configs.trainer(
            dataset=dataset,
            model=configs.model.name,
            loss=configs.loss.name,
            optimizer=configs.optimizer.name,
            loss_params=configs.loss.params,
            optimizer_params=configs.optimizer.params,
            model_params=configs.model.params,
            stopping=configs.stopper.name,
            stopping_params=configs.stopper.params,
            device=device if device is not None else 'cpu',
            output_dir= output_dir_path / "model",
            use_last_checkpoint=configs.use_last_checkpoint,
            skip_training_if_checkpoint=configs.skip_training_if_checkpoint,
            logger=logger,
            plot_params=configs.plot_params.model_dump()
        )

        train_elapsed = None
        if reference_file_path is not None:
            logger.info(f"Training model with {reference_file_path}")
            train_start = time.time()

            trainer.train(**configs.training_params.model_dump())

            train_end = time.time()
            train_elapsed = (train_end - train_start) / 60
            logger.info(f"Model trained in {train_elapsed:.1f} minutes")

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
                train_reference_file_path=reference_file_path,
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
            f.write(f"Matcha: {matcha_elapsed:.1f} minutes\n")
            f.write(f"Dataset: {dataset_elapsed:.1f} minutes\n")
            if train_elapsed is not None:
                f.write(f"Training: {train_elapsed:.1f} minutes\n")
            f.write(f"Alignment: {alignment_elapsed:.1f} minutes\n")
        logger.info(f"Times written to {times_file_path}")

        timmings = {
            "Matcha": matcha_elapsed,
            "Dataset": dataset_elapsed,
            "Training": train_elapsed if train_elapsed is not None else 0,
            "Alignment": alignment_elapsed,
            "Total": elapsed_time,
        }

        return results, timmings

# TODO rework complex actions
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


class TuningAlignmentAction:
    @staticmethod
    def run(
        source_file_path: Path,
        target_file_path: Path,
        reference_file_path: Path|None,
        candidates_file_path: Path|None,
        full_reference_file_path: Path,
        configs_file_path: Path,
        output_dir_path: Path,
        save_logs: bool = False,
        extra_dirs: list[Path] = [],
        tag: str|None = None,
    ) -> None:
        logger = logging.getLogger("matcha-dl-tuner")
        logger.setLevel(logging.INFO)

        # 1) load this one-combo YAML → model
        cfg_dict  = yaml.safe_load(configs_file_path.read_text())
        cfg_model = ConfigModel(**cfg_dict)

        # 2) make run-dir
        run_tag = tag or cfg_dict.get("tag") or datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = output_dir_path / run_tag
        run_dir.mkdir(parents=True, exist_ok=True)

        # 3) dump exact config
        (run_dir/"config.yaml").write_text(yaml.safe_dump(cfg_dict))

        # 4) copy extras in
        for d in extra_dirs:
            tgt = run_dir / d.name
            if d.is_dir(): shutil.copytree(d, tgt)
            else:           shutil.copy2(d, tgt)

        # 5) optional file logging
        if save_logs:
            fh = logging.FileHandler(run_dir/f"tuner_{run_tag}.log")
            fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            logger.addHandler(fh)

        # 6) run alignment & catch errors
        try:
            start  = time.time()
            out    = AlignmentAction.run(
                        source_file_path        = source_file_path,
                        target_file_path        = target_file_path,
                        reference_file_path     = reference_file_path,
                        candidates_file_path    = candidates_file_path,
                        full_reference_file_path= full_reference_file_path,
                        output_dir_path         = run_dir,
                        configs_file_path       = configs_file_path,
                        save_logs               = save_logs
                     )
            duration = time.time() - start
            results  = out.get("results", {})
            timings  = out.get("timings", {})
            status   = "success"
        except Exception as e:
            logger.error(f"Run {run_tag} failed: {e}", exc_info=True)
            results, timings, status = {}, {}, "error"

        # 7) flatten & append to CSV safely
        row        = {"tag": run_tag, "status": status, **flatten_config(cfg_model), **results, **timings}
        df         = pd.DataFrame([row])
        csv_path   = output_dir_path/"tuning_alignment_results.csv"
        write_hdr  = not csv_path.exists() or os.path.getsize(csv_path)==0
        with open(csv_path, "a", newline="") as f:
            df.to_csv(f, index=False, header=write_hdr)
        logger.info(f"Appended results for {run_tag}")
