
# TODO create eval action from files
    # Single Eval
    # Directory Eval
# TODO refactor delivery to allow for multiple cli and apis
# TODO add eval to list of poetry toml commands

import logging
import time
from pathlib import Path
from typing import Optional, Protocol, Union, List, Tuple

from matcha_dl.core.contracts.dataset import OWLDataset
from matcha_dl.impl.evaluator import Evaluator
from matcha_dl.core.entities.mappings import EntityMapping, ReferenceMapping


class EvaluationAction(Protocol):
    @staticmethod
    def run(
        alignment: Union[List[Tuple[ReferenceMapping, List[EntityMapping]]], List[EntityMapping], Path],
        output_dir_path: Path,
        error_on_fail: bool = False,
        K: Optional[List[int]] = None,
        source_file_path: Optional[Union[OWLDataset, Path]] = None,
        target_file_path: Optional[Union[OWLDataset, Path]] = None,
        train_reference_file_path: Optional[Path] = None,
        full_reference_file_path: Optional[Path] = None,
        reference_candidates: Optional[Path] = None,
        log_file_path: Optional[Path] = None,
        log_level: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> Optional[dict]:
        
        start_time = time.time()

        # Load logging configuration

        if logger is None:
            logger = logging.getLogger("matcha-dl-evaluation")

            if log_level is not None:
                logger.setLevel(log_level)

            if log_file_path is not None:
                file_handler = logging.FileHandler(log_file_path)
                file_handler.setLevel(log_level)
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
                logger.info(f"Logging to file {log_file_path}")

            logger.debug(f"Logging level set to {log_level}")

        # Start evaluation

        logger.info(f"Starting evaluation")

        if full_reference_file_path is not None:

            logger.info(f"Using test reference file {full_reference_file_path} for global evaluation")

            if isinstance(source_file_path, OWLDataset):
                logger.info(f"Using source ontology from memory")
                source_ontology = source_file_path.ontology
            elif isinstance(source_file_path, Path):
                logger.info(f"Using source ontology from file {source_file_path}")
                source_ontology = OWLDataset(source_file_path).ontology
            else:
                target_ontology = None
                logger.warning(f"not using source ontology for global evaluation")
            
            if isinstance(target_file_path, OWLDataset):
                logger.info(f"Using target ontology from memory")
                target_ontology = target_file_path.ontology
            elif isinstance(target_file_path, Path):
                logger.info(f"Using target ontology from file {target_file_path}")
                target_ontology = OWLDataset(target_file_path).ontology
            else:
                target_ontology = None
                logger.warning(f"not using target ontology for global evaluation")

            try:

                results = Evaluator.global_eval(
                    predictions=alignment,
                    full_reference=full_reference_file_path,
                    train_reference=train_reference_file_path,
                    source_ontology=source_ontology,
                    target_ontology=target_ontology,
                )

            except ValueError as e:
                logger.error(f"Error during evaluation: {e}", exc_info=True)
                results = None

                if error_on_fail:
                    raise e

        else:

            logger.info(f"Using aligment for local evaluation")

            try:

                results = Evaluator.local_eval(
                    reference_and_candidates=alignment,
                    reference_candidates=reference_candidates,
                    K=K,
                )
            
            except ValueError as e:
                logger.error(f"Error during evaluation: {e}", exc_info=True)
                results = None

                if error_on_fail:
                    raise e

        if results is not None:
            logger.info(f"Evaluation results: {results}")
            logger.info(f"Saving evaluation results to {output_dir_path / 'evaluation_results.csv'}")
            Evaluator.save_results(results, output_dir_path)

        else:
            logger.error(f"Error during evaluation, skipping evaluation..")

        logger.info(f"Finished evaluation in {time.time() - start_time} seconds")

        return results


class DirectoryEvaluationAction(Protocol):
    @staticmethod
    def run(
        directory: Path,
        max_heap_size: str = "4g",
        log_level: str = "INFO",
        K: Optional[List[int]] = None,
        error_on_fail: bool = False,
        log_file_path: Optional[Path] = None,
    ):
        """
        Given a directory it will look for a directory containg the following subdirectories:

        """

        pass