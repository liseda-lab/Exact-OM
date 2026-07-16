import logging
import time
from pathlib import Path
from typing import List, Optional, Protocol, Tuple, Union

from exact.core.contracts.dataset import OWLDataset
from exact.core.entities.mappings import EntityMapping, ReferenceMapping
from exact.impl.evaluator import Evaluator
from exact.utils.logs import configure_exact_logger


class EvaluationAction(Protocol):
    @staticmethod
    def run(
        alignment: Union[
            List[Tuple[ReferenceMapping, List[EntityMapping]]], List[EntityMapping], Path
        ],
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
            resolved_log_level = (
                getattr(logging, log_level) if log_level is not None else logging.INFO
            )
            logger = configure_exact_logger(
                logging.getLogger("OAEI-BIO-ML:eval"),
                resolved_log_level,
                log_file_path=log_file_path,
            )

            logger.debug(f"Logging level set to {resolved_log_level}")

        # Start evaluation

        logger.info("Starting evaluation")

        if full_reference_file_path is not None:

            logger.info(
                f"Using test reference file {full_reference_file_path} for global evaluation"
            )

            if isinstance(source_file_path, OWLDataset):
                logger.info("Using source ontology from memory")
                source_ontology = source_file_path.ontology
            elif isinstance(source_file_path, Path):
                logger.info(f"Using source ontology from file {source_file_path}")
                source_ontology = OWLDataset(str(source_file_path)).ontology
            else:
                target_ontology = None
                logger.warning("not using source ontology for global evaluation")

            if isinstance(target_file_path, OWLDataset):
                logger.info("Using target ontology from memory")
                target_ontology = target_file_path.ontology
            elif isinstance(target_file_path, Path):
                logger.info(f"Using target ontology from file {target_file_path}")
                target_ontology = OWLDataset(str(target_file_path)).ontology
            else:
                target_ontology = None
                logger.warning("not using target ontology for global evaluation")

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

            logger.info("Using aligment for local evaluation")

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
            logger.info(
                f"Saving evaluation results to {output_dir_path / 'evaluation_results.csv'}"
            )
            Evaluator.save_results(results, output_dir_path)

        else:
            logger.error("Error during evaluation, skipping evaluation..")

        logger.info(f"Finished evaluation in {(time.time() - start_time):.3f} seconds")

        return results
