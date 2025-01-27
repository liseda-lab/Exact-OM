
from typing import Dict, List, Type

from matcha_dl.core.contracts.metric import IMetric

from abc import ABC, abstractmethod
from typing import Dict, Union, List, Optional
from matcha_dl.core.entities.mappings import EntityMapping, ReferenceMapping


class IEvaluator(ABC):
    """Abstract base class for all evaluators."""

    @abstractmethod
    def __init__(self, metrics: List[Type[IMetric]]):
        """
        Initialize the evaluator with a list of metrics.

        Parameters:
            metrics (List[Type[IMetric]]): List of metric names to evaluate.
        """
        pass

    @abstractmethod
    def evaluate(
        self,
        data: Dict[str, Union[List[EntityMapping], List[ReferenceMapping], List[int]]],
        source_ontology: Optional[OWLOntology] = None,
        target_ontology: Optional[OWLOntology] = None
    ) -> Dict[str, float]:
        """
        Evaluate the provided data using the configured metrics.

        Parameters:
            data (Dict): A dictionary containing:
                - prediction_mappings (List[EntityMapping]): Predicted entity mappings.
                - reference_mappings (List[ReferenceMapping]): Reference entity mappings.
                - null_reference_mappings (Optional[List[ReferenceMapping]]): Null reference mappings.
                - reference_and_candidates (Optional[List[Tuple[ReferenceMapping, List[EntityMapping]]]]): Reference and candidate mappings.
                - K (Optional[List[int]]): List of K values for Hits@K evaluation.

            source_ontology (Optional[OWLOntology]): Source ontology to filter ignored classes.
            target_ontology (Optional[OWLOntology]): Target ontology to filter ignored classes.

        Returns:
            Dict[str, float]: A dictionary of metric results.
        """
        pass