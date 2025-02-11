
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging

from mowl.owlapi import OWLOntology

from matcha_dl.core.contracts import LoggingClass
from matcha_dl.core.entities.configs import ComponentRegistry
from matcha_dl.core.entities.evaluation import EvaluationData, MetricNames
from matcha_dl.core.entities.mappings import EntityMapping, ReferenceMapping


class IEvaluator(ABC):
    """Abstract base class for all evaluators."""

    @abstractmethod
    def __init__(self, metrics: List[MetricNames],
                 logger: Optional[logging.Logger] = None
    ):

        """
        Initialize the evaluator with the specified metrics.
        
        Parameters:
            metrics (List[MetricNames]): A list of metric names to be used for evaluation.
        """

        self.metrics = [
            ComponentRegistry.get_metric("metric", metric.value)() for metric in metrics
        ]

    @abstractmethod
    def evaluate(
        self,
        data: EvaluationData,
    ) -> Dict[str, float]:
        
        """
        Evaluate the provided data using the configured metrics.

        Parameters:
            data (EvaluationData): The data to be evaluated.

        Returns:
            Dict[str, float]: A dictionary of metric results.
        """
        pass

    @classmethod
    @abstractmethod
    def global_eval(
        cls, 
        predictions: Union[List[EntityMapping], Path],
        test_reference: Union[List[ReferenceMapping], Path],
        source_ontology: Optional[OWLOntology] = None,
        target_ontology: Optional[OWLOntology] = None,
        train_reference: Optional[List[ReferenceMapping]] = None,
        threshold: Optional[float] = None,
    ) -> Dict[str, float]:
        
        """
        Evaluate the provided data using the configured metrics.

        Parameters:
            predictions (Union[List[EntityMapping], Path]): The data to be evaluated.
            test_reference (Union[List[ReferenceMapping], Path]): The data to be evaluated.
            source_ontology (Optional[OWLOntology]): The data to be evaluated.
            target_ontology (Optional[OWLOntology]): The data to be evaluated.
            train_reference (Optional[List[ReferenceMapping]]): The data to be evaluated.
            threshold (Optional[float]): The data to be evaluated.

        Returns:
            Dict[str, float]: A dictionary of metric results.
        """
        pass

    @classmethod
    @abstractmethod
    def local_eval(
        cls,
        reference_and_candidates: Union[List[Tuple[ReferenceMapping, List[EntityMapping]]], Path],
        K: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        
        """
        Evaluate the provided data using the configured metrics.

        Parameters:
            reference_and_candidates (Union[List[Tuple[ReferenceMapping, List[EntityMapping]]], Path]): The data to be evaluated.
            K (Optional[List[int]]): The data to be evaluated.

        Returns:
            Dict[str, float]: A dictionary of metric results.
        """
        pass