
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging

from mowl.owlapi import OWLOntology

from matcha_dl.core.entities.registry import ComponentRegistry, ComponentType
from matcha_dl.core.entities.evaluation import EvaluationData, MetricNames
from matcha_dl.core.entities.mappings import EntityMapping, ReferenceMapping


class IEvaluator(ABC):
    """Abstract base class for all evaluators."""

    def __init__(self, metrics: List[MetricNames]):

        """
        Initialize the evaluator with the specified metrics.
        
        Parameters:
            metrics (List[MetricNames]): A list of metric names to be used for evaluation.
        """

        self.metrics = []

        for register in ComponentRegistry.list(ComponentType.METRIC)[1:]:

                reg_class = ComponentRegistry.get(ComponentType.METRIC, register)

                if reg_class.metric_name in metrics:
                    self.metrics.append(reg_class())

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