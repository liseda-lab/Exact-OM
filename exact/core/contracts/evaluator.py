from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from exact.core.contracts.base import SelfRegisteringComponent
from exact.core.entities.evaluation import (
    BackendEvaluation,
    EvaluationData,
    EvaluationRequest,
    MetricNames,
)
from exact.core.entities.mappings import EntityMapping, ReferenceMapping
from exact.core.entities.registry import ComponentRegistry, ComponentType


class IEvaluator(SelfRegisteringComponent, ABC):
    """Abstract base class for all evaluators."""

    component_type = ComponentType.EVALUATOR

    @classmethod
    @abstractmethod
    def run(cls, request: EvaluationRequest) -> BackendEvaluation:
        """Evaluate a backend-neutral request without writing output files."""
        raise NotImplementedError

    def __init__(self, metrics: List[MetricNames]):
        """
        Initialize the evaluator with the specified metrics.

        Parameters:
            metrics (List[MetricNames]): A list of metric names to be used for evaluation.
        """

        registered = {}
        for name in ComponentRegistry.list(ComponentType.METRIC):
            metric_class = ComponentRegistry.get(ComponentType.METRIC, name)
            metric_name = metric_class.metric_name
            if metric_name in registered:
                raise ValueError(f"Multiple metrics are registered for {metric_name.value!r}.")
            registered[metric_name] = metric_class

        missing = [metric for metric in metrics if metric not in registered]
        if missing:
            available = ", ".join(metric.value for metric in registered) or "none"
            requested = ", ".join(metric.value for metric in missing)
            raise ValueError(
                f"Metrics are not registered: {requested}. Available metrics: {available}."
            )

        self.metrics = [registered[metric]() for metric in metrics]

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
        source_ontology: Any = None,
        target_ontology: Any = None,
        train_reference: Optional[Union[List[ReferenceMapping], Path]] = None,
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
        reference_candidates: Optional[Path] = None,
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
