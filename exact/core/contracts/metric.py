"""Metric component contract used by built-in evaluators."""

from abc import abstractmethod
from typing import Dict, List

from exact.core.contracts import SelfRegisteringComponent
from exact.core.entities.evaluation import EvaluationData, MetricNames
from exact.core.entities.registry import ComponentType


class IMetric(SelfRegisteringComponent):
    """Base contract for one registered evaluation metric."""

    component_type = ComponentType.METRIC
    metric_name: MetricNames

    def prepare(self, data: EvaluationData) -> List[EvaluationData]:
        """
        Optionally preprocess the data. Default behavior is to return the input data as-is.
        """
        return [data]

    @abstractmethod
    def compute(self, data: EvaluationData) -> Dict[str, float]:
        """
        Compute the metric using the prepared data.

        Parameters:
            data (EvaluationData): Structured input data for evaluation.

        Returns:
            Dict[str, float]: Metric results as a dictionary.
        """
        pass
