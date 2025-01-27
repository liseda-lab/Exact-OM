
from abc import abstractmethod
from typing import Dict, List

from matcha_dl.core.contracts import SelfRegisteringComponent
from matcha_dl.core.entities.evaluation import EvaluationData


class IMetric(SelfRegisteringComponent):

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