from typing import Dict

from exact.core.contracts.metric import IMetric
from exact.core.entities.evaluation import EvaluationData, MetricNames
from exact.utils.eval import MetricUtils


class PrecisionMetric(IMetric):

    metric_name = MetricNames.PRECISION

    def compute(self, data: EvaluationData) -> Dict[str, float]:
        preds_set, refs_set = MetricUtils.compute_intersection_and_union(
            data.prediction_mappings, data.reference_mappings
        )
        precision = len(preds_set & refs_set) / len(preds_set) if preds_set else 0.0
        return {self.metric_name.value: round(precision, 3)}


class RecallMetric(IMetric):

    metric_name = MetricNames.RECALL

    def compute(self, data: EvaluationData) -> Dict[str, float]:
        preds_set, refs_set = MetricUtils.compute_intersection_and_union(
            data.prediction_mappings, data.reference_mappings
        )
        recall = len(preds_set & refs_set) / len(refs_set) if refs_set else 0.0
        return {self.metric_name.value: round(recall, 3)}


class F1Metric(IMetric):

    metric_name = MetricNames.F1

    """Implementation of the F1 metric."""

    def compute(self, data: EvaluationData) -> Dict[str, float]:
        preds_set, refs_set = MetricUtils.compute_intersection_and_union(
            data.prediction_mappings, data.reference_mappings, data.null_reference_mappings
        )
        precision = len(preds_set & refs_set) / len(preds_set) if preds_set else 0.0
        recall = len(preds_set & refs_set) / len(refs_set) if refs_set else 0.0
        f1_score = (
            2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        )

        return {
            MetricNames.PRECISION.value: round(precision, 3),
            MetricNames.RECALL.value: round(recall, 3),
            self.metric_name.value: round(f1_score, 3),
        }
