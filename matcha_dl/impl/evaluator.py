
from typing import Dict, List, Optional, Union

from matcha_dl.core.contracts.evaluator import IEvaluator
from matcha_dl.core.entities.mappings import EntityMapping, ReferenceMapping
from matcha_dl.core.entities.evaluation import EvaluationData
from matcha_dl.impl.metrics import MetricRegistry
from matcha_dl.impl.utils import MetricUtils


class Evaluator(IEvaluator):
    def __init__(self, metrics: List[str]):
        self.metrics = [
            MetricRegistry.get_metric(metric)() for metric in metrics
        ]

    def evaluate(
        self,
        data: EvaluationData,
        source_ontology: Optional["OWLOntology"] = None,
        target_ontology: Optional["OWLOntology"] = None
    ) -> Dict[str, float]:
        if (source_ontology and not target_ontology) or (target_ontology and not source_ontology):
            raise ValueError("Both source_ontology and target_ontology must be provided together.")

        if source_ontology and target_ontology:
            ignored_class_index = MetricUtils.get_ignored_class_index(source_ontology)
            ignored_class_index.update(MetricUtils.get_ignored_class_index(target_ontology))
            data["prediction_mappings"] = MetricUtils.remove_ignored_mappings(
                data["prediction_mappings"], ignored_class_index
            )

        results = {}
        for metric in self.metrics:
            prepared_data = metric.prepare(data)
            for partial_data in prepared_data:
                results.update(metric.compute(partial_data))
        return results