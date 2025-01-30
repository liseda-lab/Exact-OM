
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path

from mowl.owlapi import OWLOntology

from matcha_dl.core.contracts.evaluator import IEvaluator
from matcha_dl.core.contracts.dataset import IDataset
from matcha_dl.core.entities.configs import ComponentRegistry
from matcha_dl.core.entities.evaluation import EvaluationData, MetricNames
from matcha_dl.core.entities.mappings import EntityMapping, ReferenceMapping
from matcha_dl.impl.utils import MetricUtils


class Evaluator(IEvaluator):
    def __init__(self, metrics: List[MetricNames]):
        self.metrics = [
            ComponentRegistry.get_metric("metric", metric.value)() for metric in metrics
        ]

    def evaluate(
        self,
        data: EvaluationData,
    ) -> Dict[str, float]:

        results = {}
        for metric in self.metrics:
            prepared_data = metric.prepare(data)
            for partial_data in prepared_data:
                results.update(metric.compute(partial_data))
        return results
    
    @classmethod
    def global_eval(
        cls, 
        predictions: Union[List[EntityMapping], Path],
        test_reference: Union[List[ReferenceMapping], Path],
        source_ontology: Optional[OWLOntology] = None,
        target_ontology: Optional[OWLOntology] = None,
        train_reference: Optional[List[ReferenceMapping]] = None,
        threshold: Optional[float] = None,
    ) -> Dict[str, float]:
        
        if (source_ontology and not target_ontology) or (target_ontology and not source_ontology):
            raise ValueError("Both source_ontology and target_ontology must be provided together.")
        
        if isinstance(predictions, Path):
            predictions = EntityMapping.read_table_mappings(predictions, threshold=threshold)

        if isinstance(test_reference, Path):
            test_reference = ReferenceMapping.read_table_mappings(test_reference)

        if train_reference:
            if isinstance(train_reference, Path):
                train_reference = ReferenceMapping.read_table_mappings(train_reference)
        else:
            train_reference = []

        if source_ontology and target_ontology:
            ignored_class_index = MetricUtils.get_ignored_class_index(source_ontology)
            ignored_class_index.update(MetricUtils.get_ignored_class_index(target_ontology))
            predictions = MetricUtils.remove_ignored_mappings(predictions, ignored_class_index)

        return cls([MetricNames.F1]).evaluate(
            EvaluationData(predictions, test_reference, null_reference_mappings=train_reference)
        )
    
    @classmethod
    def local_eval(
        cls,
        reference_and_candidates: Union[List[Tuple[ReferenceMapping, List[EntityMapping]]], Path],
        K: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        
        if isinstance(reference_and_candidates, Path):
            reference_and_candidates = EntityMapping.read_table_mappings(reference_and_candidates)


        try:
            MetricUtils.ranking_result_file_check(reference_and_candidates)
        except AssertionError:
            raise ValueError("The file does not have the correct format for ranking results.")

        reference_and_candidates = MetricUtils.read_candidate_mappings(reference_and_candidates)

        return cls([MetricNames.HITS_AT_K, MetricNames.MRR]).evaluate(
            EvaluationData(reference_and_candidates=reference_and_candidates, K=K)
        )