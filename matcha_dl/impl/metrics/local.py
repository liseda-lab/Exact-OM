
from typing import Dict, List


from matcha_dl.core.contracts.metric import IMetric
from matcha_dl.core.entities.mappings import EntityMapping
from matcha_dl.core.entities.evaluation import EvaluationData, MetricNames

class HitsAtKMetric(IMetric):
    """Implementation of the Hits@K metric."""

    def prepare(self, data: EvaluationData) -> List[EvaluationData]:
        return [
            EvaluationData(
                prediction_mappings=data.prediction_mappings,
                reference_mappings=data.reference_mappings,
                null_reference_mappings=data.null_reference_mappings,
                reference_and_candidates=data.reference_and_candidates,
                K=[k]
            ) for k in data.K or [1]
        ]

    def compute(self, data: EvaluationData) -> Dict[str, float]:
        k = data.K[0]  # Single value of K
        n_hits = 0
        for ref, candidates in data.reference_and_candidates or []:
            ordered_candidates = [c.to_tuple() for c in EntityMapping.sort_entity_mappings_by_score(candidates, k=k)]
            if ref.to_tuple() in ordered_candidates[:k]:
                n_hits += 1
        hits_at_k = n_hits / len(data.reference_and_candidates) if data.reference_and_candidates else 0.0
        return {MetricNames.HITS_AT_K.value.format(k=k): round(hits_at_k, 3)}
    
class MeanReciprocalRankMetric(IMetric):
    """Implementation of the Mean Reciprocal Rank (MRR) metric."""

    def compute(self, data: EvaluationData) -> Dict[str, float]:
        reciprocal_ranks = 0
        for ref, candidates in data.reference_and_candidates or []:
            ordered_candidates = [c.to_tuple() for c in EntityMapping.sort_entity_mappings_by_score(candidates)]
            if ref.to_tuple() in ordered_candidates:
                rank = ordered_candidates.index(ref.to_tuple()) + 1
                reciprocal_ranks += 1 / rank
        mrr = reciprocal_ranks / len(data.reference_and_candidates) if data.reference_and_candidates else 0.0
        return {MetricNames.MRR.value: round(mrr, 3)}