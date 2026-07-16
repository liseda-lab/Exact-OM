"""Exact's dependency-free, backward-compatible evaluator backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from exact.core.contracts.evaluator import IEvaluator
from exact.core.entities.evaluation import (
    BackendEvaluation,
    EvaluationData,
    EvaluationRequest,
    MetricNames,
)
from exact.core.entities.mappings import EntityMapping, ReferenceMapping
from exact.impl.metrics import (  # noqa: F401 - importing registers the metrics
    F1Metric,
    HitsAtKMetric,
    MeanReciprocalRankMetric,
    PrecisionMetric,
    RecallMetric,
)
from exact.ontology import load_ontology
from exact.utils.data import save_dict_to_csv
from exact.utils.eval import MetricUtils


class BuiltinEvaluator(IEvaluator):
    """The historical Exact P/R/F1 and ranking implementation."""

    registry_name = "builtin"

    @classmethod
    def run(cls, request: EvaluationRequest) -> BackendEvaluation:
        source = (
            load_ontology(request.source) if isinstance(request.source, Path) else request.source
        )
        target = (
            load_ontology(request.target) if isinstance(request.target, Path) else request.target
        )
        if request.full_reference is not None:
            metrics = cls.global_eval(
                predictions=cast(Union[List[EntityMapping], Path], request.alignment),
                test_reference=request.full_reference,
                train_reference=request.train_reference,
                source_ontology=source,
                target_ontology=target,
            )
        else:
            metrics = cls.local_eval(
                reference_and_candidates=cast(
                    Union[List[Tuple[ReferenceMapping, List[EntityMapping]]], Path],
                    request.alignment,
                ),
                reference_candidates=request.reference_candidates,
                K=list(request.k),
            )
        return BackendEvaluation(metrics=metrics, version="builtin-1")

    def evaluate(self, data: EvaluationData) -> Dict[str, float]:
        results: Dict[str, float] = {}
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
        source_ontology: Any = None,
        target_ontology: Any = None,
        train_reference: Optional[Union[List[ReferenceMapping], Path]] = None,
        threshold: Optional[float] = None,
    ) -> Dict[str, float]:
        if (source_ontology is None) != (target_ontology is None):
            raise ValueError("Both source_ontology and target_ontology must be provided together.")

        prediction_mappings = (
            EntityMapping.read_table_mappings(predictions, threshold=threshold)
            if isinstance(predictions, Path)
            else predictions
        )
        reference_mappings = (
            ReferenceMapping.read_table_mappings(test_reference)
            if isinstance(test_reference, Path)
            else test_reference
        )
        null_reference_mappings = (
            ReferenceMapping.read_table_mappings(train_reference)
            if isinstance(train_reference, Path)
            else train_reference or []
        )

        if source_ontology is not None and target_ontology is not None:
            ignored = MetricUtils.get_ignored_class_index(source_ontology)
            ignored.update(MetricUtils.get_ignored_class_index(target_ontology))
            prediction_mappings = MetricUtils.remove_ignored_mappings(prediction_mappings, ignored)

        return cls([MetricNames.F1]).evaluate(
            EvaluationData(
                prediction_mappings,
                reference_mappings,
                null_reference_mappings=null_reference_mappings,
            )
        )

    @classmethod
    def local_eval(
        cls,
        reference_and_candidates: Union[List[Tuple[ReferenceMapping, List[EntityMapping]]], Path],
        reference_candidates: Optional[Path] = None,
        K: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        if isinstance(reference_and_candidates, Path):
            if reference_candidates is not None:
                try:
                    MetricUtils.ranking_result_file_check(
                        cand_maps_file=str(reference_and_candidates),
                        ref_cand_maps_file=str(reference_candidates),
                    )
                except AssertionError as exc:
                    raise ValueError(
                        "The file does not have the correct format for ranking results."
                    ) from exc
            local_mappings = MetricUtils.read_candidate_mappings(str(reference_and_candidates))
        else:
            local_mappings = reference_and_candidates

        return cls([MetricNames.HITS_AT_K, MetricNames.MRR]).evaluate(
            EvaluationData(
                reference_and_candidates=local_mappings,
                K=K if K is not None else [1, 5, 10],
            )
        )

    @staticmethod
    def save_results(results: Dict[str, float], output_dir: Path) -> None:
        """Write the historical CSV representation byte-for-byte."""

        save_dict_to_csv(
            data=results,
            file_path=output_dir / "evaluation_results.csv",
            columns=["Metric", "Value"],
        )
