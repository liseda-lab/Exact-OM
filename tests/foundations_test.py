import pytest
import torch

import exact.impl.metrics  # noqa: F401 - imports register metric implementations
from exact.core.contracts.evaluator import IEvaluator
from exact.core.contracts.seed import ISeedSetter
from exact.core.entities.evaluation import (
    BackendEvaluation,
    EvaluationData,
    MetricNames,
)
from exact.core.entities.mappings import EntityMapping, ReferenceMapping
from exact.core.entities.registry import ComponentRegistry, ComponentType
from exact.impl import bootstrap_components
from exact.impl.evaluator import Evaluator
from exact.impl.models.semantic_scorer import SemanticScorer


class _TestEvaluator(IEvaluator):
    @classmethod
    def run(cls, request):
        return BackendEvaluation(metrics={})

    def evaluate(self, data: EvaluationData):
        return {}

    @classmethod
    def global_eval(cls, predictions, test_reference, **kwargs):
        return {}

    @classmethod
    def local_eval(cls, reference_and_candidates, K=None):
        return {}


def test_evaluator_resolves_every_registered_metric() -> None:
    registered_classes = [
        ComponentRegistry.get(ComponentType.METRIC, name)
        for name in ComponentRegistry.list(ComponentType.METRIC)
    ]
    registered_names = [metric_class.metric_name for metric_class in registered_classes]

    evaluator = _TestEvaluator(registered_names)

    assert [metric.metric_name for metric in evaluator.metrics] == registered_names
    assert MetricNames.PRECISION in registered_names


def test_seed_setter_is_resolved_through_the_core_registry() -> None:
    bootstrap_components()

    seed_setter = ComponentRegistry.get(ComponentType.SEED_SETTER, "SeedSetter")

    assert issubclass(seed_setter, ISeedSetter)


def test_evaluation_data_uses_configured_default_k_values() -> None:
    assert EvaluationData().K == [1, 5, 10]


def test_local_evaluation_uses_all_default_k_values() -> None:
    reference = ReferenceMapping("source", "target", "=")
    candidates = [EntityMapping("source", "target", "=", 1.0)]

    results = Evaluator.local_eval([(reference, candidates)])

    assert list(results) == ["Hits@1", "Hits@5", "Hits@10", "MRR"]


def test_llm_calibration_rejects_full_reference_only_labels() -> None:
    with pytest.raises(ValueError, match="training reference"):
        SemanticScorer(
            use_llm=True,
            use_llm_calibration=True,
            llm_model_name="unused",
        )


def test_llm_calibration_samples_are_derived_only_from_training_sources() -> None:
    scorer = SemanticScorer.__new__(SemanticScorer)
    scorer._calibration_messages = []
    scorer._llm_calibration_reference_pairs = {("src-train", "tgt-positive")}
    scorer._llm_calibration_reference_sources = {"src-train"}

    samples = scorer._collect_calibration_samples(
        [0, 1, 2],
        torch.tensor([0.9, 0.2, 0.8]),
        ["src-train", "src-train", "src-test"],
        ["tgt-positive", "tgt-negative", "tgt-from-full-reference"],
    )

    assert samples is not None
    probabilities, labels = samples
    assert probabilities.tolist() == pytest.approx([0.9, 0.2])
    assert labels.tolist() == [1.0, 0.0]
