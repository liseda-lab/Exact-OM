from exact.core.entities.evaluation import EvaluationRequest
from exact.core.entities.mappings import EntityMapping, ReferenceMapping
from exact.impl.evaluators.builtin import BuiltinEvaluator


def test_builtin_global_metrics_preserve_historical_names_and_math() -> None:
    predictions = [
        EntityMapping("s1", "t1", "=", 0.9),
        EntityMapping("s2", "t2", "=", 0.8),
    ]
    reference = [ReferenceMapping("s1", "t1", "="), ReferenceMapping("s3", "t3", "=")]

    result = BuiltinEvaluator.run(
        EvaluationRequest(alignment=predictions, full_reference=reference)
    )

    assert result.metrics == {"P": 0.5, "R": 0.5, "F1": 0.5}


def test_builtin_local_metrics_use_requested_cutoffs() -> None:
    reference = ReferenceMapping("s", "gold", "=")
    candidates = [
        EntityMapping("s", "other", "=", 0.9),
        EntityMapping("s", "gold", "=", 0.8),
    ]

    result = BuiltinEvaluator.run(EvaluationRequest(alignment=[(reference, candidates)], k=(1, 2)))

    assert result.metrics == {"Hits@1": 0.0, "Hits@2": 1.0, "MRR": 0.5}
