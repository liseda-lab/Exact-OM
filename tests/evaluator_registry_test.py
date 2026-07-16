from exact.core.entities.registry import ComponentRegistry, ComponentType
from exact.impl.evaluators import BioMLEvaluator, BuiltinEvaluator


def test_evaluator_backends_are_registered_by_stable_names() -> None:
    assert ComponentRegistry.get(ComponentType.EVALUATOR, "builtin") is BuiltinEvaluator
    assert ComponentRegistry.get(ComponentType.EVALUATOR, "bioml") is BioMLEvaluator


def test_evaluator_interface_is_not_registered() -> None:
    assert "IEvaluator" not in ComponentRegistry.list(ComponentType.EVALUATOR)
