from __future__ import annotations

import importlib
import sys

import pytest

from exact.core.actions import evaluation as evaluation_actions
from exact.core.entities.mappings import EntityMapping


def _pairs(mappings: list[EntityMapping]) -> list[tuple[str, str]]:
    return EntityMapping.as_tuples(mappings)


def test_source_cardinality_ties_use_ascending_target_iri() -> None:
    mappings = [
        EntityMapping("source", "target-z", score=0.8),
        EntityMapping("source", "target-a", score=0.8),
        EntityMapping("source", "target-m", score=0.8),
    ]

    forward = EntityMapping.filter_top_n_entity_mappings(mappings, 2)
    reversed_input = EntityMapping.filter_top_n_entity_mappings(list(reversed(mappings)), 2)

    assert _pairs(forward) == [("source", "target-a"), ("source", "target-m")]
    assert _pairs(reversed_input) == _pairs(forward)


def test_target_cardinality_ties_use_ascending_source_iri() -> None:
    mappings = [
        EntityMapping("source-z", "target", score=0.8),
        EntityMapping("source-a", "target", score=0.8),
        EntityMapping("source-m", "target", score=0.8),
    ]

    forward = EntityMapping.filter_top_n_target_entity_mappings(mappings, 2)
    reversed_input = EntityMapping.filter_top_n_target_entity_mappings(list(reversed(mappings)), 2)

    assert _pairs(forward) == [("source-a", "target"), ("source-m", "target")]
    assert _pairs(reversed_input) == _pairs(forward)


def test_cardinality_tie_break_keeps_protected_pairs_first() -> None:
    mappings = [
        EntityMapping("source", "target-a", score=1.0),
        EntityMapping("source", "target-z", score=0.1),
    ]

    filtered = EntityMapping.filter_top_n_entity_mappings(
        mappings,
        1,
        protected_pairs={("source", "target-z")},
    )

    assert _pairs(filtered) == [("source", "target-z")]


def test_evaluation_action_is_a_deprecated_function_alias(monkeypatch) -> None:
    sentinel = object()
    monkeypatch.setattr(evaluation_actions, "run_evaluation", lambda **kwargs: sentinel)

    with pytest.deprecated_call(match="run_evaluation"):
        result = evaluation_actions.EvaluationAction.run(
            alignment=[],
            output_dir_path="unused",
        )

    assert result is sentinel


def test_legacy_llm_routing_module_aliases_new_subsystem() -> None:
    routing = importlib.import_module("exact.llm.routing")
    sys.modules.pop("exact.utils.llm_routing", None)

    with pytest.deprecated_call(match="exact.llm.routing"):
        legacy = importlib.import_module("exact.utils.llm_routing")

    assert legacy is routing
