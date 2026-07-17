"""Deprecated result helpers using public :mod:`pyowl_core` visitors."""

from __future__ import annotations

from collections.abc import Iterable

import pyowl_core
from pyowl_core import (
    Class,
    ClassExpression,
    ObjectIntersectionOf,
    ObjectProperty,
    ObjectSomeValuesFrom,
    StructuralNode,
    walk,
)


def named_class_iri(expression: ClassExpression) -> str | None:
    """Return the IRI of a core named class expression."""

    return expression.iri.value if isinstance(expression, Class) else None


def named_class_iris(expression: ClassExpression) -> list[str]:
    """Return named classes encountered by the core's exhaustive walker."""

    return list(
        dict.fromkeys(node.iri.value for node in walk(expression) if isinstance(node, Class))
    )


def existential_targets(expression: ClassExpression, property_iris: Iterable[str]) -> list[str]:
    """Return named fillers for matching core object-existential occurrences."""

    selected = frozenset(map(str, property_iris))
    targets: list[str] = []
    for node in walk(expression):
        if not isinstance(node, ObjectSomeValuesFrom):
            continue
        if isinstance(node.property, ObjectProperty) and node.property.iri.value in selected:
            targets.extend(named_class_iris(node.filler))
    return list(dict.fromkeys(targets))


def intersection_operands(
    expression: ClassExpression,
) -> tuple[ClassExpression, ...]:
    """Return top-level core intersection operands or the expression itself."""

    if isinstance(expression, ObjectIntersectionOf):
        return tuple(expression.operands)
    return (expression,)


def render_class_expression(expression: StructuralNode) -> str:
    """Return a stable compatibility display without a local OWL renderer."""

    if isinstance(expression, Class):
        return f"<{expression.iri.value}>"
    return f"{type(expression).__name__}(" f"{pyowl_core.structural_hexdigest(expression)})"


__all__ = [
    "existential_targets",
    "intersection_operands",
    "named_class_iri",
    "named_class_iris",
    "render_class_expression",
]
