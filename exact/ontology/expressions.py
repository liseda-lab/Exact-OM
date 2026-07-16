"""Walkers over backend-neutral OWL class expressions."""

from __future__ import annotations

from collections.abc import Iterable

from exact.ontology.parser import (
    ClassExpression,
    NamedClass,
    ObjectIntersectionOf,
    ObjectSomeValuesFrom,
)


def named_class_iri(expr: ClassExpression) -> str | None:
    """Return the IRI when *expr* is a named class, otherwise ``None``."""

    return expr.iri if isinstance(expr, NamedClass) else None


def named_class_iris(expr: ClassExpression) -> list[str]:
    """Return all named class IRIs nested in an expression, in stable order."""

    if isinstance(expr, NamedClass):
        return [expr.iri]
    if isinstance(expr, ObjectSomeValuesFrom):
        return named_class_iris(expr.filler)
    if isinstance(expr, ObjectIntersectionOf):
        values: list[str] = []
        for operand in expr.operands:
            values.extend(named_class_iris(operand))
        return list(dict.fromkeys(values))
    return []


def existential_targets(expr: ClassExpression, property_iris: Iterable[str]) -> list[str]:
    """Find named fillers of matching ``ObjectSomeValuesFrom`` expressions."""

    properties = frozenset(property_iris)
    targets: list[str] = []

    def visit(current: ClassExpression) -> None:
        if isinstance(current, ObjectSomeValuesFrom):
            if current.property_iri in properties:
                targets.extend(named_class_iris(current.filler))
            else:
                visit(current.filler)
        elif isinstance(current, ObjectIntersectionOf):
            for operand in current.operands:
                visit(operand)

    visit(expr)
    return list(dict.fromkeys(targets))


def intersection_operands(expr: ClassExpression) -> tuple[ClassExpression, ...]:
    """Return top-level intersection operands, or the expression as one operand."""

    if isinstance(expr, ObjectIntersectionOf):
        return expr.operands
    return (expr,)
