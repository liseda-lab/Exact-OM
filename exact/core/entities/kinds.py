from __future__ import annotations

import warnings
from collections.abc import Iterable
from enum import Enum
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from exact.core.contracts.knowledge import KnowledgeSource


class EntityKind(str, Enum):
    """Kinds of named entities exposed by a knowledge source."""

    CLASS = "class"
    OBJECT_PROPERTY = "object_property"
    DATA_PROPERTY = "data_property"
    ANNOTATION_PROPERTY = "annotation_property"
    INDIVIDUAL = "individual"


MATCHABLE_ENTITY_KINDS = (
    EntityKind.CLASS,
    EntityKind.OBJECT_PROPERTY,
    EntityKind.DATA_PROPERTY,
    EntityKind.INDIVIDUAL,
)


def normalize_entity_kinds(
    values: Iterable[EntityKind | str] | EntityKind | str | None,
    *,
    allow_annotation_properties: bool = False,
) -> tuple[EntityKind, ...]:
    """Validate and de-duplicate configured entity kinds in stable order.

    ``None`` intentionally means the historical class-only mode. Annotation
    properties are present in the backend contract but matching them remains a
    deferred capability, so callers must opt into accepting that enum value.
    """

    if values is None:
        return (EntityKind.CLASS,)
    if isinstance(values, (str, EntityKind)):
        values = (values,)
    normalized: list[EntityKind] = []
    for value in values:
        try:
            kind = EntityKind(value)
        except ValueError as exc:
            available = ", ".join(kind.value for kind in EntityKind)
            raise ValueError(
                f"Unknown entity kind {value!r}; expected one of: {available}"
            ) from exc
        if kind == EntityKind.ANNOTATION_PROPERTY and not allow_annotation_properties:
            raise ValueError(
                "annotation_property matching is not implemented; select class, "
                "object_property, data_property, or individual"
            )
        if kind not in normalized:
            normalized.append(kind)
    if not normalized:
        raise ValueError("At least one entity kind must be configured")
    return tuple(normalized)


def build_entity_kind_index(
    source: "KnowledgeSource",
    *,
    kinds: Iterable[EntityKind] = tuple(EntityKind),
) -> dict[str, EntityKind]:
    """Build a deterministic IRI-to-kind lookup from source signatures."""

    index: dict[str, EntityKind] = {}
    for kind in kinds:
        normalized_kind = EntityKind(kind)
        for iri in source.entities(normalized_kind):
            index.setdefault(str(iri), normalized_kind)
    return index


def infer_entity_kind(
    source: "KnowledgeSource",
    iri: str,
    *,
    primary: EntityKind = EntityKind.CLASS,
    index: dict[str, EntityKind] | None = None,
    warn: bool = True,
    warning_callback: Callable[[str], None] | None = None,
) -> EntityKind:
    """Resolve an IRI's signature kind, defaulting unknown legacy IRIs safely."""

    lookup = index if index is not None else build_entity_kind_index(source)
    resolved = lookup.get(str(iri))
    if resolved is not None:
        return resolved
    message = (
        f"Entity IRI {iri!r} is absent from the knowledge-source signature; "
        f"defaulting to configured primary kind {primary.value!r}."
    )
    if warn:
        if warning_callback is not None:
            warning_callback(message)
        else:
            warnings.warn(message, UserWarning, stacklevel=2)
    return EntityKind(primary)


__all__ = [
    "EntityKind",
    "MATCHABLE_ENTITY_KINDS",
    "build_entity_kind_index",
    "infer_entity_kind",
    "normalize_entity_kinds",
]
