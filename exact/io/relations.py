"""Deterministic hierarchy heuristic for typed alignment relations."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

from exact.core.contracts.knowledge import KnowledgeSource
from exact.io.writers._frames import validated_mapping_frame
from exact.io.writers.base import WriterOptionsError


def _closure(
    source: KnowledgeSource,
    iri: str,
    *,
    direct: Callable[[str], list[str]],
    optimized_name: str,
) -> set[str]:
    optimized = getattr(source, optimized_name, None)
    if callable(optimized):
        return {str(value) for value in optimized(str(iri))}
    seen: set[str] = set()
    pending = list(direct(str(iri)))
    while pending:
        current = str(pending.pop())
        if current in seen:
            continue
        seen.add(current)
        pending.extend(direct(current))
    return seen


def _ancestors(source: KnowledgeSource, iri: str) -> set[str]:
    return _closure(
        source,
        iri,
        direct=source.direct_parents,
        optimized_name="ancestors",
    )


def _descendants(source: KnowledgeSource, iri: str) -> set[str]:
    return _closure(
        source,
        iri,
        direct=source.direct_children,
        optimized_name="descendants",
    )


def _anchor_frame(candidates: pd.DataFrame, anchors: Any | None) -> pd.DataFrame:
    if anchors is not None:
        return validated_mapping_frame(anchors)
    for marker in ("Anchor", "anchor", "is_anchor", "protected"):
        if marker in candidates.columns:
            marked = candidates[candidates[marker].fillna(False).astype(bool)]
            if not marked.empty:
                return marked.copy()
    return (
        candidates.sort_values(
            ["Score", "TgtEntity"],
            ascending=[False, True],
            kind="mergesort",
        )
        .drop_duplicates("SrcEntity", keep="first")
        .reset_index(drop=True)
    )


def predict_relations(
    candidates: Any,
    source: KnowledgeSource,
    target: KnowledgeSource,
    *,
    mode: str = "none",
    anchors: Any | None = None,
) -> pd.DataFrame:
    """Add ``Relation`` and ``relation_confidence`` to a candidate table.

    ``hierarchy_heuristic`` uses the highest-scoring target image for each
    source (or explicitly supplied anchors). A candidate target above that
    image, or matching an anchored source ancestor, supports ``<``. The
    symmetric descendant evidence supports ``>``. Anchor scores weight the
    evidence; equal or absent directional evidence remains ``=``.

    This intentionally does not infer new correspondences or call an LLM.
    """

    frame = validated_mapping_frame(candidates)
    normalized_mode = str(mode).strip().lower()
    if normalized_mode == "none":
        frame["Relation"] = "="
        frame["relation_confidence"] = 1.0
        return frame
    if normalized_mode != "hierarchy_heuristic":
        raise WriterOptionsError("relation prediction must be 'none' or 'hierarchy_heuristic'")

    anchor_frame = _anchor_frame(frame, anchors)
    images: dict[str, list[tuple[str, float]]] = {}
    for row in anchor_frame.itertuples(index=False):
        weight = max(float(row.Score), 1e-12)
        images.setdefault(str(row.SrcEntity), []).append((str(row.TgtEntity), weight))

    source_ancestor_cache: dict[str, set[str]] = {}
    source_descendant_cache: dict[str, set[str]] = {}
    target_ancestor_cache: dict[str, set[str]] = {}
    target_descendant_cache: dict[str, set[str]] = {}

    def ancestors_of(knowledge: KnowledgeSource, iri: str, *, target_side: bool) -> set[str]:
        cache = target_ancestor_cache if target_side else source_ancestor_cache
        if iri not in cache:
            cache[iri] = _ancestors(knowledge, iri)
        return cache[iri]

    def descendants_of(knowledge: KnowledgeSource, iri: str, *, target_side: bool) -> set[str]:
        cache = target_descendant_cache if target_side else source_descendant_cache
        if iri not in cache:
            cache[iri] = _descendants(knowledge, iri)
        return cache[iri]

    relations: list[str] = []
    confidences: list[float] = []
    for row in frame.itertuples(index=False):
        src, tgt = str(row.SrcEntity), str(row.TgtEntity)
        subsumed_by = 0.0
        subsumes = 0.0
        equivalent = 0.0

        for image, weight in images.get(src, ()):
            if tgt == image:
                equivalent += weight
            elif tgt in ancestors_of(target, image, target_side=True):
                subsumed_by += weight
            elif tgt in descendants_of(target, image, target_side=True):
                subsumes += weight

        for ancestor in ancestors_of(source, src, target_side=False):
            for image, weight in images.get(ancestor, ()):
                if tgt == image:
                    subsumed_by += weight
        for descendant in descendants_of(source, src, target_side=False):
            for image, weight in images.get(descendant, ()):
                if tgt == image:
                    subsumes += weight

        total = subsumed_by + subsumes + equivalent
        if subsumed_by > max(subsumes, equivalent):
            relation, dominant = "<", subsumed_by
        elif subsumes > max(subsumed_by, equivalent):
            relation, dominant = ">", subsumes
        else:
            relation, dominant = "=", equivalent
        relations.append(relation)
        confidences.append(0.0 if total == 0.0 else dominant / total)

    frame["Relation"] = relations
    frame["relation_confidence"] = confidences
    return frame


__all__ = ["predict_relations"]
