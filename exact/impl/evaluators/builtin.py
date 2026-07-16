"""Exact's dependency-free, backward-compatible evaluator backend."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union, cast

from exact.core.contracts.evaluator import IEvaluator
from exact.core.entities.evaluation import (
    BackendEvaluation,
    EvaluationData,
    EvaluationRequest,
    MetricNames,
)
from exact.core.entities.kinds import (
    EntityKind,
    build_entity_kind_index,
    infer_entity_kind,
    normalize_entity_kinds,
)
from exact.core.entities.mappings import EntityMapping, ReferenceMapping
from exact.impl.metrics import (  # noqa: F401 - importing registers the metrics
    F1Metric,
    HitsAtKMetric,
    MeanReciprocalRankMetric,
    PrecisionMetric,
    RecallMetric,
)
from exact.io.sources import resolve as resolve_source
from exact.io.writers.oaei_rdf import read_alignment
from exact.utils.data import save_dict_to_csv
from exact.utils.eval import MetricUtils


class BuiltinEvaluator(IEvaluator):
    """The historical Exact P/R/F1 and ranking implementation."""

    registry_name = "builtin"

    @staticmethod
    def _request_entity_kinds(options: Mapping[str, Any]) -> Tuple[EntityKind, ...]:
        raw = options.get("entity_kinds")
        if raw is None:
            matching = options.get("matching")
            if isinstance(matching, Mapping):
                raw = matching.get("entity_kinds")
            elif matching is not None:
                raw = getattr(matching, "entity_kinds", None)
        return normalize_entity_kinds(raw)

    @staticmethod
    def _mapping_kind(
        mapping: EntityMapping,
        source_ontology: Any,
        target_ontology: Any,
        source_index: Dict[str, EntityKind] | None,
        target_index: Dict[str, EntityKind] | None,
        primary: EntityKind,
    ) -> Tuple[EntityKind, EntityKind]:
        if source_ontology is None or target_ontology is None:
            return (
                EntityKind(getattr(mapping, "src_kind", primary)),
                EntityKind(getattr(mapping, "tgt_kind", primary)),
            )
        src_kind = infer_entity_kind(
            source_ontology,
            mapping.head,
            primary=primary,
            index=source_index,
        )
        tgt_kind = infer_entity_kind(
            target_ontology,
            mapping.tail,
            primary=primary,
            index=target_index,
        )
        mapping.src_kind = src_kind
        mapping.tgt_kind = tgt_kind
        return src_kind, tgt_kind

    @classmethod
    def _filter_mappings_by_kind(
        cls,
        mappings: Iterable[EntityMapping],
        *,
        source_ontology: Any,
        target_ontology: Any,
        entity_kinds: Tuple[EntityKind, ...],
        label: str,
    ) -> List[EntityMapping]:
        allowed = set(entity_kinds)
        primary = entity_kinds[0]
        source_index = (
            build_entity_kind_index(source_ontology) if source_ontology is not None else None
        )
        target_index = (
            build_entity_kind_index(target_ontology) if target_ontology is not None else None
        )
        kept: List[EntityMapping] = []
        removed = 0
        for mapping in mappings:
            src_kind, tgt_kind = cls._mapping_kind(
                mapping,
                source_ontology,
                target_ontology,
                source_index,
                target_index,
                primary,
            )
            if src_kind == tgt_kind and src_kind in allowed:
                kept.append(mapping)
            else:
                removed += 1
        if removed:
            configured = ", ".join(kind.value for kind in entity_kinds)
            warnings.warn(
                f"Filtered {removed} {label} mapping(s) outside configured "
                f"within-kind evaluation pools ({configured}).",
                UserWarning,
                stacklevel=3,
            )
        return kept

    @classmethod
    def _global_metrics(
        cls,
        predictions: List[EntityMapping],
        references: List[ReferenceMapping],
        null_references: List[ReferenceMapping],
    ) -> Dict[str, float]:
        return cls([MetricNames.F1]).evaluate(
            EvaluationData(
                predictions,
                references,
                null_reference_mappings=null_references,
            )
        )

    @classmethod
    def _local_metrics(
        cls,
        mappings: List[Tuple[ReferenceMapping, List[EntityMapping]]],
        K: List[int],
    ) -> Dict[str, float]:
        return cls([MetricNames.HITS_AT_K, MetricNames.MRR]).evaluate(
            EvaluationData(reference_and_candidates=mappings, K=K)
        )

    @staticmethod
    def _json_mapping_value(record: Mapping[str, Any], *names: str) -> Any:
        for name in names:
            if name in record:
                return record[name]
        raise ValueError(f"JSON mapping is missing one of: {', '.join(names)}")

    @classmethod
    def _read_scored_alignment(cls, path: Path) -> List[EntityMapping]:
        """Read a scored global artifact emitted by a registered built-in writer."""

        suffix = path.suffix.lower()
        if suffix in {".rdf", ".xml"}:
            return EntityMapping.read_table_mappings(read_alignment(path))
        if suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, Mapping):
                payload = payload.get("mappings", payload.get("alignment"))
            if not isinstance(payload, list) or not all(
                isinstance(record, Mapping) for record in payload
            ):
                raise ValueError("JSON alignment must be a list of mapping objects")
            mappings: List[EntityMapping] = []
            for record in payload:
                source = cls._json_mapping_value(
                    record, "Src", "SrcEntity", "source", "src", "head"
                )
                target = cls._json_mapping_value(
                    record, "Tgt", "TgtEntity", "target", "tgt", "tail"
                )
                score = cls._json_mapping_value(record, "Score", "score", "confidence", "measure")
                src_kind = record.get(
                    "SrcKind", record.get("Kind", record.get("kind", EntityKind.CLASS))
                )
                mappings.append(
                    EntityMapping(
                        str(source),
                        str(target),
                        str(record.get("Relation", record.get("relation", "="))),
                        float(score),
                        src_kind=src_kind,
                        tgt_kind=record.get("TgtKind", src_kind),
                    )
                )
            return mappings
        return EntityMapping.read_table_mappings(path)

    @classmethod
    def _scored_alignment_with_candidate_pool(
        cls,
        alignment_path: Path,
        candidate_pool_path: Path,
    ) -> List[Tuple[ReferenceMapping, List[EntityMapping]]]:
        """Project sparse scored mappings onto every candidate in a ranking pool."""

        scored = cls._read_scored_alignment(alignment_path)
        pools = MetricUtils.read_candidate_mappings(str(candidate_pool_path))
        by_pair: Dict[Tuple[str, str], EntityMapping] = {}
        for mapping in scored:
            key = mapping.to_tuple()
            previous = by_pair.get(key)
            if previous is None or mapping.score > previous.score:
                by_pair[key] = mapping

        local: List[Tuple[ReferenceMapping, List[EntityMapping]]] = []
        for reference, candidates in pools:
            scored_reference = by_pair.get(reference.to_tuple())
            if scored_reference is not None:
                reference.src_kind = scored_reference.src_kind
                reference.tgt_kind = scored_reference.tgt_kind
            filled: List[EntityMapping] = []
            for candidate in candidates:
                predicted = by_pair.get(candidate.to_tuple())
                filled.append(
                    EntityMapping(
                        candidate.head,
                        candidate.tail,
                        candidate.relation,
                        predicted.score if predicted is not None else 0.0,
                        src_kind=(
                            predicted.src_kind if predicted is not None else reference.src_kind
                        ),
                        tgt_kind=(
                            predicted.tgt_kind if predicted is not None else reference.tgt_kind
                        ),
                    )
                )
            local.append((reference, EntityMapping.sort_entity_mappings_by_score(filled)))
        return local

    @classmethod
    def run(cls, request: EvaluationRequest) -> BackendEvaluation:
        source = (
            resolve_source(request.source) if isinstance(request.source, Path) else request.source
        )
        target = (
            resolve_source(request.target) if isinstance(request.target, Path) else request.target
        )
        entity_kinds = cls._request_entity_kinds(request.options)
        if request.full_reference is not None:
            metrics = cls.global_eval(
                predictions=cast(Union[List[EntityMapping], Path], request.alignment),
                test_reference=request.full_reference,
                train_reference=request.train_reference,
                source_ontology=source,
                target_ontology=target,
                entity_kinds=entity_kinds,
            )
        else:
            metrics = cls.local_eval(
                reference_and_candidates=cast(
                    Union[List[Tuple[ReferenceMapping, List[EntityMapping]]], Path],
                    request.alignment,
                ),
                reference_candidates=request.reference_candidates,
                K=list(request.k),
                source_ontology=source,
                target_ontology=target,
                entity_kinds=entity_kinds,
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
        entity_kinds: Iterable[EntityKind | str] | EntityKind | str | None = None,
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

        selected_kinds = normalize_entity_kinds(entity_kinds)
        prediction_mappings = cls._filter_mappings_by_kind(
            prediction_mappings,
            source_ontology=source_ontology,
            target_ontology=target_ontology,
            entity_kinds=selected_kinds,
            label="prediction",
        )
        reference_mappings = cast(
            List[ReferenceMapping],
            cls._filter_mappings_by_kind(
                reference_mappings,
                source_ontology=source_ontology,
                target_ontology=target_ontology,
                entity_kinds=selected_kinds,
                label="reference",
            ),
        )
        null_reference_mappings = cast(
            List[ReferenceMapping],
            cls._filter_mappings_by_kind(
                null_reference_mappings,
                source_ontology=source_ontology,
                target_ontology=target_ontology,
                entity_kinds=selected_kinds,
                label="null-reference",
            ),
        )

        if source_ontology is not None and target_ontology is not None:
            ignored = MetricUtils.get_ignored_class_index(source_ontology)
            ignored.update(MetricUtils.get_ignored_class_index(target_ontology))
            prediction_mappings = MetricUtils.remove_ignored_mappings(prediction_mappings, ignored)

        metrics = cls._global_metrics(
            prediction_mappings,
            reference_mappings,
            null_reference_mappings,
        )
        if len(selected_kinds) > 1:
            for kind in selected_kinds:
                kind_predictions = [
                    mapping for mapping in prediction_mappings if mapping.src_kind == kind
                ]
                kind_references = [
                    mapping for mapping in reference_mappings if mapping.src_kind == kind
                ]
                kind_nulls = [
                    mapping for mapping in null_reference_mappings if mapping.src_kind == kind
                ]
                for name, value in cls._global_metrics(
                    kind_predictions, kind_references, kind_nulls
                ).items():
                    metrics[f"{kind.value}.{name}"] = value
        return metrics

    @classmethod
    def local_eval(
        cls,
        reference_and_candidates: Union[List[Tuple[ReferenceMapping, List[EntityMapping]]], Path],
        reference_candidates: Optional[Path] = None,
        K: Optional[List[int]] = None,
        source_ontology: Any = None,
        target_ontology: Any = None,
        entity_kinds: Iterable[EntityKind | str] | EntityKind | str | None = None,
    ) -> Dict[str, float]:
        if isinstance(reference_and_candidates, Path):
            if reference_candidates is not None:
                try:
                    MetricUtils.ranking_result_file_check(
                        cand_maps_file=str(reference_and_candidates),
                        ref_cand_maps_file=str(reference_candidates),
                    )
                    local_mappings = MetricUtils.read_candidate_mappings(
                        str(reference_and_candidates)
                    )
                except Exception:
                    try:
                        local_mappings = cls._scored_alignment_with_candidate_pool(
                            reference_and_candidates,
                            reference_candidates,
                        )
                    except Exception as scored_error:
                        raise ValueError(
                            "The alignment is neither a valid legacy ranking result nor a "
                            "supported scored TSV, JSON, or OAEI RDF artifact."
                        ) from scored_error
            else:
                local_mappings = MetricUtils.read_candidate_mappings(str(reference_and_candidates))
        else:
            local_mappings = reference_and_candidates
        selected_kinds = normalize_entity_kinds(entity_kinds)
        filtered_local: List[Tuple[ReferenceMapping, List[EntityMapping]]] = []
        removed_references = 0
        for reference, candidates in local_mappings:
            filtered_reference = cls._filter_mappings_by_kind(
                [reference],
                source_ontology=source_ontology,
                target_ontology=target_ontology,
                entity_kinds=selected_kinds,
                label="local reference",
            )
            if not filtered_reference:
                removed_references += 1
                continue
            filtered_candidates = cls._filter_mappings_by_kind(
                candidates,
                source_ontology=source_ontology,
                target_ontology=target_ontology,
                entity_kinds=selected_kinds,
                label="local candidate",
            )
            filtered_candidates = [
                candidate
                for candidate in filtered_candidates
                if candidate.src_kind == reference.src_kind
                and candidate.tgt_kind == reference.tgt_kind
            ]
            filtered_local.append((reference, filtered_candidates))
        if removed_references:
            # The detailed warnings above identify the filtering reason; this
            # aggregate makes an all-filtered local evaluation explicit.
            warnings.warn(
                f"Skipped {removed_references} local evaluation group(s) for excluded kinds.",
                UserWarning,
                stacklevel=2,
            )

        k_values = K if K is not None else [1, 5, 10]
        metrics = cls._local_metrics(filtered_local, k_values)
        if len(selected_kinds) > 1:
            for kind in selected_kinds:
                kind_local = [
                    (reference, candidates)
                    for reference, candidates in filtered_local
                    if reference.src_kind == kind
                ]
                for name, value in cls._local_metrics(kind_local, k_values).items():
                    metrics[f"{kind.value}.{name}"] = value
        return metrics

    @staticmethod
    def save_results(results: Dict[str, float], output_dir: Path) -> None:
        """Write the historical CSV representation byte-for-byte."""

        save_dict_to_csv(
            data=results,
            file_path=output_dir / "evaluation_results.csv",
            columns=["Metric", "Value"],
        )
