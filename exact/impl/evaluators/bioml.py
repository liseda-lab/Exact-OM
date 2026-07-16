"""Adapter for the optional OAEI Bio-ML evaluation package.

This is intentionally the only Exact module that resolves upstream symbols.
The imports stay behind :func:`_load_bioml_api` so normal Exact imports never
require the pre-release package.
"""

from __future__ import annotations

import importlib
from ast import literal_eval
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence, cast, runtime_checkable

from exact.core.contracts.evaluator import IEvaluator
from exact.core.entities.evaluation import BackendEvaluation, EvaluationRequest
from exact.core.entities.mappings import EntityMapping, ReferenceMapping
from exact.utils.data import read_table


class BioMLDependencyError(RuntimeError):
    """Raised when the explicitly requested optional backend is unavailable."""


def _numeric_mapping(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise TypeError("upstream evaluator did not return a metric mapping")
    return {
        str(name): float(metric)
        for name, metric in value.items()
        if isinstance(metric, (int, float)) and not isinstance(metric, bool)
    }


@runtime_checkable
class BioMLApi(Protocol):
    """Stable seam between Exact and the evolving upstream package."""

    version: str
    missing: frozenset[str]

    def evaluate_equivalence(self, request: EvaluationRequest) -> Mapping[str, float]: ...

    def evaluate_ranking(self, request: EvaluationRequest) -> Mapping[str, float]: ...

    def evaluate_typed(self, request: EvaluationRequest) -> Mapping[str, float]: ...

    def structural_coherence(self, request: EvaluationRequest) -> Mapping[str, float]: ...


def _optional_symbol(module_name: str, symbol: str) -> Callable[..., Any] | None:
    try:
        module = importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError):
        return None
    value = getattr(module, symbol, None)
    return value if callable(value) else None


class _UpstreamBioMLApi:
    def __init__(self, package: Any):
        self.version = str(getattr(package, "__version__", "unknown"))
        self._score_global = _optional_symbol(
            "oaei_bioml_eval.equivalence.report", "score_global_files"
        )
        self._global_metric = _optional_symbol("oaei_bioml_eval.equivalence.metrics", "global_prf1")
        self._ranking_metric = _optional_symbol(
            "oaei_bioml_eval.equivalence.metrics", "local_ranking_metrics"
        )
        self._score_typed = _optional_symbol("oaei_bioml_eval.typed.report", "score_files")
        self._score_structural = _optional_symbol(
            "oaei_bioml_eval.coherence.report", "score_structural_proxy_files"
        )
        missing = set()
        if self._score_global is None and self._global_metric is None:
            missing.add("equivalence")
        if self._ranking_metric is None:
            missing.add("ranking")
        if self._score_typed is None:
            missing.add("typed")
        if self._score_structural is None:
            missing.add("structural")
        self.missing = frozenset(missing)

    def evaluate_equivalence(self, request: EvaluationRequest) -> Mapping[str, float]:
        if isinstance(request.alignment, Path) and isinstance(request.full_reference, Path):
            if self._score_global is None:
                raise NotImplementedError("upstream global file scorer is unavailable")
            return _numeric_mapping(self._score_global(request.alignment, request.full_reference))

        if self._global_metric is None:
            raise NotImplementedError("upstream pure global metric is unavailable")
        if isinstance(request.alignment, Path) or isinstance(request.full_reference, Path):
            raise TypeError("global evaluation inputs must both be paths or in-memory mappings")
        predicted = {
            (mapping.head, mapping.tail)
            for mapping in request.alignment
            if isinstance(mapping, EntityMapping)
        }
        reference = {
            (mapping.head, mapping.tail)
            for mapping in request.full_reference or []
            if isinstance(mapping, ReferenceMapping)
        }
        return _numeric_mapping(self._global_metric(predicted, reference))

    @staticmethod
    def _ranking_inputs(
        request: EvaluationRequest,
    ) -> tuple[dict[int, Sequence[str]], dict[int, str]]:
        if isinstance(request.alignment, Path):
            rows = read_table(request.alignment).values.tolist()
            parsed = []
            for source, target, candidates in rows:
                values = literal_eval(str(candidates))
                scored = []
                for index, value in enumerate(values):
                    if isinstance(value, str):
                        scored.append((value, float(len(values) - index)))
                    else:
                        scored.append((str(value[0]), float(value[1])))
                scored.sort(key=lambda item: (-round(item[1], 12), item[0]))
                parsed.append((str(source), str(target), [target for target, _ in scored]))
        else:
            parsed = []
            local_rows = cast(
                Sequence[tuple[ReferenceMapping, list[EntityMapping]]], request.alignment
            )
            for reference, candidates in local_rows:
                ranked = sorted(candidates, key=lambda item: (-round(item.score, 12), item.tail))
                parsed.append((reference.head, reference.tail, [item.tail for item in ranked]))
        rankings: dict[int, Sequence[str]] = {index: row[2] for index, row in enumerate(parsed)}
        golds = {index: row[1] for index, row in enumerate(parsed)}
        return rankings, golds

    def evaluate_ranking(self, request: EvaluationRequest) -> Mapping[str, float]:
        if self._ranking_metric is None:
            raise NotImplementedError("upstream pure ranking metric is unavailable")
        rankings, golds = self._ranking_inputs(request)
        return _numeric_mapping(self._ranking_metric(rankings, golds, hits_ks=tuple(request.k)))

    def evaluate_typed(self, request: EvaluationRequest) -> Mapping[str, float]:
        if self._score_typed is None:
            raise NotImplementedError("upstream typed scorer is unavailable")
        options = dict(request.options)
        predictions = options.pop("typed_submission_path", None)
        answers = options.pop("typed_answers_path", None)
        if predictions is None or answers is None:
            raise ValueError("typed_submission_path and typed_answers_path are required")
        allowed = {
            "preferred_pairs_path",
            "graded_relevance_path",
            "hierarchy_path",
            "submission_format",
            "relations",
            "candidate_count",
            "k",
            "max_distance",
        }
        kwargs = {key: value for key, value in options.items() if key in allowed}
        return _numeric_mapping(self._score_typed(predictions, answers, **kwargs))

    def structural_coherence(self, request: EvaluationRequest) -> Mapping[str, float]:
        if self._score_structural is None:
            raise NotImplementedError("upstream structural proxy is unavailable")
        source = getattr(request.source, "origin", request.source)
        target = getattr(request.target, "origin", request.target)
        return _numeric_mapping(self._score_structural(request.alignment, source, target))


def _load_bioml_api() -> BioMLApi:
    """Discover upstream capabilities and return a stable partial adapter."""

    try:
        package = importlib.import_module("oaei_bioml_eval")
    except ModuleNotFoundError as exc:
        if exc.name != "oaei_bioml_eval":
            raise
        raise BioMLDependencyError(
            'BioML evaluation is optional; install it with `pip install "exact-om[bioml-eval]"`.'
        ) from exc
    return _UpstreamBioMLApi(package)


class BioMLEvaluator(IEvaluator):
    """Capability-tolerant wrapper around ``oaei_bioml_eval``."""

    registry_name = "bioml"

    _PLACEHOLDERS = {
        "equivalence": ("equivalence.precision", "equivalence.recall", "equivalence.f1"),
        "ranking": ("ranking.mrr",),
        "typed": ("typed.typed_mrr", "typed.hierarchy_aware_ndcg_at_10"),
        "structural": ("structural.proxy",),
    }

    @staticmethod
    def _record(
        category: str,
        values: Mapping[str, Any],
        metrics: dict[str, float | None],
    ) -> None:
        for name, value in values.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                metrics[f"{category}.{str(name).lower()}"] = float(value)

    @classmethod
    def run(cls, request: EvaluationRequest) -> BackendEvaluation:
        api = _load_bioml_api()
        metrics: dict[str, float | None] = {}
        skipped: dict[str, str] = {}

        def execute(category: str, callback: Callable[[EvaluationRequest], Mapping[str, float]]):
            if category in api.missing:
                reason = f"upstream capability {category!r} is not available"
                for name in cls._PLACEHOLDERS[category]:
                    metrics[name] = None
                    skipped[name] = reason
                return
            try:
                values = callback(request)
            except (NotImplementedError, TypeError, ValueError) as exc:
                reason = str(exc) or f"upstream capability {category!r} is unavailable"
                for name in cls._PLACEHOLDERS[category]:
                    metrics[name] = None
                    skipped[name] = reason
                return
            cls._record(category, values, metrics)

        if request.full_reference is not None:
            execute("equivalence", api.evaluate_equivalence)
        else:
            execute("ranking", api.evaluate_ranking)

        if request.options.get("typed_submission_path") or request.options.get(
            "typed_answers_path"
        ):
            execute("typed", api.evaluate_typed)

        if request.source is not None and request.target is not None:
            execute("structural", api.structural_coherence)

        return BackendEvaluation(metrics=metrics, skipped=skipped, version=api.version)

    def evaluate(self, data):
        raise NotImplementedError("BioMLEvaluator consumes EvaluationRequest via run()")

    @classmethod
    def global_eval(cls, predictions, test_reference, **kwargs):
        request = EvaluationRequest(
            alignment=predictions,
            full_reference=test_reference,
            train_reference=kwargs.get("train_reference"),
            source=kwargs.get("source_ontology"),
            target=kwargs.get("target_ontology"),
            options=kwargs.get("options") or {},
        )
        return dict(cls.run(request).metrics)

    @classmethod
    def local_eval(cls, reference_and_candidates, reference_candidates=None, K=None, **kwargs):
        request = EvaluationRequest(
            alignment=reference_and_candidates,
            reference_candidates=reference_candidates,
            k=tuple(K or (1, 5, 10)),
            options=kwargs.get("options") or {},
        )
        return dict(cls.run(request).metrics)


__all__ = [
    "BioMLApi",
    "BioMLDependencyError",
    "BioMLEvaluator",
    "_load_bioml_api",
]
