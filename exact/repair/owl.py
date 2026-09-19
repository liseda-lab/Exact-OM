"""Policy verification using the public, shared-snapshot OWL reasoner APIs.

No logical conclusions are drawn from projected or inferred matching features.
The caller supervises these operations in a bounded worker; a backend's optional
query timeout alone is not a wall-clock bound on compilation and cleanup.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from importlib import import_module
from math import isfinite
from typing import Any

import pyowl_core as owl

from exact.ontology.versions import distribution_version


@dataclass(frozen=True)
class ObligationResult:
    """An actual query result, with the policy's expected truth value."""

    kind: str
    query_id: str
    verdict: bool | None
    expected: bool = True
    complete: bool = False
    reason: str | None = None
    class_iri: str | None = None

    @property
    def satisfied(self) -> bool | None:
        """Return unknown for incomplete results, including incomplete booleans."""
        return self.verdict == self.expected if self.complete else None


@dataclass(frozen=True)
class SupportReport:
    """Support qualified against the actual input and completed query calls."""

    reasoner: str
    package_version: str
    backend: str
    implementation_version: str
    input_supported: bool
    complete_imports: bool
    constructs: tuple[str, ...]
    query_families: tuple[str, ...]
    issues: tuple[str, ...] = ()
    diagnostics: tuple[tuple[str, str | int | float | bool], ...] = ()
    query_support: tuple[tuple[str, str, bool], ...] = ()


@dataclass(frozen=True)
class CheckReport:
    """Low-level evidence translated into the shared v2 verification record."""

    logical_status: str
    verification_scope: str
    theory_hash: str
    obligations: tuple[ObligationResult, ...]
    support: SupportReport
    schema_version: str = "exact-repair/owl-check/v2"

    @property
    def complete(self) -> bool:
        """Whether every obligation was decided on a supported complete input."""
        return self.support.input_supported and all(item.complete for item in self.obligations)

    @property
    def unsatisfiable_classes(self) -> tuple[str, ...]:
        """Return only completely decided named-class failures."""
        return tuple(
            item.class_iri
            for item in self.obligations
            if item.kind == "class_satisfiability"
            and item.complete
            and item.verdict is False
            and item.class_iri is not None
        )

    def to_dict(self) -> dict[str, Any]:
        """Expose plain serializable evidence without a live backend handle."""
        return asdict(self)


@dataclass(frozen=True)
class BaselineDiagnosis:
    """Four distinct baselines and frozen, source-only exception evidence."""

    source: CheckReport
    target: CheckReport
    union: CheckReport
    alignment: CheckReport
    exceptions: tuple[str, ...] = ()
    exception_evidence: tuple[tuple[str, str, str, str], ...] = ()
    schema_version: str = "exact-repair/owl-baseline/v2"


@lru_cache(maxsize=1)
def _empty_snapshot() -> owl.OntologyView:
    # Parse a constant empty document once, never serialize/reparse source axioms.
    # OntologyDelta preserves the original shared AST and anonymous identities.
    return owl.load_snapshot(
        b"Ontology(<urn:exact:repair:asserted-view>)",
        options=owl.LoadOptions(backend=owl.BackendPreference.PYTHON),
    )


def _declarations(axioms: Iterable[owl.AxiomNode]) -> set[owl.Declaration]:
    bounds = {
        owl.OWL_THING,
        owl.OWL_NOTHING,
        owl.OWL_TOP_OBJECT_PROPERTY,
        owl.OWL_BOTTOM_OBJECT_PROPERTY,
        owl.OWL_TOP_DATA_PROPERTY,
        owl.OWL_BOTTOM_DATA_PROPERTY,
    }
    return {
        owl.Declaration(entity)
        for axiom in axioms
        for entity in owl.signature(axiom)
        if entity not in bounds
    }


def snapshot_from_axioms(axioms: Iterable[owl.AxiomNode]) -> owl.OntologyView:
    """Create a fresh asserted repair view with semantically inert declarations.

    Occurrence selection must happen before this function: an axiom still emitted
    by a fixed import or another object remains present. Declarations preserve
    typed entity identities (including legal punning) for the DL profile gate.
    """
    selected = tuple(axioms)
    if not all(isinstance(axiom, owl.AxiomNode) for axiom in selected):
        raise TypeError("repair theories must contain shared pyowl-core axioms")
    return owl.apply_delta(
        _empty_snapshot(),
        owl.OntologyDelta(add_axioms=owl.CanonicalSet((*selected, *_declarations(selected)))),
    )


def named_classes(snapshot: owl.OntologyView) -> tuple[str, ...]:
    """Return the whole asserted class signature except owl:Nothing."""
    return tuple(
        sorted(
            entity.iri.value
            for entity in snapshot.signature(owl.EntityKind.CLASS)
            if entity != owl.OWL_NOTHING
        )
    )


def _class(value: str | owl.Class) -> owl.Class:
    return owl.Class(owl.IRI(value)) if isinstance(value, str) else value


def _query_id(query: owl.StructuralNode | None) -> str:
    return "consistency" if query is None else hashlib.sha256(query.canonical_bytes()).hexdigest()


def _failure(error: Exception) -> str:
    """Keep timeout, unsupported, unavailable and failed calls distinguishable."""
    code = str(getattr(error, "code", type(error).__name__))
    if isinstance(error, TimeoutError):
        category = "timeout"
    elif isinstance(error, ImportError) or "Unavailable" in type(error).__name__:
        category = "backend_unavailable"
    elif isinstance(error, NotImplementedError) or any(
        text in type(error).__name__ for text in ("Unsupported", "Profile", "Incomplete", "Import")
    ):
        category = "unsupported"
    else:
        category = "query_failure"
    return f"{category}:{code}: {error}"


class OwlVerifier:
    """Narrow complete-check adapter over optional pyHermiT or pyELK 0.2 APIs.

    HermiT accepts only its validated OWL 2 DL input and supported datatypes.
    ELK uses strict compilation and requires per-query completeness. Neither
    backend may silently discard an unsupported axiom and authorize a repair.
    """

    def __init__(
        self,
        reasoner: str = "hermit",
        *,
        backend: str = "auto",
        timeout_seconds: float | None = None,
        workers: int = 1,
    ) -> None:
        if reasoner not in {"hermit", "elk"}:
            raise ValueError("repair requires the qualified 'hermit' or 'elk' backend")
        if timeout_seconds is not None and (not isfinite(timeout_seconds) or timeout_seconds <= 0):
            raise ValueError("timeout_seconds must be finite and positive")
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
            raise ValueError("workers must be a positive integer")
        self.reasoner = reasoner
        self.backend = backend
        self.timeout_seconds = timeout_seconds
        self.workers = workers

    def _open(self, snapshot: owl.OntologyView) -> tuple[str, Any]:
        module = import_module("pyhermit" if self.reasoner == "hermit" else "pyelk")
        package_version = distribution_version(
            module, "pyhermit" if self.reasoner == "hermit" else "pyelk-reasoner"
        )
        if package_version.split(".")[:2] != ["0", "2"]:
            raise NotImplementedError("only the shared-snapshot 0.2 backend API is qualified")
        if self.reasoner == "hermit":
            config = module.ReasonerConfig(
                backend=self.backend, timeout=self.timeout_seconds, workers=self.workers
            )
        else:
            config = module.ReasonerConfig(
                backend=self.backend,
                workers=self.workers,
                unsupported="error",
                allow_incomplete_imports=False,
            )
        reasoner = module.Reasoner(snapshot, config=config)
        try:
            if reasoner.ontology is not snapshot:
                raise RuntimeError("reasoner did not retain the shared ontology view")
            if (
                self.reasoner == "hermit"
                and "full_reasoner" not in reasoner.backend.complete_features
            ):
                raise NotImplementedError("the selected HermiT backend is not a complete reasoner")
            return package_version, reasoner
        except Exception:
            self._close(reasoner)
            raise

    def _close(self, reasoner: Any) -> None:
        if self.reasoner == "hermit":
            reasoner.dispose()
        else:
            reasoner.close()

    def _execute(self, reasoner: Any, kind: str, query: Any) -> bool:
        if kind == "consistency":
            value = reasoner.is_consistent()
        elif kind in {"class_satisfiability", "active_satisfiability"}:
            value = reasoner.is_satisfiable(query)
        elif self.reasoner == "hermit":
            if not reasoner.supports_entailment(type(query)):
                raise NotImplementedError(f"unsupported entailment query: {type(query).__name__}")
            value = reasoner.entails(query)
        else:
            value = reasoner.is_entailed(query)
        if self.reasoner == "elk":
            value = value.require_complete()
        if type(value) is not bool:
            raise TypeError("reasoner query did not return a complete boolean")
        return value

    def inspect_support(
        self,
        snapshot: owl.OntologyView,
        *,
        queries: Iterable[owl.AxiomNode] = (),
        expressions: Iterable[owl.ClassExpression] = (),
    ) -> SupportReport:
        """Qualify support through actual consistency and requested query calls."""
        return self.check_theory(snapshot, (), required=queries, activated=expressions).support

    def check_theory(
        self,
        snapshot: owl.OntologyView,
        monitored_classes: Iterable[str | owl.Class] | None = None,
        *,
        required: Iterable[owl.AxiomNode] = (),
        prohibited: Iterable[owl.AxiomNode] = (),
        activated: Iterable[owl.ClassExpression] = (),
        exceptions: Iterable[str | owl.Class] = (),
    ) -> CheckReport:
        """Check consistency and every frozen policy query on one current theory.

        Passing a frozen signature retains classes that disappear after deletion.
        ``None`` monitors all named classes currently in the asserted signature.
        Only selected candidates supply ``activated`` antecedent expressions.
        """
        snapshot = owl.coerce_snapshot(snapshot)
        monitored = named_classes(snapshot) if monitored_classes is None else monitored_classes
        exempt = {_class(value) for value in exceptions}
        classes = sorted(
            {_class(value) for value in monitored} - exempt - {owl.OWL_NOTHING},
            key=lambda value: value.iri.value,
        )
        queries: list[tuple[str, Any, bool]] = [("consistency", None, True)]
        queries.extend(("class_satisfiability", value, True) for value in classes)
        queries.extend(("required_entailment", value, True) for value in required)
        queries.extend(("prohibited_entailment", value, False) for value in prohibited)
        queries.extend(
            ("active_satisfiability", value, True)
            for value in sorted(set(activated), key=lambda value: value.canonical_bytes())
        )
        constructs = tuple(
            sorted(
                {
                    type(node).__name__
                    for axiom in snapshot.iter_axioms()
                    for node in owl.walk(axiom)
                }
            )
        )
        support = SupportReport(
            self.reasoner,
            "unavailable",
            self.backend,
            "unknown",
            False,
            snapshot.is_complete,
            constructs,
            tuple(sorted({kind for kind, _, _ in queries})),
        )
        obligations: list[ObligationResult] = []
        reasoner = None
        failure: str | None = None
        try:
            if not snapshot.is_complete:
                raise NotImplementedError("incomplete import closure")
            package_version, reasoner = self._open(snapshot)
            support = replace(
                support,
                package_version=package_version,
                backend=str(reasoner.backend.name),
                implementation_version=str(reasoner.backend.implementation_version),
                input_supported=True,
            )
            for kind, query, expected in queries:
                result = ObligationResult(
                    kind,
                    _query_id(query),
                    None,
                    expected,
                    class_iri=query.iri.value if kind == "class_satisfiability" else None,
                )
                # On an inconsistent theory there is no model for any expression
                # and every logical axiom follows. Avoid calling APIs that throw.
                inconsistent = bool(obligations and obligations[0].verdict is False)
                try:
                    if (
                        inconsistent
                        and kind in {"required_entailment", "prohibited_entailment"}
                        and not isinstance(query, owl.LOGICAL_AXIOM_TYPES)
                    ):
                        raise NotImplementedError("nonlogical consequence query")
                    value = (
                        kind in {"required_entailment", "prohibited_entailment"}
                        if inconsistent
                        else self._execute(reasoner, kind, query)
                    )
                    result = replace(result, verdict=value, complete=True)
                except Exception as error:
                    result = replace(result, reason=_failure(error))
                obligations.append(result)
            support = replace(support, diagnostics=tuple(sorted(reasoner.diagnostics().items())))
        except Exception as error:
            failure = _failure(error)
        finally:
            if reasoner is not None:
                try:
                    self._close(reasoner)
                except Exception as error:
                    failure = _failure(error)
        # A constructor or resource failure cannot leave unrecorded obligations.
        for kind, query, expected in queries[len(obligations) :]:
            obligations.append(
                ObligationResult(
                    kind,
                    _query_id(query),
                    None,
                    expected,
                    reason=failure,
                    class_iri=query.iri.value if kind == "class_satisfiability" else None,
                )
            )
        issues = tuple(
            dict.fromkeys(
                (
                    *([failure] if failure else []),
                    *(item.reason for item in obligations if item.reason),
                )
            )
        )
        support = replace(
            support,
            issues=issues,
            query_support=tuple((item.kind, item.query_id, item.complete) for item in obligations),
        )
        if not obligations[0].complete:
            support = replace(support, input_supported=False)
        if support.input_supported and any(item.satisfied is False for item in obligations):
            status = "VERIFIED_INFEASIBLE"
        elif support.input_supported and all(item.satisfied is True for item in obligations):
            status = "VERIFIED_FEASIBLE"
        else:
            status = "UNKNOWN"
        scope = (
            "complete_supported_fragment"
            if support.input_supported and all(item.complete for item in obligations)
            else "partial_detection"
        )
        return CheckReport(
            status, scope, snapshot.logical_fingerprint.hex, tuple(obligations), support
        )

    def check_assignment(
        self,
        fixed_axioms: Iterable[owl.AxiomNode],
        selected: Iterable[Any],
        monitored_classes: Iterable[str | owl.Class],
        **policy: Any,
    ) -> CheckReport:
        """Reconstruct complete replacements, retaining all remaining emitters."""
        candidates = tuple(selected)
        axioms = (*fixed_axioms, *(axiom for item in candidates for axiom in item.axioms))
        activated = tuple(
            expression for item in candidates for expression in item.active_expressions
        )
        return self.check_theory(
            snapshot_from_axioms(axioms), monitored_classes, activated=activated, **policy
        )

    def diagnose_baselines(
        self,
        source: owl.OntologyView,
        target: owl.OntologyView,
        mapping_axioms: Iterable[owl.AxiomNode],
        *,
        allow_source_exceptions: bool = False,
        **policy: Any,
    ) -> BaselineDiagnosis:
        """Check source, target, their union and the original full alignment."""
        source, target = owl.coerce_snapshot(source), owl.coerce_snapshot(target)
        union = (
            source
            if source is target
            else owl.compose_views(source, target, roles=("source", "target"))
        )
        mappings = tuple(mapping_axioms)
        aligned = owl.apply_delta(
            union,
            owl.OntologyDelta(
                add_axioms=owl.CanonicalSet((*mappings, *_declarations(mappings))),
                policy=owl.DeltaPolicy.IDEMPOTENT,
            ),
        )
        source_report = self.check_theory(source)
        target_report = self.check_theory(target)
        evidence = tuple(
            (item.class_iri, side, report.theory_hash, item.query_id)
            for side, report in (("source", source_report), ("target", target_report))
            if allow_source_exceptions and report.support.input_supported
            for item in report.obligations
            if item.kind == "class_satisfiability"
            and item.complete
            and item.verdict is False
            and item.class_iri is not None
        )
        exceptions = tuple(sorted({item[0] for item in evidence}))
        return BaselineDiagnosis(
            source_report,
            target_report,
            self.check_theory(union, exceptions=exceptions),
            self.check_theory(aligned, exceptions=exceptions, **policy),
            exceptions,
            evidence,
        )
