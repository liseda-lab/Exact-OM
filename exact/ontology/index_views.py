"""Shared hierarchy semantics built from the public retained type partition."""

from __future__ import annotations

import time

import pyowl_core as core
from pyowl_core.index import (
    ClassHierarchyOptions,
    IndexBuildBudget,
    ViewBuildReport,
    ViewBuildStrategy,
)

from exact.ontology.view_contract import retain_ontology_view


class TypedClassHierarchyView(core.AssertedClassHierarchyView):
    """Keep core's queries and constructor; avoid three full closure decoding passes."""

    SCHEMA_NAME = "exact/asserted-class-hierarchy/typed-partition"
    SCHEMA_VERSION = 1

    @classmethod
    def _build(
        cls,
        ontology: object,
        options: object,
        budget: IndexBuildBudget,
        cancellation_token: core.CancellationToken | None,
        started: float,
    ) -> TypedClassHierarchyView:
        """Build core's index from typed rows while retaining the original owner."""

        if not isinstance(options, ClassHierarchyOptions):
            raise TypeError("options must be ClassHierarchyOptions")
        owner = retain_ontology_view(ontology)
        index = owner.view(
            core.AxiomTypeIndex,
            scope=options.scope,
            document_key=options.document_key,
            include_origins=False,
            cancellation_token=cancellation_token,
        )
        # This is exactly the asserted-class-hierarchy schema v1 type inventory.
        rows: list[tuple[bytes, core.AxiomNode]] = []
        for kind in (core.SubClassOf, core.EquivalentClasses, core.DisjointUnion):
            for axiom in index.iter(kind):
                canonical = core.canonical_bytes(axiom)
                budget.add("asserted_records", bytes_=64 + len(canonical))
                rows.append((canonical, axiom))
        if not rows:
            budget.add("object", rows=0, bytes_=256)
        report = ViewBuildReport(
            cls.SCHEMA_NAME,
            cls.SCHEMA_VERSION,
            ViewBuildStrategy.FULL_BUILD,
            budget.rows,
            budget.shared_rows,
            budget.bytes,
            0,
            time.monotonic() - started,
            budget.tables,
        )
        return cls(
            owner,
            options,
            tuple(axiom for _, axiom in sorted(rows, key=lambda row: row[0])),
            (),
            (),
            (),
            frozenset(),
            report,
        )
