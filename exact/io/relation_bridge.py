"""Optional native HermiT bridge over locally bound OWL documents.

Only bridge axioms cross Python. Loading, OWL normalization and entailment use
native consumers; unsupported inputs and timeouts remain explicit unknowns.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd


def native_bridge(frame, source, target, *, anchor_rows, timeout_seconds):
    import pyhermit
    import pyowl_core as core

    from exact.ontology import load_ontology
    from exact.ontology.reasoning import ReasonerSettings, _create_hermit

    deadline = time.monotonic() + timeout_seconds
    results, abstentions, consistency_checks = [], [], []
    import_cache_hits = 0
    from exact.ontology.store import OwlOntologySource

    origins = [getattr(knowledge, "origin", None) for knowledge in (source, target)]
    if not all(isinstance(knowledge, OwlOntologySource) for knowledge in (source, target)) or any(
        path is None or not Path(path).is_file() for path in origins
    ):
        reason = "unsupported_profile"
    else:
        # Each native query snapshot imports the same public OWL documents and
        # omits the queried equivalence bridge.
        document = (
            "Ontology("
            + " ".join(f"Import(<{Path(path).resolve().as_uri()}>)" for path in origins)
            + ")"
        )
        reason = None
    for index, row in frame.iterrows():
        source_iri, target_iri = str(row.SrcEntity), str(row.TgtEntity)
        unknown = reason
        if (
            str(row.get("SrcKind", "class")) != "class"
            or str(row.get("TgtKind", "class")) != "class"
        ):
            unknown = "unsupported_kind"
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            unknown = "reasoning_timeout"
        if unknown is None:
            bridges = " ".join(
                f"EquivalentClasses(<{anchor['src']}> <{anchor['tgt']}>)"
                for anchor in anchor_rows
                if anchor["src_kind"].value == "class"
                and (anchor["src"], anchor["tgt"]) != (source_iri, target_iri)
                and anchor["src"] != anchor["tgt"]
            )
            # Current native consumers do not support overlay column ownership.
            # Parse the small bridge document natively; core's immutable document
            # cache can reuse imported documents between query snapshots.
            query_document = document[:-1] + " " + bridges + ")"
            reasoner = None
            try:
                view = load_ontology(
                    query_document.encode(),
                    document_iri="urn:exact:relation-bridge",
                    resolver=core.MappingResolver(
                        {Path(path).resolve().as_uri(): Path(path) for path in origins}
                    ),
                ).owl_snapshot()
                import_cache_hits += view.report.document_cache_hits
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("native bridge loading exceeded the remaining deadline")
                reasoner, _, _ = _create_hermit(
                    view, ReasonerSettings(backend="native", timeout_seconds=remaining, workers=1)
                )
                consistent = bool(reasoner.is_consistent())
                consistency_checks.append(consistent)
                if not consistent:
                    unknown = "inconsistent_bridge"
                else:
                    left, right = core.Class(core.IRI(source_iri)), core.Class(core.IRI(target_iri))
                    forward = bool(reasoner.entails(core.SubClassOf(left, right)))
                    reverse = bool(reasoner.entails(core.SubClassOf(right, left)))
                    if time.monotonic() > deadline:
                        raise TimeoutError("native bridge task deadline")
                    if forward or reverse:
                        result = row.to_dict()
                        result.update(
                            Relation="=" if forward and reverse else "<" if forward else ">",
                            relation_confidence=1.0,
                            relation_semantic_backend="native_hermit",
                            relation_evidence=json.dumps(
                                {
                                    "query_bridge_excluded": True,
                                    "backend": "native_hermit",
                                    "forward": forward,
                                    "reverse": reverse,
                                }
                            ),
                        )
                        results.append((index, result))
                    else:
                        unknown = "not_entailed"
            except (pyhermit.ReasonerTimeoutError, TimeoutError):
                unknown = "reasoning_timeout"
            except (pyhermit.OntologyProfileError, core.UnresolvedImportError):
                unknown = "unsupported_profile"
            finally:
                if reasoner is not None:
                    reasoner.dispose()
        if unknown is not None:
            abstentions.append({"source": source_iri, "target": target_iri, "reason": unknown})
    output = (
        pd.DataFrame([row for _, row in results], index=[index for index, _ in results])
        if results
        else frame.iloc[:0].copy()
    )
    if not results:
        output["relation_confidence"] = pd.Series(dtype=float)
    output.attrs["relation_abstentions"] = abstentions
    output.attrs["relation_anchor_count"] = len(anchor_rows)
    output.attrs["coherence_audit"] = {
        "profile": "native_hermit_supported_owl",
        "query_bridge_excluded": True,
        "logical_unsatisfiability": "unknown",
        "native_import_document_cache_hits": import_cache_hits,
        "logical_reason": "named_class_satisfiability_not_checked",
        "ontology_consistency": (
            "inconsistent"
            if False in consistency_checks
            else "consistent" if len(consistency_checks) == len(frame) else "unknown"
        ),
    }
    return output
