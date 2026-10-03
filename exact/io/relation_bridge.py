"""Native OWL entailment with shared anchor worlds and resumable query records.

Only the small bridge document crosses Python. Ontology parsing, compilation,
consistency, satisfiability and entailment remain native operations.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import pandas as pd


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _save(path, value):
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def native_bridge(
    frame,
    source,
    target,
    *,
    anchor_rows,
    timeout_seconds,
    import_map=None,
    checkpoint_path=None,
    max_memory_bytes=None,
    max_compile_work=None,
    preparation_identity=None,
):
    import pyhermit
    import pyowl_core as core

    from exact.ontology import load_ontology
    from exact.ontology.reasoning import (
        ReasonerSettings,
        _create_hermit,
        reasoner_cache_identity,
    )
    from exact.ontology.store import OwlOntologySource
    from exact.utils.provenance import sha256_file

    settings = ReasonerSettings(
        backend="native",
        timeout_seconds=timeout_seconds,
        workers=1,
        max_memory_bytes=max_memory_bytes,
        max_compile_work=max_compile_work,
    )
    # A path allows diagnostic admission without loading the two roots twice.
    origins = [
        (
            Path(knowledge)
            if isinstance(knowledge, (str, Path))
            else getattr(knowledge, "origin", None)
        )
        for knowledge in (source, target)
    ]
    supported = all(
        isinstance(knowledge, (str, Path, OwlOntologySource)) for knowledge in (source, target)
    ) and all(path is not None and Path(path).is_file() for path in origins)
    paths = {str(iri): Path(path).resolve() for iri, path in (import_map or {}).items()}
    if supported:
        paths.update({Path(path).resolve().as_uri(): Path(path).resolve() for path in origins})
    anchors = sorted(
        {
            (str(row["src"]), str(row["tgt"]))
            for row in anchor_rows
            if row["src_kind"] == "class"
            and row["tgt_kind"] == "class"
            and row["src"] != row["tgt"]
        }
    )
    # Store no scores/gold in the checkpoint. Current row payload is restored at output.
    queries = [
        (
            str(row.SrcEntity),
            str(row.TgtEntity),
            str(row.get("SrcKind", "class")),
            str(row.get("TgtKind", "class")),
        )
        for _, row in frame.iterrows()
    ]
    state = {"identity": None, "queries": {}}
    checkpoint = Path(checkpoint_path) if checkpoint_path else None
    if checkpoint is not None:
        identity = _digest(
            {
                "schema": "exact/native-bridge/v2",
                "implementation": sha256_file(Path(__file__)),
                "runtime": reasoner_cache_identity("hermit", settings),
                "documents": {iri: sha256_file(path) for iri, path in sorted(paths.items())},
                "roots": [str(path) for path in origins],
                "anchors": anchors,
                "queries": queries,
                "reasoning_preparation": preparation_identity,
            }
        )
        state["identity"] = identity
        if checkpoint.exists():
            state = json.loads(checkpoint.read_text())
            if state.get("identity") != identity:
                raise ValueError("Native bridge checkpoint inputs or implementation changed")
        else:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            _save(checkpoint, state)

    def record(query, value):
        state["queries"][_digest(query)] = value
        if checkpoint is not None:
            _save(checkpoint, state)

    groups = defaultdict(list)
    for query in queries:
        if _digest(query) in state["queries"]:
            continue
        if query[2:] != ("class", "class"):
            record(query, {"reason": "unsupported_kind"})
        elif not supported:
            record(query, {"reason": "unsupported_profile"})
        else:
            # Most queries are not anchors and share exactly one reasoner.
            # Anchor queries each exclude their own equivalence axiom.
            excluded = query[:2] if query[:2] in anchors else None
            groups[excluded].append(query)
    cache_hits = 0
    compiled_worlds = 0
    failures = (
        pyhermit.OntologyProfileError,
        pyhermit.UnsupportedDatatypeError,
        pyhermit.IncompleteImportClosureError,
        pyhermit.FreshEntityError,
        core.UnresolvedImportError,
        core.UnsupportedSyntaxError,
    )

    def failure(error):
        if isinstance(error, (pyhermit.ReasonerTimeoutError, TimeoutError)):
            reason = "reasoning_timeout"
        elif isinstance(error, pyhermit.FreshEntityError):
            reason = "unknown_entity"
        else:
            reason = "unsupported_profile"
        return {"reason": reason, "detail": str(error)}

    for excluded, pending in groups.items():
        bridges = " ".join(
            f"EquivalentClasses(<{src}> <{tgt}>)" for src, tgt in anchors if (src, tgt) != excluded
        )
        imports = " ".join(f"Import(<{Path(path).resolve().as_uri()}>)" for path in origins)
        reasoner = None
        try:
            view = load_ontology(
                f"Ontology({imports} {bridges})".encode(),
                document_iri="urn:exact:relation-bridge",
                resolver=core.MappingResolver(paths),
            ).owl_snapshot()
            cache_hits += view.report.document_cache_hits
            # pyHermiT resets its timeout per operation; no frame-wide deadline.
            reasoner, _, _ = _create_hermit(view, settings)
            compiled_worlds += 1
            consistent = bool(reasoner.is_consistent())
            for query in pending:
                value = {"consistent": consistent}
                if not consistent:
                    value["reason"] = "inconsistent_bridge"
                else:
                    try:
                        left, right = (core.Class(core.IRI(iri)) for iri in query[:2])
                        satisfiable = bool(reasoner.is_satisfiable(left)) and bool(
                            reasoner.is_satisfiable(right)
                        )
                        value["endpoints_satisfiable"] = satisfiable
                        if not satisfiable:
                            value["reason"] = "unsatisfiable_endpoint"
                        else:
                            forward = bool(reasoner.entails(core.SubClassOf(left, right)))
                            reverse = bool(reasoner.entails(core.SubClassOf(right, left)))
                            value.update(forward=forward, reverse=reverse)
                            if forward or reverse:
                                value["relation"] = (
                                    "=" if forward and reverse else "<" if forward else ">"
                                )
                            else:
                                value["reason"] = "not_entailed"
                    except (TimeoutError, *failures) as error:
                        value.update(failure(error))
                record(query, value)
        except (TimeoutError, *failures) as error:
            for query in pending:
                if _digest(query) not in state["queries"]:
                    record(query, failure(error))
        finally:
            if reasoner is not None:
                reasoner.dispose()

    results, abstentions, observations = [], [], []
    for (index, row), query in zip(frame.iterrows(), queries):
        observation = state["queries"][_digest(query)]
        observations.append(observation)
        if observation.get("relation"):
            result = row.to_dict()
            result.update(
                Relation=observation["relation"],
                relation_confidence=1.0,
                relation_semantic_backend="native_hermit",
                relation_evidence=json.dumps(
                    {
                        "query_bridge_excluded": True,
                        "backend": "native_hermit",
                        **observation,
                    }
                ),
            )
            results.append((index, result))
        else:
            abstentions.append(
                {
                    "source": query[0],
                    "target": query[1],
                    "reason": observation["reason"],
                    **({"detail": observation["detail"]} if "detail" in observation else {}),
                }
            )
    output = (
        pd.DataFrame([row for _, row in results], index=[index for index, _ in results])
        if results
        else frame.iloc[:0].copy()
    )
    if not results:
        output["Relation"] = pd.Series(dtype=str)
        output["relation_confidence"] = pd.Series(dtype=float)
    output.attrs["relation_abstentions"] = abstentions
    output.attrs["relation_anchor_count"] = len(anchor_rows)
    output.attrs["coherence_audit"] = {
        "profile": "native_hermit_supported_owl",
        "query_bridge_excluded": True,
        "logical_unsatisfiability": "unknown",  # No claim about all named classes.
        "logical_reason": "queried_endpoints_checked_not_full_classification",
        "unsatisfiable_query_pairs": sum(
            o.get("endpoints_satisfiable") is False for o in observations
        ),
        "native_import_document_cache_hits": cache_hits,
        "compiled_anchor_worlds": compiled_worlds,
        "ontology_consistency": (
            "inconsistent"
            if any(o.get("consistent") is False for o in observations)
            else (
                "consistent"
                if observations and all(o.get("consistent") is True for o in observations)
                else "unknown"
            )
        ),
    }
    return output
