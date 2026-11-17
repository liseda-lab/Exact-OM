"""Explicit offline preparation of immutable roots and native class taxonomy.

The only graph algorithm here finds navigation entry components in already
recorded edges. OWL entailment is exclusively the installed native HermiT adapter.
"""

from __future__ import annotations

import base64
import dataclasses
import importlib.metadata
import os
import resource
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from contextlib import closing
from pathlib import Path
from threading import Timer

from .artifacts import atomic_json
from .context_semantics import KINDS, _allowed, _json, _ref, decode_payload, typed_node
from .contracts import canonical_hash, file_hash
from .navigation import BASES, NAVIGATION_CONTRACT, NAVIGATION_FILE

_SQL = """
CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
CREATE TABLE entries(basis TEXT NOT NULL,kind TEXT NOT NULL,ordinal INTEGER NOT NULL,
 iri TEXT NOT NULL,entry_kind TEXT NOT NULL,component_size INTEGER NOT NULL,
 PRIMARY KEY(basis,kind,ordinal));
CREATE TABLE edges(id TEXT PRIMARY KEY,child TEXT NOT NULL,parent TEXT NOT NULL,
 kind TEXT NOT NULL,payload TEXT NOT NULL);
CREATE INDEX edge_child ON edges(child,kind,id);
CREATE INDEX edge_parent ON edges(parent,kind,id);
CREATE TABLE members(iri TEXT NOT NULL,kind TEXT NOT NULL,group_id TEXT NOT NULL,
 is_top INTEGER NOT NULL,is_bottom INTEGER NOT NULL,PRIMARY KEY(iri,kind));
CREATE INDEX member_group ON members(group_id,iri);
"""
_TOP = "http://www.w3.org/2002/07/owl#Thing"
_BOTTOM = "http://www.w3.org/2002/07/owl#Nothing"


def copy_immutable(source, destination):
    """Use APFS copy-on-write when available, never a mutable hard link."""
    if sys.platform == "darwin":
        result = subprocess.run(
            ["/bin/cp", "-c", str(source), str(destination)], capture_output=True
        )
        if result.returncode == 0:
            return str(destination)
    return shutil.copyfile(source, destination)


def _root_components(entities, edges):
    """Iterative SCC condensation of recorded navigation, never OWL inference."""
    up = {iri: set() for iri in entities}
    down = {iri: set() for iri in entities}
    for child, parent in edges:
        if child in up and parent in up:
            up[child].add(parent)
            down[parent].add(child)
    seen, finished = set(), []
    for start in sorted(up):
        if start in seen:
            continue
        seen.add(start)
        stack = [(start, iter(up[start]))]
        while stack:
            node, iterator = stack[-1]
            child = next(iterator, None)
            if child is None:
                finished.append(node)
                stack.pop()
            elif child not in seen:
                seen.add(child)
                stack.append((child, iter(up[child])))
    assigned, components = {}, []
    for start in reversed(finished):
        if start in assigned:
            continue
        index = len(components)
        group, pending = [], [start]
        assigned[start] = index
        while pending:
            node = pending.pop()
            group.append(node)
            for child in down[node]:
                if child not in assigned:
                    assigned[child] = index
                    pending.append(child)
        components.append(group)
    for number, group in enumerate(components):
        if any(assigned[parent] != number for child in group for parent in up[child]):
            continue
        cyclic = len(group) > 1 or any(child in up[child] for child in group)
        yield group, cyclic, sum(len(down[iri]) for iri in group)


def _entries(
    context, db, policy, *, inferred=False, inferred_basis="reasoner_inferred", only_inferred=False
):
    with context._connection(vm_step_budget=50_000_000_000) as source:
        labels = {}

        def label(iri, kind):
            key = (iri, kind)
            if key not in labels:
                value = None
                if _allowed(policy, context.ontology_version_id, "labels"):
                    value = source.execute(
                        "SELECT CASE WHEN length(CAST(term AS BLOB))<=4096 THEN term ELSE NULL END FROM terms WHERE iri=? AND kind=? AND category='labels' ORDER BY (language='') DESC,language,fact_id LIMIT 1",
                        key,
                    ).fetchone()
                labels[key] = value[0] if value and value[0] else ""
            return labels[key]

        for basis in [inferred_basis] if only_inferred else BASES[:3]:
            if basis == inferred_basis and not inferred:
                continue
            for kind in sorted(KINDS):
                if basis == inferred_basis and kind != "class":
                    continue
                entities = [
                    r[0]
                    for r in source.execute(
                        "SELECT iri FROM entities WHERE kind=? ORDER BY iri", (kind,)
                    )
                ]
                if kind == "individual":
                    choices = [(iri, "entity_listing", 1, 0) for iri in entities]
                elif not _allowed(policy, context.ontology_version_id, "hierarchy"):
                    choices = []
                else:
                    if basis == inferred_basis:
                        entities = [
                            r[0]
                            for r in db.execute("SELECT iri FROM members WHERE kind=?", (kind,))
                        ]
                        rows = list(
                            db.execute("SELECT child,parent FROM edges WHERE kind=?", (kind,))
                        )
                        # Equivalent classes are one taxonomy node. These links
                        # are used only for root grouping, never emitted as facts.
                        groups = {}
                        for iri, group in db.execute(
                            "SELECT iri,group_id FROM members WHERE kind=?", (kind,)
                        ):
                            groups.setdefault(group, []).append(iri)
                        for members in groups.values():
                            for iri in members[1:]:
                                rows += [(members[0], iri), (iri, members[0])]
                    else:
                        categories = [
                            c[0]
                            for c in source.execute("SELECT DISTINCT category FROM axioms")
                            if _allowed(policy, context.ontology_version_id, c[0])
                        ]
                        marks = ",".join("?" for _ in categories)
                        rows = source.execute(
                            "SELECT e.child,e.parent FROM edges e JOIN axioms a ON a.id=e.axiom_id WHERE e.kind=? AND e.basis=? AND a.category IN ("
                            + marks
                            + ")",
                            (kind, basis, *categories),
                        )
                    choices = []
                    for group, cyclic, children in _root_components(entities, rows):
                        labelled = [(label(iri, kind), iri) for iri in group]
                        # A labelled real entity is a representative, not a new
                        # synthetic root or an asserted edge to owl:Thing.
                        _, representative = min(
                            labelled,
                            key=lambda item: (not bool(item[0]), item[0].casefold(), item[1]),
                        )
                        choices.append(
                            (
                                representative,
                                "component_entry" if cyclic else "named_root",
                                len(group),
                                children,
                            )
                        )

                def order(item):
                    iri, entry_kind, _, children = item
                    text = label(iri, kind)
                    priority = (
                        iri in {_TOP, _BOTTOM},
                        not bool(children),
                        entry_kind != "named_root",
                        not bool(text),
                    )
                    return (*priority, text.casefold(), iri)

                choices.sort(key=order)
                db.executemany(
                    "INSERT INTO entries VALUES (?,?,?,?,?,?)",
                    (
                        (basis, kind, ordinal, iri, entry_kind, count)
                        for ordinal, (iri, entry_kind, count, _) in enumerate(choices)
                    ),
                )


def _logical_ast(ast):
    """Compare exact core structures with nonlogical annotations omitted."""
    if isinstance(ast, dict):
        return {k: _logical_ast(v) for k, v in ast.items() if k != "annotations"}
    if isinstance(ast, list):
        return [_logical_ast(value) for value in ast]
    return ast


def _admitted_input(context, destination, policy, source_snapshot=None, *, deadline=None):
    """Use native core nodes/renderer; no local OWL parser or inferred filtering.

    A source snapshot is optional recovery input for original bytes removed by
    annotation redaction. Every recovered logical structure must exactly match
    an admitted context fact before it can enter the reasoner input.
    """
    import pyowl_core as core

    template = core.load_snapshot(
        b"Ontology()",
        options=core.LoadOptions(
            backend=core.BackendPreference.NATIVE, imports=core.ImportPolicy.IGNORE
        ),
    ).root
    template = core.OntologyDocument(
        **{
            field.name: getattr(template, field.name)
            for field in dataclasses.fields(core.OntologyDocument)
            if field.init
        }
    )
    recovery = None
    support = []
    count = 0
    with destination.open("wb") as stream, context._connection(vm_step_budget=50_000_000_000) as db:
        stream.write(b"Ontology(\n")
        batch = []

        def flush():
            if not batch:
                return
            document = dataclasses.replace(template, axioms=core.CanonicalSet(batch))
            rendered = core.render_document(document, format=core.DocumentFormat.FUNCTIONAL)
            if not rendered.startswith(b"Ontology(\n") or not rendered.endswith(b")\n"):
                raise ValueError("Unsupported native OWL renderer framing")
            stream.write(rendered[len(b"Ontology(\n") : -len(b")\n")])
            batch.clear()

        for row in db.execute(
            "SELECT id,category,payload FROM axioms WHERE category IN ('hierarchy','restrictions','types','assertions','domains','ranges','characteristics','axioms') ORDER BY id"
        ):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("Admitted input preparation deadline reached")
            if not _allowed(policy, context.ontology_version_id, row["category"]):
                continue
            payload = decode_payload(row["payload"])
            ast = payload["ast"]
            if ast["type"] in {
                "AnnotationAssertion",
                "SubAnnotationPropertyOf",
                "AnnotationPropertyDomain",
                "AnnotationPropertyRange",
            }:
                continue
            original = payload.get("original_syntax")
            if original:
                axiom = core.decode_canonical(base64.b64decode(original, validate=True))
            else:
                if source_snapshot is None:
                    raise ValueError(
                        "Logical axiom original bytes are unavailable; supply the admitted source snapshot"
                    )
                if recovery is None:
                    recovery = {
                        canonical_hash(_logical_ast(typed_node(a))): a
                        for a in source_snapshot.iter_axioms()
                    }
                axiom = recovery.get(canonical_hash(_logical_ast(ast)))
                if axiom is None:
                    raise ValueError("Source snapshot does not contain an admitted logical axiom")
                axiom = core.decode_canonical(core.canonical_bytes(axiom))
            if _logical_ast(typed_node(axiom)) != _logical_ast(ast):
                raise ValueError("Native axiom bytes differ from their admitted context structure")
            if hasattr(axiom, "annotations"):
                axiom = dataclasses.replace(axiom, annotations=core.CanonicalSet(()))
            batch.append(axiom)
            support.append(row["id"])
            count += 1
            if len(batch) == 1000:
                flush()
        # Signature declarations neither add subclass assertions nor import
        # outside facts. They permit native DL admission of undeclared terms.
        constructors = {
            "class": core.Class,
            "object_property": core.ObjectProperty,
            "data_property": core.DataProperty,
            "individual": core.NamedIndividual,
        }
        for iri, kind in db.execute("SELECT iri,kind FROM entities ORDER BY iri,kind"):
            batch.append(core.Declaration(constructors[kind](core.IRI(iri))))
            if len(batch) == 1000:
                flush()
        flush()
        stream.write(b")\n")
        stream.flush()
        os.fsync(stream.fileno())
    return {
        "input_sha256": file_hash(destination),
        "admitted_axiom_count": count,
        "admitted_fact_ids_sha256": canonical_hash(support),
        "construction": "admitted_context_axioms_without_annotations_and_typed_signature_declarations",
        "imports": "flattened_declared_scope_no_external_resolution",
    }


def _classify(
    context, db, directory, policy, *, timeout_seconds, max_memory_bytes, source_snapshot
):
    import pyowl_core as core

    from exact.ontology.reasoning import load_reasoner
    from exact.ontology.store import OwlOntologySource

    started = time.monotonic()
    deadline = started + timeout_seconds

    def remaining():
        value = deadline - time.monotonic()
        if value <= 0:
            raise TimeoutError("Total hierarchy preparation deadline reached")
        return value

    input_path = directory / "admitted.ofn"
    input_receipt = _admitted_input(context, input_path, policy, source_snapshot, deadline=deadline)
    snapshot = core.load_snapshot(
        input_path,
        options=core.LoadOptions(
            backend=core.BackendPreference.NATIVE,
            imports=core.ImportPolicy.RESOLVE_STRICT,
            limits=core.ParseLimits(
                max_memory_bytes=max_memory_bytes, deadline_seconds=remaining()
            ),
        ),
    )
    settings = {
        "backend": "native",
        "timeout_seconds": remaining(),
        "max_memory_bytes": max_memory_bytes,
        "fallback": "error",
    }
    reasoner = load_reasoner("hermit", OwlOntologySource(snapshot), settings=settings)
    timer = Timer(max(0, deadline - time.monotonic()), reasoner.shared_reasoner.interrupt)
    timer.daemon = True
    timer.start()
    try:
        native = reasoner.shared_reasoner
        remaining()
        if not native.is_consistent():
            return {
                **input_receipt,
                "status": "failed",
                "reason": "Admitted ontology is inconsistent",
                "consistency": "inconsistent",
                "completion": "not_completed",
                "reasoner": reasoner.provenance,
            }
        with context._connection() as source:
            classes = {
                row[0] for row in source.execute("SELECT iri FROM entities WHERE kind='class'")
            }
        bottom = {e.iri.value for e in native.unsatisfiable_classes()}
        top = {e.iri.value for e in native.equivalent_classes(core.Class(core.IRI(_TOP)))}
        for ordinal, iri in enumerate(sorted(classes)):
            if time.monotonic() - started > timeout_seconds:
                raise TimeoutError("Total hierarchy preparation deadline reached")
            members = sorted(
                e.iri.value
                for e in native.equivalent_classes(core.Class(core.IRI(iri)))
                if e.iri.value in classes
            )
            group = canonical_hash(
                {
                    "ontology": context.ontology_version_id,
                    "input": input_receipt["input_sha256"],
                    "members": members,
                }
            )
            db.execute(
                "INSERT INTO members VALUES (?,?,?,?,?)",
                (iri, "class", group, int(iri in top), int(iri in bottom)),
            )
            parents = set(reasoner.direct_parents(iri))
            # The shared matcher's narrow adapter intentionally hides OWL
            # bounds. Browsing retains a bound when it actually occurs in this
            # admitted signature; an implicit bound never creates a new entity.
            if classes & {_TOP, _BOTTOM}:
                parents.update(
                    member.iri.value
                    for group in native.superclasses(core.Class(core.IRI(iri)), direct=True)
                    for member in group
                    if member.iri.value in classes & {_TOP, _BOTTOM}
                )
            for parent in sorted(parents):
                if parent not in classes:
                    raise ValueError("Reasoner returned a nonadmitted named class")
                edge = {
                    "child": _ref(context.ontology_version_id, iri, "class"),
                    "parent": _ref(context.ontology_version_id, parent, "class"),
                    "basis": "reasoner_inferred",
                    "relation": "subclass",
                    "direct": True,
                    "axiom_id": None,
                    "interpretation": {
                        "kind": "reasoner_inferred",
                        "provider": "hermit",
                        "input_sha256": input_receipt["input_sha256"],
                        "premises": [],
                        "premises_unavailable": True,
                    },
                    "origins": [],
                }
                edge["id"] = canonical_hash(edge)
                db.execute(
                    "INSERT INTO edges VALUES (?,?,?,?,?)",
                    (edge["id"], iri, parent, "class", _json(edge)),
                )
            if ordinal % 500 == 0:
                db.commit()
        return {
            **input_receipt,
            "status": "available",
            "reason": None,
            "consistency": "consistent",
            "completion": "complete_for_admitted_scope",
            "supported_fragment": "native_pyhermit_OWL2_DL_admission",
            "unsupported_constructs": [],
            "reasoner": reasoner.provenance,
            "class_count": len(classes),
            "unsatisfiable_class_count": len(bottom & classes),
            "equivalence_group_count": db.execute(
                "SELECT count(DISTINCT group_id) FROM members"
            ).fetchone()[0],
            "direct_edge_count": db.execute("SELECT count(*) FROM edges").fetchone()[0],
        }
    finally:
        timer.cancel()
        reasoner.close()


def prepare_navigation(
    context,
    destination,
    *,
    policy=None,
    infer=False,
    timeout_seconds=300.0,
    max_memory_bytes=2 * 1024**3,
    source_snapshot=None,
):
    """Copy a context into a new immutable package with prepared navigation.

    The original package is never edited. Repeated calls with identical inputs
    reuse the committed result; failures have a failed/unsupported receipt and no
    partial inferred edges. A retry with changed limits requires a new destination.
    """
    from .context import OntologyContext

    policy = context._effective_policy(policy)
    if policy is not None and context._prepared_policy is None:
        raise ValueError("Prepare a physically policy-filtered context before navigation")
    if policy and not policy.allows_ontology(context.ontology_version_id):
        raise ValueError("Ontology is outside the navigation policy")
    if isinstance(timeout_seconds, bool) or not 0 < timeout_seconds <= 86400:
        raise ValueError("Hierarchy timeout must be in (0, 86400] seconds")
    if type(max_memory_bytes) is not int or max_memory_bytes <= 0:
        raise ValueError("Hierarchy memory budget must be positive")
    policy_hash = policy.policy_hash if policy else canonical_hash({"policy": "local-all"})
    versions = {name: importlib.metadata.version(name) for name in ("pyowl-core",)} if infer else {}
    if infer:
        try:
            versions["pyhermit"] = importlib.metadata.version("pyhermit")
        except importlib.metadata.PackageNotFoundError:
            versions["pyhermit"] = "unavailable"
    identity = canonical_hash(
        {
            "contract": NAVIGATION_CONTRACT,
            "context": context.manifest["artifacts"]["context.sqlite"],
            "policy": policy_hash,
            "infer": infer,
            "versions": versions,
            "timeout": timeout_seconds,
            "memory": max_memory_bytes,
        }
    )
    destination = Path(destination)
    if destination.exists():
        previous = OntologyContext(destination)
        if previous.manifest.get("navigation", {}).get("preparation_key") != identity:
            raise ValueError("Existing navigation package has different inputs")
        return previous
    destination.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix=".navigation-", dir=destination.parent) as tmp:
        staging = Path(tmp)
        copy_immutable(context.database, staging / "context.sqlite")
        inference = {
            "status": "not_run",
            "reason": "Inference was not requested",
            "completion": "not_requested",
            "consistency": "not_checked",
        }
        with closing(sqlite3.connect(staging / NAVIGATION_FILE)) as db:
            db.executescript(_SQL)
            if infer:
                try:
                    inference = _classify(
                        context,
                        db,
                        staging,
                        policy,
                        timeout_seconds=float(timeout_seconds),
                        max_memory_bytes=max_memory_bytes,
                        source_snapshot=source_snapshot,
                    )
                except Exception as error:
                    name = type(error).__name__
                    unsupported = name in {
                        "OntologyProfileError",
                        "UnsupportedDatatypeError",
                        "ReasonerUnavailableError",
                        "ModuleNotFoundError",
                    }
                    inference = {
                        "status": "unsupported" if unsupported else "failed",
                        "reason": (
                            "Native hierarchy preparation rejected the admitted input"
                            if unsupported
                            else "Native hierarchy preparation did not complete"
                        ),
                        "failure_code": name,
                        "completion": "not_completed",
                        "consistency": "not_established",
                    }
                    if (staging / "admitted.ofn").is_file():
                        inference["input_sha256"] = file_hash(staging / "admitted.ofn")
                if inference["status"] != "available":
                    db.execute("DELETE FROM edges")
                    db.execute("DELETE FROM members")
            _entries(context, db, policy, inferred=inference["status"] == "available")
            if inference["status"] != "available":
                db.execute("DELETE FROM entries WHERE basis='reasoner_inferred'")
            inference.update(
                {
                    "supported_kinds": ["class"],
                    "semantics": "direct_named_class_taxonomy_between_equivalence_groups",
                    "builtin_bounds": "bounds_in_admitted_signature_retained_otherwise_top_bottom_metadata_only",
                    "justifications": "not_exported",
                    "versions": versions,
                    "options": {
                        "backend": "native",
                        "fallback": "error",
                        "timeout_seconds": timeout_seconds,
                        "max_memory_bytes": max_memory_bytes,
                    },
                    "scope": context.manifest["scope"],
                    "source_imports_complete": context.manifest["completeness"]["imports_complete"],
                }
            )
            navigation = {
                "contract_version": NAVIGATION_CONTRACT,
                "preparation_key": identity,
                "context_sha256": context.manifest["artifacts"]["context.sqlite"],
                "ontology_version_id": context.ontology_version_id,
                "policy_hash": policy_hash,
                "source_root_sha256": context.manifest["identity"]["root_sha256"],
                "source_documents": sorted(context.manifest["identity"]["documents"]),
                "entry_points": "available",
                "order": "root_priority_label_iri",
                "inference": inference,
                "runtime": {
                    "preparation_seconds": time.monotonic() - started,
                    "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    * (1 if sys.platform == "darwin" else 1024),
                },
            }
            db.execute("INSERT INTO metadata VALUES (?,?)", ("binding", canonical_hash(navigation)))
            db.execute("ANALYZE")
            db.commit()
        (staging / "admitted.ofn").unlink(missing_ok=True)
        manifest = {
            **context.manifest,
            "navigation": navigation,
            "capabilities": {
                **context.manifest["capabilities"],
                "navigation": {
                    "contract_version": NAVIGATION_CONTRACT,
                    "entry_points": "available",
                },
                "reasoner_inferred": inference,
            },
            "artifacts": {
                "context.sqlite": context.manifest["artifacts"]["context.sqlite"],
                NAVIGATION_FILE: file_hash(staging / NAVIGATION_FILE),
            },
            "portable": {
                **context.manifest["portable"],
                "browsing": ["manifest.json", "context.sqlite", NAVIGATION_FILE],
            },
        }
        atomic_json(staging / "manifest.json", manifest)
        OntologyContext(staging)
        os.replace(staging, destination)
    return OntologyContext(destination)
