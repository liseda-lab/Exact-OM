"""Explicit offline ontology preparation; serving never imports the OWL runtime."""

from __future__ import annotations

import base64
import fcntl
import importlib.metadata
import json
import os
import resource
import shutil
import sqlite3
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .context_semantics import (
    ANNOTATION_REGISTRY,
    CATEGORIES,
    CONTEXT_SCHEMA,
    KINDS,
    _category,
    _edge_candidates,
    _entities,
    _iri,
    _json,
    _kind,
    _ref,
    expression_text,
    typed_node,
)
from .contracts import canonical_hash, file_hash

if TYPE_CHECKING:
    from .context import OntologyContext

_SQL = """
CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE entities(iri TEXT NOT NULL,kind TEXT NOT NULL,eligible INTEGER NOT NULL,
 PRIMARY KEY(iri,kind));
CREATE TABLE terms(iri TEXT NOT NULL,kind TEXT NOT NULL,term TEXT NOT NULL,
 normalized TEXT NOT NULL,language TEXT NOT NULL,category TEXT NOT NULL,fact_id TEXT NOT NULL);
CREATE TABLE axioms(id TEXT PRIMARY KEY, category TEXT NOT NULL, predicate TEXT,
 lexical TEXT, datatype TEXT, language TEXT,payload TEXT NOT NULL,
 payload_bytes INTEGER NOT NULL,fact TEXT NOT NULL,fact_bytes INTEGER NOT NULL,original_digest TEXT NOT NULL);
CREATE TABLE refs(iri TEXT NOT NULL,kind TEXT NOT NULL,axiom_id TEXT NOT NULL,
 PRIMARY KEY(iri,kind,axiom_id));
CREATE TABLE edges(id TEXT PRIMARY KEY,child TEXT NOT NULL,parent TEXT NOT NULL,
 kind TEXT NOT NULL,basis TEXT NOT NULL,axiom_id TEXT NOT NULL,payload TEXT NOT NULL);
"""
_INDEX_SQL = """
CREATE INDEX entity_kind ON entities(kind,iri);
CREATE INDEX term_search ON terms(normalized,iri,kind);
CREATE INDEX term_entity ON terms(iri,kind,language,category);
CREATE INDEX axiom_original ON axioms(original_digest,id);
CREATE INDEX ref_axiom ON refs(axiom_id,iri,kind);
CREATE INDEX edge_child ON edges(child,kind,basis,id);
CREATE INDEX edge_parent ON edges(parent,kind,basis,id);
"""


def build_context_package(
    snapshot_or_source: Any,
    destination: str | Path,
    *,
    scope: str = "root",
    matcher_scope: str | None = None,
    alignment_eligible: set[tuple[str, str]] | None = None,
    ontology_name: str | None = None,
    source_derivation: Mapping[str, Any] | None = None,
    checkpoint_interval: int = 10000,
    cache_mib: int = 256,
    work_directory: str | Path | None = None,
    durable_interval: int = 500000,
    stop_requested: Any = None,
) -> "OntologyContext":
    """Freeze a reusable context from an already loaded supported OWL snapshot.

    Browsing requires only manifest.json and context.sqlite. Regeneration requires
    the exact source documents and recorded parser/import options. Existing packages
    are never silently replaced or merged across versions.
    """
    import pyowl_core as core

    from .context import OntologyContext

    if scope not in {"root", "closure"} or matcher_scope not in {None, "root", "closure"}:
        raise ValueError("scope and matcher_scope must be root or closure")
    snapshot = core.coerce_snapshot(snapshot_or_source)
    if not isinstance(snapshot, core.OntologySnapshot):
        raise ValueError("Context preparation requires a retained source OntologySnapshot")
    if snapshot.load_options.allow_partial_rdf_mapping:
        raise ValueError("Partial RDF mapping cannot establish a faithful OWL context package")
    if checkpoint_interval < 1:
        raise ValueError("checkpoint_interval must be positive")
    if durable_interval < checkpoint_interval or durable_interval % checkpoint_interval:
        raise ValueError("durable_interval must be a positive multiple of checkpoint_interval")
    if isinstance(cache_mib, bool) or not isinstance(cache_mib, int) or not 1 <= cache_mib <= 8192:
        raise ValueError("SQLite cache must be between 1 and 8192 MiB")
    started = time.monotonic()
    axiom_scope = core.AxiomScope.ROOT if scope == "root" else core.AxiomScope.CLOSURE
    selected_documents = [
        record
        for record in snapshot.import_manifest.documents
        if scope == "closure" or record.document_key == snapshot.root_document_key
    ]
    document_records = {record.document_key: typed_node(record) for record in selected_documents}
    sources = [
        {
            "document_key": record.document_key,
            "source_sha256": "sha256:" + record.source_sha256.hex(),
            "ontology_id": typed_node(record.ontology_id),
            "format": record.format.value,
        }
        for record in selected_documents
    ]
    options = typed_node(snapshot.load_options)
    identity = {
        "schema": CONTEXT_SCHEMA,
        "scope": scope,
        "documents": sorted((source["source_sha256"] for source in sources)),
        "root_sha256": next(
            source["source_sha256"]
            for source in sources
            if source["document_key"] == snapshot.root_document_key
        ),
        "import_policy": snapshot.import_manifest.policy.value,
        "imports": [
            {
                "import_iri": edge.import_iri.value,
                "status": edge.status.value,
                "source_sha256": next(
                    (
                        source["source_sha256"]
                        for source in sources
                        if source["document_key"] == edge.resolved_document_key
                    ),
                    None,
                ),
            }
            for edge in snapshot.import_manifest.edges
        ],
        "parser": {
            "package": "pyowl-core",
            "version": importlib.metadata.version("pyowl-core"),
            "backend": snapshot.capabilities.backend,
            "options": options,
        },
        "structural_fingerprint": snapshot.structural_fingerprint.hex,
        "logical_fingerprint": snapshot.logical_fingerprint.hex,
    }
    version = canonical_hash(identity)
    preparation_inputs: dict[str, Any] = {
        "ontology_version_id": version,
        "matcher_scope": matcher_scope,
        "alignment_eligible": sorted(alignment_eligible or ()),
        "source_derivation": source_derivation,
        "ontology_name": ontology_name,
    }
    # Retain the exact existing identity for unbound builds/checkpoints. Explicit
    # empty sets must differ from absent eligibility input for all future builds.
    if alignment_eligible is not None:
        preparation_inputs["alignment_eligibility_bound"] = True
    preparation_key = canonical_hash(preparation_inputs)
    destination = Path(destination)
    if destination.exists():
        existing = OntologyContext(destination)
        if existing.ontology_version_id != version:
            raise ValueError("Existing context belongs to a different ontology version")
        if existing.manifest.get("preparation_key") != preparation_key:
            raise ValueError("Existing context has incompatible preparation coverage")
        return existing
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = (destination.parent / ("." + destination.name + ".lock")).open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise ValueError("Another writer owns this context preparation") from None
    staging = destination.parent / ("." + destination.name + ".building")
    staging.mkdir(exist_ok=True)
    persistent_database = staging / "context.sqlite"
    workspace = None
    if work_directory is not None:
        Path(work_directory).mkdir(parents=True, exist_ok=True)
        workspace = Path(tempfile.mkdtemp(prefix="context-", dir=work_directory))
        database = workspace / "context.sqlite"
        if persistent_database.exists():
            # Recover a hot rollback journal before copying a interrupted direct build.
            # Copying only database pages would otherwise omit its rollback state.
            if persistent_database.with_name(persistent_database.name + "-journal").exists():
                recovery = sqlite3.connect(persistent_database)
                try:
                    recovery.execute("SELECT count(*) FROM sqlite_schema").fetchone()
                finally:
                    recovery.close()
            shutil.copyfile(persistent_database, database)
    else:
        database = persistent_database
    fresh = not database.exists()
    connection = sqlite3.connect(database)
    healthy = False
    durable_count = 0
    persisted_stat = None

    def persist() -> None:
        """Publish a committed local database atomically to the durable staging path."""
        nonlocal durable_count, persisted_stat
        if workspace is not None:
            stat = database.stat()
            signature = (stat.st_size, stat.st_mtime_ns)
            if signature == persisted_stat:
                return
            temporary = staging / ".checkpoint.sqlite"
            shutil.copyfile(database, temporary)
            with temporary.open("rb") as copied:
                os.fsync(copied.fileno())
            if temporary.stat().st_size != database.stat().st_size:
                raise ValueError("Incomplete durable context checkpoint copy")
            os.replace(temporary, persistent_database)
            persisted_stat = signature
        durable_count = int(
            connection.execute("SELECT value FROM metadata WHERE key='axiom_count'").fetchone()[0]
        )

    try:
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute(f"PRAGMA cache_size={-1024 * cache_mib}")
        connection.execute("PRAGMA temp_store=MEMORY")
        if fresh:
            connection.executescript(_SQL)
            connection.executemany(
                "INSERT INTO metadata VALUES (?,?)",
                [
                    ("ontology_version_id", version),
                    ("preparation_key", preparation_key),
                    ("axiom_count", "0"),
                    ("source_span_count", "0"),
                    ("last_axiom_digest", ""),
                    ("entities_ready", "0"),
                ],
            )
            connection.commit()
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        if metadata.get("preparation_key") != preparation_key:
            raise ValueError("Checkpoint has incompatible inputs or preparation coverage")
        if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("Context checkpoint failed SQLite integrity verification")
        healthy = True
        durable_count = int(metadata["axiom_count"])
        # Secondary lookup indexes are constructed once after streaming all records.
        # Maintaining random-key indexes per chunk is prohibitively expensive on NFS.
        for statement in _INDEX_SQL.strip().split(";"):
            if statement.strip():
                connection.execute("DROP INDEX IF EXISTS " + statement.split()[2])
        connection.commit()
        prior_count = int(metadata["axiom_count"])
        attempt_path = staging / "attempts.jsonl"
        with attempt_path.open("a", encoding="utf-8") as attempts:
            attempts.write(
                _json(
                    {
                        "started_at": time.time(),
                        "pid": os.getpid(),
                        "resume_axiom_count": prior_count,
                    }
                )
                + "\n"
            )
        entities = (
            snapshot.signature(scope=axiom_scope, include_builtins=False)
            if metadata["entities_ready"] != "1"
            else ()
        )
        for entity in entities:
            kind = "individual" if entity.kind.value == "named_individual" else entity.kind.value
            if kind in KINDS:
                connection.execute(
                    "INSERT OR IGNORE INTO entities VALUES (?,?,?)",
                    (
                        entity.iri.value,
                        kind,
                        int(
                            alignment_eligible is not None
                            and (entity.iri.value, kind) in alignment_eligible
                        ),
                    ),
                )
        connection.execute("UPDATE metadata SET value='1' WHERE key='entities_ready'")
        connection.commit()
        if not persistent_database.exists():
            persist()
        axiom_count = prior_count
        source_span_count = int(metadata["source_span_count"])
        last_digest = metadata["last_axiom_digest"]
        chunk_started = time.monotonic()
        checkpoint_path = staging / "checkpoints.jsonl"

        def checkpoint(*, force: bool = False) -> None:
            nonlocal chunk_started
            commit_started = time.monotonic()
            connection.executemany(
                "UPDATE metadata SET value=? WHERE key=?",
                [
                    (str(axiom_count), "axiom_count"),
                    (str(source_span_count), "source_span_count"),
                    (last_digest, "last_axiom_digest"),
                ],
            )
            connection.commit()
            local_commit_seconds = time.monotonic() - commit_started
            if workspace is None or force or axiom_count % durable_interval == 0:
                persist()
            with checkpoint_path.open("a", encoding="utf-8") as metrics:
                metrics.write(
                    _json(
                        {
                            "axiom_count": axiom_count,
                            "chunk_seconds": time.monotonic() - chunk_started,
                            "commit_seconds": local_commit_seconds,
                            "persistence_seconds": time.monotonic()
                            - commit_started
                            - local_commit_seconds,
                            "durable_axiom_count": durable_count,
                            "workspace": "local" if workspace is not None else "durable",
                            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                            * 1024,
                        }
                    )
                    + "\n"
                )
            chunk_started = time.monotonic()

        origins_index = snapshot.origin_index
        for ordinal, axiom in enumerate(snapshot.iter_axioms(scope=axiom_scope)):
            if ordinal < prior_count:
                if (
                    ordinal == prior_count - 1
                    and core.structural_digest(axiom).hex() != last_digest
                ):
                    raise ValueError("Checkpoint axiom ordering differs from the pinned snapshot")
                continue
            if stop_requested is not None and stop_requested():
                checkpoint(force=True)
                raise InterruptedError(f"Context preparation stopped; resume at {destination}")
            ast = typed_node(axiom)
            category = _category(ast)
            axiom_id = canonical_hash(
                {"ontology_version_id": version, "basis": "asserted", "axiom": ast}
            )
            original_digest = core.structural_digest(axiom)
            origins = []
            for origin in origins_index.entries.get(original_digest, ()):
                if origin.document_key in document_records:
                    item = typed_node(origin)
                    item["source_sha256"] = next(
                        source["source_sha256"]
                        for source in sources
                        if source["document_key"] == origin.document_key
                    )
                    origins.append(item)
                    source_span_count += int(origin.span is not None)
            if not origins:
                # Origins can be disabled in the loader. Membership remains exact.
                for record in selected_documents:
                    if snapshot.contains(
                        axiom, scope=core.AxiomScope.DOCUMENT, document_key=record.document_key
                    ):
                        origins.append(
                            {
                                "document_key": record.document_key,
                                "source_sha256": "sha256:" + record.source_sha256.hex(),
                                "span": None,
                                "status": "occurrence_unavailable",
                            }
                        )
            payload = {
                "id": axiom_id,
                "fact_id": axiom_id,
                "axiom_id": axiom_id,
                "ontology_version_id": version,
                "category": category,
                "interpretation": {"kind": "asserted", "scope": scope},
                "availability": "available",
                "ast": ast,
                "origins": origins,
                "original_axiom_digest": "sha256:" + original_digest.hex(),
                "original_syntax": base64.b64encode(core.canonical_bytes(axiom)).decode("ascii"),
                "original_format": "pyowl-core/canonical-bytes;model=2",
                "rendering": expression_text(ast),
            }
            refs = set(_entities(ast))
            predicate = lexical = datatype = language = None
            if ast["type"] == "AnnotationAssertion":
                predicate = _iri(ast["property"])
                value = ast["value"]
                subject_iri = _iri(ast["subject"])
                payload.update(
                    {
                        "predicate": predicate,
                        "value": value,
                        "qualifiers": ast["annotations"],
                        "synonym_scope": ANNOTATION_REGISTRY.get(predicate or "", (None, None))[1],
                    }
                )
                if subject_iri:
                    refs.update(
                        (subject_iri, row[0])
                        for row in connection.execute(
                            "SELECT kind FROM entities WHERE iri=?", (subject_iri,)
                        )
                    )
                    payload["subject_iri"] = subject_iri
                if value.get("type") == "Literal":
                    lexical = value["lexical_form"]
                    datatype = _iri(value["datatype"])
                    language = value.get("language") or ""
                    if category in {"labels", "synonyms"}:
                        term_refs = set(refs)
                        if subject_iri and not any(iri == subject_iri for iri, _ in term_refs):
                            term_refs.add((subject_iri, "annotation_property"))
                        for iri, kind in term_refs:
                            if iri == subject_iri:
                                connection.execute(
                                    "INSERT INTO terms VALUES (?,?,?,?,?,?,?)",
                                    (
                                        iri,
                                        kind,
                                        lexical,
                                        lexical.casefold(),
                                        language,
                                        category,
                                        axiom_id,
                                    ),
                                )
            compact = {
                key: value
                for key, value in payload.items()
                if key not in {"ast", "original_syntax", "rendering"}
            }
            compact["ast"] = {
                key: value
                for key, value in ast.items()
                if key in {"type", "source", "target", "individual"}
            }
            payload_json, compact_json = _json(payload), _json(compact)
            connection.execute(
                "INSERT OR IGNORE INTO axioms VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    axiom_id,
                    category,
                    predicate,
                    lexical,
                    datatype,
                    language,
                    payload_json,
                    len(payload_json.encode("utf-8")),
                    compact_json,
                    len(compact_json.encode("utf-8")),
                    payload["original_axiom_digest"],
                ),
            )
            for iri, kind in refs:
                connection.execute(
                    "INSERT OR IGNORE INTO refs VALUES (?,?,?)", (iri, kind, axiom_id)
                )
            for child, parent, basis, relation, rule in _edge_candidates(ast):
                child_iri, parent_iri, entity_kind = _iri(child), _iri(parent), _kind(child)
                assert child_iri is not None and parent_iri is not None and entity_kind is not None
                edge = {
                    "child": _ref(version, child_iri, entity_kind),
                    "parent": _ref(version, parent_iri, entity_kind),
                    "basis": basis,
                    "relation": relation,
                    "axiom_id": axiom_id,
                    "category": "hierarchy",
                    "interpretation": {
                        "kind": "asserted" if rule is None else "structurally_derived",
                        "rule": rule,
                        "provider": "exact-context",
                        "version": 1,
                        "premises": [axiom_id],
                        "scope": scope,
                    },
                    "origins": origins,
                }
                edge["id"] = canonical_hash(
                    {
                        "version": version,
                        "child": child_iri,
                        "parent": parent_iri,
                        "kind": entity_kind,
                        "basis": basis,
                        "relation": relation,
                        "axiom": axiom_id,
                    }
                )
                connection.execute(
                    "INSERT OR IGNORE INTO edges VALUES (?,?,?,?,?,?,?)",
                    (edge["id"], child_iri, parent_iri, entity_kind, basis, axiom_id, _json(edge)),
                )
            axiom_count += 1
            last_digest = payload["original_axiom_digest"].removeprefix("sha256:")
            if axiom_count % checkpoint_interval == 0:
                checkpoint()
        checkpoint(force=True)
        connection.executescript(_INDEX_SQL)
        connection.execute("ANALYZE")
        connection.commit()
        persist()
        entity_count = connection.execute("SELECT count(*) FROM entities").fetchone()[0]
        connection.close()
        manifest = {
            "schema": CONTEXT_SCHEMA,
            "preparation_key": preparation_key,
            "ontology_version_id": version,
            "identity": identity,
            "source_derivation": dict(source_derivation) if source_derivation else None,
            "name": ontology_name,
            "sources": sources,
            "scope": scope,
            "matcher_scope": matcher_scope,
            "alignment_eligibility_bound": alignment_eligible is not None,
            "context_extension": matcher_scope is not None and matcher_scope != scope,
            "capabilities": {
                "provider": "owl",
                "typed_expressions": True,
                "categories": list(CATEGORIES),
                "kinds": sorted(KINDS),
                "source_spans": source_span_count > 0,
                "reasoner_inferred": {
                    "status": "not_run",
                    "reason": "Optional inference was not requested",
                },
            },
            "completeness": {
                "extraction": "complete",
                "imports_complete": snapshot.is_complete,
                "scope": scope,
                "domain_completeness": "not_established",
                "entity_count": entity_count,
                "axiom_count": axiom_count,
            },
            "diagnostics": typed_node(snapshot.diagnostics),
            "runtime": {
                "preparation_seconds": time.monotonic() - started,
                "sqlite_cache_mib": cache_mib,
                "local_workspace": workspace is not None,
                "durable_interval": (
                    durable_interval if workspace is not None else checkpoint_interval
                ),
                "resumed_axiom_count": prior_count,
                "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                "checkpoint_metrics": [
                    json.loads(line) for line in checkpoint_path.read_text().splitlines()
                ],
                "attempts": [json.loads(line) for line in attempt_path.read_text().splitlines()],
                "parser_timings": typed_node(snapshot.timings),
            },
            "portable": {
                "browsing": ["manifest.json", "context.sqlite"],
                "regeneration": "Exact source bytes for the declared documents and parser/import options",
            },
            "artifacts": {"context.sqlite": file_hash(persistent_database)},
        }
        (staging / "manifest.json").write_text(_json(manifest) + "\n", encoding="utf-8")
        os.replace(staging, destination)
    except BaseException:
        if healthy and workspace is not None:
            try:
                connection.rollback()
                persist()
            except sqlite3.ProgrammingError:
                pass  # Connection was already closed after the final durable copy.
        connection.close()
        # Committed durable chunks survive even when the optional local workspace is lost.
        raise
    finally:
        lock.close()
        if workspace is not None:
            shutil.rmtree(workspace, ignore_errors=True)
    return OntologyContext(destination)


def prepare_context(
    path: str | Path,
    destination: str | Path,
    *,
    imports: Mapping[str, Any] | None = None,
    scope: str = "root",
    expected_hash: str | None = None,
    matcher_scope: str | None = None,
    ontology_name: str | None = None,
    source_derivation: Mapping[str, Any] | None = None,
    cache_mib: int = 256,
    work_directory: str | Path | None = None,
) -> OntologyContext:
    """Explicit offline native preparation, with pinned root and local import receipts.

    Import values may be local paths or {path, sha256} records. The latter verify
    an existing lock; plain paths freeze the bytes acquired by this preparation.
    No parser or resolver is invoked by OntologyContext reads.
    """
    import pyowl_core as core

    source = Path(path)
    actual_hash = file_hash(source)
    if expected_hash and actual_hash.removeprefix("sha256:") != expected_hash.removeprefix(
        "sha256:"
    ):
        raise ValueError("Ontology input hash does not match its pin")
    if scope not in {"root", "closure"}:
        raise ValueError("scope must be root or closure")
    pinned_imports: dict[Any, Any] = {}
    import_hashes = {}
    for iri, record in (imports or {}).items():
        import_path = Path(record["path"] if isinstance(record, Mapping) else record)
        digest = file_hash(import_path)
        if isinstance(record, Mapping) and record.get("sha256"):
            if digest.removeprefix("sha256:") != record["sha256"].removeprefix("sha256:"):
                raise ValueError("Ontology import hash does not match its pin")
        pinned_imports[iri] = import_path
        import_hashes[iri] = digest
    snapshot = core.load_snapshot(
        source,
        options=core.LoadOptions(
            backend=core.BackendPreference.NATIVE,
            preserve_source_map=True,
            offline=True,
            imports=(
                core.ImportPolicy.IGNORE if scope == "root" else core.ImportPolicy.RESOLVE_STRICT
            ),
        ),
        resolver=core.MappingResolver(pinned_imports),
    )
    if file_hash(source) != actual_hash or any(
        file_hash(pinned_imports[iri]) != digest for iri, digest in import_hashes.items()
    ):
        raise ValueError("Ontology input changed during preparation")
    return build_context_package(
        snapshot,
        destination,
        scope=scope,
        matcher_scope=matcher_scope,
        ontology_name=ontology_name,
        source_derivation=source_derivation,
        cache_mib=cache_mib,
        work_directory=work_directory,
    )
