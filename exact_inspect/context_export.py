"""Publish physically policy-filtered portable contexts in fresh SQLite files.

Raw indexes remain authorized preparation inputs. The portable view contains only
permitted facts, terms and nested annotations, including its stored original syntax.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING

from .artifacts import atomic_json
from .context_prepare import _INDEX_SQL, _SQL
from .context_semantics import (
    _allowed,
    _entities,
    _json,
    _visible_node,
    decode_payload,
    encode_payload,
    expression_text,
)
from .contracts import DomainError, VisibilityPolicy, canonical_hash, file_hash

if TYPE_CHECKING:
    from .context import OntologyContext


EXPORT_SCHEMA = "exact-policy-context/1"


def export_policy_context(
    context: OntologyContext, destination: str | Path, policy: VisibilityPolicy
) -> OntologyContext:
    """Create a verified, bounded-memory view without copying forbidden SQLite pages.

    Entity and original fact identities survive filtering. A fresh context checksum
    binds the modified representation, and redacted original bytes remain unavailable.
    """
    from .context import OntologyContext

    version = context.ontology_version_id
    if not policy.allows_ontology(version):
        raise DomainError("policy_denied", "Ontology is outside the portable package policy", 403)
    previous = context.manifest.get("policy_filter")
    if previous:
        source_policy = VisibilityPolicy.model_validate(previous["policy"])
        if any(
            _allowed(policy, version, category) and not _allowed(source_policy, version, category)
            for category in policy.categories
        ):
            raise DomainError(
                "policy_unavailable", "A filtered context cannot restore omitted facts", 409
            )
    source_hash = context.manifest["artifacts"]["context.sqlite"]
    identity = canonical_hash(
        {
            "schema": EXPORT_SCHEMA,
            "source_context_sha256": source_hash,
            "policy": policy.model_dump(mode="json"),
        }
    )
    destination = Path(destination)
    if destination.exists():
        existing = OntologyContext(destination)
        if existing.manifest.get("preparation_key") != identity:
            raise DomainError(
                "export_conflict", "Existing portable context has different inputs", 409
            )
        return existing
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".policy-context-", dir=destination.parent
    ) as temporary:
        staging = Path(temporary)
        database = staging / "context.sqlite"
        counts: dict[str, int] = {}
        source_spans = False
        with context._connection(vm_step_budget=50_000_000_000) as source, closing(
            sqlite3.connect(database)
        ) as output:
            output.execute("PRAGMA synchronous=FULL")
            output.execute("PRAGMA cache_size=-262144")
            output.executescript(_SQL)
            # The entities of an allowed ontology remain navigable even when some
            # fact categories are hidden; their labels are separately policy-filtered.
            output.executemany(
                "INSERT INTO entities VALUES (?,?,?)",
                (
                    tuple(row)
                    for row in source.execute(
                        "SELECT iri,kind,eligible FROM entities ORDER BY rowid"
                    )
                ),
            )
            retained_axioms = 0
            # Physical row order reads large source packages sequentially; hashes
            # remain identities and do not require random source-page traversal.
            for row in source.execute("SELECT * FROM axioms ORDER BY rowid"):
                category = row["category"]
                if not _allowed(policy, version, category):
                    continue
                payload = decode_payload(row["payload"])
                visible = _visible_node(payload["ast"], policy, version)
                if visible != payload["ast"]:
                    payload["ast"] = visible
                    payload["qualifiers"] = visible.get("annotations", [])
                    payload["original_syntax"] = None
                    payload["original_availability"] = "not_exported"
                    payload["rendering"] = expression_text(visible)
                compact = {
                    key: value
                    for key, value in payload.items()
                    if key not in {"ast", "original_syntax", "rendering"}
                }
                compact["ast"] = {
                    key: value
                    for key, value in visible.items()
                    if key in {"type", "source", "target", "individual"}
                }
                compact_json = _json(compact)
                output.execute(
                    "INSERT INTO axioms VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        row["id"],
                        category,
                        row["predicate"],
                        row["lexical"],
                        row["datatype"],
                        row["language"],
                        encode_payload(payload),
                        len(_json(payload).encode()),
                        compact_json,
                        len(compact_json.encode()),
                        row["original_digest"],
                    ),
                )
                references = set(_entities(visible))
                if payload.get("subject_iri"):
                    references.update(
                        (payload["subject_iri"], entity[0])
                        # Resolve against the already copied local entity index,
                        # avoiding millions of random reads from network storage.
                        for entity in output.execute(
                            "SELECT kind FROM entities WHERE iri=?", (payload["subject_iri"],)
                        )
                    )
                output.executemany(
                    "INSERT OR IGNORE INTO refs VALUES (?,?,?)",
                    ((iri, kind, row["id"]) for iri, kind in references),
                )
                counts[category] = counts.get(category, 0) + 1
                source_spans = source_spans or any(
                    origin.get("span") for origin in payload.get("origins", [])
                )
                retained_axioms += 1
                if retained_axioms % 5000 == 0:
                    output.commit()
            # SQL joins copy only records whose supporting axiom survived. No old
            # database page, deleted row, checkpoint or unrestricted count is copied.
            output.execute(
                "ATTACH DATABASE ? AS original",
                (context.database.resolve().as_uri() + "?mode=ro&immutable=1",),
            )
            output.execute(
                "INSERT INTO terms SELECT t.* FROM original.terms t JOIN main.axioms a ON a.id=t.fact_id ORDER BY t.rowid"
            )
            if _allowed(policy, version, "hierarchy"):
                output.execute(
                    "INSERT INTO edges SELECT e.* FROM original.edges e JOIN main.axioms a ON a.id=e.axiom_id ORDER BY e.rowid"
                )
            output.executemany(
                "INSERT INTO metadata VALUES (?,?)",
                [
                    ("ontology_version_id", version),
                    ("preparation_key", identity),
                    ("policy_hash", policy.policy_hash),
                    ("axiom_count", str(sum(counts.values()))),
                    ("entities_ready", "1"),
                ],
            )
            output.executescript(_INDEX_SQL)
            output.execute("ANALYZE main")
            entity_count = output.execute("SELECT count(*) FROM entities").fetchone()[0]
            output.commit()
            if output.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise ValueError("Filtered context failed SQLite integrity verification")
        manifest = {
            key: context.manifest[key]
            for key in (
                "schema",
                "ontology_version_id",
                "identity",
                "name",
                "sources",
                "scope",
                "matcher_scope",
                "alignment_eligibility_bound",
                "context_extension",
                "portable",
            )
            if key in context.manifest
        }
        manifest.update(
            {
                "preparation_key": identity,
                "payload_encoding": "exact-zlib/1",
                "policy_filter": {
                    "schema": EXPORT_SCHEMA,
                    "policy": policy.model_dump(mode="json"),
                    "policy_hash": policy.policy_hash,
                    "source_context_sha256": source_hash,
                },
                "capabilities": {
                    **context.manifest["capabilities"],
                    "categories": sorted(counts),
                    "source_spans": bool(source_spans),
                },
                "completeness": {
                    "extraction": "complete_for_policy",
                    "imports_complete": context.manifest["completeness"]["imports_complete"],
                    "scope": context.manifest["scope"],
                    "domain_completeness": "not_established",
                    "entity_count": entity_count,
                    "axiom_count": sum(counts.values()),
                    "category_counts": counts,
                },
                "artifacts": {"context.sqlite": file_hash(database)},
            }
        )
        atomic_json(staging / "manifest.json", manifest)
        # Revalidate the new bytes before the package becomes discoverable.
        OntologyContext(staging)
        os.replace(staging, destination)
    return OntologyContext(destination)
