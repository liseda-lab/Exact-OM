"""Bounded, read-only queries over portable independent ontology context packages.

The public preparation helpers are reexported for a single integration entry point;
only explicit preparation imports pyowl and traverses source axioms.
"""

from __future__ import annotations

import json
import sqlite3
from collections import OrderedDict
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .context_prepare import build_context_package, prepare_context
from .context_semantics import (
    ANNOTATION_REGISTRY,
    CATEGORIES,
    CONTEXT_SCHEMA,
    KINDS,
    _allowed,
    _fact_view,
    _ref,
    _ref_dict,
    _visible_node,
    decode_payload,
    decoded_chunks,
    expression_text,
    typed_node,
)
from .contracts import (
    CONTRACT_VERSION,
    DomainError,
    EntityRef,
    VisibilityPolicy,
    canonical_hash,
    decode_cursor,
    encode_cursor,
    file_hash,
)
from .sqlite_safety import CONTEXT_TABLES, readonly_connection

__all__ = [
    "OntologyContext",
    "build_context_package",
    "prepare_context",
    "CONTEXT_SCHEMA",
    "ANNOTATION_REGISTRY",
    "typed_node",
    "expression_text",
]

_VERIFIED_FILES: OrderedDict[tuple[str, tuple[int, ...]], str] = OrderedDict()
_FACT_PAGE_BYTES = 64 * 1024
_AXIOM_DETAIL_BYTES = 2 * 1024 * 1024


class OntologyContext:
    """Verified context package; all public reads use independent bounded SQL queries."""

    def __init__(self, path: str | Path, *, verified_database_hash: str | None = None):
        self.path = Path(path)
        self.manifest = json.loads((self.path / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("schema") != CONTEXT_SCHEMA:
            raise ValueError("Unsupported ontology context schema")
        filtered = self.manifest.get("policy_filter")
        self._prepared_policy = (
            VisibilityPolicy.model_validate(filtered["policy"]) if filtered else None
        )
        if (
            filtered
            and self._prepared_policy
            and self._prepared_policy.policy_hash != filtered.get("policy_hash")
        ):
            raise ValueError("Ontology context visibility policy mismatch")
        self.ontology_version_id = self.manifest["ontology_version_id"]
        if canonical_hash(self.manifest["identity"]) != self.ontology_version_id:
            raise ValueError("Ontology context identity mismatch")
        self.database = self.path / "context.sqlite"
        self._verified_stat = self._database_stat()
        verification_key = (str(self.database.resolve()), self._verified_stat)
        # A containing bundle may already have verified these exact database bytes.
        digest = verified_database_hash or _VERIFIED_FILES.get(verification_key)
        if digest is None:
            digest = file_hash(self.database)
            if self._database_stat() != self._verified_stat:
                raise ValueError("Ontology index changed during verification")
            _VERIFIED_FILES[verification_key] = digest
            while len(_VERIFIED_FILES) > 128:
                _VERIFIED_FILES.popitem(last=False)
        if digest != self.manifest["artifacts"]["context.sqlite"]:
            raise ValueError("Ontology context checksum mismatch")
        with self._connection() as connection:
            if (
                connection.execute(
                    "SELECT value FROM metadata WHERE key='ontology_version_id'"
                ).fetchone()[0]
                != self.ontology_version_id
            ):
                raise ValueError("Ontology index identity mismatch")

    def _effective_policy(self, policy: Any) -> Any:
        if self._prepared_policy is None:
            return policy
        if policy is not None and policy.policy_hash != self._prepared_policy.policy_hash:
            raise DomainError(
                "context_policy_mismatch",
                "Context was prepared for a different visibility policy",
                status_code=409,
            )
        return self._prepared_policy

    def _database_stat(self) -> tuple[int, ...]:
        stat = self.database.stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

    @contextmanager
    def _connection(self, *, vm_step_budget: int = 50_000_000) -> Iterator[sqlite3.Connection]:
        if self._database_stat() != self._verified_stat:
            raise ValueError("Ontology index changed after verification")
        with readonly_connection(
            self.database, CONTEXT_TABLES, vm_step_budget=vm_step_budget
        ) as connection:
            yield connection

    def _scope(self, query: dict[str, Any], policy: Any, basis: str = "asserted") -> dict[str, Any]:
        return {
            "ontology_version_id": self.ontology_version_id,
            "context_revision": self.manifest["artifacts"]["context.sqlite"],
            "basis": "literal_asserted" if basis == "asserted" else basis,
            "filter_id": canonical_hash(query),
            "visibility_policy_hash": (
                policy.policy_hash
                if policy is not None
                else canonical_hash({"policy": "local-all"})
            ),
        }

    @staticmethod
    def _bound(limit: int, maximum: int = 100) -> None:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= maximum:
            raise ValueError(f"limit must be between 1 and {maximum}")

    def _entity(self, entity: Any, policy: Any = None) -> dict[str, Any]:
        policy = self._effective_policy(policy)
        ref = _ref_dict(entity)
        EntityRef.model_validate(ref)
        if ref["ontology_version_id"] != self.ontology_version_id:
            raise KeyError("Unknown ontology entity")
        if (
            policy is not None
            and policy.ontology_ids
            and self.ontology_version_id not in policy.ontology_ids
        ):
            raise KeyError("Unknown ontology entity")
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM entities WHERE iri=? AND kind=?", (ref["iri"], ref["kind"])
            ).fetchone()
        if row is None:
            raise KeyError("Unknown ontology entity")
        return ref

    def _page(
        self,
        items: list[Any],
        total: int | None,
        scope: dict[str, Any],
        *,
        next_key: Any = None,
        status: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        return {
            "items": items,
            "returned_count": len(items),
            "total_count": total,
            "next_cursor": encode_cursor(scope, next_key) if next_key is not None else None,
            "truncated": next_key is not None,
            "scope": scope,
            "status": status or ("available" if items else "absent_in_scope"),
            "reason": reason,
        }

    def search(
        self,
        *,
        kind: str | None = None,
        term: str = "",
        language: str | None = None,
        limit: int = 20,
        cursor: str | None = None,
        policy: Any = None,
    ) -> dict[str, Any]:
        """Search exact IRIs or normalized term prefixes, retaining distinct identities."""
        policy = self._effective_policy(policy)
        self._bound(limit)
        if kind is not None and kind not in KINDS:
            raise ValueError("Unsupported entity kind")
        if len(term) > 512:
            raise ValueError("Search term exceeds 512 characters")
        scope = self._scope(
            {"operation": "search", "kind": kind, "term": term, "language": language}, policy
        )
        if (
            policy is not None
            and policy.ontology_ids
            and self.ontology_version_id not in policy.ontology_ids
        ):
            return self._page([], 0, scope)
        last = decode_cursor(cursor, scope) if cursor else ["", ""]
        if (
            not isinstance(last, list)
            or len(last) != 2
            or not all(isinstance(key, str) for key in last)
        ):
            raise DomainError("invalid_cursor", "Invalid search cursor")
        predicates, args = [], []
        if kind:
            predicates.append("e.kind=?")
            args.append(kind)
        if term:
            allowed = [
                category
                for category in ("labels", "synonyms")
                if _allowed(policy, self.ontology_version_id, category)
            ]
            sql = "SELECT iri,kind FROM entities WHERE iri=?"
            term_args: list[Any] = [term]
            if allowed:
                sql += (
                    " UNION SELECT t.iri,t.kind FROM terms t WHERE t.normalized>=? AND t.normalized<? AND t.category IN ("
                    + ",".join("?" for _ in allowed)
                    + ")"
                )
                term_args += [term.casefold(), term.casefold() + "\U0010ffff", *allowed]
                if language is not None:
                    sql += " AND t.language=?"
                    term_args.append(language)
            predicates.append("(e.iri,e.kind) IN (" + sql + ")")
            args.extend(term_args)
        where = " AND ".join(predicates) or "1"
        with self._connection() as connection:
            total = connection.execute(
                "SELECT count(*) FROM entities e WHERE " + where, args
            ).fetchone()[0]
            rows = connection.execute(
                "SELECT e.* FROM entities e WHERE "
                + where
                + " AND (e.iri,e.kind)>(?,?) ORDER BY e.iri,e.kind LIMIT ?",
                [*args, *last, limit + 1],
            ).fetchall()
        more = len(rows) > limit
        items = []
        for row in rows[:limit]:
            ref = _ref(self.ontology_version_id, row["iri"], row["kind"])
            items.append(
                {
                    "entity": ref,
                    "preferred_label": self._label(ref, language, policy),
                    **self._alignment_eligibility(row["eligible"]),
                }
            )
        key = [rows[limit - 1]["iri"], rows[limit - 1]["kind"]] if more else None
        return self._page(items, total, scope, next_key=key)

    def _alignment_eligibility(self, stored: Any) -> dict[str, Any]:
        bound = self.manifest.get("alignment_eligibility_bound")
        if bound is True or stored:
            return {"alignment_eligible": bool(stored), "alignment_eligibility_status": "available"}
        return {
            "alignment_eligible": None,
            "alignment_eligibility_status": "not_requested" if bound is False else "not_exported",
        }

    def _label(self, ref: dict[str, Any], language: str | None, policy: Any) -> dict[str, Any]:
        if not _allowed(policy, self.ontology_version_id, "labels"):
            return {"status": "not_exported", "value": None}
        with self._connection() as connection:
            row = connection.execute(
                "SELECT CASE WHEN length(CAST(term AS BLOB))<=4096 THEN term ELSE NULL END term,language,fact_id FROM terms WHERE iri=? AND kind=? AND category='labels' ORDER BY (language=?) DESC,(language='') DESC,language,fact_id LIMIT 1",
                (ref["iri"], ref["kind"], language or ""),
            ).fetchone()
        if row is None:
            return {"status": "absent_in_scope", "value": None}
        return {
            "status": "available" if row["term"] is not None else "partial",
            "value": row["term"],
            "language": row["language"] or None,
            "fact_id": row["fact_id"],
            "requested_language": language,
            "language_fallback": language is not None and row["language"] != language,
        }

    def facts(
        self,
        entity: Any,
        *,
        category: str | None = None,
        basis: str = "asserted",
        limit: int = 20,
        cursor: str | None = None,
        policy: Any = None,
    ) -> dict[str, Any]:
        """Return complete typed fact references in deterministic keyset pages."""
        policy = self._effective_policy(policy)
        self._bound(limit)
        ref = self._entity(entity, policy)
        if category is not None and category not in CATEGORIES:
            raise ValueError("Unknown fact category")
        scope = self._scope(
            {"operation": "facts", "entity": ref, "category": category}, policy, basis
        )
        if basis not in {"asserted", "literal_asserted"}:
            return self._page(
                [], None, scope, status="not_run" if basis == "reasoner_inferred" else "unsupported"
            )
        categories = [
            item for item in CATEGORIES if _allowed(policy, self.ontology_version_id, item)
        ]
        if category and category != "usage":
            categories = [category] if category in categories else []
        if category == "usage" and not _allowed(policy, self.ontology_version_id, "usage"):
            categories = []
        if not categories:
            return self._page([], None, scope, status="not_exported")
        last = decode_cursor(cursor, scope) if cursor else ""
        if not isinstance(last, str):
            raise DomainError("invalid_cursor", "Invalid collection cursor")
        placeholders = ",".join("?" for _ in categories)
        where = f"r.iri=? AND r.kind=? AND a.category IN ({placeholders})"
        args = [ref["iri"], ref["kind"], *categories]
        with self._connection() as connection:
            total = connection.execute(
                "SELECT count(*) FROM refs r JOIN axioms a ON a.id=r.axiom_id WHERE " + where, args
            ).fetchone()[0]
            rows = connection.execute(
                "SELECT a.id,a.category,CASE WHEN a.fact_bytes<=? THEN a.fact ELSE NULL END fact FROM refs r JOIN axioms a ON a.id=r.axiom_id WHERE "
                + where
                + " AND a.id>? ORDER BY a.id LIMIT ?",
                [_FACT_PAGE_BYTES // 2, *args, last, limit + 1],
            ).fetchall()
        items: list[dict[str, Any]] = []
        page_bytes = 0
        oversized = False
        for row in rows[:limit]:
            if row["fact"] is None:
                oversized = True
                item = {
                    "id": row["id"],
                    "fact_id": row["id"],
                    "subject": ref,
                    "predicate_iri": None,
                    "value": {"term_type": "expression_ref", "expression_id": row["id"]},
                    "axiom_ref": row["id"],
                    "axiom_id": row["id"],
                    "category": row["category"],
                    "interpretation": "asserted",
                    "origins": [],
                    "availability": "partial",
                    "reason": "Fact exceeds inline byte budget; original remains in the axiom artifact",
                }
            else:
                item = _fact_view(json.loads(row["fact"]), ref, policy)
            item_bytes = len(json.dumps(item, ensure_ascii=False).encode("utf-8"))
            if items and page_bytes + item_bytes > _FACT_PAGE_BYTES:
                break
            items.append(item)
            page_bytes += item_bytes
        return self._page(
            items,
            total,
            scope,
            next_key=rows[len(items) - 1]["id"] if len(rows) > len(items) else None,
            status="partial" if oversized else None,
            reason="Oversized values require explicit axiom detail" if oversized else None,
        )

    def entity_context(
        self, entity: Any, *, language: str | None = None, policy: Any = None
    ) -> dict[str, Any]:
        """Summarize an entity independently of any matching run or scored pair."""
        policy = self._effective_policy(policy)
        ref = self._entity(entity, policy)
        with self._connection() as connection:
            eligible = bool(
                connection.execute(
                    "SELECT eligible FROM entities WHERE iri=? AND kind=?",
                    (ref["iri"], ref["kind"]),
                ).fetchone()[0]
            )
        categories = {
            category: self.facts(ref, category=category, limit=20, policy=policy)
            for category in CATEGORIES
            if category != "usage"
        }
        label_facts = categories["labels"]["items"]
        parents = self.hierarchy(ref, direction="parents", policy=policy)
        parent_facts = []
        for edge in parents["items"]:
            with self._connection() as connection:
                row = connection.execute(
                    "SELECT fact FROM axioms WHERE id=? AND fact_bytes<=?",
                    (edge["axiom_id"], _FACT_PAGE_BYTES // 2),
                ).fetchone()
            if row is None:
                continue
            fact = _fact_view(json.loads(row[0]), ref, policy)
            fact["value"] = {"term_type": "iri", "iri": edge["parent"]["iri"]}
            parent_facts.append(fact)
        parent_page = {
            key: value
            for key, value in parents.items()
            if key not in {"nodes", "basis", "node_budget_reached"}
        }
        parent_page["items"] = parent_facts
        parent_page["returned_count"] = len(parent_facts)
        return {
            "artifact_type": "entity_context",
            "contract_version": CONTRACT_VERSION,
            "labels": [
                {
                    "text": fact["value"]["lexical_form"],
                    "language": fact["value"].get("language"),
                    "predicate_iri": fact["predicate_iri"],
                }
                for fact in label_facts
                if fact.get("value", {}).get("term_type") == "literal"
            ],
            "definitions": categories["definitions"],
            "synonyms": categories["synonyms"],
            "parents": parent_page,
            "context_scope": self._scope(
                {"operation": "entity_context", "entity": ref, "language": language},
                policy,
                "root_document" if self.manifest["scope"] == "root" else "resolved_closure",
            ),
            "fixture_provenance": "Prepared from a pyowl-core source snapshot; source hashes and original axiom identities are retained",
            "entity": ref,
            "ontology_version_id": self.ontology_version_id,
            "preferred_label": self._label(ref, language, policy),
            **self._alignment_eligibility(eligible),
            "categories": categories,
            "capabilities": {
                "typed_expressions": "available",
                "asserted_browsing": "available",
                "reasoner_inferred": "not_run",
                "source_spans": (
                    "available" if self.manifest["capabilities"]["source_spans"] else "not_exported"
                ),
            },
            "capabilities_metadata": self.manifest["capabilities"],
            "completeness": {
                "scope": self.manifest["scope"],
                "extraction": self.manifest["completeness"]["extraction"],
                "imports_complete": self.manifest["completeness"]["imports_complete"],
                "domain_completeness": "not_established",
            },
        }

    def fact(self, axiom_id: str, entity: Any, *, policy: Any = None) -> dict[str, Any]:
        """Resolve a stored fact only when its axiom actually references this typed entity."""
        policy = self._effective_policy(policy)
        ref = self._entity(entity, policy)
        with self._connection() as connection:
            row = connection.execute(
                "SELECT 1 FROM refs WHERE iri=? AND kind=? AND axiom_id=?",
                (ref["iri"], ref["kind"], axiom_id),
            ).fetchone()
        if row is None:
            raise KeyError("Unknown fact")
        return _fact_view(self.axiom(axiom_id, policy=policy), ref, policy)

    def axiom(self, axiom_id: str, *, policy: Any = None) -> dict[str, Any]:
        """Read the faithful typed/original axiom after category visibility checks."""
        policy = self._effective_policy(policy)
        with self._connection() as connection:
            row = connection.execute(
                "SELECT category,payload_bytes,CASE WHEN payload_bytes<=? THEN payload ELSE NULL END payload FROM axioms WHERE id=?",
                (_AXIOM_DETAIL_BYTES, axiom_id),
            ).fetchone()
        if row is None or not _allowed(policy, self.ontology_version_id, row["category"]):
            raise KeyError("Unknown axiom")
        if row["payload"] is None:
            raise DomainError(
                "axiom_detail_too_large",
                "Axiom exceeds the inline detail budget; use the prepared ontology resource or explicit axiom stream",
                status_code=413,
            )
        result = decode_payload(row["payload"], max_bytes=_AXIOM_DETAIL_BYTES)
        if policy is not None:
            visible = _visible_node(result["ast"], policy, self.ontology_version_id)
            if visible != result["ast"]:
                result["ast"] = visible
                result["qualifiers"] = visible.get("annotations", [])
                result["original_syntax"] = None
                result["original_availability"] = "not_exported"
                result["rendering"] = expression_text(visible)
        return result

    def axiom_chunks(self, axiom_id: str, *, chunk_bytes: int = 64 * 1024, policy: Any = None):
        """Stream exact local preparation bytes without materializing large expressions.

        A policy-scoped stream requires a physically filtered package.
        """
        policy = self._effective_policy(policy)
        if policy is not None and self._prepared_policy is None:
            raise DomainError(
                "unfiltered_axiom_stream",
                "Policy-scoped streaming requires a filtered context package",
                status_code=403,
            )
        self._bound(chunk_bytes, _AXIOM_DETAIL_BYTES)
        with self._connection() as connection:
            row = connection.execute(
                "SELECT length(CAST(payload AS BLOB)),typeof(payload),category FROM axioms WHERE id=?",
                (axiom_id,),
            ).fetchone()
            if row is None or not _allowed(policy, self.ontology_version_id, row[2]):
                raise KeyError("Unknown axiom")

            def chunks():
                for offset in range(0, row[0], chunk_bytes):
                    yield connection.execute(
                        "SELECT substr(CAST(payload AS BLOB),?,?) FROM axioms WHERE id=?",
                        (offset + 1, chunk_bytes, axiom_id),
                    ).fetchone()[0]

            yield from decoded_chunks(
                chunks(), compressed=row[1] == "blob", chunk_bytes=chunk_bytes
            )

    def hierarchy(
        self,
        entity: Any,
        *,
        direction: str = "parents",
        basis: str = "literal_asserted",
        limit: int = 50,
        cursor: str | None = None,
        policy: Any = None,
    ) -> dict[str, Any]:
        """Read named adjacency or a bounded cycle-safe ancestor subgraph."""
        policy = self._effective_policy(policy)
        self._bound(limit, 1000 if direction == "ancestors" else 200)
        ref = self._entity(entity, policy)
        if direction not in {"parents", "children", "ancestors"}:
            raise ValueError("Unsupported hierarchy direction")
        if basis not in {"literal_asserted", "structural_navigation", "reasoner_inferred"}:
            raise ValueError("Unsupported hierarchy basis")
        scope = self._scope(
            {
                "operation": "hierarchy",
                "entity": ref,
                "direction": direction,
                "node_limit": limit if direction == "ancestors" else None,
            },
            policy,
            basis,
        )
        if not _allowed(policy, self.ontology_version_id, "hierarchy"):
            return self._page([], None, scope, status="not_exported")
        if basis == "reasoner_inferred":
            return self._page(
                [], None, scope, status="not_run", reason="No inference artifact was prepared"
            )
        last = decode_cursor(cursor, scope) if cursor else ""
        if not isinstance(last, str):
            raise DomainError("invalid_cursor", "Invalid collection cursor")
        column = "child" if direction != "children" else "parent"
        categories = [
            category
            for category in CATEGORIES
            if _allowed(policy, self.ontology_version_id, category)
        ]
        marks_allowed = ",".join("?" for _ in categories)
        axiom_allowed = f"EXISTS(SELECT 1 FROM axioms a WHERE a.id=axiom_id AND a.category IN ({marks_allowed}))"
        with self._connection() as connection:
            if direction == "ancestors":
                # Recursive UNION deduplicates cycles; LIMIT bounds expansion before edge reads.
                node_rows = connection.execute(
                    "WITH RECURSIVE reached(iri) AS (SELECT ? UNION SELECT e.parent FROM edges e JOIN reached r ON e.child=r.iri WHERE e.kind=? AND e.basis=? AND "
                    + axiom_allowed
                    + " LIMIT ?) SELECT iri FROM reached",
                    (ref["iri"], ref["kind"], basis, *categories, limit + 1),
                ).fetchall()
                node_iris = [row[0] for row in node_rows[:limit]]
                bounded = len(node_rows) > limit
                marks = ",".join("?" for _ in node_iris)
                # Preserve every edge joining returned nodes, with separate bounded edge paging.
                where = f"child IN ({marks}) AND parent IN ({marks}) AND kind=? AND basis=?"
                args = [*node_iris, *node_iris, ref["kind"], basis]
            else:
                bounded = False
                where = f"{column}=? AND kind=? AND basis=?"
                args = [ref["iri"], ref["kind"], basis]
            where += " AND " + axiom_allowed
            args.extend(categories)
            total = connection.execute(
                "SELECT count(*) FROM edges WHERE " + where, args
            ).fetchone()[0]
            rows = connection.execute(
                "SELECT id,payload FROM edges WHERE " + where + " AND id>? ORDER BY id LIMIT ?",
                [*args, last, min(limit, 200) + 1],
            ).fetchall()
        edge_limit = min(limit, 200)
        items = [json.loads(row["payload"]) for row in rows[:edge_limit]]
        page = self._page(
            items,
            total,
            scope,
            next_key=rows[edge_limit - 1]["id"] if len(rows) > edge_limit else None,
            status="partial" if bounded else None,
            reason="Ancestor node budget reached" if bounded else None,
        )
        nodes = {(ref["iri"], ref["kind"]): ref}
        for edge in items:
            for side in ("child", "parent"):
                node = edge[side]
                nodes[(node["iri"], node["kind"])] = node
        page["nodes"] = [nodes[key] for key in sorted(nodes)]
        page["basis"] = basis
        page["node_budget_reached"] = bounded
        page["truncated"] = page["truncated"] or bounded
        return page

    def resolve_predicate(
        self, iri: str, *, language: str | None = None, policy: Any = None
    ) -> dict[str, Any]:
        """Explain imported unlabelled predicates without inventing their descriptions."""
        policy = self._effective_policy(policy)
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT DISTINCT kind FROM terms WHERE iri=?", (iri,)
            ).fetchall()
        labels = [
            self._label(_ref(self.ontology_version_id, iri, row[0]), language, policy)
            for row in rows
        ]
        available = next((label for label in labels if label["status"] == "available"), None)
        if available:
            return {"iri": iri, **available}
        unresolved = [
            item for item in self.manifest["identity"]["imports"] if item["status"] != "resolved"
        ]
        return {
            "iri": iri,
            "value": None,
            "status": "unresolved_import" if unresolved else "absent_in_scope",
            "reason": "No predicate label in the frozen scope",
            "unresolved_imports": unresolved,
        }

    def resolve_feature(self, item: Mapping[str, Any], entity: Any) -> dict[str, Any]:
        """Resolve only exact semantic matches; historical display text cannot prove identity."""
        ref = self._entity(entity)
        matches: list[str] = []
        with self._connection() as connection:
            original_ids = item.get("source_axiom_refs", [])
            if not isinstance(original_ids, (list, tuple)) or len(original_ids) > 100:
                raise ValueError("At most 100 original axiom references are supported")
            if original_ids:
                placeholders = ",".join("?" for _ in original_ids)
                rows = connection.execute(
                    "SELECT a.id FROM axioms a WHERE a.original_digest IN (" + placeholders + ")",
                    [*original_ids],
                ).fetchall()
                matches = [row[0] for row in rows]
            elif (item.get("prop_iri") or item.get("property_iri")) and "value" in item:
                rows = connection.execute(
                    "SELECT a.id FROM refs r JOIN axioms a ON a.id=r.axiom_id WHERE r.iri=? AND r.kind=? AND a.predicate=? AND a.lexical=? AND COALESCE(a.language,'')=? AND a.datatype=?",
                    (
                        ref["iri"],
                        ref["kind"],
                        item.get("prop_iri", item.get("property_iri")),
                        item["value"],
                        item.get("language") or "",
                        item.get("datatype"),
                    ),
                ).fetchall()
                matches = [row[0] for row in rows]
            elif (
                item.get("subject_iri")
                and item.get("object_iri")
                and (
                    item.get("family") == "is_a"
                    or item.get("rel_iri")
                    in {"http://www.w3.org/2000/01/rdf-schema#subClassOf", "http://subclassof"}
                )
            ):
                rows = connection.execute(
                    "SELECT DISTINCT axiom_id FROM edges WHERE child=? AND parent=? AND kind=? AND basis='structural_navigation'",
                    (item["subject_iri"], item["object_iri"], ref["kind"]),
                ).fetchall()
                matches = [row[0] for row in rows]
        return {
            "status": "available" if matches else "not_exported",
            "fact_ids": sorted(matches),
            "ontology_version_id": self.ontology_version_id,
            "reason": None if matches else "No unambiguous exact semantic origin was resolved",
        }
