"""Bounded reads and wire contracts for explicitly prepared ontology navigation.

This module deliberately has no OWL/reasoner imports. Native classification is an
offline operation in navigation_prepare; the immutable result is policy-bound.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any, Literal

from pydantic import Field

from .context_semantics import KINDS, _allowed, _ref
from .contracts import EntityRef, Page, WireModel, canonical_hash, decode_cursor, file_hash
from .models import HierarchyPage, PreferredLabel
from .sqlite_safety import readonly_connection

NAVIGATION_CONTRACT = "exact-hierarchy/1"
NAVIGATION_FILE = "navigation.sqlite"
BASES = ("literal_asserted", "structural_navigation", "reasoner_inferred")
TABLES = {
    "metadata": ("key", "value"),
    "entries": ("basis", "kind", "ordinal", "iri", "entry_kind", "component_size"),
    "edges": ("id", "child", "parent", "kind", "payload"),
    "members": ("iri", "kind", "group_id", "is_top", "is_bottom"),
}


class EntryPoint(WireModel):
    entity: EntityRef
    preferred_label: PreferredLabel
    entry_kind: Literal["named_root", "component_entry", "entity_listing"]
    component_size: int = Field(ge=1)


class EntryPointPage(Page[EntryPoint]):
    navigation_contract: Literal["exact-hierarchy/1"] = NAVIGATION_CONTRACT
    default_entity: EntityRef | None = None
    basis: str
    completeness: dict[str, Any]
    inference: dict[str, Any]


class PreparedHierarchyPage(HierarchyPage):
    navigation_contract: Literal["exact-hierarchy/1"] = NAVIGATION_CONTRACT
    inference: dict[str, Any] | None = None
    entity_semantics: dict[str, Any] | None = None


class PreparedNavigation:
    def __init__(self, context):
        self.context = context
        self.metadata = context.manifest["navigation"]
        self.path = context.path / NAVIGATION_FILE
        if self.path.is_symlink() or self.path.resolve().parent != context.path.resolve():
            raise ValueError("Navigation artifact must be inside its context package")
        if self.metadata.get("contract_version") != NAVIGATION_CONTRACT:
            raise ValueError("Unsupported prepared navigation contract")
        if self.metadata.get("context_sha256") != context.manifest["artifacts"]["context.sqlite"]:
            raise ValueError("Navigation belongs to another context revision")
        self.signature = self._stat()
        expected = context.manifest["artifacts"].get(NAVIGATION_FILE)
        if not expected or file_hash(self.path) != expected or self._stat() != self.signature:
            raise ValueError("Navigation artifact checksum mismatch")
        self.identity = expected
        with self.connection() as db:
            stored = dict(db.execute("SELECT key,value FROM metadata"))
        if stored.get("binding") != canonical_hash(self.metadata):
            raise ValueError("Navigation metadata binding mismatch")

    def _stat(self):
        s = self.path.stat()
        return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns

    @contextmanager
    def connection(self):
        if self._stat() != self.signature:
            raise ValueError("Navigation artifact changed after verification")
        with readonly_connection(self.path, TABLES) as db:
            yield db

    def matches(self, policy):
        actual = policy.policy_hash if policy else canonical_hash({"policy": "local-all"})
        return actual == self.metadata["policy_hash"]


def metadata(context, policy=None):
    """Advertise only the result prepared for this exact effective policy."""
    policy = context._effective_policy(policy)
    navigation = getattr(context, "_navigation", None)
    if navigation and navigation.matches(policy):
        with navigation.connection():
            pass
        return navigation.metadata
    return {
        "contract_version": NAVIGATION_CONTRACT,
        "entry_points": "not_run",
        "inference": {
            "status": "not_run",
            "reason": "No navigation artifact was prepared for this context and policy",
            "supported_kinds": ["class"],
        },
    }


def entry_points(
    context,
    *,
    kind="class",
    basis="literal_asserted",
    language=None,
    limit=20,
    cursor=None,
    policy=None,
):
    policy = context._effective_policy(policy)
    context._bound(limit)
    if kind not in KINDS or basis not in BASES:
        raise ValueError("Unsupported entry-point kind or basis")
    info = metadata(context, policy)
    navigation = getattr(context, "_navigation", None)
    scope = context._scope(
        {
            "operation": "entry_points",
            "kind": kind,
            "language": language,
            "navigation": navigation.identity if navigation else None,
        },
        policy,
        basis,
    )
    extra = {
        "navigation_contract": NAVIGATION_CONTRACT,
        "basis": basis,
        "default_entity": None,
        "completeness": context.manifest["completeness"],
        "inference": info["inference"],
    }
    if not _allowed(policy, context.ontology_version_id, "hierarchy") and kind != "individual":
        return {**context._page([], None, scope, status="not_exported"), **extra}
    if info["entry_points"] != "available":
        return {
            **context._page([], None, scope, status="not_run", reason=info["inference"]["reason"]),
            **extra,
        }
    if basis == "reasoner_inferred" and (
        kind != "class" or info["inference"]["status"] != "available"
    ):
        status = "unsupported" if kind != "class" else info["inference"]["status"]
        return {
            **context._page(
                [],
                None,
                scope,
                status=status,
                reason=(
                    "Inferred taxonomy supports classes only"
                    if kind != "class"
                    else info["inference"].get("reason")
                ),
            ),
            **extra,
        }
    last = decode_cursor(cursor, scope) if cursor else -1
    if type(last) is not int or last < -1:
        from .contracts import DomainError

        raise DomainError("invalid_cursor", "Invalid entry-point cursor")
    with navigation.connection() as db:
        total = db.execute(
            "SELECT count(*) FROM entries WHERE basis=? AND kind=?", (basis, kind)
        ).fetchone()[0]
        rows = db.execute(
            "SELECT * FROM entries WHERE basis=? AND kind=? AND ordinal>? ORDER BY ordinal LIMIT ?",
            (basis, kind, last, limit + 1),
        ).fetchall()
        default = db.execute(
            "SELECT iri FROM entries WHERE basis=? AND kind=? ORDER BY ordinal LIMIT 1",
            (basis, kind),
        ).fetchone()
    items = []
    for row in rows[:limit]:
        entity = _ref(context.ontology_version_id, row["iri"], kind)
        items.append(
            {
                "entity": entity,
                "preferred_label": context._label(entity, language, policy),
                "entry_kind": row["entry_kind"],
                "component_size": row["component_size"],
            }
        )
    extra["default_entity"] = (
        _ref(context.ontology_version_id, default[0], kind) if default else None
    )
    return {
        **context._page(
            items, total, scope, next_key=rows[limit - 1]["ordinal"] if len(rows) > limit else None
        ),
        **extra,
    }


def inferred_hierarchy(context, ref, *, direction, limit, cursor, policy):
    info = metadata(context, policy)["inference"]
    navigation = getattr(context, "_navigation", None)
    scope = context._scope(
        {
            "operation": "hierarchy",
            "entity": ref,
            "direction": direction,
            "node_limit": limit if direction == "ancestors" else None,
            "navigation": navigation.identity if navigation else None,
        },
        policy,
        "reasoner_inferred",
    )
    extra = {
        "navigation_contract": NAVIGATION_CONTRACT,
        "inference": info,
        "basis": "reasoner_inferred",
        "entity_semantics": None,
    }
    if ref["kind"] != "class" or info["status"] != "available":
        return {
            **context._page(
                [],
                None,
                scope,
                status="unsupported" if ref["kind"] != "class" else info["status"],
                reason=(
                    "Inferred taxonomy supports classes only"
                    if ref["kind"] != "class"
                    else info.get("reason")
                ),
            ),
            **extra,
        }
    last = decode_cursor(cursor, scope) if cursor else ""
    if not isinstance(last, str):
        from .contracts import DomainError

        raise DomainError("invalid_cursor", "Invalid inferred hierarchy cursor")
    with navigation.connection() as db:
        member = db.execute(
            "SELECT * FROM members WHERE iri=? AND kind=?", (ref["iri"], ref["kind"])
        ).fetchone()
        if member:
            count = db.execute(
                "SELECT count(*) FROM members WHERE group_id=?", (member["group_id"],)
            ).fetchone()[0]
            extra["entity_semantics"] = {
                "equivalence_group_id": member["group_id"],
                "equivalent_entity_count": count,
                "is_top": bool(member["is_top"]),
                "is_bottom": bool(member["is_bottom"]),
            }
        bounded = False
        if direction == "ancestors":
            reached = db.execute(
                "WITH RECURSIVE reached(iri) AS (SELECT ? UNION SELECT e.parent FROM edges e JOIN reached r ON e.child=r.iri WHERE e.kind=? LIMIT ?) SELECT iri FROM reached",
                (ref["iri"], ref["kind"], limit + 1),
            ).fetchall()
            nodes = [r[0] for r in reached[:limit]]
            bounded = len(reached) > limit
            marks = ",".join("?" for _ in nodes)
            where, args = f"child IN ({marks}) AND parent IN ({marks}) AND kind=?", [
                *nodes,
                *nodes,
                ref["kind"],
            ]
        else:
            where = ("child" if direction == "parents" else "parent") + "=? AND kind=?"
            args = [ref["iri"], ref["kind"]]
        total = db.execute("SELECT count(*) FROM edges WHERE " + where, args).fetchone()[0]
        page_limit = min(limit, 200)
        rows = db.execute(
            "SELECT id,payload FROM edges WHERE " + where + " AND id>? ORDER BY id LIMIT ?",
            [*args, last, page_limit + 1],
        ).fetchall()
    items = [json.loads(row["payload"]) for row in rows[:page_limit]]
    result = context._page(
        items,
        total,
        scope,
        next_key=rows[page_limit - 1]["id"] if len(rows) > page_limit else None,
        status="partial" if bounded else None,
        reason="Ancestor node budget reached" if bounded else None,
    )
    nodes = {ref["iri"]: ref}
    for edge in items:
        for side in ("child", "parent"):
            nodes[edge[side]["iri"]] = edge[side]
    return {
        **result,
        **extra,
        "nodes": [nodes[key] for key in sorted(nodes)],
        "node_budget_reached": bounded,
        "truncated": result["truncated"] or bounded,
    }


def equivalent_entities(context, entity, *, limit=20, cursor=None, policy=None):
    """Page the named members of the selected entity's recorded taxonomy node."""
    policy = context._effective_policy(policy)
    context._bound(limit)
    ref = context._entity(entity, policy)
    info = metadata(context, policy)["inference"]
    navigation = getattr(context, "_navigation", None)
    scope = context._scope(
        {
            "operation": "equivalents",
            "entity": ref,
            "navigation": navigation.identity if navigation else None,
        },
        policy,
        "reasoner_inferred",
    )
    if not _allowed(policy, context.ontology_version_id, "hierarchy"):
        return context._page([], None, scope, status="not_exported")
    if ref["kind"] != "class" or info["status"] != "available":
        return context._page(
            [],
            None,
            scope,
            status="unsupported" if ref["kind"] != "class" else info["status"],
            reason=(
                "Inferred taxonomy supports classes only"
                if ref["kind"] != "class"
                else info.get("reason")
            ),
        )
    last = decode_cursor(cursor, scope) if cursor else ""
    if not isinstance(last, str):
        from .contracts import DomainError

        raise DomainError("invalid_cursor", "Invalid equivalence-group cursor")
    with navigation.connection() as db:
        group = db.execute(
            "SELECT group_id FROM members WHERE iri=? AND kind=?", (ref["iri"], ref["kind"])
        ).fetchone()
        if group is None:
            raise ValueError("Prepared taxonomy omitted an admitted class")
        total = db.execute("SELECT count(*) FROM members WHERE group_id=?", (group[0],)).fetchone()[
            0
        ]
        rows = db.execute(
            "SELECT iri FROM members WHERE group_id=? AND iri>? ORDER BY iri LIMIT ?",
            (group[0], last, limit + 1),
        ).fetchall()
    return context._page(
        [_ref(context.ontology_version_id, row[0], "class") for row in rows[:limit]],
        total,
        scope,
        next_key=rows[limit - 1][0] if len(rows) > limit else None,
    )
