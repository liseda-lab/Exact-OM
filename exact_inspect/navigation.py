"""Bounded reads and wire contracts for explicitly prepared ontology navigation.

This module deliberately has no OWL/reasoner imports. Native classification is an
offline operation in navigation_prepare; the immutable result is policy-bound.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from contextlib import contextmanager
from threading import RLock
from typing import Any, Literal

from pydantic import Field, model_validator

from .context_semantics import KINDS, _allowed, _ref
from .contracts import (
    EntityRef,
    Page,
    WireModel,
    canonical_hash,
    decode_cursor,
    file_hash,
)
from .models import HierarchyPage, PreferredLabel
from .sqlite_safety import readonly_connection

NAVIGATION_CONTRACT = "exact-hierarchy/1"
NAVIGATION_FILE = "navigation.sqlite"
PARTIAL_BASIS = "elk_inferred_partial"
PARTIAL_CONTRACT = "exact-partial-hierarchy/1"
PARTIAL_FILE = "partial-navigation.sqlite"
BASES = ("literal_asserted", "structural_navigation", "reasoner_inferred", PARTIAL_BASIS)
TABLES = {
    "metadata": ("key", "value"),
    "entries": ("basis", "kind", "ordinal", "iri", "entry_kind", "component_size"),
    "edges": ("id", "child", "parent", "kind", "payload"),
    "members": ("iri", "kind", "group_id", "is_top", "is_bottom"),
}


# An explicit review-navigation choice, not an obsolescence classifier or OWL edge.
# Bind both ontology identity and prepared bytes: unrelated DOID versions do not inherit it.
CURATED_DEFAULTS = {
    "sha256:26054e89dafcade53c821c0cc1a0b00574ec59becc81f6a930fc7474e44ab500": {
        "context_sha256": "sha256:d18cc163a3bcc4546029e08e347fe7d896848353fcb03df95cda3bd25660c99a",
        "iri": "http://purl.obolibrary.org/obo/DOID_4",
        "rationale": "Curated review starting point: disease. All recorded roots remain accessible; this choice makes no claim about other entities' obsolescence.",
        "revision": "review-navigation/20261013",
    }
}


class EntrySelection(WireModel):
    kind: Literal["prepared_order", "curated_default"]
    revision: str | None = None
    rationale: str | None = None
    assertion: Literal["navigation_only"] = "navigation_only"


class EntryPoint(WireModel):
    entity: EntityRef
    preferred_label: PreferredLabel
    entry_kind: Literal["named_root", "component_entry", "entity_listing"]
    component_size: int = Field(ge=1)


def _validate_partial_page(value):
    if value.basis == PARTIAL_BASIS and value.status == "partial":
        from .postmigration_contracts import PartialInferenceMetadata

        PartialInferenceMetadata.model_validate(value.inference)
    return value


class EntryPointPage(Page[EntryPoint]):
    navigation_contract: Literal["exact-hierarchy/1", "exact-partial-hierarchy/1"] = (
        NAVIGATION_CONTRACT
    )
    default_entity: EntityRef | None = None
    selection: EntrySelection = Field(default_factory=lambda: EntrySelection(kind="prepared_order"))
    basis: str
    completeness: dict[str, Any]
    inference: dict[str, Any]

    _partial = model_validator(mode="after")(_validate_partial_page)


class PreparedHierarchyPage(HierarchyPage):
    navigation_contract: Literal["exact-hierarchy/1", "exact-partial-hierarchy/1"] = (
        NAVIGATION_CONTRACT
    )
    inference: dict[str, Any] | None = None
    entity_semantics: dict[str, Any] | None = None

    _partial = model_validator(mode="after")(_validate_partial_page)


class PreparedEquivalentsPage(Page[EntityRef]):
    navigation_contract: Literal["exact-hierarchy/1", "exact-partial-hierarchy/1"] = (
        NAVIGATION_CONTRACT
    )
    basis: str = "reasoner_inferred"
    inference: dict[str, Any] | None = None

    _partial = model_validator(mode="after")(_validate_partial_page)


_VERIFIED_NAVIGATION = OrderedDict()
_VERIFICATION_LOCK = RLock()


class PreparedNavigation:
    def __init__(self, context, *, partial=False):
        self.context = context
        self.metadata = context.manifest["partial_navigation" if partial else "navigation"]
        filename = PARTIAL_FILE if partial else NAVIGATION_FILE
        self.path = context.path / filename
        if self.path.is_symlink() or self.path.resolve().parent != context.path.resolve():
            raise ValueError("Navigation artifact must be inside its context package")
        if self.metadata.get("contract_version") != (
            PARTIAL_CONTRACT if partial else NAVIGATION_CONTRACT
        ):
            raise ValueError("Unsupported prepared navigation contract")
        if self.metadata.get("context_sha256") != context.manifest["artifacts"]["context.sqlite"]:
            raise ValueError("Navigation belongs to another context revision")
        self.signature = self._stat()
        expected = context.manifest["artifacts"].get(filename)
        key = (str(self.path.resolve()), self.signature)
        with _VERIFICATION_LOCK:
            digest = _VERIFIED_NAVIGATION.get(key)
            if digest is None:
                digest = file_hash(self.path)
                if self._stat() != self.signature:
                    raise ValueError("Navigation artifact changed during verification")
                _VERIFIED_NAVIGATION[key] = digest
                while len(_VERIFIED_NAVIGATION) > 128:
                    _VERIFIED_NAVIGATION.popitem(last=False)
        if not expected or digest != expected or self._stat() != self.signature:
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


def prepared(context, basis):
    return getattr(
        context, "_partial_navigation" if basis == PARTIAL_BASIS else "_navigation", None
    )


def metadata(context, policy=None, *, basis=None):
    """Advertise only the result prepared for this exact effective policy."""
    policy = context._effective_policy(policy)
    navigation = prepared(context, basis)
    if navigation and navigation.matches(policy):
        with navigation.connection():
            pass
        info = dict(navigation.metadata)
        if basis is None and getattr(context, "_partial_navigation", None):
            info["partial_inference"] = metadata(context, policy, basis=PARTIAL_BASIS)["inference"]
        return info
    info = {
        "contract_version": PARTIAL_CONTRACT if basis == PARTIAL_BASIS else NAVIGATION_CONTRACT,
        "entry_points": "not_run",
        "inference": {
            "status": "not_run",
            "reason": "No navigation artifact was prepared for this context and policy",
            "supported_kinds": ["class"],
        },
    }

    if basis is None and getattr(context, "_partial_navigation", None):
        info["partial_inference"] = metadata(context, policy, basis=PARTIAL_BASIS)["inference"]
    return info


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
    info = metadata(context, policy, basis=basis)
    navigation = prepared(context, basis)
    choice = CURATED_DEFAULTS.get(context.ontology_version_id)
    if not (
        choice
        and kind == "class"
        and basis in {"literal_asserted", "structural_navigation"}
        and choice["context_sha256"] == context.manifest["artifacts"]["context.sqlite"]
    ):
        choice = None
    scope = context._scope(
        {
            "operation": "entry_points",
            "entry_selection": canonical_hash(choice) if choice else None,
            "kind": kind,
            "language": language,
            "navigation": navigation.identity if navigation else None,
        },
        policy,
        basis,
    )
    last = decode_cursor(cursor, scope) if cursor else -1
    extra = {
        "navigation_contract": PARTIAL_CONTRACT if basis == PARTIAL_BASIS else NAVIGATION_CONTRACT,
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
    if basis in {"reasoner_inferred", PARTIAL_BASIS} and (
        kind != "class"
        or info["inference"]["status"] != ("partial" if basis == PARTIAL_BASIS else "available")
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
    if type(last) is not int or last < -1:
        from .contracts import DomainError

        raise DomainError("invalid_cursor", "Invalid entry-point cursor")
    with navigation.connection() as db:
        total = db.execute(
            "SELECT count(*) FROM entries WHERE basis=? AND kind=?", (basis, kind)
        ).fetchone()[0]
        preferred = (
            db.execute(
                "SELECT * FROM entries WHERE basis=? AND kind=? AND iri=?",
                (basis, kind, choice["iri"]),
            ).fetchone()
            if choice
            else None
        )
        # Only promote an already admitted root; never manufacture a new hierarchy entry.
        excluded = preferred["iri"] if preferred else ""
        rows = list(
            db.execute(
                "SELECT * FROM entries WHERE basis=? AND kind=? AND ordinal>? AND iri<>? ORDER BY ordinal LIMIT ?",
                (basis, kind, last, excluded, limit + 1),
            ).fetchall()
        )
        if preferred and cursor is None:
            rows.insert(0, {**dict(preferred), "ordinal": -1})
        default = (
            preferred
            or db.execute(
                "SELECT * FROM entries WHERE basis=? AND kind=? ORDER BY ordinal LIMIT 1",
                (basis, kind),
            ).fetchone()
        )
        if preferred:
            extra["selection"] = {
                "kind": "curated_default",
                "revision": choice["revision"],
                "rationale": choice["rationale"],
                "assertion": "navigation_only",
            }
    # First page is the declared starting choice. Continuation exposes the complete
    # original inventory, including obsolete or unclassified roots, without hiding any.
    page_limit = 1 if preferred and cursor is None else limit
    items = []
    for row in rows[:page_limit]:
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
        _ref(context.ontology_version_id, default["iri"], kind) if default else None
    )
    return {
        **context._page(
            items,
            total,
            scope,
            next_key=rows[page_limit - 1]["ordinal"] if len(rows) > page_limit else None,
            status="partial" if basis == PARTIAL_BASIS else None,
            reason="Partial taxonomy; absence is unknown" if basis == PARTIAL_BASIS else None,
        ),
        **extra,
    }


def inferred_hierarchy(
    context, ref, *, direction, limit, cursor, policy, basis="reasoner_inferred"
):
    info = metadata(context, policy, basis=basis)["inference"]
    navigation = prepared(context, basis)
    scope = context._scope(
        {
            "operation": "hierarchy",
            "entity": ref,
            "direction": direction,
            "node_limit": limit if direction == "ancestors" else None,
            "navigation": navigation.identity if navigation else None,
        },
        policy,
        basis,
    )
    last = decode_cursor(cursor, scope) if cursor else ""
    extra = {
        "navigation_contract": PARTIAL_CONTRACT if basis == PARTIAL_BASIS else NAVIGATION_CONTRACT,
        "inference": info,
        "basis": basis,
        "entity_semantics": None,
    }
    if ref["kind"] != "class" or info["status"] != (
        "partial" if basis == PARTIAL_BASIS else "available"
    ):
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
        status="partial" if bounded or basis == PARTIAL_BASIS else None,
        reason=(
            "Ancestor node budget reached"
            if bounded
            else ("Partial taxonomy; absence is unknown" if basis == PARTIAL_BASIS else None)
        ),
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


def equivalent_entities(
    context, entity, *, limit=20, cursor=None, policy=None, basis="reasoner_inferred"
):
    """Page the named members of the selected entity's recorded taxonomy node."""
    policy = context._effective_policy(policy)
    context._bound(limit)
    ref = context._entity(entity, policy)
    if basis not in {"reasoner_inferred", PARTIAL_BASIS}:
        raise ValueError("Unsupported equivalence basis")
    info = metadata(context, policy, basis=basis)["inference"]
    navigation = prepared(context, basis)
    scope = context._scope(
        {
            "operation": "equivalents",
            "entity": ref,
            "navigation": navigation.identity if navigation else None,
        },
        policy,
        basis,
    )
    last = decode_cursor(cursor, scope) if cursor else ""
    extra = {
        "basis": basis,
        "inference": info,
        "navigation_contract": PARTIAL_CONTRACT if basis == PARTIAL_BASIS else NAVIGATION_CONTRACT,
    }
    if not _allowed(policy, context.ontology_version_id, "hierarchy"):
        return {**context._page([], None, scope, status="not_exported"), **extra}
    if ref["kind"] != "class" or info["status"] != (
        "partial" if basis == PARTIAL_BASIS else "available"
    ):
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
    if not isinstance(last, str):
        from .contracts import DomainError

        raise DomainError("invalid_cursor", "Invalid equivalence-group cursor")
    with navigation.connection() as db:
        group = db.execute(
            "SELECT group_id FROM members WHERE iri=? AND kind=?", (ref["iri"], ref["kind"])
        ).fetchone()
        if group is None:
            if basis == PARTIAL_BASIS:
                return {
                    **context._page(
                        [],
                        0,
                        scope,
                        status="partial",
                        reason="Class not represented in partial taxonomy",
                    ),
                    "basis": basis,
                    "inference": info,
                    "navigation_contract": PARTIAL_CONTRACT,
                }
            raise ValueError("Prepared taxonomy omitted an admitted class")
        total = db.execute("SELECT count(*) FROM members WHERE group_id=?", (group[0],)).fetchone()[
            0
        ]
        rows = db.execute(
            "SELECT iri FROM members WHERE group_id=? AND iri>? ORDER BY iri LIMIT ?",
            (group[0], last, limit + 1),
        ).fetchall()
    result = context._page(
        [_ref(context.ontology_version_id, row[0], "class") for row in rows[:limit]],
        total,
        scope,
        next_key=rows[limit - 1][0] if len(rows) > limit else None,
        status="partial" if basis == PARTIAL_BASIS else None,
        reason="Partial equivalence; absence is unknown" if basis == PARTIAL_BASIS else None,
    )
    if basis == PARTIAL_BASIS:
        result.update(basis=basis, inference=info, navigation_contract=PARTIAL_CONTRACT)
    return result
