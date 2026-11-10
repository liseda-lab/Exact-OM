"""Bounded researcher-only publication inventory without participant or key data."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from .forms import FORMS3_VERSIONS
from .models import Identifier, StrictModel


class RevisionSummary(StrictModel):
    study_revision: Identifier
    contract_version: str
    software_version: str
    information_version: str
    form_version: str
    tutorial_version: str | None
    setup_version: str | None
    export_version: str
    source_export_version: str
    supported_analysis_schemas: list[str]
    frozen_hash: str
    closed: bool
    synthetic: bool
    closes_at: str
    allocation_count: int
    test_sessions: int
    participant_sessions: int


class RevisionPage(StrictModel):
    items: list[RevisionSummary]
    next_cursor: str | None
    order: str = "study_revision_ascending"


class InvitationSummary(StrictModel):
    session_id: str
    generation: int
    test: bool
    stage: str


class InvitationPage(StrictModel):
    items: list[InvitationSummary]
    next_cursor: str | None
    study_closed: bool


def invitations(store, revision, *, limit=50, cursor=None):
    """List unrevoked links in one study without secrets or participant answers."""
    with store.transaction() as db:
        row, study = store._study(db, revision)
        closed = bool(row["closed"]) or datetime.fromisoformat(study["closes_at"]) <= datetime.now(
            timezone.utc
        )
        if closed:
            return {"items": [], "next_cursor": None, "study_closed": True}
        stage = (
            "CAST(state AS json)->>'stage'" if store.postgres else "json_extract(state, '$.stage')"
        )
        rows = db.execute(
            "SELECT id, generation, test, state FROM sessions "
            f"WHERE study_revision = ? AND revoked = 0 AND {stage} <> 'closed' AND id > ? ORDER BY id LIMIT ?",
            (revision, cursor or "", limit + 1),
        ).fetchall()
        return {
            "items": [
                {
                    "session_id": r["id"],
                    "generation": r["generation"],
                    "test": bool(r["test"]),
                    "stage": json.loads(r["state"])["stage"],
                }
                for r in rows[:limit]
            ],
            "next_cursor": rows[limit - 1]["id"] if len(rows) > limit else None,
            "study_closed": False,
        }


def revisions(store, *, limit=50, cursor=None):
    """Keyset-page frozen metadata; never return publication text, invitations or keys."""
    with store.transaction() as db:
        rows = db.execute(
            "SELECT revision, payload, frozen_hash, allocation_count, closed FROM studies "
            "WHERE revision > ? ORDER BY revision LIMIT ?",
            (cursor or "", limit + 1),
        ).fetchall()
        items = []
        for row in rows[:limit]:
            definition = json.loads(row["payload"])
            counts = {
                int(c["test"]): c["count"]
                for c in db.execute(
                    "SELECT test, COUNT(*) AS count FROM sessions WHERE study_revision = ? GROUP BY test",
                    (row["revision"],),
                ).fetchall()
            }
            protocol = definition["contract_version"]
            versions = definition.get("protocol_versions", {})
            items.append(
                {
                    "study_revision": row["revision"],
                    "contract_version": protocol,
                    "software_version": definition["software_version"],
                    "information_version": definition["information_version"],
                    "form_version": definition["form_version"],
                    "tutorial_version": (definition.get("tutorial") or {}).get("version"),
                    "setup_version": versions.get("setup"),
                    "export_version": versions.get("export", "exact-study-analysis/1"),
                    "source_export_version": versions.get("export", "exact-study-analysis/1"),
                    "supported_analysis_schemas": (
                        ["exact-study-analysis/4"]
                        if definition.get("form_version") in FORMS3_VERSIONS
                        else (
                            ["exact-study-analysis/2", "exact-study-analysis/3"]
                            if protocol == "exact-study/2.0"
                            else ["exact-study-analysis/1"]
                        )
                    ),
                    "frozen_hash": row["frozen_hash"],
                    "closed": bool(row["closed"]),
                    "synthetic": definition["synthetic"],
                    "closes_at": definition["closes_at"],
                    "allocation_count": row["allocation_count"],
                    "test_sessions": counts.get(1, 0),
                    "participant_sessions": counts.get(0, 0),
                }
            )
        return {
            "items": items,
            "next_cursor": items[-1]["study_revision"] if len(rows) > limit else None,
            "order": "study_revision_ascending",
        }
