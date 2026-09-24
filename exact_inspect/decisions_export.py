"""Physical policy filtering for portable matcher evidence, without changing decisions."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, Mapping

from .artifacts import atomic_json
from .contracts import VisibilityPolicy, canonical_hash, file_hash
from .decisions import DecisionStore


def export_policy_run(
    source: DecisionStore,
    destination: Path,
    *,
    contexts: Mapping[str, Any],
    policy: VisibilityPolicy,
) -> DecisionStore:
    """Write new SQLite pages containing only evidence linked to admitted original facts.

    Raw historic display strings cannot prove their own visibility. Portable clients
    render linked originals instead. Decision scores and final memberships are copied
    exactly; missing or withheld evidence does not change matcher behavior.
    """
    destination.mkdir(parents=True, exist_ok=True)
    database = destination / "decisions.sqlite"
    if database.exists():
        database.unlink()  # This directory is unpublished preparation staging.
    manifest = source.manifest()
    retained = 0
    pair_count = 0
    with source._connect() as original, closing(sqlite3.connect(database)) as target:
        for (schema,) in original.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL"
        ):
            target.execute(schema)
        for row in original.execute("SELECT * FROM sources"):
            target.execute("INSERT INTO sources VALUES (?,?,?,?)", row)
        for row in original.execute("SELECT * FROM pairs"):
            candidate, trace, items = json.loads(row[5]), json.loads(row[6]), json.loads(row[7])
            safe = []
            for item in items:
                ids = item.get("fact_ids", [])
                context = contexts.get(item["entity"]["ontology_version_id"])
                if not ids or context is None:
                    continue
                if (
                    context.manifest.get("policy_filter", {}).get("policy_hash")
                    != policy.policy_hash
                ):
                    raise ValueError("Portable evidence requires a physically filtered context")
                with context._connection() as facts:
                    marks = ",".join("?" for _ in ids)
                    available = {
                        r[0]
                        for r in facts.execute(
                            "SELECT id FROM axioms WHERE id IN (" + marks + ")", ids
                        )
                    }
                if set(ids) != available:
                    continue
                visible = {
                    key: item[key]
                    for key in (
                        "evidence_id",
                        "channel",
                        "side",
                        "role",
                        "entity",
                        "feature_id",
                        "fact_ids",
                        "source_axiom_refs",
                        "values",
                        "interpretation",
                        "provenance_status",
                        "status",
                        "reason",
                    )
                    if key in item
                }
                visible.update(
                    {
                        "display": {},
                        "semantic_terms": {},
                        "axiom_origins": [],
                        "historical_item_alias": None,
                    }
                )
                safe.append(visible)
            candidate["evidence_counts"] = {
                "visible_selected": len(safe),
                "scope": "prepared_visibility_policy",
            }
            candidate["evidence_status"] = "available" if safe else "not_exported"
            target.execute(
                "INSERT INTO pairs VALUES (?,?,?,?,?,?,?,?)",
                (*row[:5], json.dumps(candidate), json.dumps(trace), json.dumps(safe)),
            )
            pair_count += 1
            retained += len(safe)
        for (schema,) in original.execute(
            "SELECT sql FROM sqlite_master WHERE type='index' AND sql IS NOT NULL"
        ):
            target.execute(schema)
        target.commit()
    filtered = {
        **manifest,
        "revision": canonical_hash(
            {
                "source_revision": manifest["revision"],
                "policy_hash": policy.policy_hash,
                "exporter": "exact-policy-decisions/1",
            }
        ),
        "counts": {
            "pairs": pair_count,
            "selected_evidence": retained,
            "scope": "prepared_visibility_policy",
        },
        "policy_filter": {
            "policy_hash": policy.policy_hash,
            "source_revision": manifest["revision"],
        },
        "database_hash": file_hash(database),
    }
    atomic_json(destination / "decisions.manifest.json", filtered)
    return DecisionStore(destination)
