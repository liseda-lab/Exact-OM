"""Disk-backed global reductions for the deployed greedy/mutual-best strategies.

The input and final public API may be materialized by the caller; intermediate
deduplication, thresholding and competition use a bounded SQLite page cache. This
is disposable scratch, never a scientific checkpoint or a chunk-local matcher.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory

from exact.core.entities.mappings.entity import EntityMapping


def extract_indexed(
    mappings, *, directory, mode, threshold, protected_pairs, source_cardinality,
    target_cardinality, anchor_conflict_policy, assignment_component_cap,
):
    from exact.impl.extraction import (
        ExtractionResult, _kind_text, _protected_exact_conflicts,
        _validate_protected_exact_constraints,
    )

    Path(directory).mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="exact-extraction-", dir=directory) as temporary:
        db = sqlite3.connect(str(Path(temporary) / "edges.sqlite"))
        try:
            db.execute("PRAGMA cache_size=-16384")
            db.execute("PRAGMA temp_store=FILE")
            db.execute("PRAGMA journal_mode=OFF")  # disposable, recomputed on failure
            db.execute("""CREATE TABLE edges (
                s TEXT, sk TEXT, t TEXT, tk TEXT, score REAL, relation TEXT,
                PRIMARY KEY(s,sk,t,tk)) WITHOUT ROWID""")
            db.executemany(
                """INSERT INTO edges VALUES(?,?,?,?,?,?)
                ON CONFLICT(s,sk,t,tk) DO UPDATE SET
                    score=excluded.score, relation=excluded.relation
                WHERE excluded.score > edges.score OR
                    (excluded.score=edges.score AND excluded.relation < edges.relation)""",
                ((str(m.head), _kind_text(m.src_kind), str(m.tail), _kind_text(m.tgt_kind),
                  float(m.score), str(m.relation)) for m in mappings),
            )
            db.execute("CREATE TABLE protected(s TEXT,t TEXT,PRIMARY KEY(s,t)) WITHOUT ROWID")
            db.executemany("INSERT OR IGNORE INTO protected VALUES(?,?)", sorted(protected_pairs))
            protected = set((str(s), str(t)) for s, t in protected_pairs)

            def materialize(query):
                return [EntityMapping(s, t, relation=rel, score=score, src_kind=sk, tgt_kind=tk)
                        for s, sk, t, tk, score, rel in db.execute(query)]

            anchors = materialize("SELECT e.* FROM edges e JOIN protected p USING(s,t)")
            source_conflicts, target_conflicts, conflicts = _protected_exact_conflicts(
                anchors, protected
            )
            if anchor_conflict_policy == "error":
                _validate_protected_exact_constraints(source_conflicts, target_conflicts)
            else:
                db.executemany("DELETE FROM protected WHERE s=? AND t=?", sorted(conflicts))
            db.execute("CREATE TABLE anchors AS SELECT e.* FROM edges e JOIN protected p USING(s,t)")
            db.execute("CREATE INDEX anchor_source ON anchors(s,sk)")
            db.execute("CREATE INDEX anchor_target ON anchors(t,tk)")
            residual_condition = "NOT EXISTS(SELECT 1 FROM protected p WHERE e.s=p.s AND e.t=p.t)"
            if mode != "threshold":
                residual_condition += """ AND NOT EXISTS(
                    SELECT 1 FROM anchors a WHERE a.s=e.s AND a.sk=e.sk)
                    AND NOT EXISTS(SELECT 1 FROM anchors a WHERE a.t=e.t AND a.tk=e.tk)"""
            db.execute(f"CREATE TABLE residual AS SELECT * FROM edges e WHERE {residual_condition}")
            threshold_removed = 0
            if threshold is not None:
                threshold_removed = db.execute(
                    "SELECT count(*) FROM residual WHERE score < ?", (float(threshold),)
                ).fetchone()[0]
                db.execute("DELETE FROM residual WHERE score < ?", (float(threshold),))

            if mode == "threshold":
                query = "SELECT * FROM residual"
            elif mode == "mutual_best":
                # Typed nodes use (IRI, kind), exactly as the in-memory strategy.
                query = """SELECT s,sk,t,tk,score,relation FROM (
                    SELECT *, row_number() OVER (
                        PARTITION BY s,sk ORDER BY score DESC,t,tk,relation) AS sr,
                    row_number() OVER (
                        PARTITION BY t,tk ORDER BY score DESC,s,sk,relation) AS tr
                    FROM residual) WHERE sr=1 AND tr=1"""
            else:
                # Greedy is source top-n followed by target top-n. The trailing
                # columns reproduce Python's stable ties from _deduplicate order.
                source_query = "SELECT * FROM residual"
                if source_cardinality is not None:
                    source_query = f"""SELECT s,sk,t,tk,score,relation FROM (
                        SELECT *,row_number() OVER (
                            PARTITION BY s,sk ORDER BY score DESC,t,tk,relation) AS sr
                        FROM residual) WHERE sr <= {max(1, int(source_cardinality))}"""
                db.execute(f"CREATE TABLE source_selected AS {source_query}")
                query = "SELECT * FROM source_selected"
                if target_cardinality is not None:
                    query = f"""SELECT s,sk,t,tk,score,relation FROM (
                        SELECT *,row_number() OVER (
                            PARTITION BY t,tk ORDER BY score DESC,s,sk,relation) AS tr
                        FROM source_selected) WHERE tr <= {max(1, int(target_cardinality))}"""
            db.execute(f"CREATE TABLE selected AS {query}")
            db.execute("INSERT INTO selected SELECT * FROM anchors")
            output = materialize("SELECT * FROM selected ORDER BY sk,s,tk,t,score DESC,relation")
            selected_pairs = {(str(m.head), str(m.tail)) for m in output}
            diagnostics = {
                "mode": mode, "anchor_conflict_policy": anchor_conflict_policy,
                "source_cardinality": source_cardinality, "target_cardinality": target_cardinality,
                "anchor_source_conflicts": [
                    {"source": list(s), "targets": [list(t) for t in targets]}
                    for s, targets in sorted(source_conflicts.items())],
                "anchor_target_conflicts": [
                    {"target": list(t), "sources": [list(s) for s in sources]}
                    for t, sources in sorted(target_conflicts.items())],
                "conflicting_anchor_pairs": [list(p) for p in sorted(conflicts)],
                "selected_conflicting_anchor_pairs": [list(p) for p in sorted(conflicts & selected_pairs)],
                "suppressed_conflicting_anchor_pairs": [list(p) for p in sorted(conflicts - selected_pairs)],
                "input_mappings": len(mappings),
                "deduplicated_mappings": db.execute("SELECT count(*) FROM edges").fetchone()[0],
                "protected_mappings": db.execute("SELECT count(*) FROM anchors").fetchone()[0],
                "selected_mappings": len(output), "threshold": threshold,
                "threshold_removed": threshold_removed, "assignment_components": 0,
                "assignment_fallback_components": 0, "assignment_component_cap": int(assignment_component_cap),
                "assignment_objective": None, "assignment_fallback": "threshold_first_greedy",
                "unmatched_sources": db.execute("""SELECT count(*) FROM (
                    SELECT s,sk FROM edges EXCEPT SELECT s,sk FROM selected)""").fetchone()[0],
            }
            return ExtractionResult(mappings=output, diagnostics=diagnostics)
        finally:
            db.close()
