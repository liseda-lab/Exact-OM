"""Operator-only reassignment of untouched invitations; participant secrets never change."""

import json


def rebind_unstarted(store, old_revision, new_revision, expected_ids):
    """Atomically move precisely the audited, untouched real invitations, or change nothing."""
    if old_revision == new_revision or not expected_ids:
        raise ValueError("Distinct revisions and audited session identifiers are required")
    with store.transaction() as db:
        old_row, old = store._study(db, old_revision, lock=True)
        new_row, new = store._study(db, new_revision, lock=True)
        if new_row["closed"]:
            raise ValueError("Destination study is closed")
        for key in old:
            if (
                key
                not in {
                    "study_revision",
                    "software_version",
                    "form_version",
                    "components",
                    "workspace_scopes",
                    "questionnaires",
                    "protocol_versions",
                }
                and old[key] != new[key]
            ):
                raise ValueError("Study protocol changed beyond the approved panels")
        if {k: v for k, v in old["protocol_versions"].items() if k != "forms"} != {
            k: v for k, v in new["protocol_versions"].items() if k != "forms"
        }:
            raise ValueError("Protocol changed beyond the new form version")
        rows = db.lock(
            "SELECT * FROM sessions WHERE study_revision=? AND revoked=0 AND test=0",
            (old_revision,),
        ).fetchall()
        if {r["id"] for r in rows} != set(expected_ids):
            raise ValueError("Active invitations changed since the preflight audit")
        for row in rows:
            state = json.loads(row["state"])
            if (
                row["revision"] != 0
                or state.get("revision", 0) != 0
                or state.get("stage") != "welcome"
                or state.get("consent") is not None
                or state.get("assignment") is not None
            ):
                raise ValueError("A participant has started; invitations were not changed")
            for key in (
                "setup",
                "questionnaires",
                "protocol_versions",
                "rankings",
                "consultations",
            ):
                if state.get(key):
                    raise ValueError("Session already contains participant answers")
            if db.execute(
                "SELECT COUNT(*) AS n FROM mutations WHERE session_id=?", (row["id"],)
            ).fetchone()["n"]:
                raise ValueError("Session contains a saved mutation")
        for row in rows:
            db.execute("UPDATE sessions SET study_revision=? WHERE id=?", (new_revision, row["id"]))
            after = db.execute("SELECT * FROM sessions WHERE id=?", (row["id"],)).fetchone()
            assert all(after[k] == row[k] for k in row.keys() if k != "study_revision")
        return len(rows)
