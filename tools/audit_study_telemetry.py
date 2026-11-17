"""Read-only telemetry health report. No answers, invitation secrets or session IDs are printed."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
from collections import Counter
from pathlib import Path


def read_environment(path):
    values = {}
    for line in path.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, raw = line.split("=", 1)
            parts = shlex.split(raw)
            values[key] = parts[0] if parts else ""
    return values


def audit(database_url, revision):
    import psycopg
    from psycopg.rows import dict_row

    with psycopg.connect(database_url, row_factory=dict_row) as db:
        db.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
        now = db.execute("SELECT now() AS value").fetchone()["value"]
        rows = db.execute(
            "SELECT id,state FROM sessions WHERE study_revision=%s AND test=0 AND revoked=0 ORDER BY id",
            (revision,),
        ).fetchall()
        sessions = []
        for row in rows:
            state = json.loads(row["state"])
            events = [
                json.loads(r["payload"])
                for r in db.execute(
                    "SELECT payload FROM events WHERE session_id=%s ORDER BY received_at",
                    (row["id"],),
                ).fetchall()
            ]
            segments = [
                json.loads(r["payload"])
                for r in db.execute(
                    "SELECT payload FROM timing_segments WHERE session_id=%s ORDER BY received_at",
                    (row["id"],),
                ).fetchall()
            ]
            cases, warnings = [], []
            for position, presentation in enumerate(
                (state.get("assignment") or {}).get("presentations", []), 1
            ):
                cid, pid = presentation["case_id"], presentation["presentation_id"]
                ranking = (state.get("rankings") or {}).get(cid) or {}
                es = [e for e in events if e.get("presentation_id") == pid]
                ss = [
                    s for s in segments if s.get("presentation_id") == pid and s["stage"] == "case"
                ]
                ready = [e for e in es if e["type"] == "case_ready"]
                observed = [s for s in ss if s.get("availability", "observed") == "observed"]
                submitted = ranking.get("workflow_state") == "submitted"
                if submitted and not ready:
                    warnings.append(f"Case {position}: submitted without an accepted ready event")
                if submitted and not observed:
                    warnings.append(f"Case {position}: submitted without observed timing")
                cases.append(
                    dict(
                        position=position,
                        condition=presentation["condition"],
                        submitted=submitted,
                        submitted_at=ranking.get("submitted_at"),
                        events=len(es),
                        ready_events=len(ready),
                        timing_segments=len(ss),
                        observed_page_seconds=round(
                            sum(
                                (s["monotonic_end_ms"] - s["monotonic_start_ms"]) / 1000
                                for s in observed
                            ),
                            3,
                        ),
                        last_event=max((e["server_received_at"] for e in es), default=None),
                        last_timing=max((s["server_received_at"] for s in ss), default=None),
                    )
                )
            sessions.append(
                dict(
                    audit_label=hashlib.sha256(row["id"].encode()).hexdigest()[:12],
                    stage=state["stage"],
                    completed_cases=sum(c["submitted"] for c in cases),
                    events=len(events),
                    event_types=dict(Counter(e["type"] for e in events)),
                    timing_segments=len(segments),
                    last_event=max((e["server_received_at"] for e in events), default=None),
                    last_timing=max((s["server_received_at"] for s in segments), default=None),
                    warnings=warnings,
                    cases=cases,
                )
            )
        return dict(
            checked_at=now.isoformat(),
            study_revision=revision,
            read_only=True,
            session_stages=dict(Counter(s["stage"] for s in sessions)),
            sessions=sessions,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--require-idle",
        action="store_true",
        help="Fail if any real participant has started but has not completed or closed the study",
    )
    args = parser.parse_args()
    report = audit(read_environment(args.environment)["EXACT_STUDY_DATABASE_URL"], args.revision)
    print(json.dumps(report, indent=2))
    if args.require_idle and any(
        s["stage"] not in {"welcome", "completed", "closed"} for s in report["sessions"]
    ):
        raise SystemExit("A real participant is still in progress. Activation is not allowed.")
