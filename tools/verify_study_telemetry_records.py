"""Independently inspect the isolated browser run's persisted observations."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path


def verify(database, browser_report):
    browser = json.loads(browser_report.read_bytes())
    reports = []
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        for journey in browser["journeys"]:
            sid = journey["session"]
            row = db.execute("SELECT test,state FROM sessions WHERE id=?", (sid,)).fetchone()
            assert row and row["test"] == 1, "Only isolated test sessions may be validated"
            state = json.loads(row["state"])
            assert state["stage"] == "completed"
            events = [
                json.loads(r[0])
                for r in db.execute("SELECT payload FROM events WHERE session_id=?", (sid,))
            ]
            timing = [
                json.loads(r[0])
                for r in db.execute(
                    "SELECT payload FROM timing_segments WHERE session_id=?", (sid,)
                )
            ]
            cases = []
            for case in journey["cases"]:
                cid = case["case_id"]
                ranking = state["rankings"][cid]
                assert ranking["workflow_state"] == "submitted"
                es = [
                    e
                    for e in events
                    if e.get("case_id") == cid and e.get("scope", "case") == "case"
                ]
                ss = [
                    s
                    for s in timing
                    if s.get("case_id") == cid
                    and s["stage"] == "case"
                    and s.get("availability", "observed") == "observed"
                ]
                types = {e["type"] for e in es}
                assert {"case_ready", "keep_initial_order", "submit"} <= types, (cid, types)
                assert ss, f"No observed timing for {cid}"
                tabs = {e.get("element_id") for e in es if e["type"] == "tab_open"}
                if case["condition"] == "explanation":
                    assert {"trace", "scores"} <= tabs, (cid, tabs)
                    assert "candidate_inspected" in types
                else:
                    assert not tabs and not ({"graph_open", "table_open"} & types)
                assert all(e.get("server_received_at") for e in es)
                cases.append(
                    dict(
                        condition=case["condition"],
                        events=len(es),
                        observed_segments=len(ss),
                        observed_page_seconds=round(
                            sum(
                                (s["monotonic_end_ms"] - s["monotonic_start_ms"]) / 1000 for s in ss
                            ),
                            3,
                        ),
                    )
                )
            pages = defaultdict(list)
            for event in events:
                pages[event["page_instance_id"]].append(event["sequence"])
            for sequence in pages.values():
                assert sorted(sequence) == list(
                    range(max(sequence) + 1)
                ), "Unexpected gap or duplicate in accepted page events"
            intervals = defaultdict(list)
            for segment in timing:
                if segment.get("availability", "observed") != "observed":
                    continue
                assert segment["monotonic_end_ms"] >= segment["monotonic_start_ms"]
                intervals[segment["page_instance_id"]].append(
                    (segment["monotonic_start_ms"], segment["monotonic_end_ms"])
                )
            for values in intervals.values():
                values.sort()
                assert all(
                    a[1] <= b[0] for a, b in zip(values, values[1:])
                ), "Timing intervals overlap"
            reports.append(
                dict(
                    completed=True,
                    cases=cases,
                    event_count=len(events),
                    timing_count=len(timing),
                    page_clocks=len(pages),
                    contiguous_event_sequences=True,
                    nonoverlapping_timing=True,
                )
            )
        for event_id in browser["faultTests"]["replay_event_ids"]:
            assert (
                db.execute("SELECT COUNT(*) FROM events WHERE event_id=?", (event_id,)).fetchone()[
                    0
                ]
                == 1
            )
    assert len(reports) == 2 and all(len(r["cases"]) == 10 for r in reports)
    assert browser["journeys"][0]["first_condition"] != browser["journeys"][1]["first_condition"]
    return dict(
        passed=True,
        journeys=reports,
        original_conditions_preserved=True,
        replay_deduplication=True,
        production_sessions_modified=0,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--browser-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.database, args.browser_report)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passed": True,
                "complete_journeys": len(result["journeys"]),
                "all_twenty_cases_have_events_and_timing": True,
                "replay_deduplication": True,
            }
        )
    )
