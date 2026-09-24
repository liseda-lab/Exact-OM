"""Deterministic CSV analysis archive derived from an immutable JSON export."""

from __future__ import annotations

import csv
import hashlib
import io
import zipfile

from .store import canonical


def _cell(value):
    if isinstance(value, (dict, list)):
        return canonical(value)
    if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r")):
        return "'" + value
    return value


def csv_archive(export):
    """Include tables, schemas and per-file checksums without exporting credentials."""
    sessions, cases, questionnaires, events, segments = [], [], [], [], []
    for session in export["data"]["sessions"]:
        sid = session["session_id"]
        sessions.append(
            {
                "session_id": sid,
                "test": session["test"],
                "stage": session["stage"],
                "assignment": session["assignment"],
                "consent": session["consent"],
                "setup": session["setup"],
            }
        )
        for row in session["cases"]:
            cases.append({"session_id": sid, **row})
        for form_id, form in session["questionnaires"].items():
            questionnaires.append({"session_id": sid, "form_id": form_id, **form})
        events.extend({"session_id": sid, **event} for event in session["events"])
        segments.extend({"session_id": sid, **segment} for segment in session["timing_segments"])
    tables = {
        "sessions.csv": sessions,
        "cases.csv": cases,
        "questionnaires.csv": questionnaires,
        "events.csv": events,
        "timing_segments.csv": segments,
        "summary.csv": [export["data"]["summary"]],
    }
    files, columns = {}, {}
    for name, rows in tables.items():
        fields = sorted({key for row in rows for key in row})
        columns[name] = fields
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: _cell(value) for key, value in row.items()} for row in rows)
        files[name] = stream.getvalue().encode("utf-8")
    files["data-dictionary.json"] = canonical(
        {
            "columns": columns,
            "semantics": export["data"]["data_dictionary"],
            "nested_cells": "Nested objects and lists use canonical JSON; absent scalars are empty cells",
            "spreadsheet_safety": "Formula-leading text is prefixed with an apostrophe",
            "questionnaire_definitions": export["data"]["questionnaire_definitions"],
        }
    ).encode()
    if "researcher_case_keys" in export["data"]:
        files["researcher-case-keys.json"] = canonical(
            export["data"]["researcher_case_keys"]
        ).encode()
    files["manifest.json"] = canonical(
        {
            **export["manifest"],
            "archive_schema": "exact-study-csv/1",
            "files": {
                name: {"sha256": hashlib.sha256(content).hexdigest(), "size_bytes": len(content)}
                for name, content in files.items()
            },
        }
    ).encode()
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        for name in sorted(files):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            zipped.writestr(info, files[name])
    return archive.getvalue()
