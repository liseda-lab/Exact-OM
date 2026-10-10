"""Deterministic CSV analysis archive derived from an immutable JSON export."""

from __future__ import annotations

import csv
import hashlib
import io
import math
import zipfile

from .store import canonical


def observation_summary(segments, *, stages, case_id=None):
    """Sum accepted page-local observations without claiming elapsed coverage.

    The store admits bounded non-overlapping intervals and deduplicates segment IDs.
    The page identifier orders output only: offsets from different clocks are never
    compared, merged or mapped to receipt timestamps.
    """
    pages = {}
    for segment in segments:
        if (
            segment["stage"] not in stages
            or segment.get("case_id") != case_id
            or segment.get("availability") == "unavailable"
        ):
            continue
        pages.setdefault(segment["page_instance_id"], []).append(
            (segment["monotonic_end_ms"] - segment["monotonic_start_ms"]) / 1000
        )
    per_page = [
        {"page_instance_id": page, "seconds": math.fsum(pages[page])} for page in sorted(pages)
    ]
    count = len(per_page)
    return {
        "page_observation_seconds": math.fsum(item["seconds"] for item in per_page),
        "per_page_observation_seconds": per_page,
        "page_instance_count": count,
        "coverage_status": "not_established",
        "coverage_reason": (
            "no_eligible_observations"
            if count == 0
            else "single_page_clock_unmapped" if count == 1 else "multiple_page_clocks"
        ),
        "unique_elapsed_coverage_seconds": None,
        "unobserved_elapsed_seconds": None,
        "active_duration_known": False,
    }


def analysis3_dictionary():
    """Authoritative interpretation for newly derived analysis-3 JSON and CSV."""
    return {
        "raw_elapsed_seconds": "Seconds from the first accepted usable-content server receipt to ranking submission for that presentation, including breaks and gaps; null when either endpoint is missing. Later page-ready events do not restart this interval.",
        "page_observation_seconds": "Page-seconds: sum of accepted bounded, deduplicated, non-overlapping eligible client intervals, grouped by stage/case. Excludes availability=unavailable; zero means no observed page-seconds, never zero elapsed task time. May exceed raw elapsed seconds.",
        "per_page_observation_seconds": "Deterministically ordered page_instance_id and seconds pairs from the same eligible intervals as page_observation_seconds; each monotonic clock is local to one page. Never compare offsets across pages or infer concurrency/disjointness.",
        "page_instance_count": "Count of page clocks with eligible intervals for this stage/case; not a concurrency measure.",
        "coverage_status": "Always not_established: no validated mapping from client monotonic clocks to the server ready/submit interval exists.",
        "coverage_reason": "no_eligible_observations for zero contributing pages, single_page_clock_unmapped for one, multiple_page_clocks for more than one; neither overlap nor disjointness is established.",
        "unique_elapsed_coverage_seconds": "Null means unknown elapsed coverage in seconds, even for one page. It is never the raw page sum or a clamped value.",
        "unobserved_elapsed_seconds": "Null means unknown elapsed time without observations in seconds; never computed by subtracting page-seconds from raw elapsed seconds.",
        "active_duration_known": "Always false: page observations, visibility and receipt timestamps do not establish attention or active task duration; hidden pages are not implicitly paused.",
        "observed_segment_seconds": "Compatibility alias of case timing.page_observation_seconds in analysis/3; page-seconds, not unique elapsed coverage or active duration.",
        "tutorial_observed_seconds": "Compatibility alias of session tutorial_timing.page_observation_seconds; tutorial/practice page-seconds, excluded from scored case time.",
        "consultation_observed_seconds": "Compatibility alias of case consultation_timing.page_observation_seconds; consultation page-seconds, excluded from scored case time.",
        "tutorial_timing": "Observation summary for tutorial/practice stages with no scored case, independently grouped per session. Preparation outcomes do not change ranking-quality denominators.",
        "consultation_timing": "Observation summary for consultation stage of this case only, independent of scored case observations.",
        "unavailable_intervals": "Raw late/pre-ready intervals and their reasons are retained; unavailable intervals never contribute page counts or page-seconds.",
        "protocol_versions": "Frozen source publication protocol versions copied unchanged at data.protocol_versions; the export derivation schema is separate.",
        "source_protocol_versions": "Manifest copy of the unchanged source publication versions, including its frozen export default; source study_revision and content_sha256 bind the derived result.",
        "analysis_schema": "manifest.schema=exact-study-analysis/3 identifies corrected derived semantics; archive_schema=exact-study-csv/3 is its CSV representation. Saved archives retain their original schema and hash.",
        "csv_nulls": "Nested JSON summary values retain null. Top-level nullable scalar CSV cells are empty, meaning unknown/missing rather than zero; dictionaries and JSON are authoritative.",
    }


def _cell(value):
    if isinstance(value, (dict, list)):
        return canonical(value)
    if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r")):
        return "'" + value
    return value


def csv_archive(export):
    """Include tables, schemas and per-file checksums without exporting credentials."""
    schema = export["manifest"]["schema"]
    archive_schemas = {
        "exact-study-analysis/1": "exact-study-csv/1",
        "exact-study-analysis/2": "exact-study-csv/2",
        "exact-study-analysis/3": "exact-study-csv/3",
        "exact-study-analysis/4": "exact-study-csv/4",
    }
    if schema not in archive_schemas:
        raise ValueError("Unsupported saved analysis export schema")
    corrected_timing = schema in {"exact-study-analysis/3", "exact-study-analysis/4"}
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
                **(
                    {
                        "tutorial_timing": session["tutorial_timing"],
                        "tutorial_observed_seconds": session["tutorial_observed_seconds"],
                    }
                    if corrected_timing
                    else {}
                ),
            }
        )
        for row in session["cases"]:
            cases.append({"session_id": sid, **row})
        for form_id, form in sorted(session["questionnaires"].items()):
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
    if schema in {"exact-study-analysis/2", "exact-study-analysis/3", "exact-study-analysis/4"}:
        tables["tutorial_attempts.csv"] = [
            {"session_id": s["session_id"], **attempt}
            for s in export["data"]["sessions"]
            for attempt in s["tutorial_progress"]["attempts"]
        ]
        tables["tutorial_outcomes.csv"] = [
            {
                "session_id": s["session_id"],
                "question_id": q,
                **result,
                "completed_at": s["tutorial_progress"]["completed_at"],
                "tutorial_observed_seconds": s["tutorial_observed_seconds"],
                **({"tutorial_timing": s["tutorial_timing"]} if corrected_timing else {}),
            }
            for s in export["data"]["sessions"]
            for q, result in sorted(s["tutorial_outcomes"].items())
        ]
        tables["consultation_drafts.csv"] = [
            {"session_id": s["session_id"], "case_id": cid, **draft}
            for s in export["data"]["sessions"]
            for cid, draft in sorted(s["consultation_drafts"].items())
        ]
    if schema == "exact-study-analysis/4":
        tables["method_responses.csv"] = export["data"]["method_responses"]
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
            "archive_schema": archive_schemas[schema],
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


def analysis4_dictionary():
    """Restricted method-name derivation, retaining all analysis-3 timing semantics."""
    return {
        "analysis_schema": "manifest.schema=exact-study-analysis/4 and archive_schema=exact-study-csv/4 add study-methods/3 codebooks and explicit named-method opt-in. Analysis-3 page-clock uncertainty, null coverage and quality denominators are unchanged.",
        "method_responses": "One setup-familiarity record per session plus each saved consultation draft/final record, associated with source versions, stable codes, labels, session, case, presentation and assigned condition. Familiarity and practice are never treated as actual case use.",
        "method_codes": "study-methods/3: protege=Protégé; plain_files=Text editor; other_method=Other method; none=None of these in setup familiarity only. plain_files now means text-editor use only; historical direct-file/query/reasoner/viewer answers are not recoded.",
        "experience_codes": "Forms/3 completed-year bands: none=None; less_than_1=Less than 1 year; 1_3=1–3 years; 4_5=4–5 years; 6_9=6–9 years; 10_plus=10+ years. Zero is none; a positive duration below one year is less_than_1. Historical 1_2/3_4/5_9 codes cannot be mapped losslessly and remain unchanged.",
        "method_answer_state": "unanswered means no explicit choice; none_of_these is exclusive stated unfamiliarity; no means explicitly no external consultation for that case; yes means reported external use; answered means one or more selected familiarity codes. Empty optional familiarity is unanswered, not None.",
        "named_detail_state": "hidden means Other is not selected and no current detail is retained; unanswered means selected Other in an incomplete draft with no name; redacted means a recorded name is omitted by the default export; included means explicitly requested restricted named-method detail.",
        "includes_named_method_details": "Only an authenticated researcher may request include_named_methods=true on analysis/4. It includes current selected method names from setup and consultation drafts/finals, never general background/specialty/role/comment free text, hidden previous details, invitation credentials or tutorial action names. Names are absent from the manifest and normal exports.",
        "other_method": "A participant's short plain-text list of method names, not an institutional identifier. Unicode and punctuation are retained; CSV formula-leading scalars are prefixed with an apostrophe. Null in an opted-in record means no current named detail; the named_detail_state supplies its meaning.",
        "source_method_versions": "form_version, setup_version, consultation_version and method_version are copied from the frozen source publication. Labels come from its frozen dictionary, never a reinterpreted historical codebook.",
    }


def method_response_rows(state, study, session_id, *, include_named_methods):
    """Project only explicitly collected methods, with no inference or broad free text."""
    versions = study["protocol_versions"]
    labels = {**study["questionnaires"]["method_dictionary"], "none": "None of these"}
    presentations = {
        p["case_id"]: p for p in (state.get("assignment") or {}).get("presentations", [])
    }
    common = {
        "session_id": session_id,
        "form_version": versions["forms"],
        "setup_version": versions["setup"],
        "consultation_version": versions["consultation"],
        "method_version": versions["methods"],
    }

    def record(stage, value, case_id=None):
        setup = stage == "setup_familiarity"
        methods = value.get("familiar_methods" if setup else "methods", [])
        selected = "other_method" in methods
        name = value.get("familiar_other_method" if setup else "other_method") if selected else None
        if setup:
            answer_state = (
                "unanswered"
                if not methods
                else "none_of_these" if methods == ["none"] else "answered"
            )
        else:
            answer = value.get("consulted_external_ontologies")
            answer_state = "unanswered" if answer is None else "yes" if answer else "no"
        detail_state = (
            "hidden"
            if not selected
            else "unanswered" if not name else "included" if include_named_methods else "redacted"
        )
        presentation = presentations.get(case_id, {})
        return {
            **common,
            "stage": stage,
            "case_id": case_id,
            "presentation_id": presentation.get("presentation_id"),
            "condition": presentation.get("condition"),
            "method_answer_state": answer_state,
            "method_codes": methods,
            "method_labels": [labels[m] for m in methods],
            "named_detail_state": detail_state,
            **({"other_method": name} if include_named_methods else {}),
        }

    rows = [record("setup_familiarity", state.get("setup") or {})]
    for field, stage in (
        ("consultations", "consultation"),
        ("consultation_drafts", "consultation_draft"),
    ):
        rows.extend(
            record(stage, value, case_id) for case_id, value in sorted(state.get(field, {}).items())
        )
    return rows
