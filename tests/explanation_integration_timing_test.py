"""B13/R04 observations retain page-clock uncertainty and frozen export versions."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import subprocess
import sys
import zipfile
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from exact_inspect.study import StudyError, StudyStore
from exact_inspect.study.administration import revisions
from exact_inspect.study.api import create_study_app
from exact_inspect.study.exports import _cell, csv_archive
from exact_inspect.study.models import Publish, Ranking
from exact_inspect.study.store import canonical, digest
from exact_inspect.study.telemetry import (
    EventBatchV2,
    TimingSegmentV2,
    save_events,
    save_timing,
)
from exact_inspect.study.v2_models import ConsultationDraftV2
from tests.explanation_study_test import ADMIN, ORIGIN, SECRET, publication
from tests.explanation_study_v2_test import prepare, train
from tools.build_explanation_v2_fixture import publication_v2

SCHEMA2 = "exact-study-analysis/2"
SCHEMA3 = "exact-study-analysis/3"


@pytest.fixture(scope="module")
def frozen_package(tmp_path_factory):
    directory = tmp_path_factory.mktemp("timing-integration-resources")
    return directory, publication_v2(directory)


@pytest.fixture
def study(tmp_path, frozen_package):
    directory, template = frozen_package
    frozen = template.model_copy(deep=True)
    frozen.definition.study_revision = "timing-integration-" + uuid4().hex
    url = os.environ.get("EXACT_STUDY_TEST_DATABASE_URL", f"sqlite:///{tmp_path / 'study.sqlite'}")
    store = StudyStore(url, directory, allow_test_sqlite=True)
    store.publish(frozen)
    invitation = store.issue(frozen.definition.study_revision)["invitations"][0]
    sid, generation, _ = store.exchange(invitation["invitation"].split("=", 1)[1])
    return store, frozen, (sid, generation)


def segment(store, identity, *, stage, page, seconds, start=0, case=None):
    body = TimingSegmentV2(
        segment_id=uuid4().hex,
        page_instance_id=page,
        stage=stage,
        case_id=case["case_id"] if case else None,
        presentation_id=case["presentation_id"] if case else None,
        monotonic_start_ms=start * 1000,
        monotonic_end_ms=(start + seconds) * 1000,
    )
    return body, save_timing(store, *identity, body)


def assert_summary(summary, pages):
    assert summary == {
        "page_observation_seconds": sum(pages.values()),
        "per_page_observation_seconds": [
            {"page_instance_id": page, "seconds": seconds}
            for page, seconds in sorted(pages.items())
        ],
        "page_instance_count": len(pages),
        "coverage_status": "not_established",
        "coverage_reason": (
            "no_eligible_observations"
            if not pages
            else "single_page_clock_unmapped" if len(pages) == 1 else "multiple_page_clocks"
        ),
        "unique_elapsed_coverage_seconds": None,
        "unobserved_elapsed_seconds": None,
        "active_duration_known": False,
    }


def assert_csv_matches(export):
    archive = csv_archive(export)
    assert archive == csv_archive(export)
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        manifest = json.loads(zipped.read("manifest.json"))
        assert manifest["archive_schema"] == "exact-study-csv/3"
        assert manifest["source_protocol_versions"] == export["data"]["protocol_versions"]
        for name, receipt in manifest["files"].items():
            raw = zipped.read(name)
            assert hashlib.sha256(raw).hexdigest() == receipt["sha256"]
            assert len(raw) == receipt["size_bytes"]

        def rows(name):
            return list(csv.DictReader(io.StringIO(zipped.read(name).decode())))

        session = export["data"]["sessions"][0]
        assert json.loads(rows("sessions.csv")[0]["tutorial_timing"]) == session["tutorial_timing"]
        for row, source in zip(rows("cases.csv"), session["cases"], strict=True):
            for field in ("timing", "consultation_timing", "score"):
                assert json.loads(row[field]) == source[field]
        attempts = rows("tutorial_attempts.csv")
        assert len(attempts) == len(session["tutorial_progress"]["attempts"]) == 5
        for row in rows("tutorial_outcomes.csv"):
            assert json.loads(row["tutorial_timing"]) == session["tutorial_timing"]
            for field in ("first", "final"):
                assert (
                    json.loads(row[field])
                    == session["tutorial_outcomes"][row["question_id"]][field]
                )
        assert len(rows("consultation_drafts.csv")) == len(session["consultation_drafts"])
        raw_segments = rows("timing_segments.csv")
        assert len(raw_segments) == len(session["timing_segments"])
        assert sum(s["availability"] == "unavailable" for s in raw_segments) == sum(
            s["availability"] == "unavailable" for s in session["timing_segments"]
        )
        dictionary = json.loads(zipped.read("data-dictionary.json"))
        assert dictionary["semantics"] == export["data"]["data_dictionary"]
        assert "empty" in dictionary["semantics"]["csv_nulls"]
        assert "null" in rows("cases.csv")[0]["timing"]
    scalar_csv = io.StringIO()
    writer = csv.writer(scalar_csv, lineterminator="\n")
    writer.writerow([_cell(None), _cell("=unsafe")])
    assert scalar_csv.getvalue() == ",'=unsafe\n"
    return archive


@pytest.mark.parametrize(
    "durations,submitted",
    [([60], True), ([60, 60], True), ([20, 30], True), ([], True), ([60], False)],
)
def test_observation_oracles_survive_reload_retry_and_restart(
    study, monkeypatch, durations, submitted
):
    store, frozen, identity = study
    prepare(store, frozen, identity)
    tutorial_pages = {}
    for index, seconds in enumerate(durations):
        page = f"tutorial-{index}"
        body, receipt = segment(store, identity, stage="tutorial", page=page, seconds=seconds)
        assert receipt["availability"] == "observed"
        assert save_timing(store, *identity, body) == receipt
        tutorial_pages[page] = seconds
    train(store, frozen, identity)
    # A queued observation after tutorial completion remains raw unavailable evidence.
    _, receipt = segment(store, identity, stage="tutorial", page="tutorial-late", seconds=20)
    assert receipt["availability"] == "unavailable"
    case = store.current_case(*identity)
    first = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def ready(page, sequence, at):
        monkeypatch.setattr(
            "exact_inspect.study.telemetry.utcnow",
            lambda: (first + timedelta(seconds=at)).isoformat(),
        )
        batch = EventBatchV2(
            events=[
                dict(
                    event_id=uuid4().hex,
                    page_instance_id=page,
                    sequence=sequence,
                    case_id=case["case_id"],
                    presentation_id=case["presentation_id"],
                    type="case_ready",
                    client_monotonic_ms=0,
                    build_version=frozen.definition.software_version,
                )
            ]
        )
        receipt = save_events(store, *identity, batch)
        assert save_events(store, *identity, batch) == receipt

    ready("case-0", 0, 0)
    case_pages = {}
    original = None
    for index, seconds in enumerate(durations):
        page = f"case-{index}"
        if index:
            ready(page, 0, 20)
        body, receipt = segment(
            store, identity, stage="case", page=page, seconds=seconds, case=case
        )
        assert receipt["availability"] == "observed"
        assert save_timing(store, *identity, body) == receipt
        original = body, receipt
        case_pages[page] = seconds
    # Navigation/reconnect readiness from the same page cannot restart elapsed time.
    ready("case-0", 1, 30)
    # Pre-ready page segment is excluded and its reason is preserved.
    _, receipt = segment(
        store, identity, stage="case", page="case-not-ready", seconds=20, case=case
    )
    assert receipt["reason"] == "case_not_ready"
    consultation_pages = {}
    if submitted:
        monkeypatch.setattr(
            "exact_inspect.study.store.utcnow", lambda: (first + timedelta(seconds=100)).isoformat()
        )
        state = store.state(*identity)
        store.mutate(
            *identity,
            "submit",
            Ranking(
                idempotency_key=uuid4().hex,
                expected_revision=state["revision"],
                presentation_id=case["presentation_id"],
                response_type="none_of_these",
            ),
            case["case_id"],
        )
        for index, seconds in enumerate(durations):
            page = f"consultation-{index}"
            body, receipt = segment(
                store, identity, stage="consultation", page=page, seconds=seconds, case=case
            )
            assert receipt["availability"] == "observed"
            assert save_timing(store, *identity, body) == receipt
            consultation_pages[page] = seconds
        state = store.state(*identity)
        store.mutate(
            *identity,
            "consultation_draft",
            ConsultationDraftV2(
                idempotency_key=uuid4().hex,
                expected_revision=state["revision"],
                presentation_id=case["presentation_id"],
                form_version="exact-study-forms/2",
                consulted_external_ontologies=False,
            ),
            case["case_id"],
        )
        _, late = segment(
            store, identity, stage="case", page="case-0", start=60, seconds=20, case=case
        )
        assert late["reason"] == "stage_no_longer_current"
        if original:
            assert save_timing(store, *identity, original[0]) == original[1]
    legacy = store.export(frozen.definition.study_revision, include_test=True)
    legacy_bytes = canonical(legacy)
    legacy_csv = csv_archive(legacy)
    exported = store.export(
        frozen.definition.study_revision, include_test=True, analysis_schema=SCHEMA3
    )
    session = exported["data"]["sessions"][0]
    output_case = session["cases"][0]
    assert_summary(session["tutorial_timing"], tutorial_pages)
    assert_summary(output_case["consultation_timing"], consultation_pages)
    timing = output_case["timing"]
    assert_summary({k: timing[k] for k in session["tutorial_timing"]}, case_pages)
    assert timing["raw_elapsed_seconds"] == (100 if submitted else None)
    assert timing["observed_segment_seconds"] == sum(durations)
    assert timing["consultation_observed_seconds"] == sum(consultation_pages.values())
    assert session["tutorial_observed_seconds"] == sum(durations)
    assert exported["data"]["summary"] == legacy["data"]["summary"]
    assert (
        exported["manifest"]["source_protocol_versions"]
        == frozen.definition.protocol_versions.model_dump()
    )
    assert exported["data"]["protocol_versions"]["export"] == SCHEMA2
    assert digest(canonical(exported["data"])) == exported["manifest"]["content_sha256"]
    archive = assert_csv_matches(exported)
    # A fresh store process reads exactly the same frozen exports, never recomputes them.
    script = "from exact_inspect.study import StudyStore; from exact_inspect.study.store import canonical; import sys; s=StudyStore(sys.argv[1], sys.argv[2],allow_test_sqlite=True); print(canonical(s.saved_export(sys.argv[3])))"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            store.database_url,
            str(store.assets_dir),
            exported["manifest"]["export_id"],
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert canonical(json.loads(result.stdout)) == canonical(exported)
    reopened = StudyStore(store.database_url, store.assets_dir, allow_test_sqlite=True)
    assert csv_archive(reopened.saved_export(exported["manifest"]["export_id"])) == archive
    assert canonical(reopened.saved_export(legacy["manifest"]["export_id"])) == legacy_bytes
    assert csv_archive(reopened.saved_export(legacy["manifest"]["export_id"])) == legacy_csv
    if durations == [60, 60]:
        assert (
            legacy["data"]["sessions"][0]["cases"][0]["timing"]["unobserved_elapsed_seconds"] == 0
        )
        assert timing["page_observation_seconds"] == 120 > timing["raw_elapsed_seconds"]
        assert timing["unobserved_elapsed_seconds"] is None


def test_export_selector_preserves_publication_defaults_and_saved_artifacts(study, tmp_path):
    store, old, _ = study
    raw = old.model_dump(mode="json")
    raw["definition"]["protocol_versions"].pop("export")
    parsed = Publish.model_validate(raw)
    assert parsed.model_dump(mode="json") == old.model_dump(mode="json")
    before = store.publish(old)
    assert store.publish(parsed) == before
    corrected = old.model_copy(deep=True)
    corrected.definition.study_revision = "timing-corrected-" + uuid4().hex
    corrected.definition.protocol_versions.export = SCHEMA3
    store.publish(corrected)
    legacy = publication(tmp_path)
    StudyStore(store.database_url, tmp_path, allow_test_sqlite=True).publish(legacy)
    app = create_study_app(
        store.database_url, SECRET, ADMIN, ORIGIN, store.assets_dir, allow_test_sqlite=True
    )
    client = TestClient(app, base_url=ORIGIN, headers={"Authorization": f"Bearer {ADMIN}"})
    for frozen, default, allowed in [
        (old, SCHEMA2, {SCHEMA2, SCHEMA3}),
        (corrected, SCHEMA3, {SCHEMA2, SCHEMA3}),
        (legacy, "exact-study-analysis/1", {"exact-study-analysis/1"}),
    ]:
        route = f"/api/v1/admin/studies/{frozen.definition.study_revision}/exports"
        default_result = client.post(route)
        assert default_result.status_code == 200, default_result.text
        assert default_result.json()["manifest"]["schema"] == default
        for schema in ("exact-study-analysis/1", SCHEMA2, SCHEMA3, "exact-study-analysis/4", ""):
            result = client.post(route, params={"analysis_schema": schema})
            assert result.status_code == (200 if schema in allowed else 422), result.text
            if schema in allowed:
                exported = result.json()
                saved_route = "/api/v1/admin/exports/" + exported["manifest"]["export_id"]
                assert client.get(saved_route).json() == exported
                assert client.get(saved_route, params={"format": "csv"}).content == csv_archive(
                    exported
                )
        direct = client.post(route, params={"analysis_schema": default, "format": "csv"})
        assert direct.status_code == 200
        with zipfile.ZipFile(io.BytesIO(direct.content)) as zipped:
            assert json.loads(zipped.read("manifest.json"))["schema"] == default
        with pytest.raises(StudyError, match="unsupported"):
            store.export(frozen.definition.study_revision, analysis_schema="arbitrary")
    metadata = {x["study_revision"]: x for x in revisions(store)["items"]}
    assert metadata[old.definition.study_revision]["source_export_version"] == SCHEMA2
    assert metadata[old.definition.study_revision]["supported_analysis_schemas"] == [
        SCHEMA2,
        SCHEMA3,
    ]
    assert metadata[corrected.definition.study_revision]["source_export_version"] == SCHEMA3
    assert store.publish(old) == before


@pytest.mark.parametrize("selected", [None, SCHEMA3])
def test_researcher_cli_export_selector_is_explicit(tmp_path, monkeypatch, selected):
    from urllib.parse import parse_qs, urlsplit

    from exact_inspect.study.__main__ import main

    output = tmp_path / "export.json"
    args = ["exact-study", "export", "--study", "synthetic", "--output", str(output)]
    if selected is not None:
        args.extend(["--analysis-schema", selected])
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setenv("EXACT_STUDY_ORIGIN", ORIGIN)
    monkeypatch.setenv("EXACT_STUDY_RESEARCHER_TOKEN", ADMIN)
    calls = []

    def request(req, **kwargs):
        calls.append(req)
        return io.BytesIO(b'{"synthetic": true}')

    monkeypatch.setattr("urllib.request.urlopen", request)
    main()
    query = parse_qs(urlsplit(calls[0].full_url).query)
    assert query.get("analysis_schema") == ([selected] if selected else None)
    assert json.loads(output.read_text()) == {"synthetic": True}
    assert output.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("selected", [None, SCHEMA2])
def test_new_fixture_cli_freezes_export_three_explicitly(
    tmp_path, monkeypatch, frozen_package, selected
):
    from tools import build_explanation_v2_fixture as builder

    _, template = frozen_package
    args = ["build-explanation-v2-fixture", str(tmp_path)]
    if selected is not None:
        args.extend(["--export-version", selected])
    monkeypatch.setattr(sys, "argv", args)

    def build(root, *, export_version, **kwargs):
        result = template.model_copy(deep=True)
        result.definition.protocol_versions.export = export_version
        return result

    monkeypatch.setattr(builder, "publication_v2", build)
    builder.main()
    written = Publish.model_validate_json((tmp_path / "publication.json").read_text())
    assert written.definition.protocol_versions.export == (selected or SCHEMA3)
