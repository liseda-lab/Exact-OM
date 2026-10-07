"""Record synthetic HTTP examples for the S1 backend handoff, never human data.

Requires the development test dependencies. Setup/training uses the backend test
helpers; this is deliberately not evidence of the pending browser acceptance.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient

from exact_inspect.study.api import create_study_app
from exact_inspect.study.models import Publish
from tests.explanation_corrective_http_test import mutate
from tests.explanation_study_test import ADMIN, ORIGIN, SECRET, invite
from tests.explanation_study_v2_test import prepare, train


def record(fixture, output):
    publication_path = fixture / "publication.json"
    publication = Publish.model_validate_json(publication_path.read_text())
    if not publication.definition.synthetic:
        raise ValueError("Only explicit synthetic publications may be recorded")
    examples = {}

    def capture(name, response, *, fields=None, request=None, expected=200):
        assert response.status_code == expected, response.text
        body = response.json()
        examples[name] = {
            "method": response.request.method,
            "path": response.request.url.raw_path.decode(),
            "status": response.status_code,
            "response_projection": fields or "entire body",
            "body": {k: body[k] for k in fields} if fields else body,
        }
        if request is not None:
            examples[name]["request"] = request
        return body

    with tempfile.TemporaryDirectory(prefix="exact-integration-examples-") as temporary:
        app = create_study_app(
            f"sqlite:///{temporary}/study.sqlite",
            SECRET,
            ADMIN,
            ORIGIN,
            fixture,
            allow_test_sqlite=True,
        )
        store = app.state.study_store
        published = store.publish(publication)
        client = TestClient(app, base_url=ORIGIN, headers={"Origin": ORIGIN})
        state = client.post(
            "/api/v1/study/session", json={"secret": invite(store, publication)}
        ).json()
        sid = state["session_id"]
        client.headers["X-Study-Session"] = sid
        prepare(store, publication, (sid, 1))
        state = client.get("/api/v1/study/state").json()
        for name, position in (
            (
                "lesson_position",
                {
                    "view": "lesson",
                    "lesson_id": state["tutorial"]["lessons"][1]["lesson_id"],
                    "question_id": None,
                },
            ),
            (
                "assessment_position",
                {
                    "view": "assessment",
                    "lesson_id": None,
                    "question_id": state["tutorial"]["assessment"][0]["question_id"],
                },
            ),
        ):
            payload = dict(
                idempotency_key=uuid4().hex,
                expected_revision=client.get("/api/v1/study/state").json()["revision"],
                tutorial_version=state["tutorial"]["version"],
                position=position,
            )
            capture(
                name,
                client.put("/api/v1/study/tutorial/progress", json=payload),
                fields=[
                    "contract_version",
                    "integration_contract",
                    "revision",
                    "tutorial_progress",
                ],
                request=payload,
            )
        capture(
            "position_omitted",
            mutate(
                client,
                "PUT",
                "tutorial/progress",
                tutorial_version=state["tutorial"]["version"],
                help_opened=True,
            ),
            fields=["revision", "tutorial_progress"],
        )
        capture(
            "position_null",
            mutate(
                client,
                "PUT",
                "tutorial/progress",
                tutorial_version=state["tutorial"]["version"],
                position=None,
            ),
            expected=422,
        )
        train(store, publication, (sid, 1))
        case = capture("explanation_case", client.get("/api/v1/study/cases/current"))
        assert case["condition"] == "explanation"
        base = "/api/v1/study/workspace/" + case["workspace"]["scope_id"]
        capture("capabilities", client.get(base + "/capabilities"))
        cursor = client.get(
            base + "/hierarchy", params={**case["source"], "direction": "children", "limit": 1}
        ).json()["next_cursor"]
        if cursor is None:
            raise ValueError("Build the fixture with --navigation-size 65 for continuation")
        capture(
            "mismatched_cursor",
            client.get(
                base + "/hierarchy",
                params={**case["source"], "direction": "parents", "cursor": cursor},
            ),
            expected=409,
        )
        capture(
            "terminal_absence",
            client.get(
                base + "/explanations",
                params={
                    **case["source"],
                    "task": "entity_profile",
                    "counterpart_ontology_version_id": case["candidates"][0]["entity"][
                        "ontology_version_id"
                    ],
                    "counterpart_iri": case["candidates"][0]["entity"]["iri"],
                },
            ),
        )
        original = store._study

        def missing(db, revision, **kwargs):
            row, study = original(db, revision, **kwargs)
            study = copy.deepcopy(study)
            study["workspace_scopes"] = [
                s for s in study["workspace_scopes"] if s["case_id"] != case["case_id"]
            ]
            return row, study

        with patch.object(store, "_study", side_effect=missing):
            capture(
                "missing_required_workspace_injected",
                client.get("/api/v1/study/cases/current"),
                expected=503,
            )
        with patch(
            "exact_inspect.study.telemetry.utcnow", return_value="2026-01-01T00:00:00+00:00"
        ):
            for page in ("page-a", "page-b"):
                response = client.post(
                    "/api/v1/study/events",
                    json={
                        "events": [
                            {
                                "event_id": uuid4().hex,
                                "page_instance_id": page,
                                "sequence": 0,
                                "type": "case_ready",
                                "case_id": case["case_id"],
                                "presentation_id": case["presentation_id"],
                                "client_monotonic_ms": 0,
                                "build_version": publication.definition.software_version,
                            }
                        ]
                    },
                )
                assert response.status_code == 200, response.text
                response = client.post(
                    "/api/v1/study/timing",
                    json={
                        "segment_id": uuid4().hex,
                        "page_instance_id": page,
                        "stage": "case",
                        "case_id": case["case_id"],
                        "presentation_id": case["presentation_id"],
                        "monotonic_start_ms": 0,
                        "monotonic_end_ms": 60000,
                    },
                )
                assert response.status_code == 200, response.text
        with patch("exact_inspect.study.store.utcnow", return_value="2026-01-01T00:01:40+00:00"):
            assert (
                mutate(
                    client,
                    "POST",
                    f"cases/{case['case_id']}/submit",
                    presentation_id=case["presentation_id"],
                    response_type="none_of_these",
                    ranked_candidate_ids=[],
                ).status_code
                == 200
            )
        export = capture(
            "analysis_3",
            client.post(
                "/api/v1/admin/studies/" + publication.definition.study_revision + "/exports",
                params={"analysis_schema": "exact-study-analysis/3", "include_test": True},
                headers={"Authorization": "Bearer " + ADMIN},
            ),
        )
        assert export.get("manifest", {}).get("schema") == "exact-study-analysis/3", export
        while case["condition"] == "explanation":
            if client.get("/api/v1/study/state").json()["stage"] == "case":
                assert (
                    mutate(
                        client,
                        "POST",
                        f"cases/{case['case_id']}/submit",
                        presentation_id=case["presentation_id"],
                        response_type="none_of_these",
                        ranked_candidate_ids=[],
                    ).status_code
                    == 200
                )
            assert (
                mutate(
                    client,
                    "PUT",
                    f"cases/{case['case_id']}/consultation",
                    presentation_id=case["presentation_id"],
                    form_version="exact-study-forms/2",
                    consulted_external_ontologies=False,
                ).status_code
                == 200
            )
            case = client.get("/api/v1/study/cases/current").json()
        capture("baseline_case", client.get("/api/v1/study/cases/current"))
        capture("stale_scope_denied", client.get(base + "/capabilities"), expected=403)
        receipt = {
            "synthetic_only": True,
            "transport": "FastAPI TestClient through actual study router; backend test helpers complete preparation/training",
            "publication_file_sha256": hashlib.sha256(publication_path.read_bytes()).hexdigest(),
            "publication_receipt": published,
            "examples": examples,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; preserve previous recordings")
    record(args.fixture.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
