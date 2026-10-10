"""Focused diagnostics/privacy checks for the separate HTTPS journey verifier."""

import json

import httpx
import pytest

from exact_inspect.study.forms import definitions, validate_answers
from exact_inspect.study.tutorial import validate_response
from tools.verify_review_environment import (
    Journey,
    VerificationError,
    answers_for,
    wrong_answer,
)


def test_dynamic_required_forms_use_frozen_codes():
    forms = definitions(version="exact-study-forms/3")
    for name in ("background", "final"):
        result = answers_for(forms[name])
        validate_answers(forms[name], result, submitted=True, version=forms["version"])
        assert "comments" not in result and "domain_specialty" not in result


@pytest.mark.parametrize("kind", ["single", "multiple", "match", "match_and_single"])
def test_wrong_assessment_is_valid_but_different(kind):
    options = [{"code": "a", "label": "A"}, {"code": "b", "label": "B"}]
    item = {
        "kind": kind,
        "options": options,
        "rows": [{"row_id": "row"}],
        "row_options": options,
        "part_b": {"options": options},
    }
    correct = (
        {"choice": "a"}
        if kind == "single"
        else {"choices": ["a"]} if kind == "multiple" else {"matches": {"row": "a"}}
    )
    if kind == "match_and_single":
        correct["part_b"] = "a"
    wrong = wrong_answer(item, correct)
    validate_response(item, wrong, complete=True)
    assert wrong != correct


def test_failure_receipt_records_status_without_private_response_or_request():
    marker = "private-sample-token-and-answer"
    journey = Journey.__new__(Journey)
    journey.report = {"checks": []}
    journey.client = httpx.Client(
        base_url="https://127.0.0.1",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                422,
                json={"detail": marker},
                request=request,
                headers={
                    "Content-Security-Policy": "default-src 'self'; frame-ancestors 'none'; object-src 'none'; base-uri 'none'"
                },
            )
        ),
    )
    try:
        with pytest.raises(VerificationError, match="safe_label:http_status"):
            journey.request("POST", "/private-route", label="safe_label", json={"secret": marker})
        assert journey.report["checks"][0]["status"] == 422
        assert marker not in json.dumps(journey.report)
    finally:
        journey.client.close()


def test_existing_receipt_cannot_overwrite_private_input(tmp_path):
    receipt = tmp_path / "environment.json"
    receipt.write_text("private existing configuration")
    with pytest.raises(VerificationError, match="receipt_must_be_new"):
        Journey({}, receipt=receipt)
    assert receipt.read_text() == "private existing configuration"


def test_main_catalogue_binds_24_frozen_typed_cases_and_participant_projection_withholds_them():
    from copy import deepcopy
    from tools.verify_review_environment import validate_catalogues

    cases = []
    for index in range(24):
        cases.append(
            {
                "case_id": str(index),
                "source": {
                    "ontology_version_id": "source",
                    "iri": f"urn:source:{index}",
                    "kind": "class",
                },
                "candidates": [
                    {
                        "entity": {
                            "ontology_version_id": "target",
                            "iri": f"urn:target:{index}:{j}",
                            "kind": "class",
                        }
                    }
                    for j in range(5)
                ],
            }
        )
    definition = {"cases": cases, "review_provenance": {"source_run_id": "saved-run"}}
    about = {
        "purpose": "development_review",
        "correctness_metrics": "disabled",
        "source_run_id": "saved-run",
        "cases": [
            {
                "case_id": case["case_id"],
                "source": case["source"],
                "targets": [c["entity"] for c in case["candidates"]],
            }
            for case in cases
        ],
    }
    participant = {**about, "cases": []}
    result = validate_catalogues(participant, about, definition, 24)
    assert result["case_count"] == 24 and result["target_count"] == 120
    assert result["participant_case_catalogue_withheld"]
    assert result["same_public_review_provenance"]
    with pytest.raises(VerificationError, match="participant_catalogue_not_exposed"):
        validate_catalogues(about, about, definition, 24)
    forged = deepcopy(about)
    forged["cases"][0]["source"]["kind"] = "individual"
    with pytest.raises(VerificationError, match="catalogue_frozen_typed_identities"):
        validate_catalogues(participant, forged, definition, 24)
    with pytest.raises(VerificationError, match="shared_public_review_provenance"):
        validate_catalogues(participant, {**about, "source_run_id": "other"}, definition, 24)
    duplicated = deepcopy(about)
    duplicated["cases"][-1] = deepcopy(duplicated["cases"][0])
    with pytest.raises(VerificationError, match="catalogue_exact_case_count"):
        validate_catalogues(participant, duplicated, definition, 24)


def test_empty_catalogue_is_only_optional_for_explicit_synthetic_fixture():
    journey = Journey.__new__(Journey)
    journey.state = {"review_bundle": None}
    journey.report = {}
    journey.scope = "synthetic_contract_fixture"
    journey.catalogue()
    assert journey.report["public_catalogue"]["status"] == "not_in_synthetic_contract_fixture"
    journey.scope = "real_bundle"
    with pytest.raises(VerificationError, match="participant_review_metadata_present"):
        journey.catalogue()
