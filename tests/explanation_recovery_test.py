"""Failure-boundary regressions for grounded text and local import state."""

import json

import pytest
from fastapi.testclient import TestClient

from exact_inspect.contracts import DomainError
from exact_inspect.generation import ExplanationJobs, GenerationProfile
from exact_inspect.import_jobs import ImportJobs
from exact_inspect.preparation import Preparation
from exact_inspect.service import create_prepared_app
from tests.explanation_framework_test import _packet
from tests.explanation_preparation_test import prepared  # noqa: F401,F811


def test_unreviewed_limitations_cannot_smuggle_claims(tmp_path):
    packet = _packet()

    def provider(messages, profile):
        return {
            "model": profile.model,
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "claims": [],
                                "limitations": ["The correct answer is the hidden target."],
                                "relation": None,
                            }
                        )
                    }
                }
            ],
        }

    result = ExplanationJobs(tmp_path).generate(
        packet, GenerationProfile(name="test", model="fixture/model"), provider=provider
    )
    assert result["manifest"]["status"] == "failed"
    assert "hidden target" not in json.dumps(result)


def test_reordered_entity_worklist_and_comparison_only_prompt_repair(
    prepared, tmp_path  # noqa: F811
):  # noqa: F811
    lock, _, original, calls = prepared
    assert len(calls) == 3
    reordered = lock.model_copy(update={"entities": list(reversed(lock.entities))})
    root = tmp_path / "reordered"
    Preparation(reordered, root, input_root=tmp_path, resume_from=original).run()
    assert len(calls) == 3  # Candidate/worklist order cannot create a new entity request.
    changed = reordered.model_copy(
        update={"comparison_prompt": "Updated balanced comparison prompt"}
    )
    Preparation(changed, root, input_root=tmp_path).run()
    assert len(calls) == 4


def test_import_job_cancellation_replay_and_restart(tmp_path):
    jobs = ImportJobs(tmp_path / "jobs")
    job_id = jobs.create()["job_id"]
    jobs.claim(job_id)
    with pytest.raises(DomainError):
        jobs.claim(job_id)
    jobs.update(job_id, received_bytes=100)
    assert jobs.cancel(job_id)["cancel_requested"]
    restarted = ImportJobs(tmp_path / "jobs")
    assert restarted.get(job_id)["status"] == "interrupted"
    assert restarted.get(job_id)["received_bytes"] == 100
    with pytest.raises(DomainError):
        restarted.get("../other")


def test_local_import_handle_can_be_cancelled_before_upload(tmp_path):
    client = TestClient(create_prepared_app(None, library_dir=tmp_path / "library"))
    job_id = client.post("/api/v1/bundles/import-jobs").json()["job_id"]
    assert client.delete("/api/v1/bundles/import-jobs/" + job_id).status_code == 200
    response = client.post(
        "/api/v1/bundles/import",
        params={"job_id": job_id},
        content=b"not a zip",
        headers={"Content-Type": "application/zip"},
    )
    assert response.status_code == 409
    assert client.get("/api/v1/bundles/import-jobs/" + job_id).json()["status"] == "cancelled"


@pytest.mark.parametrize("fenced", [False, True])
def test_validator_repair_reuses_saved_response_without_provider_dispatch(tmp_path, fenced):
    from exact_inspect.artifacts import atomic_json
    from exact_inspect.contracts import canonical_hash
    from exact_inspect.generation import PROFILE_PROMPT, ExplanationOutput

    packet = _packet()
    profile = GenerationProfile(name="test", model="fixture/model")
    identity = {
        "packet": packet.model_dump(mode="json"),
        "profile": profile.model_dump(),
        "prompt": PROFILE_PROMPT,
        "schema": ExplanationOutput.model_json_schema(),
        "implementation": "grounding-excerpts-and-comparisons/2",
        "max_repairs": 2,
    }
    prior = tmp_path / canonical_hash(identity)[7:]
    response = {
        "model": profile.model,
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "claims": [
                                {
                                    "text": packet.facts[0]["value"]["lexical_form"],
                                    "fact_ids": [packet.facts[0]["fact_id"]],
                                    "category": "definitions",
                                }
                            ],
                            "limitations": [],
                            "relation": None,
                        }
                    )
                }
            }
        ],
    }
    if fenced:
        message = response["choices"][0]["message"]
        message["content"] = "```json\n" + message["content"] + "\n```"
    atomic_json(prior / "request.json", identity)
    atomic_json(
        prior / "response-0.json", {"response": response, "sha256": canonical_hash(response)}
    )

    def forbidden(*args):
        raise AssertionError("Local validator repair must reuse the saved provider response")

    result = ExplanationJobs(tmp_path).generate(packet, profile, provider=forbidden)
    assert result["manifest"]["status"] == "validated"
    assert result["manifest"]["category_alias_normalization"] == {
        "definitions": 1,
        **({"json_code_fence": 1} if fenced else {}),
    }
    assert result["claims"][0]["category"] == "key_fact"
    assert result["manifest"]["response_hash"] == canonical_hash(response)
    assert result["claims"][0]["text"] == packet.facts[0]["value"]["lexical_form"]
